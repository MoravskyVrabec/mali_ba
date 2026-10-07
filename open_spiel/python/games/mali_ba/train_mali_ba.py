import os
import re
import sys
import argparse
import configparser
import random
import time
import collections
import pickle
import gzip
from datetime import datetime

# Path setup should be done early
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
build_dir = os.path.join(project_root, "build", "python")
python_games_dir = os.path.join(project_root, "open_spiel", "python", "games")
if build_dir not in sys.path: sys.path.insert(0, build_dir)
if python_games_dir not in sys.path: sys.path.insert(0, python_games_dir)

# We need to import the MP components at the top level for the __main__ guard
import multiprocessing as mp
from multiprocessing import Process, Queue

# Constant for gathering updated weights
WEIGHTS_UPDATE_INTERVAL_SECONDS = 15

# --- Child Process Functions ---

def trainer_process(args, initial_game_params, replay_buffer_queue, weights_queue, stats_queue,
                    trainer_signal_queue=None):
    """A dedicated process for training the model."""
    # --- IMPORTS ARE THE VERY FIRST THING ---
    import tensorflow as tf
    import pyspiel
    from mali_ba.training_utils import SimpleAgent
    from pyspiel.mali_ba import log, LogLevel
    import pyspiel as _pyspiel
    if getattr(args, 'debug', False):
        _pyspiel.mali_ba.set_log_level(_pyspiel.mali_ba.LogLevel.DEBUG)
    # This import can be removed, as the ReplayBuffer is local now
    # from mali_ba.classes.classes_other import ReplayBuffer

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Trainer GPU setup error: {e}")

    # Report the compute device explicitly. A CPU-only trainer still trains, just
    # roughly an order of magnitude slower per gradient step, and until now nothing
    # in the log revealed which one you got -- it went unnoticed across many runs
    # because the plain `tensorflow` pip wheel has CUDA code paths (so
    # is_built_with_cuda() is True) but none of the nvidia-*-cu12 runtime
    # libraries, leaving list_physical_devices('GPU') silently empty.
    if gpus:
        _dev_names = []
        for _g in gpus:
            try:
                _d = tf.config.experimental.get_device_details(_g)
                _nm = _d.get('device_name') or _g.name
                _cc = _d.get('compute_capability')
                _dev_names.append(f"{_nm} (sm_{_cc[0]}{_cc[1]})" if _cc else str(_nm))
            except Exception:
                _dev_names.append(_g.name)
        log(LogLevel.INFO,
            f"Trainer: COMPUTE DEVICE = GPU x{len(gpus)} — {', '.join(_dev_names)}, "
            f"memory_growth=on, TF {tf.__version__}, keras {getattr(tf.keras, 'version', lambda: '?')()}")
    else:
        log(LogLevel.WARN,
            f"Trainer: COMPUTE DEVICE = CPU ONLY — no GPU visible to TensorFlow "
            f"{tf.__version__} (built_with_cuda={tf.test.is_built_with_cuda()}). "
            f"Gradient steps will be far slower than on a GPU. If this machine has an "
            f"NVIDIA GPU, the CUDA runtime wheels are probably missing: "
            f"pip install 'tensorflow[and-cuda]=={tf.__version__}'")

    log(LogLevel.INFO, "Trainer process started.")

    temp_game = pyspiel.load_game(args.game_name, initial_game_params)
    # Size of one observation as THIS game produces it; used to refuse a saved
    # replay buffer written under a different observation layout.
    expected_obs_size = 1
    for _d in temp_game.observation_tensor_shape():
        expected_obs_size *= _d
    agent = SimpleAgent(
        temp_game.observation_tensor_shape(),
        temp_game.num_distinct_actions(),
        temp_game.num_players(),
        args.learning_rate,
        aux_targets=getattr(args, 'aux_targets', False),
        aux_win_type_weight=getattr(args, 'aux_win_type_weight', 0.5),
        aux_moves_left_weight=getattr(args, 'aux_moves_left_weight', 1.0),
    )
    del temp_game

    _pol_file = args.load_model_path.replace("weights.h5", "_policy.weights.h5") \
        if args.load_model_path else None
    _val_file = args.load_model_path.replace("weights.h5", "_value.weights.h5") \
        if args.load_model_path else None
    if args.load_model_path and (os.path.exists(_pol_file) or os.path.exists(_val_file)):
        try:
            _loaded = agent.load_model(args.load_model_path)
            if len(_loaded) == 2:
                log(LogLevel.INFO, "Trainer loaded initial model weights (policy and value).")
            elif _loaded:
                log(LogLevel.INFO,
                    f"Trainer loaded initial model weights: {_loaded[0]} head only. The "
                    f"other head starts from random initialisation -- intentional when the "
                    f"value-target definition has changed (see gamma in mali_ba.ini).")
            else:
                log(LogLevel.WARN, "Trainer: no weight files found. Starting from scratch.")
        except Exception as e:
            log(LogLevel.WARN, f"Could not load model weights: {e}. Starting from scratch.")
    else:
        log(LogLevel.WARN, f"Can't find load model file. Starting from scratch.")
    
    # ### <<< CORRECTION 1: Put a TUPLE of weights on the queue.
    weights_queue.put((agent.policy_model.get_weights(), agent.value_model.get_weights()))

    # Use the two-pool ReplayBuffer from training_utils (not the old one in classes_other).
    from mali_ba.training_utils import ReplayBuffer

    local_replay_buffer = ReplayBuffer(args.replay_buffer_size,
                                        mcts_buffer_fraction=args.mcts_buffer_fraction,
                                        near_win_pool_fraction=args.near_win_pool_fraction,
                                        raregoods_pool_fraction=args.raregoods_pool_fraction,
                                        replace_bootstrap_with_mcts=args.replace_bootstrap_with_mcts,
                                        timbuktu_pool_fraction=getattr(args, 'timbuktu_pool_fraction', 0.0))
    training_counter = 0
    last_save_time = time.time()
    last_checkpoint_time = time.time()
    last_train_time = time.time()
    TRAIN_INTERVAL_SECONDS = getattr(args, 'train_interval_seconds', 10)
    # Moving windows of game lengths (last 20 games) for adaptive fraction tuning.
    bootstrap_lengths = collections.deque(maxlen=20)
    mcts_lengths = collections.deque(maxlen=20)

    # Restore buffer from a previous run if available.
    if args.save_buffer_path and os.path.exists(args.save_buffer_path):
        try:
            with gzip.open(args.save_buffer_path, 'rb') as f:
                saved = pickle.load(f)

            # Refuse a buffer whose observations don't match this game's layout
            # (e.g. after the observation planes changed). Nothing downstream
            # catches it: 192 planes is exactly twice 96, so reshaping a batch of
            # old observations "succeeds" by gluing pairs of unrelated positions
            # into fake ones. Move the file aside rather than leave it, so the
            # next buffer save cannot overwrite it.
            _saved_obs_size = None
            for _k in ('mcts_timeout_buffer', 'mcts_natural_buffer', 'mcts_nearwin_buffer',
                       'mcts_raregoods_buffer', 'mcts_timbuktu_buffer', 'mcts_buffer',
                       'bootstrap_buffer'):
                for _exp in saved.get(_k, []):
                    _o = _exp[0]      # numpy is not imported in trainer_process
                    _saved_obs_size = int(_o.size) if hasattr(_o, 'size') else len(_o)
                    break
                if _saved_obs_size is not None:
                    break
            if _saved_obs_size is not None and _saved_obs_size != expected_obs_size:
                del saved                      # don't keep the rejected buffer in memory
                _aside = f"{args.save_buffer_path}.obs{_saved_obs_size}"
                _n = 1
                while os.path.exists(_aside):
                    _n += 1
                    _aside = f"{args.save_buffer_path}.obs{_saved_obs_size}.{_n}"
                os.replace(args.save_buffer_path, _aside)
                raise ValueError(
                    f"saved observations have {_saved_obs_size} values but this game "
                    f"produces {expected_obs_size} (observation planes changed?). "
                    f"Starting with an EMPTY buffer; the old file was moved to {_aside}")
            # Convert any older format (e.g. the pre-2026-10-01 "natural" pool of
            # Timbuktu wins + ordinary timeouts) through the shared helper, then copy
            # each pool into a deque sized for THIS run, so its maxlen is respected
            # rather than the one baked into the save.
            from mali_ba.buffer_format import (normalize_saved_buffer, saved_keep_every,
                                               thin_pools)
            _pools_in, _notes = normalize_saved_buffer(saved)
            _have_keep = saved_keep_every(saved)
            del saved
            for _n in _notes:
                log(LogLevel.INFO, f"Trainer: Converting saved buffer -- {_n}.")
            # A buffer saved with every position kept (or fewer thinned than now) is
            # thinned to match buffer_keep_every. Otherwise its games would dominate
            # training for hours: new games add only 1 in N positions, so it takes
            # ~N times longer for them to push the old ones out.
            _, _notes = thin_pools(_pools_in, _have_keep, max(1, getattr(args, 'buffer_keep_every', 1)))
            for _n in _notes:
                log(LogLevel.INFO, f"Trainer: Saved buffer {_n}.")
            _targets = (('bootstrap_buffer',      'bootstrap'),
                        ('mcts_timeout_buffer',   'MCTS-timeout'),
                        ('mcts_nearwin_buffer',   'MCTS-nearwin'),
                        ('mcts_raregoods_buffer', 'MCTS-raregoods'),
                        ('mcts_timbuktu_buffer',  'MCTS-timbuktu'))
            for _key, _label in _targets:
                _cap = getattr(local_replay_buffer, _key).maxlen
                _src = _pools_in[_key]
                if len(_src) > _cap:
                    if _key == 'bootstrap_buffer':
                        log(LogLevel.WARN,
                            f"Trainer: Saved {_label} pool ({len(_src)}) exceeds new capacity "
                            f"({_cap}). Sampling {_cap} entries randomly.")
                        _src = random.sample(_src, _cap)
                    else:
                        log(LogLevel.WARN,
                            f"Trainer: Saved {_label} pool ({len(_src)}) exceeds new capacity "
                            f"({_cap}). Keeping most recent {_cap} entries.")
                        _src = _src[-_cap:]
                setattr(local_replay_buffer, _key, collections.deque(_src, maxlen=_cap))
            log(LogLevel.INFO,
                f"Trainer: Restored buffer from {args.save_buffer_path} — "
                f"bootstrap={len(local_replay_buffer.bootstrap_buffer)}, "
                f"mcts_timeout={len(local_replay_buffer.mcts_timeout_buffer)}, "
                f"mcts_nearwin={len(local_replay_buffer.mcts_nearwin_buffer)}, "
                f"mcts_raregoods={len(local_replay_buffer.mcts_raregoods_buffer)}, "
                f"mcts_timbuktu={len(local_replay_buffer.mcts_timbuktu_buffer)} experiences.")
            # Release the loaded copy. These locals live as long as trainer_process,
            # so without this every restored experience stays referenced after the
            # pools evict it: once the pools turned over the trainer held two full
            # buffers (B008: 51 GB resident for a ~18 GB buffer), including every
            # entry dropped when a pool was shrunk at load.
            del _pools_in, _src
            import gc
            gc.collect()
        except Exception as e:
            log(LogLevel.WARN, f"Trainer: Could not restore buffer from {args.save_buffer_path}: {e}")

    log(LogLevel.INFO, "Trainer: entering main loop. Waiting for experiences...")

    _trainer_dbg_count = 0   # limit verbose per-item debug to first 20 items
    while True:
        # Process all available experiences without blocking
        experiences_processed = 0
        max_experiences_per_cycle = args.batch_size * 4  # Process in chunks
        
        try:
            while experiences_processed < max_experiences_per_cycle:
                experience = replay_buffer_queue.get_nowait()  # Non-blocking get
                if experience is None:
                    log(LogLevel.INFO, "Trainer received shutdown signal. Saving final model.")
                    agent.save_model(args.save_model_path)
                    if args.save_buffer_path:
                        try:
                            tmp_path = args.save_buffer_path + ".tmp"
                            with gzip.open(tmp_path, 'wb',
                                           compresslevel=getattr(args, 'buffer_compresslevel', 1)) as f:
                                pickle.dump({
                                    'bootstrap_buffer':      local_replay_buffer.bootstrap_buffer,
                                    'mcts_timeout_buffer':   local_replay_buffer.mcts_timeout_buffer,
                                    'mcts_nearwin_buffer':   local_replay_buffer.mcts_nearwin_buffer,
                                    'mcts_raregoods_buffer': local_replay_buffer.mcts_raregoods_buffer,
                                    'mcts_timbuktu_buffer':  local_replay_buffer.mcts_timbuktu_buffer,
                                    'keep_every':            max(1, getattr(args, 'buffer_keep_every', 1)),
                                }, f)
                            os.replace(tmp_path, args.save_buffer_path)
                            log(LogLevel.INFO, f"Trainer: Buffer saved on shutdown to {args.save_buffer_path}.")
                        except Exception as e:
                            log(LogLevel.ERROR, f"Trainer: Buffer save on shutdown failed: {e}")
                    # The main process has stopped reading these; don't let exit wait
                    # on their feeder threads (it then sat until the 180 s join timeout).
                    for _q in (weights_queue, stats_queue):
                        try:
                            _q.cancel_join_thread()
                        except Exception:
                            pass
                    return
                # Per-game summary sent after all experiences for a game.
                if _trainer_dbg_count < 20:
                    log(LogLevel.INFO, f"  [DBG Trainer] dequeued item: type={type(experience[0]).__name__} "
                        f"len={len(experience)} first_elem_repr={repr(experience[0])[:40]}")
                    _trainer_dbg_count += 1
                if isinstance(experience, tuple) and isinstance(experience[0], str) and experience[0] == 'GAME_END':
                    # Format: ('GAME_END', game_length, is_bootstrap_game [, is_near_win])
                    is_bootstrap_game = experience[2]
                    game_length       = experience[1]
                    game_is_nearwin   = experience[3] if len(experience) > 3 else False
                    if is_bootstrap_game:
                        bootstrap_lengths.append(game_length)
                    elif not game_is_nearwin:
                        # Only track natural-win MCTS lengths for adaptive fraction.
                        mcts_lengths.append(game_length)
                    # Adapt mcts_fraction when both windows are populated.
                    if len(bootstrap_lengths) >= 10 and len(mcts_lengths) >= 10:
                        bootstrap_avg = sum(bootstrap_lengths) / len(bootstrap_lengths)
                        mcts_avg = sum(mcts_lengths) / len(mcts_lengths)
                        if mcts_avg < bootstrap_avg * 0.90:
                            new_fraction = min(0.95, local_replay_buffer.mcts_fraction + 0.05)
                            if new_fraction > local_replay_buffer.mcts_fraction:
                                local_replay_buffer.mcts_fraction = new_fraction
                                log(LogLevel.INFO,
                                    f"Adaptive fraction: MCTS avg={mcts_avg:.0f} moves < "
                                    f"Bootstrap avg={bootstrap_avg:.0f} moves. "
                                    f"mcts_fraction → {new_fraction:.2f}")
                    continue
                # experience is (obs, policy, value, is_bootstrap
                #                [, is_near_win [, is_rare_goods [, is_timbuktu]]])
                is_bootstrap  = experience[3] if len(experience) > 3 else False
                is_near_win   = experience[4] if len(experience) > 4 else False
                is_rare_goods = experience[5] if len(experience) > 5 else False
                is_timbuktu   = experience[6] if len(experience) > 6 else False
                local_replay_buffer.add(experience[:3], is_bootstrap=is_bootstrap,
                                        is_near_win=is_near_win, is_rare_goods=is_rare_goods,
                                        is_timbuktu=is_timbuktu)
                experiences_processed += 1
        except Exception as _trainer_exc:
            import queue as _queue_mod
            if not isinstance(_trainer_exc, _queue_mod.Empty):
                log(LogLevel.ERROR, f"Trainer: unexpected exception in experience loop: "
                    f"{type(_trainer_exc).__name__}: {_trainer_exc}")
        
        if experiences_processed > 0:
            # Log every batch so we can confirm trainer is receiving data
            log(LogLevel.INFO, f"Trainer processed {experiences_processed} new experiences. "
                f"Bootstrap: {len(local_replay_buffer.bootstrap_buffer)}  "
                f"MCTS-timeout: {len(local_replay_buffer.mcts_timeout_buffer)}  "
                f"MCTS-nearwin: {len(local_replay_buffer.mcts_nearwin_buffer)}  "
                f"MCTS-raregoods: {len(local_replay_buffer.mcts_raregoods_buffer)}  "
                f"MCTS-timbuktu: {len(local_replay_buffer.mcts_timbuktu_buffer)}  "
                f"| batch shares to/nw/rg/tb: "
                + "/".join(f"{x:.2f}" for x in local_replay_buffer.effective_pool_shares))

        # Train in batches
        if len(local_replay_buffer) >= args.batch_size:
            # Train when we have a large batch of new data, or on a regular time interval
            time_since_train = time.time() - last_train_time
            if experiences_processed > args.batch_size // 2 or time_since_train >= TRAIN_INTERVAL_SECONDS:
                log(LogLevel.INFO, f"Trainer: Starting training with buffer size {len(local_replay_buffer)}")
                try:
                    loss = agent.train(local_replay_buffer, args.batch_size) # Use the configured batch size
                    if loss is not None:
                        log(LogLevel.INFO, f"Trainer: Training successful, loss = {loss:.4f}")
                        stats_queue.put({"loss": loss})
                    else:
                        log(LogLevel.WARN, "Trainer: agent.train() returned None")
                except Exception as e:
                    log(LogLevel.ERROR, f"Trainer: Training failed with error: {e}")
                    import traceback
                    log(LogLevel.ERROR, f"Trainer: Full traceback: {traceback.format_exc()}")
                last_train_time = time.time()

                # Update weights for actors after a successful training step
                if weights_queue.empty():
                    # ### <<< CORRECTION 2: Put a TUPLE of weights on the queue here as well.
                    weights_queue.put((agent.policy_model.get_weights(), agent.value_model.get_weights()))

        # Bootstrap-complete checkpoint: save weights as soon as bootstrap ends
        # so MCTS actors receive a model trained on bootstrap data, not random weights.
        if trainer_signal_queue is not None and not trainer_signal_queue.empty():
            signal = trainer_signal_queue.get_nowait()
            if signal == 'bootstrap_done':
                log(LogLevel.INFO, "Trainer: Bootstrap complete signal received. Saving bootstrap checkpoint.")
                try:
                    bootstrap_path = args.save_model_path.replace(".weights.h5", "_bootstrap.weights.h5")
                    agent.save_model(bootstrap_path)
                    agent.save_model(args.save_model_path)  # also overwrite the main path
                    # Push fresh weights to actors immediately
                    weights_queue.put((agent.policy_model.get_weights(), agent.value_model.get_weights()))
                    log(LogLevel.INFO, f"Trainer: Bootstrap checkpoint saved to {bootstrap_path}. Weights pushed to actors.")
                except Exception as e:
                    log(LogLevel.ERROR, f"Trainer: Bootstrap checkpoint save failed: {e}")

        # Periodic model saving
        current_time = time.time()
        if current_time - last_save_time > args.save_every * 60:
            log(LogLevel.INFO, f"Trainer: Save interval of {args.save_every} minutes reached. Attempting to save model.")
            try:
                agent.save_model(args.save_model_path)
                last_save_time = current_time # Update time ONLY on successful save attempt
            except Exception as e:
                # The agent's save_model will also log, but we add one here too.
                log(LogLevel.ERROR, f"Trainer: agent.save_model failed inside periodic save. Error: {e}")
            if args.save_buffer_path:
                try:
                    tmp_path = args.save_buffer_path + ".tmp"
                    log(LogLevel.INFO, f"Trainer: Saving buffer to {tmp_path} ...")
                    # compresslevel matters a lot here: gzip's default of 9 costs about
                    # 41s for 34k entries and ~4 minutes for 200k, and this save is
                    # synchronous, so that time is training time lost. Level 1 saves
                    # 200k in ~32s -- faster than level 9 managed for 34k -- for a file
                    # roughly 2x larger, which is irrelevant at these sizes.
                    _save_t0 = time.time()
                    with gzip.open(tmp_path, 'wb',
                                   compresslevel=getattr(args, 'buffer_compresslevel', 1)) as f:
                        pickle.dump({
                            'bootstrap_buffer':      local_replay_buffer.bootstrap_buffer,
                            'mcts_timeout_buffer':   local_replay_buffer.mcts_timeout_buffer,
                            'mcts_nearwin_buffer':   local_replay_buffer.mcts_nearwin_buffer,
                            'mcts_raregoods_buffer': local_replay_buffer.mcts_raregoods_buffer,
                            'mcts_timbuktu_buffer':  local_replay_buffer.mcts_timbuktu_buffer,
                            'keep_every':            max(1, getattr(args, 'buffer_keep_every', 1)),
                        }, f)
                    os.replace(tmp_path, args.save_buffer_path)
                    _save_el = time.time() - _save_t0
                    _save_mb = os.path.getsize(args.save_buffer_path) / (1024 * 1024)
                    log(LogLevel.INFO,
                        f"Trainer: BUFFER SAVE took {_save_el:.1f}s for {len(local_replay_buffer):,} "
                        f"entries ({_save_mb:.1f} MB, compresslevel="
                        f"{getattr(args, 'buffer_compresslevel', 1)})")
                    log(LogLevel.INFO,
                        f"Trainer: Buffer saved to {args.save_buffer_path} — "
                        f"bootstrap={len(local_replay_buffer.bootstrap_buffer)}, "
                        f"mcts_timeout={len(local_replay_buffer.mcts_timeout_buffer)}, "
                        f"mcts_nearwin={len(local_replay_buffer.mcts_nearwin_buffer)}, "
                        f"mcts_raregoods={len(local_replay_buffer.mcts_raregoods_buffer)}, "
                        f"mcts_timbuktu={len(local_replay_buffer.mcts_timbuktu_buffer)} experiences.")
                except Exception as e:
                    log(LogLevel.ERROR, f"Trainer: Buffer save failed: {e}")

        # Every 2 hours, save a timestamped checkpoint of the model weights.
        if args.save_model_path and current_time - last_checkpoint_time > 2 * 3600:
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                checkpoint_path = args.save_model_path.replace(".weights.h5", f"_ckpt_{ts}.weights.h5")
                agent.save_model(checkpoint_path)
                last_checkpoint_time = current_time
                log(LogLevel.INFO, f"Trainer: 2-hour checkpoint saved to {checkpoint_path}.")
            except Exception as e:
                log(LogLevel.ERROR, f"Trainer: 2-hour checkpoint save failed: {e}")

        # If there's nothing to do, sleep briefly to prevent busy-waiting
        if experiences_processed == 0 and len(local_replay_buffer) < args.batch_size:
            time.sleep(0.1)


def log_pass_diagnostic(state, player, root, actor_id, episode_num, move_count):
    """Log diagnostic info when MCTS chooses Pass, to help diagnose forced-pass situations."""
    import pyspiel
    from pyspiel.mali_ba import log, LogLevel

    legal_actions = state.legal_actions()
    action_strs = [state.action_to_string(player, a) for a in legal_actions]

    # Categorise legal moves
    categories = {}
    for a, s in zip(legal_actions, action_strs):
        key = s.split('_')[0] if '_' in s else s
        categories.setdefault(key, 0)
        categories[key] += 1

    # Full visit distribution from MCTS
    visit_map = {}
    if hasattr(root, 'children'):
        for child in root.children:
            visit_map[child.action] = child.explore_count
    total_visits = sum(visit_map.values()) or 1
    pass_action = next((a for a, s in zip(legal_actions, action_strs) if s.lower() == 'pass'), None)
    pass_visits = visit_map.get(pass_action, 0) if pass_action is not None else 0

    # Mali-Ba state info
    try:
        ms = pyspiel.mali_ba.downcast_state(state)
        phase = ms.current_phase()
        extra = f"phase={phase}"
    except Exception:
        extra = "(phase unavailable)"

    log(LogLevel.INFO,
        f"  [PASS_DIAG] Actor {actor_id} Game {episode_num} Move {move_count} P{player}: "
        f"pass={pass_visits}/{total_visits} visits ({100*pass_visits/total_visits:.0f}%) | "
        f"legal={len(legal_actions)} moves: {categories} | {extra}")

    # If pass has >90% of visits, show the full legal move list — likely forced
    if pass_visits / total_visits > 0.90:
        log(LogLevel.INFO,
            f"  [PASS_DIAG] Near-forced pass. All legal actions: {action_strs}")


def actor_process(actor_id, game_params, args, job_queue, result_queue, games_per_actor,
                  arena=None, server_weights_queue=None, inference_slot=None):
    # --- (Delayed imports are the same and correct) ---
    import numpy as np
    import random
    from open_spiel.python.algorithms import mcts
    import pyspiel
    from pyspiel import mali_ba
    from pyspiel.mali_ba import log, LogLevel
    # Constrain this actor's own TF thread pools before TF is imported/
    # initialized. Actor inference is many small independent per-move calls,
    # which thrashes badly if every actor process spawns its own multi-
    # threaded BLAS/TF pool (measured: ~50 OS threads/actor unconstrained vs.
    # 2 constrained, and a real ~29% wall-clock speedup on Hetzner with this
    # set). The trainer process is a separate mp.Process (spawned fresh, see
    # mp.set_start_method('spawn')) with its own later `import tensorflow`,
    # so it's unaffected by this and keeps full multi-threading for its own
    # batched gradient step. setdefault (not direct assignment) so an
    # explicit env var set by the caller still wins.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf
    from mali_ba.training_utils import AlphaZeroEvaluator, create_mali_ba_policy_network, create_mali_ba_value_network

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')
    if getattr(args, 'debug', False):
        pyspiel.mali_ba.set_log_level(pyspiel.mali_ba.LogLevel.DEBUG)
    log(LogLevel.INFO, f"Actor {actor_id} started, configured for CPU-only execution "
                       f"(inference arena: {'attached' if arena is not None else 'none'}).")

    # --- Create ONE game object and ONE model for the actor's lifetime ---
    log(LogLevel.INFO, f"Actor {actor_id}: Initializing its game instance and model.")
    game = pyspiel.load_game(args.game_name, game_params)

    # Batched GPU inference client -- must come after `game`, whose observation shape
    # it needs. Local models are still built below: they serve the periodic
    # early-termination value checks and debug dumps, roughly 21 calls per game.
    # Only the MCTS evaluator, which is ~99.8% of move-loop time, uses the server.
    _infer_client = None
    if arena is not None:
        try:
            from mali_ba.inference_server import InferenceClient
            _infer_client = InferenceClient(arena, game.observation_tensor_shape(),
                                            slot=inference_slot)
            log(LogLevel.INFO,
                f"Actor {actor_id}: using batched inference server (slot {_infer_client.slot}).")
        except Exception as e:
            log(LogLevel.WARN,
                f"Actor {actor_id}: could not attach to inference server ({e}); "
                f"falling back to local CPU inference.")
            _infer_client = None
        # Get the max game length
    max_game_length = game.max_game_length()
    log(LogLevel.INFO, f"Using max game length: {max_game_length}")

    # Create instances of both networks
    policy_model = create_mali_ba_policy_network(game.observation_tensor_shape(), game.num_distinct_actions())
    value_model = create_mali_ba_value_network(game.observation_tensor_shape(), game.num_players())

    for _ in range(args.games_per_actor):
        job = job_queue.get()
        if job is None:  break

        episode_num, (policy_weights, value_weights), game_rng_seed = job[:3]
        _rg_heuristic_override = job[4]  # None for non-focused games, set at dispatch (line ~1511)
        _is_low_guidance_test = job[5] if len(job) > 5 else False
        # Compute weight from episode_num so it reflects actual game position,
        # not the dispatch-time value (which is stale due to job queue pre-fill).
        _bootstrap_eps = getattr(args, 'bootstrap_episodes', 0)
        _mcts_game_num = max(0, episode_num - _bootstrap_eps)
        if _is_low_guidance_test:
            # Absolute override, not additive -- this slice exists specifically to
            # measure the network's own policy/value standing on its own, so it
            # ignores the normal decay-scheduled base weight entirely.
            job_heuristic_weight = getattr(args, 'low_guidance_test_weight', 0.02)
            log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}: LOW-GUIDANCE-TEST game, "
                f"heuristic_guidance_weight={job_heuristic_weight:.3f} (absolute override, not additive)")
        elif _rg_heuristic_override is not None:
            _base_weight = compute_heuristic_weight(_mcts_game_num, args)
            job_heuristic_weight = _base_weight + _rg_heuristic_override
            log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}: RARE-GOODS-FOCUSED game, "
                f"heuristic_guidance_weight={job_heuristic_weight:.3f} "
                f"(base={_base_weight:.3f} + added={_rg_heuristic_override:.3f})")
        else:
            job_heuristic_weight = compute_heuristic_weight(_mcts_game_num, args)
            log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}: heuristic_guidance_weight={job_heuristic_weight:.3f}")

        # Feed the local inference server, if this machine runs one. On the trainer
        # host main() owns the weights and forwards them directly, but a remote
        # worker has no such hub: weights only ever arrive inside the jobs its
        # actors pull off the queue. So each actor republishes the weights from its
        # job to the local server, which coalesces to the newest. One push per game
        # per actor, against games lasting minutes, so the cost is irrelevant.
        if server_weights_queue is not None:
            try:
                server_weights_queue.put((policy_weights, value_weights))
            except Exception as _e:
                log(LogLevel.WARN,
                    f"Actor {actor_id}: could not publish weights to local inference "
                    f"server: {_e}")

        # Set weights on the two separate models
        policy_model.set_weights(policy_weights)
        value_model.set_weights(value_weights)
        
        random.seed(game_rng_seed)
        np.random.seed(game_rng_seed)
        move_count = 0
        # Playout-cap accounting, logged at game end so the speedup is visible.
        _n_fast_moves = 0
        _n_full_moves = 0
        _n_route_exempt = 0
        _total_sims = 0
        # Wall-clock buckets. External sampling profilers disagreed with production
        # by more than an order of magnitude, so the actor now measures itself.
        _t_search = 0.0      # inside bot.mcts_search()
        _t_obs = 0.0         # state.observation_tensor() at the top of each move
        _t_pick = 0.0        # visit-count extraction, action choice, policy target
        _t_apply = 0.0       # state.apply_action() + state.rewards()
        _t_replay = 0.0      # state.serialize() + replay write
        _t_move_all = 0.0    # whole move-loop iteration
        
        # --- Use the existing game object to create a new state ---
        # DO NOT RELOAD THE GAME.
        # seeded_game_params = game_params.copy()
        # seeded_game_params["rng_seed"] = game_rng_seed
        # game = pyspiel.load_game(args.game_name, seeded_game_params)
        # INSTEAD:
        state = game.new_initial_state()

        # NOTE: For MCTS, if you need the C++ RNG to be different for each game,
        # you would need a way to re-seed the game object's internal RNG.
        # A simple `game.set_rng_state(str(game_rng_seed))` exposed via pybind11
        # would be the proper way to handle this. For now, we proceed as the
        # MCTS bot's own randomness (dirichlet noise, policy sampling) will
        # provide sufficient exploration.

        # Pass both models to the evaluator
        evaluator = AlphaZeroEvaluator(game, policy_model, value_model,
                                       heuristic_guidance_weight=job_heuristic_weight,
                                       client=_infer_client)
        
        bot = mcts.MCTSBot(
            game=game, uct_c=args.uct_c, max_simulations=args.max_simulations,
            evaluator=evaluator, solve=False,
            dirichlet_noise=(0.2, 0.25),
            child_selection_fn=mcts.SearchNode.puct_value, verbose=False)

        state = game.new_initial_state()
        
        # Chance node startup
        episode_trajectory = []
        _setup_layout = None      # "Setup_k"; logged with the token placements below
        _setup_tokens = {}        # player -> ["(x,y,z)", ...]
        if state.is_chance_node():
            # The opening chance node picks the meeple layout (one of 65,536).
            # Sample it: taking legal_actions()[0] would start every game from
            # the same board.
            _layout_action = random.choice(state.legal_actions())
            _setup_layout = state.action_to_string(state.current_player(), _layout_action)
            state.apply_action(_layout_action)

        # Now do place tokens
        mali_ba_state = pyspiel.mali_ba.downcast_state(state)
        if mali_ba_state.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
            log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}: Starting token placement phase.")
            
            # --- START OF OPTIMIZATION ---
            while mali_ba_state.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
                if mali_ba_state.is_terminal(): break
                
                # For token placement, a simple uniform random choice is much faster.
                legal_actions = mali_ba_state.legal_actions()
                if not legal_actions:
                    log(LogLevel.WARN, f"Actor {actor_id}, Game {episode_num}: No legal actions in placement phase.")
                    break
                
                # Use Python's random for this, as it's already seeded.
                action = random.choice(legal_actions)
                
                # Optional: Log the placement move
                action_str = mali_ba_state.action_to_string(mali_ba_state.current_player(), action)
                log(LogLevel.DEBUG, f"Actor {actor_id}, Game {episode_num}: Placing token with action '{action_str}'")
                _setup_tokens.setdefault(mali_ba_state.current_player(), []).append(
                    action_str.replace('PlaceToken_', ''))

                mali_ba_state.apply_action(action)
            # --- END OF OPTIMIZATION ---

            # One line per game for analyze_placements.py: where each seat's tokens
            # started. Placement is uniform random here, so win rate by placement is a
            # clean (randomised) comparison. Must not contain " plays ": the remote
            # log relay drops such lines.
            log(LogLevel.INFO,
                f"Actor {actor_id}, Game {episode_num}: SETUP layout={_setup_layout} tokens "
                + " ".join(f"P{p}=" + ";".join(_setup_tokens.get(p, []))
                           for p in range(game.num_players())))

        log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}: Starting main play phase.")
        move_count = 0
        early_terminated = False
        _declining_best = []   # max(val) from last 2 value checks at move >= 360
        _stalled_best   = []   # max(val) from last 2 value checks at move >= 400
        _nk_thresh = getattr(args, 'random_no_kill_thresh', 0.0)
        _no_kill = (random.random() < _nk_thresh) if _nk_thresh > 0.0 else False
        if _no_kill:
            log(LogLevel.INFO,
                f"Actor {actor_id}, Game {episode_num}: NO-KILL game (thresh={_nk_thresh:.2f}).")

        # --- Replay file initialisation ---
        _replay_n = getattr(args, 'replay_game_n', 0)
        replay_file = None
        replay_temp_path = None
        replay_move_num = 0
        if _replay_n > 0 and getattr(args, 'replay_counts', None) is not None:
            _counts = args.replay_counts
            if any(_counts.get(k, 0) < _replay_n
                   for k in ('natural_short', 'natural_long', 'near_win', 'timeout')):
                _rdir = getattr(args, 'replay_dir', './replays')
                replay_temp_path = os.path.join(
                    _rdir, f"_tmp_{actor_id}_{episode_num}.mali_ba_replay")
                try:
                    os.makedirs(_rdir, exist_ok=True)
                    replay_file = open(replay_temp_path, 'w')
                    # create_setup_json() (not plain serialize()) so the [setup]
                    # section includes static board layout (valid_hexes/cities/
                    # grid_radius/num_players), not just dynamic state -- needed
                    # by the GUI replay loader (see main.py's MODE_GUI_REPLAY
                    # fallback, which becomes unnecessary once this is rebuilt).
                    _setup_json = pyspiel.mali_ba.downcast_state(state).create_setup_json()
                    replay_file.write(f"[setup]\n{_setup_json}\n")
                except Exception as _e:
                    log(LogLevel.WARN,
                        f"Actor {actor_id}: Could not open replay temp file: {_e}")
                    if replay_file:
                        try: replay_file.close()
                        except: pass
                    replay_file = None
                    replay_temp_path = None

        # Switch to PLAY mode
        while not state.is_terminal():
            _t_iter0 = time.perf_counter()
            _t0 = time.perf_counter()
            observation = np.array(state.observation_tensor(), dtype=np.float32)
            _t_obs += time.perf_counter() - _t0
            player = state.current_player()

            # --- Early termination checks (every 20 moves) ---
            if move_count > 0 and move_count % 20 == 0:
                _near_win = False
                try:
                    _near_win = pyspiel.mali_ba.downcast_state(state).is_near_win(args.near_win_rare_regions)
                except Exception:
                    pass
                # Value-head checks: evaluate from move 200 for lifecycle monitoring;
                # termination logic only applies from hopeless_move1 onwards.
                _val = None
                _clear_winner = False
                _hm1 = getattr(args, 'hopeless_move1', 200)
                _hm2 = getattr(args, 'hopeless_move2', 300)
                _hm3 = getattr(args, 'hopeless_move3', 360)
                _cwt = getattr(args, 'clear_winner_thresh', 0.35)
                if move_count >= 200:
                    try:
                        _obs_r = np.reshape(observation, game.observation_tensor_shape())
                        _obs_b = np.expand_dims(_obs_r, 0)
                        _val = value_model(_obs_b, training=False)[0].numpy()
                        _clear_winner = any(v > _cwt for v in _val)
                    except Exception as _ve:
                        log(LogLevel.WARN,
                            f"Actor {actor_id}: Value head check failed: {_ve}")
                if move_count >= _hm1 and _val is not None:
                    if move_count >= _hm3:
                        _thresh = getattr(args, 'hopeless_thresh3', 0.30)
                        _nw_override = getattr(args, 'hopeless_nearwin_override3', True)
                    elif move_count >= _hm2:
                        _thresh = getattr(args, 'hopeless_thresh2', 0.20)
                        _nw_override = getattr(args, 'hopeless_nearwin_override2', True)
                    else:
                        _thresh = getattr(args, 'hopeless_thresh1', 0.10)
                        _nw_override = getattr(args, 'hopeless_nearwin_override1', True)
                    _nw_exempt = _near_win and _nw_override
                    _exempt = 'near_win' if _nw_exempt else ('clear_winner' if _clear_winner else None)
                    log(LogLevel.INFO,
                        f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                        f"Value check: {[f'{v:.3f}' for v in _val]} "
                        f"(thresh={_thresh:.2f}"
                        f"{f', exempt={_exempt}' if _exempt else ''})")
                    if not _nw_exempt and not _clear_winner and all(v < _thresh for v in _val):
                        _msg = (
                            f"value head hopeless "
                            f"{[f'{v:.3f}' for v in _val]}"
                            f"{' (near_win not exempt at this level)' if _near_win else ''}"
                        )
                        if _no_kill:
                            log(LogLevel.INFO,
                                f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                                f"NO-KILL — would have terminated: {_msg}.")
                        else:
                            log(LogLevel.INFO,
                                f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                                f"Early termination — {_msg}.")
                            early_terminated = True
                            break
                    # Declining-best rule: if the best value has been negative and
                    # worsening for 2 consecutive checks, abandon regardless of near-win.
                    if move_count >= 360:
                        _declining_best.append(max(_val))
                        if len(_declining_best) > 2:
                            _declining_best.pop(0)
                        # Threshold from the ini (declining_best_thresh, default 0.0 =
                        # the original "negative" rule). The calibrated value head
                        # predicts ~0.00 for the best player in an ordinary late
                        # position, so 0.0 culls a large share of normal games.
                        _dthr = getattr(args, 'declining_best_thresh', 0.0)
                        if (len(_declining_best) == 2
                                and all(v < _dthr for v in _declining_best)
                                and _declining_best[1] < _declining_best[0]):
                            _msg = (
                                f"best value negative and declining "
                                f"{[f'{v:.3f}' for v in _declining_best]}"
                                f"{' (near_win overridden)' if _near_win else ''}"
                            )
                            if _no_kill:
                                log(LogLevel.INFO,
                                    f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                                    f"NO-KILL — would have terminated: {_msg}.")
                            else:
                                log(LogLevel.INFO,
                                    f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                                    f"Early termination — {_msg}.")
                                early_terminated = True
                                break
                    if move_count >= 400:
                        _stalled_best.append(max(_val))
                        if len(_stalled_best) > 2:
                            _stalled_best.pop(0)
                        if (len(_stalled_best) == 2
                                and all(v < 0.20 for v in _stalled_best)):
                            _msg = (
                                f"near_win stalled, max value below 0.20 "
                                f"{[f'{v:.3f}' for v in _stalled_best]}"
                                f"{' (near_win overridden)' if _near_win else ''}"
                            )
                            if _no_kill:
                                log(LogLevel.INFO,
                                    f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                                    f"NO-KILL — would have terminated: {_msg}.")
                            else:
                                log(LogLevel.INFO,
                                    f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                                    f"Early termination — near_win stalled, max value below 0.20 "
                                    f"{_msg}.")
                                early_terminated = True
                                break
                elif move_count >= 200 and _val is not None:
                    log(LogLevel.INFO,
                        f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                        f"Value check: {[f'{v:.3f}' for v in _val]} (monitoring)")
                # No-near-win termination past move threshold (protected if a clear winner exists)
                if not _near_win and not _clear_winner and move_count >= getattr(args, 'early_termination_move', 380):
                    if _no_kill:
                        log(LogLevel.INFO,
                            f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                            f"NO-KILL — would have terminated: no near-win after threshold.")
                    else:
                        log(LogLevel.INFO,
                            f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                            f"Early termination — no near-win after threshold.")
                        early_terminated = True
                        break

            # Near-win / clear-winner extension decision at base max_play_moves
            _base_max = getattr(args, 'base_max_play_moves', 430)
            if move_count == _base_max:
                _ext_thresh = getattr(args, 'near_win_extension_value_thresh', 0.2)
                _ext_moves  = getattr(args, 'near_win_extension_moves', 20)
                _ext_near_win = False
                try:
                    _ext_near_win = pyspiel.mali_ba.downcast_state(state).is_near_win(args.near_win_rare_regions)
                except Exception:
                    pass
                _ext_val = None
                try:
                    _obs_r = np.reshape(observation, game.observation_tensor_shape())
                    _obs_b = np.expand_dims(_obs_r, 0)
                    _ext_val = value_model(_obs_b, training=False)[0].numpy()
                except Exception as _ve:
                    log(LogLevel.WARN, f"Actor {actor_id}: Extension value check failed: {_ve}")
                _ext_leader_val = max(_ext_val) if _ext_val is not None else None
                _val_str = f'{_ext_leader_val:.3f}' if _ext_leader_val is not None else 'N/A'
                _ext_cwt = getattr(args, 'clear_winner_thresh', 0.35)
                _ext_clear_winner = (any(v > _ext_cwt for v in _ext_val)
                                     if _ext_val is not None else False)
                _ext_qualifies = (_ext_near_win or _ext_clear_winner) and _ext_leader_val is not None and _ext_leader_val > _ext_thresh
                if _ext_qualifies:
                    _why = '+'.join(filter(None, [
                        'near_win' if _ext_near_win else '',
                        'clear_winner' if _ext_clear_winner else '',
                    ]))
                    log(LogLevel.INFO,
                        f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                        f"Extension — continuing {_ext_moves} more moves "
                        f"({_why}, max_val={_val_str} > {_ext_thresh:.2f})")
                else:
                    _reason = f"near_win={_ext_near_win}, clear_winner={_ext_clear_winner}, max_val={_val_str}"
                    if (_ext_near_win or _ext_clear_winner) and (_ext_leader_val is None or _ext_leader_val <= _ext_thresh):
                        _reason += f" <= {_ext_thresh:.2f}"
                    log(LogLevel.INFO,
                        f"Actor {actor_id}, Game {episode_num}, Move {move_count}: "
                        f"Early termination — base max reached, no qualifying extension ({_reason})")
                    early_terminated = True
                    break

            # Phase-keyed override: OPTIONAL_ROUTE decisions can have up to 300
            # legal candidates (declaring a trade route), far more than any other
            # phase, and this branching factor doesn't track move number -- routes
            # get declared throughout the game, not just late. Measured directly
            # from the replay buffer (2026-07-25): the move-tier budget alone left
            # these decisions poorly converged (median top-1 visit share ~21%,
            # normalized entropy ~0.93). Only boost when there are enough
            # candidates to justify the extra cost -- cheap route decisions (few
            # candidates) fall through to the normal move-tier budget below.
            if move_count < getattr(args, 'sim_tier2_start', 100):
                _t1 = getattr(args, 'sim_tier1_sims', 150)
                if _rg_heuristic_override is not None:
                    _t1 += getattr(args, 'rare_goods_added_tier1_sims', 0)
                bot.max_simulations = _t1
            elif move_count < getattr(args, 'sim_tier3_start', 300):
                bot.max_simulations = getattr(args, 'sim_tier2_sims', 300)
            else:
                bot.max_simulations = getattr(args, 'sim_tier3_sims', 500)
            # Route-decision floor: OPTIONAL_ROUTE decisions with enough candidates
            # get AT LEAST sim_route_decision_sims, but never less than whatever
            # the move-tier above already provides (e.g. tier3's late-game budget,
            # which can legitimately be higher than this floor).
            _route_exempt = False
            if pyspiel.mali_ba.downcast_state(state).current_phase() == pyspiel.mali_ba.Phase.OPTIONAL_ROUTE:
                _n_route_candidates = len(state.legal_actions())
                if _n_route_candidates >= getattr(args, 'sim_route_decision_min_candidates', 20):
                    bot.max_simulations = max(bot.max_simulations,
                                               getattr(args, 'sim_route_decision_sims', 500))
                    # Large route decisions are exempt from the playout cap below:
                    # they always get this floor and are always recorded. With the
                    # cap on 75% of them were searched at 40 sims, and D006's first
                    # ~2,500 games swapped Timbuktu wins (12.4% -> 8.1%) for
                    # rare-goods wins (2.9% -> 7.1%).
                    _route_exempt = getattr(args, 'playout_cap_exempt_route_decisions', True)

            # --- Playout cap randomization (KataGo, Wu 2019) ---
            # Deliberately the LAST word on the simulation budget: it overrides
            # every tier and floor above, which is the point -- a fast move has
            # to actually be cheap.
            #
            # Most moves get a small budget and are NOT recorded as policy
            # targets (their visit counts are too unconverged to learn from).
            # A minority get the full tiered budget and are recorded. Value
            # targets come from the game outcome, so every move still trains
            # the value head either way. Net effect: far fewer simulations per
            # game for roughly the same number of usable policy targets per
            # unit of compute.
            record_policy = True
            _fast_search = (getattr(args, 'playout_cap_enabled', False)
                            and not _route_exempt
                            and random.random() >= getattr(args, 'playout_cap_full_prob', 0.25))
            if _route_exempt and getattr(args, 'playout_cap_enabled', False):
                _n_route_exempt += 1
            if _fast_search:
                bot.max_simulations = max(1, getattr(args, 'playout_cap_fast_sims', 40))
                record_policy = False

            if _fast_search:
                _n_fast_moves += 1
            else:
                _n_full_moves += 1
            _total_sims += bot.max_simulations

            # Root Dirichlet noise exists to diversify the recorded policy target.
            # A fast search records nothing, so the noise would only add variance
            # to the move actually played.
            _saved_dirichlet = bot._dirichlet_noise
            if _fast_search:
                bot._dirichlet_noise = None
            _t0 = time.perf_counter()
            try:
                root = bot.mcts_search(state)
            except Exception as e:
                import traceback
                log(LogLevel.ERROR, f"Actor {actor_id}, Game {episode_num}, Move {move_count}: MCTS search crashed!")
                log(LogLevel.ERROR, f"  Player: {player}, Phase: {pyspiel.mali_ba.downcast_state(state).current_phase()}")
                log(LogLevel.ERROR, f"  Legal actions: {state.legal_actions()}")
                log(LogLevel.ERROR, f"  Error: {e}")
                log(LogLevel.ERROR, f"  Traceback: {traceback.format_exc()}")
                raise  # Re-raise so the process still dies (and gets respawned) but we now see why
            finally:
                bot._dirichlet_noise = _saved_dirichlet
                _t_search += time.perf_counter() - _t0
            
            temperature = 1.0 if move_count < 150 else 0.5

            # ** Create the action map for this state **
            legal_actions = state.legal_actions()
            if not legal_actions:
                log(LogLevel.WARN, f"Actor {actor_id} found no legal actions for a non-terminal state. Breaking game loop.")
                break
            #===============================================================
            # DEBUG to see what choices the bot has
            #===============================================================
            if getattr(args, 'debug', False) and move_count < 20 and player >= 0:  # Only debug first moves
                #legal_actions = state.legal_actions()
                log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}, Move {move_count}: DEBUG")
                log(LogLevel.INFO, f"  Legal actions ({len(legal_actions)}): {legal_actions[:10]}...")  # Show first 10
                
                # Get neural network predictions
                observation = np.array(state.observation_tensor(), dtype=np.float32)
                obs_reshaped = np.reshape(observation, game.observation_tensor_shape())
                obs_batch = np.expand_dims(obs_reshaped, 0)
                
                try:
                    # Get policy and value from their separate, dedicated models
                    policy_pred = policy_model(obs_batch, training=False)
                    value_pred = value_model(obs_batch, training=False)
                    
                    log(LogLevel.INFO, f"  Neural network value prediction: {value_pred[0].numpy()}")
                    
                    # START DEBUG =================================================================================
                    # Show policy values for legal actions
                    policy_flat = policy_pred[0].numpy()

                    # Categorize actions by type and find the best of each
                    action_categories = {
                        'income': [],
                        'mancala': [],
                        'upgrade': [],
                        'pass': [],
                        'other': []
                    }

                    # Categorize all legal actions
                    for action in legal_actions:
                        if 0 <= action < len(policy_flat):
                            action_str = state.action_to_string(player, action).lower()
                            policy_val = policy_flat[action]
                            
                            if "income" in action_str:
                                action_categories['income'].append((action, policy_val, action_str))
                            elif "mancala" in action_str:
                                action_categories['mancala'].append((action, policy_val, action_str))
                            elif "upgrade" in action_str:
                                action_categories['upgrade'].append((action, policy_val, action_str))
                            elif "pass" in action_str:
                                action_categories['pass'].append((action, policy_val, action_str))
                            else:
                                action_categories['other'].append((action, policy_val, action_str))

                    # Find and log the top action for each category
                    top_by_category = {}
                    for category, actions in action_categories.items():
                        if actions:
                            # Sort by policy value (descending) and take the top one
                            top_action = max(actions, key=lambda x: x[1])
                            top_by_category[category] = f"{category.upper()}: {top_action[2]}: {top_action[1]:.4f}"

                    if top_by_category:
                        log(LogLevel.INFO, f"   Top policy by type: {list(top_by_category.values())}")

                    # Also show overall top 5 actions across all types
                    all_legal_with_policy = [(action, policy_flat[action], state.action_to_string(player, action)) 
                                            for action in legal_actions if 0 <= action < len(policy_flat)]
                    top_5_overall = sorted(all_legal_with_policy, key=lambda x: x[1], reverse=True)[:5]
                    top_5_strings = [f"{action_str}: {policy_val:.4f}" for _, policy_val, action_str in top_5_overall]
                    log(LogLevel.INFO, f"   Top 5 overall policies: {top_5_strings}")

                    # Special check for income actions (keep your existing logic)
                    income_actions = [a for a in legal_actions if "income" in state.action_to_string(player, a).lower()]
                    if income_actions:
                        income_action = income_actions[0]
                        income_policy = policy_flat[income_action] if income_action < len(policy_flat) else 0
                        log(LogLevel.INFO, f"   Income action {income_action} policy value: {income_policy:.4f}")
            #===============================================================
            # END DEBUG to see what choices the bot has
            #===============================================================
                        
                except Exception as e:
                    log(LogLevel.ERROR, f"  Neural network prediction failed: {e}")
            
            _t0 = time.perf_counter()
            action_map = {action: i for i, action in enumerate(legal_actions)}

            visit_counts = np.zeros(len(legal_actions))
            for child in root.children:
                if child.action in action_map:
                    visit_counts[action_map[child.action]] = child.explore_count

            if np.sum(visit_counts) > 0:
                powered_policy = np.power(visit_counts, 1.0 / temperature)
                mcts_policy_compact = powered_policy / np.sum(powered_policy)
                # Choose an action from the *compact* index space
                chosen_compact_index = np.random.choice(len(legal_actions), p=mcts_policy_compact)
                action = legal_actions[chosen_compact_index]
            else:
                action = random.choice(legal_actions)

            #===============================================================
            # DEBUG to see what choices the bot has
            #===============================================================
            if getattr(args, 'debug', False) and move_count < 200 and player >= 0:
                chosen_action_str = state.action_to_string(player, action)
                log(LogLevel.DEBUG, f"  MCTS chose: {chosen_action_str} (action {action})")

                # Show MCTS visit counts for top actions
                if hasattr(root, 'children') and len(root.children) > 0:
                    visit_counts = [(child.action, child.explore_count) for child in root.children]
                    visit_counts.sort(key=lambda x: x[1], reverse=True)
                    top_visits = []
                    for act, count in visit_counts[:3]:
                        act_str = state.action_to_string(player, act)
                        top_visits.append(f"{act_str}: {count} visits")
                    log(LogLevel.DEBUG, f"  MCTS top visits: {top_visits}")

            # # Diagnostic: log player situation whenever MCTS chooses Pass
            # if player >= 0 and state.action_to_string(player, action).lower() == 'pass':
            #     log_pass_diagnostic(state, player, root, actor_id, episode_num, move_count)
            #===============================================================
            # DEBUG to see what choices the bot has
            #===============================================================

            # For the replay buffer, we need the policy over the FULL action space.
            # A fast (playout-capped) move stays all-zero: the trainer reads that
            # as "value target only" and masks the row out of the policy loss.
            mcts_policy_full = np.zeros(game.num_distinct_actions())
            if record_policy:
                for child in root.children:
                     if 0 <= child.action < game.num_distinct_actions():
                        mcts_policy_full[child.action] = child.explore_count
                if np.sum(mcts_policy_full) > 0:
                    mcts_policy_full /= np.sum(mcts_policy_full)
                else: # Fallback for states with no visits (should be rare)
                    prob = 1.0 / len(legal_actions)
                    for act in legal_actions:
                        mcts_policy_full[act] = prob
            
            if action != pyspiel.INVALID_ACTION:
                action_str = state.action_to_string(player, action)
                log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}, Move {move_count}: Player {player} plays '{action_str}'")

            if action == pyspiel.INVALID_ACTION: break

            _t_pick += time.perf_counter() - _t0
            _t0 = time.perf_counter()
            state.apply_action(action)
            reward_vector = state.rewards()  # Called after apply_action so Rewards() sees the move in moves_history_
            _t_apply += time.perf_counter() - _t0
            episode_trajectory.append((observation, player, mcts_policy_full, reward_vector))
            _t0 = time.perf_counter()
            if replay_file:
                try:
                    replay_move_num += 1
                    _sj = pyspiel.mali_ba.downcast_state(state).serialize()
                    replay_file.write(
                        f"[move{replay_move_num}]\naction={action_str}\nstate={_sj}\n")
                except Exception as _e:
                    log(LogLevel.WARN,
                        f"Actor {actor_id}: Replay write failed at move {replay_move_num}: {_e}")
            _t_replay += time.perf_counter() - _t0
            move_count += 1
            _t_move_all += time.perf_counter() - _t_iter0

        _t_other = max(0.0, _t_move_all - (_t_search + _t_obs + _t_pick
                                          + _t_apply + _t_replay))
        def _pc(x):
            return f"{100.0*x/_t_move_all:.1f}%" if _t_move_all > 0 else "n/a"
        log(LogLevel.INFO,
            f"Actor {actor_id}, Game {episode_num}: TIME BREAKDOWN \u2014 "
            f"total={_t_move_all:.0f}s "
            f"search={_t_search:.0f}s({_pc(_t_search)}) "
            f"obs={_t_obs:.0f}s({_pc(_t_obs)}) "
            f"pick={_t_pick:.0f}s({_pc(_t_pick)}) "
            f"apply={_t_apply:.0f}s({_pc(_t_apply)}) "
            f"replay={_t_replay:.0f}s({_pc(_t_replay)}) "
            f"other={_t_other:.0f}s({_pc(_t_other)})"
            + (f" | per-sim search={1000.0*_t_search/_total_sims:.2f}ms "
               f"total={1000.0*_t_move_all/_total_sims:.2f}ms" if _total_sims else ""))

        _cache_hits, _cache_misses, _cache_rate = evaluator.cache_stats()
        log(LogLevel.INFO,
            f"Actor {actor_id}, Game {episode_num}: SEARCH COST — "
            f"moves={move_count} (full={_n_full_moves}, fast={_n_fast_moves}) "
            f"sims={_total_sims} "
            f"nn_evals={_cache_misses} cache_hits={_cache_hits} "
            f"hit_rate={_cache_rate:.3f} route_exempt={_n_route_exempt}")

        # Close replay file before classification (must be closed before rename on some OSes)
        if replay_file:
            try:
                replay_file.close()
            except Exception:
                pass
            replay_file = None

        # --- Early termination: discard game, get next job ---
        if early_terminated:
            if replay_temp_path:
                try:
                    os.remove(replay_temp_path)
                except FileNotFoundError:
                    pass
            log(LogLevel.INFO,
                f"Actor {actor_id}, Game {episode_num}: Discarded after {move_count} moves (early termination).")
            continue

        returns = state.returns()
        near_win_flag = False
        finished_msg = f"Actor {actor_id}, Game {episode_num}: FINISHED (non-terminal exit) in {move_count} moves. heuristic_weight={job_heuristic_weight:.3f}"
        if state.is_terminal():
            mali_ba_state_terminal = pyspiel.mali_ba.downcast_state(state)
            reason = mali_ba_state_terminal.get_game_end_reason()
            trigger_player = mali_ba_state_terminal.get_game_end_triggering_player()
            try:
                near_win_flag = mali_ba_state_terminal.is_near_win(args.near_win_rare_regions)
            except Exception as e:
                log(LogLevel.WARN, f"Actor {actor_id}: is_near_win check failed: {e}")

            winner_str = "Tie/Draw" # Default
            max_return = max(returns)
            if max_return >= 1.0:
                winner_player_id = returns.index(max_return)
                winner_str = f"Player {winner_player_id}"
            elif reason == "Max game length reached":
                 winner_str = "Tie/Draw (Max Length)"

            # --- Replay classification: rename or discard temp file ---
            if replay_temp_path and getattr(args, 'replay_counts', None) is not None:
                _short_thresh = getattr(args, 'replay_short_threshold', 150)
                if max_return >= 1.0:
                    _cat = 'natural_short' if move_count < _short_thresh else 'natural_long'
                elif near_win_flag:
                    _cat = 'near_win'
                elif reason == "Max game length reached":
                    _cat = 'timeout'
                else:
                    _cat = None

                _replay_n = getattr(args, 'replay_game_n', 0)
                if _cat and _replay_n > 0:
                    with args.replay_lock:
                        _cur = args.replay_counts.get(_cat, 0)
                        if _cur < _replay_n:
                            _new = _cur + 1
                            args.replay_counts[_cat] = _new
                            _rdir = getattr(args, 'replay_dir', './replays')
                            _fname = (f"replay_{_cat}_{_new}"
                                      f"_game{episode_num}_moves{move_count}.mali_ba_replay")
                            _fpath = os.path.join(_rdir, _fname)
                            try:
                                os.rename(replay_temp_path, _fpath)
                                log(LogLevel.INFO,
                                    f"Actor {actor_id}: Saved replay → {_fname}")
                                replay_temp_path = None
                            except Exception as _e:
                                log(LogLevel.WARN,
                                    f"Actor {actor_id}: Failed to save replay: {_e}")

            # --- BEGIN intermediate rewards calculation ---
            try:
                # Calculate the sum of intermediate rewards for each player from the trajectory
                num_players = game.num_players()
                total_intermediate_rewards = [0.0] * num_players
                non_zero_reward_steps = 0

                # The trajectory is a list of (observation, player, policy, reward_vector)
                for _, _, _, reward_vector in episode_trajectory:
                    if any(r != 0 for r in reward_vector):
                        non_zero_reward_steps += 1
                    for i in range(num_players):
                        total_intermediate_rewards[i] += reward_vector[i]
                
                # Format the rewards for readable logging
                formatted_rewards = [f"{r:.4f}" for r in total_intermediate_rewards]

            except Exception as e:
                log(LogLevel.WARN, f"Actor {actor_id}, Game {episode_num}: Failed to generate reward summary. Error: {e}")
            # --- END intermediate rewards calculation ---
            
            finished_msg = (
                f"Actor {actor_id}, Game {episode_num}: FINISHED in {move_count} moves. "
                f"heuristic_weight={job_heuristic_weight:.3f}. "
                f"Winner: {winner_str}. Reason: '{reason}'. "
                f"Triggered by: Player {trigger_player}. Final Returns: {returns}. "
                f"Intermediate Rewards: [{', '.join(formatted_rewards)}], "
                f"Rewarded Steps: {non_zero_reward_steps}"
            )

        # Delete replay temp file if not already saved (non-terminal exit or category not needed)
        if replay_temp_path:
            try:
                os.remove(replay_temp_path)
            except FileNotFoundError:
                pass
            replay_temp_path = None

        # Thin before sending (buffer_keep_every). A full game is ~71 MB (400 moves x
        # 173 KB observation), and the learner keeps only 1 position in N anyway.
        # Remote actors upload over the network at ~2 MB/s, so they spent most of
        # their time blocked here: results reached the learner a median 670 s (D005)
        # to 844 s (D006) after the game ended. Dropped steps keep their player and
        # reward vector, which the learner's backward value pass needs, but lose the
        # observation and policy; the learner keeps exactly the steps that have one.
        _keep = max(1, getattr(args, 'buffer_keep_every', 1))
        if _keep > 1:
            _off = random.randrange(_keep)
            episode_trajectory = [step if i % _keep == _off else (None, step[1], None, step[3])
                                  for i, step in enumerate(episode_trajectory)]
        result_queue.put((episode_trajectory, returns, finished_msg, near_win_flag))

    log(LogLevel.INFO, f"Actor {actor_id} completed its quota of {games_per_actor} games and is terminating.")

def heuristic_actor_process(actor_id, game_params, args, job_queue, result_queue, games_per_actor):
    """
    An actor process that plays games using the built-in C++ heuristic.
    It generates trajectories with one-hot policies to bootstrap the initial model.
    """
    # --- Delayed Imports ---
    import numpy as np
    import random
    import pyspiel
    from pyspiel import mali_ba
    from pyspiel.mali_ba import log, LogLevel
    import tensorflow as tf

    # Ensure this actor runs on CPU to leave GPU for the trainer
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')
    if getattr(args, 'debug', False):
        pyspiel.mali_ba.set_log_level(pyspiel.mali_ba.LogLevel.DEBUG)
    log(LogLevel.INFO, f"Heuristic Actor {actor_id} started (CPU-only).")

    # --- Create ONE game object for the lifetime of the actor ---
    log(LogLevel.INFO, f"Heuristic Actor {actor_id}: Initializing its game instance.")
    game = pyspiel.load_game(args.game_name, game_params)
    # Get the max game length
    max_game_length = game.max_game_length()
    log(LogLevel.INFO, f"Using max game length: {max_game_length}")
    
    for _ in range(args.games_per_actor):
        job = job_queue.get()
        if job is None:
            break

        episode_num, _, game_rng_seed = job[:3]

        # Seed Python's random for deterministic choices if needed
        random.seed(game_rng_seed)
        np.random.seed(game_rng_seed)

        state = game.new_initial_state()
        # Seed the C++ state RNG per-episode so heuristic choices vary across games
        mali_ba_state_early = pyspiel.mali_ba.downcast_state(state)
        mali_ba_state_early.seed_rng(int(game_rng_seed % (2**31 - 1)))
        
        # The C++ state's internal RNG will be seeded by the game object's RNG state,
        # which is usually set once at creation. For true per-game randomness from C++,
        # you might need a game.set_rng_state() method if the initial seed isn't sufficient.
        # However, for the heuristic actor, this is less critical.

        # --- Play the full game to generate a trajectory ---
        
        # This list will store tuples of (observation, player, one_hot_policy)
        # The final game outcome will be added later.
        temp_trajectory = []

        # Handle initial chance node
        if state.is_chance_node():
            # The opening chance node picks the meeple layout (one of 65,536).
            # Sample it: taking legal_actions()[0] would start every game from
            # the same board.
            state.apply_action(random.choice(state.legal_actions()))

        mali_ba_state = pyspiel.mali_ba.downcast_state(state)
        # Use the heuristic for token placement so players start near cities
        while mali_ba_state.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
            if mali_ba_state.is_terminal(): break
            legal_actions = mali_ba_state.legal_actions()
            if not legal_actions: break
            action = mali_ba_state.select_heuristic_random_action()
            if action == pyspiel.INVALID_ACTION:
                action = random.choice(legal_actions)
            mali_ba_state.apply_action(action)
        
        log(LogLevel.INFO, f"Heuristic Actor {actor_id}, Game {episode_num}: Starting main play phase.")
        move_count = 0

        # Main heuristic-driven play loop
        while not state.is_terminal():
            player = state.current_player()
            observation = np.array(state.observation_tensor(), dtype=np.float32)
            # # DEBUG ======================================================================
            # log(LogLevel.INFO, f"Raw observation shape: {observation.shape}")
            # log(LogLevel.INFO, f"Raw observation range: min={np.min(observation)}, max={np.max(observation)}")
            # log(LogLevel.INFO, f"Raw observation sample: {observation[:20]}")  # First 20 values
            # # END DEBUG ======================================================================

            # Get the action from the C++ heuristic
            action = mali_ba_state.select_heuristic_random_action()
            if action == pyspiel.INVALID_ACTION:
                log(LogLevel.WARN, f"Heuristic Actor {actor_id} received invalid action. Breaking.")
                break
                
            # Get the weights for ALL legal actions
            action_weights_map = mali_ba_state.get_heuristic_action_weights()
            
            # Create the policy target vector
            policy_target = np.zeros(game.num_distinct_actions(), dtype=np.float32)
            
            total_weight = sum(action_weights_map.values())
            
            if total_weight > 0:
                for act, weight in action_weights_map.items():
                    policy_target[act] = weight / total_weight # Normalize to a probability distribution
            else:
                # Fallback for states where all weights are zero (should be rare)
                legal_actions = state.legal_actions()
                if legal_actions:
                    prob = 1.0 / len(legal_actions)
                    for act in legal_actions:
                        policy_target[act] = prob

            # Store the observation and policy for the current state (S_t)
            # Then, apply the action to transition to the next state (S_{t+1})
            state.apply_action(action)
            
            # Now get the immediate reward (R_{t+1}) received for that transition
            reward_vector = state.rewards()

            # Store the complete 4-element tuple for this step
            temp_trajectory.append((observation, player, policy_target, reward_vector))

            move_count += 1
            # END Main heuristic-driven play loop - while not state.is_terminal():

        # Game is over, get the final returns
        returns = state.returns()

        # # DEBUG Check action diversity in this game
        # action_counts = {}
        # action_names = {}
        # for obs, player, policy_target in temp_trajectory:
        #     action = np.argmax(policy_target)
        #     action_counts[action] = action_counts.get(action, 0) + 1
        #     # Get the action name for readability
        #     if action not in action_names:
        #         action_names[action] = state.action_to_string(player, action) if hasattr(state, 'action_to_string') else f"action_{action}"
        
        # # DEBUG Log the action distribution with names
        # action_summary = {action_names.get(action, f"action_{action}"): count 
        #                  for action, count in action_counts.items()}
        # log(LogLevel.DEBUG, f"Heuristic Actor {actor_id}, Game {episode_num}: Action distribution: {action_summary}")


        terminal_state = pyspiel.mali_ba.downcast_state(state)
        reason = terminal_state.get_game_end_reason()
        trigger_player = terminal_state.get_game_end_triggering_player()
        winner_str = "Tie/Draw"
        max_return = max(returns)
        if max_return >= 1.0:
            winner_str = f"Player {list(returns).index(max_return)}"

        # Summarise intermediate rewards from the trajectory
        try:
            num_players = game.num_players()
            total_intermediate_rewards = [0.0] * num_players
            non_zero_reward_steps = 0
            for _, _, _, reward_vector in temp_trajectory:
                if any(r != 0 for r in reward_vector):
                    non_zero_reward_steps += 1
                for i in range(num_players):
                    total_intermediate_rewards[i] += reward_vector[i]
            formatted_rewards = [f"{r:.4f}" for r in total_intermediate_rewards]
        except Exception as e:
            log(LogLevel.WARN, f"Heuristic Actor {actor_id}, Game {episode_num}: Failed to generate reward summary. Error: {e}")
            formatted_rewards = []
            non_zero_reward_steps = 0

        finished_msg = (
            f"Heuristic Actor {actor_id}, Game {episode_num}: FINISHED in {move_count} moves. "
            f"Winner: {winner_str}. Reason: '{reason}'. Triggered by: Player {trigger_player}. "
            f"Final Returns: {list(returns)}. "
            f"Intermediate Rewards: [{', '.join(formatted_rewards)}], "
            f"Rewarded Steps: {non_zero_reward_steps}"
        )

        # End-of-game diagnostic: per-player goods and route coverage
        try:
            all_routes = terminal_state.get_trade_routes()
            num_players = len(returns)
            for p in range(num_players):
                rare_goods = dict(terminal_state.get_player_rare_goods(p))
                common_goods = dict(terminal_state.get_player_common_goods(p))
                total_common = sum(common_goods.values())
                unique_rare = sum(1 for v in rare_goods.values() if v > 0)
                # Count active routes owned by this player index
                # TradeRoute.owner is a PlayerColor enum; compare by index position
                active_routes = sum(1 for r in all_routes
                                    if r.active and int(r.owner) == p + 1)
                rare_str = " ".join(f"{k}:{v}" for k, v in rare_goods.items() if v > 0)
                log(LogLevel.INFO,
                    f"[HEURISTIC_DIAG] Player {p}"
                    f" | unique_rare={unique_rare}"
                    f" routes={active_routes}"
                    f" common_goods={total_common}"
                    f" | rare=[{rare_str}]")
            log(LogLevel.INFO,
                f"[HEURISTIC_DIAG] Game ended: reason='{reason}' moves={move_count}")
        except Exception as e:
            log(LogLevel.DEBUG, f"[HEURISTIC_DIAG] Diagnostic failed: {e}")

        # The result queue expects a trajectory, the returns, a FINISHED log message, and a near-win flag.
        # The message is logged by the learner so it appears in the desktop log even for remote actors.
        near_win_flag = False
        try:
            near_win_flag = terminal_state.is_near_win(args.near_win_rare_regions)
        except Exception as e:
            log(LogLevel.WARN, f"Heuristic Actor {actor_id}: is_near_win check failed: {e}")
        result_queue.put((temp_trajectory, returns, finished_msg, near_win_flag))

    log(LogLevel.INFO, f"Heuristic Actor {actor_id} completed its quota and is terminating.")


def compute_heuristic_weight(mcts_games, args):
    """Linear decay of heuristic_guidance_weight from initial to final over the configured MCTS game range."""
    initial = args.heuristic_guidance_weight_initial
    final   = args.heuristic_guidance_weight_final
    start   = args.heuristic_guidance_decay_start
    end     = args.heuristic_guidance_decay_end
    if final == initial or end <= start:
        return initial
    if mcts_games <= start:
        return initial
    if mcts_games >= end:
        return final
    t = (mcts_games - start) / (end - start)
    return initial + t * (final - initial)


def spawn_actor(actor_id, initial_game_params, args, job_queue, result_queue, actor_pool,
                actor_function, arena=None, slot=None, slot_map=None):
    """
    Creates, starts, and tracks a new actor process using the specified actor function.
    """
    # Slots are assigned by the parent, not claimed by the child, so a slot freed by
    # a finished actor can be reused. Claiming in the child leaked one slot per
    # respawn, and once the arena filled the next actor silently fell back to local
    # CPU inference -- a 7x regression with no error anywhere.
    _extra = (() if (arena is None or actor_function is not actor_process)
              else (arena, None, slot))
    p = mp.Process(target=actor_function, args=(
        actor_id, initial_game_params, args, job_queue, result_queue, args.games_per_actor)
        + _extra)
    p.start()
    actor_pool[p] = actor_id # Associates the process object with its ID
    if slot_map is not None and slot is not None:
        slot_map[p] = slot
    print(f"Main: Spawned new actor (type: {actor_function.__name__}) with ID {actor_id}.")

def log_game_outcome_debug(total_games_processed, returns, game_length, max_game_length):
    try:
        from pyspiel.mali_ba import log, LogLevel
    except ImportError:
        class LogLevel: INFO, WARN = 1, 2
        def log(level, msg): print(msg)

    """Debug logging for game outcomes."""
    if total_games_processed % 50 == 0 or total_games_processed <= 10:
        length_penalty_ratio = game_length / max_game_length
        
        if any(r > 0 for r in returns):  # Someone won
            winner = returns.index(max(returns))
            #discounted_win = 1.0 - (length_penalty_ratio ** 1.5)
            log(LogLevel.INFO, f"  DECISIVE GAME: Player {winner} won in {game_length} moves")
            log(LogLevel.INFO, f"    Return: {returns[winner]:.3f} ")
        elif any(r == 0 for r in returns):  # Draw
            log(LogLevel.INFO, f"  DRAW GAME: {game_length} moves, all players get penalty")
        else:
            log(LogLevel.INFO, f"  UNUSUAL GAME: Returns {returns}")

# --- Main Orchestrator (Updated for robustness) ---
def main(args):
    import pyspiel
    try:
        from pyspiel.mali_ba import log, LogLevel
        if getattr(args, 'debug', False):
            pyspiel.mali_ba.set_log_level(pyspiel.mali_ba.LogLevel.DEBUG)
    except ImportError:
        class LogLevel: INFO, WARN = 1, 2
        def log(level, msg): print(msg)
    import queue
    from training_utils import get_training_parameters_from_game

    log(LogLevel.INFO, "--- Starting Mali-Ba MULTIPROCESS AI Training Script ---")

    # --- 1. Setup ---
    initial_game_params = {"config_file": args.config_file or ""}
    if args.players: initial_game_params["NumPlayers"] = args.players
    if args.grid_radius: initial_game_params["grid_radius"] = args.grid_radius
    initial_game_params["player_types"] = "ai,ai,ai"
    # Get training parameters from the game
    # DEBUG: Not sure if this is needed or if it's all taken care of in C++
    temp_game = pyspiel.load_game(args.game_name, initial_game_params)
    training_params = get_training_parameters_from_game(temp_game)
    max_game_length = temp_game.max_game_length()
    del temp_game
    DRAW_PENALTY = training_params['draw_penalty']
    MAX_MOVES_PENALTY = training_params['max_moves_penalty']
    QUICK_WIN_BONUS = training_params['quick_win_bonus'] 
    QUICK_WIN_THRESHOLD = training_params['quick_win_threshold']

    # log(LogLevel.INFO, f"Using training parameters from INI:")
    # log(LogLevel.INFO, f"  Draw penalty: {DRAW_PENALTY}")
    # log(LogLevel.INFO, f"  Draw penalty: {MAX_MOVES_PENALTY}")
    # log(LogLevel.INFO, f"  Quick win bonus: {QUICK_WIN_BONUS}")
    # log(LogLevel.INFO, f"  Quick win threshold: {QUICK_WIN_THRESHOLD}")

    # --- Replay collection setup ---
    args.replay_game_n = args.replay_game
    args.replay_short_threshold = QUICK_WIN_THRESHOLD
    if args.replay_game_n > 0:
        _replay_mgr = mp.Manager()
        args.replay_counts = _replay_mgr.dict({
            'natural_short': 0,
            'natural_long':  0,
            'near_win':      0,
            'timeout':       0,
        })
        args.replay_lock = _replay_mgr.Lock()
        os.makedirs(args.replay_dir, exist_ok=True)
        log(LogLevel.INFO,
            f"Replay collection active: up to {args.replay_game_n} per category, "
            f"short threshold={QUICK_WIN_THRESHOLD} moves, dir='{args.replay_dir}'.")
    else:
        args.replay_counts = None
        args.replay_lock = None

    # The job queue must be large enough for both local and remote actors.
    total_actors = args.num_actors + args.remote_actors
    max_jobs = total_actors * 3
    job_queue = mp.Queue(maxsize=max_jobs)

    # The result queue must also accommodate remote actors returning results.
    max_results = total_actors
    result_queue = mp.Queue(maxsize=max_results)

    # Remote actors relay their stdout/stderr through this queue so their C++
    # log() output (value checks, game summaries) appears in the desktop log.
    log_queue = mp.Queue(maxsize=10000)

    # Size the IPC queue to hold many full games in flight so end-game states
    # (which carry the strongest value signal) are never silently dropped.
    # Each game is ~690 moves; hold enough for all actors × 20 games each.
    max_replay_items = 690 * (args.num_actors + args.remote_actors) * 20
    replay_buffer_queue = mp.Queue(maxsize=max_replay_items)

    weights_queue = mp.Queue()

    # --- Batched GPU inference server ---------------------------------------
    # Actors run TF single-threaded on purpose, where one batch-1 pass through both
    # nets costs ~37ms and is essentially all of self-play cost. One process owning
    # the GPU and batching every actor's requests measured ~6.9x more evaluations
    # per second. Local (same-machine) actors attach via shared memory; remote
    # actors are unaffected and keep doing their own CPU inference.
    inference_arena = None
    inference_server_proc = None
    inference_weights_queue = None
    inference_stop = None
    inference_free_slots = []      # parent-managed free list
    inference_slot_by_proc = {}    # process -> slot, so a reaped actor frees its slot
    if getattr(args, 'inference_server', False):
        try:
            from mali_ba.inference_server import (InferenceArena, server_loop as
                                                  _inference_server_loop, STOP as _INF_STOP)
            _tmp_game = pyspiel.load_game(args.game_name, initial_game_params)
            _shape = _tmp_game.observation_tensor_shape()
            _obs_size = 1
            for _d in _shape:
                _obs_size *= _d
            # One slot per local actor, plus headroom for respawns claiming new slots.
            # Exactly one slot per concurrent actor (plus a little headroom), since
            # slots are now recycled when an actor is reaped.
            _n_slots = max(1, args.num_actors) + 4
            inference_arena = InferenceArena(_n_slots, _obs_size,
                                             _tmp_game.num_distinct_actions(),
                                             _tmp_game.num_players())
            inference_weights_queue = mp.Queue()
            inference_stop = mp.Event()
            inference_server_proc = mp.Process(
                target=_inference_server_loop,
                args=(inference_arena, _shape, initial_game_params,
                      inference_weights_queue, inference_stop),
                kwargs=dict(max_batch=getattr(args, 'inference_max_batch', 32),
                            log_every=300,
                            cpus=getattr(args, 'inference_cpus', 'auto')),
                # daemon=True so the server cannot outlive the parent. With
                # daemon=False a killed run (SIGTERM, timeout, Ctrl-C) left the
                # server orphaned and still holding several GB of VRAM -- observed
                # five such strays pinning the full 24GB card. The graceful
                # shutdown path below still runs first on a normal exit.
                daemon=True)
            inference_server_proc.start()
            # gpu_ok is set as soon as the server confirms a GPU, before weights
            # arrive. If it is not set, the server refused to serve (a batched CPU
            # server is far slower than per-actor inference), so drop the arena.
            if not inference_arena.gpu_ok.wait(180):
                log(LogLevel.ERROR,
                    "Main: inference server reported no usable GPU and will not serve. "
                    "Actors will use local CPU inference. Most likely the CUDA runtime "
                    "wheels are missing (pip install 'tensorflow[and-cuda]').")
                if inference_server_proc.is_alive():
                    inference_stop.set()
                    inference_server_proc.join(timeout=15)
                inference_arena = None
                inference_server_proc = None
                inference_weights_queue = None
            else:
                inference_free_slots = list(range(_n_slots))
                log(LogLevel.INFO,
                    f"Main: inference server started (slots={_n_slots}, "
                    f"max_batch={getattr(args, 'inference_max_batch', 32)}).")
        except Exception as _e:
            log(LogLevel.ERROR,
                f"Main: failed to start inference server ({_e}); actors will use "
                f"local CPU inference.")
            inference_arena = None
            inference_server_proc = None
    stats_queue = mp.Queue()
    trainer_signal_queue = mp.Queue()  # main sends control signals to trainer

    # --- Heuristic parameter randomisation for tuning experiments ---
    # Writes two floats to /tmp/mali_ba_heuristic_params.txt so the C++
    # heuristic can pick them up (read once per actor process, then cached).
    # mult_add_in in [0, 1]: added to distance-slope multipliers in kMancalaStep.
    # add_add_in  in [0.5, 2]: added to flat bonuses in kMancalaStep.
    # Set args.randomise_heuristic_params=False to disable and use defaults (0, 0).
    if getattr(args, 'randomise_heuristic_params', False):
        mult_add_in = random.uniform(0.0, 1.0)
        add_add_in  = random.uniform(0.5, 2.0)
        with open('/tmp/mali_ba_heuristic_params.txt', 'w') as _pf:
            _pf.write(f'{mult_add_in:.6f} {add_add_in:.6f}\n')
        log(LogLevel.INFO, f'Heuristic params: mult_add_in={mult_add_in:.4f}  add_add_in={add_add_in:.4f}')
    else:
        # Write tuned values so any stale file from a previous run is overwritten.
        with open('/tmp/mali_ba_heuristic_params.txt', 'w') as _pf:
            _pf.write('0.7 1.5\n')


    if args.heuristic_only:
        trainer = None
        log(LogLevel.INFO, "Heuristic-only mode: trainer/learner skipped.")
    else:
        trainer = mp.Process(target=trainer_process, args=(
            args, initial_game_params, replay_buffer_queue, weights_queue, stats_queue,
            trainer_signal_queue))
        trainer.start()
        log(LogLevel.INFO, "Launched trainer process.")

    # --- Distributed queue server (optional) ---
    remote_status = None  # set below when distributed; read by the result loop
    if args.distributed:
        import threading
        from queue_server import start_server
        shared_config = {
            'game_name':                           args.game_name,
            'initial_game_params':                 initial_game_params,
            'uct_c':                               args.uct_c,
            'max_simulations':                     args.max_simulations,
            'games_per_actor':                     args.games_per_actor,
            'heuristic_guidance_weight':           args.heuristic_guidance_weight,
            'heuristic_guidance_weight_initial':   args.heuristic_guidance_weight_initial,
            'heuristic_guidance_weight_final':     args.heuristic_guidance_weight_final,
            'heuristic_guidance_decay_start':      args.heuristic_guidance_decay_start,
            'heuristic_guidance_decay_end':        args.heuristic_guidance_decay_end,
            'bootstrap_episodes':                  args.bootstrap_episodes,
            'near_win_rare_regions':               args.near_win_rare_regions,
            'early_termination_move':              args.early_termination_move,
            'hopeless_move1':                      args.hopeless_move1,
            'hopeless_thresh1':                    args.hopeless_thresh1,
            'hopeless_move2':                      args.hopeless_move2,
            'hopeless_thresh2':                    args.hopeless_thresh2,
            'hopeless_move3':                      args.hopeless_move3,
            'hopeless_thresh3':                    args.hopeless_thresh3,
            'hopeless_nearwin_override1':          args.hopeless_nearwin_override1,
            'hopeless_nearwin_override2':          args.hopeless_nearwin_override2,
            'hopeless_nearwin_override3':          args.hopeless_nearwin_override3,
            'clear_winner_thresh':                 args.clear_winner_thresh,
            'random_no_kill_thresh':               getattr(args, 'random_no_kill_thresh', 0.0),
            'declining_best_thresh':               getattr(args, 'declining_best_thresh', 0.0),
            'rare_goods_actor_fraction':           getattr(args, 'rare_goods_actor_fraction', 0.0),
            'rare_goods_added_heuristic_weight':   getattr(args, 'rare_goods_added_heuristic_weight', 0.30),
            'rare_goods_added_tier1_sims':         getattr(args, 'rare_goods_added_tier1_sims', 0),
            'job_timeout_hours':                   getattr(args, 'job_timeout_hours', 3.0),
            'sim_tier1_sims':                      getattr(args, 'sim_tier1_sims', 150),
            'sim_tier2_start':                     getattr(args, 'sim_tier2_start', 100),
            'sim_tier2_sims':                      getattr(args, 'sim_tier2_sims', 300),
            'sim_tier3_start':                     getattr(args, 'sim_tier3_start', 300),
            'sim_tier3_sims':                      getattr(args, 'sim_tier3_sims', 500),
            'buffer_keep_every':                   getattr(args, 'buffer_keep_every', 1),
            'playout_cap_enabled':                 getattr(args, 'playout_cap_enabled', False),
            'playout_cap_full_prob':               getattr(args, 'playout_cap_full_prob', 0.25),
            'playout_cap_fast_sims':               getattr(args, 'playout_cap_fast_sims', 40),
            'playout_cap_exempt_route_decisions':  getattr(args, 'playout_cap_exempt_route_decisions', True),
            'sim_route_decision_sims':              getattr(args, 'sim_route_decision_sims', 500),
            'sim_route_decision_min_candidates':    getattr(args, 'sim_route_decision_min_candidates', 20),
            'base_max_play_moves':                 getattr(args, 'base_max_play_moves', 430),
            'near_win_extension_moves':            getattr(args, 'near_win_extension_moves', 20),
            'near_win_extension_value_thresh':     getattr(args, 'near_win_extension_value_thresh', 0.2),
            'debug':                               getattr(args, 'debug', False),
        }
        from queue_server import RemoteStatus
        remote_status = RemoteStatus()
        server_thread = threading.Thread(
            target=start_server,
            args=(job_queue, result_queue, shared_config, log_queue),
            kwargs={'host': args.bind_host, 'port': args.queue_port, 'authkey': args.authkey.encode(),
                    'remote_status': remote_status},
            daemon=True  # Dies automatically when main process exits
        )
        server_thread.start()

        def _drain_remote_logs():
            while True:
                try:
                    line = log_queue.get(timeout=1.0)
                    print(line, flush=True)
                except Exception:
                    pass

        threading.Thread(target=_drain_remote_logs, daemon=True).start()

        log(LogLevel.INFO, f"Distributed queue server started on {args.bind_host}:{args.queue_port}.")
        log(LogLevel.INFO, f"  Remote actors can connect with:")
        log(LogLevel.INFO, f"  python remote_actors.py --server_host <this machine's IP> --server_port {args.queue_port}")

    # --- 2. Learner State Init (Same) ---
    actor_pool = {}
    next_actor_id = 0
    total_games_processed = 0
    # Value-target range tracking. tanh caps the value head at [-1, +1], so any
    # target outside that range is unreachable and its gradient vanishes -- worth
    # knowing about rather than discovering it in a buffer dump months later.
    _vt_min, _vt_max = float('inf'), float('-inf')
    _vt_saturation_warned = False
    jobs_dispatched = 0
    jobs_timed_out = 0        # cumulative jobs assumed lost to spot preemption / crash
    start_time = time.time()
    last_weights_update_time = time.time()
    last_result_time = time.time()  # last time a game result was received
    last_remote_result_time = None  # last game from a remote actor (id >= 100000)
    last_remote_warn_time = 0.0
    
    if args.random_seed is None:
        master_seed = int(time.time() * 1000) % (2**32 - 1)
    else:
        master_seed = args.random_seed
            
    import numpy as np
    seed_generator = np.random.RandomState(master_seed)
    log(LogLevel.INFO, f"Master seed generator initialized with seed: {master_seed}")

    if args.heuristic_only:
        current_weights = None
        log(LogLevel.INFO, "Heuristic-only mode: skipping initial weights.")
    else:
        log(LogLevel.INFO, "Learner waiting for initial weights from trainer...")
        current_weights = weights_queue.get()
        if inference_weights_queue is not None:
            inference_weights_queue.put(current_weights)
            if inference_arena is not None and not inference_arena.ready.wait(600):
                log(LogLevel.ERROR,
                    "Main: inference server did not become ready; actors will fall back "
                    "to local CPU inference.")
                inference_arena = None
        log(LogLevel.INFO, "Learner received initial weights.")

    # --- 3. UNIFIED Main Learner Loop ---
    bootstrap_transition_logged = False
    log(LogLevel.INFO,
        f"Learner: VALUE TARGET CONFIG = gamma {getattr(args, 'gamma', 0.997)}, "
        f"buffer {args.replay_buffer_size}, batch {args.batch_size}, "
        f"train_interval {getattr(args, 'train_interval_seconds', 10)}s"
        + ("  [gamma=1.0: targets independent of moves remaining]"
           if getattr(args, 'gamma', 0.997) >= 1.0 else
           "  [gamma<1: targets depend on moves remaining, which the observation "
           "does not encode -- see mali_ba.ini]"))

    while total_games_processed < args.num_episodes:


        if total_games_processed % 10 == 0:
            log(LogLevel.DEBUG, f"Queue sizes - Jobs: {job_queue.qsize()}, "
                            f"Results: {result_queue.qsize()}, "
                            f"Replay: {replay_buffer_queue.qsize()}")

        # --- A. Actor & Job Management ---
        
        # Determine which actor function to use for any NEW spawns
        if args.heuristic_only or total_games_processed < args.bootstrap_episodes:
            current_actor_function = heuristic_actor_process
        else:
            if not bootstrap_transition_logged:
                log(LogLevel.INFO,
                    f"--- {'Bootstrap phase complete. ' if args.bootstrap_episodes > 0 else ''}MCTS phase active. "
                    f"Initial heuristic_guidance_weight={args.heuristic_guidance_weight:.3f} "
                    f"(decay {args.heuristic_guidance_decay_start}→{args.heuristic_guidance_decay_end}, "
                    f"final={args.heuristic_guidance_weight_final:.3f}) ---")
                bootstrap_transition_logged = True
                if args.bootstrap_episodes > 0:
                    # Signal the trainer to save a checkpoint now so MCTS actors
                    # start with weights that reflect the full bootstrap dataset.
                    trainer_signal_queue.put('bootstrap_done')
            current_actor_function = actor_process
            # Update heuristic weight according to decay schedule so each newly
            # spawned actor picks up the current value (args is copied at spawn time).
            mcts_games = max(0, total_games_processed - args.bootstrap_episodes)
            prev_weight = args.heuristic_guidance_weight
            args.heuristic_guidance_weight = compute_heuristic_weight(mcts_games, args)
            if abs(args.heuristic_guidance_weight - prev_weight) >= 0.01:
                log(LogLevel.INFO,
                    f"Heuristic guidance weight: {args.heuristic_guidance_weight:.3f} "
                    f"(MCTS game {mcts_games}, decay range "
                    f"{args.heuristic_guidance_decay_start}-{args.heuristic_guidance_decay_end})")
            
        # --- Find, remove, and immediately replace dead actors ---
        dead_actors = [p for p in actor_pool if not p.is_alive()]
        if dead_actors:
            log(LogLevel.INFO, f"Found {len(dead_actors)} dead actor(s). Respawning...")
            
        for p in dead_actors:
            actor_crashed_id = actor_pool[p]
            log(LogLevel.WARN, f"Actor ID {actor_crashed_id} terminated. Spawning replacement of type {current_actor_function.__name__}.")
            
            # Remove the dead process from the pool
            del actor_pool[p]
            # Recycle its inference slot before spawning the replacement.
            _freed = inference_slot_by_proc.pop(p, None)
            if _freed is not None:
                inference_free_slots.append(_freed)
            
            # Immediately spawn its replacement
            _slot = inference_free_slots.pop(0) if inference_free_slots else None
            spawn_actor(next_actor_id, initial_game_params, args, job_queue, result_queue, actor_pool, current_actor_function, arena=inference_arena, slot=_slot, slot_map=inference_slot_by_proc)
            next_actor_id += 1
            
        # This separate loop is now only necessary for the initial startup,
        # but it's harmless to keep it for ensuring the pool is always full.
        while len(actor_pool) < args.num_actors:
            log(LogLevel.INFO, f"Actor pool below target ({len(actor_pool)}/{args.num_actors}). Spawning new actor.")
            _slot = inference_free_slots.pop(0) if inference_free_slots else None
            spawn_actor(next_actor_id, initial_game_params, args, job_queue, result_queue, actor_pool, current_actor_function, arena=inference_arena, slot=_slot, slot_map=inference_slot_by_proc)
            next_actor_id += 1
            
        # --- Remote-results watchdog (warning only; remote workers restart themselves) ---
        # Remote games arrived earlier in the run but none for 15+ minutes: either the
        # remote machine was stopped, or its results are being silently lost (2026-10-06:
        # 411 laptop games over 75 minutes, no error on either machine).
        if (last_remote_result_time is not None
                and time.time() - last_remote_result_time > 900
                and time.time() - last_remote_warn_time > 900):
            last_remote_warn_time = time.time()
            log(LogLevel.WARN,
                f"REMOTE WATCHDOG: no game received from any remote actor for "
                f"{(time.time() - last_remote_result_time) / 60:.0f} min. If the remote "
                f"machine is still playing, its results are being lost; remote_actors.py "
                f"with its watchdog on will restart itself.")

        # --- Maintain a healthy job queue size ---
        # Keep dispatching until num_episodes games have been RECEIVED. Capping at
        # num_episodes jobs DISPATCHED (as before) ran out of work early: culled games
        # (~15%) and games lost in transit are dispatched but never arrive, so the run
        # sat idle until the job timeout fired. Jobs still queued at the end are
        # drained at shutdown; the queue's target size bounds how many are outstanding.
        target_job_queue_size = total_actors * 2
        while job_queue.qsize() < target_job_queue_size and total_games_processed < args.num_episodes:
            unique_seed = seed_generator.randint(0, 2**31 - 1)
            # Held-out low-guidance test slice: a small fraction of games run at a
            # fixed, near-zero heuristic weight (absolute, not additive -- overrides
            # the base weight entirely) so the network's own policy/value can be
            # measured standing on its own, continuously, without having to test a
            # guidance reduction on the whole fleet and risk a costly win-rate
            # crater across every actor. Mutually exclusive with the rare-goods
            # focus below -- a job is either a low-guidance test, RG-focused, or
            # normal, never more than one at once, so each slice stays cleanly
            # interpretable.
            _lg_fraction = getattr(args, 'low_guidance_test_fraction', 0.0)
            _is_low_guidance_test = (_lg_fraction > 0.0 and random.random() < _lg_fraction)
            if _is_low_guidance_test:
                _rg_override = None
            else:
                _rg_fraction = getattr(args, 'rare_goods_actor_fraction', 0.0)
                _rg_weight   = getattr(args, 'rare_goods_added_heuristic_weight', 0.30)
                _rg_override = _rg_weight if (_rg_fraction > 0.0 and random.random() < _rg_fraction) else None
            job_queue.put((jobs_dispatched, current_weights, unique_seed, args.heuristic_guidance_weight,
                            _rg_override, _is_low_guidance_test))
            jobs_dispatched += 1

        # --- B. Try to process a result ---
        try:
            result = result_queue.get(timeout=1.0)
            near_win_flag = result[3] if len(result) > 3 else False
            trajectory, returns, finished_msg = result[0], result[1], result[2]
            last_result_time = time.time()

            # Log the FINISHED message here so it appears in the desktop log for all actors,
            # including remote actors whose subprocess stdout goes to the laptop.
            log(LogLevel.INFO, finished_msg)
            # Record the arrival for remote watchdogs (see queue_server.RemoteStatus).
            _m_aid = re.search(r'Actor (\d+),', finished_msg)
            if _m_aid:
                _aid = int(_m_aid.group(1))
                if remote_status is not None:
                    remote_status.record(_aid)
                if _aid >= 100000:
                    last_remote_result_time = time.time()

            # Process the game result
            total_games_processed += 1
            phase_label = "Bootstrap" if total_games_processed <= args.bootstrap_episodes else "MCTS"
            log(LogLevel.INFO, f"LEARNER ({phase_label}) RECEIVED GAME #{total_games_processed}/{args.num_episodes}. "
                            f"Length: {len(trajectory)} moves. Returns: {returns}")

            # Discount factor for the value target. NOTE: with gamma < 1 the target for
            # a given position depends on how many moves remain, but the observation
            # tensor encodes no move count or progress-to-timeout -- so the network
            # cannot tell a certain win 400 moves out (target ~ +0.36 at gamma=0.997)
            # from the same win 20 moves out (target ~ +1.13), and regresses to the
            # mean of the two. gamma = 1.0 removes that unobservable dependency; the
            # "win sooner" incentive is already carried by time_penalty and
            # max_moves_penalty, so gamma < 1 is largely redundant with them.
            GAMMA = getattr(args, 'gamma', 0.997)

            # The 'returns' variable from the C++ state is the final terminal outcome.
            # Length penalty is handled in C++ via time_penalty (per-step) and
            # max_moves_penalty (at timeout), so we use returns directly here.
            game_length = len(trajectory)
            final_terminal_returns = list(returns)

            # This will store (obs, player, policy, calculated_value_vector) for each step.
            trajectory_with_values = []

            # We iterate backwards from the end of the game.
            # The value of the last state is just the final game outcome.
            next_state_discounted_returns = final_terminal_returns

            for i in range(len(trajectory) - 1, -1, -1):
                observation, player, policy_target, immediate_reward_vector = trajectory[i]
                
                # The value of a state is: (the immediate reward you get) + gamma * (the value of the state you land in).
                # Since we are iterating backwards, 'next_state_discounted_returns' holds the value of the next state.
                current_state_value_vector = [
                    r + GAMMA * next_r for r, next_r in zip(immediate_reward_vector, next_state_discounted_returns)
                ]
                
                # Add this step's data with the correctly calculated value to our list.
                trajectory_with_values.append((observation, player, policy_target, current_state_value_vector))

                _step_min = min(current_state_value_vector)
                _step_max = max(current_state_value_vector)
                if _step_min < _vt_min: _vt_min = _step_min
                if _step_max > _vt_max: _vt_max = _step_max
                if not _vt_saturation_warned and (_step_min < -1.0 or _step_max > 1.0):
                    _vt_saturation_warned = True
                    log(LogLevel.WARN,
                        f"Trainer: VALUE TARGET OUT OF RANGE — saw {_step_min:+.3f}..{_step_max:+.3f}, "
                        f"outside the value head's tanh range [-1, +1]. Those targets are "
                        f"unreachable and their gradients vanish. Reduce the reward magnitudes "
                        f"(max_moves_penalty, loss_penalty, rare_goods_bonus) so the discounted "
                        f"returns stay inside [-1, +1].")
                
                # The value we just calculated becomes the "next state value" for the previous step in the next iteration.
                next_state_discounted_returns = current_state_value_vector

            # The list is currently in reverse order, so let's put it back chronologically.
            trajectory_with_values.reverse()

            # Win-condition games are rare and carry a strong learning signal —
            # oversample them so the NN sees them proportionally more often.
            game_length = len(trajectory)
            max_len = args.num_episodes  # proxy; real check is whether winner exists by condition
            is_win_condition_game = (args.oversample_threshold > 0) and (game_length < args.oversample_threshold) and any(r > 0 for r in returns)
            oversample_factor = 3 if is_win_condition_game else 1
            if is_win_condition_game:
                log(LogLevel.INFO, f"  Win-condition game! Oversampling x{oversample_factor} into replay buffer.")

            # Now, add the correctly calculated experiences to the replay buffer.
            is_bootstrap_game = total_games_processed < args.bootstrap_episodes
            is_timeout_game = 'Max game length reached' in finished_msg
            # Natural win via Rare goods end-condition (not Timbuktu, not timeout, not bootstrap).
            is_rare_goods_game = (any(r >= 1.0 for r in returns)
                                  and not is_bootstrap_game
                                  and 'Timbuktu' not in finished_msg)
            # Natural win via the Timbuktu end-condition: routed to its own pool.
            is_timbuktu_game = (any(r >= 1.0 for r in returns)
                                and not is_bootstrap_game
                                and 'Timbuktu' in finished_msg)
            log(LogLevel.INFO, f"  [DBG] game={total_games_processed} bootstrap={is_bootstrap_game} "
                f"timeout={is_timeout_game} near_win={near_win_flag} rare_goods={is_rare_goods_game} "
                f"timbuktu={is_timbuktu_game} "
                f"skip_timeout_games={args.skip_timeout_games} returns={[round(r,3) for r in returns]}")
            # Skip timeout games during bootstrap always, and during MCTS when skip_timeout_games is set.
            # Exception: retain near-win timeout games regardless.
            if is_timeout_game and (is_bootstrap_game or args.skip_timeout_games):
                if near_win_flag:
                    source = "bootstrap" if is_bootstrap_game else "skip_timeout_games"
                    retain_pct = args.near_win_bootstrap_pct if is_bootstrap_game else args.near_win_mcts_pct
                    if random.random() < retain_pct:
                        log(LogLevel.INFO, f"  Near-win timeout game retained ({source}) [random keep].")
                    else:
                        log(LogLevel.INFO, f"  Near-win timeout game discarded ({source}) [random skip].")
                        continue
                else:
                    log(LogLevel.INFO, f"  [DBG] Skipping non-near-win timeout game (bootstrap={is_bootstrap_game} skip_timeout={args.skip_timeout_games}).")
                    continue

            # TEMPORARY: during bootstrap, only insert natural win games (skip draws/ties).
            if is_bootstrap_game and not any(r > 0 for r in returns):
                log(LogLevel.INFO, f"  [DBG] Skipping non-win bootstrap game (draw/tie). returns={returns}")
                continue

            # Keep only 1 position in buffer_keep_every per game (2026-10-01). Keeping every
            # move filled the 100k buffer with only ~240 games, and the value head learned
            # to recognise which game a position came from instead of judging it: 100%
            # winner top-pick on games it had trained on, ~50% on new games. Thinning lets
            # the same buffer span N times as many games. The offset is random per game
            # so every move number is represented across games.
            keep_every = max(1, getattr(args, 'buffer_keep_every', 1))
            # Steps keep their move index, which the aux moves-left target needs.
            indexed_steps = list(enumerate(trajectory_with_values))
            if any(step[0] is None for step in trajectory_with_values):
                # Already thinned by the actor before sending: keep the steps it kept.
                kept_steps = [(i, step) for i, step in indexed_steps if step[0] is not None]
            elif keep_every > 1:
                # Full trajectory (heuristic actors, or a remote worker running older code).
                kept_steps = indexed_steps[random.randrange(keep_every)::keep_every]
            else:
                kept_steps = indexed_steps
            # Aux "how does the game end" targets (training_utils.AUX_WIN_TYPES order).
            if getattr(args, 'aux_targets', False):
                _aux_type = (1 if is_timbuktu_game else 2 if is_rare_goods_game
                             else 0 if is_timeout_game else None)
            else:
                _aux_type = None
            _aux_max_moves = max(1, getattr(args, 'base_max_play_moves', 420))
            log(LogLevel.INFO, f"  [DBG] Game passed filters — queuing {len(kept_steps)} of "
                f"{len(trajectory_with_values)} experiences (1 in {keep_every}, oversample "
                f"x{oversample_factor}). bootstrap={is_bootstrap_game} near_win={near_win_flag}")
            queued_count = 0
            for _ in range(oversample_factor):
                for step_idx, (observation, player, policy_target, value_target_vector) in kept_steps:
                    player_value = value_target_vector[player]
                    if _aux_type is None:
                        value_data = (player, player_value, value_target_vector)
                    else:
                        _moves_left = min(1.0, (game_length - step_idx) / _aux_max_moves)
                        value_data = (player, player_value, value_target_vector,
                                      (_aux_type, _moves_left))

                    # Always send to trainer — bootstrap games are tagged so the
                    # trainer can route them to the correct buffer pool.
                    # (heuristic_only skips this entirely since there is no trainer.)
                    if not args.heuristic_only:
                        if not replay_buffer_queue.full():
                            replay_buffer_queue.put((observation, policy_target, value_data, is_bootstrap_game, near_win_flag and is_timeout_game, is_rare_goods_game, is_timbuktu_game))
                            queued_count += 1
                        else:
                            log(LogLevel.INFO, f"  [DBG] Replay buffer queue FULL after {queued_count} puts.")
                            break
            if not args.heuristic_only:
                log(LogLevel.INFO, f"  [DBG] Queued {queued_count} experiences to replay_buffer_queue.")

            if total_games_processed % 25 == 0 and _vt_max > _vt_min:
                log(LogLevel.INFO,
                    f"Trainer: VALUE TARGET RANGE = {_vt_min:+.3f}..{_vt_max:+.3f} "
                    f"(gamma={GAMMA}, tanh limit ±1.0)")

            # Send a per-game summary so the trainer can track average game lengths
            # and adaptively adjust the MCTS/bootstrap sampling fraction.
            if not args.heuristic_only:
                replay_buffer_queue.put(('GAME_END', game_length, is_bootstrap_game, near_win_flag and is_timeout_game))

            # Add detailed outcome logging
            # log_game_outcome_debug(total_games_processed, discounted_returns, game_length, max_game_length)

            # Debug logging for first few games
            if total_games_processed <= 5:
                log(LogLevel.INFO, f"Game {total_games_processed} debug:")
                log(LogLevel.INFO, f"  Returns: {returns}")

            # # Create training experiences with proper value targets
            # experiences_added = 0
            # for observation, player, policy_target in trajectory:
            #     player_value = returns[player]
            #     value_data = (player, player_value, returns)
            #     replay_buffer_queue.put((observation, policy_target, value_data))
            #     experiences_added += 1

            # # Debug log
            # if total_games_processed <= 10:  # Only for first 10 games
            #     log(LogLevel.INFO, f"MAIN: Added {experiences_added} experiences to queue. Queue size now: {replay_buffer_queue.qsize()}")


        except queue.Empty:  # Only catch timeout exceptions
            #log(LogLevel.INFO, "...")
            pass
        except Exception as e:  # Log any other exceptions
            log(LogLevel.ERROR, f"Error processing game result: {e}")
            import traceback
            log(LogLevel.ERROR, f"Traceback: {traceback.format_exc()}")


        # --- C. Weight & Stats Management (Periodic) ---
        if time.time() - last_weights_update_time > WEIGHTS_UPDATE_INTERVAL_SECONDS:
            latest_weights = None
            while not weights_queue.empty():
                try: latest_weights = weights_queue.get_nowait()
                except: break
            if latest_weights is not None:
                current_weights = latest_weights
                if inference_weights_queue is not None:
                    inference_weights_queue.put(current_weights)
                log(LogLevel.INFO, f"Learner updated to latest weights at game #{total_games_processed}. Distributing to {len(actor_pool)} actors.")

            
            while not stats_queue.empty():
                try:
                    stats = stats_queue.get_nowait()
                    if "loss" in stats:
                        log(LogLevel.INFO, f"Trainer reported loss: {stats['loss']:.4f} at game #{total_games_processed}")
                except: break

            # --- No-results alarm ---
            # Warning only: dispatch no longer depends on jobs_timed_out, it simply
            # continues until num_episodes games are received.
            _job_timeout_secs = getattr(args, 'job_timeout_hours', 3.0) * 3600
            _time_since_result = time.time() - last_result_time
            if _time_since_result > _job_timeout_secs:
                _inflight = max(0, jobs_dispatched - total_games_processed - job_queue.qsize())
                if _inflight > 0:
                    jobs_timed_out += _inflight
                    last_result_time = time.time()  # reset to avoid immediate re-fire
                    log(LogLevel.WARN,
                        f"Job timeout: no result received in {_time_since_result/3600:.1f}h. "
                        f"About {_inflight} dispatched job(s) have not come back (culled, "
                        f"lost, or actors stalled). Cumulative: {jobs_timed_out}.")

            last_weights_update_time = time.time()

    # --- 4. Final Shutdown (Same) ---
    log(LogLevel.INFO, "All episodes processed. Sending shutdown signals...")
    if not args.heuristic_only:
        replay_buffer_queue.put(None)

    # Drain leftover jobs. empty() only sees what has reached the pipe; each job
    # carries the full weights, so more can still be in the queue's feeder buffer.
    # The old `while not empty(): get_nowait()` stopped early, leaving jobs AHEAD of
    # the stop sentinels: actors started new games instead of stopping, and the
    # unsent jobs then made the process hang at exit. Drain until it stays empty.
    import queue as _queue_mod
    _drained = 0
    while True:
        try:
            job_queue.get(timeout=2.0)
            _drained += 1
        except _queue_mod.Empty:
            break
        except Exception:
            break
    log(LogLevel.INFO, f"Shutdown: drained {_drained} unstarted job(s).")

    # One sentinel per actor, remote ones included: they read the same queue and
    # can take sentinels meant for local actors, leaving those blocked.
    for _ in range(len(actor_pool) + (args.remote_actors if args.distributed else 0)):
        job_queue.put(None)

    if trainer is not None:
        log(LogLevel.INFO, "Waiting for trainer to terminate...")
        trainer.join(timeout=180)
        if trainer.is_alive():
            log(LogLevel.WARN, "Trainer did not terminate gracefully. Forcing.")
            trainer.terminate()

    # One shared deadline: per-actor 60 s timeouts ran one after another, so a few
    # stuck actors could add many minutes.
    log(LogLevel.INFO, "Waiting for actors to terminate...")
    _deadline = time.time() + 90
    for p in actor_pool:
        p.join(timeout=max(0.0, _deadline - time.time()))
    _stuck = [p for p in actor_pool if p.is_alive()]
    for p in _stuck:
        p.terminate()
    if _stuck:
        log(LogLevel.WARN, f"Shutdown: terminated {len(_stuck)} actor(s) that did not stop.")

    # Stop the inference server last: actors may still be blocked on it above, and
    # a client waiting on a dead server would only unblock on its timeout.
    if inference_server_proc is not None:
        try:
            if inference_arena is not None:
                _s = inference_arena.stats()
                log(LogLevel.INFO,
                    f"InferenceServer totals: {_s['requests']:,} evals in "
                    f"{_s['batches']:,} batches (avg batch {_s['avg_batch']:.1f}, "
                    f"{_s['avg_infer_ms']:.2f} ms/batch, "
                    f"{_s['ms_per_request']:.3f} ms/eval)")
            if inference_stop is not None:
                inference_stop.set()
            if inference_weights_queue is not None:
                from mali_ba.inference_server import STOP as _INF_STOP2
                inference_weights_queue.put(_INF_STOP2)
            inference_server_proc.join(timeout=60)
            if inference_server_proc.is_alive():
                log(LogLevel.WARN, "Inference server did not stop gracefully. Forcing.")
                inference_server_proc.terminate()
        except Exception as _e:
            log(LogLevel.WARN, f"Inference server shutdown issue: {_e}")

    # Whatever is still buffered in queues this process wrote to has no reader now.
    # Without this, interpreter exit waits on their feeder threads forever.
    for _q in (job_queue, replay_buffer_queue, inference_weights_queue):
        if _q is not None:
            try:
                _q.cancel_join_thread()
            except Exception:
                pass

    log(LogLevel.INFO, "All processes terminated.")
    end_time = time.time()
    log(LogLevel.INFO, f"Total time: {end_time - start_time:.2f} seconds")
    log(LogLevel.INFO, f"Training complete. Final model saved to {args.save_model_path}")
    

# --- Main Execution Guard ---
if __name__ == "__main__":

    try:
        # Use the mp alias we defined at the top
        mp.set_start_method('spawn', force=True) 
    except RuntimeError:
        print("Note: multiprocessing start method already set.")
        pass
    
    parser = argparse.ArgumentParser(description='Mali-Ba Multiprocess AI Training Script')
    # ... (all args are the same) ...
    parser.add_argument('--num_actors', type=int, default=max(1, os.cpu_count() - 2), help="Number of actor processes to run in parallel.")
    parser.add_argument('--game_name', type=str, default="mali_ba")
    parser.add_argument('--config_file', type=str, default="/home/robp/Projects/mali_ba/open_spiel/games/mali_ba/mali_ba.ini")
    parser.add_argument('--players', type=int, default=None, choices=range(2, 6))
    parser.add_argument('--grid_radius', type=int, default=None)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default="mali_ba_agent_v2.weights.h5")
    parser.add_argument('--replay_buffer_size', type=int, default=None)
    parser.add_argument('--save_buffer_path', type=str, default=None,
                        help="If set, save/restore the replay buffer to this file between runs.")
    parser.add_argument('--mcts_buffer_fraction', type=float, default=None,
                    help="Fraction of replay buffer capacity reserved for MCTS games "
                         "(remainder goes to bootstrap). Overrides ini value if set. Default: from ini or 0.80.")
    parser.add_argument('--replace_bootstrap_with_mcts', type=lambda x: x.lower() in ('1','true','yes'),
                    default=None,
                    help="When the MCTS timeout pool is full, promote evicted entries into "
                         "the bootstrap pool instead of discarding them. Overrides ini value if set.")

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_every', type=int, default=10, help="Save a model checkpoint every N minutes in the trainer process.")
    parser.add_argument('--update_workers_every', type=int, default=50, help="This argument is now effectively unused as weights are updated via a queue.")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Initial learning rate for the schedule.")
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--uct_c', type=float, default=2.0)
    parser.add_argument('--max_simulations', type=int, default=500)
    parser.add_argument('--sim_tier1_sims', type=int, default=None,
                        help="MCTS simulations for play-phase moves before sim_tier2_start. Overrides ini.")
    parser.add_argument('--sim_tier2_start', type=int, default=None,
                        help="Play-phase move count where tier 2 begins. Overrides ini.")
    parser.add_argument('--sim_tier2_sims', type=int, default=None,
                        help="MCTS simulations for tier 2 moves. Overrides ini.")
    parser.add_argument('--sim_tier3_start', type=int, default=None,
                        help="Play-phase move count where tier 3 begins. Overrides ini.")
    parser.add_argument('--sim_tier3_sims', type=int, default=None,
                        help="MCTS simulations for tier 3 moves (late game). Overrides ini.")
    parser.add_argument('--buffer_compresslevel', type=int, default=None, choices=range(0, 10),
                        help="gzip level for replay-buffer saves (0-9). The save is synchronous, "
                             "so this is training time: level 9 costs ~4 min for a 200k buffer, "
                             "level 1 about 32s. Overrides ini.")
    parser.add_argument('--inference_server', action='store_true', default=None,
                        help="Route MCTS neural-network evaluation through a single "
                             "batched GPU inference server instead of each actor running "
                             "its own single-threaded CPU forward passes. Measured ~6.9x "
                             "more evaluations per second. Overrides ini.")
    parser.add_argument('--no_inference_server', dest='inference_server',
                        action='store_false',
                        help="Disable the batched inference server even if the ini enables it.")
    parser.set_defaults(inference_server=None)
    parser.add_argument('--inference_max_batch', type=int, default=None,
                        help="Maximum batch the inference server assembles. Overrides ini.")
    parser.add_argument('--inference_cpus', type=str, default=None,
                        help="CPUs to pin the inference server to: 'auto' (performance cores "
                             "or the large-L3 cores, if the CPU has them), 'none', or a list "
                             "like '0-7,16-23'. Overrides ini.")
    parser.add_argument('--gamma', type=float, default=None,
                        help="Discount factor for value targets. 1.0 makes a position's target "
                             "independent of how many moves remain, which the observation tensor "
                             "does not encode. Overrides ini.")
    parser.add_argument('--playout_cap_enabled', action='store_true', default=None,
                        help="Enable playout cap randomization: most moves get a cheap "
                             "search and are used as value-only samples, a minority get "
                             "the full tiered budget and supply the policy targets. "
                             "Overrides ini.")
    parser.add_argument('--playout_cap_full_prob', type=float, default=None,
                        help="Probability a move gets the full tiered simulation budget "
                             "and is recorded as a policy target. Overrides ini.")
    parser.add_argument('--no_playout_cap', dest='playout_cap_enabled', action='store_false',
                        help="Disable playout cap randomization even if the ini enables it.")
    parser.set_defaults(playout_cap_enabled=None)
    parser.add_argument('--aux_targets', type=int, default=None,
                        help="1 = train extra value-network heads on how the game ends "
                             "(win type, moves left); 0 = off. Overrides ini.")
    parser.add_argument('--playout_cap_exempt_route_decisions', type=int, default=None,
                        help="1 = OPTIONAL_ROUTE decisions with >= sim_route_decision_min_candidates "
                             "candidates always get the full search and are recorded, even with "
                             "the playout cap on; 0 = cap them like any move. Overrides ini.")
    parser.add_argument('--playout_cap_fast_sims', type=int, default=None,
                        help="Simulation budget for fast (non-recorded) moves. Overrides ini.")
    parser.add_argument('--sim_route_decision_sims', type=int, default=None,
                        help="MCTS simulations for OPTIONAL_ROUTE decisions with at least "
                             "sim_route_decision_min_candidates legal candidates, overriding "
                             "whatever the move-tier budget would otherwise give. Overrides ini.")
    parser.add_argument('--sim_route_decision_min_candidates', type=int, default=None,
                        help="Minimum legal-action count for an OPTIONAL_ROUTE decision to "
                             "receive the sim_route_decision_sims boost. Overrides ini.")
    parser.add_argument('--games_per_actor', type=int, default=10, 
                    help="Number of games each actor process plays before self-terminating to free memory.")
    parser.add_argument('--bootstrap_episodes', type=int, default=0,
                    help="Number of initial episodes to generate using the C++ heuristic for bootstrapping.")
    parser.add_argument('--skip_timeout_games', action='store_true',
                         help="Do not add max-length (timeout) MCTS games to the replay buffer.")
    parser.add_argument('--near_win_rare_regions', type=int, default=None,
                         help="Threshold of rare-good regions for a timeout game to be considered near-win. "
                              "Overrides ini value if set. Default: from ini or 4.")
    parser.add_argument('--near_win_bootstrap_pct', type=float, default=None,
                         help="Fraction of near-win timeout games from bootstrap to retain. "
                              "Overrides ini value if set. Default: from ini or 0.50.")
    parser.add_argument('--near_win_mcts_pct', type=float, default=None,
                         help="Fraction of near-win timeout games from MCTS to retain. "
                              "Overrides ini value if set. Default: from ini or 0.50.")
    parser.add_argument('--near_win_pool_fraction', type=float, default=None,
                         help="Fraction of the MCTS buffer capacity reserved for near-win games. "
                              "Overrides ini value if set. Default: from ini or 0.30.")
    parser.add_argument('--raregoods_pool_fraction', type=float, default=None,
                         help="Fraction of the MCTS buffer capacity reserved for Rare goods natural wins. "
                              "Overrides ini value if set. Default: from ini or 0.15.")
    parser.add_argument('--buffer_keep_every', type=int, default=None,
                         help="Keep 1 position in N from each finished game in the replay "
                              "buffer, so it spans N times as many games. Default: from ini or 1 "
                              "(keep every position).")
    parser.add_argument('--timbuktu_pool_fraction', type=float, default=None,
                         help="Fraction of the MCTS buffer capacity (and of every batch) for "
                              "Timbuktu wins. 0 keeps them in the timeout pool (old behaviour). "
                              "Default: from ini or 0.")
    parser.add_argument('--heuristic_guidance_weight', type=float, default=None,
                         help="Starting weight of heuristic policy in MCTS prior. Default: from ini or 0.40.")
    parser.add_argument('--heuristic_guidance_weight_final', type=float, default=None,
                         help="Floor value for heuristic weight after decay. Default: from ini or same as initial.")
    parser.add_argument('--low_guidance_test_fraction', type=float, default=None,
                         help="Fraction of dispatched jobs held out as a continuous low-guidance test "
                              "slice, run at an absolute (not additive) fixed low heuristic weight, to "
                              "measure whether the network can carry play without heuristic support. "
                              "Mutually exclusive with rare-goods-focused selection. 0.0 disables. "
                              "Overrides ini value if set. Default: from ini or 0.0.")
    parser.add_argument('--low_guidance_test_weight', type=float, default=None,
                         help="Absolute heuristic_guidance_weight used for low-guidance-test jobs "
                              "(replaces, not adds to, the normal decay-scheduled weight). "
                              "Overrides ini value if set. Default: from ini or 0.02.")
    parser.add_argument('--heuristic_guidance_decay_start', type=int, default=None,
                         help="MCTS game number where heuristic weight decay begins. Default: from ini or 300.")
    parser.add_argument('--heuristic_guidance_decay_end', type=int, default=None,
                         help="MCTS game number where heuristic weight reaches its floor. Default: from ini or 2500.")
    parser.add_argument('--rare_goods_added_tier1_sims', type=int, default=None,
                         help="Extra tier-1 simulations added for rare-goods-focused games. "
                              "0 disables. Overrides ini value if set.")
    parser.add_argument('--rare_goods_actor_fraction', type=float, default=None,
                         help="Fraction of MCTS games played with an elevated heuristic weight to bias "
                              "exploration toward Rare goods wins. 0.0 disables. Overrides ini value if set.")
    parser.add_argument('--rare_goods_added_heuristic_weight', type=float, default=None,
                         help="Heuristic guidance weight used for rare-goods-focused games "
                              "(replaces the decay-scheduled weight for those games). Overrides ini value if set.")
    parser.add_argument('--heuristic_only', action='store_true',
                    help="Run heuristic games only — skip the trainer/learner entirely. "
                         "Use with --bootstrap_episodes for fast heuristic diagnostics.")
    parser.add_argument('--randomise_heuristic_params', action='store_true',
                    help="Randomise kMancalaStep heuristic tuning params each run and "
                         "write them to /tmp/mali_ba_heuristic_params.txt for the C++ side.")
    parser.add_argument('--distributed', action='store_true',
                    help="Start a queue server so remote machines can contribute actor processes.")
    parser.add_argument('--bind_host', type=str, default='0.0.0.0',
                    help="Interface for the distributed queue server to bind to. "
                         "'0.0.0.0' accepts connections on any interface (required on cloud VMs).")
    parser.add_argument('--queue_port', type=int, default=50000,
                    help="TCP port for the distributed queue server (used with --distributed).")
    parser.add_argument('--authkey', type=str, default='malibatraining2024',
                    help="Shared secret for the distributed queue server.")
    parser.add_argument('--remote_actors', type=int, default=0,
                    help="Expected number of remote actor processes (used to size queues correctly).")
    parser.add_argument('--debug', action='store_true',
                    help="Set log level to DEBUG in all processes (Python and C++).")
    parser.add_argument('--replay_game', type=int, default=0,
                    help="Save up to N replay files per category (natural-short, natural-long, "
                         "near-win, timeout) for GUI playback. 0 disables.")
    parser.add_argument('--replay_dir', type=str, default='./replays',
                    help="Directory where --replay_game writes replay files.")
    parser.add_argument('--job_timeout_hours', type=float, default=None,
                    help="Hours without a result before orphaned in-flight jobs are re-queued. "
                         "Useful when using spot/preemptible VMs. Default: from ini or 3.0.")


    parsed_args = parser.parse_args()

    # Load ML training params from the [MLTraining] ini section.
    # CLI args take precedence; ini provides defaults; hardcoded values are the last resort.
    _ini = configparser.ConfigParser()
    _ini.read(parsed_args.config_file)
    def _ini_float(key, fallback):
        try:
            return _ini.getfloat('MLTraining', key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    def _ini_int(key, fallback):
        try:
            return _ini.getint('MLTraining', key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback
    def _ini_bool(key, fallback):
        try:
            return _ini.getboolean('MLTraining', key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return fallback

    if parsed_args.near_win_rare_regions is None:
        parsed_args.near_win_rare_regions = _ini_int('near_win_rare_regions', 4)
    if parsed_args.near_win_bootstrap_pct is None:
        parsed_args.near_win_bootstrap_pct = _ini_float('near_win_bootstrap_pct', 0.50)
    if parsed_args.near_win_mcts_pct is None:
        parsed_args.near_win_mcts_pct = _ini_float('near_win_mcts_pct', 0.50)
    if parsed_args.near_win_pool_fraction is None:
        parsed_args.near_win_pool_fraction = _ini_float('near_win_pool_fraction', 0.30)
    if parsed_args.raregoods_pool_fraction is None:
        parsed_args.raregoods_pool_fraction = _ini_float('raregoods_pool_fraction', 0.15)
    if parsed_args.buffer_keep_every is None:
        parsed_args.buffer_keep_every = _ini_int('buffer_keep_every', 1)
    if parsed_args.timbuktu_pool_fraction is None:
        parsed_args.timbuktu_pool_fraction = _ini_float('timbuktu_pool_fraction', 0.0)
    if parsed_args.heuristic_guidance_weight is None:
        parsed_args.heuristic_guidance_weight = _ini_float('heuristic_guidance_weight', 0.40)
    if parsed_args.heuristic_guidance_weight_final is None:
        parsed_args.heuristic_guidance_weight_final = _ini_float(
            'heuristic_guidance_weight_final', parsed_args.heuristic_guidance_weight)
    if parsed_args.heuristic_guidance_decay_start is None:
        parsed_args.heuristic_guidance_decay_start = _ini_int('heuristic_guidance_decay_start', 300)
    if parsed_args.heuristic_guidance_decay_end is None:
        parsed_args.heuristic_guidance_decay_end = _ini_int('heuristic_guidance_decay_end', 2500)
    # Save the initial value so the schedule always interpolates from the same baseline.
    parsed_args.heuristic_guidance_weight_initial = parsed_args.heuristic_guidance_weight
    if parsed_args.mcts_buffer_fraction is None:
        parsed_args.mcts_buffer_fraction = _ini_float('mcts_buffer_fraction', 0.80)
    if parsed_args.replace_bootstrap_with_mcts is None:
        parsed_args.replace_bootstrap_with_mcts = _ini_bool('replace_bootstrap_with_mcts', False)

    def _ini_training_int(key, fallback):
        try:
            raw = _ini.get('Training', key).split('#')[0].split(';')[0].strip()
            return int(raw)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback

    parsed_args.base_max_play_moves = _ini_training_int('max_play_moves', 430)
    parsed_args.near_win_extension_moves = _ini_training_int('near_win_extension_moves', 20)
    parsed_args.near_win_extension_value_thresh = _ini_float('near_win_extension_value_thresh', 0.2)
    parsed_args.train_interval_seconds = _ini_int('train_interval_seconds', 10)
    parsed_args.early_termination_move = _ini_int('early_termination_move', 380)
    parsed_args.hopeless_move1   = _ini_int('hopeless_move1', 200)
    parsed_args.hopeless_thresh1 = _ini_float('hopeless_thresh1', 0.10)
    parsed_args.hopeless_nearwin_override1 = _ini_bool('hopeless_nearwin_override1', True)
    parsed_args.hopeless_move2   = _ini_int('hopeless_move2', 300)
    parsed_args.hopeless_thresh2 = _ini_float('hopeless_thresh2', 0.20)
    parsed_args.hopeless_nearwin_override2 = _ini_bool('hopeless_nearwin_override2', True)
    parsed_args.hopeless_move3   = _ini_int('hopeless_move3', 360)
    parsed_args.hopeless_thresh3 = _ini_float('hopeless_thresh3', 0.30)
    parsed_args.hopeless_nearwin_override3 = _ini_bool('hopeless_nearwin_override3', True)
    parsed_args.clear_winner_thresh = _ini_float('clear_winner_thresh', 0.35)
    parsed_args.random_no_kill_thresh = _ini_float('random_no_kill_thresh', 0.0)
    parsed_args.declining_best_thresh = _ini_float('declining_best_thresh', 0.0)
    if parsed_args.sim_tier1_sims is None:
        parsed_args.sim_tier1_sims = _ini_int('sim_tier1_sims', 150)
    if parsed_args.sim_tier2_start is None:
        parsed_args.sim_tier2_start = _ini_int('sim_tier2_start', 100)
    if parsed_args.sim_tier2_sims is None:
        parsed_args.sim_tier2_sims = _ini_int('sim_tier2_sims', 300)
    if parsed_args.sim_tier3_start is None:
        parsed_args.sim_tier3_start = _ini_int('sim_tier3_start', 300)
    if parsed_args.sim_tier3_sims is None:
        parsed_args.sim_tier3_sims = _ini_int('sim_tier3_sims', 500)
    if parsed_args.inference_server is None:
        parsed_args.inference_server = _ini_bool('inference_server', False)
    if parsed_args.inference_max_batch is None:
        parsed_args.inference_max_batch = _ini_int('inference_max_batch', 32)
    if parsed_args.inference_cpus is None:
        parsed_args.inference_cpus = _ini.get('MLTraining', 'inference_cpus', fallback='auto')
    if parsed_args.buffer_compresslevel is None:
        parsed_args.buffer_compresslevel = _ini_int('buffer_compresslevel', 1)
    if parsed_args.gamma is None:
        parsed_args.gamma = _ini_float('gamma', 0.997)
    if parsed_args.playout_cap_enabled is None:
        parsed_args.playout_cap_enabled = _ini_bool('playout_cap_enabled', False)
    if parsed_args.playout_cap_full_prob is None:
        parsed_args.playout_cap_full_prob = _ini_float('playout_cap_full_prob', 0.25)
    if parsed_args.playout_cap_fast_sims is None:
        parsed_args.playout_cap_fast_sims = _ini_int('playout_cap_fast_sims', 40)
    if parsed_args.playout_cap_exempt_route_decisions is None:
        parsed_args.playout_cap_exempt_route_decisions = _ini_bool('playout_cap_exempt_route_decisions', True)
    else:
        parsed_args.playout_cap_exempt_route_decisions = bool(parsed_args.playout_cap_exempt_route_decisions)
    if parsed_args.aux_targets is None:
        parsed_args.aux_targets = _ini_bool('aux_targets', False)
    else:
        parsed_args.aux_targets = bool(parsed_args.aux_targets)
    parsed_args.aux_win_type_weight = _ini_float('aux_win_type_weight', 0.5)
    parsed_args.aux_moves_left_weight = _ini_float('aux_moves_left_weight', 1.0)
    if parsed_args.sim_route_decision_sims is None:
        parsed_args.sim_route_decision_sims = _ini_int('sim_route_decision_sims', 500)
    if parsed_args.sim_route_decision_min_candidates is None:
        parsed_args.sim_route_decision_min_candidates = _ini_int('sim_route_decision_min_candidates', 20)
    if parsed_args.rare_goods_added_tier1_sims is None:
        parsed_args.rare_goods_added_tier1_sims = _ini_int('rare_goods_added_tier1_sims', 0)
    if parsed_args.rare_goods_actor_fraction is None:
        parsed_args.rare_goods_actor_fraction = _ini_float('rare_goods_actor_fraction', 0.0)
    if parsed_args.rare_goods_added_heuristic_weight is None:
        parsed_args.rare_goods_added_heuristic_weight = _ini_float('rare_goods_added_heuristic_weight', 0.30)
    if parsed_args.low_guidance_test_fraction is None:
        parsed_args.low_guidance_test_fraction = _ini_float('low_guidance_test_fraction', 0.0)
    if parsed_args.low_guidance_test_weight is None:
        parsed_args.low_guidance_test_weight = _ini_float('low_guidance_test_weight', 0.02)
    if parsed_args.job_timeout_hours is None:
        parsed_args.job_timeout_hours = _ini_float('job_timeout_hours', 3.0)

    if parsed_args.replay_buffer_size is None:
        parsed_args.replay_buffer_size = _ini_int('replay_buffer_size', None)

    # Verify no required MLTraining parameter is still unresolved.
    _required = {
        'replay_buffer_size':               parsed_args.replay_buffer_size,
        'mcts_buffer_fraction':             parsed_args.mcts_buffer_fraction,
        'near_win_rare_regions':            parsed_args.near_win_rare_regions,
        'near_win_bootstrap_pct':           parsed_args.near_win_bootstrap_pct,
        'near_win_mcts_pct':                parsed_args.near_win_mcts_pct,
        'near_win_pool_fraction':           parsed_args.near_win_pool_fraction,
        'heuristic_guidance_weight':        parsed_args.heuristic_guidance_weight,
        'heuristic_guidance_weight_final':  parsed_args.heuristic_guidance_weight_final,
        'heuristic_guidance_decay_start':   parsed_args.heuristic_guidance_decay_start,
        'heuristic_guidance_decay_end':     parsed_args.heuristic_guidance_decay_end,
    }
    _missing = [k for k, v in _required.items() if v is None]
    if _missing:
        print()
        print('ERROR: the following MLTraining parameters have no value (not in ini and not on CLI):')
        for k in _missing:
            print(f'  {k}')
        print('Set them in [MLTraining] in the ini file or pass them as CLI arguments.')
        sys.exit(1)

    print()
    print('--- MLTraining parameters (Ctrl-C to abort) ---')
    print(f'  replay_buffer_size          : {parsed_args.replay_buffer_size:,}')
    print(f'  mcts_buffer_fraction        : {parsed_args.mcts_buffer_fraction}')
    print(f'  near_win_rare_regions       : {parsed_args.near_win_rare_regions}')
    print(f'  near_win_bootstrap_pct      : {parsed_args.near_win_bootstrap_pct}')
    print(f'  near_win_mcts_pct           : {parsed_args.near_win_mcts_pct}')
    print(f'  near_win_pool_fraction      : {parsed_args.near_win_pool_fraction}')
    print(f'  raregoods_pool_fraction     : {parsed_args.raregoods_pool_fraction}')
    print(f'  timbuktu_pool_fraction      : {parsed_args.timbuktu_pool_fraction}')
    print(f'  buffer_keep_every           : {parsed_args.buffer_keep_every}')
    if parsed_args.rare_goods_actor_fraction > 0.0:
        _rg_t1_str = (f', tier1_sims={parsed_args.sim_tier1_sims}+{parsed_args.rare_goods_added_tier1_sims}'
                      if parsed_args.rare_goods_added_tier1_sims > 0 else '')
        print(f'  rare_goods_actor_fraction   : {parsed_args.rare_goods_actor_fraction} '
              f'(+heuristic={parsed_args.rare_goods_added_heuristic_weight}{_rg_t1_str})')
    else:
        print(f'  rare_goods_actor_fraction   : 0.0 (disabled)')
    if parsed_args.low_guidance_test_fraction > 0.0:
        print(f'  low_guidance_test_fraction  : {parsed_args.low_guidance_test_fraction} '
              f'(absolute weight={parsed_args.low_guidance_test_weight}, mutually exclusive with rare-goods focus)')
    else:
        print(f'  low_guidance_test_fraction  : 0.0 (disabled)')
    print(f'  heuristic_guidance_weight   : {parsed_args.heuristic_guidance_weight} → {parsed_args.heuristic_guidance_weight_final}')
    print(f'  heuristic_guidance_decay    : MCTS games {parsed_args.heuristic_guidance_decay_start}–{parsed_args.heuristic_guidance_decay_end}')
    print(f'  sim tiers (move → sims)     : '
          f'[0,{parsed_args.sim_tier2_start})→{parsed_args.sim_tier1_sims}  '
          f'[{parsed_args.sim_tier2_start},{parsed_args.sim_tier3_start})→{parsed_args.sim_tier2_sims}  '
          f'[{parsed_args.sim_tier3_start},∞)→{parsed_args.sim_tier3_sims}')
    print(f'  sim route-decision override : OPTIONAL_ROUTE with '
          f'≥{parsed_args.sim_route_decision_min_candidates} candidates → '
          f'{parsed_args.sim_route_decision_sims} sims (overrides move-tier budget above)')
    if parsed_args.playout_cap_enabled:
        print(f'  playout cap                 : ON  full search on {parsed_args.playout_cap_full_prob:.0%} '
              f'of moves, else {parsed_args.playout_cap_fast_sims} sims (value-only); large route '
              f'decisions {"exempt" if parsed_args.playout_cap_exempt_route_decisions else "capped"}')
    else:
        print(f'  playout cap                 : off')
    if parsed_args.aux_targets:
        print(f'  aux targets                 : ON  win type (weight {parsed_args.aux_win_type_weight:g}), '
              f'moves left (weight {parsed_args.aux_moves_left_weight:g})')
    else:
        print(f'  aux targets                 : off')
    print(f'  job_timeout_hours           : {parsed_args.job_timeout_hours}')
    print()

    try:
        raw = input('Oversampling threshold? [0] for none, default [350]: ').strip()
        if raw == '':
            parsed_args.oversample_threshold = 350
        elif raw == '0':
            parsed_args.oversample_threshold = 0
        else:
            val = int(raw)
            parsed_args.oversample_threshold = val if val > 0 else 350
    except (ValueError, EOFError):
        parsed_args.oversample_threshold = 350

    ot = parsed_args.oversample_threshold
    if ot > 0:
        print(f'  Oversampling threshold set to {ot} (wins < {ot} moves inserted 3x into buffer).')
    else:
        print('  Oversampling disabled.')

    main(parsed_args)