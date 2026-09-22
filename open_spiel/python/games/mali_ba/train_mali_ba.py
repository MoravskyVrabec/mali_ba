import os
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

    log(LogLevel.INFO, "Trainer process started.")

    temp_game = pyspiel.load_game(args.game_name, initial_game_params)
    agent = SimpleAgent(
        temp_game.observation_tensor_shape(),
        temp_game.num_distinct_actions(),
        temp_game.num_players(),
        args.learning_rate
    )
    del temp_game

    if args.load_model_path and os.path.exists(args.load_model_path.replace("weights.h5", "_policy.weights.h5")):
        try:
            agent.load_model(args.load_model_path)
            log(LogLevel.INFO, "Trainer loaded initial model weights.")
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
                                        replace_bootstrap_with_mcts=args.replace_bootstrap_with_mcts)
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
            # Copy into correctly-sized deques so the current run's maxlen is respected,
            # not the maxlen that was baked into the saved deques.
            saved_bootstrap = list(saved['bootstrap_buffer'])
            # Support both old single-pool saves and new two-sub-pool saves.
            saved_natural = list(saved.get('mcts_natural_buffer',
                                           saved.get('mcts_buffer', [])))
            saved_nearwin   = list(saved.get('mcts_nearwin_buffer', []))
            saved_raregoods = list(saved.get('mcts_raregoods_buffer', []))
            # Migration: old buffers lack the raregoods pool — split natural 50/50 to seed it.
            if 'mcts_raregoods_buffer' not in saved and saved_natural:
                half = len(saved_natural) // 2
                saved_raregoods = saved_natural[half:]
                saved_natural   = saved_natural[:half]
                log(LogLevel.INFO,
                    f"Trainer: Migrating old buffer — split {len(saved_natural) + len(saved_raregoods)} "
                    f"natural entries 50/50 into natural ({len(saved_natural)}) "
                    f"and raregoods ({len(saved_raregoods)}) pools.")
            bootstrap_cap  = local_replay_buffer.bootstrap_buffer.maxlen
            natural_cap    = local_replay_buffer.mcts_natural_buffer.maxlen
            nearwin_cap    = local_replay_buffer.mcts_nearwin_buffer.maxlen
            raregoods_cap  = local_replay_buffer.mcts_raregoods_buffer.maxlen
            if len(saved_bootstrap) > bootstrap_cap:
                log(LogLevel.WARN,
                    f"Trainer: Saved bootstrap pool ({len(saved_bootstrap)}) exceeds new capacity "
                    f"({bootstrap_cap}). Sampling {bootstrap_cap} entries randomly.")
                saved_bootstrap = random.sample(saved_bootstrap, bootstrap_cap)
            if len(saved_natural) > natural_cap:
                log(LogLevel.WARN,
                    f"Trainer: Saved MCTS-natural pool ({len(saved_natural)}) exceeds new capacity "
                    f"({natural_cap}). Keeping most recent {natural_cap} entries.")
                saved_natural = saved_natural[-natural_cap:]
            if len(saved_nearwin) > nearwin_cap:
                log(LogLevel.WARN,
                    f"Trainer: Saved MCTS-nearwin pool ({len(saved_nearwin)}) exceeds new capacity "
                    f"({nearwin_cap}). Keeping most recent {nearwin_cap} entries.")
                saved_nearwin = saved_nearwin[-nearwin_cap:]
            if len(saved_raregoods) > raregoods_cap:
                log(LogLevel.WARN,
                    f"Trainer: Saved MCTS-raregoods pool ({len(saved_raregoods)}) exceeds new capacity "
                    f"({raregoods_cap}). Keeping most recent {raregoods_cap} entries.")
                saved_raregoods = saved_raregoods[-raregoods_cap:]
            local_replay_buffer.bootstrap_buffer      = collections.deque(saved_bootstrap, maxlen=bootstrap_cap)
            local_replay_buffer.mcts_natural_buffer   = collections.deque(saved_natural,   maxlen=natural_cap)
            local_replay_buffer.mcts_nearwin_buffer   = collections.deque(saved_nearwin,   maxlen=nearwin_cap)
            local_replay_buffer.mcts_raregoods_buffer = collections.deque(saved_raregoods, maxlen=raregoods_cap)
            log(LogLevel.INFO,
                f"Trainer: Restored buffer from {args.save_buffer_path} — "
                f"bootstrap={len(local_replay_buffer.bootstrap_buffer)}, "
                f"mcts_natural={len(local_replay_buffer.mcts_natural_buffer)}, "
                f"mcts_nearwin={len(local_replay_buffer.mcts_nearwin_buffer)}, "
                f"mcts_raregoods={len(local_replay_buffer.mcts_raregoods_buffer)} experiences.")
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
                            with gzip.open(tmp_path, 'wb') as f:
                                pickle.dump({
                                    'bootstrap_buffer':      local_replay_buffer.bootstrap_buffer,
                                    'mcts_natural_buffer':   local_replay_buffer.mcts_natural_buffer,
                                    'mcts_nearwin_buffer':   local_replay_buffer.mcts_nearwin_buffer,
                                    'mcts_raregoods_buffer': local_replay_buffer.mcts_raregoods_buffer,
                                }, f)
                            os.replace(tmp_path, args.save_buffer_path)
                            log(LogLevel.INFO, f"Trainer: Buffer saved on shutdown to {args.save_buffer_path}.")
                        except Exception as e:
                            log(LogLevel.ERROR, f"Trainer: Buffer save on shutdown failed: {e}")
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
                # experience is (obs, policy, value, is_bootstrap [, is_near_win [, is_rare_goods]])
                is_bootstrap  = experience[3] if len(experience) > 3 else False
                is_near_win   = experience[4] if len(experience) > 4 else False
                is_rare_goods = experience[5] if len(experience) > 5 else False
                local_replay_buffer.add(experience[:3], is_bootstrap=is_bootstrap,
                                        is_near_win=is_near_win, is_rare_goods=is_rare_goods)
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
                f"MCTS-natural: {len(local_replay_buffer.mcts_natural_buffer)}  "
                f"MCTS-nearwin: {len(local_replay_buffer.mcts_nearwin_buffer)}  "
                f"MCTS-raregoods: {len(local_replay_buffer.mcts_raregoods_buffer)}")

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
                    with gzip.open(tmp_path, 'wb') as f:
                        pickle.dump({
                            'bootstrap_buffer':      local_replay_buffer.bootstrap_buffer,
                            'mcts_natural_buffer':   local_replay_buffer.mcts_natural_buffer,
                            'mcts_nearwin_buffer':   local_replay_buffer.mcts_nearwin_buffer,
                            'mcts_raregoods_buffer': local_replay_buffer.mcts_raregoods_buffer,
                        }, f)
                    os.replace(tmp_path, args.save_buffer_path)
                    log(LogLevel.INFO,
                        f"Trainer: Buffer saved to {args.save_buffer_path} — "
                        f"bootstrap={len(local_replay_buffer.bootstrap_buffer)}, "
                        f"mcts_natural={len(local_replay_buffer.mcts_natural_buffer)}, "
                        f"mcts_nearwin={len(local_replay_buffer.mcts_nearwin_buffer)}, "
                        f"mcts_raregoods={len(local_replay_buffer.mcts_raregoods_buffer)} experiences.")
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


def actor_process(actor_id, game_params, args, job_queue, result_queue, games_per_actor):
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
    log(LogLevel.INFO, f"Actor {actor_id} started, configured for CPU-only execution.")

    # --- Create ONE game object and ONE model for the actor's lifetime ---
    log(LogLevel.INFO, f"Actor {actor_id}: Initializing its game instance and model.")
    game = pyspiel.load_game(args.game_name, game_params)
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

        # Set weights on the two separate models
        policy_model.set_weights(policy_weights)
        value_model.set_weights(value_weights)
        
        random.seed(game_rng_seed)
        np.random.seed(game_rng_seed)
        move_count = 0
        
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
                                       heuristic_guidance_weight=job_heuristic_weight)
        
        bot = mcts.MCTSBot(
            game=game, uct_c=args.uct_c, max_simulations=args.max_simulations,
            evaluator=evaluator, solve=False,
            dirichlet_noise=(0.2, 0.25),
            child_selection_fn=mcts.SearchNode.puct_value, verbose=False)

        state = game.new_initial_state()
        
        # Chance node startup
        episode_trajectory = []
        if state.is_chance_node():
            state.apply_action(state.legal_actions()[0])

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

                mali_ba_state.apply_action(action)
            # --- END OF OPTIMIZATION ---

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
            observation = np.array(state.observation_tensor(), dtype=np.float32)
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
                        if (len(_declining_best) == 2
                                and all(v < 0 for v in _declining_best)
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
            if pyspiel.mali_ba.downcast_state(state).current_phase() == pyspiel.mali_ba.Phase.OPTIONAL_ROUTE:
                _n_route_candidates = len(state.legal_actions())
                if _n_route_candidates >= getattr(args, 'sim_route_decision_min_candidates', 20):
                    bot.max_simulations = max(bot.max_simulations,
                                               getattr(args, 'sim_route_decision_sims', 500))
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

            # For the replay buffer, we need the policy over the FULL action space
            mcts_policy_full = np.zeros(game.num_distinct_actions())
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

            state.apply_action(action)
            reward_vector = state.rewards()  # Called after apply_action so Rewards() sees the move in moves_history_
            episode_trajectory.append((observation, player, mcts_policy_full, reward_vector))
            if replay_file:
                try:
                    replay_move_num += 1
                    _sj = pyspiel.mali_ba.downcast_state(state).serialize()
                    replay_file.write(
                        f"[move{replay_move_num}]\naction={action_str}\nstate={_sj}\n")
                except Exception as _e:
                    log(LogLevel.WARN,
                        f"Actor {actor_id}: Replay write failed at move {replay_move_num}: {_e}")
            move_count += 1

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
            state.apply_action(state.legal_actions()[0])

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


def spawn_actor(actor_id, initial_game_params, args, job_queue, result_queue, actor_pool, actor_function):
    """
    Creates, starts, and tracks a new actor process using the specified actor function.
    """
    p = mp.Process(target=actor_function, args=(
        actor_id, initial_game_params, args, job_queue, result_queue, args.games_per_actor))
    p.start()
    actor_pool[p] = actor_id # Associates the process object with its ID
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
            'rare_goods_actor_fraction':           getattr(args, 'rare_goods_actor_fraction', 0.0),
            'rare_goods_added_heuristic_weight':   getattr(args, 'rare_goods_added_heuristic_weight', 0.30),
            'rare_goods_added_tier1_sims':         getattr(args, 'rare_goods_added_tier1_sims', 0),
            'job_timeout_hours':                   getattr(args, 'job_timeout_hours', 3.0),
            'sim_tier1_sims':                      getattr(args, 'sim_tier1_sims', 150),
            'sim_tier2_start':                     getattr(args, 'sim_tier2_start', 100),
            'sim_tier2_sims':                      getattr(args, 'sim_tier2_sims', 300),
            'sim_tier3_start':                     getattr(args, 'sim_tier3_start', 300),
            'sim_tier3_sims':                      getattr(args, 'sim_tier3_sims', 500),
            'sim_route_decision_sims':              getattr(args, 'sim_route_decision_sims', 500),
            'sim_route_decision_min_candidates':    getattr(args, 'sim_route_decision_min_candidates', 20),
            'base_max_play_moves':                 getattr(args, 'base_max_play_moves', 430),
            'near_win_extension_moves':            getattr(args, 'near_win_extension_moves', 20),
            'near_win_extension_value_thresh':     getattr(args, 'near_win_extension_value_thresh', 0.2),
            'debug':                               getattr(args, 'debug', False),
        }
        server_thread = threading.Thread(
            target=start_server,
            args=(job_queue, result_queue, shared_config, log_queue),
            kwargs={'host': args.bind_host, 'port': args.queue_port, 'authkey': args.authkey.encode()},
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
    jobs_dispatched = 0
    jobs_timed_out = 0        # cumulative jobs assumed lost to spot preemption / crash
    start_time = time.time()
    last_weights_update_time = time.time()
    last_result_time = time.time()  # last time a game result was received
    
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
        log(LogLevel.INFO, "Learner received initial weights.")

    # --- 3. UNIFIED Main Learner Loop ---
    bootstrap_transition_logged = False
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
            
            # Immediately spawn its replacement
            spawn_actor(next_actor_id, initial_game_params, args, job_queue, result_queue, actor_pool, current_actor_function)
            next_actor_id += 1
            
        # This separate loop is now only necessary for the initial startup,
        # but it's harmless to keep it for ensuring the pool is always full.
        while len(actor_pool) < args.num_actors:
            log(LogLevel.INFO, f"Actor pool below target ({len(actor_pool)}/{args.num_actors}). Spawning new actor.")
            spawn_actor(next_actor_id, initial_game_params, args, job_queue, result_queue, actor_pool, current_actor_function)
            next_actor_id += 1
            
        # --- Maintain a healthy job queue size ---
        # This logic remains the same and is correct.
        target_job_queue_size = total_actors * 2
        while job_queue.qsize() < target_job_queue_size and jobs_dispatched < args.num_episodes + jobs_timed_out:
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

            # Process the game result
            total_games_processed += 1
            phase_label = "Bootstrap" if total_games_processed <= args.bootstrap_episodes else "MCTS"
            log(LogLevel.INFO, f"LEARNER ({phase_label}) RECEIVED GAME #{total_games_processed}/{args.num_episodes}. "
                            f"Length: {len(trajectory)} moves. Returns: {returns}")

            GAMMA = 0.997  # Discount factor. Rewards further in the future are worth slightly less.

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
            log(LogLevel.INFO, f"  [DBG] game={total_games_processed} bootstrap={is_bootstrap_game} "
                f"timeout={is_timeout_game} near_win={near_win_flag} rare_goods={is_rare_goods_game} "
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

            log(LogLevel.INFO, f"  [DBG] Game passed filters — queuing {len(trajectory_with_values)} experiences "
                f"(oversample x{oversample_factor}). bootstrap={is_bootstrap_game} near_win={near_win_flag}")
            queued_count = 0
            for _ in range(oversample_factor):
                for observation, player, policy_target, value_target_vector in trajectory_with_values:
                    player_value = value_target_vector[player]
                    value_data = (player, player_value, value_target_vector)

                    # Always send to trainer — bootstrap games are tagged so the
                    # trainer can route them to the correct buffer pool.
                    # (heuristic_only skips this entirely since there is no trainer.)
                    if not args.heuristic_only:
                        if not replay_buffer_queue.full():
                            replay_buffer_queue.put((observation, policy_target, value_data, is_bootstrap_game, near_win_flag and is_timeout_game, is_rare_goods_game))
                            queued_count += 1
                        else:
                            log(LogLevel.INFO, f"  [DBG] Replay buffer queue FULL after {queued_count} puts.")
                            break
            if not args.heuristic_only:
                log(LogLevel.INFO, f"  [DBG] Queued {queued_count} experiences to replay_buffer_queue.")

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
                log(LogLevel.INFO, f"Learner updated to latest weights at game #{total_games_processed}. Distributing to {len(actor_pool)} actors.")

            
            while not stats_queue.empty():
                try:
                    stats = stats_queue.get_nowait()
                    if "loss" in stats:
                        log(LogLevel.INFO, f"Trainer reported loss: {stats['loss']:.4f} at game #{total_games_processed}")
                except: break

            # --- Job timeout: re-queue jobs assumed lost to spot preemption ---
            _job_timeout_secs = getattr(args, 'job_timeout_hours', 3.0) * 3600
            _time_since_result = time.time() - last_result_time
            if _time_since_result > _job_timeout_secs:
                _inflight = max(0, jobs_dispatched - total_games_processed - job_queue.qsize())
                if _inflight > 0:
                    jobs_timed_out += _inflight
                    last_result_time = time.time()  # reset to avoid immediate re-fire
                    log(LogLevel.WARN,
                        f"Job timeout: no result received in {_time_since_result/3600:.1f}h. "
                        f"Assuming {_inflight} orphaned job(s) lost (spot preemption?). "
                        f"Cumulative lost: {jobs_timed_out}. Re-queuing.")

            last_weights_update_time = time.time()

    # --- 4. Final Shutdown (Same) ---
    log(LogLevel.INFO, "All episodes processed. Sending shutdown signals...")
    if not args.heuristic_only:
        replay_buffer_queue.put(None)

    while not job_queue.empty():
        try: job_queue.get_nowait()
        except: break

    for _ in range(len(actor_pool)):
        job_queue.put(None)

    if trainer is not None:
        log(LogLevel.INFO, "Waiting for trainer to terminate...")
        trainer.join(timeout=180)
        if trainer.is_alive():
            log(LogLevel.WARN, "Trainer did not terminate gracefully. Forcing.")
            trainer.terminate()

    log(LogLevel.INFO, "Waiting for actors to terminate...")
    for p in actor_pool:
        p.join(timeout=60)
        if p.is_alive(): p.terminate()

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
                    help="When the MCTS natural-win pool is full, promote evicted entries into "
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