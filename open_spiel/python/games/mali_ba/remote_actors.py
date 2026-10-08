"""
Remote actor worker for distributed Mali-Ba training.

Run this on a second machine to contribute actor processes to an existing
train_mali_ba.py session that was started with --distributed.

Both machines must have the same codebase and compiled pyspiel/mali_ba.

Actors always run MCTS on the CPU. Neural-network inference is the real cost:
with TF single-threaded (which actors force, to avoid oversubscribing the CPU)
one batch-1 forward pass through both nets costs ~37ms, and that is essentially
all of self-play time. If this machine has a GPU, pass --inference_server to run
one process that owns it and batches every local actor's evaluations over shared
memory; measured ~7x more games on the trainer host. Without it each actor does
its own single-threaded CPU inference.

Setup on the remote machine
---------------------------
1. Clone/sync the repo so the code matches the training server exactly.
2. Activate the same conda/venv environment (same Python, TF, OpenSpiel).
3. Run:

    cd <repo>/open_spiel/python/games/mali_ba
    python remote_actors.py --server_host 192.168.1.100 --num_actors 8

Usage
-----
    python remote_actors.py \\
        --server_host <IP of training machine> \\
        [--server_port 50000] \\
        [--num_actors 8] \\
        [--authkey malibatraining2024] \\
        [--inference_server] [--inference_max_batch 32] \\
        [--cpu_only]
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
import types

# Log lines from remote actors that match any of these substrings are dropped
# from the relay to keep the desktop log uncluttered.  Remove entries here if
# you ever need those lines in the desktop log.
_LOG_FILTER_SUBSTRINGS = [
    ' plays ',       # per-move "Player X plays action Y" lines
    'plays action',
]


# Watchdog (see --watchdog_minutes). The env var counts restarts across os.execv.
_WATCHDOG_ENV = 'MALIBA_REMOTE_WATCHDOG_RESTARTS'
_WATCHDOG_POLL_S = 120          # how often to ask the trainer
_WATCHDOG_MAX_FAILURES = 5      # consecutive unanswered checks (~10 min) before restarting
_RECONNECT_WINDOW_S = 30 * 60   # after a restart, keep retrying the connection this long


def _should_relay(line):
    for frag in _LOG_FILTER_SUBSTRINGS:
        if frag in line:
            return False
    return True


def _actor_worker(actor_id, initial_game_params, actor_args,
                  job_queue, result_queue, games_per_actor, log_queue=None,
                  arena=None, server_weights_queue=None, inference_slot=None):
    """
    Thin wrapper that imports actor_process from train_mali_ba and runs it.
    Defined at module level so it is picklable for mp.Process on all platforms.

    If log_queue is provided, stdout and stderr are redirected to a pipe and
    a background thread relays each line back to the desktop via log_queue so
    that C++ log() output (value checks, game summaries, etc.) appears in the
    desktop's tee'd training log.
    """
    import threading

    relay_thread = None
    if log_queue is not None:
        orig_fd = os.dup(1)   # save original stdout before redirect
        r_fd, w_fd = os.pipe()
        os.dup2(w_fd, 1)  # redirect stdout → pipe write end
        os.dup2(w_fd, 2)  # redirect stderr → pipe write end
        os.close(w_fd)    # close the extra copy; fds 1 & 2 are the only write ends

        def _relay():
            with os.fdopen(r_fd, 'r', errors='replace', buffering=1) as pipe, \
                 os.fdopen(orig_fd, 'w', errors='replace', buffering=1) as local_out:
                for line in pipe:
                    stripped = line.rstrip('\n')
                    if not stripped:
                        continue
                    local_out.write(stripped + '\n')  # always visible on laptop terminal
                    local_out.flush()
                    if _should_relay(stripped):       # filtered subset to desktop
                        try:
                            log_queue.put(stripped, timeout=2.0)
                        except Exception:
                            pass  # drop if queue is full or connection lost

        relay_thread = threading.Thread(target=_relay, daemon=True)
        relay_thread.start()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from train_mali_ba import actor_process
    try:
        actor_process(actor_id, initial_game_params, actor_args,
                      job_queue, result_queue, games_per_actor,
                      arena=arena, server_weights_queue=server_weights_queue,
                      inference_slot=inference_slot)
    finally:
        if relay_thread is not None:
            # Replace fd 1/2 with devnull so the relay thread sees EOF and exits.
            dn = os.open(os.devnull, os.O_WRONLY)
            os.dup2(dn, 1)
            os.dup2(dn, 2)
            os.close(dn)
            relay_thread.join(timeout=5.0)


def main():
    parser = argparse.ArgumentParser(
        description='Remote actor worker for distributed Mali-Ba training')
    parser.add_argument('--server_host', required=True,
                        help='IP address or hostname of the training server')
    parser.add_argument('--server_port', type=int, default=50000)
    parser.add_argument('--num_actors', type=int, default=8,
                        help='Number of parallel actor processes to run on this machine')
    parser.add_argument('--authkey', type=str, default='malibatraining2024',
                        help='Shared secret — must match --authkey on the server')
    parser.add_argument('--actor_id_start', type=int, default=100000,
                        help='Starting actor ID for this remote process '
                             '(use different values per machine to avoid log collisions)')
    parser.add_argument('--inference_server', action='store_true',
                        help='Run a batched inference server on this machine\'s GPU and '
                             'route every local actor through it. Actors otherwise do '
                             'their own single-threaded CPU inference at ~37ms per call, '
                             'which is essentially all of self-play cost.')
    parser.add_argument('--inference_max_batch', type=int, default=32,
                        help='Largest batch the inference server assembles (default 32).')
    parser.add_argument('--inference_cpus', type=str, default='auto',
                        help="CPUs to pin the inference server to: 'auto' (default: "
                             "performance cores on an Intel hybrid CPU, or the large-L3 cores "
                             "on a split-cache Ryzen; otherwise unpinned), 'none', or a list "
                             "like '0-15'.")
    parser.add_argument('--inference_vram_mb', type=int, default=4096,
                        help='VRAM cap for the inference server (default 4096 MB).')
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU-only TF even if a GPU is present')
    parser.add_argument('--watchdog_minutes', type=float, default=20.0,
                        help='Restart this worker (stop its actors, reconnect, respawn) when '
                             'the trainer has received none of its games for this long. '
                             'Needs a trainer with RemoteStatus; 0 disables (default 20).')
    args = parser.parse_args()

    if args.cpu_only and args.inference_server:
        print("[Remote] --cpu_only with --inference_server would put the server on the "
              "CPU, where batching is no faster than per-actor inference "
              "(batch-32 single-threaded is 36.6ms/sample vs 37.1 at batch 1). "
              "Ignoring --cpu_only; actors remain CPU-only regardless.")
    elif args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # --- Connect to the queue server ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from queue_server import connect_client

    print(f"[Remote] Connecting to queue server at {args.server_host}:{args.server_port}...")
    # After a watchdog restart the network may still be recovering, so keep trying
    # for a while; a first manual start fails fast as before.
    restarted = os.environ.get(_WATCHDOG_ENV) is not None
    deadline = time.time() + (_RECONNECT_WINDOW_S if restarted else 0)
    while True:
        try:
            manager = connect_client(
                host=args.server_host,
                port=args.server_port,
                authkey=args.authkey.encode()
            )
            break
        except OSError as e:
            if time.time() < deadline:
                print(f"[Remote] Connect failed ({e}); retrying in 30 s...", flush=True)
                time.sleep(30)
                continue
            print(f"[Remote] ERROR: Could not connect ({e}). "
                  f"Is train_mali_ba.py running with --distributed?")
            sys.exit(1)

    job_queue    = manager.get_job_queue()
    result_queue = manager.get_result_queue()

    # get_config() returns a proxy to the dict on the server.
    # We call _getvalue() once to get a plain local copy.
    config = manager.get_config()._getvalue()

    print(f"[Remote] Connected. Config received:")
    print(f"  game_name:       {config['game_name']}")
    print(f"  max_simulations: {config['max_simulations']}")
    print(f"  uct_c:           {config['uct_c']}")
    print(f"  games_per_actor: {config['games_per_actor']}")

    # Try to get the log relay queue (requires server running updated queue_server.py).
    try:
        log_queue = manager.get_log_queue()
        print(f"[Remote] Log relay enabled — actor output will appear in desktop log.")
    except Exception:
        log_queue = None
        print(f"[Remote] Log relay not available (server may be running older code).")

    # Watchdog: ask the trainer when it last received a game from each of our actors.
    status_proxy = None
    wd_t0 = None
    if args.watchdog_minutes > 0:
        try:
            status_proxy = manager.get_remote_status()
            wd_t0 = status_proxy.snapshot()['now']   # trainer clock; ignore older arrivals
            print(f"[Remote] Watchdog on: restarts this worker if the trainer receives none "
                  f"of its games for {args.watchdog_minutes:g} min"
                  + (f" (restart #{os.environ.get(_WATCHDOG_ENV)})" if restarted else "")
                  + ".", flush=True)
        except Exception as e:
            status_proxy = None
            print(f"[Remote] Watchdog off: trainer does not report arrivals "
                  f"(older train_mali_ba.py / queue_server.py?): {e}", flush=True)

    # Reconstruct an args-like namespace for actor_process.
    # Only the fields actor_process actually reads are needed.
    actor_args = types.SimpleNamespace(
        game_name=config['game_name'],
        uct_c=config['uct_c'],
        max_simulations=config['max_simulations'],
        games_per_actor=config['games_per_actor'],
        heuristic_guidance_weight=config.get('heuristic_guidance_weight', 0.40),
        heuristic_guidance_weight_initial=config.get('heuristic_guidance_weight_initial', config.get('heuristic_guidance_weight', 0.40)),
        heuristic_guidance_weight_final=config.get('heuristic_guidance_weight_final', config.get('heuristic_guidance_weight', 0.40)),
        heuristic_guidance_decay_start=config.get('heuristic_guidance_decay_start', 0),
        heuristic_guidance_decay_end=config.get('heuristic_guidance_decay_end', 1),
        bootstrap_episodes=config.get('bootstrap_episodes', 0),
        near_win_rare_regions=config.get('near_win_rare_regions', 4),
        early_termination_move=config.get('early_termination_move', 380),
        hopeless_move1=config.get('hopeless_move1', 200),
        hopeless_thresh1=config.get('hopeless_thresh1', 0.10),
        hopeless_move2=config.get('hopeless_move2', 300),
        hopeless_thresh2=config.get('hopeless_thresh2', 0.20),
        hopeless_move3=config.get('hopeless_move3', 360),
        hopeless_thresh3=config.get('hopeless_thresh3', 0.30),
        clear_winner_thresh=config.get('clear_winner_thresh', 0.35),
        declining_best_thresh=config.get('declining_best_thresh', 0.0),
        stalled_thresh=config.get('stalled_thresh', 0.20),
        # Sent by the server but previously not copied, so remote actors used the
        # defaults (exempt near-win games at every level, never no-kill) while
        # desktop actors followed the ini -- the two machines culled differently.
        hopeless_nearwin_override1=config.get('hopeless_nearwin_override1', True),
        hopeless_nearwin_override2=config.get('hopeless_nearwin_override2', True),
        hopeless_nearwin_override3=config.get('hopeless_nearwin_override3', True),
        random_no_kill_thresh=config.get('random_no_kill_thresh', 0.0),
        debug=config.get('debug', False),
    )
    # Copy every other key the server sends. The list above predates the sim tiers,
    # route-decision budget, playout cap, rare-goods extras and near-win extension;
    # the server sent them but they were dropped here, so remote actors silently ran
    # actor_process's getattr defaults (e.g. tier3 500 sims instead of the ini's 600,
    # and playout cap always off).
    for _k, _v in config.items():
        if not hasattr(actor_args, _k):
            setattr(actor_args, _k, _v)
    initial_game_params = config['initial_game_params']

    # --- Optional batched inference server on this machine's GPU ---------------
    # Unlike the trainer host, this process has no direct source of weights: they
    # only arrive inside the jobs the actors pull off the queue. So each actor
    # republishes its job's weights to the server, which coalesces to the newest
    # (see actor_process). That means the server sits idle until the first actor
    # picks up a job, which is expected.
    inference_arena = None
    inference_server_proc = None
    inference_weights_queue = None
    inference_stop = None
    inference_free_slots = []
    inference_slot_by_proc = {}
    if args.inference_server:
        try:
            import pyspiel
            from inference_server import InferenceArena, server_loop, STOP as INF_STOP
            _g = pyspiel.load_game(config['game_name'], initial_game_params)
            _shape = _g.observation_tensor_shape()
            _obs = 1
            for _d in _shape:
                _obs *= _d
            _n_slots = max(1, args.num_actors) + 4
            inference_arena = InferenceArena(_n_slots, _obs,
                                             _g.num_distinct_actions(), _g.num_players())
            inference_weights_queue = mp.Queue()
            inference_stop = mp.Event()
            inference_server_proc = mp.Process(
                target=server_loop,
                args=(inference_arena, _shape, initial_game_params,
                      inference_weights_queue, inference_stop),
                kwargs=dict(max_batch=args.inference_max_batch,
                            gpu_memory_limit_mb=args.inference_vram_mb,
                            log_every=300,
                            cpus=args.inference_cpus),
                # daemon so a killed worker cannot leave the server orphaned and
                # still holding VRAM.
                daemon=True)
            inference_server_proc.start()
            # The server sets gpu_ok as soon as it confirms a GPU, before any
            # weights arrive. If it does not, it has refused to serve (a batched
            # CPU server is far slower than per-actor inference), so drop the arena
            # and let actors infer locally.
            if not inference_arena.gpu_ok.wait(180):
                print("[Remote] Inference server reported no usable GPU, so it will not "
                      "serve. Actors will use local CPU inference instead. See the "
                      "InferenceServer error above -- most likely the CUDA runtime "
                      "wheels are missing (pip install 'tensorflow[and-cuda]').",
                      flush=True)
                inference_arena = None
                inference_weights_queue = None
                if inference_server_proc.is_alive():
                    inference_stop.set()
                    inference_server_proc.join(timeout=15)
                inference_server_proc = None
            else:
                inference_free_slots = list(range(_n_slots))
                print(f"[Remote] Inference server started (slots={_n_slots}, "
                      f"max_batch={args.inference_max_batch}, "
                      f"vram_cap={args.inference_vram_mb}MB). Waiting for the first "
                      f"actor's weights...", flush=True)
        except Exception as e:
            print(f"[Remote] ERROR: could not start inference server ({e}). "
                  f"Actors will use local CPU inference.", flush=True)
            inference_arena = None
            inference_server_proc = None

    # --- Actor pool management ---
    actor_pool = {}
    next_actor_id = args.actor_id_start

    def spawn():
        nonlocal next_actor_id
        # Slots are assigned here rather than claimed in the child so a finished
        # actor's slot can be reused; claiming in the child leaks one per respawn
        # and silently drops later actors back to CPU inference once the arena fills.
        slot = inference_free_slots.pop(0) if inference_free_slots else None
        p = mp.Process(
            target=_actor_worker,
            args=(next_actor_id, initial_game_params, actor_args,
                  job_queue, result_queue, actor_args.games_per_actor, log_queue,
                  inference_arena, inference_weights_queue, slot),
            daemon=False
        )
        p.start()
        actor_pool[p] = next_actor_id
        if slot is not None:
            inference_slot_by_proc[p] = slot
        print(f"[Remote] Spawned actor {next_actor_id}"
              + (f" (inference slot {slot})" if slot is not None else ""), flush=True)
        next_actor_id += 1

    # Initial spawn
    print(f"[Remote] Starting {args.num_actors} actor(s)...")
    for _ in range(args.num_actors):
        spawn()

    def shutdown():
        for p in list(actor_pool):
            p.terminate()
        for p in list(actor_pool):
            p.join(timeout=5)
        if inference_server_proc is not None:
            try:
                if inference_arena is not None:
                    st = inference_arena.stats()
                    print(f"[Remote] InferenceServer totals: {st['requests']:,} evals in "
                          f"{st['batches']:,} batches (avg batch {st['avg_batch']:.1f}, "
                          f"{st['ms_per_request']:.3f} ms/eval)")
                if inference_stop is not None:
                    inference_stop.set()
                if inference_weights_queue is not None:
                    inference_weights_queue.put(INF_STOP)
                inference_server_proc.join(timeout=30)
                if inference_server_proc.is_alive():
                    inference_server_proc.terminate()
            except Exception as e:
                print(f"[Remote] Inference server shutdown issue: {e}")

    def watchdog_stale_seconds():
        """Seconds since the trainer last received a game from one of our actors (by
        the trainer's clock, counting only arrivals since this process started), or
        None if the trainer could not be asked. The call runs in a thread with a
        timeout, since a broken connection can make it hang rather than fail."""
        import threading
        out = {}

        def ask():
            try:
                out['snap'] = status_proxy.snapshot()
            except Exception as e:
                out['err'] = e
        t = threading.Thread(target=ask, daemon=True)
        t.start()
        t.join(timeout=60)
        snap = out.get('snap')
        if snap is None:
            print(f"[Remote] Watchdog: could not reach the trainer "
                  f"({out.get('err', 'no reply within 60 s')})", flush=True)
            return None
        mine = [t for aid, t in snap['last'].items()
                if args.actor_id_start <= aid < next_actor_id and t >= wd_t0]
        return snap['now'] - (max(mine) if mine else wd_t0)

    # Keep the pool full: respawn actors that finish their quota
    last_stat = time.time()
    last_wd = time.time()
    wd_failures = 0
    try:
        while True:
            dead = [p for p in list(actor_pool) if not p.is_alive()]
            for p in dead:
                aid = actor_pool.pop(p)
                p.join()
                freed = inference_slot_by_proc.pop(p, None)
                if freed is not None:
                    inference_free_slots.append(freed)
                print(f"[Remote] Actor {aid} finished, respawning...", flush=True)
                spawn()
            if inference_arena is not None and time.time() - last_stat >= 300:
                st = inference_arena.stats()
                print(f"[Remote] InferenceServer: {st['requests']:,} evals in "
                      f"{st['batches']:,} batches (avg batch {st['avg_batch']:.1f}, "
                      f"{st['avg_infer_ms']:.2f} ms/batch, "
                      f"{st['ms_per_request']:.3f} ms/eval)", flush=True)
                last_stat = time.time()
            if status_proxy is not None and time.time() - last_wd >= _WATCHDOG_POLL_S:
                last_wd = time.time()
                stale = watchdog_stale_seconds()
                wd_failures = wd_failures + 1 if stale is None else 0
                limit = args.watchdog_minutes * 60
                if (stale is not None and stale > limit) or wd_failures >= _WATCHDOG_MAX_FAILURES:
                    why = (f"the trainer has received none of this worker's games for "
                           f"{stale / 60:.1f} min" if stale is not None else
                           f"the trainer has not answered {wd_failures} checks in a row")
                    n = int(os.environ.get(_WATCHDOG_ENV, '0')) + 1
                    print(f"[Remote] WATCHDOG: {why}. Restarting this worker "
                          f"(restart #{n}): stopping actors and reconnecting.", flush=True)
                    shutdown()
                    os.environ[_WATCHDOG_ENV] = str(n)
                    sys.stdout.flush()
                    sys.stderr.flush()
                    os.execv(sys.executable, [sys.executable] + sys.argv)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[Remote] Shutting down...")
        shutdown()
        print("[Remote] All actors stopped.")


if __name__ == '__main__':
    # 'spawn' is safer than 'fork' for processes that load TF/CUDA.
    mp.set_start_method('spawn', force=True)
    main()
