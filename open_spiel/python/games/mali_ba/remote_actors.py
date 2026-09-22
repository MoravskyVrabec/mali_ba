"""
Remote actor worker for distributed Mali-Ba training.

Run this on a second machine to contribute actor processes to an existing
train_mali_ba.py session that was started with --distributed.

Both machines must have the same codebase and compiled pyspiel/mali_ba.
The remote actors use CPU for MCTS (same as local actors); if the remote
machine has a GPU it will be used for the neural network inference inside
the evaluator unless you pass --cpu_only.

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


def _should_relay(line):
    for frag in _LOG_FILTER_SUBSTRINGS:
        if frag in line:
            return False
    return True


def _actor_worker(actor_id, initial_game_params, actor_args,
                  job_queue, result_queue, games_per_actor, log_queue=None):
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
                      job_queue, result_queue, games_per_actor)
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
    parser.add_argument('--cpu_only', action='store_true',
                        help='Force CPU-only TF even if a GPU is present')
    args = parser.parse_args()

    if args.cpu_only:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    # --- Connect to the queue server ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from queue_server import connect_client

    print(f"[Remote] Connecting to queue server at {args.server_host}:{args.server_port}...")
    try:
        manager = connect_client(
            host=args.server_host,
            port=args.server_port,
            authkey=args.authkey.encode()
        )
    except ConnectionRefusedError:
        print(f"[Remote] ERROR: Could not connect. Is train_mali_ba.py running with --distributed?")
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
        debug=config.get('debug', False),
    )
    initial_game_params = config['initial_game_params']

    # --- Actor pool management ---
    actor_pool = {}
    next_actor_id = args.actor_id_start

    def spawn():
        nonlocal next_actor_id
        p = mp.Process(
            target=_actor_worker,
            args=(next_actor_id, initial_game_params, actor_args,
                  job_queue, result_queue, actor_args.games_per_actor, log_queue),
            daemon=False
        )
        p.start()
        actor_pool[p] = next_actor_id
        print(f"[Remote] Spawned actor {next_actor_id}", flush=True)
        next_actor_id += 1

    # Initial spawn
    print(f"[Remote] Starting {args.num_actors} actor(s)...")
    for _ in range(args.num_actors):
        spawn()

    # Keep the pool full: respawn actors that finish their quota
    try:
        while True:
            dead = [p for p in list(actor_pool) if not p.is_alive()]
            for p in dead:
                aid = actor_pool.pop(p)
                p.join()
                print(f"[Remote] Actor {aid} finished, respawning...", flush=True)
                spawn()
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n[Remote] Shutting down...")
        for p in list(actor_pool):
            p.terminate()
        for p in list(actor_pool):
            p.join(timeout=5)
        print("[Remote] All actors stopped.")


if __name__ == '__main__':
    # 'spawn' is safer than 'fork' for processes that load TF/CUDA.
    mp.set_start_method('spawn', force=True)
    main()
