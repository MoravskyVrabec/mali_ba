"""Head-to-head strength evaluation for Mali-Ba agents.

Why this exists
---------------
Self-play win rate cannot measure whether an agent is improving. Both sides improve
together, so the figure mostly reflects how often games reach a win condition before
the move cap -- it sat near 34% across runs 113, A002, A003 and A004 while the agent
was, as far as anything showed, learning normally.

Strength has to be measured against a FIXED reference. This plays one agent against
another and reports a record.

Three-player note
-----------------
Mali-Ba has three seats, so "A vs B" means A takes some seats and B the rest. With A in
one seat and B in two, two equally strong agents each win a seat's share, so A's
expected win rate is 1/3. Above that means A is stronger. Seats are rotated so each
agent plays every position an equal number of times -- the game has a known turn-order
advantage, and without rotation that alone would decide the result.

Agents are identified by the --save_model_path style stem, e.g.
mali_ba_agent_vA004.weights.h5, from which the _policy and _value files are derived.
Pass "heuristic" instead of a path to use the built-in heuristic player as a baseline
that needs no weights.

Examples
--------
    # current weights against the six-month-old v113, 60 games, A in one seat
    python ab_eval.py --agent_a mali_ba_agent_vA004.weights.h5 \
                      --agent_b mali_ba_agent_v113.weights.h5 --games 60

    # against the heuristic player, deeper search
    python ab_eval.py --agent_a mali_ba_agent_vA004.weights.h5 \
                      --agent_b heuristic --games 30 --sims 300

    # give A two seats instead of one (expected win rate 2/3 if equal)
    python ab_eval.py --agent_a A.weights.h5 --agent_b B.weights.h5 --a_seats 2
"""

import argparse
import itertools
import math
import multiprocessing as mp
import os
import random
import sys
import time

HEURISTIC = "heuristic"


def _wilson(k, n, z=1.96):
    """Wilson score interval: behaves sensibly at small n and near 0 or 1,
    unlike the normal approximation."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def _build_agent(spec, game, shape, sims, uct_c, cache_size, heur_w):
    """Returns a callable state -> action, or None for the heuristic player."""
    import numpy as np
    import pyspiel
    from open_spiel.python.algorithms import mcts
    from mali_ba.training_utils import (AlphaZeroEvaluator,
                                        create_mali_ba_policy_network,
                                        create_mali_ba_value_network)

    if spec == HEURISTIC:
        def act(state):
            return pyspiel.mali_ba.downcast_state(state).select_heuristic_random_action()
        return act, "heuristic"

    pol = spec.replace("weights.h5", "_policy.weights.h5")
    val = spec.replace("weights.h5", "_value.weights.h5")
    for p in (pol, val):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing weights file: {p}")
    pm = create_mali_ba_policy_network(shape, game.num_distinct_actions())
    vm = create_mali_ba_value_network(shape, game.num_players())
    pm.load_weights(pol)
    vm.load_weights(val)
    # heur_w=0 measures the NETWORKS alone; the training value (~0.35) measures the
    # agent as actually deployed. Both are defensible, but they must match across the
    # two agents, and at 0 the agents are weak enough that most games hit the move cap
    # and the comparison yields no decided games.
    ev = AlphaZeroEvaluator(game, pm, vm, heuristic_guidance_weight=heur_w,
                            cache_size=cache_size)
    bot = mcts.MCTSBot(game=game, uct_c=uct_c, max_simulations=sims, evaluator=ev,
                       solve=False, dirichlet_noise=None,
                       child_selection_fn=mcts.SearchNode.puct_value, verbose=False)

    def act(state):
        root = bot.mcts_search(state)
        # Greedy: evaluation should play each agent's best move, not sample.
        best = max(root.children, key=lambda c: c.explore_count)
        return best.action

    return act, os.path.basename(spec).replace(".weights.h5", "")


def play_game(game, actors, seed):
    """actors: list of callables indexed by seat. Returns (winner_seat|None, moves, reason)."""
    import numpy as np
    import pyspiel
    rng = random.Random(seed)
    state = game.new_initial_state()
    while state.is_chance_node():
        # Meeple layout: sampled from this game's seed, so evaluations vary the
        # board but stay reproducible.
        la = state.legal_actions()
        state.apply_action(la[rng.randrange(len(la))])
    ms = pyspiel.mali_ba.downcast_state(state)
    # Token placement: uniform random, as in training. Same distribution for everyone.
    while ms.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
        la = state.legal_actions()
        if not la:
            break
        state.apply_action(la[rng.randrange(len(la))])
        ms = pyspiel.mali_ba.downcast_state(state)
    moves = 0
    while not state.is_terminal():
        la = state.legal_actions()
        if not la:
            break
        p = state.current_player()
        if p < 0:
            state.apply_action(la[rng.randrange(len(la))])
            continue
        try:
            a = actors[p](state)
        except Exception:
            a = la[rng.randrange(len(la))]
        if a not in la:
            a = la[rng.randrange(len(la))]
        state.apply_action(a)
        moves += 1
    ms = pyspiel.mali_ba.downcast_state(state)
    reason = ms.get_game_end_reason()
    rets = list(state.returns())
    won = "Max game length" not in reason
    winner = rets.index(max(rets)) if won else None
    return winner, moves, reason


def _worker(jobs, cfg, out_q):
    """Play a list of (rotation, seed) jobs in one process.

    Agents are built once per worker, not once per game: loading two models costs
    seconds, and a worker plays many games. TF is pinned to one thread because we run
    many workers -- the same reason the training actors do it.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    if not cfg["gpu"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import tensorflow as tf
    import pyspiel
    if not cfg["gpu"]:
        tf.config.set_visible_devices([], "GPU")
    game = pyspiel.load_game("mali_ba", {"config_file": cfg["config_file"]})
    shape = game.observation_tensor_shape()
    act_a, _ = _build_agent(cfg["agent_a"], game, shape, cfg["sims"], cfg["uct_c"],
                            cfg["cache_size"], cfg["heur_w"])
    act_b, _ = _build_agent(cfg["agent_b"], game, shape, cfg["sims"], cfg["uct_c"],
                            cfg["cache_size"], cfg["heur_w"])
    nseats = game.num_players()
    for rot, seed in jobs:
        actors = [act_a if s in rot else act_b for s in range(nseats)]
        try:
            winner, moves, reason = play_game(game, actors, seed)
        except Exception as e:
            out_q.put(("error", rot, str(e)[:120]))
            continue
        out_q.put(("ok", rot, winner, moves, reason))
    out_q.put(("done",))


def main():
    ap = argparse.ArgumentParser(description="Head-to-head evaluation of two Mali-Ba agents")
    ap.add_argument("--agent_a", required=True,
                    help='weights stem (…weights.h5) or "heuristic"')
    ap.add_argument("--agent_b", required=True,
                    help='weights stem (…weights.h5) or "heuristic"')
    ap.add_argument("--games", type=int, default=60,
                    help="total games; rounded up to a multiple of the seat rotation")
    ap.add_argument("--sims", type=int, default=100,
                    help="MCTS simulations per move, same for both agents (default 100). "
                         "Lower than training on purpose: more games matters more than "
                         "deeper search for a strength estimate.")
    ap.add_argument("--uct_c", type=float, default=1.4)
    ap.add_argument("--a_seats", type=int, default=1, choices=(1, 2),
                    help="how many of the three seats agent A takes (default 1, so an "
                         "equal-strength A wins 1/3)")
    ap.add_argument("--config_file", default="mali_ba.ini")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cache_size", type=int, default=8192)
    ap.add_argument("--heuristic_weight", type=float, default=0.35,
                    help="heuristic prior blended into the network policy, applied "
                         "identically to both agents (default 0.35, matching training). "
                         "Use 0.0 to measure the networks in isolation, but expect far "
                         "more games to hit the move cap and decide nothing.")
    ap.add_argument("--gpu", action="store_true",
                    help="allow the GPU (default CPU-only, to leave a training run alone)")
    ap.add_argument("--gpu_memory_mb", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4),
                    help="parallel game-playing processes. Each pins TF to ONE thread, so "
                         "you want roughly one worker per core: a single-threaded forward "
                         "pass costs ~37ms against ~10ms multi-threaded, which means few "
                         "workers is SLOWER than running serially with all threads "
                         "(measured: 6 workers gave only 1.16x over serial). Defaults to "
                         "cores-4. Each worker holds its own copy of both models, roughly "
                         "0.7 GB. Lower it if a training run is using the machine.")
    args = ap.parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    import tensorflow as tf
    import pyspiel
    if args.gpu:
        for g in tf.config.experimental.list_physical_devices("GPU"):
            try:
                tf.config.set_logical_device_configuration(
                    g, [tf.config.LogicalDeviceConfiguration(memory_limit=args.gpu_memory_mb)])
            except RuntimeError:
                pass
    else:
        tf.config.set_visible_devices([], "GPU")

    game = pyspiel.load_game("mali_ba", {"config_file": args.config_file})
    shape = game.observation_tensor_shape()
    nseats = game.num_players()
    if nseats != 3:
        print(f"note: this game has {nseats} seats; seat rotation adapts automatically")

    # The parent only resolves display names; the workers load the models.
    def _name(spec):
        return (HEURISTIC if spec == HEURISTIC
                else os.path.basename(spec).replace(".weights.h5", ""))
    for spec in (args.agent_a, args.agent_b):
        if spec != HEURISTIC:
            for suf in ("_policy", "_value"):
                f = spec.replace("weights.h5", suf + ".weights.h5")
                if not os.path.exists(f):
                    print(f"ERROR: missing weights file: {f}")
                    sys.exit(1)
    name_a, name_b = _name(args.agent_a), _name(args.agent_b)

    # Every distinct assignment of A to `a_seats` of the seats, so each agent plays
    # every position equally often.
    rotations = list(itertools.combinations(range(nseats), args.a_seats))
    per_rot = max(1, math.ceil(args.games / len(rotations)))
    total = per_rot * len(rotations)
    expected = args.a_seats / nseats

    print(f"A = {name_a}")
    print(f"B = {name_b}")
    print(f"{args.sims} sims/move, uct_c={args.uct_c}, "
          f"heuristic_weight={args.heuristic_weight}, "
          f"A in {args.a_seats} of {nseats} seats")
    print(f"{total} games ({per_rot} per seat rotation), expected A win rate if equal: "
          f"{expected:.3f}")
    print()

    a_wins = 0
    b_wins = 0
    draws = 0
    errors = 0
    by_seat = {s: [0, 0] for s in range(nseats)}
    a_by_rot = {r: [0, 0] for r in rotations}
    lengths = []

    # Build the full job list, then deal it round-robin so every worker gets a mix of
    # rotations -- otherwise a worker that finishes early would bias which seats are
    # represented if the run is cut short.
    jobs = []
    k = 0
    for rot in rotations:
        for _ in range(per_rot):
            jobs.append((rot, args.seed + k * 7919))
            k += 1
    nw = max(1, min(args.workers, len(jobs)))
    buckets = [jobs[i::nw] for i in range(nw)]
    cfg = dict(agent_a=args.agent_a, agent_b=args.agent_b, sims=args.sims,
               uct_c=args.uct_c, cache_size=args.cache_size,
               heur_w=args.heuristic_weight, config_file=args.config_file,
               gpu=args.gpu)
    print(f"running {nw} worker process(es)")
    print()

    t0 = time.time()
    out_q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(b, cfg, out_q), daemon=True)
             for b in buckets if b]
    for pr in procs:
        pr.start()
    n = 0
    finished = 0
    while finished < len(procs):
        msg = out_q.get()
        if msg[0] == "done":
            finished += 1
            continue
        if msg[0] == "error":
            errors += 1
            print(f"  worker error on A@{','.join(map(str,msg[1]))}: {msg[2]}", flush=True)
            continue
        _, rot, winner, moves, reason = msg
        n += 1
        lengths.append(moves)
        a_by_rot[rot][1] += 1
        if winner is None:
            draws += 1
        else:
            by_seat[winner][0] += 1
            if winner in rot:
                a_wins += 1
                a_by_rot[rot][0] += 1
            else:
                b_wins += 1
        for s in range(nseats):
            by_seat[s][1] += 1
        el = time.time() - t0
        dec = a_wins + b_wins
        print(f"  {n:>4}/{total}  A@{','.join(map(str, rot))}  {moves:>3} moves  "
              f"winner={'draw' if winner is None else f'seat {winner}'}  "
              f"[A {a_wins}-{b_wins} of {dec} decided, {el/max(1,n):.0f}s/game wall]",
              flush=True)
    for pr in procs:
        pr.join(timeout=30)
        if pr.is_alive():
            pr.terminate()

    decided = a_wins + b_wins
    print()
    print("=" * 66)
    print(f"  games played      : {n}   (decided {decided}, timed out {draws})")
    if decided:
        p, lo, hi = _wilson(a_wins, decided)
        print(f"  A wins            : {a_wins}/{decided} = {p:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"  B wins            : {b_wins}/{decided} = {b_wins/decided:.3f}")
        exp_dec = args.a_seats / nseats
        verdict = ("A stronger" if lo > exp_dec else
                   "B stronger" if hi < exp_dec else
                   "no significant difference")
        print(f"  expected if equal : {exp_dec:.3f}   ->  {verdict}")
    else:
        print("  every game timed out; raise --sims or --games")
    print()
    print(f"  wins by seat (turn-order effect): "
          + "  ".join(f"seat {s}: {v[0]}" for s, v in sorted(by_seat.items())))
    print(f"  A win rate by rotation: "
          + "  ".join(f"A@{','.join(map(str,r))}: {v[0]}/{v[1]}" for r, v in a_by_rot.items()))
    if lengths:
        lengths.sort()
        print(f"  game length: median {lengths[len(lengths)//2]}  "
              f"min {lengths[0]}  max {lengths[-1]}")
    if errors:
        print(f"  worker errors: {errors}")
    print(f"  wall time: {(time.time()-t0)/60:.1f} min "
          f"({nw} workers, {(time.time()-t0)/max(1,n):.0f}s/game wall)")
    print("=" * 66)


if __name__ == "__main__":
    # spawn: workers import TF fresh, so the parent never initialises it
    mp.set_start_method("spawn", force=True)
    main()
