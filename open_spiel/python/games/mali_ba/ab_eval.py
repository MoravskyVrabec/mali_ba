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

    # same network, different search: does a lower uct_c pick stronger moves?
    python ab_eval.py --agent_a D.weights.h5 --agent_b D.weights.h5 \
                      --uct_c_a 0.5 --uct_c_b 2.0 --heuristic_weight 0.30
"""

import argparse
import itertools
import math
import multiprocessing as mp
import os
import queue
import random
import sys
import time

HEURISTIC = "heuristic"


class _Progress:
    """One self-updating status line on the terminal, e.g.
        game 123/504 (24%) | 12m elapsed, ~38m left | A 41 B 50 of 91 decided

    Written to /dev/tty, so it shows even when stdout and stderr are redirected to a
    log (the usual way ab_eval is run) and never ends up in that log. Without a
    terminal (background or nohup runs) it does nothing.
    """

    def __init__(self, total):
        self.total = total
        self.t0 = time.time()
        self.n = 0
        self.score = "starting: loading networks and playing the first games..."
        try:
            self.tty = open('/dev/tty', 'w')
        except OSError:
            self.tty = None
        self.update(0, self.score)

    def tick(self):
        """Redraw with the latest counts, so the clock moves between results."""
        self.update(self.n, self.score)

    @staticmethod
    def _fmt(sec):
        sec = int(sec)
        return f"{sec // 3600}h{(sec % 3600) // 60:02d}m" if sec >= 3600 else f"{sec // 60}m{sec % 60:02d}s"

    def update(self, n, score):
        self.n, self.score = n, score
        if self.tty is None:
            return
        el = time.time() - self.t0
        # The first results arrive only after start-up, so early estimates are wild.
        left = (f"~{self._fmt(el / n * (self.total - n))} left" if n >= 10
                else "estimating time left...")
        line = (f"game {n}/{self.total} ({100 * n // max(1, self.total)}%) | "
                f"{self._fmt(el)} elapsed, {left} | {score}")
        try:
            self.tty.write("\r" + line + "\033[K")
            self.tty.flush()
        except OSError:
            self.tty = None

    def done(self):
        if self.tty is not None:
            try:
                self.tty.write("\n")
                self.tty.close()
            except OSError:
                pass


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


def _build_agent(spec, game, shape, sims, uct_c, cache_size, heur_w, client=None):
    """Returns a callable state -> action, or None for the heuristic player.

    With an InferenceClient, forward passes go to the batched GPU server and the
    evaluator never touches local models, so none are built here.
    """
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

    if client is not None:
        pm = vm = None
    else:
        pm, vm = _load_models(spec, game, shape)
    # heur_w=0 measures the NETWORKS alone; the training value (~0.35) measures the
    # agent as actually deployed. Both are defensible, but they must match across the
    # two agents, and at 0 the agents are weak enough that most games hit the move cap
    # and the comparison yields no decided games.
    ev = AlphaZeroEvaluator(game, pm, vm, heuristic_guidance_weight=heur_w,
                            cache_size=cache_size, client=client)
    bot = mcts.MCTSBot(game=game, uct_c=uct_c, max_simulations=sims, evaluator=ev,
                       solve=False, dirichlet_noise=None,
                       child_selection_fn=mcts.SearchNode.puct_value, verbose=False)

    def act(state):
        root = bot.mcts_search(state)
        # Greedy: evaluation should play each agent's best move, not sample.
        best = max(root.children, key=lambda c: c.explore_count)
        return best.action

    return act, os.path.basename(spec).replace(".weights.h5", "")


def _load_models(spec, game, shape):
    from mali_ba.training_utils import (create_mali_ba_policy_network,
                                        create_mali_ba_value_network)
    pol = spec.replace("weights.h5", "_policy.weights.h5")
    val = spec.replace("weights.h5", "_value.weights.h5")
    for p in (pol, val):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing weights file: {p}")
    pm = create_mali_ba_policy_network(shape, game.num_distinct_actions())
    vm = create_mali_ba_value_network(shape, game.num_players())
    pm.load_weights(pol)
    vm.load_weights(val)
    return pm, vm


def _start_server(spec, game, shape, n_slots, config_file):
    """Start a batched GPU inference server holding one agent's weights.

    Returns (arena, proc, weights_queue, stop_event), or None if the server finds no
    GPU -- it refuses to serve on CPU, which would be far slower than the workers
    each inferring locally.
    """
    from mali_ba.inference_server import InferenceArena, server_loop
    obs_size = 1
    for d in shape:
        obs_size *= d
    arena = InferenceArena(n_slots, obs_size, game.num_distinct_actions(),
                           game.num_players())
    wq = mp.Queue()
    stop = mp.Event()
    # max_batch = n_slots: every worker waiting on this agent fits in one batch.
    proc = mp.Process(target=server_loop,
                      args=(arena, shape, {"config_file": config_file}, wq, stop),
                      kwargs=dict(max_batch=n_slots, cpus='auto'), daemon=True)
    proc.start()
    if not arena.gpu_ok.wait(180):
        stop.set()
        proc.join(timeout=15)
        return None
    pm, vm = _load_models(spec, game, shape)
    wq.put((pm.get_weights(), vm.get_weights()))
    if not arena.ready.wait(600):
        raise RuntimeError("inference server did not become ready within 600s")
    return arena, proc, wq, stop


def _start_servers(args, side_specs, game, shape, nw, name_fn):
    """One inference server per distinct network. Returns (arenas by side, servers)."""
    arenas, servers = {}, []
    if not args.inference_server:
        return arenas, servers
    by_spec = {}
    for side, spec in side_specs:
        if spec == HEURISTIC:
            continue
        if spec not in by_spec:
            print(f"starting inference server for {name_fn(spec)} ...", flush=True)
            srv = _start_server(spec, game, shape, nw, args.config_file)
            if srv is None:
                print("  no usable GPU: the server refused to serve, so workers "
                      "will infer on the CPU")
                for _, (_, proc, _, stop) in servers:
                    stop.set()
                    proc.join(timeout=15)
                return {}, []
            by_spec[spec] = srv
            servers.append((name_fn(spec), srv))
        arenas[side] = by_spec[spec][0]
    return arenas, servers


def _stop_servers(servers):
    stats = []
    for name, (arena, proc, wq, stop) in servers:
        stats.append((name, arena.stats()))
        stop.set()
        from mali_ba.inference_server import STOP
        wq.put(STOP)
        proc.join(timeout=30)
        if proc.is_alive():
            proc.terminate()
    return stats


def _main_three_way(args, game, shape, nseats, name_fn, uct_a, uct_b, uct_c):
    """A, B and C each take one seat; all 6 seat orders are played equally often."""
    specs = {"a": args.agent_a, "b": args.agent_b, "c": args.agent_c}
    names = {s: name_fn(sp) for s, sp in specs.items()}
    ucts = {"a": uct_a, "b": uct_b, "c": uct_c}
    orders = list(itertools.permutations("abc"))          # seating per seat 0..2
    per_order = max(1, math.ceil(args.games / len(orders)))
    total = per_order * len(orders)
    for s in "abc":
        print(f"{s.upper()} = {names[s]}" + ("" if specs[s] == HEURISTIC
                                             else f"  (uct_c={ucts[s]})"))
    print(f"{args.sims} sims/move, heuristic_weight={args.heuristic_weight}, "
          f"three-way: one seat each")
    print(f"{total} games ({per_order} per seat order), expected win share if equal: 0.333")
    print()
    # Orders interleaved, so a partial run is balanced across seat orders.
    jobs = []
    for i in range(per_order):
        for j, order in enumerate(orders):
            jobs.append((order, order, args.seed + (j * per_order + i) * 7919))
    nw = max(1, min(args.workers, len(jobs)))
    buckets = [jobs[i::nw] for i in range(nw)]
    cfg = dict(agent_a=specs["a"], agent_b=specs["b"], agent_c=specs["c"], sims=args.sims,
               uct_c_a=uct_a, uct_c_b=uct_b, uct_c_c=uct_c, cache_size=args.cache_size,
               heur_w=args.heuristic_weight, config_file=args.config_file, gpu=args.gpu)
    progress = _Progress(total)   # shown from the start, before servers load
    arenas, servers = _start_servers(args, tuple(specs.items()), game, shape, nw, name_fn)
    print(f"running {nw} worker process(es), inference: "
          f"{'GPU server' if servers else 'CPU in each worker'}")
    print()

    wins = {s: 0 for s in "abc"}
    seat_wins = {s: [0] * nseats for s in "abc"}     # wins by the seat the agent sat in
    seat_games = {s: [0] * nseats for s in "abc"}
    by_seat = [0] * nseats
    draws = errors = n = 0
    lengths = []
    t0 = time.time()
    out_q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(b, cfg, out_q, i, arenas), daemon=True)
             for i, b in enumerate(buckets) if b]
    for pr in procs:
        pr.start()
    finished = 0
    while finished < len(procs):
        try:
            msg = out_q.get(timeout=5)
        except queue.Empty:
            progress.tick()
            continue
        if msg[0] == "done":
            finished += 1
            continue
        if msg[0] == "error":
            errors += 1
            print(f"  worker error on order {''.join(msg[1]).upper()}: {msg[2]}", flush=True)
            continue
        _, _, seating, winner, moves, reason = msg
        n += 1
        lengths.append(moves)
        for seat, side in enumerate(seating):
            seat_games[side][seat] += 1
        if winner is None:
            draws += 1
        else:
            side = seating[winner]
            wins[side] += 1
            seat_wins[side][winner] += 1
            by_seat[winner] += 1
        dec = sum(wins.values())
        print(f"  {n:>4}/{total}  order {''.join(seating).upper()}  {moves:>3} moves  "
              f"winner={'draw' if winner is None else seating[winner].upper()}  "
              f"[A {wins['a']} B {wins['b']} C {wins['c']} of {dec} decided, "
              f"{(time.time() - t0) / max(1, n):.0f}s/game wall]", flush=True)
        progress.update(n, f"A {wins['a']} B {wins['b']} C {wins['c']} of {dec} decided")
    progress.done()
    for pr in procs:
        pr.join(timeout=30)
        if pr.is_alive():
            pr.terminate()
    server_stats = _stop_servers(servers)

    decided = sum(wins.values())
    print()
    print("=" * 66)
    print(f"  games played      : {n}   (decided {decided}, timed out {draws})")
    if decided:
        for s in sorted("abc", key=lambda s: -wins[s]):
            p, lo, hi = _wilson(wins[s], decided)
            verdict = ("above equal share" if lo > 1 / 3 else
                       "below equal share" if hi < 1 / 3 else "within equal share")
            print(f"  {s.upper()} {names[s]:<34} {wins[s]:>4}/{decided} = {p:.3f}  "
                  f"95% CI [{lo:.3f}, {hi:.3f}]  {verdict}")
        print("  (equal strength = 0.333 each; the three shares sum to 1, so their CIs are")
        print("   not independent. For a firm A-vs-B verdict, follow up with a two-way run.)")
    else:
        print("  every game timed out; raise --sims or --games")
    print()
    print(f"  wins by seat (turn-order effect): "
          + "  ".join(f"seat {s}: {by_seat[s]}" for s in range(nseats)))
    for s in "abc":
        print(f"  {s.upper()} wins from seat 0/1/2: "
              + "  ".join(f"{seat_wins[s][k]}/{seat_games[s][k]}" for k in range(nseats)))
    if lengths:
        lengths.sort()
        print(f"  game length: median {lengths[len(lengths)//2]}  "
              f"min {lengths[0]}  max {lengths[-1]}")
    for name, st in server_stats:
        print(f"  inference server ({name}): {st['requests']:,} evals, "
              f"avg batch {st['avg_batch']:.1f}, {st['ms_per_request']:.3f} ms/eval")
    if errors:
        print(f"  worker errors: {errors}")
    print(f"  wall time: {(time.time()-t0)/60:.1f} min "
          f"({nw} workers, {(time.time()-t0)/max(1,n):.0f}s/game wall)")
    print("=" * 66)


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


def _worker(jobs, cfg, out_q, slot=None, arenas=None):
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
    # One client per arena: when both agents share a network they share a server,
    # and two clients on one slot would keep separate request sequence numbers.
    clients = {}
    arenas = arenas or {}
    sides = ("a", "b", "c") if cfg.get("agent_c") else ("a", "b")
    for side in sides:
        ar = arenas.get(side)
        if ar is not None and id(ar) not in clients:
            from mali_ba.inference_server import InferenceClient
            clients[id(ar)] = InferenceClient(ar, shape, slot=slot)
    def _client(side):
        ar = arenas.get(side)
        return clients[id(ar)] if ar is not None else None
    acts = {}
    for side in sides:
        acts[side], _ = _build_agent(cfg["agent_" + side], game, shape, cfg["sims"],
                                     cfg["uct_c_" + side], cfg["cache_size"], cfg["heur_w"],
                                     _client(side))
    # A job's seating names the agent in each seat, e.g. ('a', 'b', 'b') or ('c', 'a', 'b').
    for key, seating, seed in jobs:
        actors = [acts[side] for side in seating]
        try:
            winner, moves, reason = play_game(game, actors, seed)
        except Exception as e:
            out_q.put(("error", key, str(e)[:120]))
            continue
        out_q.put(("ok", key, seating, winner, moves, reason))
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
    ap.add_argument("--uct_c", type=float, default=2.0,
                    help="PUCT exploration constant for both agents (default 2.0, the "
                         "training default; results before 2026-10-05 used 1.4)")
    ap.add_argument("--uct_c_a", type=float, default=None,
                    help="uct_c for agent A only (overrides --uct_c)")
    ap.add_argument("--uct_c_b", type=float, default=None,
                    help="uct_c for agent B only (overrides --uct_c)")
    ap.add_argument("--agent_c", default=None,
                    help='three-way mode: a third agent (weights stem or "heuristic"). Each '
                         'game seats A, B and C once each, cycling through all 6 seat orders '
                         'so the first-mover advantage cancels; each agent wins 1/3 of '
                         'decided games if all are equal. --a_seats is ignored.')
    ap.add_argument("--uct_c_c", type=float, default=None,
                    help="uct_c for agent C only (overrides --uct_c)")
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
    ap.add_argument("--inference_server", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="serve every worker's forward passes from a batched GPU "
                         "inference server, one per distinct network (default on). "
                         "Single-threaded CPU inference costs ~37ms a call and is "
                         "nearly all of the search cost; training saw ~7x from the "
                         "server. Falls back to CPU inference if no GPU is found.")
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

    # Hiding the GPU through the environment would also hide it from the inference
    # servers, which inherit it. With a server, the parent hides it from its own TF
    # below instead, and workers hide it from themselves.
    if not args.gpu and not args.inference_server:
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
    three_way = args.agent_c is not None
    for spec in (args.agent_a, args.agent_b) + ((args.agent_c,) if three_way else ()):
        if spec != HEURISTIC:
            for suf in ("_policy", "_value"):
                f = spec.replace("weights.h5", suf + ".weights.h5")
                if not os.path.exists(f):
                    print(f"ERROR: missing weights file: {f}")
                    sys.exit(1)
    name_a, name_b = _name(args.agent_a), _name(args.agent_b)
    uct_a = args.uct_c if args.uct_c_a is None else args.uct_c_a
    uct_b = args.uct_c if args.uct_c_b is None else args.uct_c_b
    uct_c3 = args.uct_c if args.uct_c_c is None else args.uct_c_c
    if three_way:
        if nseats != 3:
            print("ERROR: three-way mode needs a 3-player game")
            sys.exit(1)
        return _main_three_way(args, game, shape, nseats, _name, uct_a, uct_b, uct_c3)

    # Every distinct assignment of A to `a_seats` of the seats, so each agent plays
    # every position equally often.
    rotations = list(itertools.combinations(range(nseats), args.a_seats))
    per_rot = max(1, math.ceil(args.games / len(rotations)))
    total = per_rot * len(rotations)
    expected = args.a_seats / nseats

    # uct_c means nothing to the heuristic player, so don't print one for it.
    print(f"A = {name_a}" + ("" if args.agent_a == HEURISTIC else f"  (uct_c={uct_a})"))
    print(f"B = {name_b}" + ("" if args.agent_b == HEURISTIC else f"  (uct_c={uct_b})"))
    print(f"{args.sims} sims/move, "
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

    # Build the job list with rotations INTERLEAVED (A@0, A@1, A@2, A@0, ...), then deal
    # it round-robin to workers, so the games finished at any point are balanced across
    # seats. Queuing rotation by rotation made every worker play its A@0 games first:
    # on 2026-10-06 a uct_c test read 0.51 at the halfway point (seat 0 is the strong
    # seat) and finished at 0.346. Each game keeps the seed it had before (indexed by
    # rotation, then game), so boards are unchanged from earlier runs.
    jobs = []
    for i in range(per_rot):
        for j, rot in enumerate(rotations):
            seating = tuple('a' if s in rot else 'b' for s in range(nseats))
            jobs.append((rot, seating, args.seed + (j * per_rot + i) * 7919))
    nw = max(1, min(args.workers, len(jobs)))
    buckets = [jobs[i::nw] for i in range(nw)]
    cfg = dict(agent_a=args.agent_a, agent_b=args.agent_b, agent_c=None, sims=args.sims,
               uct_c_a=uct_a, uct_c_b=uct_b, cache_size=args.cache_size,
               heur_w=args.heuristic_weight, config_file=args.config_file,
               gpu=args.gpu)
    progress = _Progress(total)   # shown from the start, before servers load
    arenas, servers = _start_servers(args, (("a", args.agent_a), ("b", args.agent_b)),
                                     game, shape, nw, _name)
    print(f"running {nw} worker process(es), inference: "
          f"{'GPU server' if servers else 'CPU in each worker'}")
    print()

    t0 = time.time()
    out_q = mp.Queue()
    procs = [mp.Process(target=_worker, args=(b, cfg, out_q, i, arenas), daemon=True)
             for i, b in enumerate(buckets) if b]
    for pr in procs:
        pr.start()
    n = 0
    finished = 0
    while finished < len(procs):
        try:
            msg = out_q.get(timeout=5)
        except queue.Empty:
            progress.tick()
            continue
        if msg[0] == "done":
            finished += 1
            continue
        if msg[0] == "error":
            errors += 1
            print(f"  worker error on A@{','.join(map(str,msg[1]))}: {msg[2]}", flush=True)
            continue
        _, rot, _seating, winner, moves, reason = msg
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
        progress.update(n, f"A {a_wins} B {b_wins} of {dec} decided")
    progress.done()
    for pr in procs:
        pr.join(timeout=30)
        if pr.is_alive():
            pr.terminate()
    server_stats = _stop_servers(servers)

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
    for name, st in server_stats:
        print(f"  inference server ({name}): {st['requests']:,} evals, "
              f"avg batch {st['avg_batch']:.1f}, {st['ms_per_request']:.3f} ms/eval")
    if errors:
        print(f"  worker errors: {errors}")
    print(f"  wall time: {(time.time()-t0)/60:.1f} min "
          f"({nw} workers, {(time.time()-t0)/max(1,n):.0f}s/game wall)")
    print("=" * 66)


if __name__ == "__main__":
    # spawn: workers import TF fresh, so the parent never initialises it
    mp.set_start_method("spawn", force=True)
    main()
