"""Measure whether a Mali-Ba checkpoint is learning, without playing any games.

Why this exists
---------------
Neither of the two numbers visible during a training run can tell you whether the
agent is improving:

  * Self-play win rate is a treadmill. Both sides improve together, so it mostly
    reflects how often a game reaches a win condition before the move cap. It sat
    near 34-37% across runs 113, A002, A003, A004 and A005 regardless.
  * Training loss is measured against a moving target drawn from a buffer whose
    contents change. Measured 2026-09-27, it tracks self-play THROUGHPUT rather than
    model quality: A005's loss rose from 2.339 to 2.387 purely because a finishing
    evaluation freed 24 cores, tripling the novel experiences per gradient step.

Head-to-head play (see ab_eval.py) does measure strength, but needs ~700 games to
resolve a moderate gain, and the checkpoints must be separated by enough training for
a gap to exist -- v113 vs vA004 (~2-3k games apart) came back at 0.336 against a
0.333 null.

This probe measures the thing training is actually optimising, on a FROZEN set of
positions that never changes between checkpoints. AlphaZero learns by distilling
search into the network: MCTS produces a better move distribution than the raw policy
head, and the policy head is trained to match it. So the gap between them is the
learning signal. As the network improves, it agrees with search more often and its
cross-entropy against the search distribution falls.

Because the position set is fixed, this is a genuine held-out validation metric -- the
one thing the training log does not provide. It needs no games and runs in minutes.

What it reports
---------------
  top-1 agreement   raw policy's best legal move == MCTS's most-visited move.
                    Blunt but easy to interpret. Higher is better.
  top-3 agreement   raw policy's best move is among MCTS's three most-visited.
  policy CE         cross-entropy of the raw policy against the MCTS visit
                    distribution, over legal actions. This IS the policy training
                    loss, but on fixed positions. LOWER is better, and it is far more
                    sensitive than top-1 because it uses the whole distribution.
  value MAE         |value head for the player to move - MCTS root value estimate|.
                    Lower is better.

All four are reported overall and split by game phase, since early and late positions
are very different problems.

Two-step use
------------
    # 1. Build the frozen position set ONCE. Uses the heuristic player only, so the
    #    set is not biased toward any checkpoint, and a fixed seed makes it
    #    reproducible. Commit or keep this file -- every later comparison needs it.
    python policy_probe.py --generate 200 --out probe_positions.json

    # 2. Score any number of checkpoints against that same set.
    python policy_probe.py --positions probe_positions.json \
        --agents mali_ba_agent_v113.weights.h5 \
                 mali_ba_agent_vA004.weights.h5 \
                 mali_ba_agent_vA005.weights.h5

Reading the result: a checkpoint that is learning shows rising top-1 agreement and
falling policy CE relative to older checkpoints. If those are flat across checkpoints
separated by thousands of games, the network is not distilling search any better than
it used to, and that is real evidence of stalled learning -- unlike a flat win rate.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time

# Absolute, and next to this script: pyspiel SEGFAULTS on a missing config_file
# rather than raising, so a relative default that misses is very hard to diagnose.

# --------------------------------------------------------------------------- #
#  pyspiel resolution
# --------------------------------------------------------------------------- #
# There are TWO pyspiel builds on this machine. The one in the conda env's
# site-packages is a stale copy that SEGFAULTS (not raises) inside load_game while
# parsing the current mali_ba.ini, because it predates ini keys such as
# timeout_leader_reward. Only the live build tree works. The training run gets this
# right via PYTHONPATH; a script run by hand usually does not, so fix it here before
# pyspiel is imported anywhere. Workers use the "spawn" start method and re-import
# this module, so module-level placement covers them too.
BUILD_PYSPIEL_DIRS = [
    "/media/robp/UD/Projects/open_spiel/build/python",
]


def _setup_paths():
    """Make both imports work in this process and in every spawned worker.

    Two separate things are needed:
      * the live pyspiel build, ahead of the stale site-packages copy;
      * the directory ABOVE this script, so `import mali_ba.training_utils`
        resolves -- this script lives inside the mali_ba package, so its own
        directory is not enough.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    parent = os.path.dirname(here)          # .../python/games, which holds mali_ba/
    for d in (here, parent):
        if d not in sys.path:
            sys.path.insert(0, d)

    cands = [d for d in os.environ.get("PYTHONPATH", "").split(os.pathsep) if d]
    cands += BUILD_PYSPIEL_DIRS
    for d in cands:
        if os.path.exists(os.path.join(d, "pyspiel.so")):
            if sys.path[0] != d:
                sys.path.insert(0, d)
            return d
    return None


_PYSPIEL_DIR = _setup_paths()

DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "mali_ba.ini")


def _check_config(path):
    """load_game segfaults on a missing config file, so check before calling it."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"config_file not found: {path}\n"
            f"pyspiel.load_game segfaults rather than raising on a missing config, "
            f"so this is checked up front. Pass --config_file with the ini the "
            f"training run uses.")


# --------------------------------------------------------------------------- #
#  Position set generation
# --------------------------------------------------------------------------- #

def generate_positions(n, config_file, seed, min_move, stride, max_move):
    """Play heuristic games and snapshot positions at a spread of move numbers.

    The heuristic player is used deliberately: it needs no weights, so the position
    set is independent of every checkpoint being compared. Positions are stored as
    action histories and replayed, which avoids depending on state serialisation.
    """
    import pyspiel

    _check_config(config_file)
    game = pyspiel.load_game("mali_ba", {"config_file": config_file})
    rng = random.Random(seed)
    out = []
    games_played = 0

    while len(out) < n:
        state = game.new_initial_state()
        while state.is_chance_node():
            # Sample the meeple layout. It is recorded in the history as a chance
            # outcome, so replay() reproduces the same board in any process.
            la = state.legal_actions()
            state.apply_action(la[rng.randrange(len(la))])
        ms = pyspiel.mali_ba.downcast_state(state)
        while ms.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
            la = state.legal_actions()
            if not la:
                break
            state.apply_action(la[rng.randrange(len(la))])
            ms = pyspiel.mali_ba.downcast_state(state)

        # Snapshot at min_move, min_move+stride, ... so the set spans the whole game.
        want = set(range(min_move, max_move + 1, stride))
        moves = 0
        games_played += 1
        while not state.is_terminal() and moves <= max_move and len(out) < n:
            if moves in want and state.current_player() >= 0:
                out.append({"history": [int(a) for a in state.history()],
                            "move": moves,
                            "player": int(state.current_player())})
            la = state.legal_actions()
            if not la:
                break
            p = state.current_player()
            if p < 0:
                state.apply_action(la[rng.randrange(len(la))])
                continue
            ms = pyspiel.mali_ba.downcast_state(state)
            try:
                a = ms.select_heuristic_random_action()
            except Exception:
                a = la[rng.randrange(len(la))]
            if a not in la:
                a = la[rng.randrange(len(la))]
            state.apply_action(a)
            moves += 1

    return out[:n], games_played


def replay(game, history):
    """Rebuild a state by replaying its full action history, chance nodes included."""
    state = game.new_initial_state()
    for a in history:
        state.apply_action(a)
    return state


# --------------------------------------------------------------------------- #
#  Scoring one checkpoint on the frozen set
# --------------------------------------------------------------------------- #

def _score_worker(agent_spec, positions, cfg, out_q):
    """Score one agent over a slice of positions. Runs in its own process."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    _setup_paths()

    import numpy as np
    import tensorflow as tf
    import pyspiel
    from open_spiel.python.algorithms import mcts
    from mali_ba.training_utils import (AlphaZeroEvaluator,
                                        create_mali_ba_policy_network,
                                        create_mali_ba_value_network)

    tf.config.set_visible_devices([], "GPU")
    _check_config(cfg["config_file"])
    game = pyspiel.load_game("mali_ba", {"config_file": cfg["config_file"]})
    shape = game.observation_tensor_shape()

    pol = agent_spec.replace("weights.h5", "_policy.weights.h5")
    val = agent_spec.replace("weights.h5", "_value.weights.h5")
    pm = create_mali_ba_policy_network(shape, game.num_distinct_actions())
    vm = create_mali_ba_value_network(shape, game.num_players())
    pm.load_weights(pol)
    vm.load_weights(val)

    for item in positions:
        idx = item["_idx"]
        try:
            state = replay(game, item["history"])
            if state.is_terminal() or state.current_player() < 0:
                out_q.put(("skip", idx))
                continue
            legal = state.legal_actions()
            if len(legal) < 2:
                out_q.put(("skip", idx))
                continue
            player = state.current_player()

            # --- raw network outputs, read BEFORE any search touches this state ---
            # A fresh evaluator per position keeps the cache from carrying anything
            # over, and entry[1] (the full-width raw policy) is discarded by prior(),
            # so it has to be read first.
            ev_raw = AlphaZeroEvaluator(game, pm, vm, heuristic_guidance_weight=0.0,
                                        cache_size=4)
            entry = ev_raw._entry(state)
            value_vec = np.asarray(entry[0], dtype=np.float64)
            raw_full = np.asarray(entry[1], dtype=np.float64)
            raw_legal = np.array([max(raw_full[a], 0.0) for a in legal])
            if raw_legal.sum() <= 0:
                raw_legal = np.ones(len(legal))
            raw_legal = raw_legal / raw_legal.sum()
            raw_best = legal[int(np.argmax(raw_legal))]

            # --- MCTS, with the heuristic mix the run actually deploys, because the
            # visit counts it produces are what the policy head is trained on ---
            ev_s = AlphaZeroEvaluator(game, pm, vm,
                                      heuristic_guidance_weight=cfg["heur_w"],
                                      cache_size=cfg["cache_size"])
            bot = mcts.MCTSBot(game=game, uct_c=cfg["uct_c"],
                               max_simulations=cfg["sims"], evaluator=ev_s,
                               solve=False, dirichlet_noise=None,
                               child_selection_fn=mcts.SearchNode.puct_value,
                               random_state=np.random.RandomState(1000 + idx),
                               verbose=False)
            root = bot.mcts_search(state)
            visits = {c.action: c.explore_count for c in root.children}
            tot = sum(visits.values())
            if tot <= 0:
                out_q.put(("skip", idx))
                continue
            ranked = sorted(visits.items(), key=lambda kv: -kv[1])
            mcts_best = ranked[0][0]
            mcts_top3 = {a for a, _ in ranked[:3]}

            # cross-entropy of raw policy against the MCTS visit distribution
            ce = 0.0
            for i, a in enumerate(legal):
                pi = visits.get(a, 0) / tot
                if pi > 0:
                    ce -= pi * math.log(max(raw_legal[i], 1e-12))

            root_v = root.total_reward / root.explore_count if root.explore_count else 0.0
            v_head = float(value_vec[player]) if value_vec.size > player else 0.0

            # Sharpness. CE alone cannot separate "the policy got better" from "the
            # policy got flatter": a flat distribution is penalised less by
            # cross-entropy AND, because the raw policy feeds the search prior, it
            # flattens the very target CE is measured against. Entropy of both sides
            # disambiguates. Reported in nats; ln(len(legal)) is the uniform ceiling.
            h_raw = -sum(q * math.log(max(q, 1e-12)) for q in raw_legal)
            pv = [visits.get(a, 0) / tot for a in legal]
            h_mcts = -sum(q * math.log(max(q, 1e-12)) for q in pv if q > 0)
            p_max = float(max(raw_legal))

            out_q.put(("ok", idx, item["move"],
                       int(raw_best == mcts_best),
                       int(raw_best in mcts_top3),
                       ce, abs(v_head - root_v), len(legal),
                       h_raw, h_mcts, p_max, math.log(len(legal))))
        except Exception as e:
            out_q.put(("err", idx, str(e)[:140]))
    out_q.put(("done",))


def score_agent(agent_spec, positions, cfg, workers):
    """Fan the position set across processes and collect per-position results."""
    chunks = [[] for _ in range(workers)]
    for i, p in enumerate(positions):
        item = dict(p)
        item["_idx"] = i
        chunks[i % workers].append(item)

    q = mp.Queue()
    procs = []
    for ch in chunks:
        if not ch:
            continue
        pr = mp.Process(target=_score_worker, args=(agent_spec, ch, cfg, q),
                        daemon=True)
        pr.start()
        procs.append(pr)

    rows, errs, skipped, done = [], [], 0, 0
    per_pos = {}
    while done < len(procs):
        msg = q.get()
        if msg[0] == "done":
            done += 1
        elif msg[0] == "ok":
            # report() expects (move, top1, top3, ce, vmae, nlegal, ...) at 0..n,
            # but keep idx->top1 separately so agents can be compared PAIRED: both
            # are scored on identical positions, which is far more powerful than
            # treating them as two independent samples.
            per_pos[msg[1]] = msg[3]
            rows.append(msg[2:])
        elif msg[0] == "skip":
            skipped += 1
        else:
            errs.append(msg[2])
    for pr in procs:
        pr.join(timeout=10)
    return rows, errs, skipped, per_pos



def _snapshot_weights(specs, dest):
    """Copy each checkpoint's weights aside before scoring.

    The live training run rewrites its --save_model_path every --save_every
    episodes. Scoring that file directly means measuring a moving target (observed
    2026-09-27: vA005 read 35.5% then 43.5% on identical positions because the file
    changed between runs) and risks reading a partially written file. Copying first
    pins exactly one version, and the mtime is reported so it is on the record.
    """
    import shutil
    os.makedirs(dest, exist_ok=True)
    out = []
    print("  checkpoint files (copied aside so a live run cannot move them):")
    for spec in specs:
        base = os.path.basename(spec)
        newspec = os.path.join(dest, base)
        for suf in ("_policy.weights.h5", "_value.weights.h5"):
            src = spec.replace("weights.h5", suf)
            dst = newspec.replace("weights.h5", suf)
            shutil.copy2(src, dst)
            if suf == "_policy.weights.h5":
                st = os.stat(src)
                age = time.time() - st.st_mtime
                warn = "   <-- WRITTEN IN THE LAST 10 MIN, LIKELY LIVE" if age < 600 else ""
                print(f"    {base:34} mtime {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))}"
                      f"  {st.st_size:,} bytes{warn}")
        out.append(newspec)
    print()
    return out


def _mcnemar(a_correct, b_correct):
    """Paired two-sided exact test on which positions each agent got right.

    b = baseline right / other wrong, c = baseline wrong / other right. Only the
    disagreements carry information, which is exactly why this beats comparing two
    independent proportions.
    """
    keys = set(a_correct) & set(b_correct)
    b = sum(1 for k in keys if a_correct[k] and not b_correct[k])
    c = sum(1 for k in keys if not a_correct[k] and b_correct[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    # exact binomial, two-sided, p=0.5
    lo = min(b, c)
    tail = sum(math.comb(n, i) for i in range(lo + 1)) / (2 ** n)
    return b, c, min(1.0, 2 * tail)


# =========================================================================== #
#  FIXED-REFERENCE MODE  (added 2026-09-27 after the random-weight control)
# =========================================================================== #
# Why this exists
# ---------------
# The original mode scored a network against MCTS visits produced by a search
# that used THAT network as 65% of its prior. A confident network steers the
# search toward its own preference, so agreement is partly self-fulfilling. The
# control proved it: a randomly initialised net scored 40% top-1 against a
# chance rate of 11%, and it did so because it was the SHARPEST net measured
# (entropy 1.064, max p 0.556). Sharpness inflates top-1 through the prior loop
# and inflates CE through confident errors, so neither number is comparable
# across nets of differing sharpness.
#
# Here the reference distribution is computed ONCE by a search that contains no
# neural network at all -- heuristic prior, heuristic rollouts -- and cached.
# Every candidate is then scored against that identical target. The circularity
# is gone, and because no MCTS runs per candidate, scoring is one forward pass
# per position: seconds per checkpoint instead of minutes.
#
# The reference is not "ground truth"; it is a fixed, network-independent
# yardstick. That is all a comparison across checkpoints needs.


class HeuristicRolloutEvaluator:
    """MCTS evaluator with no neural network: heuristic prior, heuristic rollouts.

    Deliberately network-free so the reference target can never depend on the
    checkpoint being measured.
    """

    def __init__(self, game, n_rollouts=2, max_depth=160, seed=0):
        import numpy as np
        self._game = game
        self._n = n_rollouts
        self._max_depth = max_depth
        self._np = np
        import random as _r
        self._rng = _r.Random(seed)
        self.terminal_hits = 0
        self.rollouts_done = 0

    def prior(self, state):
        import pyspiel
        legal = state.legal_actions()
        if not legal:
            return []
        try:
            w = pyspiel.mali_ba.downcast_state(state).get_heuristic_action_weights()
            tot = sum(w.values())
        except Exception:
            w, tot = {}, 0
        if tot > 0:
            out = [(a, max(w.get(a, 0.0), 0.0) / tot) for a in legal]
            s = sum(p for _, p in out)
            if s > 0:
                return [(a, p / s) for a, p in out]
        u = 1.0 / len(legal)
        return [(a, u) for a in legal]

    def evaluate(self, state):
        import pyspiel
        np = self._np
        if state.is_terminal():
            return np.array(state.returns(), dtype=np.float64)
        total = np.zeros(self._game.num_players(), dtype=np.float64)
        for _ in range(self._n):
            w = state.clone()
            depth = 0
            while not w.is_terminal() and depth < self._max_depth:
                la = w.legal_actions()
                if not la:
                    break
                if w.current_player() < 0:
                    w.apply_action(la[self._rng.randrange(len(la))])
                    continue
                try:
                    a = pyspiel.mali_ba.downcast_state(w).select_heuristic_random_action()
                except Exception:
                    a = la[self._rng.randrange(len(la))]
                if a not in la:
                    a = la[self._rng.randrange(len(la))]
                w.apply_action(a)
                depth += 1
            self.rollouts_done += 1
            if w.is_terminal():
                self.terminal_hits += 1
                total += np.array(w.returns(), dtype=np.float64)
            # non-terminal rollout contributes 0 (neutral) -- identical for every
            # candidate, so it biases the yardstick, not the comparison
        return total / max(1, self._n)


def _ref_worker(positions, cfg, out_q):
    """Build reference visit distributions for a slice of positions."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    _setup_paths()
    import numpy as np
    import pyspiel
    from open_spiel.python.algorithms import mcts

    _check_config(cfg["config_file"])
    game = pyspiel.load_game("mali_ba", cfg["game_params"])
    for item in positions:
        idx = item["_idx"]
        try:
            state = replay(game, item["history"])
            if state.is_terminal() or state.current_player() < 0:
                out_q.put(("skip", idx)); continue
            legal = state.legal_actions()
            if len(legal) < 2:
                out_q.put(("skip", idx)); continue
            ev = HeuristicRolloutEvaluator(game, n_rollouts=cfg["ref_rollouts"],
                                          max_depth=cfg["ref_depth"], seed=5000 + idx)
            bot = mcts.MCTSBot(game=game, uct_c=cfg["uct_c"],
                               max_simulations=cfg["ref_sims"], evaluator=ev,
                               solve=False, dirichlet_noise=None,
                               child_selection_fn=mcts.SearchNode.puct_value,
                               random_state=np.random.RandomState(5000 + idx),
                               verbose=False)
            root = bot.mcts_search(state)
            visits = {int(c.action): int(c.explore_count) for c in root.children
                      if c.explore_count > 0}
            if not visits:
                out_q.put(("skip", idx)); continue
            frac = ev.terminal_hits / max(1, ev.rollouts_done)
            out_q.put(("ok", idx, item["move"], visits,
                       [int(a) for a in legal], frac))
        except Exception as e:
            out_q.put(("err", idx, str(e)[:140]))
    out_q.put(("done",))


def build_reference(positions, cfg, workers, out_path):
    """Run the network-free search once per position and cache the result."""
    chunks = [[] for _ in range(workers)]
    for i, pos in enumerate(positions):
        it = dict(pos); it["_idx"] = i
        chunks[i % workers].append(it)
    q = mp.Queue(); procs = []
    for ch in chunks:
        if not ch:
            continue
        pr = mp.Process(target=_ref_worker, args=(ch, cfg, q), daemon=True)
        pr.start(); procs.append(pr)
    ref, errs, skipped, done = {}, [], 0, 0
    fracs = []
    t0 = time.time()
    while done < len(procs):
        m = q.get()
        if m[0] == "done":
            done += 1
        elif m[0] == "ok":
            ref[str(m[1])] = {"move": m[2], "visits": {str(k): v for k, v in m[3].items()},
                              "legal": m[4]}
            fracs.append(m[5])
        elif m[0] == "skip":
            skipped += 1
        else:
            errs.append(m[2])
    for pr in procs:
        pr.join(timeout=10)
    meta = {"ref_sims": cfg["ref_sims"], "ref_rollouts": cfg["ref_rollouts"],
            "ref_depth": cfg["ref_depth"], "uct_c": cfg["uct_c"],
            "config_file": cfg["config_file"], "n": len(ref),
            "rollout_terminal_fraction": (sum(fracs)/len(fracs)) if fracs else 0.0}
    with open(out_path, "w") as f:
        json.dump({"meta": meta, "ref": ref}, f)
    print(f"  built reference for {len(ref)} positions in {time.time()-t0:.0f}s "
          f"({skipped} skipped, {len(errs)} errors)")
    print(f"  rollouts reaching a terminal state: "
          f"{100*meta['rollout_terminal_fraction']:.1f}%  "
          f"(the rest score 0, identically for every candidate)")
    if errs:
        print(f"  first error: {errs[0]}")
    print(f"  wrote {out_path}")
    return meta


def _cand_worker(agent_spec, positions, ref, cfg, out_q):
    """Score one candidate's RAW policy against the cached reference.

    No MCTS here -- one forward pass per position, so this is fast.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    _setup_paths()
    import numpy as np
    import tensorflow as tf
    import pyspiel
    from mali_ba.training_utils import (create_mali_ba_policy_network,
                                        create_mali_ba_value_network)
    tf.config.set_visible_devices([], "GPU")
    _check_config(cfg["config_file"])
    game = pyspiel.load_game("mali_ba", cfg["game_params"])
    shape = game.observation_tensor_shape()
    pm = create_mali_ba_policy_network(shape, game.num_distinct_actions())
    vm = create_mali_ba_value_network(shape, game.num_players())
    pm.load_weights(agent_spec.replace("weights.h5", "_policy.weights.h5"))
    vm.load_weights(agent_spec.replace("weights.h5", "_value.weights.h5"))

    for item in positions:
        idx = item["_idx"]
        r = ref.get(str(idx))
        if r is None:
            out_q.put(("skip", idx)); continue
        try:
            state = replay(game, item["history"])
            legal = [int(a) for a in state.legal_actions()]
            if set(legal) != set(r["legal"]):
                # position replayed to a different node than the reference saw
                out_q.put(("skip", idx)); continue
            cp = state.current_player()
            obs = np.asarray(state.observation_tensor(cp), dtype=np.float32)
            pol, val = pm(obs.reshape((1, *shape)), training=False),                        vm(obs.reshape((1, *shape)), training=False)
            pol = np.asarray(pol)[0].astype(np.float64)
            val = np.asarray(val)[0].astype(np.float64)

            raw = np.array([max(pol[a], 0.0) for a in legal])
            if raw.sum() <= 0:
                raw = np.ones(len(legal))
            raw = raw / raw.sum()
            raw_best = legal[int(np.argmax(raw))]

            visits = {int(k): v for k, v in r["visits"].items()}
            tot = sum(visits.values())
            ranked = sorted(visits.items(), key=lambda kv: -kv[1])
            ref_best = ranked[0][0]
            ref_top3 = {a for a, _ in ranked[:3]}

            ce = 0.0
            for i, a in enumerate(legal):
                pi = visits.get(a, 0) / tot
                if pi > 0:
                    ce -= pi * math.log(max(raw[i], 1e-12))
            h_raw = -sum(q * math.log(max(q, 1e-12)) for q in raw)
            out_q.put(("ok", idx, r["move"], int(raw_best == ref_best),
                       int(raw_best in ref_top3), ce, 0.0, len(legal),
                       h_raw, 0.0, float(max(raw)), math.log(len(legal))))
        except Exception as e:
            out_q.put(("err", idx, str(e)[:140]))
    out_q.put(("done",))


def score_against_reference(agent_spec, positions, ref, cfg, workers):
    chunks = [[] for _ in range(workers)]
    for i, pos in enumerate(positions):
        it = dict(pos); it["_idx"] = i
        chunks[i % workers].append(it)
    q = mp.Queue(); procs = []
    for ch in chunks:
        if not ch:
            continue
        pr = mp.Process(target=_cand_worker, args=(agent_spec, ch, ref, cfg, q),
                        daemon=True)
        pr.start(); procs.append(pr)
    rows, errs, skipped, done, per_pos = [], [], 0, 0, {}
    while done < len(procs):
        m = q.get()
        if m[0] == "done":
            done += 1
        elif m[0] == "ok":
            per_pos[m[1]] = m[3]
            rows.append(m[2:])
        elif m[0] == "skip":
            skipped += 1
        else:
            errs.append(m[2])
    for pr in procs:
        pr.join(timeout=10)
    return rows, errs, skipped, per_pos


def _phase(move):
    if move < 120:
        return "early (<120)"
    if move < 280:
        return "mid (120-279)"
    return "late (280+)"


def report(name, rows, errs, skipped, elapsed):
    import statistics as st
    if not rows:
        print(f"  {name}: no scoreable positions "
              f"({skipped} skipped, {len(errs)} errors)")
        if errs:
            print(f"    first error: {errs[0]}")
        return None

    top1 = [r[1] for r in rows]
    top3 = [r[2] for r in rows]
    ce = [r[3] for r in rows]
    vmae = [r[4] for r in rows]
    nlegal = [r[5] for r in rows]
    h_raw = [r[6] for r in rows]
    h_mcts = [r[7] for r in rows]
    p_max = [r[8] for r in rows]
    h_unif = [r[9] for r in rows]

    print(f"  {name}")
    print(f"    positions scored : {len(rows)}"
          + (f"   ({skipped} skipped, {len(errs)} errors)" if skipped or errs else ""))
    print(f"    top-1 agreement  : {100*sum(top1)/len(top1):6.2f}%   (higher = better)")
    print(f"    top-3 agreement  : {100*sum(top3)/len(top3):6.2f}%")
    print(f"    policy CE        : {sum(ce)/len(ce):6.4f}   (LOWER = better; "
          f"uniform would be {st.mean([math.log(n) for n in nlegal]):.4f})")
    if any(v != 0.0 for v in vmae):
        print(f"    value MAE        : {sum(vmae)/len(vmae):6.4f}   (lower = better)")
    print(f"    mean legal moves : {st.mean(nlegal):6.1f}")
    hr, hm, hu = st.mean(h_raw), st.mean(h_mcts), st.mean(h_unif)
    print(f"    raw policy entropy: {hr:6.4f} nats  ({100*hr/hu:5.1f}% of uniform "
          f"{hu:.4f})   <- higher = flatter/less decisive")
    if hm != 0.0:
        print(f"    MCTS visit entropy: {hm:6.4f} nats  ({100*hm/hu:5.1f}% of uniform)")
    print(f"    raw policy max p  : {st.mean(p_max):6.4f}   <- lower = less confident")
    by = {}
    for r in rows:
        by.setdefault(_phase(r[0]), []).append(r)
    for ph in ("early (<120)", "mid (120-279)", "late (280+)"):
        if ph not in by:
            continue
        g = by[ph]
        _vm = sum(x[4] for x in g)/len(g)
        print(f"      {ph:14} n={len(g):4d}  top1 {100*sum(x[1] for x in g)/len(g):5.1f}%"
              f"   CE {sum(x[3] for x in g)/len(g):6.4f}"
              + (f"   vMAE {_vm:6.4f}" if _vm != 0.0 else ""))
    print(f"    wall time        : {elapsed:.0f}s")
    if errs:
        print(f"    first error      : {errs[0]}")
    print()
    return {"top1": sum(top1)/len(top1), "top3": sum(top3)/len(top3),
            "ce": sum(ce)/len(ce), "vmae": sum(vmae)/len(vmae), "n": len(rows),
            "h_raw": st.mean(h_raw), "h_mcts": st.mean(h_mcts),
            "p_max": st.mean(p_max), "h_unif": st.mean(h_unif)}


def main():
    ap = argparse.ArgumentParser(
        description="Measure policy/search agreement on a frozen position set")
    ap.add_argument("--generate", type=int, default=0,
                    help="build a position set of this many positions and exit")
    ap.add_argument("--out", default="probe_positions.json",
                    help="where --generate writes the set")
    ap.add_argument("--positions", default="probe_positions.json",
                    help="frozen position set to score against")
    ap.add_argument("--agents", nargs="+",
                    help="one or more weights stems (…weights.h5), oldest first")
    ap.add_argument("--sims", type=int, default=200,
                    help="MCTS simulations per position (default 200). More sims make "
                         "the reference distribution sharper and the metric cleaner.")
    ap.add_argument("--uct_c", type=float, default=1.4)
    ap.add_argument("--heur_w", type=float, default=0.35,
                    help="heuristic mix for the SEARCH, matching the training run "
                         "(default 0.35). The raw policy is always read unmixed.")
    ap.add_argument("--cache_size", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=8,
                    help="processes (default 8, deliberately low so a training run "
                         "keeps most of its cores)")
    ap.add_argument("--config_file", default=DEFAULT_CONFIG)
    ap.add_argument("--seed", type=int, default=20260927,
                    help="only affects --generate")
    ap.add_argument("--build_reference", metavar="OUT",
                    help="compute the network-free reference target for the position "
                         "set and cache it to OUT, then exit")
    ap.add_argument("--reference", metavar="FILE",
                    help="score candidates against this cached reference instead of "
                         "against each candidate's own search (recommended: removes "
                         "the prior-feedback circularity)")
    ap.add_argument("--ref_sims", type=int, default=800,
                    help="sims for the reference search (default 800; it runs once "
                         "for the whole set, so it can afford to be deep)")
    ap.add_argument("--ref_rollouts", type=int, default=2,
                    help="heuristic rollouts per leaf in the reference search")
    ap.add_argument("--ref_depth", type=int, default=160,
                    help="max rollout depth before scoring 0")
    ap.add_argument("--no_snapshot", action="store_true",
                    help="score the weights files in place instead of copying them "
                         "aside first (not recommended while a run is training)")
    ap.add_argument("--snapshot_dir", default="/tmp",
                    help="where to copy weights before scoring")
    ap.add_argument("--min_move", type=int, default=40)
    ap.add_argument("--stride", type=int, default=40)
    ap.add_argument("--max_move", type=int, default=400)
    args = ap.parse_args()

    import pyspiel as _ps
    if "site-packages" in _ps.__file__:
        print(f"WARNING: pyspiel resolved to {_ps.__file__}\n"
              f"  That build is stale and segfaults parsing the current ini. "
              f"Set PYTHONPATH to the build tree, e.g.\n"
              f"  PYTHONPATH=/media/robp/UD/Projects/open_spiel/build/python",
              file=sys.stderr)

    if args.generate:
        t0 = time.time()
        print(f"Generating {args.generate} positions with the heuristic player "
              f"(seed {args.seed})…")
        pos, ngames = generate_positions(args.generate, args.config_file, args.seed,
                                         args.min_move, args.stride, args.max_move)
        with open(args.out, "w") as f:
            json.dump({"config_file": args.config_file, "seed": args.seed,
                       "min_move": args.min_move, "stride": args.stride,
                       "max_move": args.max_move, "positions": pos}, f)
        moves = [p["move"] for p in pos]
        print(f"Wrote {len(pos)} positions to {args.out} from {ngames} games "
              f"in {time.time()-t0:.0f}s")
        print(f"  move numbers: min {min(moves)}  median {sorted(moves)[len(moves)//2]}"
              f"  max {max(moves)}")
        print(f"\nKeep this file. Every comparison must use the same one, or the "
              f"numbers are not comparable.")
        return

    if args.build_reference:
        with open(args.positions) as f:
            _d = json.load(f)
        gp = {"config_file": args.config_file, "player_types": "ai,ai,ai"}
        cfg = {"config_file": args.config_file, "game_params": gp,
               "uct_c": args.uct_c, "ref_sims": args.ref_sims,
               "ref_rollouts": args.ref_rollouts, "ref_depth": args.ref_depth}
        print("=" * 70)
        print(f"  BUILDING NETWORK-FREE REFERENCE for {len(_d['positions'])} positions")
        print(f"  {args.ref_sims} sims, heuristic prior + {args.ref_rollouts} heuristic "
              f"rollouts (depth {args.ref_depth}), {args.workers} workers")
        print("=" * 70)
        build_reference(_d["positions"], cfg, args.workers, args.build_reference)
        return

    if not args.agents:
        ap.error("--agents is required unless --generate is used")

    with open(args.positions) as f:
        data = json.load(f)
    positions = data["positions"]
    if data.get("config_file") != args.config_file:
        print(f"NOTE: set was generated with config_file={data.get('config_file')!r}, "
              f"scoring with {args.config_file!r}")

    for spec in args.agents:
        for p in (spec.replace("weights.h5", "_policy.weights.h5"),
                  spec.replace("weights.h5", "_value.weights.h5")):
            if not os.path.exists(p):
                raise FileNotFoundError(f"missing weights file: {p}")

    cfg = {"config_file": args.config_file, "sims": args.sims, "uct_c": args.uct_c,
           "heur_w": args.heur_w, "cache_size": args.cache_size}
    paired = {}

    print("=" * 70)
    print(f"  POLICY / SEARCH AGREEMENT on {len(positions)} frozen positions")
    print(f"  {args.sims} sims per position, heuristic mix {args.heur_w} in search, "
          f"{args.workers} workers")
    print("=" * 70)

    results = {}
    agent_specs = args.agents if args.no_snapshot else _snapshot_weights(
        args.agents, os.path.join(args.snapshot_dir, "probe_weights"))
    _ref = None
    if args.reference:
        with open(args.reference) as f:
            _rj = json.load(f)
        _ref = _rj["ref"]; _meta = _rj["meta"]
        cfg["game_params"] = {"config_file": args.config_file,
                             "player_types": "ai,ai,ai"}
        print(f"  reference: {args.reference}")
        print(f"    {_meta['n']} positions, {_meta['ref_sims']} sims, network-free "
              f"(heuristic prior + rollouts)")
        print(f"    rollouts reaching terminal: "
              f"{100*_meta.get('rollout_terminal_fraction',0):.1f}%")
        print(f"    -> identical target for every candidate, no prior feedback loop")
        print()

    for spec in agent_specs:
        name = os.path.basename(spec).replace(".weights.h5", "")
        t0 = time.time()
        if _ref is not None:
            rows, errs, skipped, per_pos = score_against_reference(
                spec, positions, _ref, cfg, args.workers)
        else:
            rows, errs, skipped, per_pos = score_agent(spec, positions, cfg,
                                                      args.workers)
        paired[name] = per_pos
        results[name] = report(name, rows, errs, skipped, time.time() - t0)

    ok = {k: v for k, v in results.items() if v}
    if len(ok) > 1:
        print("=" * 70)
        print("  CHANGE relative to the first checkpoint listed")
        print("=" * 70)
        names = list(ok)
        base = ok[names[0]]
        print(f"  {'checkpoint':28} {'top-1':>9} {'top-3':>9} {'policy CE':>11} "
              f"{'value MAE':>11} {'pol entropy':>12} {'max p':>8}")
        for n in names:
            r = ok[n]
            if n == names[0]:
                print(f"  {n:28} {100*r['top1']:8.2f}% {100*r['top3']:8.2f}% "
                      f"{r['ce']:11.4f} {r['vmae']:11.4f} {r['h_raw']:12.4f} "
                      f"{r['p_max']:8.4f}   (baseline)")
            else:
                print(f"  {n:28} {100*r['top1']:8.2f}% {100*r['top3']:8.2f}% "
                      f"{r['ce']:11.4f} {r['vmae']:11.4f} {r['h_raw']:12.4f} "
                      f"{r['p_max']:8.4f}   "
                      f"[top1 {100*(r['top1']-base['top1']):+.2f}pp, "
                      f"entropy {r['h_raw']-base['h_raw']:+.4f}]")
        print()
        print("  PAIRED top-1 comparison vs the baseline (McNemar, exact two-sided).")
        print("  Both agents see identical positions, so only the positions where they")
        print("  DISAGREE carry information -- much more powerful than comparing the two")
        print("  percentages as if they were independent samples.")
        bkey = names[0]
        for n_ in names[1:]:
            b, c, pv = _mcnemar(paired[bkey], paired[n_])
            verdict = ("baseline better" if b > c else "other better") if pv < 0.05 \
                else "no significant difference"
            print(f"    {bkey} vs {n_}:")
            print(f"      baseline-only right {b}, {n_}-only right {c}, "
                  f"p = {pv:.4f}  ->  {verdict}")
        print()
        print("  Run-to-run noise on a FROZEN checkpoint was ~2-2.5pp on top-1 "
              "(measured 2026-09-27).")
        print("  A checkpoint flagged as live above is NOT comparable -- copy it aside "
              "or stop the run.")
        print(f"  Uniform entropy on this set is {base['h_unif']:.4f} nats; a raw policy")
        print("  entropy near that means the network is barely distinguishing moves.")
        print("  CAUTION: policy CE can FALL while the policy gets worse, because a")
        print("  flatter distribution is penalised less and also flattens the search")
        print("  prior it is scored against. Read entropy and top-1 together, not CE alone.")
        print("  Learning looks like: top-1 UP, policy CE DOWN, across checkpoints")
        print("  separated by thousands of self-play games.")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
