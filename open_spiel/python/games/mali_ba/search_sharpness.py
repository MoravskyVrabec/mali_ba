"""
search_sharpness.py

How decisive is MCTS? Runs the trainer's search (MCTSBot + AlphaZeroEvaluator, PUCT,
no Dirichlet noise) on the frozen v2 probe positions with one checkpoint, at several
uct_c values, and reports how flat the visit counts are compared with the prior and
how far apart the children's Q values are. The "pruned" columns show the recorded
target after policy-target pruning (training_utils.prune_policy_target).

Why (2026-10-02): every checkpoint's raw policy sat at ~92% of uniform entropy, and
the MCTS visit targets in the D005 buffer were 90-91% of uniform (96% median), so the
policy head had almost nothing to learn. With uct_c = 2.0 the children's Q values
differed by only ~0.04 (median), far less than the exploration term, so visits just
followed the (flat) prior. Measured with the D005 18:56 checkpoint, 120 positions,
300 sims, heuristic 0.30:

    uct_c  visit entropy  prior entropy  top move share  child Q spread  same best move as c=2
     2.00            89%            93%             41%           0.077                   100%
     1.00            82%            93%             47%           0.087                    83%
     0.50            70%            93%             59%           0.111                    73%
     0.25            59%            93%             67%           0.125                    69%

Sharper is not automatically better (the Q values come from an imperfect value head);
head-to-head play decides that. This only shows how much the search discriminates.

Usage (from this directory, with the trainer's environment):
    python search_sharpness.py --agent mali_ba_agent_vD005.weights.h5
    python search_sharpness.py --agent ... --uct_c 2.0 1.0 0.5 --sims 300 --every 5 --workers 4
"""
import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def ent_ratio(p):
    """Entropy as a share of the uniform entropy over the non-zero entries."""
    p = np.asarray(p, float)
    p = p[p > 0]
    if len(p) < 2:
        return np.nan
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def worker(items, cfg, out_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    for d in (HERE, os.path.dirname(HERE)):
        if d not in sys.path:
            sys.path.insert(0, d)
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2)
    import pyspiel
    from open_spiel.python.algorithms import mcts
    from mali_ba.training_utils import (AlphaZeroEvaluator, create_mali_ba_policy_network,
                                        prune_policy_target,
                                        create_mali_ba_value_network)
    game = pyspiel.load_game("mali_ba", {"config_file": cfg['config_file']})
    shape = game.observation_tensor_shape()
    pm = create_mali_ba_policy_network(shape, game.num_distinct_actions())
    vm = create_mali_ba_value_network(shape, game.num_players())
    pm.load_weights(cfg['agent'].replace("weights.h5", "_policy.weights.h5"))
    vm.load_weights(cfg['agent'].replace("weights.h5", "_value.weights.h5"))
    for idx, hist in items:
        # Positions are action histories recorded under the old setup rules; under
        # other rules an action becomes illegal. Report it rather than apply it
        # (applying it crashed workers and hung the run).
        state = game.new_initial_state()
        mismatch = False
        for a in hist:
            if a not in state.legal_actions():
                mismatch = True
                break
            state.apply_action(a)
        if mismatch:
            out_q.put({'mismatch': idx})
            continue
        if state.is_terminal() or state.current_player() < 0 or len(state.legal_actions()) < 2:
            continue
        row = {'idx': idx, 'n_legal': len(state.legal_actions())}
        for c in cfg['uct_c']:
            ev = AlphaZeroEvaluator(game, pm, vm, heuristic_guidance_weight=cfg['heur_w'],
                                    cache_size=20000, blend_normalized=cfg['blend_normalized'],
                                    mover_relative=cfg['mover_relative'])
            bot = mcts.MCTSBot(game=game, uct_c=c, max_simulations=cfg['sims'], evaluator=ev,
                               solve=False, dirichlet_noise=cfg['dirichlet'],
                               child_selection_fn=mcts.SearchNode.puct_value,
                               random_state=np.random.RandomState(1000 + idx), verbose=False)
            if cfg['mover_relative']:
                ev.set_relabel_seat(state.current_player())   # one view per search
            root = bot.mcts_search(state)
            visits = [ch.explore_count for ch in root.children]
            priors = [ch.prior for ch in root.children]
            qs = [ch.total_reward / ch.explore_count for ch in root.children if ch.explore_count > 0]
            pruned = list(prune_policy_target(root, c).values())
            clean = dict(ev.prior(state))
            pruned_c = list(prune_policy_target(root, c, priors=clean).values())
            row[c] = dict(visit_ent=ent_ratio(visits), prior_ent=ent_ratio(priors),
                          pruned_ent=ent_ratio(pruned),
                          pruned_top=max(pruned) / max(1, sum(pruned)),
                          pruned_kept=sum(pruned) / max(1, sum(visits)),
                          clean_ent=ent_ratio(pruned_c),
                          clean_top=max(pruned_c) / max(1, sum(pruned_c)),
                          clean_kept=sum(pruned_c) / max(1, sum(visits)),
                          top_share=max(visits) / max(1, sum(visits)),
                          q_spread=(max(qs) - min(qs)) if len(qs) > 1 else np.nan,
                          best=root.best_child().action)
        out_q.put(row)
    out_q.put(None)


def main():
    ap = argparse.ArgumentParser(description="Measure how decisive MCTS is at several uct_c values.")
    ap.add_argument('--agent', required=True, help="Checkpoint, e.g. mali_ba_agent_vD005.weights.h5")
    ap.add_argument('--positions', default=os.path.join(HERE, 'probe_data/probe_positions_v2_600.json'))
    # The v2 positions were recorded before the 2026-10-06 setup rules; they only
    # replay under the old rules (see probe_data/README.md).
    ap.add_argument('--config_file', default=os.path.join(HERE, 'probe_data/mali_ba_oldrules.ini'))
    ap.add_argument('--uct_c', type=float, nargs='+', default=[2.0, 1.0, 0.5, 0.25])
    ap.add_argument('--sims', type=int, default=300)
    ap.add_argument('--heur_w', type=float, default=0.30, help="Heuristic guidance weight in the prior")
    ap.add_argument('--guidance_blend_normalized', action='store_true',
                    help="Rescale the network policy over the legal moves before mixing in the "
                         "heuristic, so --heur_w is the heuristic's true share (training's "
                         "guidance_blend_normalized). Needed for D021 and later, whose policies "
                         "put most of their probability on illegal moves.")
    ap.add_argument('--mover_relative', action='store_true',
                    help="The agent was trained with mover_relative_obs: evaluate every position "
                         "from its player to move's point of view (required for such networks).")
    ap.add_argument('--dirichlet', type=float, nargs=2, default=None, metavar=('EPS', 'ALPHA'),
                    help="Root Dirichlet noise as in self-play full searches (training uses "
                         "0.2 0.25). Off by default.")
    ap.add_argument('--every', type=int, default=5, help="Use every Nth probe position")
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    cfg = dict(agent=os.path.abspath(args.agent), config_file=args.config_file,
               uct_c=args.uct_c, sims=args.sims, heur_w=args.heur_w,
               blend_normalized=args.guidance_blend_normalized,
               dirichlet=tuple(args.dirichlet) if args.dirichlet else None,
               mover_relative=args.mover_relative)

    pos = json.load(open(args.positions))['positions']
    items = [(i, p['history']) for i, p in enumerate(pos) if i % args.every == 0]
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(items[w::args.workers], cfg, q))
             for w in range(args.workers)]
    for p in procs:
        p.start()
    import queue as _queue
    rows, done, mismatches = [], 0, 0
    while done < args.workers:
        try:
            r = q.get(timeout=10)
        except _queue.Empty:
            if not any(p.is_alive() for p in procs):
                print("ERROR: worker processes exited without finishing (crashed?)")
                sys.exit(1)
            continue
        if r is None:
            done += 1
        elif 'mismatch' in r:
            mismatches += 1
        else:
            rows.append(r)
    for p in procs:
        p.join()
    if mismatches > 0.05 * len(items):
        print(f"ERROR: {mismatches} of {len(items)} positions do not replay under "
              f"{args.config_file}: they were recorded under different game rules. "
              f"Use the default --config_file (probe_data/mali_ba_oldrules.ini).")
        sys.exit(2)

    ref = args.uct_c[0]
    print(f"{os.path.basename(args.agent)}: {len(rows)} positions, mean legal moves "
          f"{np.mean([r['n_legal'] for r in rows]):.1f}, {args.sims} sims, heuristic {args.heur_w}"
          f"{' (normalized blend)' if args.guidance_blend_normalized else ''}"
          f"{f', Dirichlet noise {args.dirichlet}' if args.dirichlet else ''}")
    print(f"{'uct_c':>6} {'visit entropy':>14} {'prior entropy':>14} {'top move share':>15} "
          f"{'child Q spread':>15} {'same best move as c=' + str(ref):>24} "
          f"{'pruned entropy':>15} {'pruned top':>11} {'visits kept':>12} "
          f"{'clean-prior pruned':>19} {'top':>5} {'kept':>5}")
    for c in args.uct_c:
        g = lambda k: np.nanmean([r[c][k] for r in rows])
        same = np.mean([r[c]['best'] == r[ref]['best'] for r in rows])
        print(f"{c:6.2f} {100 * g('visit_ent'):13.0f}% {100 * g('prior_ent'):13.0f}% "
              f"{100 * g('top_share'):14.0f}% {g('q_spread'):15.3f} {100 * same:23.0f}% "
              f"{100 * g('pruned_ent'):14.0f}% {100 * g('pruned_top'):10.0f}% "
              f"{100 * g('pruned_kept'):11.0f}% "
              f"{100 * g('clean_ent'):18.0f}% {100 * g('clean_top'):4.0f}% {100 * g('clean_kept'):4.0f}%")
    qs = np.array([r[ref]['q_spread'] for r in rows])
    print(f"child Q spread at uct_c={ref} (percentiles 10/50/90):",
          np.round(np.nanpercentile(qs, [10, 50, 90]), 3))


if __name__ == '__main__':
    main()
