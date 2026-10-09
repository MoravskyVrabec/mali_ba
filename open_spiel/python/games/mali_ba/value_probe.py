"""Compare value heads across checkpoints on one fixed, held-out set of positions.

Why this exists (2026-10-08)
----------------------------
analyze_log's value checks score a run's value head on that run's own games. When a run
changes what is played or kept (culling off, pool shares), the test changes too, so
those numbers do not compare across runs: D011's looked better than D010's yet they
were level head to head, and D013's looked worse than D011's yet it won. This probe
scores every checkpoint on the SAME positions, so differences are the networks'.

The positions come from head-to-head games (`ab_eval.py --record_positions FILE`),
which are never trained on, so the set is held out for every checkpoint. They are
stored as observations (not move histories), so a later rules change cannot break the
set; it only needs the same observation layout (192 planes).

What it reports, per checkpoint
-------------------------------
  top-pick   how often the value head rates highest the player who finished with the
             best return (the winner, or the score leader at a timeout). Chance is 33%.
             Overall, by moves remaining, and by how the game ended.
  MSE        mean squared error against the final returns (time penalties still to
             come are not included, identically for every checkpoint).
  calib      average prediction minus average final return, over all players.
  vs first   top-pick difference from the first checkpoint listed, with a 95% interval
             from resampling whole GAMES (positions within a game are correlated, so
             treating them as independent would overstate the precision).

Usage (from this directory, with the trainer's environment):
    python value_probe.py --positions /media/robp/UD/Projects/open_spiel/value_probe_set.npz \\
        --agents mali_ba_agent_vD011.weights.h5 mali_ba_agent_vD013.weights.h5

A probe is no substitute for head-to-head play: it measures judgement on positions,
not play.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKETS = (("early (>200 left)", 201, 9999), ("mid (61-200)", 61, 200), ("late (<=60)", 0, 60))


def main():
    ap = argparse.ArgumentParser(description="Value-head probe on a fixed held-out set")
    ap.add_argument("--positions", required=True, help="npz written by ab_eval.py --record_positions")
    ap.add_argument("--agents", nargs="+", required=True, help="checkpoint stems (...weights.h5)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")   # small job; leave the GPU alone
    sys.path.insert(0, os.path.dirname(HERE))
    from mali_ba.training_utils import create_mali_ba_value_network

    d = np.load(args.positions, allow_pickle=False)
    meta = json.loads(str(d["meta"]))
    X = d["X"]
    if X.ndim == 2:   # flat observations: reshape to the recorded / standard layout
        X = X.reshape((-1, *meta.get("obs_shape", (192, 15, 15))))
    rets, game, to_end, ending = d["returns"], d["game"], d["moves_to_end"], d["ending"]
    leader = rets.argmax(1)
    shape = X.shape[1:]
    n_games = len(np.unique(game))
    print(f"{len(X):,} positions from {n_games} games "
          f"(recorded {meta.get('created')}, agents {', '.join(os.path.basename(a) for a in meta.get('agents', []))}; "
          f"endings: " + ", ".join(f"{k} {int((ending == k).sum())}" for k in ("timeout", "timbuktu", "rare_goods")) + ")")

    correct, rows = {}, []
    for spec in args.agents:
        name = os.path.basename(spec).replace(".weights.h5", "")
        path = spec.replace("weights.h5", "_value.weights.h5")
        if not os.path.exists(path):
            print(f"  missing {path}, skipped")
            continue
        vm = create_mali_ba_value_network(shape, rets.shape[1])
        vm.load_weights(path)
        pred = np.concatenate([vm(X[i:i + 512].astype(np.float32), training=False).numpy()
                               for i in range(0, len(X), 512)])
        ok = pred.argmax(1) == leader
        correct[name] = ok
        r = {"name": name, "top": 100 * ok.mean(),
             "mse": float(np.mean((pred - rets) ** 2)),
             "calib": float(pred.mean() - rets.mean())}
        for label, lo, hi in BUCKETS:
            m = (to_end >= lo) & (to_end <= hi)
            r[label] = 100 * ok[m].mean() if m.any() else float("nan")
        for k in ("timeout", "timbuktu", "rare_goods"):
            m = ending == k
            r[k] = 100 * ok[m].mean() if m.any() else float("nan")
        rows.append(r)

    if not rows:
        return
    hdr = (f"\n{'checkpoint':<28} {'top-pick':>8} " + " ".join(f"{b[0]:>18}" for b in BUCKETS)
           + f" {'timeout':>8} {'Timbuktu':>9} {'rare':>6} {'MSE':>7} {'calib':>7}")
    print(hdr)

    def pct(v, w):
        return f"{'-':>{w}} " if np.isnan(v) else f"{v:{w}.1f}%"
    for r in rows:
        print(f"{r['name']:<28} {r['top']:7.1f}% " + " ".join(pct(r[b[0]], 17) for b in BUCKETS)
              + f" {pct(r['timeout'], 7)} {pct(r['timbuktu'], 8)} {pct(r['rare_goods'], 5)}"
              f" {r['mse']:7.4f} {r['calib']:+7.3f}")

    if len(rows) > 1:
        base = rows[0]["name"]
        games = np.unique(game)
        idx_by_game = [np.where(game == g)[0] for g in games]
        rng = np.random.RandomState(0)
        print(f"\nTop-pick difference vs {base} (95% interval from resampling games):")
        for r in rows[1:]:
            diff = correct[r["name"]].astype(float) - correct[base].astype(float)
            per_game = np.array([diff[ix].sum() for ix in idx_by_game])
            per_n = np.array([len(ix) for ix in idx_by_game])
            boots = []
            for _ in range(args.bootstrap):
                s = rng.randint(0, len(games), len(games))
                boots.append(100 * per_game[s].sum() / per_n[s].sum())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            est = 100 * diff.mean()
            verdict = "better" if lo > 0 else "worse" if hi < 0 else "no clear difference"
            print(f"  {r['name']:<28} {est:+5.1f} points  [{lo:+.1f}, {hi:+.1f}]  {verdict}")
    print("\nThe winner here is the best final return (the timeout leader counts). "
          "A probe measures judgement on positions, not play: head-to-head decides strength.")


if __name__ == "__main__":
    main()
