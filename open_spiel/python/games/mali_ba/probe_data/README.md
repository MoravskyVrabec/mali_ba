# Probe data

Frozen inputs for `policy_probe.py`. A checkpoint's score is only comparable to
another score made against **the same files**.

## Current set (use these)

| file | contents |
|---|---|
| `probe_positions_v2_600.json` | 600 positions (moves 40-400, median 200) from 61 games, stored as action histories |
| `probe_reference_v2_600.json` | network-free reference search: 590 usable, 600 sims, heuristic prior + 2 heuristic rollouts, depth 300 (80.1% of rollouts reach a terminal state) |

Built 2026-09-30, after the meeple layout became a recorded chance outcome
(commit 342e132). Each game's layout is the first action in its history, so
replaying a position reproduces the exact board in any process. **Scores are
exactly repeatable**: the same checkpoint scored three times, with 16 and 5
workers, gave identical top-1 (31.53%) and CE (1.8322).

The reference contains no network, so it serves any observation layout: load
the game with the ini flags that match the checkpoint being scored (192 planes
for the C-series, 96 for the B-series).

**Untrained baseline, 192-plane (C-series) architecture, 5 random inits:**
top-1 17.6 / 31.5 / 32.2 / 27.5 / 15.3% -> **24.8% +/- 7.8**. A random net's
arbitrary move preferences can line up with the reference by chance, so a
trained checkpoint should clearly beat ~32% (the best untrained init) before it
counts as learning. Use top-1, not policy CE: CE rewards flat distributions.

The reference is heuristic-guided search, so it cannot credit a network for
learning things the heuristic does not know. Head-to-head play (`ab_eval.py`)
is the check for that.

## Stale sets (kept for the record; do not use for new comparisons)

`probe_positions.json` / `probe_reference.json` (200) and
`probe_positions_600.json` / `probe_reference_600.json` (600) were built before
342e132. The meeple layout was then drawn from a clock-seeded generator and not
recorded, so every process replayed these histories onto a different board: the
reference searches and every checkpoint's scoring ran on different boards. The
roughly 3.5-point run-to-run variation seen with them, first attributed to
floating-point effects, came from this. Their histories now replay onto layout
0, which is reproducible but is not the board the reference was built on.
Results recorded against them (B002/B005, a 96-plane floor of 28.4% +/- 1.1)
are noisier than they appeared.
