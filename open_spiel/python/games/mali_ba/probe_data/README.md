# Probe data

Frozen inputs for `policy_probe.py`. Every checkpoint score is only comparable to
another score made against **the same files**, so keep these fixed.

| file | contents |
|---|---|
| `probe_positions_600.json` | 600 positions (121 early / 240 mid / 239 late), stored as action histories |
| `probe_reference_600.json` | network-free reference search for them: 593 usable, 600 sims, heuristic prior + 2 heuristic rollouts, depth 300 (80.1% of rollouts reach a terminal state) |
| `probe_positions.json` / `probe_reference.json` | earlier 200-position set, used for the B002 results of 2026-09-28 |

**These cannot be regenerated.** The C++ heuristic's randomness is not seeded from
Python, so `--generate` with the same seed produces a different set. The files
are the reproducible artifact, not the seed.

Baseline on the 600 set (2026-09-30): untrained networks score **28.4% +/- 1.1**
top-1 (5 random inits). Use top-1, not policy CE: CE rewards flat distributions,
and two untrained nets beat every trained checkpoint on it.

The reference is heuristic-guided search, so it cannot credit a network for
learning things the heuristic does not know. Head-to-head play (`ab_eval.py`) is
the check for that.
