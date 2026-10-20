# Value head: where it stands and what to try (19 October 2026)

Prepared while D042 runs, for use if D042 ties D041. Read-only analysis; nothing the
training uses was changed.

## Findings

**1. The value head improved through D039 and has been flat since.** Probe v3
(15,492 held-out positions from D031 vs D028 games), all agents mover-relative:

| | top-pick | early (>200 left) | mid (61-200) | late (<=60) | balanced | MSE |
|---|---|---|---|---|---|---|
| D032 | 72.8% | 60.6% | 72.2% | 86.8% | 66.4% | 0.361 |
| D036 | 76.1% | 64.3% | 76.0% | 88.8% | 68.3% | 0.322 |
| D039 | 76.7% | 64.8% | 76.2% | 90.0% | 69.6% | 0.314 |
| D040 | 75.3% | 60.5% | 75.5% | 90.3% | 69.2% | 0.311 |
| D041 | 75.9% | 61.3% | 76.0% | 90.9% | 69.6% | 0.307 |

D036, D039, D040 and D041 all beat D032 significantly (top-pick +2.5 to +3.8, balanced
+1.8 to +3.2; intervals from resampling games); D039 to D041 do not differ. Play
followed the same shape: gains through D039, then ties (D040 vs D039 0.358, D041 vs D040
0.352). Late positions keep improving slightly; mid-game is flat at 76%.

**2. Early positions got worse after D039** (64.8% -> ~61%). Self-play games now end
around move 150-210, so training sees few positions with more than 200 moves still to
play; the newer heads are less practised there. Partly a probe-set artefact (v3 games
are longer than current ones).

**3. Not memorising; flat on its own training data.** Training value MSE by fifth of
each run: D039 0.441 -> 0.435, D040 0.432 -> 0.424, D041 0.419 -> 0.428. It is higher
than the held-out probe MSE (~0.31; different mix of positions, so only roughly
comparable), so there is no sign of overfitting. The head sits near the floor its
targets allow: the target is the game's final outcome, which from a mid-game position
is a noisy single sample of what could happen.

## Proposal (D043, if D042 ties)

**Blend the search's own value into the value target** ("z + q" targets, used by
several AlphaZero reimplementations to cut target noise): for each stored position,
target = (1 - lam) * outcome + lam * root search value, for the player to move;
other players keep the outcome. Start with lam = 0.5 on full-search moves only (their
root value comes from 300-800 simulations; fast-search moves keep the plain outcome).

Work needed (none while a run is going):
- Actors: record the root value of each full search (the searching player's average
  reward at the root) alongside the policy target. MCTS backs up a per-player value,
  so only the mover's value is available; that is the one the blend uses.
- Buffer entry: add the root value (or store the blended vector directly), with
  buffer_format.py converting older saves (no root value = plain outcome).
- Trainer: unchanged if the blend is stored; ini setting `value_target_search_weight`
  (0 = current behaviour).
- Measure: probe top-pick / balanced (fresh set), training value MSE (expected to drop,
  as the targets get less noisy), and head-to-head as always.

Cheaper alternatives, if the blend is too invasive: a higher value-loss weight, or a
value-only learning-rate bump; both are ini/trainer changes but don't address noisy
targets.

## Fresh probe set

v3 no longer looks like current play (longer games, seat 2 rarely best). Record a new
set from the D042 vs D041 head-to-head by adding
`--record_positions /media/robp/UD/Projects/open_spiel/value_probe_set_v4.npz`
(a new file name: --record_positions overwrites).
