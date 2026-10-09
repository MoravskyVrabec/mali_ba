# Mali-Ba diagnostic tools

What each measurement tool does, how it works, how to run it, and how to read it.
All of them live in `open_spiel/python/games/mali_ba/`.

## Which tool answers which question

| Question | Tool | Cost |
|---|---|---|
| Is the current run healthy, and is the value head improving? | `analyze_log.py` | 2-5 min, reads a log |
| Is checkpoint X **stronger** than checkpoint Y? | `ab_eval.py` | ~55-80 min for 501 games, needs the whole machine |
| Is the **policy** head learning? | `policy_probe.py` | seconds per checkpoint |
| Did the **value** head improve between runs? | `value_probe.py` | seconds per checkpoint |
| How decisive is the search? | `search_sharpness.py` | minutes |
| Do starting token positions matter? | `analyze_placements.py` | about a minute, reads logs |

Only `ab_eval.py` measures playing strength. Everything else measures something
that *usually* goes with strength, and has been misleading before: D011's value
head looked better than D010's in its own log, yet the two were level head to head.

## Running them

All tools need the trainer's environment. From `open_spiel/python/games/mali_ba`:

```bash
export PYTHONPATH=/media/robp/UD/Projects/open_spiel/build/python:/media/robp/UD/Projects/open_spiel:/media/robp/UD/Projects/mali_ba/open_spiel/python/games
/home/robp/miniconda3/envs/mali_ba/bin/python <tool>.py ...
```

The default `python` on the path has no TensorFlow. `ab_eval.py` and
`search_sharpness.py` compete with a training run for the CPU and GPU; run them
with training stopped. The others are light.

---

## analyze_log.py: what happened in a training run

Reads a training log (`train_runD0xx.log`) and prints a report; optionally shows charts.

```bash
python analyze_log.py /media/robp/UD/Projects/open_spiel/train_runD012.log [--no-plot]
```

Sections worth reading, in order of usefulness:

- **VALUE HEAD CALIBRATION.** Built from the actors' mid-game *value checks*: every
  20 moves from move 200, an actor asks the value head who will win, on a game that
  is being played for the first time (so it is a held-out test, not training data).
  - *top-pick*: how often the value head rates the eventual winner (or the timeout
    leader) highest. Chance is 33%. D007-D011 sat at 69-73%.
  - *avg per player* gap: average prediction minus average training target. Positive
    means "too optimistic about everyone", i.e. it expects more games to end in a
    win than do. 0.07-0.11 has been normal; D008 reached 0.14 together with a
    top-pick drop, which was the sign of trouble (memorization).
  - *thirds*: the same numbers for the first, middle and last third of the run.
    Flat thirds mean no change during the run.
- **VALUE HEAD TRAJECTORY.** The eventual winner's predicted value every 20 moves,
  split by ending type. Should rise towards the actual result late in the game;
  a late dip means the head does not see that kind of win coming.
- **AUX VALUE HEADS** (runs with `aux_targets` on). The extra heads that predict how
  the game ends. Their win-type accuracy should beat *always-majority* (always
  guessing "timeout"); the *gain* column is the margin. Early in a run, compare the
  win-type loss with the base-rate loss (~0.60-0.75) instead: it shows learning
  before accuracy moves.
- **WIN CONDITIONS / PLAYER WIN BALANCE.** How games ended and seat win shares.
  Self-play win rate is *not* a strength measure (both sides improve together).
- **SAMPLE REUSE.** How many times each stored position is trained on. ~2-2.5 has
  been healthy; ~5.5 went with memorization (D005, D008).
- **EARLY TERMINATIONS / NO-KILL.** Culling is off since 2026-10-08
  (`random_no_kill_thresh = 1.0`); games that would have been culled are marked
  "NO-KILL - would have terminated" and play on.

Also: it writes a value-check CSV next to the log, prints the `taskset` command to
pin the inference server (or says it is already pinned), and its charts show the
value checks, search cost and a 60-game win-rate average.

**Pitfalls.** Loss curves do not measure strength (a rising loss usually means the
buffer is filling). These are self-play numbers on the run's own games.

---

## ab_eval.py: head-to-head strength

Plays two (or three) checkpoints against each other and reports who wins.

```bash
# two-way: A takes one seat, B the other two
python ab_eval.py --agent_a mali_ba_agent_vD012.weights.h5 \
    --agent_b mali_ba_agent_vD011.weights.h5 \
    --sims 300 --heuristic_weight 0.30 --games 501 --workers 64 \
    --config_file $PWD/mali_ba.ini > ab_D012_vs_D011.log 2>&1

# three-way: A, B and C one seat each
python ab_eval.py --agent_a ...D012... --agent_b ...D011... --agent_c ...D010... \
    --sims 300 --heuristic_weight 0.30 --games 504 --workers 64 \
    --config_file $PWD/mali_ba.ini > ab3.log 2>&1
```

**How it works.**
- Each game is a full game of Mali-Ba. Every seat is played by an MCTS search
  (300 simulations per move, heuristic guidance 0.30, `uct_c` 2.0 by default) using
  the checkpoint assigned to that seat. Moves are chosen greedily (most-visited), with
  no exploration noise, so each side plays its best.
- **Seat rotation.** Seat 0 has a large first-mover advantage. In two-way mode, A
  sits in each seat equally often (A@0, A@1, A@2); in three-way mode, all 6 seat
  orders are played equally often. Jobs are *interleaved* across rotations, so a
  partial run is balanced and a mid-run reading is meaningful (fixed 2026-10-08;
  before that, early results were mostly A in seat 0).
- **Boards.** Each game's meeple layout and token placement come from a fixed seed,
  so the same command plays the same boards.
- **Inference.** One batched GPU inference server per distinct checkpoint; it pins
  itself to the fast cores. 64 workers play games in parallel.

**Reading the result.**
- Only *decided* games count (timeouts at move 420 have no winner).
- Two-way: if equal, A wins 1/3 of decided games. The 95% interval (Wilson) decides:
  entirely above 0.333 means A is stronger, entirely below means B is, otherwise no
  significant difference. With ~450 decided games the interval is about +/-4.5
  points, so A needs ~38% to show a gain.
- Three-way: each agent's share against 1/3. The shares sum to 1, so the intervals
  are not independent; use it to rank, and settle a close pair with a two-way run.
- The terminal shows a live progress line (game n/total, time left, score); the log
  gets one line per game and the summary.

**Pitfalls.**
- Rules: play under the ini you want. Checkpoints trained before the 2026-10-06
  setup rules are at a disadvantage under the new rules (and vice versa); to check
  that a gain is not just familiarity with the rules, run it under both (the
  old-rules ini is `probe_data/mali_ba_oldrules.ini`).
- Use the *final* weights of a stopped run (`mali_ba_agent_vD0xx.weights.h5`), not a
  checkpoint of a run that is still training.
- Results so far: D007 > D005 (0.499), D008 < D007 (0.252), D010 > D007 (0.431),
  D011 = D010, D012 < D011 and D010 (three-way, 0.286).

---

## policy_probe.py: is the policy head learning?

Answers one narrow question without playing any games: is the policy head getting
better at predicting good moves?

```bash
python policy_probe.py --positions probe_data/probe_positions_v2_600.json \
    --reference probe_data/probe_reference_v2_600.json \
    --config_file $PWD/probe_data/mali_ba_oldrules.ini --workers 12 \
    --agents mali_ba_agent_vD007.weights.h5 mali_ba_agent_vD012.weights.h5
```

**How it works.**
1. **A frozen set of 600 positions** (`probe_positions_v2_600.json`), taken from 61
   heuristic-played games at moves 40-400. They are stored as move sequences;
   the probe rebuilds each position by replaying its moves. Because the set never
   changes, every checkpoint is tested on exactly the same positions.
2. **A fixed reference answer for each position** (`probe_reference_v2_600.json`),
   computed once: a 600-simulation search that uses **no network** (heuristic prior
   plus heuristic rollouts). Its most-visited move is the reference move. Using a
   network-free search matters: a search guided by the network being tested would
   partly agree with itself.
3. **For each checkpoint**, the probe runs the policy head once per position (no
   search) and compares its move preferences with the reference.

**What it reports.**
- *top-1*: how often the policy's favourite legal move is the reference move. The
  main number. An untrained network scores 24.8% +/- 7.8 (best random init 32%), so
  a trained network should clearly beat ~32-36% to count as learning.
- *top-3*: the policy's favourite is among the reference's three most-visited.
- *policy CE*: cross-entropy against the reference visit distribution. Do not read
  it alone: a flatter policy is penalised less.
- *entropy* (as % of uniform): how decisive the policy is. 100% = no preference.
  This is the key number for "is the policy learning at all". Every checkpoint
  D007-D012 sat at 92-93%.
- *max p*: the policy's largest probability.
- *by phase*: early (<120), mid (120-279), late (280+) moves.
- **Paired comparison** (McNemar test): for each pair of checkpoints, counts the
  positions only one of them got right. Because both see identical positions, this
  is much more sensitive than comparing two percentages. Run-to-run noise on the
  same checkpoint is ~2-2.5 points of top-1.

**Pitfalls.**
- **Use `probe_data/mali_ba_oldrules.ini`** for the v2 set (it was recorded before
  the setup rules changed). With the wrong rules the probe now stops within seconds
  with a clear message; before 2026-10-08 it hung.
- The reference is heuristic-guided, so the probe cannot credit a network for
  learning things the heuristic does not know. It measures policy learning, not
  strength; head-to-head play is the strength test.
- See `probe_data/README.md` for the set's history and the stale sets not to use.

---

## value_probe.py: did the value head improve between runs?

analyze_log's value numbers score each run on its own games, so they do not compare
across runs that play or keep different games (D011 looked better than D010 but was
level head to head; D013 looked worse than D011 but won). This probe scores every
checkpoint's value head on the **same** held-out positions.

```bash
# 1. record a set: any head-to-head can do it as a side effect
python ab_eval.py --agent_a ... --agent_b ... --sims 300 --heuristic_weight 0.30 \
    --games 501 --workers 64 --config_file $PWD/mali_ba.ini \
    --record_positions /media/robp/UD/Projects/open_spiel/value_probe_set.npz > ab.log 2>&1
# 2. score checkpoints on it (CPU, seconds each)
python value_probe.py --positions /media/robp/UD/Projects/open_spiel/value_probe_set.npz \
    --agents mali_ba_agent_vD011.weights.h5 mali_ba_agent_vD013.weights.h5
```

**How it works.** `--record_positions` saves every 8th move of each head-to-head game
(the mover's observation) with the game's final result. Head-to-head games are never
trained on, so the set is held out for every checkpoint. Positions are stored as
observations, not move histories, so a rules change cannot break the set (unlike the
policy probe); it only needs the same 192-plane layout. A 501-game run gives ~25,000
positions (~18 MB).

**What it reports.** Top-pick (does the value head rate highest the player with the
best final return, timeout leader included), overall, by moves remaining and by how the
game ended; MSE against final returns; calibration (average prediction minus average
result). Against the first checkpoint listed, the top-pick difference with a 95%
interval from resampling whole games (positions in one game are correlated).

**Pitfalls.** The set reflects the two agents that played it; refresh it now and then
from recent head-to-heads. It measures judgement on positions, not play.

---

## search_sharpness.py: how decisive is the search?

Runs the training search (same settings as the actors, no exploration noise) with one
checkpoint on the probe positions, at several `uct_c` values, and reports how flat
the visit counts are compared with the prior and how far apart the moves' Q values
are. It explains *why* the policy has nothing to learn: at `uct_c` 2.0 visits are
~89% of uniform and the moves' Q values differ by only ~0.08.

```bash
python search_sharpness.py --agent mali_ba_agent_vD011.weights.h5 --uct_c 2.0 1.0 0.5
```

Sharper is not automatically better: lower `uct_c` was tested head to head with D005
and D007 and did not play better. Like the probe, it uses the v2 positions, so its
`--config_file` defaults to `probe_data/mali_ba_oldrules.ini`; with other rules it
stops with an error instead of hanging.

---

## analyze_placements.py: do starting token positions matter?

Training actors place tokens **uniformly at random** and, since 2026-10-07, log one
line per game:

```
Actor 52, Game 1234: SETUP layout=Setup_40213 tokens P0=(x,y,z);(..);(..) P1=... P2=...
```

The script pairs each SETUP line with that game's result and reports win rates by
placement feature. Because placement is random, these are *randomised* comparisons:
a difference is caused by the placement, not by stronger players choosing better spots.

```bash
python analyze_placements.py /media/robp/UD/Projects/open_spiel/train_runD011.log [more logs]
```

**How it works.**
- For each seat it measures: distance from its tokens to the nearest city (closest
  token, and the average of the three), distance from the closest token to Timbuktu,
  spread (average distance between its own tokens), and how many different cultures
  its nearest cities belong to. Features that never vary are skipped.
- **Seat adjustment.** Seats differ in strength, so for each bucket it compares the
  win rate with what those seats would win on average ("expected"). The difference
  is the *effect*, with a +/-2 standard-error margin; `*` marks effects beyond it.
- "Won" means best return: the winner of a Timbuktu or rare-goods game, or the
  leader at a timeout. It also shows outright wins and the Timbuktu/rare-goods split.

**Regression section.** Because the features overlap (spread-out tokens also tend to
sit farther from Timbuktu and near more cultures), the script ends with a logistic
regression: each feature's effect with the others held fixed, adjusted for seat, with
standard errors clustered by game. Read this, not the single-feature tables, when
deciding which feature matters. It is reported for best-return wins and for outright
wins.

**Result (D011-D013 pooled, 18,973 games):** spreading tokens apart is the main effect
(+4.3 points across its typical range, and the only one that raises outright wins);
each token near its own city (-3.2 points for being farther); a little better away
from Timbuktu (+1.8); culture diversity is mostly spread in disguise.

---

## One-off analyses

Some questions were answered with short scripts against the saved replay buffer
(`/media/robp/UD/Projects/open_spiel/mali_ba_buffer.pkl.gz`), using
`pretrain_heuristic.load_eval_buffer` to read it. They are not kept as tools, but
the method is reusable:
- *Score planes vs value head* (2026-10-06): the value head beats simply reading the
  score at every stage except the last ~20 moves.
- *Is the current score leader the final timeout leader?* (2026-10-08): yes, 96%+
  of the time with a lead of 125+ points at moves 320-359, 75+ at 360-399, 50+ at
  400+; small leads are near coin flips. Basis for ending settled games early.
