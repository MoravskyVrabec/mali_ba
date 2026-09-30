# Open items

Ordered by expected value, highest first. Each entry records what was measured, so a
future session doesn't have to re-derive it.

---

## 1. Convolutions run over the wrong axes  (IMPORTANT, unproven, needs a full retrain)

**STATUS 2026-09-30: IMPLEMENTED.** `layers.Permute((2, 3, 1))` is in both networks as of the
B-series runs (B001 onward), bundled with the move-count plane and the heuristic-weight decay,
so a result cannot be attributed to any one of the three. Whether it helped is still open: on
the fixed-reference probe, B002/B005 sit ~4-5pp top-1 above an untrained net and flat within
runs. Note that probe's reference is heuristic-guided search, so it cannot credit learning that
departs from the heuristic -- head-to-head play is the test that can.

`observation_tensor_shape()` is `[95, 15, 15]` — planes, height, width. The Python
model feeds that straight into Keras, which defaults to `channels_last` and therefore
reads the **last** axis as the feature count:

```
model.input_shape = (None, 95, 15, 15)
first Conv2D: data_format=channels_last, kernel=(3, 3, 15, 128)
                                                     ^^ 15 = board width, used as CHANNELS
```

So Keras sees a 95x15 image with 15 channels, when the data is a 15x15 board with 95
feature planes. Each 3x3 kernel spans **3 adjacent plane indices x 3 board columns**,
treating unrelated feature planes (say "player 1 posts" and "player 2 posts") as
spatially adjacent, and handling the two board axes asymmetrically.

Intended layout is `(15, 15, 95)`.

**Fix** — one line in `create_mali_ba_policy_network` / `create_mali_ba_value_network`:

```python
x = layers.Permute((2, 3, 1))(inputs)   # (95,15,15) -> (15,15,95)
```

The model's declared input shape stays `(95,15,15)`, so `AlphaZeroEvaluator`, the
inference server and the trainer all keep passing observations unchanged. No C++
change, no rebuild. The **buffer stays valid** (observations are stored flat and the
tensor shape is unchanged). **All existing weights become unusable** — the stem kernel
changes from `(3,3,15,128)` to `(3,3,95,128)` — so this is a retrain from scratch.

**Why it is not obviously urgent.** An offline A/B on 24k buffer samples was
inconclusive: mid-game AUC 0.591 (current) vs 0.594 (permuted), which is inside the
+/-0.033 noise on a 304-position probe. The permuted net had not converged (test MSE
0.104 vs 0.025 after six epochs), so its accuracy was not fairly measured. Separately,
the current value head already picks the eventual winner at AUC 0.644, close to a
logistic regression fitted on hand-chosen features (0.686) — if the axes were crippling
it, that should not be true.

Plausible reading: global aggregate features (counting rare goods across planes) survive
the scrambling, which is what the rare-goods win condition needs; genuine 2D spatial
reasoning does not, which may be why Timbuktu (connectivity) wins are the ones the value
head reads worst.

**To settle it:** train both variants to convergence and compare mid-game AUC. Needs
hours of GPU time that self-play is currently using.

---

## 2. `clear_winner_thresh` is discarding ~47% of self-play compute

**STATUS 2026-09-30: set back to 0.30 in mali_ba.ini.**

Measured on runA003: 73 games culled at a **median of move 400**, discarding 28,460 MCTS
moves against 32,017 kept. Games are culled after nearly all their cost has been paid,
and the whole trajectory is thrown away.

`clear_winner_thresh = 0.60` was raised from 0.30 because a timeout leader's target sat
at +0.40 and 0.30 would have wrongly protected doomed games. `timeout_leader_reward =
0.7` dropped that target to +0.10, so the justification is gone.

**Set it back to 0.30.** Note also that it gates on a signal with AUC ~0.45 for the
game-level "will anyone win" question, so any value is close to arbitrary — which argues
for culling less rather than tuning the number.

---

## 3. No way to tell whether the agent is improving

**STATUS 2026-09-30: `policy_probe.py` added** (see its docstring). Use `--reference` mode: the
self-search mode is circular (a random net scored 40% top-1 that way). Top-1 is the usable
metric; policy CE rewards flat distributions even against the fixed reference, and two
untrained nets beat every trained checkpoint on CE. The random-init floor is 28.4% +/- 1.1
top-1 on the 593-position reference (5 inits). Pre-B001 checkpoints cannot be scored (95 planes).

**Self-play win rate cannot measure strength.** Both sides improve together, so it mostly
reflects how often games reach a win condition before the move cap. It has sat near 34%
across runs 113, A002, A003 and A004 — which is what you would expect from an agent
improving steadily.

Needs head-to-head play against a **fixed** reference (an old checkpoint, or the
heuristic bot), rotating seats to cancel the known turn-order advantage. Checkpoints back
to v101 exist and all share the same architecture and observation shape, so any of them
works as an opponent. See `ab_eval.py`.

---

## 4. Value-head trajectory table in `analyze_log.py` is misleading

The `VALUE HEAD TRAJECTORY` section compares mid-game predictions against each game's
**eventual** terminal return. At move 200 the outcome is not yet determined, so a
prediction near the base rate is correct behaviour, and the table scores it as a failure.
It led to two wrong diagnoses in one session ("the value head is compressed", "the value
head is anti-predictive").

Either drop the section or replace it with a discrimination metric — per-player AUC for
"does this player win", which is what the network is actually being asked to do.

---

## 5. Smaller items

- **Trainer spins a core.** The idle-guard at the end of `trainer_process` only sleeps
  when the buffer is below `batch_size`, which never happens once training is underway,
  so the loop busy-polls `get_nowait()`. One-line fix: sleep briefly whenever nothing was
  drained.
- **Actor processes can orphan.** They are not `daemon=True`, so a parent killed abruptly
  leaves them running at ~1.2 GB each (44 such strays were found once, holding 22 GB).
  The inference server was fixed; actors were left alone because they are respawned
  deliberately during a run.
- **Value net is 16x smaller than the policy net** (323,592 params vs 5,056,836) with a
  3-block/64-filter trunk and `Dense(64)` head. Not currently believed to be a
  bottleneck — it explains 98% of target variance — but noted.
- **`timeout_leader_reward` and win/timeout contrast.** Currently 0.7, giving 0.90 of
  separation between a real win (+1.00) and a timeout leader (+0.10). Lower values raise
  the contrast further but invert the ordering below 0.5 (being ahead at the timeout would
  score worse than losing a decided game).
