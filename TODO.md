# Mali-Ba to-do

## Starting token placement (added 2026-10-07)

Training actors place tokens uniformly at random, and from the run after D010 each
game logs a `SETUP` line. `open_spiel/python/games/mali_ba/analyze_placements.py`
reports seat-adjusted win rates by placement feature.

1. **Find out whether placement matters.** After a few thousand games with SETUP
   lines, run `analyze_placements.py`. If every effect is within about ±2 points,
   random placement is fine and the rest of this item can be dropped.
   - **First result (D011, 6,156 games, 2026-10-07): it matters, modestly.**
     Effects up to ~3-5 points, so a little beyond the ±2 threshold:
     - spreading tokens apart is the main effect: widest spread +2.3, clustered
       -3.2 (outright wins 9.5% vs 6.6%);
     - nearest cities of 3 distinct cultures +0.9, of 2 cultures -2.6 (largely
       the same effect as spread);
     - all three tokens close to cities helps a little (+1.7 vs -3.6);
     - a token right next to Timbuktu is slightly worse (-2.0).
   - **Next:** pool D011 + D012 for tighter margins (`analyze_placements.py`
     takes several logs), then step 2.
2. **If it matters, fit a placement heuristic from the data.** Add a logistic
   regression to `analyze_placements.py`: win probability from a few placement
   features (distance to Timbuktu, spread, culture diversity, ...), adjusted for
   seat. Its handful of weights become a readable scoring formula. A separate
   neural network is not worth it: a few thousand games can fit a few weights, but
   a network on raw positions would memorise (65,536 layouts).
   - The C++ heuristic already has a placement rule ("near cities in new
     regions", used by heuristic actors); the fitted weights could set it.
3. **Keep randomness if bots start placing deliberately in training.** Sample in
   proportion to the heuristic score, or place fully at random in ~20% of games,
   so the randomised comparison can be re-checked as play improves and the
   network keeps seeing varied openings. Serpentine order only starts to matter
   once placement is deliberate.
4. **Later:** let the search choose placements (reacting to and blocking
   opponents), once the value head is reliable early in the game; today it is
   weakest there (~60% top-pick before move 260).

## Rules document: rare-goods end condition (added 2026-10-07)

The rules document's Game End section (2a) says the end triggers when a player has
"five different rare goods". The code and training runs use "a rare good from five
different regions" (`end_game_cond_rare_good_each_region = true`,
`end_game_cond_rare_good_num_regions = 5`, `end_game_cond_num_rare_goods = -1`),
which the user prefers. Update the rules document to match.

## Get the policy learning (added 2026-10-08)

The policy head still learns essentially nothing: probe on 2026-10-08 (D007, D010,
D011, D012) gave top-1 31-34% (no significant differences, inside the untrained
range) and entropy 92.0% -> 93.2% of uniform, i.e. slightly *flatter* over time.
All strength gains so far (D007, D010) came through the value head. The policy
trains on MCTS visit counts, and at uct_c 2.0 those are nearly flat (~90% of
uniform), so there is nothing sharp to learn.

- Keep heuristic guidance at 0.30 until the policy carries real preferences
  (probe entropy clearly below ~90%, top-1 clearly above ~36%); then test lower
  guidance with a head-to-head (needs a per-agent --heuristic_weight in ab_eval).
- Options, roughly in order: sharpen policy targets (visit counts raised to a
  power, or temperature); KataGo-style forced playouts with policy-target pruning;
  revisit uct_c once the value head is stronger (0.5 and 1.0 vs 2.0 were null on
  D005 and D007).
- Judge any change by probe entropy/top-1 AND head-to-head (sharper targets can
  just make flat preferences look confident).
- Consider recording a new probe set under the current rules (the v2 set needs
  probe_data/mali_ba_oldrules.ini).

## Smaller items (carried over from the 2026-09-30 list, 2026-10-08)

- **Trainer busy-waits.** The idle sleep at the end of `trainer_process`'s loop
  only fires when the buffer holds fewer than `batch_size` positions, which never
  happens once training is under way, so the loop polls `get_nowait()` constantly
  and keeps a core busy. One-line fix: sleep briefly whenever nothing was drained.
- **Actor processes can be orphaned.** They are not started with `daemon=True`, so
  if the trainer is killed abruptly they keep running at ~1.2 GB each (44 strays,
  22 GB, were found once). The inference server was fixed; actors were left alone
  because they are respawned deliberately during a run.
- **Value network is 16x smaller than the policy network** (323,592 parameters vs
  5,056,836: a 3-block, 64-filter trunk and a `Dense(64)` head). Noted on
  2026-09-30 as not believed to be a bottleneck. Worth revisiting, since every
  strength gain so far (D007, D010) came through the value head.
