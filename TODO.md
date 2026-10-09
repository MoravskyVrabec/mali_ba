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
   - **Pooled D011-D013 (18,973 games, 2026-10-08): confirmed**, and every run
     agrees on its own (spread: D011 -3.2/+2.3, D012 -2.8/+2.1, D013 -3.5/+2.0).
2. **Regression done (2026-10-08, in `analyze_placements.py`).** Each feature with
   the others held fixed, seat-adjusted, standard errors clustered by game.
   Effect on winning over the feature's typical range (10th-90th percentile):
   - spread (avg distance between own tokens): **+4.3 pts** (+1.2 per hex, z 7.9);
     the only feature that also raises *outright* wins (+1.7 pts on ~7.7%)
   - average distance to nearest city: **-3.2 pts** (-4.8 per hex, z -5.6)
   - closest token to Timbuktu: **+1.8 pts** (+0.6 per hex, z 3.5), points wins only
   - distinct cultures: +1.4 pts (z 3.0); mostly spread in disguise, no effect on
     outright wins
   - seat P2 still -3.2 vs P0 in the pooled data (D013 alone was nearly even)

   Candidate heuristic score (win-probability points):
   `+1.2*spread - 4.8*avg_city_distance + 0.6*timbuktu_distance + 1.4*cultures`.
   **Built (2026-10-08), off by default:** `placement_mode = heuristic` in the
   ini makes bots sample placements by this score (spread and Timbuktu distance
   capped at 6 and 4, where the data levelled off), with
   `placement_random_fraction = 0.20` of games random as a control group and
   `placement_temperature = 1.0`. Offline, it moves spread 4.9 -> 6.2, city
   distance 1.23 -> 1.03, Timbuktu distance 2.5 -> 3.5, cultures 2.75 -> 2.96, with
   293/300 distinct openings. **Planned as D015's one change**; judge by the
   PLACEMENT MODES table in `analyze_placements.py` (heuristic vs random games, same
   run), seat balance (serpentine order now matters) and a head-to-head vs D014.
   *(Original plan for step 2, kept for context:)*
   **If it matters, fit a placement heuristic from the data.** Add a logistic
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
  - **Tested 2026-10-08 (offline, by-game split of the saved buffer): not the
    limit.** Bigger value nets overfit faster and generalised worse (held-out
    top-pick: current 54.1%, wider head 53.9%, 6x128 52.0%); all peaked after 1-2
    epochs. The value head is data-limited: the buffer spans only ~4,000 games.
    Revisit size only once the buffer spans far more games. Next experiment:
    `buffer_keep_every = 32` (~8,000 games in the same memory).

## Human play: separate the engine from the presentation layer (added 2026-10-08)

Design for review: `docs/design/ENGINE_UI_SEPARATION.md`. Goal: humans vs bots in
the local GUI and later a web front end, with move-by-move legal-move feedback and
the trained network as an opponent.

- **Blocking now:** the local GUI sends whole-turn move strings (`place (x,y,z)`,
  `mancala a:b:c post`) that the engine no longer accepts (it uses step-by-step
  actions such as `PlaceToken_(x,y,z)`, `StartMancala_`, `MancalaDir_k`); human moves
  in the GUI are very likely broken.
- GUI bots are the C++ heuristic (via a pygame timer), with a second, inconsistent
  network-bot path patched into `GameInterface`.
- Plan: GameSession + one bot interface (heuristic, network + MCTS, data-fitted
  placement) with headless tests; engine changes (per-decision move pruning,
  describe-action); then move the GUI onto it; then a web server.

## Parked: extra training compute (added 2026-10-08)

Decide only once training is clearly improving (after the policy-learning work).
First trial a rented GPU box (marketplace, a few hours) as a remote worker to
measure games per hour per dollar; then compare with owning a small-form-factor
PC (e.g. business SFF with i7-12700/13700, 32-64 GB, low-profile RTX A2000;
~$650-900 used). Workers need a GPU now (inference server); count physical
cores, not vCPUs. Rule of thumb: owning wins if a worker would run most days for
more than ~4-6 months.
Earlier distributed setups (Google Cloud spot workers and Hetzner Cloud, networked
with Tailscale) are documented in `docs/guides/GCP_SETUP.md` (section 9 = Hetzner)
and `docs/guides/DISTRIBUTED_TRAINING.md`, with scripts in
`open_spiel/python/games/mali_ba/gcp/`. They predate the GPU inference server (July
workers were CPU-only), so refresh them for GPU workers before reuse.
