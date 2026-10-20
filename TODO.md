# Mali-Ba to-do

## FIRST: desktop memory errors (added 2026-10-11)

**Status 2026-10-19: no recurrence.** No machine-check errors in the kernel logs since
2026-10-12, `HardwareCorrupted: 0 kB` (since the reboot of 2026-10-19 18:04), and runs
D021-D041 finished cleanly. The user thinks it was a one-off; kept here to watch. Quick
check: `grep HardwareCorrupted /proc/meminfo` and
`journalctl -k --since YYYY-MM-DD | grep -i "memory corruption"`.

The desktop is reporting uncorrectable RAM errors. On 2026-10-11 the kernel killed
68 Python processes with machine-check exceptions ("MCE: Killing python ... due to
hardware memory corruption fault"), from 16:14 on; most were D020 actors (silently
respawned), then at 19:41 the trainer itself, mid-save. There was an earlier one on
2026-10-09 19:05 (during D016). `/proc/meminfo` showed `HardwareCorrupted: 8 kB`:
the kernel had retired two physical pages. The error records say `UE ... Poison`
(uncorrectable). Until this is fixed any run can lose its trainer, and corruption
the hardware does not catch is possible too.

Steps:

1. **Record the memory configuration:**
   `sudo dmidecode -t memory | grep -E "Locator|Size|Speed|Part Number"`
   (number of modules, rated vs configured speed, part numbers).
2. **Check the BIOS memory settings.** With 4 DDR5 modules on AM5 (7950X3D),
   EXPO/XMP overclocks are a common cause of exactly this. Try EXPO off, or a lower
   speed (e.g. 4800 MT/s), before suspecting a module.
3. **Run MemTest86+** (Ubuntu's boot menu, or a USB stick from memtest.org): at
   least one full pass at the current settings, to confirm the fault, then again
   after any change.
4. **If one module shows errors:** reseat it, retest; test modules one at a time
   to find the bad one; replace it (the kit may be under warranty).
5. **After a fix, watch for recurrence** during the next runs:
   `journalctl -k | grep -iE "MCE|Hardware Error|Memory failure"` and
   `grep HardwareCorrupted /proc/meminfo` (should stay 0 kB after a reboot).
6. Since 2026-10-11 a run stops itself with an error when its trainer dies, rather
   than playing on for hours; resume from the last 2-hour checkpoint (the main
   weights file may be cut off: D020's `_policy` file was 1.5 MB instead of 9.2 MB, and its 19:41
   checkpoint was renamed to replace the main `vD020` files).

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

**Status 2026-10-19: largely achieved.** Since uct_c 1.0 -> 0.5, policy-target pruning,
more full searches (0.55) and guidance 0, the targets are sharp (target H ~0.59) and
the policy follows them (gap ~0.22, lowest 0.218 in D041); the models now beat a human
easily. Remaining work is the value head (see "Value head" below). The history below is
kept for reference.

The policy head still learns essentially nothing: probe on 2026-10-08 (D007, D010,
D011, D012) gave top-1 31-34% (no significant differences, inside the untrained
range) and entropy 92.0% -> 93.2% of uniform, i.e. slightly *flatter* over time.
All strength gains so far (D007, D010) came through the value head. The policy
trains on MCTS visit counts, and at uct_c 2.0 those are nearly flat (~90% of
uniform), so there is nothing sharp to learn.

- **2026-10-11 finding:** most of the policy loss was not about move choice.
  Policy loss ~2.78 = target entropy ~1.75 (the search's own indecision, ~91% of
  flat) + a ~1.0 gap, and ~0.67 of that gap was the network putting ~52% of its
  probability on illegal actions (624-way softmax, never masked). The same flaw
  inflated guidance: the heuristic's real share was ~0.38 at setting 0.20, ~0.50
  at 0.30, ~0.55 at ab_eval's 0.35. Fixed in 1411db0: `policy_legal_mask = true`
  (loss over legal moves only), `guidance_blend_normalized = true` with the weight
  at 0.38 (same real share as before). Judge D021 on by the "gap" in analyze_log's
  POLICY FIT section, not the loss.
- **First masked step (smoke test, D020 checkpoint):** gap over legal moves only
  0.11 (target entropy 1.59, CE over legal 1.70). The network already matches its
  targets over the legal moves; **the flatness is in the search targets.** Next
  levers, roughly in order: a better value head (bigger Q differences between
  moves), more simulations on recorded full-search moves, Gumbel-style policy
  targets (prior + searched values instead of visit counts), per-node Q
  normalization (test head to head; lower uct_c was null).
- Guidance results so far (real shares): 0.27 vs 0.50 at play time lost (D017);
  training at ~0.38 vs ~0.50 (D018) did not help by itself.
- **2026-10-12 (D021 done):** the POLICY FIT gap stayed at ~0.09 for the whole
  run and target entropy rose slightly (1.74 -> 1.79). D021 / D020 / D019 were
  even three-way (0.350 / 0.327 / 0.323), and the value probe puts D017-D021 level.
  Legal moves per position: median 5, p90 7; only ~5% have 20+ (mostly route
  decisions). Self-play looked nothing like strong play: D021 games ~465 moves, 70%
  timeouts; greedy 300-sim head-to-heads median 246 moves, 2% timeouts. Every
  self-play move, including 40-sim fast moves, was sampled at temperature 1.0 for
  150 moves, then 0.5, over near-flat visits.
- **D022 (stopped after ~20 min, 2026-10-12):** self-play move selection only. `selfplay_temp_moves = 30`,
  `selfplay_temp_late = 0.25`, `playout_cap_fast_greedy = true` (fast moves play
  the most-visited move). Check: win-condition share and natural game length in
  analyze_log, POLICY FIT target entropy over the run, head-to-head vs D021. Watch
  for lost game variety (value checks worse, near-identical games).
- **D023 (from D021):** D022's move selection **plus rebalanced buffer pools**:
  near-win 0.20, rare goods 0.35, Timbuktu 0.30, timeout 0.15 (was 0.35 / 0.20 /
  0.10 / 0.35). D022's first 326 games: 77% natural wins (49% of games rare goods,
  28% Timbuktu, 23% timeouts), median win 322 moves, 1,222 games/hr. At the old
  shares the timeout pools (70% of every batch) would have stayed mostly D021 data
  for ~16 h, with new timeout positions trained ~5.8x and win positions ~1x.
  train_interval_seconds stays 12 (reuse 2.6x average). Two changes in one run:
  a D023-vs-D021 result can't separate them.
- **Candidates for later runs, one per run:**
  1. **Policy-target pruning (KataGo).** Before recording a target, cut each
     non-best child's visits down to what its PUCT score justifies against the
     best child, removing visits caused by Dirichlet noise and forced exploration.
     Changes recorded targets only, not play. Python only, ~30 lines in the actor.
  2. **First-play urgency (FPU).** OpenSpiel's `puct_value` values unvisited
     children at 0, while the average value is ~-0.47, so the search visits every
     legal move once first. Negligible for the typical 5-7 legal moves; it matters
     for large route decisions (up to 300 candidates). Fix: value an unvisited
     child at its visited siblings' mean Q minus `fpu_reduction` (default off), in
     a small `MCTSBot` subclass in mali_ba (the selection function can't see
     siblings). Add to ab_eval per agent; test with search_sharpness.py and a
     same-network head-to-head before training with it.
  3. **Larger fast budget** (`playout_cap_fast_sims` 40 -> 80): ~7 visits per legal
     move is thin. Costs compute.
  4. **Gumbel-style completed-Q targets.** A bigger change, and with Q spreads of
     0.04-0.08 they would come out nearly flat until the value head improves.
  5. **More simulations on recorded moves.** Costly; at uct_c 2.0 the gain is
     modest (search_sharpness: 89% vs 93% of uniform entropy).
  - Revisit uct_c once the value head is stronger (0.5 and 1.0 vs 2.0 were null
    on D005 and D007, measured with FPU at 0).
  - **2026-10-13 search_sharpness on D023** (120 old-rules probe positions, 300
    sims, normalized blend). Guidance barely changes target sharpness: visit
    entropy 91 / 90 / 90 / 87% of uniform at guidance 0 / 0.10 / 0.38 / 1.0
    (prior 95% for the network, 89% for the heuristic alone). uct_c does, at
    guidance 0.10: 2.0 / 1.0 / 0.5 / 0.25 -> 90 / 86 / 79 / 70%, same best move as
    at 2.0 in 100 / 88 / 74 / 68% of positions; median child Q spread only 0.022.
    Next: same-network head-to-head D023 uct_c 1.0 vs 2.0 at guidance 0.55; if 1.0
    wins or ties, train a run at 1.0.
  - Not recommended: sharpening targets with a temperature (visits^(1/T)). It adds
    no information and amplifies search noise.
- Judge any change by probe entropy/top-1 AND head-to-head (sharper targets can
  just make flat preferences look confident).
- Consider recording a new probe set under the current rules (the v2 set needs
  probe_data/mali_ba_oldrules.ini).

## Seat 2 is getting weaker (added 2026-10-12)

Seat 2's share of self-play wins: D017 30.4%, D019 29.4%, D021 26.0%; of
rare-goods wins 19.5% -> 12.8% -> 8.5% (Timbuktu 29.1% -> 23.6% -> 19.5%). In the
D021/D020/D019 three-way it won 30 of 492 decided games (6%), and every agent was
weak from that seat. Candidate causes: the last seat's starting goods or serpentine
placement backfiring in stronger play; rare goods running out before seat 2 gets
to them; games ending (median 246 moves at 300 sims) before seat 2's advantages
pay off. First checks: analyze_placements.py by seat, per-seat goods over the game
in the eval logs. Re-check after D022, whose self-play is closer to strong play.

**Resolved 2026-10-15 (cause found; see docs/handoffs/HANDOFF-20261016.html):** the
collapse was learned, not structural. Rule changes barely helped (seat 2 at 1-17%
under extra starting goods, 2.3% with finish-the-round + seat 2 moving first); the
heuristic alone gives seat 2 its share; and D028 shown seat-2 positions as player 0
(once per search) won 24% from seat 2 instead of 1.6%. The absolute per-seat
observation and value outputs let the network learn separate play per seat.
Mover-relative training (mover_relative_obs) gives even seats (D030 self-play
33.5 / 32.6 / 33.8%) but D029/D030 lost strength (D030 vs D028: 0.105) because they
started from D028 searched per position. Next: **D031 with the self-play teacher**
(selfplay_teacher_weights = D028, per-search relabel), training from D028's weights
on a fresh buffer. **Update 2026-10-16 evening:** the lost strength was mostly the
per-position search (each node from its own mover's view). Searched one view per
search, D031 is level with D028 (0.362) and D030 scores 0.256 (was 0.105); fixed in
116938a. D032 continues with normal self-play from D031 (teacher off). Then:
- Evaluate D031 mover-relative vs D028 and D030; record value_probe_set_v3.npz from
  one of those games (v2 is biased against mover-relative models).
- If D031 is near D028: teacher off, continue normal self-play from D031.
- Once a seat-neutral network is strong, re-measure seat balance with
  seat_balance.py --mover_relative; only then consider the rule options built for
  this (starting_*_goods_by_seat, end_game_finish_round, main_play_start_seat).

## Smaller items (carried over from the 2026-09-30 list, 2026-10-08)

- **Make the normalized guidance blend the default (added 2026-10-13; not while a
  run is active, actors import training_utils.py live).** `AlphaZeroEvaluator`
  defaults to `blend_normalized=False`, the old blend. Since the masked loss,
  D021+ policies put ~90% of their probability on illegal moves, so any tool that
  builds an evaluator without the option gives the heuristic a much larger real
  share than its weight (0.10 acts like ~0.5). Training and ab_eval pass it;
  search_sharpness.py got `--guidance_blend_normalized` in f9f8444;
  **policy_probe.py still uses the old blend**. Fix: default to normalized, keep the
  old blend as an opt-in for reproducing pre-D021 results; optionally mask the
  network output to legal moves inside the evaluator. policy_probe's entropy also
  needs to be over legal moves only.

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

## Human play: game server and front ends (DONE 2026-10-18; follow-ups below)

Built (commits d57dd11, 8e38db9, 7be38fb, 23d7063): `play/` package -- GameSession
(any mix of human and bot seats; legal moves described for display; route options carry
their hexes, read from the engine by applying each to a cloned state), heuristic and
network bots (ab_eval's search, 300 sims default), a FastAPI/WebSocket game server
(protocol v1, `docs/design/GAME_SERVER_PROTOCOL.md`), a pygame client and a web client
served by the server. No C++ changes were needed.

Since then (2026-10-18/19): protocol 2 (Restart Turn / Complete Turn; each token
placement is a turn), End Game with confirmation, the ini rule
`block_consecutive_income` (off for human play via the server's PLAY_RULES; built),
a hex menu before upgrades and mancala starts, centers drawn larger, city names behind
the pieces (Show City Names), Show Regions. Both clients get every change.

- **Build pending (after D042):** region bindings (5d65fc7, `games_mali_ba.cc`). Then
  restart the server and check Show Regions appears; the engine should report 6
  regions of 9, 9, 18, 12, 15 and 19 hexes.
- The old GUI (`main.py`, `ui/visualizer.py`, `utils/cpp_interface.py`) is untouched:
  its live modes send whole-turn move strings the engine no longer accepts, and its
  rule copies (`ui/visualizer_other.py`) are stale. Replay mode may still work. Decide:
  retire the live modes, or port replay to the new clients (games the server saves to
  `play/games_played/` are JSON with the full move history).
- Placement for network bots uses the heuristic (self-play places tokens at random).
- No login on the server: expose it only on the LAN or Tailscale.
- Ideas: undo for hot-seat games; a replay viewer for saved games; show each player's
  goods in detail; a model's value estimate as an optional "who's winning" bar.

## Value head: search value in the target (added 2026-10-19)

The value head has been flat since D039 (probe v3 top-pick ~76%, balanced ~69.6%;
training value error ~0.43 for three runs) and is not overfitting: its targets (one
game's final outcome) are the noisy limit. If D042 ties D041, D043 should mix the
search's root value into the target: (1 - lam) * outcome + lam * search value, for the
player to move only, lam = 0.5, full-search moves only. Work: actors store the root
value; buffer entry + `buffer_format.py` conversion of older saves; ini
`value_target_search_weight` (0 = current behaviour). Measure training value error,
probe v4 (record it in the D042 vs D041 head-to-head, new file name), head-to-head.
Analysis: `docs/notes/VALUE_HEAD_2026-10-19.md`.

## Rules: make points matter (added 2026-10-19; the user is considering)

Both win conditions end the game at once, so points decide only the ~3% of games that
reach the move cap (D040 self-play: rare goods 71%, Timbuktu 26%, move cap 3%). A human
playing for points loses to a race: in the first human game D040 won by Timbuktu to
coast after ~9 mancala turns each. Options, gentlest first:
- `end_game_finish_round = true` (exists): everyone else gets one more turn.
- A win condition becomes a large point bonus that ends the game after the round, so
  points decide every game.
- Raise the bar: minimum route length or regions crossed, centers at both ends, or a
  prerequisite (`end_game_req_num_routes` exists).
- More interaction around Timbuktu (limit posts there; blocking).
Make each an ini switch. A network trained under the old rules underplays new ones:
judge candidate rules from a short fine-tuning run from the current model (win-type
mix, game length, whether points decide), not from games against today's model.

## Quick-play option for experienced players (added 2026-10-19)

Both clients now ask before upgrades and mancala starts: a click on such a hex opens a
menu beside it, even when it is the only move there (web 4780bb7, pygame since the
parity commit of 2026-10-19). Add a per-player setting
("quick play"), remembered per browser, under which a hex with exactly one move plays
it on one click, as before. Mancala steps and token placements stay one click either
way. Possible extras: double-click to play a menu's first choice; keyboard shortcuts
(Enter = Complete Turn already exists; R = Restart Turn).

## AI skill levels (added 2026-10-18)

Offer "Level 1-5" opponents instead of model names (raw model/sims stay available).
Each level is a preset of three knobs, fixed for the whole game so a bot doesn't swing
between strong and weak:
- **Model:** the heuristic for the lowest level, then older D checkpoints, then the best
  model. Old models make natural-looking mistakes.
- **Simulations:** fewer = weaker (already per seat, e.g. `D039@50`); strength flattens
  at very low counts, where the bot just follows its network's first instinct.
- **Temperature:** sample moves in proportion to search visits instead of always
  playing the most-visited (0 = greedy, as now). Limit sampling to moves within a margin
  of the best so a weak level never makes absurd blunders (Stockfish "Skill Level" uses
  a similar capped-mistake idea).

Work: temperature sampling in `play/bots.py` NetworkBot; a levels table in the server
(and a level picker in both clients); calibrate the presets with `ab_eval` round-robins
so the steps are evenly spaced (an Elo-style ladder), then check by playing them.
Later options: adjust the level automatically after a few wins or losses; human-like
models trained on saved human games, per rating band (as Lichess's Maia), once there
are enough of them.

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
