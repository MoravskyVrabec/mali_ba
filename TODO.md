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
