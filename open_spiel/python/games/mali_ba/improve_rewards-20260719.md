# Improving the Reward Structure (2026-07-19)

I now have the full picture — rules, ini, and the actual `Returns()`/`Rewards()` implementations in `mali_ba_state_core.cc:1580`. Here's what the model currently receives, and where I think the structure works against closing games out.

## What the model gets today

| Signal | Value | When |
|---|---|---|
| Win (highest score) | +1.0 | terminal |
| Loss | −0.1 | terminal |
| Timeout penalty | −0.7 to **all** players | terminal, on top of the above |
| Rare good from new region | **+0.30 each** | during play |
| Trade route created | +0.05 | during play |
| Upgrade to center | +0.02 | during play |
| Key-location post | +0.03 | during play |
| Time penalty | −0.0005/move, escalating ×1→×7 | during play |

Two structural problems jump out.

**1. Approaching the win pays about as much as winning.** `new_rare_region_reward = 0.30` can pay out up to 6 times — a player who farms rare goods from 5–6 regions banks **+1.5 to +1.8 in cumulative shaping without ever ending the game**, more than the +1.0 win itself. The near-win state (rare goods from 4+ regions) is exactly the state this reward drives toward, and the marginal payoff for converting it into an actual win is small relative to what was already collected on the way. The value head learns "trajectories that approach the win are high-value" — which is literally what the data shows: high value estimates at move 430, 0% conversion.

**2. Being ahead at timeout is still profitable.** At timeout the code scores normally, gives the point leader +1.0, then applies −0.7 to everyone — so the stalling leader still nets **+0.3**. The model is never taught "a win you don't close isn't a win." Meanwhile a trailing player who *could* trigger the end gets −0.1 for closing — and here the point about the ender not always being the winner matters a lot: in game 125, if neither route-builder was the point leader, refusing to complete the route was arguably *correct* play under these returns. Stalling is a rational equilibrium of the current reward structure, not necessarily a search failure.

## Recommended changes, in priority order

1. **Make timeout strictly unprofitable for everyone, including the leader** (one-line ini change): raise `max_moves_penalty` from −0.7 to **−1.0 or −1.2**. The point leader at timeout then nets 0.0 to −0.2 instead of +0.3. This is the single most direct encoding of "close it out or it doesn't count." The ordering that matters stays intact (losing-by-closing −0.1 still beats losing-at-timeout ≈ −1.1, so trailing players still prefer to close rather than stall).

2. **Shrink the rare-region shaping and move the incentive to the terminal.** Cut `new_rare_region_reward` from 0.30 to **~0.05**, and turn on the already-implemented `rare_goods_bonus` (currently 0.0) at **+0.2 or +0.3**. That bonus is conditional on actually *winning* by rare goods — it's the right shape for the same intent: bias toward the rare-goods win without paying for approach-and-stall. The shaping reward was scaffolding for exploration; at 2000+ MCTS games it's now being farmed.

3. **Reward the closer, conditional on winning.** The C++ already tracks `GetGameEndTriggeringPlayer()`. Add a small terminal bonus (+0.1–0.15) to the game-ender *if they also win*. This sharpens the leader's incentive to pull the trigger rather than coast, without creating a kingmaker incentive for losing players to end games spitefully.

4. **Widen the loss differential, or make returns margin-aware.** −0.1 for losing is nearly indistinguishable from a draw; losers barely feel it, and 2nd vs 3rd place is identical, so a player who can't win has no gradient at all. Options, mildest first: `loss_penalty = -0.5`; rank-based returns (+1.0 / −0.3 / −0.7); or margin-based returns (normalized score differences, e.g. `(score_p − mean) / spread` scaled into ±1 with a win bonus on top). Margin-based is the most interesting for this specific quirk — because the winner is determined by points, a value head trained only on discrete outcomes has to infer point-standing indirectly, but the decision "should I close now?" *depends entirely on current point standing*. Margin returns give it that signal directly.

5. **Related but not reward structure:** check whether the observation tensor includes current score (or its components). The unique-goods set scoring is complex enough (repeated exponential sets) that asking the network to re-derive point standing from raw board state is a big ask — and per #4, closing decisions hinge on it. If scores aren't in the observation, adding them may matter as much as any reward change.

## Cautions

- **Change one thing per run.** Take #1 + #2 together as a single "fix the mis-specification" step (they're two halves of the same problem), then evaluate before touching #3/#4.
- **Every threshold in the training pipeline is denominated in current return units** — `clear_winner_thresh = 0.30`, the hopeless thresholds, `near_win_extension_value_thresh`. Rescaling returns (especially #4) means re-tuning those, or the early-termination logic will start culling the wrong games.
- **Reward changes fix incentives, not execution.** The search-side bottleneck found earlier (the winning route declaration is only reachable from the right `last_action_hex_`, chosen among up to 300 candidates at 300 sims) is untouched by any of this. If incentives improve and conversion stays at ~0%, that's confirmation the bottleneck is search, not reward.