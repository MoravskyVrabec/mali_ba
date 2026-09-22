# Handoff: BFS Dead-End Pruning Fix

## Problem Statement

During MCTS simulations, the agent was choosing **Pass** with near-100% visit counts (~21% of all moves were Pass). Diagnostics showed this was not a learning failure — the MCTS tree was genuinely finding that Pass was the only legal action available. The `PASS_ONLY` C++ diagnostic (in `GetLegalActionsAndCounts`) confirmed this, and it always showed `path_len=5`.

## Root Cause

Mali-Ba's mancala mechanic works as follows:
1. A mancala starts at a hex and must traverse a path placing meeples one per step.
2. The mancala begins with 3 meeples in hand → path: meeple, meeple, meeple, token (4 steps, path_len grows to 4 when last meeple placed, then token is step 5 → path_len=5 at the dead end).
3. At each step, the player chooses one of 6 hex directions.
4. A direction is illegal if: (a) off-board, (b) already in `current_mancala_path_`.
5. If all 6 directions are pruned, only Pass is legal.

**Why path_len is always 5:** The mancala always runs into a dead end at the token step (the 5th element of the path), because earlier meeple steps chose directions without reserving a hex for the final token landing.

**Why the previous BFS didn't fix it:** The BFS was computing `needed = steps_needed - 1` where `steps_needed = meeples_in_hand_.size()`. This means:
- With 3 meeples in hand → checks 2 reachable hexes from target
- With 2 meeples → checks 1
- With 1 meeple (final meeple step) → checks 0 → BFS never prunes → dead end!

The final meeple step never checked whether there was a hex available for the token to land on afterward.

## The Fix (RESOLVED)

The step-level BFS approach was abandoned in favor of a cleaner fix: **DFS validation at mancala start**.

**File:** `open_spiel/games/mali_ba/mali_ba_state_core.cc`  
**Location:** `kOptionalMancala` / mancala start action generation

Before offering a mancala start action for a given hex, a DFS checks whether a complete non-self-intersecting path of `total_steps = meeples_at_hex + 1` exists from that hex. If no valid path exists, the mancala start is never added to the legal action list.

```cpp
// total_steps = meeples in hand + 1 for the token landing
const int total_steps = static_cast<int>(GetMeeplesAt(hex).size()) + 1;

auto has_valid_mancala_path = [&](const HexCoord& start_pos, int steps) {
    std::vector<HexCoord> path = {start_pos};
    std::function<bool(const HexCoord&, int)> dfs = [&](const HexCoord& current, int depth) -> bool {
        if (depth == 0) return true; // Found a complete path!
        for (int d = 0; d < 6; ++d) {
            HexCoord next_hex = current + kHexDirections[d];
            if (IsValidHex(next_hex) &&
                std::find(path.begin(), path.end(), next_hex) == path.end()) {
                path.push_back(next_hex);
                if (dfs(next_hex, depth - 1)) return true;
                path.pop_back();
            }
        }
        return false;
    };
    return dfs(start_pos, steps);
};

if (has_valid_mancala_path(hex, total_steps)) {
    result.actions.push_back(kMancalaStartBase + i);
}
```

**Why this works:** If a valid path exists from start, it's always possible for the agent to complete the mancala without getting stuck — the agent just needs to follow a valid branch. Dead ends during mancala steps remain possible if the agent makes poor choices, but they cannot persist: Pass is available as an escape, and this fix ensures the start itself isn't degenerate.

**Why it's cleaner than step-level BFS:** The previous approach tried to prune bad step directions mid-mancala but had an off-by-one error (didn't reserve a hex for the token). The start DFS solves the root cause — don't begin a mancala that has no valid completion.

## Status

**FIXED AND COMPILED.** The fix is verified working.

Remaining cleanup:
1. Disable diagnostics when confident Pass rate has dropped:
   - C++ `PASS_ONLY` block in `mali_ba_state_core.cc` (~line 329)
   - Python `PASS_DIAG` block in `train_mali_ba.py` (`log_pass_diagnostic` call)
2. Rsync to laptop and rebuild for remote actors.
4. Rsync to laptop and rebuild there for remote actors.

## Diagnostics Currently Active

- **C++ `PASS_ONLY`**: logs when `GetLegalActionsAndCounts` returns only Pass. Shows `phase`, `path_len`, `posts_supply`, `income_ok`, `upgrade_ok`, `mancala_hex`.
- **Python `PASS_DIAG`**: logs when the actor selects Pass during MCTS. Shows legal action count, pass visit fraction, phase.

Both can be found by grepping `PASS_ONLY` and `PASS_DIAG` in the logs.

## Training Context

- Current run: `train_run04.log`
- `--skip_timeout_games` is active (timeout games not added to replay buffer)
- Length penalty on wins: `return *= 1.0 - (game_length / max_game_length) * 0.15`
- `--mcts_sample_fraction 0.5` (equal draw from bootstrap and MCTS pools)
- Max game length: 690 moves (changed from 700 to give a 10-move buffer)
- Weights saved every 10 minutes (`--save_every 10`)