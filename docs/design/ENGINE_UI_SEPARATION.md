# Design: separating the game engine from the presentation layer

Status: **proposal, for review** (2026-10-08). Nothing here is implemented yet.

## Goal

Play Mali-Ba as humans against bots (any mix of seats) in two front ends that share
everything except drawing and input:

- the **local GUI** (pygame) that exists today, and
- a **web front end** (browser) to be added later.

Both must give the human move-by-move feedback on what is legal (which hexes a
mancala may continue to, whether a trading post can be placed, which goods can pay
for an upgrade, ...), and both must be able to play against the **trained network**,
not only the hand-written heuristic. Neither front end may contain game rules.

## Current structure (as of 2026-10-08)

Code is under `open_spiel/python/games/mali_ba/` unless noted.

| Piece | Where | Role |
|---|---|---|
| Engine | C++ `open_spiel/games/mali_ba/` via `pyspiel` | rules, legal moves, state; serialises the full state as JSON |
| Engine wrapper | `utils/cpp_interface.py` (`GameInterface`) | load game, apply a move string, return state JSON, play a heuristic move |
| GUI | `main.py`, `ui/visualizer*.py`, `ui/gui_other.py` | pygame drawing and input; rebuilds its picture from the JSON (`GameStateCache`) |
| Network bot | `utils/configurable_ai_bot.py`, `utils/simple_ai_integration.py` | MCTS + trained network; patched into `GameInterface` |
| Setup | `main.py`, `utils/player_config.py`, `utils/board_config.py` | players, board, config |

### What is already right

- **The engine is the single source of truth.** The GUI never changes game state
  itself: it sends a move string to the engine and rebuilds everything from the
  engine's JSON (`serialize()`; keys `currentPhase`, `currentPlayerId`, `hexMeeples`,
  `tradePosts`, `tradeRoutes`, `playerTokens`, `commonGoods`, `rareGoods`,
  `playerPostsSupply`, `midTurnState`, `history`, `version`). This boundary carries
  over to the web unchanged: a browser can send the same moves and receive the same
  JSON.

### Problems

1. **The GUI's move format no longer matches the engine (blocking).** The GUI
   composes a whole turn as one string, e.g. `place (x,y,z)` or
   `mancala (a):(b):(c) post` (`ui/visualizer.py`, `submit_move`). The engine now
   models a turn as a series of single decisions (`PlaceToken_(x,y,z)`,
   `StartMancala_(hex)`, `MancalaDir_k` per step, `PlacePost`, `UpgradePost_(hex)`,
   `PayGood_k`, `DeclareRoute_k`, `TakeIncome`, `Pass`), and its string parser only
   matches those names. Verified: `PlaceToken_(-5,0,5)` is accepted, `place (-5,0,5)`
   is rejected ("Couldn't find an action matching"). Human moves in the local GUI are
   therefore very likely broken today.
2. **Rules are duplicated in the GUI.** `ui/visualizer_other.py` has simplified copies
   of engine rules (`can_start_mancala_at`, `is_valid_mancala_step`,
   `can_select_for_upgrade`, `can_add_to_trade_route`) to decide what a click may do.
   They can disagree with the engine, and a web client would need them rewritten.
3. **Bot play is tangled with the presentation layer, along two inconsistent paths.**
   - Path A (GUI): a pygame timer calls `play_heuristic_move`, i.e. the C++
     heuristic, for every "ai" or "heuristic" seat.
   - Path B (`add_ai_support`): replaces `GameInterface.apply_action` with a wrapper
     that plays one MCTS + network move right after a *human's* move. It cannot make
     the first move or a second bot move in a row (those fall back to path A), uses
     its own settings (heuristic guidance 0.40, not training's 0.30) and a hard-coded
     OpenSpiel path.
   Which bot plays a move depends on turn order, and none of it is usable from a server.
4. **Move pruning is game-wide.** `prune_moves_for_ai` (engine parameter) makes an
   upgrade offer only the single best trade route, as in training; off ("GUI mode")
   it offers every route. Humans need every option, while the network was trained
   with pruning on. In a mixed game it must apply per decision.
5. **Some choices cannot be shown from their names.** `DeclareRoute_3` and
   `PayGood_2` say nothing about which hexes or which good they mean.
6. **The engine wrapper sits among GUI utilities** (`utils/`), and setup logic is
   spread over three files.

## Target structure

```
Engine (C++ via pyspiel)                rules, legal moves, state JSON
        │
Game service  (pure Python, no pygame, no web code)
   GameSession   one game: seats, state, legal moves, move descriptions, apply, undo
   Bots          one interface: heuristic, network + MCTS, placement heuristic
        │
   ┌────┴─────────────────────────────┐
Local GUI (pygame)                 Web server (later)
calls GameSession in-process       wraps GameSession behind HTTP / WebSocket;
                                   a browser client renders the same JSON
```

Both front ends: render the state JSON, highlight what `legal_moves()` returns, send
one decision at a time, and get bot moves from the session. No rules in either.

### GameSession (sketch)

```python
class GameSession:
    def __init__(self, config_file, seats):  # seats: ["human", "bot:network:D013", "bot:heuristic"]
    def state(self) -> dict                   # the engine's state JSON, parsed
    def current_seat(self) -> int             # -1 when terminal
    def is_human_turn(self) -> bool
    def legal_moves(self) -> list[dict]       # one entry per legal decision, e.g.
        # {"action": 812, "name": "MancalaDir_2", "kind": "mancala_step",
        #  "hex": (1,-1,0), "label": "Continue the mancala to (1,-1,0)"}
        # {"action": 4021, "name": "DeclareRoute_3", "kind": "declare_route",
        #  "hexes": [(0,0,0), (1,-1,0), ...], "label": "Route of 5 hexes ..."}
    def apply(self, action_or_name) -> dict   # validates against legal_moves(); returns new state
    def bot_move(self) -> dict                # plays the current bot seat's move; returns new state
    def undo(self) -> dict                    # optional, for the GUI
    def result(self) -> dict | None           # winner, end condition, scores when terminal
```

Notes:
- `legal_moves()` is the feedback mechanism. Because the engine splits a turn into
  single decisions, at each step it is exactly the set of hexes / buttons the human
  may use next: valid starting hexes for a mancala, then each allowed next hex, then
  whether `PlacePost` is available, then which goods can pay, and so on.
- `kind` and `hex` / `hexes` / `good` let a front end map an action to something it
  can draw, without parsing names or knowing rules.
- Bot moves can be slow (a network search takes seconds), so the GUI calls
  `bot_move()` from a worker thread and redraws when it returns; the web server does
  the same per session.

### Bots (one interface)

```python
class Bot:
    def choose(self, session_state) -> int    # an action id
```
- **HeuristicBot**: the C++ heuristic (`select_heuristic_random_action`).
- **NetworkBot**: MCTS with the trained network, reusing the training code's
  `AlphaZeroEvaluator` and the same settings as `ab_eval.py` (heuristic guidance 0.30,
  `uct_c` 2.0, a simulation budget chosen per difficulty), greedy move choice.
  Loads `mali_ba_agent_vD0xx.weights.h5`. Inference on CPU by default; a server with
  a GPU can share one inference server across games.
- **Placement**: during token placement, bots use the data-fitted placement heuristic
  (`train_mali_ba._choose_placement`, moved to a shared `placement.py`) at a low
  temperature (e.g. 0.5) so they place well, instead of the C++ placement rule.
- Difficulty levels map to bot type and simulation budget.

### Engine (C++) changes needed

1. **Per-decision pruning.** Replace the game-wide `prune_moves_for_ai` with a
   per-state setting (e.g. `state.set_prune_moves(bool)`), so a human's decision gets
   the full move list and a bot's gets the pruned list it was trained on.
2. **Describe an action.** A binding such as `state.describe_action(action)` returning
   structured detail: the hex for placement / mancala / upgrade actions, the hex list
   for `DeclareRoute_k`, the good for `PayGood_k`. (Mancala direction to hex can also
   be computed in Python from the current position; doing it in the engine keeps it
   in one place.)
3. **Optional: explain why something is illegal** (e.g. "need 3 common goods to
   upgrade here"), for richer messages than highlighting alone.
4. **Optional: expose `ComputeScores()`** to show live scores (the observation already
   carries them, but a direct call is clearer).

## Migration steps

Each step leaves something working and testable on its own.

1. **GameSession + bots, with tests, no UI.** New package (e.g. `mali_ba/engine/` or
   `mali_ba/service/`) holding `GameSession`, the bot classes and `placement.py`.
   Engine changes 1 and 2. Test by playing complete games headless: humans simulated
   by picking from `legal_moves()`, bots of each type, mixed seats, pruning per seat,
   descriptions for every action kind.
2. **Point the local GUI at GameSession.** Input becomes step-by-step: highlight
   `legal_moves()` hexes, submit one decision per click; bot turns via `bot_move()` in
   a thread. Delete the GUI's rule copies (`visualizer_other.py` checks) and both
   old bot paths (`play_heuristic_move` timer, `add_ai_support` patching). This also
   fixes problem 1.
3. **Web server.** A thin HTTP/WebSocket layer over GameSession (one session per
   game; messages are the state JSON, `legal_moves()` and moves). A browser client
   renders the board from the JSON. Framework to decide (FastAPI with WebSockets is a
   natural fit).
4. **Clean-up.** Move setup out of `main.py`; retire `GameInterface` once nothing uses
   it; keep `config.py` constants shared.

## Open questions

- Which web framework, and is the web version single-machine (local network) or
  internet-facing (then authentication and hosting matter)?
- Bot strength for humans: which checkpoint, how many simulations, and whether to
  offer difficulty levels.
- Undo for humans: allow it in local games only?
- Should bots' move choice keep a little randomness for variety, as in training, or
  always play their best?
- Replays: the GUI has a replay mode (`SimpleReplayManager`); should GameSession own
  move history and replay for both front ends?
