# Mali-Ba game server protocol (version 2)

The game server (`open_spiel/python/games/mali_ba/play/server.py`) runs games of Mali-Ba.
Any seat can be a human or a bot. Humans connect from any front end (the pygame GUI, a
browser, a phone app) on any machine that can reach the server. Bots play on the server.

The C++ engine holds every rule. A front end needs none: it draws the board, shows the
legal moves the server sends, and sends back one of them. The server validates every
move against the engine, so a client with a bug cannot make an illegal move.

## Running the server

From the open_spiel tree, in the `mali_ba` conda env (needs `fastapi` and `uvicorn`):

```
PYTHONPATH=build/python:open_spiel/python/games:. \
    python -m mali_ba.play.server --host 0.0.0.0 --port 8765
```

| Option | Default | Meaning |
|---|---|---|
| `--host` | `127.0.0.1` | `0.0.0.0` accepts other machines (LAN or Tailscale) |
| `--port` | `8765` | |
| `--config_file` | `mali_ba/mali_ba.ini` | game rules (the same ini the training uses) |
| `--models_dir` | the mali_ba directory | where `mali_ba_agent_v*.weights.h5` files are found |
| `--max_sims` | 1600 | the most simulations a seat may ask for |
| `--record_dir` | `play/games_played` | finished games saved as JSON; `''` turns it off |
| `--gpu` | off | let TensorFlow use the GPU (the default leaves it to training) |
| `--rule KEY=VALUE` | `block_consecutive_income=false` | change one ini value for these games (repeatable) |

The server plays by the same ini as training, except for the values in `--rule`. By
default only one differs: `block_consecutive_income` (a training aid that forbids taking
income twice in a row) is off. The server writes the changed copy to
`play/games_played/play_mali_ba.ini`; edit the original, not the copy.

There is no login. Anyone who can reach the port can create and watch games, so expose it
only on networks you trust (the LAN, or Tailscale for remote players).

## Flow

1. `GET /api/models` lists the networks that can fill a seat.
2. `POST /api/games` creates a game. Bot seats start playing right away; play stops at a
   human seat until that human moves.
3. Each human claims a seat with `POST /api/games/{id}/join`, which returns a private
   seat token.
4. The human opens the WebSocket `/ws/{id}?token=TOKEN`. Without a token the connection
   watches the game.
5. On every change the server sends each connection a `state` message. When it is your
   seat's turn, the message's `view.legal_moves` is non-empty; send one with a `move`
   message.
6. A human's turn is several decisions. When the last one passes play on, the turn is
   done (`view.turn.complete`) but the game waits: send `complete_turn` to let the next
   player move. Until then, `restart_turn` puts the game back to the start of the turn,
   as often as you like.
7. When the game ends, `view.result` is set. A human whose move ends the game completes
   that turn first.

A token stays valid for the whole game, so a client that drops can reconnect with it.

## HTTP

All bodies are JSON. Errors return a 4xx status with `{"detail": "..."}`.

`GET /api/info` returns:
```json
{"protocol": 2, "default_sims": 300, "max_sims": 1600, "num_seats": 3,
 "config_file": "play_mali_ba.ini", "rules": {"block_consecutive_income": "false"}}
```

`GET /api/board` returns the board's fixed parts, which a front end needs once:
```json
{"grid_radius": 7, "valid_hexes": [[-5, 0, 5], ...],
 "cities": [{"id": 1, "name": "Agadez", "culture": "Tuareg", "location": [4, -4, 0],
             "common_good": "Iron work", "rare_good": "Silver cross"}, ...],
 "player_colors": ["Red", "Green", "Blue"],
 "meeple_colors": ["solid black", "clear black", ...],
 "post_types": {"1": "post", "2": "center"},
 "map_rect": [-11.214, -7.147, 21.824, 16.034],
 "map_url": "/api/map"}
```
`map_rect` places the background map (`GET /api/map`, a JPEG) relative to the hexes:
x, y, width and height, in hex radii from the centre of hex (0,0,0). Screen y points
down, and hexes are flat-topped, so hex (q,r,s) is centred at (1.5 q, √3 (r − s) / 2).
The rectangle reproduces the original visualizer's alignment.
In the board JSON, seats are numbers (0 = the first of `player_colors`), meeples are
indexes into `meeple_colors`, and a trade post's `type` is a key of `post_types`.

`GET /api/models` returns models with the newest first:
```json
[{"id": "D040", "mover_relative": true}, {"id": "D039", "mover_relative": true}, ...]
```

`POST /api/games` takes one seat spec per seat, in seat order:
```json
{"seats": [{"type": "human", "name": "Rob"},
           {"type": "network", "model": "D039", "sims": 300},
           {"type": "heuristic"}],
 "seed": 7,
 "bot_delay": 0.0}
```
- `type` is `human`, `network` (needs `model`; `sims` defaults to 300) or `heuristic`.
- `seed` (optional) fixes the board layout and token draws.
- `bot_delay` sets the minimum seconds between bot moves (0 to 10), so people can follow
  an all-bot game.

It returns the game summary (see `GET /api/games`). Creating a game with a model that
isn't loaded yet takes a few seconds.

`GET /api/games` lists the games, newest first:
```json
[{"game_id": "URkJ6yb5", "created": 1792371000.1, "seed": 7,
  "seats": [{"seat": 0, "type": "human", "name": "Rob", "claimed": false}, ...],
  "current_seat": 0, "moves": 12, "over": false, "watchers": 1}]
```

`GET /api/games/{id}` returns `{"summary": ..., "view": ...}`. The view is the spectator
view, so it has no legal moves.

`POST /api/games/{id}/join` takes `{"seat": 0, "name": "Rob"}`. Both fields are
optional: without `seat`, it claims the first open human seat. It returns
`{"game_id": ..., "seat": 0, "token": "..."}`. A seat that is already taken returns 409,
and a bot seat returns 400.

## WebSocket `/ws/{id}?token=TOKEN`

Every message is a JSON object with a `type`.

Messages the client sends:

| Message | Meaning |
|---|---|
| `{"type": "move", "action": 812}` | play a move: its `action` id or its `name` (e.g. `"MancalaDir_2"`) from `legal_moves` |
| `{"type": "complete_turn"}` | your turn is done (`view.turn.complete`): pass play to the next player |
| `{"type": "restart_turn"}` | put the game back to the start of your turn (any time during it, also once it is done) |
| `{"type": "end_game"}` | stop the game for everyone, from any human seat (clients should ask "are you sure?" first); no winner, the game is saved and leaves the games list |
| `{"type": "view"}` | ask for the current state again |
| `{"type": "ping"}` | the server answers `{"type": "pong"}` |

Messages the server sends:

| Message | When |
|---|---|
| `{"type": "welcome", "protocol": 2, "game_id", "seat", "seq", "board_config", "view"}` | on connecting; `seat` is `null` for spectators; `board_config` is what `/api/board` returns |
| `{"type": "state", "seq", "last_move", "view"}` | after every move by anyone; after a seat is claimed, and after `complete_turn` or `restart_turn`, `last_move` is `null` and `event` names the command |
| `{"type": "error", "message"}` | a rejected move or a bad message; the game is unchanged |

A move is rejected when it isn't your turn, the game is over, the connection is a
spectator, the move isn't legal now, or your turn is done and waiting for
`complete_turn`. `complete_turn` is rejected while the turn isn't finished.

`seq` numbers the state messages of a game; the highest is the newest. A client with
several connections (one per seat at a hot-seat screen) should show the view with the
highest `seq`, not the longest history: a restarted turn makes the history shorter.

### The view

```json
{"board": {...},
 "phase": "MANCALA_STEP",
 "current_seat": 0,
 "seats": [{"seat": 0, "type": "human", "name": "Rob"}, ...],
 "legal_moves": [{"action": 812, "name": "MancalaDir_2", "kind": "mancala_step",
                  "direction": "southwest", "hex": [0, 1, -1],
                  "label": "Continue the mancala to (0,1,-1)"}, ...],
 "final_round": false,
 "history": [{"n": 41, "seat": 1, "action": 77, "name": "PlaceToken_(3,2,-5)",
              "kind": "place_token", "label": "Place a token on (3,2,-5)", "hex": [3, 2, -5],
              "t": 1792371002.3}, ...],
 "result": null,
 "goods": ["Casava/peanut", ...],
 "turn": {"seat": 0, "moves": 3, "can_restart": true, "complete": false}}
```

- `board` is the engine's own state JSON (`Mali_BaState::Serialize`). Its keys are
  `currentPhase`, `currentPlayerId`, `hexMeeples`, `playerTokens`, `tradePosts`,
  `tradeRoutes`, `commonGoods`, `rareGoods`, `playerPostsSupply`, `midTurnState` (the
  mancala in progress: `mancalaHex`, `mancalaPath`, `meeplesInHand`, `lastActionHex`),
  `history` and `version`. Hexes are cube coordinates `x,y,z` with x+y+z = 0.
- `legal_moves` is non-empty only for the seat to move, on its own connection.
- `history` holds the last 200 decisions. A restarted turn's decisions are removed from it.
- `turn` is set during a human's turn (otherwise `null`): `moves` made so far this turn,
  `can_restart` (at least one move to undo) and `complete` (the turn is done and waits
  for `complete_turn`; `legal_moves` is then empty and `current_seat` is still this
  seat). Bots' turns pass on without waiting. Each token placement is a turn of its own,
  including the two placements in a row at a serpentine bend.
- `result` is set when the game ends:
  `{"winner": 0 | null, "reason", "returns", "score", "moves"}`. `winner` is `null` when
  the game reached the move cap. `score` is the engine's score breakdown text. A game
  stopped with `end_game` has `ended_by` (the seat), no winner, `returns` `null`, and
  the score so far.

### Moves

The engine splits a turn into single decisions. Each legal move carries a `kind`:

| kind | Engine action | Extra fields |
|---|---|---|
| `place_token` | `PlaceToken_(x,y,z)` (set-up) | `hex` |
| `start_mancala` | `StartMancala_(x,y,z)` | `hex` (the token that moves) |
| `mancala_step` | `MancalaDir_k` | `hex` (where the step goes), `direction` |
| `place_post` | `PlacePost` (at the mancala's end hex) | |
| `pay_good` | `PayGood_k` | `good` |
| `upgrade` | `UpgradePost_(x,y,z)` | `hex` |
| `declare_route` | `DeclareRoute_k` | `route_index`, `hexes` (the route, in order), `cities` (the cities on it) |
| `take_income` | `TakeIncome` | |
| `pass` | `Pass` (decline a post or route, or end a mancala) | |

A front end can highlight `hex` on the board and show `label` as text.

A route choice can offer well over 100 options late in a game, so a list alone is hard to
use. Each option's `hexes` come from the engine: the server applies the option to a copy
of the state and reads the new route. A front end can draw an option on the board, or
narrow the list to the routes through hexes the player clicks, as the pygame client does.
History records keep `hex`, `hexes` and `good` when the move had them.

## Bots

- **`heuristic`** is the engine's heuristic player. It moves instantly.
- **`network`** uses MCTS guided by a trained network, searched the way `ab_eval`
  evaluates models: PUCT, uct_c 0.5, no Dirichlet noise, heuristic guidance 0, and it
  plays the most-visited move. Models from D029 on search mover-relative. Token
  placement uses the heuristic, because self-play places tokens at random and the
  networks never learned it.

At 300 simulations on the CPU, a decision takes about 0.8 s on average and 2 s at most
(measured while training was running). Bots search one at a time across all games.

## Front ends

**Web** (`play/web/index.html`) is served by the game server itself: open
`http://HOST:8765/` in any browser, including on a phone. The lobby creates games (any
mix of humans, networks and the heuristic) and lists the server's games with Join, Watch
and Resume buttons. Seat tokens are kept in the browser, so reloading the page rejoins
the game. `http://HOST:8765/#g=GAME_ID` is an invite link: it opens the game and offers
the open seats. To play, click a highlighted hex or a move button. The wheel zooms;
right-drag, shift-drag or a touch drag pans. Hovering a move shows it on the board.
Restart Turn and Complete Turn sit under the prompt; Enter also completes a turn.
"End game…" at the bottom of the sidebar stops the game after a confirmation.

**Pygame** (`play/pygame_client.py`) needs only pygame, websockets and the mali_ba package
(`PYTHONPATH=open_spiel/python/games`); it does not need the engine build.
```
python -m mali_ba.play.pygame_client --new human,D039,heuristic           # you vs two bots
python -m mali_ba.play.pygame_client --new human,human,D039@100 --hotseat  # two humans at one screen
python -m mali_ba.play.pygame_client --join GAME_ID --name Alice           # take an open seat
python -m mali_ba.play.pygame_client --watch GAME_ID
python -m mali_ba.play.pygame_client --new D040,D039,heuristic --bot_delay 1
```
Add `--server http://HOST:8765` for a server on another machine. To play, click a
highlighted hex or a move button. The wheel zooms the board and scrolls the move list;
right-drag pans. Restart Turn and Complete Turn (or Enter) sit in the sidebar; "End
game..." in its bottom-right corner stops the game after a confirmation. F fits
the board, R toggles trade routes, H hides the final score.

## Versioning

`protocol` (in `/api/info` and `welcome`) changes when a message changes incompatibly.
New optional fields can be added without a version change, so clients should ignore
fields they don't know.
