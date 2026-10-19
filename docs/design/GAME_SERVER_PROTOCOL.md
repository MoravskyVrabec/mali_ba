# Mali-Ba game server protocol (version 1)

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
6. When the game ends, `view.result` is set.

A token stays valid for the whole game, so a client that drops can reconnect with it.

## HTTP

All bodies are JSON. Errors return a 4xx status with `{"detail": "..."}`.

`GET /api/info` returns:
```json
{"protocol": 1, "default_sims": 300, "max_sims": 1600, "num_seats": 3, "config_file": "mali_ba.ini"}
```

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
| `{"type": "view"}` | ask for the current state again |
| `{"type": "ping"}` | the server answers `{"type": "pong"}` |

Messages the server sends:

| Message | When |
|---|---|
| `{"type": "welcome", "protocol": 1, "game_id", "seat", "view"}` | on connecting; `seat` is `null` for spectators |
| `{"type": "state", "last_move", "view"}` | after every move by anyone, and after a seat is claimed (`last_move` is then `null`) |
| `{"type": "error", "message"}` | a rejected move or a bad message; the game is unchanged |

A move is rejected when it isn't your turn, the game is over, the connection is a
spectator, or the move isn't legal now.

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
 "goods": ["Casava/peanut", ...]}
```

- `board` is the engine's own state JSON (`Mali_BaState::Serialize`). Its keys are
  `currentPhase`, `currentPlayerId`, `hexMeeples`, `playerTokens`, `tradePosts`,
  `tradeRoutes`, `commonGoods`, `rareGoods`, `playerPostsSupply`, `midTurnState` (the
  mancala in progress: `mancalaHex`, `mancalaPath`, `meeplesInHand`, `lastActionHex`),
  `history` and `version`. Hexes are cube coordinates `x,y,z` with x+y+z = 0.
- `legal_moves` is non-empty only for the seat to move, on its own connection.
- `history` holds the last 200 decisions.
- `result` is set when the game ends:
  `{"winner": 0 | null, "reason", "returns", "score", "moves"}`. `winner` is `null` when
  the game reached the move cap. `score` is the engine's score breakdown text.

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
| `declare_route` | `DeclareRoute_k` | `route_index`; `hexes` is `null` for now |
| `take_income` | `TakeIncome` | |
| `pass` | `Pass` (decline a post or route, or end a mancala) | |

A front end can highlight `hex` on the board and show `label` as text. A planned engine
binding will add each route option's hexes, so `declare_route` can be drawn on the board
instead of shown as "option k".

## Bots

- **`heuristic`** is the engine's heuristic player. It moves instantly.
- **`network`** uses MCTS guided by a trained network, searched the way `ab_eval`
  evaluates models: PUCT, uct_c 0.5, no Dirichlet noise, heuristic guidance 0, and it
  plays the most-visited move. Models from D029 on search mover-relative. Token
  placement uses the heuristic, because self-play places tokens at random and the
  networks never learned it.

At 300 simulations on the CPU, a decision takes about 0.8 s on average and 2 s at most
(measured while training was running). Bots search one at a time across all games.

## Versioning

`protocol` (in `/api/info` and `welcome`) changes when a message changes incompatibly.
New optional fields can be added without a version change, so clients should ignore
fields they don't know.
