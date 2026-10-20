"""Mali-Ba game server: humans join from any front end, bots play on the server.

Run (from the open_spiel tree, mali_ba env):
    PYTHONPATH=build/python:open_spiel/python/games:. \\
        python -m mali_ba.play.server --host 0.0.0.0 --port 8765

Protocol: docs/design/GAME_SERVER_PROTOCOL.md (version PROTOCOL below). In short:
  GET  /api/info                      protocol version, server settings
  GET  /api/board                     the fixed board: hexes, cities, colour names
  GET  /api/models                    network bots that can fill a seat
  POST /api/games                     create a game: {"seats": [...], "seed", "bot_delay"}
  GET  /api/games                     games and their open human seats
  GET  /api/games/{id}                one game's view (as a spectator)
  POST /api/games/{id}/join           claim a human seat -> {"token"}
  WS   /ws/{id}?token=...             play that seat (no token: watch)

Mali-Ba has no hidden information, so every connection sees the whole board; the seat
token only decides who may move. Bots search one at a time on a worker thread, so a
long search never blocks the server.
"""
import argparse
import asyncio
import concurrent.futures
import glob
import json
import os
import re
import secrets
import time

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mali_ba.play.bots import DEFAULT_SIMS, model_name
from mali_ba.play.session import GameSession, IllegalMove, board_config

PROTOCOL = 2      # 2: human turns end with complete_turn; restart_turn
HERE = os.path.dirname(os.path.abspath(__file__))
MALI_BA_DIR = os.path.dirname(HERE)
WEB_DIR = os.path.join(HERE, "web")

# Set by main(); module-level so the app can be imported by uvicorn for testing.
SETTINGS = {
    "config_file": os.path.join(MALI_BA_DIR, "mali_ba.ini"),
    "models_dir": MALI_BA_DIR,
    "max_sims": 1600,
    "record_dir": None,
}
_bot_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="bot")
_game_cache = {}


def _game():
    import pyspiel
    key = SETTINGS["config_file"]
    if key not in _game_cache:
        _game_cache[key] = pyspiel.load_game("mali_ba", {"config_file": key})
    return _game_cache[key]


# ---- models ----------------------------------------------------------------------

def list_models():
    """Network models in models_dir: [{"id", "file", "mover_relative"}], newest D first.
    A model is usable when both its _policy and _value weights exist."""
    out = []
    for pol in glob.glob(os.path.join(SETTINGS["models_dir"], "mali_ba_agent_v*._policy.weights.h5")):
        spec = pol.replace("_policy.weights.h5", "weights.h5")
        if not os.path.exists(spec.replace("weights.h5", "_value.weights.h5")):
            continue
        mid = model_name(spec)
        m = re.match(r"D(\d+)", mid)
        # Networks from D029 on were trained mover-relative.
        out.append({"id": mid, "file": os.path.basename(spec),
                    "mover_relative": bool(m and int(m.group(1)) >= 29),
                    "mtime": os.path.getmtime(pol)})
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def _model_spec(model_id):
    for m in list_models():
        if m["id"] == model_id:
            return os.path.join(SETTINGS["models_dir"], m["file"]), m["mover_relative"]
    raise HTTPException(400, f"unknown model {model_id!r}; see /api/models")


# ---- games -----------------------------------------------------------------------

class Game:
    def __init__(self, gid, session, bot_delay):
        self.id = gid
        self.session = session
        self.bot_delay = bot_delay
        self.lock = asyncio.Lock()
        self.tokens = {}            # token -> seat
        self.claimed = {}           # seat -> player name
        self.conns = {}             # websocket -> seat (None = spectator)
        self.bot_task = None
        self.created = time.time()
        self.recorded = False
        self.seq = 0                # numbers the state messages: newest = highest

    def summary(self):
        s = self.session
        return {"game_id": self.id, "created": self.created, "seed": s.seed,
                "seats": [dict(seat, claimed=(seat["type"] != "human" or seat["seat"] in self.claimed))
                          for seat in s.seats],
                "current_seat": s.current_seat(), "moves": len(s.history),
                "over": s.is_over(), "watchers": len(self.conns)}


GAMES = {}


class SeatSpec(BaseModel):
    type: str                       # "human" | "heuristic" | "network"
    name: str | None = None         # humans: a display name (optional)
    model: str | None = None        # network: a model id from /api/models, e.g. "D039"
    sims: int | None = None         # network: simulations per decision (default 300)


class CreateGame(BaseModel):
    seats: list[SeatSpec]
    seed: int | None = None
    bot_delay: float = 0.0          # seconds between bot moves (to watch bots play)


class Join(BaseModel):
    seat: int | None = None         # None: the first open human seat
    name: str | None = None


app = FastAPI(title="Mali-Ba game server")


@app.get("/api/info")
def info():
    return {"protocol": PROTOCOL, "default_sims": DEFAULT_SIMS, "max_sims": SETTINGS["max_sims"],
            "num_seats": _game().num_players(), "config_file": os.path.basename(SETTINGS["config_file"]),
            "rules": SETTINGS.get("rules", {})}


MAP_FILE = os.path.join(MALI_BA_DIR, "ui", "mali_ba_map.jpg")


def _board_config():
    cfg = board_config(_game(), SETTINGS["config_file"])
    cfg["map_url"] = "/api/map" if os.path.exists(MAP_FILE) else None
    return cfg


@app.get("/api/board")
def board():
    return _board_config()


@app.get("/api/map")
def board_map():
    return FileResponse(MAP_FILE)


@app.get("/api/models")
def models():
    return [{k: m[k] for k in ("id", "mover_relative")} for m in list_models()]


@app.post("/api/games")
async def create_game(req: CreateGame):
    seats = []
    for sp in req.seats:
        if sp.type == "human":
            seats.append({"type": "human", "name": sp.name})
        elif sp.type == "heuristic":
            seats.append({"type": "heuristic"})
        elif sp.type == "network":
            if not sp.model:
                raise HTTPException(400, "a network seat needs a model")
            path, rel = _model_spec(sp.model)
            sims = DEFAULT_SIMS if sp.sims is None else sp.sims
            if not 1 <= sims <= SETTINGS["max_sims"]:
                raise HTTPException(400, f"sims must be 1..{SETTINGS['max_sims']}")
            seats.append({"type": "network", "weights": path, "sims": sims, "mover_relative": rel})
        else:
            raise HTTPException(400, f"unknown seat type {sp.type!r}")
    loop = asyncio.get_running_loop()
    try:
        # Loading a network takes a few seconds the first time; keep the loop free.
        session = await loop.run_in_executor(
            _bot_pool, lambda: GameSession(SETTINGS["config_file"], seats, seed=req.seed, game=_game()))
    except ValueError as e:
        raise HTTPException(400, str(e))
    gid = secrets.token_urlsafe(6)
    g = Game(gid, session, max(0.0, min(req.bot_delay, 10.0)))
    GAMES[gid] = g
    _kick_bots(g)
    return g.summary()


@app.get("/api/games")
def games():
    return [g.summary() for g in sorted(GAMES.values(), key=lambda g: -g.created)]


@app.get("/api/games/{gid}")
def game_view(gid: str):
    g = _get(gid)
    return {"summary": g.summary(), "view": g.session.view(for_seat=-1)}


@app.post("/api/games/{gid}/join")
async def join(gid: str, req: Join):
    g = _get(gid)
    humans = [s["seat"] for s in g.session.seats if s["type"] == "human"]
    seat = req.seat
    if seat is None:
        open_seats = [s for s in humans if s not in g.claimed]
        if not open_seats:
            raise HTTPException(409, "no open human seat")
        seat = open_seats[0]
    if seat not in humans:
        raise HTTPException(400, f"seat {seat} is not a human seat")
    if seat in g.claimed:
        raise HTTPException(409, f"seat {seat} is already taken")
    token = secrets.token_urlsafe(16)
    g.tokens[token] = seat
    g.claimed[seat] = req.name or g.session.seats[seat]["name"]
    g.session.seats[seat]["name"] = g.claimed[seat]
    await _broadcast(g, None)
    return {"game_id": gid, "seat": seat, "token": token}


def _get(gid):
    g = GAMES.get(gid)
    if g is None:
        raise HTTPException(404, "no such game")
    return g


# ---- live play -------------------------------------------------------------------

async def _send(ws, msg):
    try:
        await ws.send_text(json.dumps(msg))
    except Exception:
        pass


async def _broadcast(g, last_move, event=None):
    g.seq += 1
    for ws, seat in list(g.conns.items()):
        msg = {"type": "state", "seq": g.seq, "last_move": last_move,
               "view": g.session.view(for_seat=-1 if seat is None else seat)}
        if event:
            msg["event"] = event        # "complete_turn" or "restart_turn"
        await _send(ws, msg)
    if g.session.is_over() and not g.recorded:
        g.recorded = True
        _record(g)


def _record(g):
    if not SETTINGS["record_dir"]:
        return
    os.makedirs(SETTINGS["record_dir"], exist_ok=True)
    s = g.session
    path = os.path.join(SETTINGS["record_dir"], f"game_{time.strftime('%Y%m%d_%H%M%S')}_{g.id}.json")
    with open(path, "w") as f:
        json.dump({"game_id": g.id, "seed": s.seed, "seats": s.seats, "config_file": s.config_file,
                   "history": s.history, "result": s.result()}, f, indent=1)


def _kick_bots(g):
    if g.bot_task is None or g.bot_task.done():
        g.bot_task = asyncio.create_task(_drive_bots(g))


async def _drive_bots(g):
    loop = asyncio.get_running_loop()
    while True:
        async with g.lock:
            if not g.session.is_bot_turn():
                return
            t0 = time.time()
            rec = await loop.run_in_executor(_bot_pool, g.session.bot_move)
            await _broadcast(g, rec)
        wait = g.bot_delay - (time.time() - t0)
        await asyncio.sleep(max(0.0, wait))


@app.websocket("/ws/{gid}")
async def play(ws: WebSocket, gid: str, token: str | None = None):
    await ws.accept()
    g = GAMES.get(gid)
    if g is None:
        await _send(ws, {"type": "error", "message": "no such game"})
        await ws.close()
        return
    seat = None
    if token is not None:
        if token not in g.tokens:
            await _send(ws, {"type": "error", "message": "unknown seat token"})
            await ws.close()
            return
        seat = g.tokens[token]
    g.conns[ws] = seat
    await _send(ws, {"type": "welcome", "protocol": PROTOCOL, "game_id": gid, "seat": seat, "seq": g.seq,
                     "board_config": _board_config(),
                     "view": g.session.view(for_seat=-1 if seat is None else seat)})
    try:
        while True:
            try:
                msg = json.loads(await ws.receive_text())
            except (ValueError, TypeError):
                await _send(ws, {"type": "error", "message": "messages must be JSON"})
                continue
            kind = msg.get("type")
            if kind == "ping":
                await _send(ws, {"type": "pong"})
            elif kind == "move":
                if seat is None:
                    await _send(ws, {"type": "error", "message": "spectators cannot move"})
                    continue
                async with g.lock:
                    try:
                        rec = g.session.apply(msg.get("action"), seat=seat)
                    except IllegalMove as e:
                        await _send(ws, {"type": "error", "message": str(e)})
                        continue
                    await _broadcast(g, rec)
                _kick_bots(g)
            elif kind in ("complete_turn", "restart_turn"):
                if seat is None:
                    await _send(ws, {"type": "error", "message": "spectators cannot play"})
                    continue
                async with g.lock:
                    try:
                        getattr(g.session, kind)(seat=seat)
                    except IllegalMove as e:
                        await _send(ws, {"type": "error", "message": str(e)})
                        continue
                    await _broadcast(g, None, event=kind)
                _kick_bots(g)
            elif kind == "end_game":
                # Any seated human may stop the game for everyone (clients ask "are you
                # sure?" first). It is saved like a finished game and leaves the list.
                if seat is None:
                    await _send(ws, {"type": "error", "message": "spectators cannot end the game"})
                    continue
                async with g.lock:
                    try:
                        g.session.end_game(seat)
                    except IllegalMove as e:
                        await _send(ws, {"type": "error", "message": str(e)})
                        continue
                    await _broadcast(g, None, event="end_game")
                GAMES.pop(g.id, None)
            elif kind == "view":
                await _send(ws, {"type": "state", "seq": g.seq, "last_move": None,
                                 "view": g.session.view(for_seat=-1 if seat is None else seat)})
            else:
                await _send(ws, {"type": "error", "message": f"unknown message type {kind!r}"})
    except WebSocketDisconnect:
        pass
    finally:
        g.conns.pop(ws, None)


# The web client (step 4) is served from play/web/ when it exists.
if os.path.isdir(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))


# Human play differs from training only in these ini values (each can be changed with
# --rule KEY=VALUE). block_consecutive_income is a training aid, not a game rule.
PLAY_RULES = {"block_consecutive_income": "false"}


def play_config(path, overrides, out_dir):
    """A copy of the ini with some values replaced, so human play and training share one
    ini. A key that isn't in the file is added to [Rules]. Returns the copy's path."""
    lines = open(path).read().splitlines()
    done = set()
    for i, line in enumerate(lines):
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=", line)
        if m and m.group(1) in overrides:
            lines[i] = f"{m.group(1)} = {overrides[m.group(1)]}"
            done.add(m.group(1))
    missing = [f"{k} = {v}" for k, v in overrides.items() if k not in done]
    if missing:
        at = next((i + 1 for i, l in enumerate(lines) if l.strip() == "[Rules]"), None)
        if at is None:
            lines += ["[Rules]"] + missing
        else:
            lines[at:at] = missing
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "play_" + os.path.basename(path))
    with open(out, "w") as f:
        f.write("# Written by play/server.py from " + path + "; edit that file instead.\n")
        f.write("\n".join(lines) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 to accept other machines (LAN / Tailscale)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--config_file", default=SETTINGS["config_file"])
    ap.add_argument("--models_dir", default=SETTINGS["models_dir"])
    ap.add_argument("--max_sims", type=int, default=SETTINGS["max_sims"])
    ap.add_argument("--record_dir", default=os.path.join(HERE, "games_played"),
                    help="finished games are saved here as JSON ('' to turn off)")
    ap.add_argument("--gpu", action="store_true", help="let TensorFlow use the GPU")
    ap.add_argument("--rule", action="append", default=[], metavar="KEY=VALUE",
                    help="override an ini value for these games (repeatable); by default "
                         + ", ".join(f"{k}={v}" for k, v in PLAY_RULES.items()))
    args = ap.parse_args()
    rules = dict(PLAY_RULES)
    for r in args.rule:
        k, sep, v = r.partition("=")
        if not sep:
            ap.error(f"--rule needs KEY=VALUE, got {r!r}")
        rules[k.strip()] = v.strip()
    config = play_config(os.path.abspath(args.config_file), rules,
                         os.path.join(HERE, "games_played"))
    SETTINGS["rules"] = rules
    SETTINGS.update(config_file=config,
                    models_dir=os.path.abspath(args.models_dir),
                    max_sims=args.max_sims, record_dir=args.record_dir or None)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    if not args.gpu:
        tf.config.set_visible_devices([], "GPU")
    import pyspiel
    pyspiel.mali_ba.set_log_level(pyspiel.mali_ba.LogLevel.WARN)
    import uvicorn
    print(f"Mali-Ba server on http://{args.host}:{args.port}  (protocol {PROTOCOL}, "
          f"{len(list_models())} models in {SETTINGS['models_dir']})")
    print(f"rules from {args.config_file}, with " + ", ".join(f"{k} = {v}" for k, v in rules.items()))
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
