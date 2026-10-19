"""Pygame front end for the Mali-Ba game server: play or watch from any machine.

It holds no rules. It draws the board the server sends, highlights the hexes of the legal
moves, and sends the one you pick. The board drawing is the existing ui/visualizer_drawing.

  # a new game: you against D039 and the heuristic
  python -m mali_ba.play.pygame_client --new human,D039,heuristic
  # several humans at this one screen (hot seat)
  python -m mali_ba.play.pygame_client --new human,human,D039@100 --hotseat
  # join a game someone else created / watch one
  python -m mali_ba.play.pygame_client --join GAME_ID [--seat 1] --name Alice
  python -m mali_ba.play.pygame_client --watch GAME_ID
  # three bots, one move a second
  python -m mali_ba.play.pygame_client --new D040,D039,heuristic --bot_delay 1

Seat specs: human[:name], heuristic, or a model id from /api/models with an optional
@sims (D039@100). --server defaults to http://127.0.0.1:8765; for another machine give
its address, e.g. http://192.168.0.10:8765 or its Tailscale name.

Needs only pygame and websockets plus the mali_ba package (PYTHONPATH=open_spiel/python/games);
no engine build. Mouse: click a highlighted hex or a move button; the wheel zooms the board
and scrolls the move list; right-drag
pans. Keys: F fits the board, R toggles trade routes, H hides the final score, Esc quits.
"""
import argparse
import json
import math
import os
import queue
import sys
import threading
import urllib.error
import urllib.request

import pygame
from websockets.sync.client import connect

from mali_ba.config import (HEX_SIZE, PLAYER_COLOR_DICT, PlayerColor, WHITE, BLACK,
                            LIGHT_GRAY, DARK_GRAY, GRAY, YELLOW)
from mali_ba.classes.classes_other import City, HexCoord
from mali_ba.classes.game_state import GameStateCache
from mali_ba.ui import visualizer_drawing
from mali_ba.ui.visualizer_drawing import draw_board_state
from mali_ba.ui.visualizer_other import parse_and_update_state_from_json

SIDEBAR_W = 340
MAP_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "ui", "mali_ba_map.jpg")


# ---- server access -----------------------------------------------------------------

class Server:
    def __init__(self, base):
        self.base = base.rstrip("/")
        self.ws_base = "ws" + self.base[4:] if self.base.startswith("http") else self.base

    def call(self, method, path, body=None):
        req = urllib.request.Request(self.base + path, method=method,
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            try:
                detail = json.loads(e.read()).get("detail")
            except Exception:
                detail = str(e)
            sys.exit(f"server: {detail}")
        except urllib.error.URLError as e:
            sys.exit(f"cannot reach {self.base}: {e.reason}")


class Connection(threading.Thread):
    """One WebSocket (one seat, or a spectator). Received messages go to inbox as
    (seat, message); send() may be called from the main thread."""

    def __init__(self, url, seat, inbox):
        super().__init__(daemon=True)
        self.url, self.seat, self.inbox = url, seat, inbox
        self.ws = None

    def run(self):
        try:
            with connect(self.url, max_size=None, open_timeout=30) as ws:
                self.ws = ws
                for raw in ws:
                    self.inbox.put((self.seat, json.loads(raw)))
        except Exception as e:
            self.inbox.put((self.seat, {"type": "closed", "message": str(e)}))
        self.inbox.put((self.seat, {"type": "closed", "message": "connection closed"}))

    def send(self, msg):
        if self.ws is not None:
            self.ws.send(json.dumps(msg))


def parse_seat(tok):
    tok = tok.strip()
    if tok.startswith("human"):
        name = tok.split(":", 1)[1] if ":" in tok else None
        return {"type": "human", "name": name}
    if tok == "heuristic":
        return {"type": "heuristic"}
    model, _, sims = tok.partition("@")
    spec = {"type": "network", "model": model}
    if sims:
        spec["sims"] = int(sims)
    return spec


# ---- the window ----------------------------------------------------------------------

class Client:

    def __init__(self, server, game_id, tokens, board_cfg, title):
        self.server, self.game_id = server, game_id
        pygame.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((1280, 820), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.small = pygame.font.Font(None, 18)
        self.big = pygame.font.Font(None, 30)
        self.fonts = {"font": self.font, "small_font": self.small, "large_font": self.big}
        self.font_sizes = {"font": 22, "small_font": 18, "large_font": 30}

        self.cfg = board_cfg
        n = len(board_cfg["player_colors"])
        self.cache = GameStateCache(n)
        self.cache.update_num_players(n)
        self.cache.game_player_colors = [PlayerColor.from_int(i) for i in range(n)]
        self.cache.valid_hexes = {HexCoord(*h) for h in board_cfg["valid_hexes"]}
        self.cache.grid_radius = board_cfg["grid_radius"]
        self.cache.cities = [City(id_=c["id"], name=c["name"], culture=c["culture"],
                                  location=HexCoord(*c["location"]),
                                  common_good=c["common_good"], rare_good=c["rare_good"])
                             for c in board_cfg["cities"]]
        self.city_at = {c.location: c for c in self.cache.cities}
        self.map_img = pygame.image.load(MAP_FILE).convert() if os.path.exists(MAP_FILE) else None
        self.map_scaled = (None, None)  # (size, surface): rescaled only when the zoom changes
        if self.map_img is not None:
            # The shared drawing code makes hexes see-through when its map is set, and draws
            # that map centred in the window; this client pins the map to the hexes instead.
            visualizer_drawing.BACKGROUND_MAP = self.map_img
            visualizer_drawing.draw_background_map = lambda *a, **k: None

        self.inbox = queue.Queue()
        self.conns = {}
        if tokens:
            for seat, tok in tokens.items():
                self.conns[seat] = Connection(f"{server.ws_base}/ws/{game_id}?token={tok}",
                                              seat, self.inbox)
        else:
            self.conns[None] = Connection(f"{server.ws_base}/ws/{game_id}", None, self.inbox)
        for c in self.conns.values():
            c.start()

        self.views = {}              # seat (or None) -> latest view from that connection
        self.view = None
        self.selected_hex = None
        self.status = "connecting..."
        self.show_routes = True
        self.zoom, self.offset = 1.0, [0.0, 0.0]
        self.buttons = []            # [(rect, move)]
        self.scroll, self.list_key = 0, None
        self.route_filter = set()    # hexes a route must pass through (route choice)
        self.show_result = True
        self.hover = None
        self.fit()

    # -- geometry (flat-top cube coordinates, as in ui/visualizer.py) --
    def board_rect(self):
        w, h = self.screen.get_size()
        return pygame.Rect(0, 0, w - SIDEBAR_W, h)

    def hex_to_pixel(self, h):
        r = self.board_rect()
        size = HEX_SIZE / 2.0 * self.zoom
        return (round(r.centerx + self.offset[0] + size * 1.5 * h.x),
                round(r.centery + self.offset[1] + size * math.sqrt(3) * (h.y - h.z) / 2))

    def pixel_to_hex(self, x, y):
        r = self.board_rect()
        size = HEX_SIZE / 2.0 * self.zoom
        ax, ay = x - r.centerx - self.offset[0], y - r.centery - self.offset[1]
        xf = (2.0 / 3.0 * ax) / size
        zf = (-1.0 / 3.0 * ax + math.sqrt(3) / 3.0 * ay) / size
        yf = -xf - zf
        rx, ry, rz = round(xf), round(yf), round(zf)
        dx, dy, dz = abs(rx - xf), abs(ry - yf), abs(rz - zf)
        if dx > dy and dx > dz:
            rx = -ry - rz
        elif dy > dz:
            ry = -rx - rz
        else:
            rz = -rx - ry
        h = HexCoord(int(rx), int(rz), int(ry))     # the visualizer's axis convention
        return h if h in self.cache.valid_hexes else None

    def fit(self):
        hexes = self.cache.valid_hexes
        if not hexes:
            return
        s = HEX_SIZE / 2.0
        xs = [s * 1.5 * h.x for h in hexes]
        ys = [s * math.sqrt(3) * (h.y - h.z) / 2 for h in hexes]
        w, hgt = max(xs) - min(xs) + 2 * s, max(ys) - min(ys) + math.sqrt(3) * s
        r = self.board_rect()
        self.zoom = max(0.2, min((r.width - 20) / w, (r.height - 20) / hgt))
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
        self.offset = [-cx * self.zoom, -cy * self.zoom]

    # -- state --
    def pump(self):
        while True:
            try:
                seat, msg = self.inbox.get_nowait()
            except queue.Empty:
                return
            t = msg.get("type")
            if t in ("welcome", "state"):
                self.views[seat] = msg["view"]
                if t == "welcome":
                    self.status = "connected" + ("" if seat is None else f" as seat {seat}")
                lm = msg.get("last_move")
                if lm:
                    who = self.views[seat]["seats"][lm["seat"]]["name"]
                    self.status = f"{who}: {lm['label']}"
            elif t == "error":
                self.status = "Rejected: " + msg["message"]
            elif t == "closed":
                self.status = "Disconnected: " + msg["message"]

    def refresh_view(self):
        if not self.views:
            return
        any_view = max(self.views.values(), key=lambda v: len(v["history"]) and v["history"][-1]["n"])
        cur = any_view["current_seat"]
        v = self.views.get(cur) if cur in self.views else any_view
        if v is not self.view:
            self.view = v
            parse_and_update_state_from_json(json.dumps(v["board"]), self.cache)
            if self.selected_hex and not self.moves_at(self.selected_hex):
                self.selected_hex = None
            if not self.route_moves():
                self.route_filter = set()

    def my_moves(self):
        return (self.view or {}).get("legal_moves") or []

    def route_moves(self):
        return [m for m in self.my_moves() if m.get("hexes")]

    def listed_moves(self):
        """The moves shown as buttons: those with no hex (routes narrowed by the hexes
        clicked), plus every move on the selected hex when it has several."""
        out = []
        for m in self.my_moves():
            if m.get("hex"):
                continue
            if m.get("hexes") and self.route_filter and                     not self.route_filter <= {HexCoord(*h) for h in m["hexes"]}:
                continue
            out.append(m)
        return out + (self.moves_at(self.selected_hex) if self.selected_hex else [])

    def moves_at(self, h):
        return [m for m in self.my_moves() if m.get("hex") and HexCoord(*m["hex"]) == h]

    def send_move(self, m):
        cur = self.view["current_seat"]
        conn = self.conns.get(cur)
        if conn is None:
            return
        conn.send({"type": "move", "action": m["action"]})
        self.selected_hex = None
        self.route_filter = set()
        self.status = "sent: " + m["label"]

    # -- input --
    def click(self, pos):
        if self.view and self.view.get("result") and self.board_rect().collidepoint(pos):
            self.show_result = not self.show_result      # the score box covers the board
            return
        for rect, m in self.buttons:
            if rect.collidepoint(pos):
                self.send_move(m)
                return
        if not self.board_rect().collidepoint(pos):
            return
        h = self.pixel_to_hex(*pos)
        if h is None:
            return
        if any(h == HexCoord(*x) for m in self.route_moves() for x in m["hexes"]):
            self.route_filter ^= {h}       # narrow the route list to routes through h
            self.scroll = 0
            return
        here = self.moves_at(h)
        if len(here) == 1:
            self.send_move(here[0])
        elif here:
            self.selected_hex = h          # several moves on this hex: pick one in the sidebar
        else:
            self.selected_hex = None

    # -- drawing --
    def draw(self):
        self.screen.fill(WHITE)
        moves = self.my_moves()
        highlight = [HexCoord(*m["hex"]) for m in moves if m.get("hex")]
        highlight += list({HexCoord(*h) for m in moves if m.get("hexes") for h in m["hexes"]})
        mid = (self.view or {}).get("board", {}).get("midTurnState") or {}
        mancala_hex = (HexCoord(*map(int, mid["mancalaHex"].split(",")))
                       if self.view and "MANCALA" in self.view["phase"] and mid.get("mancalaHex")
                       else None)
        self.draw_map()
        draw_board_state(self.screen, self.cache, self.hex_to_pixel, self.zoom, self.fonts,
                         self.font_sizes, highlight, self.selected_hex or mancala_hex,
                         show_trade_routes=self.show_routes)
        self.draw_sidebar(moves)
        self.draw_route_marks()
        if self.view and self.view.get("result") and self.show_result:
            self.draw_result(self.view["result"])
        pygame.display.flip()

    def draw_map(self):
        """The background map, pinned to the hexes by board_config["map_rect"]."""
        if self.map_img is None or "map_rect" not in self.cfg:
            return
        x, y, w, h = self.cfg["map_rect"]
        size = HEX_SIZE / 2.0 * self.zoom
        dims = (max(1, round(w * size)), max(1, round(h * size)))
        if self.map_scaled[0] != dims:
            self.map_scaled = (dims, pygame.transform.smoothscale(self.map_img, dims))
        ox, oy = self.hex_to_pixel(HexCoord(0, 0, 0))
        self.screen.set_clip(self.board_rect())
        self.screen.blit(self.map_scaled[1], (round(ox + x * size), round(oy + y * size)))
        self.screen.set_clip(None)

    def draw_route_marks(self):
        size = HEX_SIZE / 2.0 * self.zoom
        for h in self.route_filter:
            pygame.draw.circle(self.screen, BLACK, self.hex_to_pixel(h), int(size * 0.8), 3)
        pos = pygame.mouse.get_pos()
        for rect, m in self.buttons:
            if m.get("hexes") and rect.collidepoint(pos):
                pts = [self.hex_to_pixel(HexCoord(*h)) for h in m["hexes"]]
                col = PLAYER_COLOR_DICT.get(PlayerColor.from_int(self.view["current_seat"]), GRAY)
                for p in pts:
                    pygame.draw.circle(self.screen, BLACK, p, int(size * 0.45))
                    pygame.draw.circle(self.screen, col, p, int(size * 0.38))
                if len(pts) > 1:
                    pygame.draw.lines(self.screen, BLACK, False, pts, 7)
                    pygame.draw.lines(self.screen, col, False, pts, 4)

    def text(self, s, x, y, font=None, color=BLACK, max_w=None):
        font = font or self.font
        if max_w:
            while s and font.size(s)[0] > max_w:
                s = s[:-4] + "..."
        self.screen.blit(font.render(s, True, color), (x, y))
        return y + font.get_linesize()

    def draw_sidebar(self, moves):
        w, h = self.screen.get_size()
        x0 = w - SIDEBAR_W
        pygame.draw.rect(self.screen, (245, 242, 235), (x0, 0, SIDEBAR_W, h))
        pygame.draw.line(self.screen, DARK_GRAY, (x0, 0), (x0, h))
        x, y, mw = x0 + 12, 10, SIDEBAR_W - 24
        y = self.text(f"Mali-Ba  game {self.game_id}", x, y, self.big)
        v = self.view
        self.buttons = []
        if v is None:
            self.text(self.status, x, y + 10, max_w=mw)
            return
        mine = [s for s in self.conns if s is not None]
        y += 6
        sj = v["board"]
        for seat in v["seats"]:
            i = seat["seat"]
            col = PLAYER_COLOR_DICT.get(PlayerColor.from_int(i), GRAY)
            pygame.draw.circle(self.screen, col, (x + 8, y + 9), 7)
            pygame.draw.circle(self.screen, BLACK, (x + 8, y + 9), 7, 1)
            if v["current_seat"] == i:      # whose turn: a pointer left of the colour
                pygame.draw.polygon(self.screen, BLACK, [(x - 9, y + 3), (x - 9, y + 15), (x - 2, y + 9)])
            you = " (you)" if i in mine else ""
            y = self.text(f"{seat['name']}{you}", x + 20, y, max_w=mw - 20)
            cg = sum(sj["commonGoods"][i].values()) if i < len(sj["commonGoods"]) else 0
            rg = sj["rareGoods"][i] if i < len(sj["rareGoods"]) else {}
            posts = sj["playerPostsSupply"][i] if i < len(sj["playerPostsSupply"]) else "?"
            y = self.text(f"common {cg}  rare {sum(rg.values())} ({len(rg)} kinds)  posts left {posts}",
                          x + 20, y, self.small, DARK_GRAY, mw - 20)
        y += 4
        phase = v["phase"].replace("_", " ").lower()
        y = self.text(f"Phase: {phase}" + ("   — final round" if v.get("final_round") else ""),
                      x, y, self.small, DARK_GRAY)
        y = self.text(self.status, x, y, self.small, (120, 40, 40), mw)
        y += 8

        # Moves: buttons for moves with no hex, and for a hex that has several moves.
        if moves:
            hex_moves = [m for m in moves if m.get("hex")]
            listed = self.listed_moves()
            if self.route_moves():
                y = self.text("Choose a trade route: hover to see it;", x, y, self.small)
                y = self.text("click hexes to show only routes through them", x, y, self.small)
                if self.route_filter:
                    y = self.text(f"({len(self.route_filter)} hex filter; click again to remove)",
                                  x, y, self.small, DARK_GRAY)
            elif hex_moves:
                y = self.text("Click a highlighted hex" + (" or:" if listed else ""), x, y, self.font)
            else:
                y = self.text("Your move:", x, y, self.font)
            # The list scrolls (mouse wheel over the sidebar): a route choice can offer dozens.
            key = tuple(m["action"] for m in listed)
            if key != self.list_key:
                self.list_key, self.scroll = key, 0
            room = max(1, (h - 170 - y) // 30)
            if len(listed) > room:
                room = max(1, room - 2)          # space for the two "more" lines
            self.scroll = max(0, min(self.scroll, len(listed) - room))
            if len(listed) > room:
                y = self.text(f"   ^ {self.scroll} more (scroll)" if self.scroll else "",
                              x, y, self.small, DARK_GRAY)
            shown = listed[self.scroll:self.scroll + room]
            for m in shown:
                r = pygame.Rect(x, y + 2, mw, 26)
                hov = r.collidepoint(pygame.mouse.get_pos())
                pygame.draw.rect(self.screen, YELLOW if hov else LIGHT_GRAY, r, border_radius=5)
                pygame.draw.rect(self.screen, DARK_GRAY, r, 1, border_radius=5)
                self.text(m["label"], r.x + 8, r.y + 5, self.font, max_w=mw - 16)
                self.buttons.append((r, m))
                y += 30
            hidden = len(listed) - self.scroll - len(shown)
            if hidden > 0:
                y = self.text(f"   v {hidden} more (scroll)", x, y, self.small, DARK_GRAY)
        elif not v.get("result"):
            cur = v["current_seat"]
            if cur is not None:
                y = self.text(f"Waiting for {v['seats'][cur]['name']}…", x, y, self.font, DARK_GRAY)

        # Hovered hex: its city, if any.
        if self.hover:
            c = self.city_at.get(self.hover)
            yy = h - 150
            self.text(f"hex {self.hover.x},{self.hover.y},{self.hover.z}", x, yy, self.small, DARK_GRAY)
            if c:
                self.text(f"{c.name} ({c.culture})", x, yy + 16, self.small)
                self.text(f"common: {c.common_good}   rare: {c.rare_good}", x, yy + 32, self.small,
                          max_w=mw)
        # Recent moves.
        yy = h - 96
        self.text("Recent moves", x, yy, self.small, DARK_GRAY)
        for rec in v["history"][-4:]:
            yy += 18
            col = PLAYER_COLOR_DICT.get(PlayerColor.from_int(rec["seat"]), GRAY)
            pygame.draw.circle(self.screen, col, (x + 5, yy + 6), 4)
            self.text(rec["label"], x + 14, yy, self.small, max_w=mw - 14)

    def draw_result(self, res):
        lines = [l for l in (res.get("score") or "").splitlines() if l.strip() and "====" not in l]
        win = res.get("winner")
        head = ("Game over: no winner (move cap)" if win is None
                else f"{self.view['seats'][win]['name']} wins — {res['reason']}")
        bw = self.board_rect().width
        box = pygame.Rect(30, 30, min(620, bw - 60), 40 + 18 * len(lines))
        s = pygame.Surface(box.size, pygame.SRCALPHA)
        s.fill((255, 255, 255, 235))
        self.screen.blit(s, box.topleft)
        pygame.draw.rect(self.screen, BLACK, box, 2)
        y = self.text(head, box.x + 12, box.y + 10, self.big, max_w=box.width - 24)
        for l in lines:
            y = self.text(l, box.x + 12, y, self.small, max_w=box.width - 24)

    # -- loop --
    def run(self):
        dragging, last = False, None
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                    pygame.quit()
                    return
                if e.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)
                    self.fit()
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_f:
                    self.fit()
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_h:
                    self.show_result = not self.show_result
                elif e.type == pygame.KEYDOWN and e.key == pygame.K_r:
                    self.show_routes = not self.show_routes
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    self.click(e.pos)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                    dragging, last = True, e.pos
                elif e.type == pygame.MOUSEBUTTONUP and e.button == 3:
                    dragging = False
                elif e.type == pygame.MOUSEMOTION:
                    if dragging:
                        self.offset[0] += e.pos[0] - last[0]
                        self.offset[1] += e.pos[1] - last[1]
                        last = e.pos
                    self.hover = (self.pixel_to_hex(*e.pos)
                                  if self.board_rect().collidepoint(e.pos) else None)
                elif e.type == pygame.MOUSEWHEEL and not self.board_rect().collidepoint(pygame.mouse.get_pos()):
                    self.scroll = max(0, self.scroll - e.y * 3)
                elif e.type == pygame.MOUSEWHEEL:
                    f = 1.1 if e.y > 0 else 1 / 1.1
                    self.zoom *= f
                    self.offset = [self.offset[0] * f, self.offset[1] * f]
            self.pump()
            self.refresh_view()
            self.draw()
            self.clock.tick(30)


def main():
    ap = argparse.ArgumentParser(description="Mali-Ba pygame client",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("\n\n", 1)[1])
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--new", help="comma-separated seat specs, e.g. human,D039,heuristic")
    g.add_argument("--join", metavar="GAME_ID")
    g.add_argument("--watch", metavar="GAME_ID")
    ap.add_argument("--seat", type=int, help="the human seat to take (default: the first open)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--hotseat", action="store_true", help="take every human seat of a --new game")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--bot_delay", type=float, default=0.3,
                    help="seconds between bot moves, so you can follow them")
    args = ap.parse_args()

    srv = Server(args.server)
    info = srv.call("GET", "/api/info")
    if info["protocol"] != 1:
        sys.exit(f"server speaks protocol {info['protocol']}; this client speaks 1")
    board_cfg = srv.call("GET", "/api/board")
    tokens = {}
    if args.new:
        seats = [parse_seat(t) for t in args.new.split(",")]
        if args.name:
            for s in seats:
                if s["type"] == "human" and not s.get("name"):
                    s["name"] = args.name
                    break
        print("creating the game (loading networks can take a few seconds)...")
        game = srv.call("POST", "/api/games", {"seats": seats, "seed": args.seed,
                                               "bot_delay": args.bot_delay})
        gid = game["game_id"]
        humans = [i for i, s in enumerate(seats) if s["type"] == "human"]
        take = humans if args.hotseat else ([args.seat] if args.seat is not None else humans[:1])
        for seat in take:
            tokens[seat] = srv.call("POST", f"/api/games/{gid}/join",
                                    {"seat": seat, "name": seats[seat].get("name")})["token"]
        others = [s for s in humans if s not in take]
        print(f"game {gid}" + (f"; open seats {others}: others join with "
                               f"--join {gid} --server {args.server}" if others else ""))
    elif args.join:
        gid = args.join
        j = srv.call("POST", f"/api/games/{gid}/join", {"seat": args.seat, "name": args.name})
        tokens[j["seat"]] = j["token"]
        print(f"joined game {gid} as seat {j['seat']}")
    else:
        gid = args.watch
    title = f"Mali-Ba — {'watching' if not tokens else 'seat ' + ','.join(map(str, tokens))} — {gid}"
    Client(srv, gid, tokens, board_cfg, title).run()


if __name__ == "__main__":
    main()
