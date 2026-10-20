"""GameSession: one game of Mali-Ba with any mix of human and bot seats.

The engine is the single source of truth. A front end asks for view() (the board, whose
turn it is, and -- on a human's turn -- the legal moves, described for display), sends
one choice with apply(), and lets bots move with bot_move(). It needs no rules: every
move it can send comes from legal_moves().

    s = GameSession("mali_ba.ini", seats=[
            {"type": "human", "name": "Rob"},
            {"type": "network", "weights": "mali_ba_agent_vD039.weights.h5", "sims": 300},
            {"type": "heuristic"}], seed=7)
    while not s.is_over():
        if s.awaiting_complete is not None:
            s.complete_turn()                 # the human confirms the finished turn
        elif s.is_human_turn():
            moves = s.legal_moves()           # [{"action", "name", "kind", "label", ...}]
            s.apply(moves[0]["action"])       # or the move's name, e.g. "MancalaDir_2"
        else:
            s.bot_move()
    print(s.result())

A human's turn runs until play passes to another seat, or is one token placement (each
placement is a turn, also two in a row at a serpentine bend); the game then waits on
that human's complete_turn(). Until then restart_turn() puts the game back to the start of
the turn (any number of times).

Not thread-safe by itself: a server holds one lock per session (bot_move can take
seconds; run it off the event loop).
"""
import json
import random
import time

from mali_ba.play.bots import make_bot
from mali_ba.play.describe import common_goods_list, describe_action
from mali_ba.play.goods import GoodsSummarizer, read_rules, regions_from_ini


class IllegalMove(ValueError):
    pass


# Engine enums (mali_ba_common.h): seats are PlayerColor 0..; meeples in board JSON
# are MeepleColor ints; trade post "type" 1 = post, 2 = center.
PLAYER_COLORS = ["Red", "Green", "Blue", "Violet", "Pink"]
# Where the background map (ui/mali_ba_map.jpg) sits relative to the hexes: x, y, width,
# height in hex radii from the centre of hex (0,0,0), screen axes (y down, flat-top hexes:
# centre x = 1.5*q, y = sqrt(3)*(y-z)/2). Measured from the original visualizer's layout,
# whose nudges were tuned to line the map up with the board.
MAP_RECT = [-11.214, -7.147, 21.824, 16.034]
MEEPLE_COLORS = ["solid black", "clear black", "solid silver", "clear silver", "clear white",
                 "solid gold", "clear gold", "solid bronze", "clear bronze", "clear tan"]


def board_config(game, config_file=None):
    """The board's fixed parts, which never change during a game: the hexes, the
    cities, the regions, and the names behind the board JSON's numbers."""
    import pyspiel
    g = pyspiel.mali_ba.downcast_game(game)
    regions, source = _regions(g), "engine"
    if not regions and config_file:
        regions, source = regions_from_ini(config_file), "ini"
    return {
        "grid_radius": g.get_grid_radius(),
        "valid_hexes": sorted([h.x, h.y, h.z] for h in g.get_valid_hexes()),
        "cities": [{"id": c.id, "name": c.name, "culture": c.culture,
                    "location": [c.location.x, c.location.y, c.location.z],
                    "common_good": c.common_good, "rare_good": c.rare_good}
                   for c in g.get_cities()],
        "player_colors": PLAYER_COLORS[:game.num_players()],
        "meeple_colors": MEEPLE_COLORS,
        "post_types": {"1": "post", "2": "center"},
        "map_rect": MAP_RECT,
        "regions": regions,
        # "ini" until the engine is built with its region bindings (5d65fc7); same regions
        "regions_source": source if regions else None,
    }


def _regions(g):
    """[{"id", "name", "hexes"}] from the engine; [] with an engine built before the
    region bindings (2026-10-19); board_config then reads them from the ini."""
    if not hasattr(g, "get_region_for_hex"):
        return []
    by_id = {}
    for h in g.get_valid_hexes():
        rid = g.get_region_for_hex(h)
        if rid >= 0:
            by_id.setdefault(rid, []).append([h.x, h.y, h.z])
    return [{"id": rid, "name": g.get_region_name(rid), "hexes": sorted(by_id.get(rid, []))}
            for rid in g.get_valid_region_ids()]


class GameSession:

    def __init__(self, config_file, seats, seed=None, game=None):
        import pyspiel
        self._pyspiel = pyspiel
        self.config_file = config_file
        self.game = game or pyspiel.load_game("mali_ba", {"config_file": config_file})
        n = self.game.num_players()
        if len(seats) != n:
            raise ValueError(f"this game has {n} seats; got {len(seats)} seat specs")
        self.seed = random.randrange(2**31) if seed is None else int(seed)
        self._rng = random.Random(self.seed)
        self.goods_list = common_goods_list(self.game)
        cfg = board_config(self.game, config_file)
        self._city_names = {tuple(c["location"]): c["name"] for c in cfg["cities"]}
        self._goods = GoodsSummarizer(cfg["cities"], cfg["regions"], read_rules(config_file))

        self.seats = []
        self._bots = {}
        for i, spec in enumerate(seats):
            spec = dict(spec)
            if spec.get("type") == "human":
                self.seats.append({"seat": i, "type": "human",
                                   "name": spec.get("name") or f"Player {i + 1}"})
            else:
                bot = make_bot(self.game, spec)
                self._bots[i] = bot
                self.seats.append({"seat": i, "type": bot.kind, "name": bot.name})

        self.state = self.game.new_initial_state()
        self._ms().seed_rng(self.seed & 0xFFFFFFFF)   # the heuristic's own randomness
        self.history = []          # [{"n", "seat", "action", "name", "label", "t"}]
        # A human's turn: a copy of the game from its start (for Restart Turn), and, once
        # their moves have passed play on, the seat waiting to press Complete Turn. Until
        # then the game stays with them: bots don't move and nothing else is legal.
        self._turn = None          # {"seat", "state", "history_len"}
        self.awaiting_complete = None
        self.ended_by = None       # a human seat that ended the game early
        self._resolve_chance()
        self._start_turn()

    # ---- queries -------------------------------------------------------------------

    def _ms(self):
        return self._pyspiel.mali_ba.downcast_state(self.state)

    def state_json(self):
        return json.loads(self._ms().serialize())

    def phase(self):
        return str(self._ms().current_phase()).split(".")[-1]

    def is_over(self):
        if self.ended_by is not None:
            return True
        return self.state.is_terminal() and self.awaiting_complete is None

    def current_seat(self):
        """The seat to move, or None when the game is over. A human whose turn is done
        but not yet completed is still the seat to move."""
        if self.ended_by is not None:
            return None
        if self.awaiting_complete is not None:
            return self.awaiting_complete
        if self.state.is_terminal():
            return None
        p = self.state.current_player()
        return p if p >= 0 else None

    def is_human_turn(self):
        s = self.current_seat()
        return s is not None and s not in self._bots

    def is_bot_turn(self):
        s = self.current_seat()
        return s is not None and s in self._bots

    def legal_moves(self, state_json=None):
        """The current seat's legal moves, described (see describe.py). None while a
        finished turn waits for Complete Turn."""
        if self.current_seat() is None or self.awaiting_complete is not None:
            return []
        sj = state_json if state_json is not None else self.state_json()
        moves = [describe_action(self._ms(), a, sj, self.goods_list)
                 for a in self.state.legal_actions()]
        if any(m["kind"] == "declare_route" for m in moves):
            self._add_route_hexes(moves, sj)
        return moves

    def _add_route_hexes(self, moves, sj):
        """Each DeclareRoute_k names the k-th route the engine found. Its hexes come from
        the engine itself: apply the move to a copy of the state and read the new route
        from the copy's JSON. (About 0.25 ms per option.)"""
        import pyspiel
        before = {r["id"] for r in sj.get("tradeRoutes", [])}
        for m in moves:
            if m["kind"] != "declare_route":
                continue
            c = self.state.clone()
            c.apply_action(m["action"])
            after = json.loads(pyspiel.mali_ba.downcast_state(c).serialize()).get("tradeRoutes", [])
            new = [r for r in after if r["id"] not in before]
            if not new:
                continue
            hexes = [[int(v) for v in h.split(",")] for h in new[0]["hexes"]]
            cities = [self._city_names[tuple(h)] for h in hexes if tuple(h) in self._city_names]
            m["hexes"] = hexes
            m["cities"] = cities
            where = (" - ".join(cities) if len(cities) > 1 else f"through {cities[0]}" if cities
                     else f"from ({new[0]['hexes'][0]}) to ({new[0]['hexes'][-1]})")
            m["label"] = f"Route {where} ({len(hexes)} hexes)"

    def result(self):
        """None while playing; then winner (None if the move cap ended it), reason,
        returns, the score breakdown and the number of moves."""
        if not self.is_over():
            return None
        ms = self._ms()
        if self.ended_by is not None:
            name = self.seats[self.ended_by]["name"]
            return {"winner": None, "reason": f"Ended early by {name}", "ended_by": self.ended_by,
                    "returns": None, "score": ms.get_score_breakdown_string(),
                    "moves": len(self.history)}
        reason = ms.get_game_end_reason()
        rets = [float(r) for r in self.state.returns()]
        winner = None if "Max game length" in reason else rets.index(max(rets))
        return {"winner": winner, "reason": reason, "returns": rets,
                "score": ms.get_score_breakdown_string(), "moves": len(self.history)}

    def board_config(self):
        return board_config(self.game, self.config_file)

    def view(self, for_seat=None):
        """Everything a front end draws. Legal moves are included when it is a human's
        turn and for_seat is None or that seat (a networked client sees moves only on
        its own turn). Mali-Ba has no hidden information, so the board is the same for
        everyone."""
        sj = self.state_json()
        seat = self.current_seat()
        show_moves = self.is_human_turn() and (for_seat is None or for_seat == seat)
        return {
            "board": sj,
            "phase": self.phase(),
            "current_seat": seat,
            "seats": self.seats,
            "legal_moves": self.legal_moves(sj) if show_moves else [],
            "final_round": bool(self._ms().is_final_round()),
            "history": self.history[-200:],
            "result": self.result(),
            "goods": self.goods_list,
            "players": self._goods.summary(sj),   # goods, sets, rare regions (goods.py)
            "turn": self.turn_info(),
        }

    def turn_info(self):
        """For a human's turn: {"seat", "moves" (decisions so far this turn),
        "can_restart", "complete" (done: waiting for Complete Turn)}; else None."""
        if self._turn is None:
            return None
        made = len(self.history) - self._turn["history_len"]
        return {"seat": self._turn["seat"], "moves": made, "can_restart": made > 0,
                "complete": self.awaiting_complete is not None}

    # ---- moves ---------------------------------------------------------------------

    def apply(self, move, seat=None):
        """Applies one decision for the human to move. move: an action id or its name.
        seat (optional): the seat sending it -- rejected if it is not that seat's turn.
        Returns the history record."""
        cur = self.current_seat()
        if cur is None:
            raise IllegalMove("the game is over")
        if seat is not None and seat != cur:
            raise IllegalMove(f"it is seat {cur}'s turn, not seat {seat}'s")
        if cur in self._bots:
            raise IllegalMove(f"seat {cur} is a bot")
        if self.awaiting_complete is not None:
            raise IllegalMove("the turn is done: complete it or restart it")
        rec = self._apply(self._to_action(move))
        # The turn is done when play passes on, and after every token placement: with
        # serpentine placement a player places twice in a row, but each is its own turn.
        if (self.state.is_terminal() or self.state.current_player() != cur
                or rec["kind"] == "place_token"):
            self.awaiting_complete = cur      # wait for Complete Turn
        return rec

    def complete_turn(self, seat=None):
        """The human whose turn is done hands play to the next player."""
        cur = self.awaiting_complete
        if cur is None:
            raise IllegalMove("the turn isn't finished yet")
        if seat is not None and seat != cur:
            raise IllegalMove(f"it is seat {cur}'s turn, not seat {seat}'s")
        self.awaiting_complete = None
        self._turn = None
        self._start_turn()

    def restart_turn(self, seat=None):
        """Puts the game back to the start of the current human's turn."""
        if self._turn is None:
            raise IllegalMove("it is not a human's turn")
        if seat is not None and seat != self._turn["seat"]:
            raise IllegalMove(f"it is seat {self._turn['seat']}'s turn, not seat {seat}'s")
        self.state = self._turn["state"].clone()
        del self.history[self._turn["history_len"]:]
        self.awaiting_complete = None

    def end_game(self, seat):
        """A human player stops the game for everyone. No winner; the score so far is
        reported."""
        if self.is_over():
            raise IllegalMove("the game is already over")
        if seat is None or seat in self._bots or not 0 <= seat < len(self.seats):
            raise IllegalMove("only a human player can end the game")
        self.ended_by = seat
        self.awaiting_complete = None
        self._turn = None

    def _start_turn(self):
        """At the start of a human's turn, keep a copy of the game for Restart Turn."""
        if self._turn is None and self.is_human_turn():
            self._turn = {"seat": self.current_seat(), "state": self.state.clone(),
                          "history_len": len(self.history)}

    def bot_move(self):
        """The bot to move makes one decision. Returns the history record."""
        cur = self.current_seat()
        if cur is None or cur not in self._bots:
            raise IllegalMove("it is not a bot's turn")
        legal = self.state.legal_actions()
        a = self._bots[cur].choose(self.state)
        if a not in legal:          # never expected; keep the game going if it happens
            a = self._rng.choice(legal)
        rec = self._apply(a)
        self._start_turn()
        return rec

    def run_bots(self, on_move=None, max_moves=None):
        """Bots move until a human's turn or the end. on_move(record) after each."""
        n = 0
        while self.is_bot_turn() and (max_moves is None or n < max_moves):
            rec = self.bot_move()
            n += 1
            if on_move:
                on_move(rec)
        return n

    def _to_action(self, move):
        legal = self.state.legal_actions()
        if isinstance(move, str) and not move.lstrip("-").isdigit():
            p = self.state.current_player()
            by_name = {self.state.action_to_string(p, a): a for a in legal}
            if move not in by_name:
                raise IllegalMove(f"{move!r} is not a legal move now")
            return by_name[move]
        a = int(move)
        if a not in legal:
            raise IllegalMove(f"action {a} is not a legal move now")
        return a

    def _apply(self, a):
        seat = self.current_seat()
        sj = self.state_json()
        desc = describe_action(self._ms(), a, sj, self.goods_list)
        if desc["kind"] == "declare_route":
            self._add_route_hexes([desc], sj)
        self.state.apply_action(a)
        rec = {"n": len(self.history), "seat": seat, "action": int(a),
               "name": desc["name"], "kind": desc["kind"], "label": desc["label"],
               "t": time.time()}
        for k in ("hex", "hexes", "good"):
            if desc.get(k) is not None:
                rec[k] = desc[k]
        self.history.append(rec)
        self._resolve_chance()
        return rec

    def _resolve_chance(self):
        # The meeple layout is a chance outcome; sample it from the session's seed.
        while self.state.is_chance_node():
            la = self.state.legal_actions()
            self.state.apply_action(la[self._rng.randrange(len(la))])
