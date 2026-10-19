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
        if s.is_human_turn():
            moves = s.legal_moves()           # [{"action", "name", "kind", "label", ...}]
            s.apply(moves[0]["action"])       # or the move's name, e.g. "MancalaDir_2"
        else:
            s.bot_move()
    print(s.result())

Not thread-safe by itself: a server holds one lock per session (bot_move can take
seconds; run it off the event loop).
"""
import json
import random
import time

from mali_ba.play.bots import make_bot
from mali_ba.play.describe import common_goods_list, describe_action


class IllegalMove(ValueError):
    pass


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
        self._resolve_chance()

    # ---- queries -------------------------------------------------------------------

    def _ms(self):
        return self._pyspiel.mali_ba.downcast_state(self.state)

    def state_json(self):
        return json.loads(self._ms().serialize())

    def phase(self):
        return str(self._ms().current_phase()).split(".")[-1]

    def is_over(self):
        return self.state.is_terminal()

    def current_seat(self):
        """The seat to move, or None when the game is over."""
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
        """The current seat's legal moves, described (see describe.py)."""
        if self.current_seat() is None:
            return []
        sj = state_json if state_json is not None else self.state_json()
        return [describe_action(self._ms(), a, sj, self.goods_list)
                for a in self.state.legal_actions()]

    def result(self):
        """None while playing; then winner (None if the move cap ended it), reason,
        returns, the score breakdown and the number of moves."""
        if not self.state.is_terminal():
            return None
        ms = self._ms()
        reason = ms.get_game_end_reason()
        rets = [float(r) for r in self.state.returns()]
        winner = None if "Max game length" in reason else rets.index(max(rets))
        return {"winner": winner, "reason": reason, "returns": rets,
                "score": ms.get_score_breakdown_string(), "moves": len(self.history)}

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
        }

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
        return self._apply(self._to_action(move))

    def bot_move(self):
        """The bot to move makes one decision. Returns the history record."""
        cur = self.current_seat()
        if cur is None or cur not in self._bots:
            raise IllegalMove("it is not a bot's turn")
        legal = self.state.legal_actions()
        a = self._bots[cur].choose(self.state)
        if a not in legal:          # never expected; keep the game going if it happens
            a = self._rng.choice(legal)
        return self._apply(a)

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
        desc = describe_action(self._ms(), a, self.state_json(), self.goods_list)
        self.state.apply_action(a)
        rec = {"n": len(self.history), "seat": seat, "action": int(a),
               "name": desc["name"], "kind": desc["kind"], "label": desc["label"],
               "t": time.time()}
        if "hex" in desc:
            rec["hex"] = desc["hex"]
        self.history.append(rec)
        self._resolve_chance()
        return rec

    def _resolve_chance(self):
        # The meeple layout is a chance outcome; sample it from the session's seed.
        while self.state.is_chance_node():
            la = self.state.legal_actions()
            self.state.apply_action(la[self._rng.randrange(len(la))])
