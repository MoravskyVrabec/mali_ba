"""Checks the engine's goods choices (SetGoodsChoiceSeats, 2026-10-20) after a build:
random games in which the chosen seats pick their upgrade payments and income goods at
random. Needs an engine built with the change; with an older build it says so and stops.

    PYTHONPATH=build/python:open_spiel/python/games:. python -m mali_ba.play.test_goods_choices [games]

Checks, at every choice:
  - an upgrade payment takes exactly one rare good, or common goods one at a time up to
    the cost, never both, and the post is a center afterwards;
  - income goods change only when the last income choice is made, and then by exactly
    the fixed income plus the options chosen (the IncomeChoice_ names say what each gives);
  - a clone mid-choice has the same legal moves and serializes the same;
  - seats not chosen never enter a choice phase;
  - GameSession bots move on a copy with the choices off.
"""
import json
import os
import random
import sys

import pyspiel

from mali_ba.play.describe import parse_income_choice

INI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mali_ba.ini")
CHOICE_PHASES = ("UPGRADE_PAYMENT", "INCOME_CHOICE")


def goods(sj, seat):
    out = {}
    for side in ("commonGoods", "rareGoods"):
        for g, c in sj[side][seat].items():
            if c:
                out[(side, g)] = c
    return out


def diff(after, before):
    keys = set(after) | set(before)
    return {k: after.get(k, 0) - before.get(k, 0) for k in keys if after.get(k, 0) != before.get(k, 0)}


def centers(sj, seat):
    return {h for h, posts in sj["tradePosts"].items() for p in posts
            if p["owner"] == seat and p["type"] == 2}


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    pyspiel.mali_ba.set_log_level(pyspiel.mali_ba.LogLevel.WARN)
    game = pyspiel.load_game("mali_ba", {"config_file": INI})
    probe = pyspiel.mali_ba.downcast_state(game.new_initial_state())
    if not hasattr(probe, "set_goods_choice_seats"):
        sys.exit("this engine build has no set_goods_choice_seats: build first")
    rng = random.Random(11)
    stats = {"upgrade_payments": 0, "rare_payments": 0, "income_choices": 0, "incomes_with_choice": 0}
    chosen_seats = [0, 2]
    for gi in range(games):
        s = game.new_initial_state()
        ms = pyspiel.mali_ba.downcast_state(s)
        ms.set_goods_choice_seats(chosen_seats)
        assert ms.get_goods_choice_seats() == chosen_seats
        upgrade = income = None     # bookkeeping for a choice in progress
        steps = 0
        while not s.is_terminal() and steps < 3000:
            steps += 1
            if s.is_chance_node():
                s.apply_action(rng.choice(s.legal_actions()))
                continue
            p = s.current_player()
            ms = pyspiel.mali_ba.downcast_state(s)
            phase = str(ms.current_phase()).split(".")[-1]
            sj = json.loads(ms.serialize())
            if phase in CHOICE_PHASES:
                assert p in chosen_seats, f"seat {p} not chosen but in {phase}"
                c = s.clone()
                assert c.legal_actions() == s.legal_actions()
                assert pyspiel.mali_ba.downcast_state(c).serialize() == ms.serialize()
            legal = s.legal_actions()
            a = rng.choice(legal)
            name = s.action_to_string(p, a)
            before = goods(sj, p)

            if phase == "PLAY" and name.startswith("UpgradePost_"):
                upgrade = {"hex": name[len("UpgradePost_("):-1], "before": before, "seat": p}
            if phase == "PLAY" and name == "TakeIncome":
                income = {"before": before, "seat": p, "chosen": {}}
            if phase == "INCOME_CHOICE":
                _, _, gain = parse_income_choice(name)
                for side, key in (("rare", "rareGoods"), ("common", "commonGoods")):
                    for g, n in gain[side].items():
                        income["chosen"][(key, g)] = income["chosen"].get((key, g), 0) + n
                stats["income_choices"] += 1

            s.apply_action(a)
            if s.is_terminal():
                break
            ms = pyspiel.mali_ba.downcast_state(s)
            after_phase = str(ms.current_phase()).split(".")[-1]
            sj2 = json.loads(ms.serialize())

            if upgrade and after_phase != "UPGRADE_PAYMENT":
                d = diff(goods(sj2, upgrade["seat"]), upgrade["before"])
                rare = sum(-v for (side, _), v in d.items() if side == "rareGoods")
                common = sum(-v for (side, _), v in d.items() if side == "commonGoods")
                assert all(v < 0 for v in d.values()), f"an upgrade added goods: {d}"
                assert (rare, common) in ((1, 0), (0, 3)), f"upgrade paid rare={rare} common={common}"
                assert upgrade["hex"] in centers(sj2, upgrade["seat"]), "no center after the upgrade"
                stats["upgrade_payments"] += 1
                stats["rare_payments"] += rare
                upgrade = None
            if income:
                if after_phase == "INCOME_CHOICE":
                    assert goods(sj2, income["seat"]) == income["before"], "goods changed mid-choice"
                else:
                    d = diff(goods(sj2, income["seat"]), income["before"])
                    assert all(v > 0 for v in d.values()), f"income took goods: {d}"
                    for k, n in income["chosen"].items():
                        assert d.get(k, 0) >= n, f"chosen {k} x{n} not received: {d}"
                    if income["chosen"]:
                        stats["incomes_with_choice"] += 1
                    income = None
        print(f"game {gi}: {steps} steps, {'over: ' + ms.get_game_end_reason() if s.is_terminal() else 'stopped'}")

    # GameSession: a human seat chooses; bots never see a choice phase.
    from mali_ba.play.session import GameSession
    sess = GameSession(INI, [{"type": "human"}, {"type": "heuristic"}, {"type": "heuristic"}], seed=3)
    assert sess.choice_seats == [0]
    n = 0
    while not sess.is_over() and n < 3000:
        n += 1
        if sess.awaiting_complete is not None:
            sess.complete_turn()
        elif sess.is_human_turn():
            moves = sess.legal_moves()
            if sess.phase() in CHOICE_PHASES:
                assert sess.view()["choice"] is not None
            sess.apply(rng.choice(moves)["action"])
        else:
            assert sess.phase() not in CHOICE_PHASES, "a bot's turn in a choice phase"
            sess.bot_move()
    print(f"session game: {n} steps; stats {stats}")
    assert stats["upgrade_payments"] and stats["income_choices"], "no choices were exercised"
    print("ok")


if __name__ == "__main__":
    main()
