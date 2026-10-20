"""Checks goods.py against the engine: set points must equal the engine's "Unique goods
sets" score for every player, over random games.

    PYTHONPATH=build/python:open_spiel/python/games:. python -m mali_ba.play.test_goods [games]
"""
import json
import os
import random
import re
import sys

import pyspiel

from mali_ba.play.goods import GoodsSummarizer, read_rules, set_points
from mali_ba.play.session import board_config

INI = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mali_ba.ini")
RE_SETS = re.compile(r"Unique goods sets:\s+(-?[\d.]+) pts")


def main():
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    pyspiel.mali_ba.set_log_level(pyspiel.mali_ba.LogLevel.WARN)
    game = pyspiel.load_game("mali_ba", {"config_file": INI})
    cfg = board_config(game, INI)
    summ = GoodsSummarizer(cfg["cities"], cfg["regions"], read_rules(INI))
    rng = random.Random(1)
    checked = 0
    for g in range(games):
        state = game.new_initial_state()
        moves = 0
        while not state.is_terminal() and moves < 600:
            state.apply_action(rng.choice(state.legal_actions()))
            moves += 1
            if moves % 25 or state.is_chance_node():
                continue
            ms = pyspiel.mali_ba.downcast_state(state)
            engine = [float(x) for x in RE_SETS.findall(ms.get_score_breakdown_string())]
            sj = json.loads(ms.serialize())
            mine = [p["set_points"] for p in summ.summary(sj)]
            assert engine == mine, f"game {g} move {moves}: engine {engine}, goods.py {mine}"
            for p in summ.summary(sj):
                counts = {}
                for d in p["goods"]:
                    counts[d["name"]] = counts.get(d["name"], 0) + d["count"]
                for d in p["goods"]:
                    less = dict(counts)
                    less[d["name"]] -= 1
                    assert d["spend_cost"] == p["set_points"] - set_points(less)
            checked += 1
    print(f"ok: {checked} positions in {games} random games; set points match the engine")


if __name__ == "__main__":
    main()
