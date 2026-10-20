"""Describe engine actions for display.

The engine splits a turn into single decisions and names each one (PlaceToken_(x,y,z),
StartMancala_(x,y,z), MancalaDir_k, PlacePost, PayGood_k, UpgradePost_(x,y,z),
DeclareRoute_k, TakeIncome, Pass, and for seats that choose their own goods PayRare_k and
IncomeChoice_<center|post>_(x,y,z)_<goods>). describe_action() turns one into a dict a
front end can draw without knowing any rules:

    {"action": 812, "name": "MancalaDir_2", "kind": "mancala_step",
     "hex": [0, 1, -1], "label": "Continue the mancala to (0,1,-1)"}

Kinds: place_token, start_mancala, mancala_step, place_post, pay_good, pay_rare, upgrade,
declare_route, take_income, income_choice, pass. "hex" is set for actions tied to one
hex (a front end highlights it); "good" for payments; "route_index" for route
declarations (GameSession adds each route's "hexes" and "cities" by applying it to a copy
of the state); an income choice has "source" (center or post), "source_hex" (not "hex":
it is a list choice, not a hex click) and "gain" ({"rare": {...}, "common": {...}}).
"""
import re

# Same order as kHexDirections in open_spiel/games/mali_ba/hex_grid.h.
HEX_DIRECTIONS = ((1, -1, 0), (1, 0, -1), (0, 1, -1), (-1, 1, 0), (-1, 0, 1), (0, -1, 1))
DIRECTION_NAMES = ("east", "southeast", "southwest", "west", "northwest", "northeast")

_HEX_IN_NAME = re.compile(r"\((-?\d+),(-?\d+),(-?\d+)\)")


def parse_hex(text):
    """'(1,-2,1)' or '1,-2,1' -> (1, -2, 1); None if there is none."""
    if text is None:
        return None
    m = _HEX_IN_NAME.search(text) or re.search(r"(-?\d+),(-?\d+),(-?\d+)", text)
    return tuple(int(v) for v in m.groups()) if m else None


def common_goods_list(game):
    """Common goods in the engine's PayGood_k order: the cities' common goods, unique,
    sorted alphabetically (GoodsManager in mali_ba_game.cc)."""
    import pyspiel
    cities = pyspiel.mali_ba.downcast_game(game).get_cities()
    return sorted({c.common_good for c in cities})


def rare_goods_list(game):
    """Rare goods in the engine's PayRare_k order (sorted alphabetically, as GoodsManager)."""
    import pyspiel
    cities = pyspiel.mali_ba.downcast_game(game).get_cities()
    return sorted({c.rare_good for c in cities})


def _goods_text(goods):
    """{"Millet": 2} -> "2 Millet"; {"Millet": 1, "Kola nuts": 1} -> "Kola nuts and Millet"."""
    parts = [f"{n} {g}" if n > 1 else g for g, n in sorted(goods.items())]
    return " and ".join(parts)


def parse_income_choice(name):
    """'IncomeChoice_center_(1,-2,1)_rare=Kora' -> ("center", (1,-2,1), {"rare": {"Kora": 1},
    "common": {}}); None if the name has another form."""
    m = re.match(r"IncomeChoice_(center|post)_\((-?\d+),(-?\d+),(-?\d+)\)_(.*)$", name)
    if not m:
        return None
    gain = {"rare": {}, "common": {}}
    for part in m.group(5).split(";"):
        kind, _, goods = part.partition("=")
        if kind in gain:
            for g in goods.split(","):
                if g:
                    gain[kind][g] = gain[kind].get(g, 0) + 1
    return m.group(1), tuple(int(v) for v in m.group(2, 3, 4)), gain


def _hex_str(h):
    return f"({h[0]},{h[1]},{h[2]})"


def describe_action(state, action, state_json, goods_list, rare_list=()):
    """One legal action -> description dict (see module docstring).

    state: the pyspiel state; state_json: its parsed serialize() output (for the
    mid-turn mancala position); goods_list: common_goods_list(game); rare_list:
    rare_goods_list(game).
    """
    player = state.current_player()
    name = state.action_to_string(player, action)
    d = {"action": int(action), "name": name}
    phase = str(state.current_phase()) if hasattr(state, "current_phase") else ""

    if name.startswith("PlaceToken_"):
        h = parse_hex(name)
        d.update(kind="place_token", hex=list(h), label=f"Place a token on {_hex_str(h)}")
    elif name.startswith("StartMancala_"):
        h = parse_hex(name)
        d.update(kind="start_mancala", hex=list(h), label=f"Start a mancala from {_hex_str(h)}")
    elif name.startswith("MancalaDir_"):
        k = int(name.split("_")[1])
        here = parse_hex((state_json.get("midTurnState") or {}).get("mancalaHex"))
        to = tuple(a + b for a, b in zip(here, HEX_DIRECTIONS[k])) if here else None
        d.update(kind="mancala_step", direction=DIRECTION_NAMES[k],
                 hex=list(to) if to else None,
                 label=(f"Continue the mancala to {_hex_str(to)}" if to
                        else f"Continue the mancala {DIRECTION_NAMES[k]}"))
    elif name == "PlacePost":
        d.update(kind="place_post", label="Place a trading post here")
    elif name.startswith("PayGood_"):
        k = int(name.split("_")[1])
        good = goods_list[k] if 0 <= k < len(goods_list) else f"good {k}"
        d.update(kind="pay_good", good=good, label=f"Pay with {good}")
    elif name.startswith("PayRare_"):
        k = int(name.split("_")[1])
        good = rare_list[k] if 0 <= k < len(rare_list) else f"rare good {k}"
        d.update(kind="pay_rare", good=good, label=f"Pay with {good} (rare)")
    elif name.startswith("IncomeChoice_"):
        parsed = parse_income_choice(name)
        if parsed is None:
            d.update(kind="income_choice", label=name)
        else:
            source, h, gain = parsed
            what = " and ".join(t for t in (
                _goods_text(gain["rare"]) + (" (rare)" if gain["rare"] else ""),
                _goods_text(gain["common"])) if t)
            d.update(kind="income_choice", source=source, source_hex=list(h), gain=gain,
                     label=f"Take {what}")
    elif name.startswith("UpgradePost_"):
        h = parse_hex(name)
        d.update(kind="upgrade", hex=list(h), label=f"Upgrade the post at {_hex_str(h)} to a center")
    elif name.startswith("DeclareRoute_"):
        k = int(name.split("_")[1])
        d.update(kind="declare_route", route_index=k, hexes=None,
                 label=f"Declare trade route option {k + 1}")
    elif name == "TakeIncome":
        d.update(kind="take_income", label="Take income")
    elif name == "Pass":
        if "MANCALA" in phase:
            label = "Stop here (no further move possible)"
        elif "OPTIONAL_POST" in phase:
            label = "Don't place a post"
        elif "OPTIONAL_ROUTE" in phase or "ROUTE" in phase:
            label = "Don't declare a trade route"
        else:
            label = "Pass"
        d.update(kind="pass", label=label)
    else:
        d.update(kind="other", label=name)
    return d
