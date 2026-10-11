"""How far each player is from the Timbuktu end condition (aux training target).

The engine ends the game when a player declares an active trade route whose hexes
include Timbuktu, a desert city (Agadez or Oudane), at least two other cities and a
coastal hex (mali_ba_state_core.cc, "Trade route Timbuktu to coast"). A route is any
set of 3-8 of the player's trading centers; nothing requires them to be adjacent
(mali_ba_state_trade.cc, FindPossibleTradeRoutes). So the condition is a checklist
of where the player has trading centers, and the distance to it is a count of
build steps:

    a hex where the player has a trading center   0 steps
    a hex where the player has a trading post     1 step  (upgrade)
    any other hex                                 2 steps (place a post, upgrade)

distance = cost(Timbuktu) + min cost(desert city)
           + min over two other cities o1, o2 and a coastal hex k of
             cost(o1) + cost(o2) + (0 if o1 or o2 is coastal, else cost(k))

It ranges 0..10 and is reported as a fraction of 10. It ignores what the rules may
block (goods needed for an upgrade, other players' posts, the shared-centers limit
between routes, declaring the route itself), so it is a training signal, not a rule.

Why (2026-10-10): the value head learned to anticipate rare-goods wins (whose
progress is in the goods planes) better than Timbuktu wins. The observation marks
every city alike, so the network has to work out from absolute board position which
city is Timbuktu, which are desert cities and which hexes are coastal. Predicting
this distance for every player pushes the shared trunk to learn exactly that.
"""

import json
import re

MAX_ROUTE_DISTANCE = 10
_DESERT_CITIES = ("Agadez", "Oudane")
_CENTER, _POST = 2, 1   # TradePostType in mali_ba_common.h


def _parse_hex(s):
    return tuple(int(v) for v in str(s).strip().strip("()").split(","))


def read_coastal_hexes(config_file):
    """The [Board] coastal_hexes list from the game ini, as (x, y, z) tuples."""
    with open(config_file) as f:
        for line in f:
            m = re.match(r"\s*coastal_hexes\s*=\s*(.*)", line)
            if m:
                return {_parse_hex(h) for h in m.group(1).split(":") if h.strip()}
    return set()


class RouteDistance:
    """Computes the distance for every player from a Mali-Ba state.

    Build once per game object (cities and coast are fixed), then call per state.
    """

    def __init__(self, game, config_file, num_players):
        cities = {c.name: _parse_hex(str(c.location)) for c in game.get_cities()}
        self.num_players = num_players
        self.timbuktu = cities.get("Timbuktu")
        self.desert = [cities[n] for n in _DESERT_CITIES if n in cities]
        self.others = [h for n, h in cities.items()
                       if n != "Timbuktu" and n not in _DESERT_CITIES]
        self.coast = read_coastal_hexes(config_file)
        self.enabled = bool(self.timbuktu and self.desert and len(self.others) >= 2
                            and self.coast)

    def from_json(self, state_json):
        """Distances (fractions of MAX_ROUTE_DISTANCE), one per player, from the
        state's serialize() JSON (a str or an already-parsed dict)."""
        if not self.enabled:
            return None
        data = json.loads(state_json) if isinstance(state_json, str) else state_json
        best = [dict() for _ in range(self.num_players)]   # hex -> 0 (center) or 1 (post)
        for key, posts in (data.get("tradePosts") or {}).items():
            hx = _parse_hex(key)
            for post in posts:
                p, t = post.get("owner"), post.get("type")
                if not isinstance(p, int) or not 0 <= p < self.num_players:
                    continue
                c = 0 if t == _CENTER else 1 if t == _POST else 2
                if c < best[p].get(hx, 2):
                    best[p][hx] = c
        return [self._distance(best[p]) / MAX_ROUTE_DISTANCE
                for p in range(self.num_players)]

    def from_state(self, mali_ba_state):
        """Same, from a downcast Mali_BaState."""
        return self.from_json(mali_ba_state.serialize())

    def _distance(self, cost_at):
        def cost(h):
            return cost_at.get(h, 2)
        coast_cost = min(cost(k) for k in self.coast)
        others = sorted(self.others, key=cost)
        pair = MAX_ROUTE_DISTANCE
        for i in range(len(others)):
            for j in range(i + 1, len(others)):
                o1, o2 = others[i], others[j]
                extra = 0 if (o1 in self.coast or o2 in self.coast) else coast_cost
                pair = min(pair, cost(o1) + cost(o2) + extra)
        return cost(self.timbuktu) + min(cost(d) for d in self.desert) + pair
