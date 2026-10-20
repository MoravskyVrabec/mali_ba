"""Each player's goods, as a front end shows them: what they hold, the sets those goods
make and what the sets score, and which regions their rare goods come from.

goods_summary() works from the board JSON alone (serialize()); it needs no engine
calls, so it costs nothing per view. The set scoring mirrors the engine's
(GetScoreBreakdownString, mali_ba_state_core.cc): common and rare goods count together;
each pass takes one of every distinct good still held and scores the size of that set
(1, 3, 6, 11, 19, 30, 45, 60, 75, 90, 110 points for 1-11 kinds, 20 more per kind after
11), until no goods are left. play/tests check it against the engine.

    {"seat": 0, "total": 9, "kinds": 6, "set_points": 44,
     "sets": [{"size": 6, "points": 30}, {"size": 2, "points": 3}, ...],
     "goods": [{"name": "Kora", "kind": "rare", "count": 1, "city": "Sikasso",
                "culture": "Senoufo", "region": 3, "spend_cost": 19}, ...],
     "regions": {"have": 3, "needed": 5, "list": [{"id": 1, "name": "Desert West",
                 "have": true, "goods": ["Silver cross"]}, ...]},
     "routes": {"active": 1, "needed": 2}}

"spend_cost" is what paying one of that good would cost in set points now (0 for a
duplicate that only sits in a small set's top layer, more for the last of a kind).
"""
import configparser

SET_POINTS = {1: 1, 2: 3, 3: 6, 4: 11, 5: 19, 6: 30, 7: 45, 8: 60, 9: 75, 10: 90, 11: 110}
SET_BONUS_PER_KIND = 20      # each kind beyond 11 in one set


def set_score(size):
    if size <= 0:
        return 0
    if size > 11:
        return SET_POINTS[11] + (size - 11) * SET_BONUS_PER_KIND
    return SET_POINTS[size]


def sets_of(counts):
    """{good: count} -> [set sizes], largest first (one set per pass)."""
    left = {g: c for g, c in counts.items() if c > 0}
    sizes = []
    while left:
        sizes.append(len(left))
        left = {g: c - 1 for g, c in left.items() if c > 1}
    return sizes


def set_points(counts):
    return sum(set_score(s) for s in sets_of(counts))


def read_rules(config_file):
    """The [Rules] values the summary needs, from the ini the game was loaded with."""
    cp = configparser.ConfigParser(inline_comment_prefixes=("#", ";"), strict=False,
                                   interpolation=None)
    try:
        cp.read(config_file)
    except (configparser.Error, OSError):
        pass

    def get(key, default):
        try:
            return cp.get("Rules", key).strip()
        except (configparser.Error, KeyError):
            return default
    each_region = get("end_game_cond_rare_good_each_region", "false").lower() == "true"
    return {"regions_needed": int(get("end_game_cond_rare_good_num_regions", "5")) if each_region else None,
            "routes_needed": int(get("end_game_req_numroutes", "0"))}


def regions_from_ini(config_file):
    """[{"id", "name", "hexes"}] from the ini's custom_hexesN / regionN lines: the same
    regions the engine reads, for engines built before the region bindings."""
    cp = configparser.ConfigParser(inline_comment_prefixes=("#",), strict=False,
                                   interpolation=None)
    try:
        cp.read(config_file)
    except (configparser.Error, OSError):
        return []
    out = []
    for n in range(1, 21):
        try:
            raw = cp.get("Board", f"custom_hexes{n}")
        except (configparser.Error, KeyError):
            continue
        hexes = sorted([int(v) for v in t.split(",")] for t in raw.split(":") if t.strip())
        try:
            name = cp.get("Regions", f"region{n}").strip()
        except (configparser.Error, KeyError):
            name = f"Region {n}"
        out.append({"id": n, "name": name, "hexes": hexes})
    return out


class GoodsSummarizer:
    """Built once per game from board_config(); summary(state_json) per view."""

    def __init__(self, cities, regions, rules):
        region_at = {tuple(h): r["id"] for r in regions for h in r["hexes"]}
        self.regions = [{"id": r["id"], "name": r["name"]} for r in regions]
        self.rules = rules
        self.source = {}             # good name -> (kind, city, culture, region id)
        for c in cities:
            loc = tuple(c["location"])
            self.source[c["rare_good"]] = ("rare", c["name"], c["culture"], region_at.get(loc))
            self.source.setdefault(c["common_good"], ("common", c["name"], c["culture"], None))

    def summary(self, state_json):
        common = state_json.get("commonGoods") or []
        rare = state_json.get("rareGoods") or []
        routes = state_json.get("tradeRoutes") or []
        return [self._one(i, common[i] if i < len(common) else {},
                          rare[i] if i < len(rare) else {}, routes)
                for i in range(max(len(common), len(rare)))]

    def _one(self, seat, common, rare, routes):
        counts = {}
        for g, c in list(common.items()) + list(rare.items()):
            if c > 0:
                counts[g] = counts.get(g, 0) + c
        points = set_points(counts)
        goods = []
        for kind, held in (("rare", rare), ("common", common)):
            for g, c in sorted(held.items()):
                if c <= 0:
                    continue
                _, city, culture, region = self.source.get(g, (kind, None, None, None))
                less = dict(counts)
                less[g] -= 1
                goods.append({"name": g, "kind": kind, "count": c, "city": city,
                              "culture": culture, "region": region if kind == "rare" else None,
                              "spend_cost": points - set_points(less)})
        rare_by_region = {}
        for g in goods:
            if g["kind"] == "rare" and g["region"] is not None:
                rare_by_region.setdefault(g["region"], []).append(g["name"])
        active = sum(1 for r in routes if r.get("owner") == seat and r.get("active", True))
        return {
            "seat": seat,
            "total": sum(counts.values()),
            "kinds": len(counts),
            "set_points": points,
            "sets": [{"size": s, "points": set_score(s)} for s in sets_of(counts)],
            "goods": goods,
            "regions": {"have": len(rare_by_region), "needed": self.rules.get("regions_needed"),
                        "list": [{"id": r["id"], "name": r["name"], "have": r["id"] in rare_by_region,
                                  "goods": rare_by_region.get(r["id"], [])} for r in self.regions]},
            "routes": {"active": active, "needed": self.rules.get("routes_needed")},
        }


def annotate_choices(moves, state_json, seat):
    """Adds set-point effects to a seat's goods choices, in place: "spend_cost" on
    payments (pay_good, pay_rare) and "set_gain" on income choices, each also appended to
    the label. Income choices count the goods already chosen this income (held back by
    the engine until the last choice, midTurnState.incomePending*)."""
    if seat is None or not any(m["kind"] in ("pay_good", "pay_rare", "income_choice") for m in moves):
        return moves
    counts = {}
    for side in ("commonGoods", "rareGoods"):
        held = (state_json.get(side) or [])
        for g, c in (held[seat] if seat < len(held) else {}).items():
            if c > 0:
                counts[g] = counts.get(g, 0) + c
    mid = state_json.get("midTurnState") or {}
    pending = dict(counts)
    for side in ("incomePendingCommon", "incomePendingRare"):
        for g, c in (mid.get(side) or {}).items():
            pending[g] = pending.get(g, 0) + c
    now, base = set_points(counts), set_points(pending)
    for m in moves:
        if m["kind"] in ("pay_good", "pay_rare") and counts.get(m.get("good"), 0) > 0:
            less = dict(counts)
            less[m["good"]] -= 1
            m["spend_cost"] = now - set_points(less)
            m["label"] += f"  (−{m['spend_cost']} set pts)" if m["spend_cost"] else "  (no set points lost)"
        elif m["kind"] == "income_choice" and m.get("gain"):
            more = dict(pending)
            for side in ("rare", "common"):
                for g, c in m["gain"][side].items():
                    more[g] = more.get(g, 0) + c
            m["set_gain"] = set_points(more) - base
            m["label"] += f"  (+{m['set_gain']} set pts)"
    return moves
