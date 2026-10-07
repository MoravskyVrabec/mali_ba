"""Win rates by starting token placement.

Actors log one line per game after token placement (from 2026-10-07):
    Actor 52, Game 1234: SETUP layout=Setup_40213 tokens P0=(x,y,z);(..);(..) P1=... P2=...
and the learner logs each result:
    Actor 52, Game 1234: FINISHED in 342 moves. ... Reason: '...'. ... Final Returns: [..]

Training actors place tokens uniformly at random, so these are randomised comparisons:
a difference in win rate between placements is caused by the placement, not by
stronger players choosing better spots. Culled games (~15%) never report a result
and are left out.

"Won" means the seat finished with the best return: the winner of a Timbuktu or
rare-goods game, or the score leader at a timeout. Seats differ in strength (first
mover), so each bucket is compared with what its seats would win on average
("expected"), and the gap is the placement effect.

Usage (from the mali_ba directory, with the usual PYTHONPATH):
    python analyze_placements.py /media/robp/UD/Projects/open_spiel/train_runD011.log [more logs]
"""

import argparse
import math
import os
import re
from collections import defaultdict
from itertools import combinations

RE_SETUP = re.compile(r'Game (\d+): SETUP layout=(\S+) tokens (.*)$')
RE_FINISHED = re.compile(r'Game (\d+): FINISHED in (\d+) moves\..*?Reason: \'([^\']*)\'.*?'
                         r'Final Returns: \[([^\]]+)\]')
RE_COORD = re.compile(r'\((-?\d+),(-?\d+),(-?\d+)\)')


def hex_dist(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2]))


def parse(paths):
    setups, results = {}, {}
    for path in paths:
        with open(path, errors='ignore') as f:
            for line in f:
                if 'SETUP layout=' in line:
                    m = RE_SETUP.search(line)
                    if m:
                        seats = {}
                        for part in m.group(3).split():
                            seat, _, coords = part.partition('=')
                            seats[int(seat[1:])] = [tuple(map(int, c)) for c in RE_COORD.findall(coords)]
                        setups[(path, m.group(1))] = seats
                elif 'FINISHED in' in line:
                    m = RE_FINISHED.search(line)
                    if m:
                        rets = [float(x) for x in m.group(4).split(',')]
                        reason = m.group(3)
                        kind = ('timbuktu' if 'Timbuktu' in reason else
                                'rare_goods' if 'Rare good' in reason else 'timeout')
                        results[(path, m.group(1))] = (rets, kind, int(m.group(2)))
    return setups, results


def features(tokens, cities):
    city_locs = [c[1] for c in cities]
    timbuktu = next((c[1] for c in cities if c[0] == 'Timbuktu'), None)
    near = [min(hex_dist(t, c) for c in city_locs) for t in tokens]
    f = {
        'nearest city, closest token': min(near),
        'nearest city, average of 3': sum(near) / len(near),
        'tokens on a city hex': sum(1 for d in near if d == 0),
        'spread (avg distance between own tokens)':
            sum(hex_dist(a, b) for a, b in combinations(tokens, 2)) / 3,
        'distinct cultures of nearest cities': len({
            min(cities, key=lambda c: hex_dist(t, c[1]))[2] for t in tokens}),
    }
    if timbuktu is not None:
        f['Timbuktu, closest token'] = min(hex_dist(t, timbuktu) for t in tokens)
    return f


def main():
    ap = argparse.ArgumentParser(description='Win rates by starting token placement')
    ap.add_argument('logs', nargs='+')
    ap.add_argument('--config_file', default='mali_ba.ini')
    ap.add_argument('--min_n', type=int, default=150,
                    help='hide buckets with fewer seat-games than this (default 150)')
    args = ap.parse_args()

    import pyspiel
    game = pyspiel.load_game('mali_ba', {'config_file': os.path.abspath(args.config_file)})
    cities = [(c.name, (c.location.x, c.location.y, c.location.z), c.culture)
              for c in game.get_cities()]

    setups, results = parse(args.logs)
    games = [(setups[k], results[k]) for k in setups if k in results]
    print(f'{len(setups)} games with a SETUP line, {len(results)} results, '
          f'{len(games)} matched (unmatched setups are mostly culled games).')
    if not games:
        return

    # Seat baselines, so seat strength does not masquerade as a placement effect.
    nseats = len(games[0][0])
    seat_wins = defaultdict(int)
    for seats, (rets, _, _) in games:
        seat_wins[rets.index(max(rets))] += 1
    seat_rate = {s: seat_wins[s] / len(games) for s in range(nseats)}
    print('Seat win shares: ' + '  '.join(f'P{s} {100 * seat_rate[s]:.1f}%' for s in range(nseats)))

    rows = []   # (features, seat, won, outright_win, kind)
    for seats, (rets, kind, _) in games:
        w = rets.index(max(rets))
        for s, toks in seats.items():
            if len(toks) == 3:
                rows.append((features(toks, cities), s, w == s, w == s and kind != 'timeout', kind))
    names = list(rows[0][0])

    for name in names:
        vals = sorted({r[0][name] for r in rows})
        if len(vals) < 2:
            print(f'\n  {name}: always {vals[0]:g} (placement rules fix it), skipped')
            continue
        if len(vals) > 6:   # continuous: split into quintiles
            cut = sorted(r[0][name] for r in rows)
            qs = [cut[int(len(cut) * q / 5)] for q in range(1, 5)]
            def bucket(v, qs=qs):
                i = sum(v > q for q in qs)
                return i
            labels = {}
            for r in rows:
                b = bucket(r[0][name]); lo_hi = labels.setdefault(b, [r[0][name], r[0][name]])
                lo_hi[0] = min(lo_hi[0], r[0][name]); lo_hi[1] = max(lo_hi[1], r[0][name])
            label = {b: f'{v[0]:.1f}-{v[1]:.1f}' for b, v in labels.items()}
        else:
            bucket = lambda v: v
            label = {v: f'{v:g}' for v in vals}
        groups = defaultdict(list)
        for r in rows:
            groups[bucket(r[0][name])].append(r)
        print(f'\n  {name}')
        print(f'    {"value":>10} {"seat-games":>10} {"won":>7} {"expected":>8} {"effect":>7} '
              f'{"±2se":>6} {"outright":>8} {"Tim/rare share of their wins":>28}')
        for b in sorted(groups):
            g = groups[b]
            n = len(g)
            if n < args.min_n:
                continue
            won = sum(r[2] for r in g) / n
            exp = sum(seat_rate[r[1]] for r in g) / n
            se = math.sqrt(max(exp * (1 - exp), 1e-9) / n)
            outright = sum(r[3] for r in g) / n
            wins = [r for r in g if r[2]]
            tim = sum(r[4] == 'timbuktu' for r in wins); rare = sum(r[4] == 'rare_goods' for r in wins)
            mix = f'{100 * tim / len(wins):.0f}% / {100 * rare / len(wins):.0f}%' if wins else '-'
            flag = '  *' if abs(won - exp) > 2 * se else ''
            print(f'    {label[b]:>10} {n:>10} {100 * won:6.1f}% {100 * exp:7.1f}% '
                  f'{100 * (won - exp):+6.1f} {200 * se:5.1f} {100 * outright:7.1f}% {mix:>28}{flag}')
    print('\n  effect = won - expected for those seats; * = beyond 2 standard errors.')
    print('  Rows are seat-games (3 per game), so a bucket and its complement are not independent.')
    print('  The measures overlap (e.g. widely spread tokens tend to sit farther from Timbuktu),')
    print('  so an effect in one can partly be another\'s; compare them before concluding.')


if __name__ == '__main__':
    main()
