#!/usr/bin/env python3
"""Analyze a mali_ba training log and print statistics and trends."""

import re
import sys
import argparse
from collections import defaultdict, Counter
from datetime import datetime


# ── regex patterns ────────────────────────────────────────────────────────────

RE_FINISHED = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] (Heuristic )?Actor (\d+), Game (\d+): '
    r'FINISHED in (\d+) moves\. '
    r'(?:heuristic_weight=[\d.]+\. )?'
    r'Winner: ([^.]+)\. Reason: \'([^\']+)\'\. '
    r'Triggered by: Player ([-\d]+)\. Final Returns: \[([^\]]+)\]'
    r'(?:\.? ?Intermediate Rewards: \[([^\]]+)\], Rewarded Steps: (\d+))?'
)
RE_RECEIVED = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] LEARNER \((Bootstrap|MCTS)\) RECEIVED GAME '
    r'#(\d+)/(\d+)\. Length: (\d+) moves\. Returns: \[([^\]]+)\]'
)
RE_LOSS = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] Training: Total Loss=([\d.]+) '
    r'\(Policy=([\d.]+), Value=([\d.]+)\)'
)
RE_PASS_ONLY = re.compile(
    r"MCTS top visits: \['Pass: \d+ visits'\]"
)
RE_BUFFER_STATE = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] Trainer processed \d+ new experiences\. '
    r'Bootstrap: (\d+)\s+MCTS-natural: (\d+)\s+MCTS-nearwin: (\d+)'
    r'(?:\s+MCTS-raregoods: (\d+))?'
)
RE_HEURISTIC_GUIDANCE = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] Actor (\d+), Game (\d+): '
    r'heuristic_guidance_weight=([\d.]+)'
)
RE_TRAINER_LOSS = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] Trainer: Training successful, loss = ([\d.]+)'
)
RE_ADAPTIVE_FRACTION = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] Adaptive fraction: .*?mcts_fraction \u2192 ([\d.]+)'
)
RE_TOP_ACTION = re.compile(
    r"MCTS top visits: \['([^:]+): \d+ visits'"
)
RE_EARLY_TERM_REASON = re.compile(
    r'\d{8}-\d{6} \[INFO\] \[Python:0\] Actor (\d+), Game (\d+), Move (\d+): '
    r'Early termination — (.+)\.'
)
RE_EARLY_TERM_DISCARDED = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] Actor (\d+), Game (\d+): '
    r'Discarded after (\d+) moves \(early termination\)\.'
)
RE_NO_KILL_GAME = re.compile(
    r'\d{8}-\d{6} \[INFO\] \[Python:0\] Actor (\d+), Game (\d+): '
    r'NO-KILL game \(thresh=([\d.]+)\)\.'
)
RE_NO_KILL_WOULD = re.compile(
    r'\d{8}-\d{6} \[INFO\] \[Python:0\] Actor (\d+), Game (\d+), Move (\d+): '
    r'NO-KILL — would have terminated: (.+)\.'
)
RE_OVERSAMPLE_THRESH = re.compile(
    r'Oversampling threshold set to (\d+)|Oversampling disabled'
)
RE_SEARCH_COST = re.compile(
    r'(\d{8}-\d{6}) \[INFO\] \[Python:0\] Actor (\d+), Game (\d+): SEARCH COST — '
    r'moves=(\d+) \(full=(\d+), fast=(\d+)\) sims=(\d+) '
    r'nn_evals=(\d+) cache_hits=(\d+) hit_rate=([\d.]+)'
)
RE_VALUE_CHECK = re.compile(
    r'\[Python:\d+\] Actor (\d+), Game (\d+), Move (\d+): '
    r'Value check: \[([^\]]+)\] '
    r'\((?:thresh=([-\d.]+)(?:, (?:exempt=(\w+)|(clear winner)))?|monitoring)\)'
)


def parse_time(t):
    return datetime.strptime(t, '%Y%m%d-%H%M%S')


def parse_returns(s):
    results = []
    for x in s.split(','):
        x = x.strip()
        try:
            results.append(float(x))
        except ValueError:
            break  # Stop at first malformed token (log corruption)
    return results if results else [0.0]


# ── main parse ────────────────────────────────────────────────────────────────

def parse_log(path):
    games = []          # list of dicts from FINISHED lines
    received = []       # list of dicts from RECEIVED GAME lines
    losses = []         # (game_num, total, policy, value)
    trainer_losses = [] # (timestamp, loss) from "Training successful" lines
    pass_only_count = 0
    near_win_retained_count = 0
    near_win_bootstrap_count = 0
    top_action_counter = Counter()
    first_time = None
    last_time = None
    last_finished_is_win = None        # carry is_win from FINISHED to next RECEIVED line
    last_finished_actor = None         # carry actor id from FINISHED to next RECEIVED line
    last_finished_inter_rewards = None # carry inter_rewards from FINISHED to next RECEIVED line
    last_finished_rewarded_steps = None
    last_finished_game_idx = None      # index into games[] of the last FINISHED entry, so the
                                       # subsequent RECEIVED line can correct its is_bootstrap flag
                                       # (MCTS actors using heuristic guidance log "Heuristic Actor"
                                       # just like bootstrap actors, so the RECEIVED phase is the
                                       # authoritative source for bootstrap vs MCTS classification)
    last_buffer_state = None      # most recent trainer buffer pool sizes
    last_adaptive_fraction = None # most recent adaptive mcts_fraction value
    recent_heuristic_guidance = [] # last N heuristic_guidance_weight log lines
    early_terminations = []       # list of dicts for early-terminated games
    _pending_et = {}              # (actor, game_n) -> (move, reason_text)
    value_checks = []             # list of dicts from Value check lines
    no_kill_games = {}            # (actor, game_n) -> list of would-have-terminated events
    oversample_threshold = None   # set from log; None means log predates this feature
    search_costs = []             # per-game search cost from SEARCH COST lines
                                  # (absent in logs predating playout cap randomization)

    with open(path) as f:
        for line in f:
            t_match = re.match(r'(\d{8}-\d{6})', line)
            if t_match:
                t = parse_time(t_match.group(1))
                if first_time is None:
                    first_time = t
                last_time = t

            m = RE_FINISHED.search(line)
            if m:
                ts, heuristic_prefix, actor, game_n, moves, winner, reason, trigger, returns_s, \
                    inter_rewards_s, rewarded_steps_s = m.groups()
                returns = parse_returns(returns_s)
                inter_rewards = parse_returns(inter_rewards_s) if inter_rewards_s else None
                rewarded_steps = int(rewarded_steps_s) if rewarded_steps_s else None
                is_win = reason != 'Max game length reached'
                is_bootstrap = heuristic_prefix is not None
                last_finished_is_win = is_win
                last_finished_actor = int(actor)
                last_finished_inter_rewards = inter_rewards
                last_finished_rewarded_steps = rewarded_steps
                last_finished_game_idx = len(games)
                games.append({
                    'time': parse_time(ts),
                    'actor': int(actor),
                    'game_n': int(game_n),
                    'moves': int(moves),
                    'winner': winner,
                    'reason': reason,
                    'trigger': int(trigger),
                    'returns': returns,
                    'inter_rewards': inter_rewards,
                    'rewarded_steps': rewarded_steps,
                    'is_win': is_win,
                    'is_remote': int(actor) >= 100000,
                    'is_bootstrap': is_bootstrap,
                })
                continue

            m = RE_SEARCH_COST.search(line)
            if m:
                ts, actor, game_n, moves, full, fast, sims, nn_evals, hits, rate = m.groups()
                search_costs.append({
                    'time':       parse_time(ts),
                    'actor':      int(actor),
                    'game_n':     int(game_n),
                    'moves':      int(moves),
                    'full_moves': int(full),
                    'fast_moves': int(fast),
                    'sims':       int(sims),
                    'nn_evals':   int(nn_evals),
                    'cache_hits': int(hits),
                    'hit_rate':   float(rate),
                    'is_remote':  int(actor) >= 100000,
                })
                continue

            m = RE_RECEIVED.search(line)
            if m:
                ts, phase, game_num, total, length, returns_s = m.groups()
                received.append({
                    'time': parse_time(ts),
                    'phase': phase,
                    'game_num': int(game_num),
                    'total': int(total),
                    'length': int(length),
                    'returns': parse_returns(returns_s),
                    'is_win': last_finished_is_win,
                    'actor': last_finished_actor,
                    'inter_rewards': last_finished_inter_rewards,
                    'rewarded_steps': last_finished_rewarded_steps,
                })
                # Use the RECEIVED phase as the authoritative bootstrap/MCTS classifier.
                # MCTS actors with heuristic guidance log "Heuristic Actor" in their
                # FINISHED lines (same as bootstrap actors), so the prefix alone is
                # unreliable. Correct the preceding FINISHED entry now that we know the phase.
                if last_finished_game_idx is not None:
                    games[last_finished_game_idx]['is_bootstrap'] = (phase == 'Bootstrap')
                last_finished_game_idx = None
                last_finished_is_win = None
                last_finished_actor = None
                last_finished_inter_rewards = None
                last_finished_rewarded_steps = None
                continue

            m = RE_LOSS.search(line)
            if m:
                ts, total, policy, value = m.groups()
                game_num = received[-1]['game_num'] if received else 0
                losses.append((game_num, float(total), float(policy), float(value)))
                continue

            m = RE_TRAINER_LOSS.search(line)
            if m:
                ts, loss = m.groups()
                trainer_losses.append((parse_time(ts), float(loss)))
                continue

            if RE_PASS_ONLY.search(line):
                pass_only_count += 1
                continue

            if "Near-win timeout game retained" in line:
                near_win_retained_count += 1
                continue

            if "Near-win bootstrap timeout game retained" in line:
                near_win_bootstrap_count += 1
                continue

            m = RE_TOP_ACTION.search(line)
            if m:
                top_action_counter[m.group(1)] += 1
                continue

            m = RE_BUFFER_STATE.search(line)
            if m:
                last_buffer_state = {
                    'time': parse_time(m.group(1)),
                    'bootstrap': int(m.group(2)),
                    'mcts_natural': int(m.group(3)),
                    'mcts_nearwin': int(m.group(4)),
                    'mcts_raregoods': int(m.group(5)) if m.group(5) else None,
                }
                continue

            m = RE_HEURISTIC_GUIDANCE.search(line)
            if m:
                recent_heuristic_guidance.append({
                    'time': parse_time(m.group(1)),
                    'actor': int(m.group(2)),
                    'game': int(m.group(3)),
                    'weight': float(m.group(4)),
                })
                if len(recent_heuristic_guidance) > 10:
                    recent_heuristic_guidance.pop(0)
                continue

            m = RE_ADAPTIVE_FRACTION.search(line)
            if m:
                last_adaptive_fraction = float(m.group(2))
                continue

            m = RE_OVERSAMPLE_THRESH.search(line)
            if m:
                oversample_threshold = int(m.group(1)) if m.group(1) else 0
                continue

            m = RE_NO_KILL_GAME.search(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                no_kill_games.setdefault(key, [])
                continue

            m = RE_NO_KILL_WOULD.search(line)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                no_kill_games.setdefault(key, []).append({
                    'move':   int(m.group(3)),
                    'reason': m.group(4),
                })
                continue

            m = RE_EARLY_TERM_REASON.search(line)
            if m:
                actor, game_n, move, reason = m.group(1), m.group(2), m.group(3), m.group(4)
                _pending_et[(int(actor), int(game_n))] = (int(move), reason)
                continue

            m = RE_EARLY_TERM_DISCARDED.search(line)
            if m:
                ts, actor, game_n, move = m.group(1), m.group(2), m.group(3), m.group(4)
                key = (int(actor), int(game_n))
                reason_move, reason_text = _pending_et.pop(key, (int(move), 'unknown'))
                early_terminations.append({
                    'time': parse_time(ts),
                    'actor': int(actor),
                    'game_n': int(game_n),
                    'move': reason_move,
                    'reason': reason_text,
                })
                continue

            m = RE_VALUE_CHECK.search(line)
            if m:
                vals = [float(v.strip().strip("'")) for v in m.group(4).split(',')]
                # group(6): new format "exempt=<word>" → captures the word
                # group(7): old format "clear winner" → normalize to "clear_winner"
                exempt = m.group(6) or ('clear_winner' if m.group(7) else None)
                value_checks.append({
                    'actor':  int(m.group(1)),
                    'game_n': int(m.group(2)),
                    'move':   int(m.group(3)),
                    'p0': vals[0] if len(vals) > 0 else None,
                    'p1': vals[1] if len(vals) > 1 else None,
                    'p2': vals[2] if len(vals) > 2 else None,
                    'thresh': float(m.group(5)) if m.group(5) else None,
                    'exempt': exempt,
                })
                continue

    return {
        'games': games,
        'received': received,
        'losses': losses,
        'pass_only_count': pass_only_count,
        'near_win_retained_count': near_win_retained_count,
        'near_win_bootstrap_count': near_win_bootstrap_count,
        'top_action_counter': top_action_counter,
        'first_time': first_time,
        'last_time': last_time,
        'last_buffer_state': last_buffer_state,
        'last_adaptive_fraction': last_adaptive_fraction,
        'recent_heuristic_guidance': recent_heuristic_guidance,
        'early_terminations': early_terminations,
        'trainer_losses': trainer_losses,
        'value_checks': value_checks,
        'no_kill_games': no_kill_games,
        'oversample_threshold': oversample_threshold,
        'search_costs': search_costs,
    }


# ── display helpers ───────────────────────────────────────────────────────────

def bar(value, max_value, width=30):
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    filled = max(0, min(width, filled))
    return '█' * filled + '░' * (width - filled)


def pct(n, d):
    return f'{100*n/d:.1f}%' if d > 0 else 'n/a'


def section(title):
    print(f'\n{"─" * 60}')
    print(f'  {title}')
    print(f'{"─" * 60}')


# ── value head trajectory ─────────────────────────────────────────────────────

def print_value_trajectory(mcts_games, value_checks):
    """Average winner value at each 20-move checkpoint, split by win condition."""
    if not value_checks:
        return

    section('VALUE HEAD TRAJECTORY (by game outcome)')

    game_lookup = {(g['actor'], g['game_n']): g for g in mcts_games}

    # Accumulate per-move winner values for each category
    cat_moves = {'timbuktu': defaultdict(list),
                 'rare':     defaultdict(list),
                 'timeout':  defaultdict(list)}
    cat_counts = {'timbuktu': 0, 'rare': 0, 'timeout': 0}

    # Group checks by game first so we can look up the outcome once per game
    game_checks = defaultdict(list)
    for vc in value_checks:
        game_checks[(vc['actor'], vc['game_n'])].append(vc)

    for key, checks in game_checks.items():
        g = game_lookup.get(key)
        if g is None:
            continue
        returns = g.get('returns', [])
        widx = returns.index(max(returns)) if returns else -1

        if g['is_win'] and 'Timbuktu' in g.get('reason', ''):
            cat = 'timbuktu'
        elif g['is_win']:
            cat = 'rare'
        else:
            cat = 'timeout'
        cat_counts[cat] += 1

        for vc in checks:
            pvals = [vc.get('p0'), vc.get('p1'), vc.get('p2')]
            pvals = [v for v in pvals if v is not None]
            if not pvals:
                continue
            if cat == 'timeout':
                cat_moves[cat][vc['move']].append(max(pvals))
            elif widx >= 0 and widx < len([vc.get('p0'), vc.get('p1'), vc.get('p2')]):
                wval = [vc.get('p0'), vc.get('p1'), vc.get('p2')][widx]
                if wval is not None:
                    cat_moves[cat][vc['move']].append(wval)

    all_moves = sorted(set(
        m for d in cat_moves.values() for m in d
    ))
    if not all_moves:
        print('  No value check data available yet.')
        return

    n_t = cat_counts['timbuktu']
    n_r = cat_counts['rare']
    n_o = cat_counts['timeout']
    print(f'  Winner\'s value at checkpoint (timeouts: max player value).')
    print(f'  Games with checks: Timbuktu={n_t}  Rare goods={n_r}  Timeout={n_o}')
    print()
    print(f'  {"Move":>5}  {"Timbuktu":>14}  {"Rare goods":>14}  {"Timeout":>14}')
    print(f'  {"─"*5}  {"─"*14}  {"─"*14}  {"─"*14}')

    def cell(d, move):
        vals = d.get(move, [])
        if not vals:
            return f'{"---":>14}'
        return f'{sum(vals)/len(vals):+.3f} (n={len(vals):2d})'

    for move in all_moves:
        print(f'  {move:>5}  {cell(cat_moves["timbuktu"], move):>14}'
              f'  {cell(cat_moves["rare"], move):>14}'
              f'  {cell(cat_moves["timeout"], move):>14}')


# ── trigger vs winner ─────────────────────────────────────────────────────────

def print_search_cost(search_costs, window=25):
    """Self-play search cost per game: simulations, NN evaluations, cache effectiveness.

    Fed by the SEARCH COST line each actor logs at the end of a game. Absent from
    logs predating playout cap randomization, in which case this section is skipped.
    """
    if not search_costs:
        return

    section('SELF-PLAY SEARCH COST')

    n = len(search_costs)
    tot_moves = sum(c['moves'] for c in search_costs)
    tot_full  = sum(c['full_moves'] for c in search_costs)
    tot_fast  = sum(c['fast_moves'] for c in search_costs)
    tot_sims  = sum(c['sims'] for c in search_costs)
    tot_nn    = sum(c['nn_evals'] for c in search_costs)
    tot_hits  = sum(c['cache_hits'] for c in search_costs)

    print(f'  Games with cost data      : {n}')
    print(f'  Simulations per game      : {tot_sims/n:,.0f} avg')
    if tot_moves > 0:
        print(f'  Simulations per move      : {tot_sims/tot_moves:,.1f} avg')
    print(f'  NN evaluations per game   : {tot_nn/n:,.0f} avg')

    # Cache effectiveness: how many evaluator calls were served without a forward pass.
    lookups = tot_nn + tot_hits
    if lookups > 0:
        print(f'  Evaluator cache hit rate  : {pct(tot_hits, lookups)}  '
              f'({tot_hits:,} hits / {lookups:,} lookups)')
        if tot_nn > 0:
            print(f'  Forward passes saved      : {tot_hits:,}  '
                  f'({lookups/tot_nn:.2f}x fewer than uncached)')

    # Playout cap split. With the cap off every move is "full", which is how you
    # tell from the log alone whether the cap is actually active on the workers.
    if tot_moves > 0:
        if tot_fast == 0:
            print(f'  Playout cap               : OFF (all {tot_full:,} moves full-search)')
        else:
            print(f'  Playout cap               : ON  — full {pct(tot_full, tot_moves)}, '
                  f'fast {pct(tot_fast, tot_moves)}')
            print(f'  Policy targets per game   : {tot_full/n:,.0f} of {tot_moves/n:,.0f} moves')
            print(f'    (value targets come from every move, so the value head still '
                  f'sees all {tot_moves/n:,.0f})')

    # Cost trend: first vs last quarter. Sims/game climbs when games get longer,
    # which is the usual reason throughput sags mid-run.
    if n >= 8:
        q = max(2, n // 4)
        first, last = search_costs[:q], search_costs[-q:]
        f_sims = sum(c['sims'] for c in first) / len(first)
        l_sims = sum(c['sims'] for c in last) / len(last)
        f_mv = sum(c['moves'] for c in first) / len(first)
        l_mv = sum(c['moves'] for c in last) / len(last)
        f_hr = sum(c['hit_rate'] for c in first) / len(first)
        l_hr = sum(c['hit_rate'] for c in last) / len(last)
        arrow = '↑' if l_sims > f_sims else ('↓' if l_sims < f_sims else '=')
        _label = f'Trend (first {q} → last {q})'
        print(f'  {_label:<26}: '
              f'sims/game {f_sims:,.0f} → {l_sims:,.0f} {arrow}  |  '
              f'moves {f_mv:.0f} → {l_mv:.0f}  |  '
              f'hit rate {f_hr:.3f} → {l_hr:.3f}')

    # Remote vs local, so a slow worker fleet is visible here too.
    remote = [c for c in search_costs if c['is_remote']]
    local  = [c for c in search_costs if not c['is_remote']]
    if remote and local:
        print(f'  Local  : {len(local):4d} games, {sum(c["sims"] for c in local)/len(local):,.0f} sims/game, '
              f'hit rate {sum(c["hit_rate"] for c in local)/len(local):.3f}')
        print(f'  Remote : {len(remote):4d} games, {sum(c["sims"] for c in remote)/len(remote):,.0f} sims/game, '
              f'hit rate {sum(c["hit_rate"] for c in remote)/len(remote):.3f}')


def print_trigger_vs_winner(mcts_games):
    """Report cases where the player who triggered the end condition differs from the winner."""
    wins = [g for g in mcts_games if g['is_win']]
    if not wins:
        return

    section('TRIGGER vs WINNER (win-condition games)')

    def winner_idx(g):
        returns = g.get('returns', [])
        return returns.index(max(returns)) if returns else -1

    match = [g for g in wins if g.get('trigger', -1) == winner_idx(g)]
    differ = [g for g in wins if g.get('trigger', -1) != winner_idx(g) and g.get('trigger', -1) >= 0]

    print(f'  Trigger = Winner : {len(match):3d} / {len(wins)}  ({pct(len(match), len(wins))})')
    print(f'  Trigger ≠ Winner : {len(differ):3d} / {len(wins)}  ({pct(len(differ), len(wins))})')
    print()

    for cond_key, label in [('Timbuktu', 'Timbuktu to coast'),
                             ('Rare',     'Rare good regions')]:
        cond_wins = [g for g in wins if cond_key in g.get('reason', '')]
        if not cond_wins:
            continue
        cond_differ = [g for g in cond_wins if g.get('trigger', -1) != winner_idx(g)
                       and g.get('trigger', -1) >= 0]
        cond_match = len(cond_wins) - len(cond_differ)
        print(f'  {label}:')
        print(f'    Trigger = Winner : {cond_match:3d} / {len(cond_wins)}  ({pct(cond_match, len(cond_wins))})')
        print(f'    Trigger ≠ Winner : {len(cond_differ):3d} / {len(cond_wins)}  ({pct(len(cond_differ), len(cond_wins))})')

    if differ:
        print()
        print(f'  Cases where trigger ≠ winner:')
        print(f'  {"Game":>5}  {"":1}{"Actor":>6}  {"Moves":>5}  {"Condition":<18}  Trig  Win   Returns')
        print(f'  {"─"*5}  {"─"*7}  {"─"*5}  {"─"*18}  {"─"*4}  {"─"*4}  {"─"*26}')
        for g in sorted(differ, key=lambda x: x['game_n']):
            rtag = 'R' if g['actor'] >= 100000 else ' '
            trig = g['trigger']
            widx = winner_idx(g)
            returns = g.get('returns', [])
            ret_str = '  '.join(
                f'[{v:+.2f}]' if i == trig else f' {v:+.2f} '
                for i, v in enumerate(returns)
            )
            cond = 'Timbuktu' if 'Timbuktu' in g['reason'] else 'Rare goods'
            print(f'  {g["game_n"]:5d}  {rtag}{g["actor"]:6d}  {g["moves"]:5d}'
                  f'  {cond:<18}  P{trig}    P{widx}   {ret_str}')


# ── win/timeout comparison ────────────────────────────────────────────────────

def print_win_timeout_comparison(mcts_games, sample_n=5):
    """Compare a sample of recent wins against recent timeouts using logged data."""
    wins     = sorted([g for g in mcts_games if g['is_win']],      key=lambda g: g['game_n'])
    timeouts = sorted([g for g in mcts_games if not g['is_win']], key=lambda g: g['game_n'])
    if not wins or not timeouts:
        return

    recent_wins     = wins[-sample_n:]
    recent_timeouts = timeouts[-sample_n:]
    has_ir = any(g.get('inter_rewards') is not None for g in recent_wins + recent_timeouts)

    section(f'WIN vs TIMEOUT COMPARISON  '
            f'(last {len(recent_wins)} wins / last {len(recent_timeouts)} timeouts)')

    def _widx(g):
        """Player index of winner (wins: trigger, timeouts: argmax returns)."""
        if g['is_win'] and g.get('trigger', -1) >= 0:
            return g['trigger']
        returns = g.get('returns', [])
        return returns.index(max(returns)) if returns else -1

    def _ir_stats(g):
        """Return (winner_ir_per100, others_avg_ir_per100) normalised by move count."""
        ir = g.get('inter_rewards')
        if not ir:
            return None, None
        widx = _widx(g)
        if widx < 0 or widx >= len(ir):
            return None, None
        moves = g['moves'] or 1
        others = [ir[i] for i in range(len(ir)) if i != widx]
        return (ir[widx] / moves * 100,
                sum(others) / len(others) / moves * 100 if others else 0.0)

    def _fmt_returns(g):
        returns = g.get('returns', [])
        if not returns:
            return 'n/a'
        widx = _widx(g)
        parts = []
        for i, v in enumerate(returns):
            s = f'{v:+.2f}'
            parts.append(f'[{s}]' if i == widx else f' {s} ')
        return '  '.join(parts)

    def _short_reason(reason):
        if 'Timbuktu' in reason:
            return 'Timbuktu→coast'
        if 'Rare good' in reason or 'rare good' in reason:
            return 'Rare good regions'
        return reason[:18]

    def _safe_avg(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    def _fn(v, fmt='.3f'):
        return f'{v:{fmt}}' if v is not None else 'n/a'

    def _print_group(group, label):
        print(f'\n  {label}')
        hdr = f'  {"Game":>5}  {"Actor":>7}  {"Moves":>5}  {"Type / outcome":<20}'
        if has_ir:
            hdr += f'  {"WIR/100":>7}  {"OIR/100":>7}'
        hdr += '  Returns'
        print(hdr)
        sep = f'  {"─"*5}  {"─"*7}  {"─"*5}  {"─"*20}'
        if has_ir:
            sep += f'  {"─"*7}  {"─"*7}'
        sep += '  ' + '─' * 24
        print(sep)
        for g in group:
            rtag = 'R' if g['actor'] >= 100000 else ' '
            if g['is_win']:
                label_str = _short_reason(g['reason'])
            else:
                widx = _widx(g)
                label_str = f'Timeout (P{widx} led)' if widx >= 0 else 'Timeout'
            row = f'  {g["game_n"]:5d}  {rtag}{g["actor"]:6d}  {g["moves"]:5d}  {label_str:<20}'
            if has_ir:
                w_ir, o_ir = _ir_stats(g)
                row += f'  {_fn(w_ir):>7}  {_fn(o_ir):>7}'
            row += f'  {_fmt_returns(g)}'
            print(row)

    _print_group(recent_wins,     f'RECENT WINS     ({len(wins)} total)')
    _print_group(recent_timeouts, f'RECENT TIMEOUTS ({len(timeouts)} total)')

    # ── averages block ──
    def _collect(group):
        out = {'moves': [], 'wr': [], 'wi': [], 'oi': []}
        for g in group:
            out['moves'].append(g['moves'])
            returns = g.get('returns', [])
            widx = _widx(g)
            if returns and 0 <= widx < len(returns):
                out['wr'].append(returns[widx])
            w_ir, o_ir = _ir_stats(g)
            if w_ir is not None:
                out['wi'].append(w_ir)
            if o_ir is not None:
                out['oi'].append(o_ir)
        return {k: _safe_avg(v) for k, v in out.items()}

    ws = _collect(recent_wins)
    ts = _collect(recent_timeouts)

    print(f'\n  AVERAGES (over sample)')
    print(f'  {"Metric":<30}  {"Wins":>8}  {"Timeouts":>8}')
    print(f'  {"─"*30}  {"─"*8}  {"─"*8}')
    print(f'  {"Moves":<30}  {_fn(ws["moves"], ".1f"):>8}  {_fn(ts["moves"], ".1f"):>8}')
    print(f'  {"Winner/leader return":<30}  {_fn(ws["wr"]):>8}  {_fn(ts["wr"]):>8}')
    if has_ir:
        print(f'  {"Winner/leader IR per 100 moves":<30}  {_fn(ws["wi"]):>8}  {_fn(ts["wi"]):>8}')
        print(f'  {"Others avg IR per 100 moves":<30}  {_fn(ws["oi"]):>8}  {_fn(ts["oi"]):>8}')

    # ── win type + player summaries ──
    win_types = Counter(_short_reason(g['reason']) for g in wins)
    if win_types:
        print(f'\n  Win type breakdown (all {len(wins)} wins):')
        for wt, cnt in win_types.most_common():
            print(f'    {wt:<22}  {cnt:4d}  ({100*cnt/len(wins):.0f}%)')

    player_wins = Counter(_widx(g) for g in wins if _widx(g) >= 0)
    if player_wins:
        print(f'\n  Winning player distribution (all {len(wins)} wins):')
        for pid, cnt in sorted(player_wins.items()):
            print(f'    P{pid}: {cnt:4d}  ({100*cnt/len(wins):.0f}%)')


# ── report ────────────────────────────────────────────────────────────────────

def report(data, window=50, show_early_terminations=False):
    games = data['games']
    received = data['received']
    losses = data['losses']
    top_actions = data['top_action_counter']
    last_buffer_state = data.get('last_buffer_state')
    last_adaptive_fraction = data.get('last_adaptive_fraction')

    # ── Overview ──────────────────────────────────────────────────────────────
    section('OVERVIEW')
    n_received = len(received)
    n_bootstrap_rcvd = sum(1 for r in received if r.get('phase') == 'Bootstrap')
    n_mcts_rcvd = n_received - n_bootstrap_rcvd
    n_target = received[-1]['total'] if received else '?'
    n_finished = len(games)
    n_bootstrap_fin = sum(1 for g in games if g['is_bootstrap'])
    n_mcts_fin = n_finished - n_bootstrap_fin
    n_remote = sum(1 for g in games if g['is_remote'] and not g['is_bootstrap'])
    n_local = n_mcts_fin - n_remote

    # Games actually added to weights = received win-condition games only
    n_bootstrap_added = sum(1 for r in received if r.get('phase') == 'Bootstrap' and r.get('is_win'))
    n_mcts_added = sum(1 for r in received if r.get('phase') != 'Bootstrap' and r.get('is_win'))

    elapsed_min = 0
    if data['first_time'] and data['last_time']:
        delta = data['last_time'] - data['first_time']
        elapsed_min = delta.total_seconds() / 60

    print(f'  Games received by learner : {n_received} / {n_target}  '
          f'(bootstrap: {n_bootstrap_rcvd}, MCTS: {n_mcts_rcvd})')
    print(f'  Added to weights          : {n_bootstrap_added + n_mcts_added}  '
          f'(bootstrap: {n_bootstrap_added}, MCTS: {n_mcts_added})')
    nw_mcts = data.get('near_win_retained_count', 0)
    nw_boot = data.get('near_win_bootstrap_count', 0)
    if nw_mcts + nw_boot > 0:
        print(f'  Near-win timeout retained : {nw_mcts + nw_boot}  '
              f'(bootstrap: {nw_boot}, MCTS: {nw_mcts})')
    print(f'  FINISHED lines parsed     : {n_finished}  '
          f'(bootstrap: {n_bootstrap_fin}, MCTS local: {n_local}, MCTS remote: {n_remote})')
    if elapsed_min > 0 and n_mcts_rcvd > 0:
        rate = n_mcts_rcvd / elapsed_min * 60
        remaining = (int(n_target) - n_mcts_rcvd) / rate if rate > 0 else float('inf')
        print(f'  Elapsed (approx)          : {elapsed_min:.0f} min')
        print(f'  Throughput (MCTS games)   : {rate:.1f} games/hr')
        print(f'  Estimated remaining       : {remaining:.0f} hr  '
              f'({remaining/24:.1f} days)')
    elif elapsed_min > 0:
        rate = n_received / elapsed_min * 60
        print(f'  Elapsed (approx)          : {elapsed_min:.0f} min')
        print(f'  Throughput                : {rate:.1f} games/hr  (bootstrap still in progress)')

    if last_buffer_state:
        bs = last_buffer_state
        rg = bs.get('mcts_raregoods')
        total = bs['bootstrap'] + bs['mcts_natural'] + bs['mcts_nearwin'] + (rg or 0)
        rg_str = f', MCTS-raregoods: {rg:,}' if rg is not None else ''
        print(f'  Replay buffer (latest)    : {total:,} experiences  '
              f'(bootstrap: {bs["bootstrap"]:,}, MCTS-natural: {bs["mcts_natural"]:,}, '
              f'MCTS-nearwin: {bs["mcts_nearwin"]:,}{rg_str})  '
              f'@ {bs["time"].strftime("%Y%m%d-%H%M%S")}')
        if last_adaptive_fraction is not None:
            print(f'  Adaptive mcts_fraction    : {last_adaptive_fraction:.2f}')

    recent_guidance = data.get('recent_heuristic_guidance', [])
    if recent_guidance:
        print(f'  Heuristic guidance (last {len(recent_guidance)}):')
        for h in recent_guidance:
            print(f'    {h["time"].strftime("%Y%m%d-%H%M%S")}  '
                  f'Actor {h["actor"]:>3}, Game {h["game"]:>4}: '
                  f'heuristic_guidance_weight={h["weight"]:.3f}')

    # ── Bootstrap summary ─────────────────────────────────────────────────────
    bootstrap_games = [g for g in games if g['is_bootstrap']]
    if bootstrap_games:
        section('BOOTSTRAP PHASE')
        b_wins = [g for g in bootstrap_games if g['is_win']]
        b_max = [g for g in bootstrap_games if not g['is_win']]
        print(f'  Total bootstrap games     : {len(bootstrap_games)}')
        print(f'  Win-condition games       : {len(b_wins)} = {pct(len(b_wins), len(bootstrap_games))}')
        print(f'  Max-length games          : {len(b_max)} = {pct(len(b_max), len(bootstrap_games))}')

        b_reason_counts = Counter(g['reason'] for g in b_wins)
        if b_reason_counts:
            b_reason_lengths = defaultdict(list)
            b_reason_player_triggers = defaultdict(Counter)
            for g in b_wins:
                b_reason_lengths[g['reason']].append(g['moves'])
                b_reason_player_triggers[g['reason']][g['trigger']] += 1

            n_players_b = max(len(g['returns']) for g in b_wins) if b_wins else 0
            player_cols_b = ''.join(f'  P{p}' for p in range(n_players_b))
            total_b_wins = len(b_wins)

            print()
            print(f'  {"Win reason":<42}  {"count":>5}  {"pct":>5}  {"avg len":>7}{player_cols_b}')
            print(f'  {"─"*42}  {"─"*5}  {"─"*5}  {"─"*7}' + '  ──' * n_players_b)
            for reason, count in b_reason_counts.most_common():
                lengths = b_reason_lengths[reason]
                avg_len = sum(lengths) / len(lengths)
                w_pct = pct(count, total_b_wins)
                ptrig = b_reason_player_triggers[reason]
                p_cols = ''.join(f'  {ptrig.get(p, 0):2d}' for p in range(n_players_b))
                print(f'  {reason:<42}  {count:5d}  {w_pct:>5}  {avg_len:7.0f}{p_cols}')

        b_win_lengths = [g['moves'] for g in b_wins]
        b_max_lengths = [g['moves'] for g in bootstrap_games]
        if b_win_lengths:
            print()
            print(f'  Win game length  : min={min(b_win_lengths)}  '
                  f'avg={sum(b_win_lengths)/len(b_win_lengths):.0f}  max={max(b_win_lengths)}')
        print(f'  All game length  : min={min(b_max_lengths)}  '
              f'avg={sum(b_max_lengths)/len(b_max_lengths):.0f}  max={max(b_max_lengths)}')

        print()
        print('  Player win balance (bootstrap):')
        b_player_wins = Counter()
        b_player_games = Counter()
        for g in bootstrap_games:
            max_r = max(g['returns'])
            sole_winner = g['returns'].count(max_r) == 1
            for i, r in enumerate(g['returns']):
                b_player_games[i] += 1
                if sole_winner and abs(r - max_r) < 1e-6:
                    b_player_wins[i] += 1
        for p in sorted(b_player_games):
            wr = b_player_wins[p] / b_player_games[p] if b_player_games[p] else 0
            print(f'    Player {p}: {b_player_wins[p]:4d} wins / {b_player_games[p]:4d} games = '
                  f'{pct(b_player_wins[p], b_player_games[p]):6s}  {bar(wr, 0.5)}')

    # ── Self-play search cost ────────────────────────────────────────────
    print_search_cost(data.get('search_costs', []))

    # ── Win conditions (MCTS only) ────────────────────────────────────────────
    mcts_games = [g for g in games if not g['is_bootstrap']]
    section('WIN CONDITIONS (MCTS games)')
    wins = [g for g in mcts_games if g['is_win']]
    max_len = [g for g in mcts_games if not g['is_win']]
    n_mcts_total = len(mcts_games)
    print(f'  Win-condition games : {len(wins)} / {n_mcts_total} = {pct(len(wins), n_mcts_total)}')
    print(f'  Max-length games    : {len(max_len)} / {n_mcts_total} = {pct(len(max_len), n_mcts_total)}')

    # Local vs remote breakdown
    local_mcts  = [g for g in mcts_games if not g['is_remote']]
    remote_mcts = [g for g in mcts_games if g['is_remote']]
    for label, subset in [('Local', local_mcts), ('Remote', remote_mcts)]:
        if subset:
            s_wins = sum(1 for g in subset if g['is_win'])
            s_max  = sum(1 for g in subset if not g['is_win'])
            n = len(subset)
            print(f'    {label:6s} actors:  win-condition {s_wins:4d} / {n:4d} = {pct(s_wins, n)}  '
                  f'max-length {s_max:4d} / {n:4d} = {pct(s_max, n)}')

    reason_counts = Counter(g['reason'] for g in wins)
    if reason_counts:
        # Per-reason stats: count, avg game length, player trigger breakdown
        reason_lengths = defaultdict(list)
        reason_player_triggers = defaultdict(Counter)
        for g in wins:
            reason_lengths[g['reason']].append(g['moves'])
            reason_player_triggers[g['reason']][g['trigger']] += 1

        # Trend: per-reason counts in first vs second half of MCTS games
        half = len(mcts_games) // 2
        early_wins = [g for g in mcts_games[:half] if g['is_win']]
        late_wins  = [g for g in mcts_games[half:] if g['is_win']]
        early_counts = Counter(g['reason'] for g in early_wins)
        late_counts  = Counter(g['reason'] for g in late_wins)

        n_players = max(len(g['returns']) for g in wins) if wins else 0
        player_cols = ''.join(f'  P{p}' for p in range(n_players))

        print()
        print(f'  {"Win reason":<42}  {"count":>5}  {"pct":>5}  {"avg len":>7}  '
              f'{"early→late":>11}{player_cols}')
        print(f'  {"─"*42}  {"─"*5}  {"─"*5}  {"─"*7}  {"─"*11}' +
              '  ──' * n_players)
        total_wins = len(wins)
        for reason, count in reason_counts.most_common():
            lengths = reason_lengths[reason]
            avg_len = sum(lengths) / len(lengths)
            w_pct = pct(count, total_wins)
            e = early_counts.get(reason, 0)
            l = late_counts.get(reason, 0)
            trend = f'{e:4d} → {l:4d}'
            ptrig = reason_player_triggers[reason]
            p_cols = ''.join(f'  {ptrig.get(p, 0):2d}' for p in range(n_players))
            print(f'  {reason:<42}  {count:5d}  {w_pct:>5}  {avg_len:7.0f}  {trend:>11}{p_cols}')

    win_lengths = [g['moves'] for g in wins]
    if win_lengths:
        print()
        sl = sorted(win_lengths)
        n  = len(sl)
        p25 = sl[int(n * 0.25)]
        p50 = sl[int(n * 0.50)]
        p75 = sl[int(n * 0.75)]
        avg = sum(sl) / n
        print(f'  Win game length: min={min(sl)}  p25={p25}  median={p50}  '
              f'p75={p75}  avg={avg:.0f}  max={max(sl)}')
        print()
        print(f'  quick_win_threshold guidance:')
        print(f'    Current threshold is set in [Training] quick_win_threshold in mali_ba.ini.')
        print(f'    Set it to the median ({p50}) so roughly half of wins earn the bonus —')
        print(f'    fast enough to be a meaningful target, common enough to train on.')
        print(f'    If wins are clustering well below the median, lower it toward p25 ({p25}).')
        print(f'    Raise it toward p75 ({p75}) only if the model rarely earns the bonus.')

    # Trend: natural-finish game length over time
    if len(wins) >= 8:
        print()
        print('  Natural-finish length trend (avg moves, bucketed across run):')
        n_buckets = min(10, len(wins))
        bucket_size = len(wins) / n_buckets
        max_len_val = max(g['moves'] for g in wins)
        for b in range(n_buckets):
            lo = int(b * bucket_size)
            hi = int((b + 1) * bucket_size)
            chunk = wins[lo:hi]
            if not chunk:
                continue
            avg = sum(g['moves'] for g in chunk) / len(chunk)
            lo_g = wins[lo]['game_n']
            hi_g = wins[hi - 1]['game_n']
            print(f'    games {lo_g:4d}-{hi_g:4d}: {avg:6.0f} moves  {bar(avg, max_len_val)}')
    elif wins:
        print(f'  (need 8+ natural-finish games for length trend; have {len(wins)})')

    # Build short reason labels for inline display
    def short_reason(reason):
        if 'Timbuktu' in reason or 'coast' in reason.lower() or 'route' in reason.lower():
            return 'Tim'
        if 'region' in reason.lower() or 'rare' in reason.lower():
            return 'Rare'
        # Fallback: first word(s) up to 6 chars
        return reason[:6].rstrip()

    # Trend: win rate over rolling windows
    if n_mcts_total >= window:
        print()
        print(f'  Win-rate trend (rolling {window}-game window, FINISHED lines):')
        step = max(1, n_mcts_total // 8)
        rows = []
        for i in range(0, n_mcts_total - window + 1, step):
            chunk = mcts_games[i:i+window]
            chunk_wins = [g for g in chunk if g['is_win']]
            wr = len(chunk_wins) / window
            if chunk_wins:
                reason_counts_chunk = Counter(short_reason(g['reason']) for g in chunk_wins)
                n_cw = len(chunk_wins)
                reason_str = ' '.join(
                    f'{r}:{round(c/n_cw*100)}%'
                    for r, c in reason_counts_chunk.most_common()
                )
                reason_part = f'({reason_str})'
            else:
                reason_part = ''
            rows.append((i, wr, reason_part))
        max_reason_w = max(len(r[2]) for r in rows) if rows else 0
        for i, wr, reason_part in rows:
            print(f'    games {i+1:4d}-{i+window:4d}: {pct(wr*window, window):6s}  '
                  f'{reason_part:{max_reason_w}s}  {bar(wr, 0.5)}')

    # ── Player win balance (MCTS) ─────────────────────────────────────────────
    section('PLAYER WIN BALANCE (MCTS games)')
    player_wins = Counter()
    player_games = Counter()
    for g in mcts_games:
        max_r = max(g['returns'])
        sole_winner = g['returns'].count(max_r) == 1
        for i, r in enumerate(g['returns']):
            player_games[i] += 1
            if sole_winner and abs(r - max_r) < 1e-6:
                player_wins[i] += 1
    for p in sorted(player_games):
        wr = player_wins[p] / player_games[p] if player_games[p] else 0
        print(f'  Player {p}: {player_wins[p]:4d} wins / {player_games[p]:4d} games = '
              f'{pct(player_wins[p], player_games[p]):6s}  {bar(wr, 0.5)}')

    # ── Actor contributions ───────────────────────────────────────────────────
    section('ACTOR CONTRIBUTIONS (MCTS games)')
    actor_counts = Counter(g['actor'] for g in mcts_games)
    local_actors = [k for k in actor_counts if k < 100000]
    remote_actors = [k for k in actor_counts if k >= 100000]
    n_local_games = sum(actor_counts[a] for a in local_actors)
    n_remote_games = sum(actor_counts[a] for a in remote_actors)
    n_local_actors = len(local_actors)
    n_remote_actors = len(remote_actors)
    print(f'  Local  actors: {n_local_actors:3d} actors   {n_local_games:4d} games')
    print(f'  Remote actors: {n_remote_actors:3d} actors   {n_remote_games:4d} games')

    # ── Loss trend ────────────────────────────────────────────────────────────
    section('LOSS TREND')
    trainer_losses = data.get('trainer_losses', [])
    if trainer_losses:
        n = len(trainer_losses)
        all_vals = [l for _, l in trainer_losses]
        n_buckets = min(10, n)
        bucket_size = n // n_buckets or 1
        print(f'  Training steps : {n}')
        print(f'  Loss range     : {min(all_vals):.4f} – {max(all_vals):.4f}')
        print()
        print(f'  {"Steps":>12}  {"Avg loss":>8}  {"Min":>7}  {"Max":>7}')
        print(f'  {"─"*12}  {"─"*8}  {"─"*7}  {"─"*7}  ')
        first_bucket_avg = last_bucket_avg = None
        for b in range(n_buckets):
            lo = b * bucket_size
            hi = lo + bucket_size if b < n_buckets - 1 else n
            bucket_vals = all_vals[lo:hi]
            avg = sum(bucket_vals) / len(bucket_vals)
            if b == 0:
                first_bucket_avg = avg
            last_bucket_avg = avg
            trend = bar(1.0 / avg if avg > 0 else 0, 0.35, 20)
            print(f'  steps {lo:4d}-{hi-1:4d}:  {avg:8.4f}  {min(bucket_vals):7.4f}  '
                  f'{max(bucket_vals):7.4f}  {trend}')

        # linear regression for slope
        xs = list(range(n))
        ys = all_vals
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
        den = sum((x - x_mean) ** 2 for x in xs)
        slope = num / den if den > 0 else 0
        first_loss = first_bucket_avg
        last_loss = last_bucket_avg
        pct_drop = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0
        direction = '↓ improving' if slope < -0.0001 else ('↑ worsening' if slope > 0.0001 else '→ flat')
        print(f'\n  First: {first_loss:.4f}  →  Latest: {last_loss:.4f}  ({pct_drop:+.1f}%)')
        print(f'  Trend slope : {slope:+.6f} per step  {direction}')
    elif losses:
        # fallback: old format bucketed by game number
        max_game = max(l[0] for l in losses)
        buckets = defaultdict(list)
        n_buckets = min(10, len(losses))
        for game_n, total, policy, value in losses:
            bucket = min(int(game_n / (max_game + 1) * n_buckets), n_buckets - 1)
            buckets[bucket].append((total, policy, value))
        print(f'  {"Game range":>14}  {"Total":>7}  {"Policy":>7}  {"Value":>7}')
        bucket_size = (max_game + 1) // n_buckets or 1
        for b in range(n_buckets):
            if b not in buckets:
                continue
            vals = buckets[b]
            avg_total = sum(v[0] for v in vals) / len(vals)
            avg_policy = sum(v[1] for v in vals) / len(vals)
            avg_value = sum(v[2] for v in vals) / len(vals)
            lo = b * bucket_size
            hi = (b + 1) * bucket_size - 1
            trend = bar(1.0 / avg_total if avg_total > 0 else 0, 0.4, 20)
            print(f'  games {lo:4d}-{hi:4d}:  {avg_total:7.4f}   {avg_policy:7.4f}   '
                  f'{avg_value:7.4f}  {trend}')
        first_loss = losses[0][1]
        last_loss = losses[-1][1]
        pct_drop = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0
        print(f'\n  First loss: {first_loss:.4f}  →  Latest loss: {last_loss:.4f}  '
              f'({pct_drop:+.1f}%)')

    # ── MCTS action preferences ───────────────────────────────────────────────
    if top_actions:
        section('MCTS TOP-ACTION PREFERENCES (most-chosen actions)')
        total_moves = sum(top_actions.values())

        # Categorise actions
        categories = defaultdict(int)
        for action, count in top_actions.items():
            if action == 'Pass':
                categories['Pass'] += count
            elif action == 'TakeIncome':
                categories['TakeIncome'] += count
            elif action.startswith('StartMancala'):
                categories['StartMancala'] += count
            elif action.startswith('MancalaDir'):
                categories['MancalaDir'] += count
            elif action.startswith('PlacePost'):
                categories['PlacePost'] += count
            elif action.startswith('UpgradePost'):
                categories['UpgradePost'] += count
            elif action.startswith('DeclareRoute'):
                categories['DeclareRoute'] += count
            elif action.startswith('PayGood'):
                categories['PayGood'] += count
            else:
                categories['Other'] += count

        print(f'  (based on {total_moves:,} MCTS top-action samples)')
        print()
        sorted_cats = sorted(categories.items(), key=lambda x: -x[1])
        max_cat = max(categories.values())
        max_count_w = max(len(f'{c:,}') for _, c in sorted_cats)
        for cat, count in sorted_cats:
            print(f'  {cat:>14}: {count:{max_count_w},}  {pct(count, total_moves):6s}  '
                  f'{bar(count, max_cat)}')

        pass_pct = categories['Pass'] / total_moves * 100 if total_moves else 0
        print(f'\n  Pass-only MCTS results (all sims → Pass): {data["pass_only_count"]:,}')
        if pass_pct > 5:
            print(f'  ⚠  Pass is {pass_pct:.1f}% of top actions — still elevated')
        else:
            print(f'  Pass frequency looks healthy ({pass_pct:.1f}%)')

    # ── Intermediate rewards summary ──────────────────────────────────────────
    games_with_rewards = [g for g in mcts_games if g.get('inter_rewards') is not None]
    if games_with_rewards:
        section('INTERMEDIATE REWARDS (MCTS games)')
        n_players = max(len(g['inter_rewards']) for g in games_with_rewards)

        per_player_totals = [[] for _ in range(n_players)]
        rewarded_steps_list = []
        for g in games_with_rewards:
            for i, r in enumerate(g['inter_rewards']):
                per_player_totals[i].append(r)
            if g['rewarded_steps'] is not None:
                rewarded_steps_list.append(g['rewarded_steps'])

        print(f'  Games with reward data: {len(games_with_rewards)}')
        print()
        print(f'  {"":8}  {"avg total":>10}  {"min":>8}  {"max":>8}  {"per move":>9}')
        print(f'  {"─"*8}  {"─"*10}  {"─"*8}  {"─"*8}  {"─"*9}')
        for i in range(n_players):
            vals = per_player_totals[i]
            moves = [g['moves'] for g in games_with_rewards]
            avg_per_move = sum(v / m for v, m in zip(vals, moves)) / len(vals) if vals else 0
            print(f'  Player {i}  {sum(vals)/len(vals):>10.4f}  {min(vals):>8.4f}  '
                  f'{max(vals):>8.4f}  {avg_per_move:>9.5f}')

        if rewarded_steps_list:
            avg_steps = sum(rewarded_steps_list) / len(rewarded_steps_list)
            print(f'\n  Avg rewarded steps/game : {avg_steps:.1f}')

        # Trend: avg intermediate reward across time buckets
        if len(games_with_rewards) >= 6:
            print()
            print('  Avg total intermediate reward trend (all players, bucketed):')
            n_buckets = min(8, len(games_with_rewards))
            bucket_size = len(games_with_rewards) / n_buckets
            for b in range(n_buckets):
                lo = int(b * bucket_size)
                hi = int((b + 1) * bucket_size)
                chunk = games_with_rewards[lo:hi]
                if not chunk:
                    continue
                all_vals = [r for g in chunk for r in g['inter_rewards']]
                avg = sum(all_vals) / len(all_vals)
                lo_g = lo + 1
                hi_g = hi
                # Bar: more negative = longer games; scale so -0.25 = empty, 0 = full
                scaled = max(0.0, min(1.0, (avg + 0.25) / 0.25))
                print(f'    games {lo_g:4d}-{hi_g:4d}: {avg:7.4f}  {bar(scaled, 1.0)}')

    # ── Early terminations ────────────────────────────────────────────────────
    early_terminations = data.get('early_terminations', [])
    if early_terminations:
        section('EARLY TERMINATIONS')
        n_et = len(early_terminations)
        n_no_near_win = sum(1 for e in early_terminations if 'no near-win' in e['reason'])
        n_hopeless = sum(1 for e in early_terminations if 'hopeless' in e['reason'].lower())
        n_other = n_et - n_no_near_win - n_hopeless
        print(f'  Total early-terminated games : {n_et}')
        print(f'    No near-win after threshold : {n_no_near_win}')
        print(f'    Value head hopeless         : {n_hopeless}')
        if n_other:
            print(f'    Other / unknown             : {n_other}')
        if show_early_terminations:
            print()
            print(f'  {"Game":>5}  {"":1}{"Actor":>6}  {"Move":>5}  Reason')
            print(f'  {"─"*5}  {"─"*7}  {"─"*5}  {"─"*50}')
            for e in early_terminations:
                reason_disp = e['reason'][:70]
                remote_tag = 'R' if e['actor'] >= 100000 else ' '
                print(f'  {e["game_n"]:5d}  {remote_tag}{e["actor"]:6d}  {e["move"]:5d}  {reason_disp}')
        else:
            print(f'  (per-game listing suppressed; pass --show-early-terminations to include it)')

        # Surviving games (those that reached a FINISHED line)
        n_natural = sum(1 for g in mcts_games if g['is_win'])
        n_timeout = len(mcts_games) - n_natural
        print()
        print(f'  Surviving games : {len(mcts_games)}  '
              f'(natural wins: {n_natural}, timeouts: {n_timeout})')
        ot = data.get('oversample_threshold')
        ot_note = '' if ot is not None else ' (threshold unknown, assumed 350)'
        ot = ot if ot is not None else 350
        win_exp = sum(
            g['moves'] * (3 if ot > 0 and g['moves'] < ot else 1)
            for g in mcts_games if g['is_win']
        )
        to_exp = sum(g['moves'] for g in mcts_games if not g['is_win'])
        print(f'  Experiences added to buffer: {win_exp:,} for natural wins, '
              f'{to_exp:,} for timeouts{ot_note}')
        if mcts_games and show_early_terminations:
            print()
            print(f'  {"Game":>5}  {"":1}{"Actor":>6}  {"Moves":>5}  Outcome')
            print(f'  {"─"*5}  {"─"*7}  {"─"*5}  {"─"*50}')
            for g in mcts_games:
                if g['is_win']:
                    outcome = f'Win  — {g["reason"][:44]}'
                else:
                    outcome = 'Timeout'
                remote_tag = 'R' if g['actor'] >= 100000 else ' '
                print(f'  {g["game_n"]:5d}  {remote_tag}{g["actor"]:6d}  {g["moves"]:5d}  {outcome}')

        print_win_timeout_comparison(mcts_games, sample_n=5)
        print_trigger_vs_winner(mcts_games)
        print_value_trajectory(mcts_games, data.get('value_checks', []))

    # ── No-kill experiment ────────────────────────────────────────────────────
    no_kill_games = data.get('no_kill_games', {})
    if no_kill_games:
        section('NO-KILL EXPERIMENT')
        # Join with game outcomes
        game_lookup = {(g['actor'], g['game_n']): g for g in mcts_games}
        n_nk = len(no_kill_games)
        n_nk_wins = sum(1 for k, g in no_kill_games.items()
                        if game_lookup.get(k, {}).get('is_win'))
        n_nk_timeouts = sum(1 for k, g in no_kill_games.items()
                            if k in game_lookup and not game_lookup[k].get('is_win'))
        n_nk_killed = n_nk - n_nk_wins - n_nk_timeouts  # still early-terminated despite no-kill
        print(f'  No-kill games tracked : {n_nk}')
        print(f'    Natural wins        : {n_nk_wins}')
        print(f'    Timeouts            : {n_nk_timeouts}')
        print(f'    In progress         : {n_nk_killed}  '
              f'(no FINISHED line yet — still running or ended before first kill check)')
        print()
        print(f'  {"Game":>5}  {"":1}{"Actor":>6}  {"Moves":>5}  {"Outcome":<18}  Would-have-terminated events')
        print(f'  {"─"*5}  {"─"*7}  {"─"*5}  {"─"*18}  {"─"*44}')
        for key in sorted(no_kill_games.keys(), key=lambda k: k[1]):
            actor, game_n = key
            g = game_lookup.get(key)
            if g:
                outcome = f'Win — {g["reason"][:12]}' if g['is_win'] else 'Timeout'
                moves = str(g['moves'])
            else:
                outcome = 'In progress'
                moves = '?'
            remote_tag = 'R' if actor >= 100000 else ' '
            events = no_kill_games[key]
            if events:
                event_str = '  |  '.join(
                    f'move {e["move"]}: {e["reason"][:35]}' for e in events
                )
            else:
                event_str = '(no kill rules fired)'
            print(f'  {game_n:5d}  {remote_tag}{actor:6d}  {moves:>5}  {outcome:<18}  {event_str}')

    # ── Recent games summary ──────────────────────────────────────────────────
    recd_games = 45
    section(f'LAST {recd_games} RECEIVED GAMES')
    for g in received[-recd_games:]:
        r = g['returns']
        max_r = max(r)
        winners = [i for i, v in enumerate(r) if abs(v - max_r) < 1e-6]
        w_str = f'P{winners[0]}' if len(winners) == 1 else 'tie'
        # is_win is set at parse time from the preceding FINISHED line
        is_natural = g.get('is_win')
        if is_natural is None:
            is_natural = max_r >= 1.0  # fallback: natural win has return ≥ 1.0
        flag = '★' if is_natural else ' '
        phase_tag = 'B' if g.get('phase') == 'Bootstrap' else 'M'
        actor = g.get('actor')
        actor_str = f'A{actor}' if actor is not None else '   ?'
        remote_tag = 'R' if actor is not None and actor >= 100000 else ' '
        ir = g.get('inter_rewards')
        ir_str = f'  inter: [{", ".join(f"{v:.3f}" for v in ir)}]' if ir else ''
        print(f'  {flag}{phase_tag}{remote_tag} Game #{g["game_num"]:4d}  {g["length"]:3d} moves  '
              f'{actor_str:<8}  winner: {w_str}  returns: {[f"{v:.1f}" for v in r]}{ir_str}')


def plot_win_rate(mcts_games, window=50, trainer_losses=None, run_start=None):
    """Plot a moving average of MCTS win rate, with loss trend on a secondary axis."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available. Install with: pip install matplotlib')
        return

    if len(mcts_games) < window:
        print(f'  Not enough MCTS games for a {window}-game moving average (have {len(mcts_games)}).')
        return

    # Use wall-clock time as the common x-axis, anchored to run start
    if run_start is None:
        run_start = mcts_games[0]['time']

    def to_hours(t):
        return (t - run_start).total_seconds() / 3600

    # Win rate moving average (x = hours since run start)
    win_xs = []
    win_ys = []
    is_win = [1 if g['is_win'] else 0 for g in mcts_games]
    for i in range(window - 1, len(mcts_games)):
        win_xs.append(to_hours(mcts_games[i]['time']))
        win_ys.append(sum(is_win[i - window + 1:i + 1]) / window * 100)

    try:
        import tkinter as _tk
        _root = _tk.Tk()
        _root.withdraw()
        _screen_h = _root.winfo_screenheight()
        _screen_w = _root.winfo_screenwidth()
        _root.destroy()
    except Exception:
        _screen_h, _screen_w = 900, 1600  # fallback if tkinter unavailable
    _dpi = 100
    _h_in = (_screen_h * 0.8) / _dpi
    _w_in = min(_h_in * (13 / 5), _screen_w * 0.95 / _dpi)
    fig, ax1 = plt.subplots(figsize=(_w_in, _h_in), dpi=_dpi)
    try:
        _mgr = plt.get_current_fig_manager()
        _w_px = int(_w_in * _dpi)
        _h_px = int(_h_in * _dpi)
        try:
            _mgr.window.state('normal')
            _mgr.window.geometry(f'{_w_px}x{_h_px}+50+50')   # TkAgg
        except AttributeError:
            try:
                _mgr.window.showNormal()                       # Qt
                _mgr.resize(_w_px, _h_px)
            except AttributeError:
                pass
    except Exception:
        pass
    ax1.plot(win_xs, win_ys, linewidth=1.5, color='steelblue',
             label=f'Win rate ({window}-game avg)')

    # Secondary, longer moving average for a steadier long-run trend line.
    _long_window = 60
    if len(mcts_games) >= _long_window:
        long_xs = []
        long_ys = []
        for i in range(_long_window - 1, len(mcts_games)):
            long_xs.append(to_hours(mcts_games[i]['time']))
            long_ys.append(sum(is_win[i - _long_window + 1:i + 1]) / _long_window * 100)
        ax1.plot(long_xs, long_ys, linewidth=1.5, color='purple',
                 label=f'Win rate ({_long_window}-game avg)')

    ax1.axhline(y=sum(is_win) / len(is_win) * 100, color='steelblue',
                linestyle='--', linewidth=1, alpha=0.5,
                label=f'Overall avg ({sum(is_win)/len(is_win)*100:.1f}%)')
    ax1.set_ylabel('Win Rate (%)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_ylim(0, 100)
    ax1.set_xlabel('Hours since run start')
    ax1.grid(True, alpha=0.3)
    _label_fontsize = ax1.xaxis.label.get_size()
    fig.text(0.5, 0.01,
             '~3.5 loss ≈ near-random  |  ~2.5–3.0 loss ≈ coarse preferences  |  ~1.5–2.5 loss ≈ meaningful strategic intent  |  <1.5 loss ≈ strong play\n'
             'likely requires many more MCTS games than you\'ll reach in early training.',
             ha='center', va='bottom', fontsize=_label_fontsize, color='tomato', linespacing=1.6)

    # Loss trend on secondary axis
    if trainer_losses and run_start:
        smooth = 50  # rolling average window for loss (reduces noise)
        loss_xs = []
        loss_ys = []
        loss_vals = [l for _, l in trainer_losses]
        for i in range(smooth - 1, len(trainer_losses)):
            t, _ = trainer_losses[i]
            loss_xs.append(to_hours(t))
            loss_ys.append(sum(loss_vals[i - smooth + 1:i + 1]) / smooth)

        ax2 = ax1.twinx()
        ax2.plot(loss_xs, loss_ys, linewidth=1.2, color='tomato', alpha=0.8,
                 label=f'Loss ({smooth}-step avg)')

        # Linear trend line over the smoothed loss
        n_l = len(loss_xs)
        if n_l >= 2:
            x_mean = sum(loss_xs) / n_l
            y_mean = sum(loss_ys) / n_l
            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(loss_xs, loss_ys))
            den = sum((x - x_mean) ** 2 for x in loss_xs)
            slope = num / den if den > 0 else 0
            intercept = y_mean - slope * x_mean
            trend_ys = [slope * x + intercept for x in loss_xs]
            ax2.plot(loss_xs, trend_ys, linewidth=1.0, color='tomato',
                     linestyle='--', alpha=0.5, label='Loss trend')

        ax2.set_ylabel('Training Loss', color='tomato')
        ax2.tick_params(axis='y', labelcolor='tomato')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')

    ax1.set_title(f'Win Rate ({window}-game avg) and Training Loss')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()


_REMOTE_SSH_HOST = '192.168.0.241'
_REMOTE_SSH_USER = 'robp'
_REMOTE_LOG_DIR  = '/media/robp/UD/Projects/open_spiel'


def plot_search_cost(search_costs, window=25, run_start=None):
    """Two stacked panels sharing the win-rate chart's x-axis (hours since run start):
    simulations and NN evaluations per game on top, cache hit rate and full-search
    share below. Makes it obvious when self-play cost creeps up as games lengthen.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available. Install with: pip install matplotlib')
        return

    if not search_costs:
        return
    if len(search_costs) < window:
        print(f'  Not enough SEARCH COST games for a {window}-game moving average '
              f'(have {len(search_costs)}).')
        return

    if run_start is None:
        run_start = search_costs[0]['time']

    def to_hours(t):
        return (t - run_start).total_seconds() / 3600

    def moving(key, scale=1.0):
        xs, ys = [], []
        vals = [c[key] * scale for c in search_costs]
        for i in range(window - 1, len(search_costs)):
            xs.append(to_hours(search_costs[i]['time']))
            ys.append(sum(vals[i - window + 1:i + 1]) / window)
        return xs, ys

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=(13, 7), dpi=100,
        gridspec_kw={'height_ratios': [3, 2]})

    xs, sims = moving('sims')
    _, nn = moving('nn_evals')
    _, moves = moving('moves')
    ax_top.plot(xs, sims, linewidth=1.5, color='darkorange',
                label=f'Simulations / game ({window}-game avg)')
    ax_top.plot(xs, nn, linewidth=1.5, color='seagreen',
                label=f'NN evaluations / game ({window}-game avg)')
    ax_top.set_ylabel('Per game')
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(loc='upper left')
    ax_top.set_title('Self-play search cost')

    # Game length on a secondary axis: the usual driver of rising cost.
    ax_len = ax_top.twinx()
    ax_len.plot(xs, moves, linewidth=1.0, color='slategray', alpha=0.6,
                linestyle='--', label=f'Moves / game ({window}-game avg)')
    ax_len.set_ylabel('Moves / game', color='slategray')
    ax_len.tick_params(axis='y', labelcolor='slategray')
    ax_len.legend(loc='upper right')

    _, hit = moving('hit_rate', 100.0)
    ax_bot.plot(xs, hit, linewidth=1.5, color='steelblue',
                label=f'Evaluator cache hit rate ({window}-game avg)')

    # Full-search share: flat at 100% with the playout cap off, near
    # playout_cap_full_prob with it on.
    full_share = []
    for i in range(window - 1, len(search_costs)):
        chunk = search_costs[i - window + 1:i + 1]
        tm = sum(c['moves'] for c in chunk)
        full_share.append(100.0 * sum(c['full_moves'] for c in chunk) / tm if tm else 0.0)
    ax_bot.plot(xs, full_share, linewidth=1.5, color='purple',
                label=f'Full-search moves ({window}-game avg)')

    ax_bot.set_ylabel('Percent')
    ax_bot.set_xlabel('Hours since run start')
    ax_bot.set_ylim(0, 100)
    ax_bot.grid(True, alpha=0.3)
    ax_bot.legend(loc='upper left')

    plt.tight_layout()
    plt.show()


def fetch_remote_value_checks(log_path, ssh_password):
    """SSH to the laptop actor host and return raw log lines containing 'Value check'.

    Requires paramiko (pip install paramiko).  The remote log is assumed to have
    the same filename as the local log, located at _REMOTE_LOG_DIR.
    Returns a list of raw log line strings (may be empty on any failure).
    """
    try:
        import paramiko
    except ImportError:
        print('  Remote log fetch skipped — run: pip install paramiko')
        return []

    import os
    log_name   = os.path.basename(log_path)
    remote_log = f'{_REMOTE_LOG_DIR}/{log_name}'

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        print(f'  Connecting to {_REMOTE_SSH_USER}@{_REMOTE_SSH_HOST} ...')
        client.connect(_REMOTE_SSH_HOST, username=_REMOTE_SSH_USER,
                       password=ssh_password, timeout=10)
        _, stdout, _ = client.exec_command(f"grep 'Value check' '{remote_log}'")
        lines = stdout.read().decode('utf-8', errors='replace').splitlines()
        client.close()
        print(f'  Fetched {len(lines)} remote value check lines from {remote_log}')
        return lines
    except Exception as e:
        print(f'  SSH fetch failed ({_REMOTE_SSH_USER}@{_REMOTE_SSH_HOST}): {e}')
        return []


def write_value_check_csv(data, log_path, ssh_password=None):
    """Write value check entries joined with game outcomes to a timestamped CSV.

    Uses the actor's internal episode number (game_n) as the 'game' column.
    Winner is determined from max(returns) rather than the trigger field.
    If ssh_password is given, also fetches remote-actor value checks from the
    laptop at _REMOTE_SSH_HOST and merges them into the output.
    """
    import csv
    import os

    local_checks = data.get('value_checks', [])

    # Optionally merge remote value checks (actors >= 100000 running on the laptop)
    remote_lines = fetch_remote_value_checks(log_path, ssh_password) if ssh_password else []
    remote_checks = []
    for line in remote_lines:
        m = RE_VALUE_CHECK.search(line)
        if m:
            vals  = [float(v.strip().strip("'")) for v in m.group(4).split(',')]
            exempt = m.group(6) or ('clear_winner' if m.group(7) else None)
            remote_checks.append({
                'actor':  int(m.group(1)),
                'game_n': int(m.group(2)),
                'move':   int(m.group(3)),
                'p0': vals[0] if len(vals) > 0 else None,
                'p1': vals[1] if len(vals) > 1 else None,
                'p2': vals[2] if len(vals) > 2 else None,
                'thresh': float(m.group(5)) if m.group(5) else None,
                'exempt': exempt,
            })

    value_checks = local_checks + remote_checks
    if not value_checks:
        print('  No value check entries found — skipping CSV export.')
        return

    # Build (actor, episode_num) → outcome info.
    # games[i] and received[i] are paired in log order.
    # We use received returns for winner determination since the surviving-games
    # display is also built from received returns.
    game_info = {}
    for i, g in enumerate(data['games']):
        rcv = data['received'][i] if i < len(data['received']) else None
        returns = rcv['returns'] if rcv and rcv.get('returns') else g['returns']
        # Always use max(returns) for winner — the trigger field records which player
        # caused the end-game condition, not necessarily the player with the highest score.
        winner_idx = returns.index(max(returns)) if returns else -1
        if g['is_win']:
            win_type = 'timbuktu' if 'Timbuktu' in g.get('reason', '') else 'rare'
        else:
            win_type = ''
        game_info[(g['actor'], g['game_n'])] = {
            'received_game_num': rcv['game_num'] if rcv else None,
            'outcome':           'win' if g['is_win'] else 'timeout',
            'win_type':          win_type,
            'winner_player':     f'P{winner_idx}' if winner_idx >= 0 else '?',
            'winner_idx':        winner_idx,
            'game_length':       g['moves'],
        }

    log_stem = os.path.splitext(os.path.basename(log_path))[0]
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if os.path.isdir('/media/robp/UD/Projects'):
        out_dir = '/media/robp/UD/Projects/open_spiel'
    else:
        _rel = os.path.join(os.path.dirname(__file__), '../../../../../../Projects/open_spiel')
        _rel = os.path.normpath(_rel)
        out_dir = _rel if os.path.isdir(_rel) else os.path.dirname(os.path.abspath(log_path))
    csv_path = os.path.join(out_dir, f'value_checks_{log_stem}_{ts}.csv')

    rows = []
    for vc in value_checks:
        info = game_info.get((vc['actor'], vc['game_n']))
        if info is None or info['received_game_num'] is None:
            continue  # no matching FINISHED/RECEIVED pair (e.g. early-terminated)
        p_vals = [vc['p0'], vc['p1'], vc['p2']]
        widx = info['winner_idx']
        winner_val = p_vals[widx] if 0 <= widx < len(p_vals) else ''
        exempt_str = f"exempt={vc['exempt']}" if vc['exempt'] else 'exempt=none'
        rows.append([
            info['outcome'], info['win_type'], vc['game_n'], vc['move'],
            f'{winner_val:.4f}' if isinstance(winner_val, float) else winner_val,
            f"{vc['p0']:.4f}", f"{vc['p1']:.4f}", f"{vc['p2']:.4f}",
            info['winner_player'], info['game_length'],
            exempt_str,
        ])

    if not rows:
        print('  No matched value check rows — skipping CSV export.')
        return None

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['outcome', 'win_type', 'game', 'move', 'winner_value',
                         'p0', 'p1', 'p2', 'winner_player', 'game_length', 'exempt'])
        for row in rows:
            writer.writerow(row)

    print(f'  Value check CSV: {csv_path}  ({len(rows)} rows)')
    return csv_path


def plot_value_checks(csv_path):
    """Line chart of winner_value vs move, one line per game, coloured by outcome."""
    import csv
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines

    # Load and sort
    rows = []
    with open(csv_path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append({
                'outcome':      row['outcome'],
                'win_type':     row.get('win_type', ''),
                'game':         int(row['game']),
                'move':         int(row['move']),
                'winner_value': float(row['winner_value']) if row['winner_value'] else None,
            })
    rows.sort(key=lambda r: (r['outcome'], r['game'], r['move']))

    # Group by game
    games = {}
    for r in rows:
        g = r['game']
        if g not in games:
            games[g] = {'outcome': r['outcome'], 'win_type': r['win_type'], 'moves': [], 'values': []}
        if r['winner_value'] is not None:
            games[g]['moves'].append(r['move'])
            games[g]['values'].append(r['winner_value'])

    # Timbuktu wins → blue, Rare goods wins → green, timeouts → red
    def _line_colour(d):
        if d['outcome'] == 'timeout':
            return 'red'
        return 'steelblue' if d['win_type'] == 'timbuktu' else 'green'

    fig, ax = plt.subplots(figsize=(10, 5))
    for g, d in sorted(games.items()):
        colour = _line_colour(d)
        lw = 3.6 if d['outcome'] == 'win' else 1.2
        ax.plot(d['moves'], d['values'], color=colour, alpha=0.6, linewidth=lw,
                label=f"G{g}")

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_xlabel('Move')
    ax.set_ylabel('Value head (winner)')
    ax.set_title('Value checks — winner value by move')

    # Legend: outcome colours only, not individual game lines
    legend_handles = [
        mlines.Line2D([], [], color='steelblue', linewidth=2.5, label='Win — Timbuktu'),
        mlines.Line2D([], [], color='green',     linewidth=2.5, label='Win — Rare goods'),
        mlines.Line2D([], [], color='red',       linewidth=1.2, label='Timeout'),
    ]
    ax.legend(handles=legend_handles)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze a mali_ba training log.')
    parser.add_argument('log_file', help='Path to the training log file')
    parser.add_argument('--window', type=int, default=15,
                        help='Rolling window size for win-rate trend (default: 30)')
    parser.add_argument('--plot', action='store_true', default=None,
                        help='Show charts without prompting')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip charts without prompting')
    parser.add_argument('--ssh-password', default=None, metavar='PASSWORD',
                        help=f'SSH password for {_REMOTE_SSH_USER}@{_REMOTE_SSH_HOST} '
                             f'to fetch remote-actor value checks and merge into CSV')
    parser.add_argument('--show-early-terminations', action='store_true', default=False,
                        help='Include the EARLY TERMINATIONS section (long; off by default)')
    args = parser.parse_args()

    print(f'Parsing {args.log_file} ...')
    data = parse_log(args.log_file)
    report(data, window=args.window, show_early_terminations=args.show_early_terminations)
    print()
    csv_path = write_value_check_csv(data, args.log_file, ssh_password=args.ssh_password)

    if args.no_plot:
        show_charts = False
    elif args.plot:
        show_charts = True
    else:
        try:
            ans = input('Show charts? [Y/n]: ').strip().lower()
            show_charts = ans in ('', 'y', 'yes')
        except (EOFError, KeyboardInterrupt):
            show_charts = False

    if show_charts:
        if csv_path:
            plot_value_checks(csv_path)
        plot_search_cost(data.get('search_costs', []), window=args.window,
                         run_start=data.get('first_time'))
        mcts_games = [g for g in data['games'] if not g['is_bootstrap']]
        plot_win_rate(mcts_games, window=args.window,
                     trainer_losses=data.get('trainer_losses', []),
                     run_start=data.get('first_time'))


if __name__ == '__main__':
    main()
