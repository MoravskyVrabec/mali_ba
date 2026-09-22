#!/usr/bin/env python3
"""Analyze a mali_ba training log and print statistics and trends."""

import re
import sys
import argparse
from collections import defaultdict, Counter
from datetime import datetime


# ── regex patterns ────────────────────────────────────────────────────────────

RE_FINISHED = re.compile(
    r'(\d{2}:\d{2}:\d{2}) \[INFO \] \[Python:0\] (Heuristic )?Actor (\d+), Game (\d+): '
    r'FINISHED in (\d+) moves\. Winner: ([^.]+)\. Reason: \'([^\']+)\'\. '
    r'Triggered by: Player ([-\d]+)\. Final Returns: \[([^\]]+)\]'
    r'(?:\.? ?Intermediate Rewards: \[([^\]]+)\], Rewarded Steps: (\d+))?'
)
RE_RECEIVED = re.compile(
    r'(\d{2}:\d{2}:\d{2}) \[INFO \] \[Python:0\] LEARNER \((Bootstrap|MCTS)\) RECEIVED GAME '
    r'#(\d+)/(\d+)\. Length: (\d+) moves\. Returns: \[([^\]]+)\]'
)
RE_LOSS = re.compile(
    r'(\d{2}:\d{2}:\d{2}) \[INFO \] \[Python:0\] Training: Total Loss=([\d.]+) '
    r'\(Policy=([\d.]+), Value=([\d.]+)\)'
)
RE_PASS_ONLY = re.compile(
    r"MCTS top visits: \['Pass: \d+ visits'\]"
)
RE_BUFFER_STATE = re.compile(
    r'(\d{2}:\d{2}:\d{2}) \[INFO \] \[Python:0\] Trainer processed \d+ new experiences\. '
    r'Bootstrap pool: (\d+)\s+MCTS pool: (\d+)'
)
RE_ADAPTIVE_FRACTION = re.compile(
    r'(\d{2}:\d{2}:\d{2}) \[INFO \] \[Python:0\] Adaptive fraction: .*?mcts_fraction \u2192 ([\d.]+)'
)
RE_TOP_ACTION = re.compile(
    r"MCTS top visits: \['([^:]+): \d+ visits'"
)


def parse_time(t):
    return datetime.strptime(t, '%H:%M:%S')


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
    pass_only_count = 0
    top_action_counter = Counter()
    first_time = None
    last_time = None
    last_finished_is_win = None        # carry is_win from FINISHED to next RECEIVED line
    last_finished_actor = None         # carry actor id from FINISHED to next RECEIVED line
    last_finished_inter_rewards = None # carry inter_rewards from FINISHED to next RECEIVED line
    last_finished_rewarded_steps = None
    last_buffer_state = None      # most recent trainer buffer pool sizes
    last_adaptive_fraction = None # most recent adaptive mcts_fraction value

    with open(path) as f:
        for line in f:
            t_match = re.match(r'(\d{2}:\d{2}:\d{2})', line)
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

            if RE_PASS_ONLY.search(line):
                pass_only_count += 1
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
                    'mcts': int(m.group(3)),
                }
                continue

            m = RE_ADAPTIVE_FRACTION.search(line)
            if m:
                last_adaptive_fraction = float(m.group(2))

    return {
        'games': games,
        'received': received,
        'losses': losses,
        'pass_only_count': pass_only_count,
        'top_action_counter': top_action_counter,
        'first_time': first_time,
        'last_time': last_time,
        'last_buffer_state': last_buffer_state,
        'last_adaptive_fraction': last_adaptive_fraction,
    }


# ── display helpers ───────────────────────────────────────────────────────────

def bar(value, max_value, width=30):
    filled = int(round(value / max_value * width)) if max_value > 0 else 0
    return '█' * filled + '░' * (width - filled)


def pct(n, d):
    return f'{100*n/d:.1f}%' if d > 0 else 'n/a'


def section(title):
    print(f'\n{"─" * 60}')
    print(f'  {title}')
    print(f'{"─" * 60}')


# ── report ────────────────────────────────────────────────────────────────────

def report(data, window=50):
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
        elapsed_min = delta.seconds / 60

    print(f'  Games received by learner : {n_received} / {n_target}  '
          f'(bootstrap: {n_bootstrap_rcvd}, MCTS: {n_mcts_rcvd})')
    print(f'  Added to weights          : {n_bootstrap_added + n_mcts_added}  '
          f'(bootstrap: {n_bootstrap_added}, MCTS: {n_mcts_added})')
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
        total = bs['bootstrap'] + bs['mcts']
        print(f'  Replay buffer (latest)    : {total:,} experiences  '
              f'(bootstrap: {bs["bootstrap"]:,}, MCTS: {bs["mcts"]:,})  '
              f'@ {bs["time"].strftime("%H:%M:%S")}')
        if last_adaptive_fraction is not None:
            print(f'  Adaptive mcts_fraction    : {last_adaptive_fraction:.2f}')

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
            print(f'  {"─"*42}  {"─"*5}  {"─"*5}  {"─"*7}' + '  ───' * n_players_b)
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
              '  ───' * n_players)
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
        print(f'  Win game length: min={min(win_lengths)}  '
              f'avg={sum(win_lengths)/len(win_lengths):.0f}  max={max(win_lengths)}')

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
        for i in range(0, n_mcts_total - window + 1, step):
            chunk = mcts_games[i:i+window]
            chunk_wins = [g for g in chunk if g['is_win']]
            wr = len(chunk_wins) / window
            # Build win-condition breakdown string
            if chunk_wins:
                reason_counts_chunk = Counter(short_reason(g['reason']) for g in chunk_wins)
                n_cw = len(chunk_wins)
                reason_str = ' '.join(
                    f'{r}:{round(c/n_cw*100)}%'
                    for r, c in reason_counts_chunk.most_common()
                )
                reason_part = f' ({reason_str})'
            else:
                reason_part = ''
            print(f'    games {i+1:4d}-{i+window:4d}: {pct(wr*window, window):6s}{reason_part}  '
                  f'{bar(wr, 0.5)}')

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
    if losses:
        # bucket losses into 10 evenly-spaced groups by game number
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
    section('MCTS TOP-ACTION PREFERENCES (most-chosen actions)')
    if top_actions:
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
        max_cat = max(categories.values())
        for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
            print(f'  {cat:>14}: {count:7,}  {pct(count, total_moves):6s}  '
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
                lo_g = chunk[0]['game_n']
                hi_g = chunk[-1]['game_n']
                # Bar: more negative = longer games; scale so -0.25 = empty, 0 = full
                scaled = max(0.0, min(1.0, (avg + 0.25) / 0.25))
                print(f'    games {lo_g:4d}-{hi_g:4d}: {avg:7.4f}  {bar(scaled, 1.0)}')

    # ── Recent games summary ──────────────────────────────────────────────────
    recd_games = 30
    section(f'LAST {recd_games} RECEIVED GAMES')
    for g in received[-recd_games:]:
        r = g['returns']
        max_r = max(r)
        winners = [i for i, v in enumerate(r) if abs(v - max_r) < 1e-6]
        w_str = f'P{winners[0]}' if len(winners) == 1 else 'tie'
        # is_win is set at parse time from the preceding FINISHED line
        is_natural = g.get('is_win')
        if is_natural is None:
            is_natural = g['length'] < 690  # fallback if FINISHED line was missing
        flag = '★' if is_natural else ' '
        phase_tag = 'B' if g.get('phase') == 'Bootstrap' else 'M'
        actor = g.get('actor')
        actor_str = f'A{actor}' if actor is not None else '   ?'
        remote_tag = 'R' if actor is not None and actor >= 100000 else ' '
        ir = g.get('inter_rewards')
        ir_str = f'  inter: [{", ".join(f"{v:.3f}" for v in ir)}]' if ir else ''
        print(f'  {flag}{phase_tag}{remote_tag} Game #{g["game_num"]:4d}  {g["length"]:3d} moves  '
              f'{actor_str:<8}  winner: {w_str}  returns: {[f"{v:.1f}" for v in r]}{ir_str}')


def plot_win_rate(mcts_games, window=50):
    """Plot a moving average of MCTS win rate over the course of the run."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available. Install with: pip install matplotlib')
        return

    if len(mcts_games) < window:
        print(f'  Not enough MCTS games for a {window}-game moving average (have {len(mcts_games)}).')
        return

    # Compute moving average at every game position
    is_win = [1 if g['is_win'] else 0 for g in mcts_games]
    xs = []
    ys = []
    for i in range(window - 1, len(is_win)):
        xs.append(i + 1)  # game number (1-indexed at end of window)
        ys.append(sum(is_win[i - window + 1:i + 1]) / window)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs, [y * 100 for y in ys], linewidth=1.5, color='steelblue', label=f'{window}-game moving avg')
    ax.axhline(y=sum(is_win) / len(is_win) * 100, color='gray', linestyle='--', linewidth=1,
               label=f'Overall avg ({sum(is_win)/len(is_win)*100:.1f}%)')
    ax.set_xlabel('MCTS Game #')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'MCTS Win Rate — {window}-game Moving Average')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze a mali_ba training log.')
    parser.add_argument('log_file', help='Path to the training log file')
    parser.add_argument('--window', type=int, default=50,
                        help='Rolling window size for win-rate trend (default: 50)')
    parser.add_argument('--plot', action='store_true', default=None,
                        help='Show moving average plot without prompting')
    parser.add_argument('--no-plot', action='store_true',
                        help='Skip moving average plot without prompting')
    args = parser.parse_args()

    print(f'Parsing {args.log_file} ...')
    data = parse_log(args.log_file)
    report(data, window=args.window)
    print()

    # Determine whether to show the plot
    if args.no_plot:
        show_plot = False
    elif args.plot:
        show_plot = True
    else:
        try:
            ans = input('Show moving average plot? [Y/n]: ').strip().lower()
            show_plot = ans in ('', 'y', 'yes')
        except (EOFError, KeyboardInterrupt):
            show_plot = False

    if show_plot:
        mcts_games = [g for g in data['games'] if not g['is_bootstrap']]
        plot_win_rate(mcts_games, window=args.window)


if __name__ == '__main__':
    main()
