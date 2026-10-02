"""
pretrain_heuristic.py

Pretrains the policy and value networks on many heuristic games, and writes an
ordinary checkpoint that train_mali_ba.py can start from (load_model_path).

Why (2026-10-01): the replay buffer holds 100k positions but only ~240 games,
because it keeps every move of every game. A value head trained on it learns to
recognise which game a position came from: 100% winner top-pick on games it trained
on, ~50% (a coin flip) on any new game. Trained on ~9,200 heuristic games instead,
the same network picked the winner of MCTS games it had never seen 71-78% of the
time. Larger networks still start to overfit those games after ~10k steps, so this
tool holds out whole games and keeps the weights from the best evaluation.

Steps
  1. Generate --games heuristic games in parallel (--workers processes). Moves are
     the C++ heuristic's weighted choice, with --random_move_prob uniformly random
     moves for variety. Only 1 position in --keep_every is kept, so the data spans
     many games without filling memory. Saved under --data_dir and reused on later
     runs with the same settings.
     Wins are rare in heuristic play (~6%), so a value head pretrained on these games
     learned that nearly every game heads for a timeout (D003: eventual winners rated
     ~-0.25 against a target of +0.93). --min_wins keeps playing until that many win
     games are stored, and --win_keep_every keeps more positions from each of them.
     Their value loss is weighted back to the natural win rate (--win_weighting), so
     the extra wins add variety without inflating how often wins are expected.
  2. Hold out --holdout of the games (whole games).
  3. Train: value head on the trainer's exact value target (V_t = r_t + gamma *
     V_{t+1}, V_end = final returns; gamma from the ini); policy head on the
     heuristic's normalised action weights, as the trainer's bootstrap actors do.
  4. Every --eval_every steps, score both heads on the held-out games (and, with
     --eval_buffer, on MCTS games from a saved replay buffer). Keep each head's best
     weights -- the value head's by its error on the MCTS games when --eval_buffer is
     given (--value_select), otherwise on the held-out games; stop after --patience
     evaluations without improvement.
  5. Save the best weights with SimpleAgent.save_model(): <out> becomes
     <stem>._policy.weights.h5 and <stem>._value.weights.h5, plus <stem>.pretrain.json.

Usage (run from this directory, with the same environment as train_mali_ba.py):
  python pretrain_heuristic.py --out mali_ba_agent_vP001.weights.h5
  python pretrain_heuristic.py --out mali_ba_agent_vP001.weights.h5 \\
         --eval_buffer ../../../../../open_spiel/mali_ba_buffer.pkl.gz
  # Win-enriched value pretraining, keeping an existing checkpoint's policy:
  python pretrain_heuristic.py --out mali_ba_agent_vP003.weights.h5 --heads value \\
         --init_from mali_ba_agent_vP001.weights.h5 --fresh_value \\
         --min_wins 1500 --win_keep_every 4 --eval_buffer ...
  # Retrain only the value head, keeping an existing checkpoint's policy:
  python pretrain_heuristic.py --out mali_ba_agent_vP002.weights.h5 --heads value \\
         --init_from mali_ba_agent_vP001.weights.h5 --fresh_value --eval_buffer ...

Memory: each kept position is ~87 KB (192x15x15 float16 + policy). The defaults
(10,000 games, 1 in 16 kept, ~27 positions per game) need ~25 GB of RAM.
"""

import argparse
import configparser
import glob
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

SHARD_FORMAT = 2   # 2: adds Y (position comes from a win game) and win enrichment


def say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------
# 1. Game generation (worker processes import pyspiel only, never TensorFlow)
# --------------------------------------------------------------------------------

def _value_targets(rewards, returns, gamma):
    """Trainer's value target for every move: V_t = r_t + gamma * V_{t+1}, with the
    value after the last move equal to the final returns (train_mali_ba.py)."""
    out = np.empty((len(rewards), len(returns)), np.float32)
    nxt = np.asarray(returns, np.float64)
    for t in range(len(rewards) - 1, -1, -1):
        nxt = np.asarray(rewards[t], np.float64) + gamma * nxt
        out[t] = nxt
    return out


def _play_game(game, num_actions, rng, cfg):
    """One heuristic game: (observations, policy targets, value targets, returns),
    or None if it did not finish."""
    import pyspiel
    state = game.new_initial_state()
    pyspiel.mali_ba.downcast_state(state).seed_rng(rng.randrange(2**31 - 1))
    obs, pol, rew = [], [], []
    while not state.is_terminal():
        legal = state.legal_actions()
        if not legal:
            break
        if state.current_player() < 0:
            # Chance node (the opening one picks one of 65,536 meeple layouts).
            state.apply_action(rng.choice(legal))
            continue
        ms = pyspiel.mali_ba.downcast_state(state)
        policy = np.zeros(num_actions, np.float32)
        weights = ms.get_heuristic_action_weights()
        total = sum(weights.values())
        if total > 0:
            for a, w in weights.items():
                policy[a] = w / total
        else:
            policy[legal] = 1.0 / len(legal)
        obs.append(np.asarray(state.observation_tensor(), np.float16))
        pol.append(policy.astype(np.float16))
        action = ms.select_heuristic_random_action()
        if action not in legal or rng.random() < cfg['random_move_prob']:
            action = rng.choice(legal)
        state.apply_action(action)
        rew.append(list(state.rewards()))
    if not state.is_terminal() or not obs:
        return None
    returns = list(state.returns())
    return obs, pol, _value_targets(rew, returns, cfg['gamma']), returns


def _generate_worker(worker, quota, cfg):
    """Play games and store this worker's share.

    Plain mode (quota['wins'] == 0): play quota['games'] games and store them all.
    Win-enriched mode: keep playing until quota['games'] timeouts AND quota['wins']
    wins have been stored (or quota['max_games'] have been played); games beyond a
    full quota are played but not stored. Win games keep 1 position in
    cfg['win_keep_every'], timeouts 1 in cfg['keep_every'].
    """
    import pyspiel
    game = pyspiel.load_game("mali_ba", cfg['game_params'])
    num_actions = game.num_distinct_actions()
    rng = random.Random(cfg['seed'] * 1000 + worker)
    X, PT, T, W, K, G, Y = [], [], [], [], [], [], []
    counts = dict(played_win=0, played_timeout=0, stored_win=0, stored_timeout=0, skipped=0)
    enriched = quota['wins'] > 0
    t0 = time.time()
    gi = 0
    while True:
        if enriched:
            if ((counts['stored_timeout'] >= quota['games'] and counts['stored_win'] >= quota['wins'])
                    or gi >= quota['max_games']):
                break
        elif gi >= quota['games']:
            break
        game_id = worker * 1_000_000 + gi
        gi += 1
        played = _play_game(game, num_actions, rng, cfg)
        if played is None:
            counts['skipped'] += 1
            continue
        obs, pol, targets, returns = played
        is_win = max(returns) >= 1.0
        counts['played_win' if is_win else 'played_timeout'] += 1
        if enriched and (counts['stored_win'] >= quota['wins'] if is_win
                         else counts['stored_timeout'] >= quota['games']):
            continue
        counts['stored_win' if is_win else 'stored_timeout'] += 1
        every = cfg['win_keep_every'] if is_win else cfg['keep_every']
        winner = int(np.argmax(returns))
        L = len(obs)
        for i in range(rng.randrange(every), L, every):
            X.append(obs[i]); PT.append(pol[i]); T.append(targets[i])
            W.append(winner); K.append(L - 1 - i); G.append(game_id); Y.append(is_win)
        if gi % 50 == 0:
            say(f"  worker {worker:2d}: {gi} games played, stored {counts['stored_timeout']} timeouts "
                f"+ {counts['stored_win']} wins ({(time.time() - t0) / gi:.2f} s/game)")
    path = os.path.join(cfg['data_dir'], f"shard_{worker:03d}.npz")
    np.savez(path, X=np.stack(X), P=np.stack(PT), T=np.array(T, np.float32),
             W=np.array(W, np.int8), K=np.array(K, np.int16), G=np.array(G, np.int64),
             Y=np.array(Y, bool))
    return worker, len(W), counts


def generate(args, cfg):
    os.makedirs(args.data_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.data_dir, "shard_*.npz")):
        os.remove(old)
    def split(total):
        return [total // args.workers + (1 if w < total % args.workers else 0)
                for w in range(args.workers)]
    games, wins, caps = split(args.games), split(args.min_wins), split(args.max_games)
    if args.min_wins:
        say(f"Generating heuristic games on {args.workers} workers until {args.games:,} timeouts "
            f"and {args.min_wins:,} wins are stored (at most {args.max_games:,} games); "
            f"random moves {args.random_move_prob:.0%}; keeping 1 position in {args.keep_every} "
            f"(timeouts) / 1 in {args.win_keep_every} (wins) ...")
    else:
        say(f"Generating {args.games:,} heuristic games on {args.workers} workers "
            f"(random moves {args.random_move_prob:.0%}, keeping 1 position in {args.keep_every}"
            + (f", 1 in {args.win_keep_every} for wins" if args.win_keep_every != args.keep_every else "")
            + ") ...")
    totals = dict(played_win=0, played_timeout=0, stored_win=0, stored_timeout=0, skipped=0)
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        results = [pool.apply_async(_generate_worker,
                                    (w, dict(games=games[w], wins=wins[w], max_games=caps[w]), cfg))
                   for w in range(args.workers)]
        n = 0
        for r in results:
            _, k, counts = r.get()
            n += k
            for key in totals:
                totals[key] += counts[key]
    meta = dict(cfg_key(args, cfg), positions=n, outcomes=totals, created=time.strftime('%Y-%m-%d %H:%M'))
    with open(os.path.join(args.data_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    played = totals['played_win'] + totals['played_timeout']
    say(f"Generated {n:,} positions. Played {played:,} games ({totals['played_win']:,} wins = "
        f"{100 * totals['played_win'] / max(1, played):.1f}%); stored {totals['stored_timeout']:,} "
        f"timeouts + {totals['stored_win']:,} wins; skipped {totals['skipped']}.")
    if args.min_wins and totals['stored_win'] < args.min_wins:
        say(f"WARNING: only {totals['stored_win']:,} of {args.min_wins:,} wins before --max_games.")


def cfg_key(args, cfg):
    """Settings that define a dataset; a saved dataset is reused only if they match."""
    return dict(format=SHARD_FORMAT, games=args.games, keep_every=args.keep_every,
                win_keep_every=args.win_keep_every, min_wins=args.min_wins,
                max_games=args.max_games if args.min_wins else None,
                random_move_prob=args.random_move_prob, gamma=cfg['gamma'],
                seed=cfg['seed'], config_file=os.path.abspath(args.config_file),
                obs_shape=cfg['obs_shape'])


def load_data(args, cfg):
    meta_path = os.path.join(args.data_dir, "meta.json")
    shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
    reuse = False
    if os.path.exists(meta_path) and shards:
        with open(meta_path) as f:
            meta = json.load(f)
        want = cfg_key(args, cfg)
        reuse = all(meta.get(k) == v for k, v in want.items())
        if not reuse:
            diff = {k: (meta.get(k), v) for k, v in want.items() if meta.get(k) != v}
            say(f"Saved data in {args.data_dir} was made with different settings {diff}; regenerating.")
    if reuse and not args.regenerate:
        say(f"Reusing {meta['positions']:,} positions in {args.data_dir} (made {meta.get('created')}).")
    else:
        generate(args, cfg)
        shards = sorted(glob.glob(os.path.join(args.data_dir, "shard_*.npz")))
        with open(meta_path) as f:
            meta = json.load(f)

    # Fill preallocated arrays shard by shard: concatenating would briefly need twice the RAM.
    sizes = []
    for s in shards:
        with np.load(s) as z:
            sizes.append(len(z['W']))
    n = sum(sizes)
    shape = tuple(cfg['obs_shape'])
    with np.load(shards[0]) as z:
        num_actions = z['P'].shape[1]
    data = dict(X=np.empty((n,) + shape, np.float16), P=np.empty((n, num_actions), np.float16),
                T=np.empty((n, cfg['num_players']), np.float32), W=np.empty(n, np.int8),
                K=np.empty(n, np.int16), G=np.empty(n, np.int64), Y=np.empty(n, bool))
    i = 0
    for s, m in zip(shards, sizes):
        with np.load(s) as z:
            data['X'][i:i + m] = z['X'].reshape((m,) + shape)
            for key in ('P', 'T', 'W', 'K', 'G', 'Y'):
                data[key][i:i + m] = z[key]
        i += m

    # Value-loss weight per position. Win games are stored more densely (win_keep_every)
    # and, with --min_wins, more often than they occur, which gives the value head many
    # more distinct win positions to learn from. Weighting them back to their natural
    # share keeps its predictions calibrated to real play (it would otherwise expect
    # wins far more often than they happen). --win_weighting none skips this.
    o = meta['outcomes']
    natural = o['played_win'] / max(1, o['played_timeout'])
    stored = o['stored_win'] / max(1, o['stored_timeout'])
    win_w = 1.0
    if args.win_weighting == 'natural' and o['stored_win'] and stored > 0:
        win_w = (args.win_keep_every / args.keep_every) * natural / stored
    data['V'] = np.where(data['Y'], win_w, 1.0).astype(np.float32)
    data['V'] /= data['V'].mean()
    n_games = len(np.unique(data['G']))
    n_win_games = len(np.unique(data['G'][data['Y']]))
    say(f"Loaded {n:,} positions from {n_games:,} games ({n_win_games:,} wins, "
        f"{int(data['Y'].sum()):,} win positions). Natural win rate "
        f"{100 * o['played_win'] / max(1, o['played_win'] + o['played_timeout']):.1f}%; "
        f"value-loss weight of a win position relative to a timeout position: {win_w:.3f}.")
    return data


# --------------------------------------------------------------------------------
# 2. Optional evaluation set: MCTS games from a saved replay buffer
# --------------------------------------------------------------------------------

def load_eval_buffer(path, obs_shape, max_play_moves, target_every=4):
    """MCTS positions from a saved replay buffer, with each game's winner.

    Games are found where the progress plane (moves / max_play_moves) drops. "Moves
    to the end" (K) is measured with that plane from the game's last stored
    position, so it stays in moves when the buffer was thinned (buffer_keep_every);
    the last stored position can then be up to N-1 moves before the true end.
    """
    import gzip
    import pickle
    from buffer_format import normalize_saved_buffer, saved_keep_every
    say(f"Loading MCTS evaluation games from {path} ...")
    with gzip.open(path, 'rb') as f:
        saved = pickle.load(f)
    pools, _ = normalize_saved_buffer(saved)
    keep = saved_keep_every(saved)
    del saved
    every = max(1, round(target_every / keep))   # aim for ~1 position in 4 moves
    planes = obs_shape[0]
    progress_plane = 95  # moves / max_play_moves
    if planes <= progress_plane:
        say("  observation has no progress plane; cannot split the buffer into games. Skipping.")
        return None
    X, T, W, K, Y = [], [], [], [], []
    n_games = n_win = 0
    for key in ('mcts_timeout_buffer', 'mcts_nearwin_buffer', 'mcts_raregoods_buffer',
                'mcts_timbuktu_buffer'):
        ent = pools[key]
        pools[key] = None
        if not ent:
            continue
        obs = [np.asarray(e[0], np.float16).reshape(obs_shape) for e in ent]
        tg = [np.asarray(e[2][2], np.float32) for e in ent]
        del ent
        prog = [float(o[progress_plane, 0, 0]) for o in obs]
        starts = [0] + [i for i in range(1, len(obs)) if prog[i] < prog[i - 1] - 1e-3]
        starts.append(len(obs))
        for a, b in zip(starts[:-1], starts[1:]):
            w = int(np.argmax(tg[b - 1]))
            is_win = float(np.max(tg[b - 1])) > 0.5   # a win's target is ~+0.8..1, a timeout's <= ~0.1
            n_games += 1
            n_win += is_win
            for i in range(a, b, every):
                X.append(obs[i]); T.append(tg[i]); W.append(w); Y.append(is_win)
                K.append(int(round((prog[b - 1] - prog[i]) * max_play_moves)))
    say(f"  {len(W):,} positions from {n_games} MCTS games ({n_win} wins); buffer kept 1 in "
        f"{keep} positions, using 1 in {every} of those.")
    return dict(X=np.stack(X), T=np.array(T), W=np.array(W), K=np.array(K), Y=np.array(Y, bool))


# --------------------------------------------------------------------------------
# 3. Training
# --------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Pretrain policy and value networks on heuristic games.")
    ap.add_argument('--out', required=True, help="Checkpoint path, e.g. mali_ba_agent_vP001.weights.h5")
    ap.add_argument('--config_file', default='mali_ba.ini')
    ap.add_argument('--games', type=int, default=10000)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument('--keep_every', type=int, default=16, help="Keep 1 position in N per game")
    ap.add_argument('--win_keep_every', type=int, default=None,
                    help="Keep 1 position in N from games that end in a win (default: --keep_every). "
                         "Lower than --keep_every gives the value head more win positions.")
    ap.add_argument('--min_wins', type=int, default=0,
                    help="Keep playing until this many win games are stored, alongside --games "
                         "timeouts (heuristic games are only ~6%% wins). 0 = just play --games games.")
    ap.add_argument('--max_games', type=int, default=None,
                    help="With --min_wins: stop after this many games regardless (default 5 x --games)")
    ap.add_argument('--win_weighting', choices=('natural', 'none'), default='natural',
                    help="natural: weight win positions in the value loss back to how often wins "
                         "really occur, so extra wins add variety without making the value head "
                         "expect wins too often. none: train on the enriched mix as is.")
    ap.add_argument('--random_move_prob', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--data_dir', default=None, help="Default: pretrain_data/heur_<games>_k<keep_every>")
    ap.add_argument('--regenerate', action='store_true', help="Regenerate even if matching data exists")
    ap.add_argument('--heads', choices=('both', 'value', 'policy'), default='both')
    ap.add_argument('--init_from', default=None, help="Checkpoint to start from (default: random init)")
    ap.add_argument('--steps', type=int, default=40000, help="Maximum training steps")
    ap.add_argument('--eval_every', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=8, help="Evaluations without improvement before stopping")
    ap.add_argument('--batch_size', type=int, default=None, help="Default: ini, else 128")
    ap.add_argument('--learning_rate', type=float, default=None, help="Default: ini, else 0.0002")
    ap.add_argument('--holdout', type=float, default=0.05, help="Fraction of games held out")
    ap.add_argument('--eval_buffer', default=None, help="Saved replay buffer of MCTS games to also score on")
    ap.add_argument('--value_select', choices=('auto', 'heldout', 'mcts'), default='auto',
                    help="Which error picks the value head's best step: held-out heuristic games, "
                         "or the --eval_buffer MCTS games. auto = mcts when --eval_buffer is given. "
                         "Selecting on the MCTS games makes their reported winner rate a few points "
                         "optimistic, since they are then also the selection set.")
    ap.add_argument('--fresh_value', action='store_true',
                    help="With --init_from: load only the policy and start the value head from "
                         "random weights")
    args = ap.parse_args()
    if args.win_keep_every is None:
        args.win_keep_every = args.keep_every
    if args.max_games is None:
        args.max_games = 5 * args.games
    if args.data_dir is None:
        name = f"heur_{args.games}_k{args.keep_every}"
        if args.win_keep_every != args.keep_every or args.min_wins:
            name += f"_wk{args.win_keep_every}_mw{args.min_wins}"
        args.data_dir = os.path.join('pretrain_data', name)

    ini = configparser.ConfigParser()
    ini.read(args.config_file)
    def ini_get(key, fallback, conv):
        try:
            return conv(ini.get('MLTraining', key))
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return fallback
    gamma = ini_get('gamma', 0.997, float)
    batch_size = args.batch_size or ini_get('batch_size', 128, int)
    lr = args.learning_rate or ini_get('learning_rate', 0.0002, float)

    import pyspiel
    game_params = {"config_file": args.config_file, "player_types": "ai,ai,ai"}
    game = pyspiel.load_game("mali_ba", game_params)
    obs_shape = list(game.observation_tensor_shape())
    cfg = dict(game_params=game_params, gamma=gamma, seed=args.seed, keep_every=args.keep_every,
               win_keep_every=args.win_keep_every,
               random_move_prob=args.random_move_prob, data_dir=args.data_dir,
               obs_shape=obs_shape, num_players=game.num_players())
    say(f"Observation {obs_shape}, {game.num_distinct_actions()} actions, gamma {gamma}, "
        f"batch {batch_size}, learning rate {lr}")

    data = load_data(args, cfg)
    games = np.unique(data['G'])
    hold_rng = np.random.default_rng(args.seed)
    held_games = games[hold_rng.random(len(games)) < args.holdout]
    held = np.isin(data['G'], held_games)
    train_idx, held_idx = np.where(~held)[0], np.where(held)[0]
    say(f"Training on {len(train_idx):,} positions; holding out {len(held_games):,} games "
        f"({len(held_idx):,} positions).")
    max_play_moves = 420
    try:
        import re
        with open(args.config_file) as f:
            m = re.search(r'^\s*max_play_moves\s*=\s*(\d+)', f.read(), re.M)
        if m:
            max_play_moves = int(m.group(1))
    except OSError:
        pass
    mcts = load_eval_buffer(args.eval_buffer, obs_shape, max_play_moves) if args.eval_buffer else None

    import tensorflow as tf
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    from training_utils import SimpleAgent
    agent = SimpleAgent(obs_shape, game.num_distinct_actions(), game.num_players(), lr)
    loaded_from_init = []
    if args.init_from:
        initial_value = agent.value_model.get_weights()
        loaded_from_init = agent.load_model(args.init_from)
        if args.fresh_value and 'value' in loaded_from_init:
            agent.value_model.set_weights(initial_value)
            loaded_from_init.remove('value')
            say("--fresh_value: value head reset to random weights.")
        say(f"Starting from {args.init_from}: loaded {loaded_from_init}")
    pm, vm = agent.policy_model, agent.value_model
    value_select = args.value_select
    if value_select == 'auto':
        value_select = 'mcts' if mcts is not None else 'heldout'
    if value_select == 'mcts' and mcts is None:
        sys.exit("--value_select mcts needs --eval_buffer")
    value_key = 'mcts_value_mse' if value_select == 'mcts' else 'value_mse'
    say(f"Value head's best step chosen by {'MCTS-game' if value_select == 'mcts' else 'held-out heuristic'} error.")
    train_policy = args.heads in ('both', 'policy')
    train_value = args.heads in ('both', 'value')
    p_opt = tf.keras.optimizers.Adam(lr)
    v_opt = tf.keras.optimizers.Adam(lr)
    cce = tf.keras.losses.CategoricalCrossentropy()
    mse = tf.keras.losses.MeanSquaredError()

    @tf.function
    def policy_step(x, p):
        with tf.GradientTape() as tape:
            loss = cce(p, pm(x, training=True))
        p_opt.apply_gradients(zip(tape.gradient(loss, pm.trainable_variables), pm.trainable_variables))
        return loss

    @tf.function
    def value_step(x, t, w):
        with tf.GradientTape() as tape:
            loss = mse(t, vm(x, training=True), sample_weight=w)
            obj = loss + tf.add_n(vm.losses) if vm.losses else loss
        v_opt.apply_gradients(zip(tape.gradient(obj, vm.trainable_variables), vm.trainable_variables))
        return loss

    def predict(model, X):
        return np.concatenate([model(tf.constant(X[j:j + 1024], tf.float32), training=False).numpy()
                               for j in range(0, len(X), 1024)])

    def top1(pred, winners, mask):
        return 100.0 * np.mean(pred[mask].argmax(1) == winners[mask]) if mask.any() else float('nan')

    def evaluate():
        r = {}
        X = data['X'][held_idx]
        if train_value:
            v = predict(vm, X)
            k, w, y = data['K'][held_idx], data['W'][held_idx], data['Y'][held_idx]
            # Weighted like training, so this measures error at the natural win rate.
            r['value_mse'] = float(np.average(np.mean((v - data['T'][held_idx]) ** 2, axis=1),
                                              weights=data['V'][held_idx]))
            r['value_top1_last20'] = top1(v, w, k <= 20)
            # Does it see wins coming? The eventual winner's predicted value in win games
            # (their target is ~+0.8..0.95), and how often it ranks that player first.
            if y.any():
                r['win_winner_pred'] = float(np.mean(v[np.where(y)[0], w[y]]))
                r['win_top1'] = top1(v, w, y)
            if mcts is not None:
                mv = predict(vm, mcts['X'])
                r['mcts_value_mse'] = float(np.mean((mv - mcts['T']) ** 2))
                r['mcts_top1_last20'] = top1(mv, mcts['W'], mcts['K'] <= 20)
                r['mcts_top1_21_60'] = top1(mv, mcts['W'], (mcts['K'] > 20) & (mcts['K'] <= 60))
                my = mcts['Y']
                if my.any():
                    r['mcts_win_winner_pred'] = float(np.mean(mv[np.where(my)[0], mcts['W'][my]]))
                    r['mcts_win_top1'] = top1(mv, mcts['W'], my)
        if train_policy:
            p = predict(pm, X)
            target = data['P'][held_idx].astype(np.float32)
            r['policy_ce'] = float(np.mean(-np.sum(target * np.log(np.clip(p, 1e-7, 1.0)), axis=1)))
            r['policy_top1'] = 100.0 * float(np.mean(p.argmax(1) == target.argmax(1)))
        return r

    # Score-plane baseline: how often simply reading the scores picks the winner.
    if len(obs_shape) == 3 and obs_shape[0] >= 99:
        k = data['K'][held_idx]
        sc = data['X'][held_idx][:, 96:99, 0, 0].astype(np.float32).argmax(1)
        say(f"Reference: reading the score planes picks the winner {top1(np.eye(3)[sc], data['W'][held_idx], k <= 20):.1f}% "
            f"of the time in the last 20 moves of held-out games"
            + (f"; {top1(np.eye(3)[mcts['X'][:, 96:99, 0, 0].astype(np.float32).argmax(1)], mcts['W'], mcts['K'] <= 20):.1f}% on MCTS games"
               if mcts is not None else "") + ".")

    best = {'value': (float('inf'), None, 0), 'policy': (float('inf'), None, 0)}
    history = []
    since_improved = 0
    rng = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    p_loss = v_loss = float('nan')
    for step in range(1, args.steps + 1):
        idx = np.sort(rng.choice(train_idx, batch_size))
        x = tf.constant(data['X'][idx], tf.float32)
        if train_policy:
            p_loss = float(policy_step(x, tf.constant(data['P'][idx], tf.float32)))
        if train_value:
            v_loss = float(value_step(x, tf.constant(data['T'][idx]), tf.constant(data['V'][idx])))
        if step % args.eval_every and step != args.steps:
            continue
        r = evaluate()
        r.update(step=step, train_policy_loss=p_loss, train_value_loss=v_loss)
        history.append(r)
        improved = []
        if train_value and r[value_key] < best['value'][0]:
            best['value'] = (r[value_key], vm.get_weights(), step); improved.append('value')
        if train_policy and r['policy_ce'] < best['policy'][0]:
            best['policy'] = (r['policy_ce'], pm.get_weights(), step); improved.append('policy')
        since_improved = 0 if improved else since_improved + 1
        parts = [f"step {step:6d} ({time.time() - t0:5.0f}s)"]
        if train_value:
            parts.append(f"value: held-out mse {r['value_mse']:.4f}, winner {r['value_top1_last20']:.1f}%"
                         + (f", win games: winner pred {r['win_winner_pred']:+.2f} top {r['win_top1']:.0f}%"
                            if 'win_winner_pred' in r else ""))
            if mcts is not None:
                parts.append(f"MCTS games: mse {r['mcts_value_mse']:.4f}, winner {r['mcts_top1_last20']:.1f}% "
                             f"(21-60 moves out {r['mcts_top1_21_60']:.1f}%)"
                             + (f", win games: winner pred {r['mcts_win_winner_pred']:+.2f} "
                                f"top {r['mcts_win_top1']:.0f}%" if 'mcts_win_winner_pred' in r else ""))
        if train_policy:
            parts.append(f"policy: held-out CE {r['policy_ce']:.3f}, top-1 {r['policy_top1']:.1f}%")
        say(" | ".join(parts) + (f"  * best {'/'.join(improved)}" if improved else ""))
        if since_improved >= args.patience:
            say(f"No improvement in {args.patience} evaluations; stopping.")
            break

    # Restore each head's best weights and save.
    if train_value and best['value'][1] is not None:
        vm.set_weights(best['value'][1])
    if train_policy and best['policy'][1] is not None:
        pm.set_weights(best['policy'][1])
    # Save only heads that were trained here or loaded from --init_from: a head left at
    # its random initialisation would otherwise be written out and later loaded by the
    # trainer as if it were real. File names match SimpleAgent.save_model().
    for name, model, keep in (("policy", pm, train_policy or "policy" in loaded_from_init),
                              ("value", vm, train_value or "value" in loaded_from_init)):
        path = args.out.replace("weights.h5", f"_{name}.weights.h5")
        if keep:
            model.save_weights(path)
        else:
            say(f"Not saving the {name} head: it was neither trained nor loaded (random weights).")
    stem = args.out[:-len("weights.h5")] if args.out.endswith("weights.h5") else args.out + "."
    report = dict(out=args.out, data_dir=args.data_dir, heads=args.heads, init_from=args.init_from,
                  fresh_value=args.fresh_value, value_select=value_select,
                  keep_every=args.keep_every, win_keep_every=args.win_keep_every,
                  min_wins=args.min_wins, win_weighting=args.win_weighting,
                  gamma=gamma, batch_size=batch_size, learning_rate=lr,
                  train_positions=int(len(train_idx)), held_out_games=int(len(held_games)),
                  best_value_step=best['value'][2], best_policy_step=best['policy'][2],
                  history=history)
    with open(stem + "pretrain.json", "w") as f:
        json.dump(report, f, indent=1)
    trained = [f"{h} from step {best[h][2]}" for h, on in (("value", train_value), ("policy", train_policy)) if on]
    say(f"Saved {args.out} ({', '.join(trained)}); history in {stem}pretrain.json")
    say(f"Start a run from it with: train_mali_ba.py ... --load_model_path {args.out}")


if __name__ == '__main__':
    main()
