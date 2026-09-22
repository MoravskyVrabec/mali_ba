# Mali-Ba

A 3-player strategic board game based on West African trade routes and mancala mechanics, implemented in OpenSpiel (C++) with AlphaZero-style self-play training in Python.

## Project Structure

```
open_spiel/python/games/mali_ba/
│
├── train_mali_ba.py        # Main training script — AlphaZero self-play loop
├── training_utils.py       # ReplayBuffer (4 pools), SimpleAgent, AlphaZeroEvaluator, networks
├── mali_ba.ini             # All configuration ([GameSettings], [Board], [Rules], [Heuristics],
│                           #   [Training] read by C++, [MLTraining] read by Python)
│
├── queue_server.py         # Distributed training: BaseManager TCP server for job/result queues
├── remote_actors.py        # Distributed training: spawns actor processes on remote machines
│
├── analyze_log.py          # Parse training logs, print stats, export value-check CSV, plot charts
├── resize_buffer.py        # Trim or expand replay buffer pools without restarting a run
├── strip_bootstrap.py      # Remove bootstrap experiences from a saved buffer file
├── plot_experiment.py      # Plot heuristic experiment results
│
├── main.py                 # GUI entry point
├── config.py               # Game constants (city IDs, cultural groups, rare goods names)
├── dev_config.py           # Development configuration overrides
├── setup.py                # Package installation
│
├── classes/
│   ├── game_state.py       # Python-side game state wrapper
│   └── classes_other.py    # Supporting game data classes
│
├── ui/
│   ├── gui_other.py        # Main GUI window and interaction
│   ├── visualizer.py       # Board rendering entry point
│   ├── visualizer_drawing.py  # Low-level drawing primitives
│   ├── visualizer_other.py    # Additional visualizer components
│   └── mali_ba_map.jpg     # Board background map
│
└── utils/
    ├── cpp_interface.py        # Interface to C++ OpenSpiel game object
    ├── configurable_ai_bot.py  # Configurable AI bot (wraps MCTS + trained model)
    ├── board_config.py         # Board hex/region configuration helpers
    ├── player_config.py        # Player configuration helpers
    ├── simple_ai_integration.py  # Simple AI integration utilities
    └── parsing.py              # INI/config parsing utilities
```

The C++ game source lives in `open_spiel/games/mali_ba/` (separate from this directory). The compiled shared library (`pyspiel.so`) is built there and imported here.

## AlphaZero Training

### Quick Start

```bash
# From the WSL training root (where pyspiel.so is accessible)
cd /media/robp/UD/Projects/open_spiel/open_spiel/python/games/mali_ba

python train_mali_ba.py \
  --config_file mali_ba.ini \
  --num_actors 24 \
  --num_episodes 5000 \
  --save_model_path mali_ba_agent_v09.weights.h5 \
  --save_buffer_path mali_ba_buffer.pkl.gz \
  2>&1 | tee ~/train_runXX.log
```

### Replay Buffer

The buffer is a gzip-pickled dict with **four pools**, saved as `mali_ba_buffer.pkl.gz`:

| Pool | Key | Contents |
|------|-----|----------|
| bootstrap | `bootstrap_buffer` | Heuristic self-play games (~95% Rare goods wins) |
| natural | `mcts_natural_buffer` | MCTS Timbuktu natural wins |
| nearwin | `mcts_nearwin_buffer` | MCTS near-win timeout games (470 moves) |
| raregoods | `mcts_raregoods_buffer` | MCTS Rare goods natural wins — guaranteed per-batch exposure |

The raregoods pool exists to break Timbuktu mode collapse: Timbuktu wins are easier to find via MCTS than Rare goods wins, which can cause the buffer to skew heavily toward Timbuktu and prevent the model from learning the Rare goods win condition. The raregoods pool guarantees a configurable fraction of every training batch comes from Rare goods wins regardless of their frequency in self-play.

Key INI parameters in `[MLTraining]`:

```ini
replay_buffer_size = 50000
mcts_buffer_fraction = 0.90       # MCTS share of total buffer (vs bootstrap)
near_win_pool_fraction = 0.20     # Nearwin share of MCTS capacity
raregoods_pool_fraction = 0.40    # Rare goods share of MCTS capacity (and batch)
```

### Resizing or Manipulating the Buffer

```bash
# Show all pool sizes and trim/expand as needed
python resize_buffer.py mali_ba_buffer.pkl.gz \
  --natural 18000 --nearwin 9000 --raregoods 18000 --bootstrap 5000

# Remove bootstrap experiences (e.g. after MCTS training is well underway)
python strip_bootstrap.py mali_ba_buffer.pkl.gz
```

### Near-Win Extension

Games reaching `max_play_moves` (430) are not automatically discarded. At that point, if `(is_near_win OR clear_winner) AND max_val > near_win_extension_value_thresh`, the game is extended by `near_win_extension_moves` (40) additional moves. Otherwise it is discarded (not added to the buffer).

The C++ `IsTerminal()` respects this: the game is only declared terminal after `max_play_moves + near_win_extension_moves` (= 470) play-phase moves.

### Early Termination

Games are culled early when they are determined to be hopeless, using a tiered value-head threshold system (every 20 moves from move 320 onward), a no-near-win cutoff at move 365, a declining-best check from move 360, and a stall check from move 400. These parameters are all in `[MLTraining]`. The system has been validated to have a zero false-positive rate (no winning games killed).

## Distributed Training

See [DISTRIBUTED_TRAINING.md](DISTRIBUTED_TRAINING.md) for the full setup guide.

```bash
# On the training server (desktop):
python train_mali_ba.py --num_actors 24 --remote_actors 25 --distributed \
  --queue_port 50000 --authkey malibatraining2024 ...

# On the remote machine (laptop), after bootstrap phase is complete:
python remote_actors.py --server_host 192.168.x.x \
  --num_actors 20 --actor_id_start 100000 \
  2>&1 | tee ~/remote_actors.log   # also writes output to a file you can tail from elsewhere
```

## Analyzing Training Runs

```bash
python analyze_log.py /path/to/train_runXX.log
```

Outputs: win rate, win type split (Timbuktu vs Rare goods), value head trajectories by game category, early termination stats, loss trend, and rolling win-rate windows.

## Running the GUI

```bash
# From the mali_ba directory
python main.py
```

Requires the C++ pyspiel extension to be built and on `PYTHONPATH`.

## Configuration

All game, board, rule, heuristic, and training parameters are in `mali_ba.ini`. The file is read by both the C++ game (`[GameSettings]`, `[Board]`, `[Cities]`, `[Rules]`, `[Heuristics]`, `[Training]`) and the Python training script (`[MLTraining]`). CLI arguments to `train_mali_ba.py` override INI values.

## Model Weights

Saved as a pair of `.h5` files per agent version:
- `<name>._policy.weights.h5` — policy network
- `<name>._value.weights.h5` — value network

The value network outputs one value per player (3 outputs), trained with discounted returns. The policy network outputs a probability distribution over all legal actions.

## Key File Relationships

```
C++ game (games/mali_ba/) ──compiled──► pyspiel.so
                                              │
                                         train_mali_ba.py
                                         ├── training_utils.py  (ReplayBuffer, SimpleAgent)
                                         ├── queue_server.py    (distributed)
                                         ├── remote_actors.py   (distributed)
                                         └── mali_ba.ini        (all config)
```
