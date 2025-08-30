# Mali-Ba Game

A strategic board game based on West African trade routes and mancala mechanics.

## Project Structure

Here's the expected directory structure for this project:

```
open_spiel/python/games/
└── mali_ba/
    ├── __init__.py  # Main package init file
    ├── config.py    # Game constants and configurations
    ├── main.py      # Entry point for the application
    ├── setup.py     # Package installation file
    ├── models/
    │   ├── __init__.py
    │   ├── game_state.py
    │   └── classes_other.py
    ├── ui/
    │   ├── __init__.py
    │   ├── visualizer.py
    │   └── gui_other.py
    └── utils/
        ├── __init__.py
        └── parsing.py
```

## Installation Options

You can install the package in development mode using:

```bash
# From the mali_ba directory (where setup.py is located)
pip install -e .
```

## Running the Game

After installation, you can run the game with:

```bash
# From the mali_ba directory
python main.py --sample
```

Or, to run without installing:

```bash
# Set PYTHONPATH to include the parent directory of mali_ba
export PYTHONPATH=/path/to/open_spiel/python:$PYTHONPATH
python main.py --sample
```