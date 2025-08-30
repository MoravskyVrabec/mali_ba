# mali_ba/dev_config.py
"""
Development configuration settings for Mali-Ba game.
These settings control development features like bypassing C++ integration.
"""

# Set this to True to bypass all C++ code integration
# This allows for testing the UI and game flow without requiring OpenSpiel
BYPASS_CPP = True

# Default number of players when in BYPASS_CPP mode
DEFAULT_PLAYERS = 3

# Default grid radius when in BYPASS_CPP mode
DEFAULT_GRID_RADIUS = 5

# Default number of tokens per player
DEFAULT_TOKENS_PER_PLAYER = 3

# When in BYPASS_CPP mode, this determines whether to load a sample state
# or generate a completely empty board
LOAD_SAMPLE_STATE = True