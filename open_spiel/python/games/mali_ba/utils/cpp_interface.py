# mali_ba/utils/cpp_interface.py
"""
Interface module that encapsulates all interactions with C++ OpenSpiel code.
Follows a client-server model where C++ is the engine and Python is the client.
"""
import sys
import json
from typing import Dict, List, Optional, Tuple, Set

# Ensure pyspiel is available
try:
    import pyspiel
    import pyspiel.mali_ba  # Import the game's C++ bindings
    _pyspiel_is_available = True
    print("cpp_interface: pyspiel module successfully imported.")
except ImportError:
    print("=" * 60)
    print("WARNING (cpp_interface): Pyspiel module not found.")
    print("C++ backend modes will not be available.")
    print("Build OpenSpiel with Python bindings to enable C++ integration.")
    print("=" * 60)
    _pyspiel_is_available = False

from mali_ba.classes.classes_other import HexCoord, City
# from mali_ba.utils.board_config import BoardConfig

class GameInterface:
    """A thin client interface to the C++ OpenSpiel game engine."""
    
    # For individual games kicked off by the Python front end I want to have move logging for replay
    # on as the default
    def __init__(self, config_file_path: Optional[str] = None, enable_move_logging: bool = True, 
                 prune_moves_for_ai: bool = True, player_types: str = "ai,ai,ai"):
        """
        Initializes the GameInterface by loading the C++ game engine.
        
        Args:
            config_file_path: Path to the .ini configuration file.
            player_types: Comma-separated string of player types (human, ai, heuristic).
        """
        self.spiel_game: Optional['pyspiel.Game'] = None
        self.spiel_state: Optional['pyspiel.State'] = None
        self.is_bypassing = not _pyspiel_is_available
        self.config_file_path = config_file_path
        self.enable_move_logging = enable_move_logging
        self.prune_moves_for_ai = prune_moves_for_ai
        self.player_types = player_types 

        if self.is_bypassing:
            print("🐍 C++ backend not available or bypassed. Running in Python-only mode is not supported by this refactor.")
            raise RuntimeError("C++ backend (pyspiel) is required for this application.")

        print("⚡ Initializing C++ game engine...")
        self._init_cpp_game()

    def _init_cpp_game(self):
        """Loads the C++ game and its initial state."""
        try:
            # The config file is the single source of truth for game parameters.
            # When playing games with the GUI enabled, set the move pruning to false.
            # It's only to make AI training more efficient.
            game_params = {"config_file": self.config_file_path or "",
                "enable_move_logging": self.enable_move_logging, 
                "prune_moves_for_ai": self.prune_moves_for_ai,
                "player_types": self.player_types
            }
            
            print(f"🔧 Loading C++ game with parameters: {game_params}")
            self.spiel_game = pyspiel.load_game("mali_ba", game_params)
            print("✅ C++ Game object created with configuration.")

            self.spiel_state = self.spiel_game.new_initial_state()
            print("🎲 Generated initial C++ state.")

            # Advance past the initial chance node to get the real starting state.
            if self.spiel_state.is_chance_node():
                print("🎲 Advancing past initial chance node...")
                chance_actions = self.spiel_state.legal_actions()
                if chance_actions:
                    self.spiel_state.apply_action(chance_actions[0])
                print("✅ Initial board state is ready.")
        
        except Exception as e:
            print(f"❌ FATAL ERROR: C++ game engine initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.is_bypassing = True
            raise


    def apply_action(self, action_string: str) -> Tuple[bool, str, Optional[str]]:
        """
        Applies an action to the C++ game state.
        Handles a special command "play_random_move" for cpp_sync_gui mode.
        """
        if self.is_bypassing:
            return False, "C++ backend is not available.", None
            
        try:
            # Special case for the C++ driven loop
            if action_string == "play_random_move":
                if self.spiel_state.is_terminal():
                    return False, "Game is terminal.", None
                
                # C++ State has a method to play a random move and serialize the result
                # We need to expose this via pybind11
                new_state_json = self.spiel_state.play_random_move_and_serialize()
                return True, "Random move applied.", new_state_json

            # Standard action application
            action_id = self.spiel_state.string_to_action(action_string)
            if action_id == pyspiel.INVALID_ACTION:
                # ... (error handling as before) ...
                return False, f"Invalid action: {action_string}", None
            
            self.spiel_state.apply_action(action_id)
            new_state_json = self.spiel_state.serialize()
            return True, "Action applied successfully.", new_state_json
            
        except Exception as e:
            # ... (exception handling as before) ...
            return False, f"C++ Error: {e}", None


    def get_current_state_string(self) -> str:
        """Gets the complete, authoritative game state as a JSON string from C++."""
        if self.is_bypassing:
            return "{}"
        return self.spiel_state.serialize()


    def get_board_config_data(self) -> Tuple[Set[HexCoord], List[City], int]:
        """
        Extracts the static board configuration (hexes, cities, radius)
        from the loaded C++ game object.
        """
        if self.is_bypassing or not self.spiel_game:
            return set(), [], 3 # Defaults

        # Cast to the specific game type to access custom methods
        mali_ba_game = pyspiel.mali_ba.downcast_game(self.spiel_game)
        
        # Get valid hexes
        valid_hexes = {HexCoord(h.x, h.y, h.z) for h in mali_ba_game.get_valid_hexes()}
        
        # Get cities
        cities = []
        for c_city in mali_ba_game.get_cities():
            cities.append(City(
                id_=c_city.id,
                name=c_city.name,
                culture=c_city.culture,
                location=HexCoord(c_city.location.x, c_city.location.y, c_city.location.z),
                common_good=c_city.common_good,
                rare_good=c_city.rare_good
            ))
        
        grid_radius = mali_ba_game.get_grid_radius()

        return valid_hexes, cities, grid_radius


    def play_heuristic_move(self) -> Tuple[bool, str, Optional[str]]:
        """
        Asks the C++ engine to select and apply one heuristic move for the current player.
        Used by the GUI for non-human players.
        """
        if self.is_bypassing or self.spiel_state.is_terminal():
            return False, "Not available or game is over.", None
        
        try:
            # This method is already exposed via pybind11 in games_mali_ba.cc
            action_id = self.spiel_state.select_heuristic_random_action()
            if action_id == pyspiel.INVALID_ACTION:
                return False, "Heuristic found no valid action.", None
            
            self.spiel_state.apply_action(action_id)
            new_state_json = self.spiel_state.serialize()
            return True, "Heuristic move applied.", new_state_json
        except Exception as e:
            print(f"ERROR in play_heuristic_move: {e}")
            return False, f"C++ Error: {e}", None


    def get_player_types(self) -> List[str]:
        """Returns the list of player types."""
        return [p_type.strip() for p_type in self.player_types.split(',')]


    def get_num_players(self) -> int:
        """Gets the number of players from the C++ game object."""
        if self.is_bypassing or not self.spiel_game:
            return 3 # Default
        return self.spiel_game.num_players()