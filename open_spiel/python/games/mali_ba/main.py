# --- START OF FILE main.py ---

import os
os.environ['ASAN_OPTIONS'] = 'allocator_may_return_null=1'
import sys
import argparse
import traceback
from typing import List
import time
import pygame

# --- Path Setup ---
# (Path setup logic remains the same)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
build_dir = os.path.join(os.path.dirname(os.path.dirname(project_root)), "build", "python")
if build_dir not in sys.path:
    sys.path.insert(0, build_dir)

try:
    import pyspiel
    print("pyspiel imported successfully.")
except ImportError:
    print("FATAL ERROR: pyspiel module not found.")
    sys.exit(1)

# --- Project Module Imports ---
from mali_ba.utils.cpp_interface import GameInterface
from mali_ba.config import PlayerColor, DEFAULT_PLAYERS
from mali_ba.ui.visualizer import BoardVisualizer
from mali_ba.utils.board_config import add_board_config_args
from mali_ba.classes.classes_other import SimpleReplayManager

# --- Mode Constants ---
MODE_GUI_SYNC_CPP = "gui_sync_cpp"
MODE_CPP_SYNC_GUI = "cpp_sync_gui"
MODE_GUI_REPLAY = "gui_replay"

def run_cpp_sync_gui_loop(game_interface: GameInterface, visualizer: BoardVisualizer):
    """
    Runs the game loop where C++ plays random moves and the GUI displays.
    """
    if game_interface.is_bypassing:
        print("ERROR: cpp_sync_gui mode requires C++ backend.")
        return

    print("--- Starting C++ Sync GUI Loop ---")
    print("Controls: ESC = Exit, P = Pause/Unpause")
    
    running = True
    paused = False
    game_is_over = False # New flag to manage the end-of-game state
    
    visualizer.control_panel.update_status("C++ playing random moves... (Press P to pause)")
    visualizer.draw()

    last_update_time = time.time()
    update_interval = 0.5

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.VIDEORESIZE:
                visualizer.handle_window_resize(event.size[0], event.size[1])
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    paused = not paused
                    if not game_is_over: # Only change status if game is not over
                        status = "PAUSED" if paused else "C++ playing..."
                        visualizer.control_panel.update_status(f"Game {status}")
        
        # --- Main Game Logic ---
        # Only process moves if the game is not over and not paused
        if not game_is_over and not paused:
            if time.time() - last_update_time > update_interval:
                # C++ makes a move
                action_string = "play_random_move"
                success, msg, new_state_json = game_interface.apply_action(action_string)
                
                if success:
                    visualizer.parse_and_update_state(new_state_json)
                else:
                    print(f"C++ move failed: {msg}")
                    running = False # Stop if something goes wrong
                last_update_time = time.time()

        # --- Check for Terminal State (after a move has been made) ---
        if not game_is_over and game_interface.spiel_state.is_terminal():
            print("\n--- GAME IS TERMINAL ---")
            game_is_over = True
            paused = True # Pause the simulation automatically
            
            # --- THIS IS THE CRUCIAL PART ---
            # Call the C++ Returns() function to trigger the score calculation and logging.
            print("--- Requesting Final Scores from C++ Backend... ---")
            final_returns = game_interface.spiel_state.returns()
            print(f"--- Python received final training returns: {final_returns} ---")
            
            visualizer.control_panel.update_status(f"GAME OVER. Final Returns: {final_returns}")

        visualizer.draw()
        visualizer.clock.tick(30)
        
    print("--- Exiting C++ Sync GUI Loop ---")


def main():
    parser = argparse.ArgumentParser(description='Mali-Ba Board Visualizer & Runner')
    parser.add_argument(
        '--mode',
        type=str,
        choices=[MODE_GUI_SYNC_CPP, MODE_CPP_SYNC_GUI, MODE_GUI_REPLAY],
        default=MODE_GUI_SYNC_CPP,
        help="Specify the operating mode."
    )
    add_board_config_args(parser)

    # Add the enable_move_logging parameter, but default to True for Python-initiated games
    parser.add_argument(
        '--enable_move_logging',
        action='store_true',
        default=True,
        help='Enable move logging to create replay files for debugging.'
    )
    
    # In replay mode, config_file argument is repurposed for the replay file
    parser.add_argument('--replay_file', type=str, help='Path to the replay log file for gui_replay mode.')

    # Add the player_types argument
    parser.add_argument(
        '--player_types', 
        type=str, 
        default='human,ai', 
        help='Comma-separated list of player types (human, ai, heuristic). E.g., "human,ai,ai"'
    )

    args = parser.parse_args()

    try:
        pygame.init()

        if args.mode == MODE_GUI_REPLAY:
            # --- Replay Mode ---
            print("🎬 Initializing GUI Replay Mode...")
            replay_manager = SimpleReplayManager()
            
            # Use --replay_file argument for clarity
            replay_file_path = args.replay_file or args.config_file
            if not replay_manager.load_replay_file(replay_file_path):
                print(f"Could not load replay file: {replay_file_path}. Exiting.")
                return

            num_players = replay_manager.get_num_players() or DEFAULT_PLAYERS
            game_player_colors = [PlayerColor.from_int(i) for i in range(num_players)]
            
            visualizer = BoardVisualizer(
                game_interface=None,
                game_player_colors=game_player_colors,
                initialize_pygame=False,
                replay_manager=replay_manager  # Pass the manager during construction
            )
            visualizer.run()

        else:
            # --- Live C++ Modes ---
            game_interface = GameInterface(config_file_path=args.config_file,
                enable_move_logging=args.enable_move_logging,
                prune_moves_for_ai=False,  # No AI so don't reduce the move space
                player_types=args.player_types
            )
            
            num_players = game_interface.get_num_players()
            game_player_colors = [PlayerColor.from_int(i) for i in range(num_players)]

            visualizer = BoardVisualizer(
                game_interface=game_interface,
                game_player_colors=game_player_colors,
                initialize_pygame=False,
                replay_manager=None # Explicitly None for live modes
            )

            if args.mode == MODE_GUI_SYNC_CPP:
                print("🎮 Starting GUI-driven loop...")
                visualizer.run()
            elif args.mode == MODE_CPP_SYNC_GUI:
                print("🤖 Starting C++ driven GUI loop...")
                run_cpp_sync_gui_loop(game_interface, visualizer)

    except Exception as e:
        print(f"\n--- An Error Occurred in Main ---")
        print(f"Error: {e}")
        traceback.print_exc()
    finally:
        pygame.quit()
        print("Pygame shutdown complete.")

if __name__ == "__main__":
    main()

# # --- Path Setup ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # Assuming the script is in mali_ba/, we need to go up to find the parent of mali_ba
# project_root = os.path.dirname(current_dir)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # Correctly add the build directory for pyspiel
# # This assumes a standard build process. Adjust if your build path is different.
# build_dir = os.path.join(os.path.dirname(os.path.dirname(project_root)), "build", "python")
# if build_dir not in sys.path:
#     sys.path.insert(0, build_dir)

# try:
#     import pyspiel
#     print("pyspiel imported successfully.")
# except ImportError:
#     print("=" * 60)
#     print(f"FATAL ERROR: pyspiel module not found. Searched in sys.path. Looked for build at: {build_dir}")
#     print("Please ensure OpenSpiel is built with Python bindings and the path is correct.")
#     sys.exit(1)

# # --- Project Module Imports ---
# from mali_ba.utils.cpp_interface import GameInterface
# from mali_ba.config import PlayerColor
# from mali_ba.ui.visualizer import BoardVisualizer
# from mali_ba.utils.board_config import add_board_config_args

# def main():
#     parser = argparse.ArgumentParser(description='Mali-Ba Board Visualizer & Runner')
    
#     # These are the only arguments needed in the new architecture
#     parser.add_argument(
#         '--mode',
#         type=str,
#         default="gui_sync_cpp",
#         help="Operating mode. 'gui_sync_cpp' is the primary mode."
#     )
#     # This function adds '--config_file' and other board-related args
#     add_board_config_args(parser)
    
#     args = parser.parse_args()
        
#     try:
#         pygame.init()

#         # 1. Initialize the Interface. It loads the C++ game using the config file.
#         game_interface = GameInterface(config_file_path=args.config_file)
        
#         # 2. Get static game info (like num_players) FROM the interface.
#         num_players = game_interface.get_num_players()
#         game_player_colors = [PlayerColor.from_int(i) for i in range(num_players)]
#         print(f"Game configured for {num_players} players via C++ engine.")

#         # 3. Initialize the Visualizer, passing the ready-to-use interface.
#         visualizer = BoardVisualizer(
#             game_interface=game_interface,
#             game_player_colors=game_player_colors,
#             initialize_pygame=False
#         )

#         # 4. Run the main loop.
#         visualizer.run()

#     except Exception as e:
#         print(f"\n--- An Error Occurred in Main ---")
#         print(f"Error: {e}")
#         traceback.print_exc()
#     finally:
#         pygame.quit()
#         print("Pygame shutdown complete.")

# if __name__ == "__main__":
#     main()


# --- END OF FILE main.py ---