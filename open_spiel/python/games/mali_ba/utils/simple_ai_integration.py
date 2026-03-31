# simple_ai_integration.py
"""
Simple integration approach - add AI support to existing GameInterface without subclassing
"""

import sys
import os

# Ensure proper imports
try:
    from mali_ba.utils.cpp_interface import GameInterface
    from mali_ba.utils.configurable_ai_bot import create_ai_player_manager_from_config
except ImportError as e:
    print(f"Import error in simple_ai_integration: {e}")
    print("Make sure all files are in the correct locations")


def create_game_interface_with_ai(config_file_path=None, enable_move_logging=True, 
                                  prune_moves_for_ai=True, player_config=None):
    """
    Factory function to create a GameInterface with AI support.
    
    Args:
        config_file_path: Path to game config file
        enable_move_logging: Whether to log moves
        prune_moves_for_ai: Whether to prune moves for AI
        player_config: GamePlayerConfig object
        
    Returns:
        GameInterface with AI capabilities added
    """
    
    # Convert player config to player_types string if provided
    if player_config:
        player_types = player_config.to_player_types_string()
    else:
        player_types = "human,ai"  # Default
    
    # Create standard GameInterface
    game_interface = GameInterface(
        config_file_path=config_file_path,
        enable_move_logging=enable_move_logging,
        prune_moves_for_ai=prune_moves_for_ai,
        player_types=player_types
    )
    
    # Add AI capabilities if player config provided
    if player_config:
        add_ai_support(game_interface, player_config)
    
    return game_interface


def add_ai_support(game_interface, player_config):
    """
    Add AI support to an existing GameInterface instance.
    
    Args:
        game_interface: Existing GameInterface instance
        player_config: GamePlayerConfig object
    """
    
    try:
        # Create AI manager
        ai_manager = create_ai_player_manager_from_config(
            game_interface.spiel_game, 
            player_config
        )
        
        if ai_manager is None:
            print("Warning: Could not create AI manager")
            return
        
        # Add AI manager to interface
        game_interface.ai_manager = ai_manager
        game_interface.player_config = player_config
        
        # Store original apply_action method
        original_apply_action = game_interface.apply_action
        
        def enhanced_apply_action(action_string: str):
            """Enhanced apply_action that handles AI moves."""
            success, message, state_json = original_apply_action(action_string)
            
            if not success:
                return success, message, state_json
            
            # Check if next player is AI and handle their move
            current_player = game_interface.spiel_state.current_player()
            
            if (current_player >= 0 and 
                ai_manager.is_ai_player(current_player) and 
                not game_interface.spiel_state.is_terminal()):
                
                try:
                    ai_action = ai_manager.get_ai_action(current_player, game_interface.spiel_state)
                    if ai_action is not None:
                        game_interface.spiel_state.apply_action(ai_action)
                        state_json = game_interface.spiel_state.serialize()
                        message += f" AI Player {current_player + 1} played action {ai_action}."
                except Exception as e:
                    print(f"Error in AI move: {e}")
            
            return success, message, state_json
        
        # Replace apply_action method
        game_interface.apply_action = enhanced_apply_action
        
        # Add helper methods
        game_interface.is_ai_player = lambda pid: ai_manager.is_ai_player(pid)
        game_interface.get_ai_thinking_status = lambda pid: ai_manager.get_thinking_status(pid)
        game_interface.get_player_info = lambda pid: ai_manager.get_player_info(pid)
        game_interface.has_ai_players = lambda: len(ai_manager.ai_bots) > 0
        
        print("AI support added to GameInterface successfully.")
        
    except Exception as e:
        print(f"Warning: Failed to add AI support: {e}")
        import traceback
        traceback.print_exc()
        game_interface.ai_manager = None


# Usage in main.py becomes very simple:
# def example_usage():
#     """Example of how to use the simple integration in main.py"""
    
#     # Get player configuration
#     player_config = get_player_config(num_players, args)
    
#     # Create game interface with AI support
#     game_interface = create_game_interface_with_ai(
#         config_file_path=args.config_file,
#         enable_move_logging=args.enable_move_logging,
#         prune_moves_for_ai=False,  # Don't prune for GUI games
#         player_config=player_config
#     )
    
    # Everything else works exactly the same as before!
    # The interface automatically handles AI moves when apply_action is called