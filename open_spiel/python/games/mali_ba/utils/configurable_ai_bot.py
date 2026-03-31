# configurable_ai_bot.py
"""
Configurable AI Bot System for Mali-Ba Game
Supports loading custom AI models and configurable thinking times.
"""

import os
import sys
import time
import numpy as np
from typing import Optional, Dict, Any
import threading
import queue

# Ensure proper path setup before imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up from utils to mali_ba
mali_ba_root = os.path.dirname(project_root)  # Go up from mali_ba to parent

# Add paths for imports
if project_root not in sys.path:
    sys.path.insert(0, project_root)
if mali_ba_root not in sys.path:
    sys.path.insert(0, mali_ba_root)

# Add the specific OpenSpiel python path BEFORE trying imports
openspiel_python_path = "/home/robp/Projects/mali_ba/open_spiel/python"
if openspiel_python_path not in sys.path:
    sys.path.insert(0, openspiel_python_path)

# Add build directory for pyspiel
build_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(project_root))), "build", "python")
if os.path.exists(build_dir) and build_dir not in sys.path:
    sys.path.insert(0, build_dir)

# Import dependencies with error handling
_imports_successful = True
PlayerConfig = None
PlayerType = None

try:
    from mali_ba.utils.player_config import PlayerConfig, PlayerType
    print("player_config imported successfully")
except ImportError as e:
    print(f"Warning: player_config import failed: {e}")
    _imports_successful = False

try:
    import tensorflow as tf
    print("tensorflow imported successfully")
except ImportError as e:
    print(f"Warning: tensorflow import failed: {e}")
    _imports_successful = False

try:
    import pyspiel
    print("pyspiel imported successfully")
except ImportError as e:
    print(f"Warning: pyspiel import failed: {e}")
    _imports_successful = False

try:
    from algorithms import mcts
    print("mcts imported successfully (direct)")
except ImportError as e:
    print(f"Warning: direct mcts import failed: {e}")
    try:
        from open_spiel.python.algorithms import mcts
        print("mcts imported successfully (full path)")
    except ImportError as e2:
        print(f"Warning: full path mcts import failed: {e2}")
        _imports_successful = False

try:
    from mali_ba.training_utils import SimpleAgent, AlphaZeroEvaluator, create_mali_ba_policy_network, create_mali_ba_value_network
    print("training_utils imported successfully")
except ImportError as e:
    print(f"Warning: training_utils import failed: {e}")
    _imports_successful = False

if not _imports_successful:
    print("Some critical imports failed. AI bot functionality may be limited.")


class ConfigurableAIBot:
    """
    AI Bot that can load custom models and has configurable thinking time.
    
    This bot uses MCTS with a neural network evaluator, where the neural network
    can be loaded from a custom model file.
    """
    
    def __init__(self, game, player_config):
        """
        Initialize the AI bot with a specific configuration.
        
        Args:
            game: The pyspiel game object
            player_config: Configuration for this AI player (PlayerConfig object)
        """
        if not _imports_successful:
            raise RuntimeError("Cannot create AI bot due to missing dependencies")
            
        self.game = game
        self.config = player_config
        self.player_id = player_config.player_id
        
        # Initialize neural network models
        self.policy_model = None
        self.value_model = None
        self.evaluator = None
        self.mcts_bot = None
        
        # Thinking simulation
        self.thinking_enabled = True
        self.is_thinking = False
        self.thinking_start_time = None
        
        # Initialize the AI components
        self._initialize_models()
        self._initialize_mcts()
    
    def _initialize_models(self):
        """Initialize the neural network models."""
        try:
            # Get game dimensions
            obs_shape = self.game.observation_tensor_shape()
            num_actions = self.game.num_distinct_actions()
            num_players = self.game.num_players()
            
            # Create model architecture
            self.policy_model = create_mali_ba_policy_network(obs_shape, num_actions)
            self.value_model = create_mali_ba_value_network(obs_shape, num_players)
            
            # Load custom weights if specified
            if self.config.model_path and os.path.exists(self.config.model_path):
                self._load_model_weights(self.config.model_path)
                print(f"AI Player {self.player_id + 1}: Loaded model from {self.config.model_path}")
            else:
                print(f"AI Player {self.player_id + 1}: Using default random weights")
                if self.config.model_path:
                    print(f"  Warning: Model file not found: {self.config.model_path}")
            
        except Exception as e:
            print(f"Error initializing models for AI Player {self.player_id + 1}: {e}")
            raise
    
    def _load_model_weights(self, model_path: str):
        """Load model weights from file."""
        try:
            # Try to load both policy and value model weights
            # The training_utils.py shows models are saved with _policy and _value suffixes
            policy_path = model_path.replace("weights.h5", "_policy.weights.h5")
            value_path = model_path.replace("weights.h5", "_value.weights.h5")
            
            if os.path.exists(policy_path) and os.path.exists(value_path):
                self.policy_model.load_weights(policy_path)
                self.value_model.load_weights(value_path)
                print(f"  Loaded separate policy and value weights")
            elif os.path.exists(model_path):
                # Try to load as a complete SimpleAgent model
                temp_agent = SimpleAgent(
                    self.game.observation_tensor_shape(),
                    self.game.num_distinct_actions(),
                    self.game.num_players()
                )
                temp_agent.load_model(model_path)
                
                # Copy weights to our models
                self.policy_model.set_weights(temp_agent.policy_model.get_weights())
                self.value_model.set_weights(temp_agent.value_model.get_weights())
                print(f"  Loaded weights from combined model file")
            else:
                raise FileNotFoundError(f"Model weights not found: {model_path}")
                
        except Exception as e:
            print(f"Warning: Failed to load model weights: {e}")
            print(f"  Using random initialization instead")
    
    def _initialize_mcts(self):
        """Initialize the MCTS bot with neural network evaluator."""
        try:
            # Check if mcts is available
            if 'mcts' not in globals():
                print(f"AI Player {self.player_id + 1}: MCTS not available, using random fallback")
                self.mcts_bot = None
                return
            
            # Create the evaluator using our loaded models
            self.evaluator = AlphaZeroEvaluator(self.game, self.policy_model, self.value_model)
            
            # Calculate number of simulations based on AI strength
            base_simulations = 50  # Default from your training script
            max_simulations = int(base_simulations * self.config.ai_strength)
            
            # Create MCTS bot
            self.mcts_bot = mcts.MCTSBot(
                self.game,
                getattr(self.config, 'uct_c', 2.0),  # Use getattr for safety
                max_simulations,
                self.evaluator
            )
            
            print(f"AI Player {self.player_id + 1}: MCTS initialized with {max_simulations} simulations")
            
        except Exception as e:
            print(f"Error initializing MCTS for AI Player {self.player_id + 1}: {e}")
            print(f"AI Player {self.player_id + 1}: Falling back to random play")
            self.mcts_bot = None
    
    def step(self, state) -> int:
        """
        Choose an action for the given state.
        
        Args:
            state: The current game state
            
        Returns:
            The chosen action ID
        """
        if state.is_terminal():
            return None
        
        # Start thinking simulation
        self._start_thinking()
        
        try:
            # Use MCTS to select action if available
            if self.mcts_bot is not None:
                action = self.mcts_bot.step(state)
            else:
                # Fallback to random action if MCTS not available
                legal_actions = state.legal_actions()
                if legal_actions:
                    action = np.random.choice(legal_actions)
                else:
                    action = None
            
            # Ensure we think for at least the configured time
            self._ensure_thinking_time()
            
            return action
            
        except Exception as e:
            print(f"Error in AI Player {self.player_id + 1} step: {e}")
            # Fallback to random action
            legal_actions = state.legal_actions()
            if legal_actions:
                return np.random.choice(legal_actions)
            return None
        finally:
            self._stop_thinking()
    
    def _start_thinking(self):
        """Start the thinking simulation."""
        if self.thinking_enabled and self.config.thinking_time > 0:
            self.is_thinking = True
            self.thinking_start_time = time.time()
    
    def _ensure_thinking_time(self):
        """Ensure the AI has been thinking for at least the configured time."""
        if not self.thinking_enabled or self.config.thinking_time <= 0:
            return
        
        if self.thinking_start_time is not None:
            elapsed = time.time() - self.thinking_start_time
            remaining = self.config.thinking_time - elapsed
            
            if remaining > 0:
                time.sleep(remaining)
    
    def _stop_thinking(self):
        """Stop the thinking simulation."""
        self.is_thinking = False
        self.thinking_start_time = None
    
    def get_thinking_status(self) -> dict:
        """Get current thinking status for display purposes."""
        if not self.is_thinking or self.thinking_start_time is None:
            return {"thinking": False, "elapsed": 0, "remaining": 0}
        
        elapsed = time.time() - self.thinking_start_time
        remaining = max(0, self.config.thinking_time - elapsed)
        
        return {
            "thinking": True,
            "elapsed": elapsed,
            "remaining": remaining,
            "total_time": self.config.thinking_time
        }
    
    def set_thinking_enabled(self, enabled: bool):
        """Enable or disable thinking simulation."""
        self.thinking_enabled = enabled


class AIPlayerManager:
    """
    Manages multiple AI players with different configurations.
    """
    
    def __init__(self, game, player_configs: list):
        """
        Initialize AI player manager.
        
        Args:
            game: The pyspiel game object
            player_configs: List of PlayerConfig objects
        """
        self.game = game
        self.player_configs = player_configs
        self.ai_bots = {}
        
        # Initialize AI bots for AI players
        for config in player_configs:
            if hasattr(config, 'player_type') and str(config.player_type).endswith('AI'):
                try:
                    bot = ConfigurableAIBot(game, config)
                    self.ai_bots[config.player_id] = bot
                    print(f"Initialized AI bot for Player {config.player_id + 1}")
                except Exception as e:
                    print(f"Failed to initialize AI bot for Player {config.player_id + 1}: {e}")
    
    def get_ai_action(self, player_id: int, state) -> Optional[int]:
        """
        Get action from AI player.
        
        Args:
            player_id: ID of the player
            state: Current game state
            
        Returns:
            Action ID or None if not an AI player
        """
        if player_id in self.ai_bots:
            return self.ai_bots[player_id].step(state)
        return None
    
    def is_ai_player(self, player_id: int) -> bool:
        """Check if a player is an AI player."""
        return player_id in self.ai_bots
    
    def get_player_type(self, player_id: int):
        """Get the type of a specific player."""
        for config in self.player_configs:
            if config.player_id == player_id:
                return config.player_type
        return "human"  # Default to human
    
    def get_thinking_status(self, player_id: int) -> dict:
        """Get thinking status for an AI player."""
        if player_id in self.ai_bots:
            return self.ai_bots[player_id].get_thinking_status()
        return {"thinking": False, "elapsed": 0, "remaining": 0}
    
    def set_thinking_enabled(self, enabled: bool):
        """Enable or disable thinking simulation for all AI players."""
        for bot in self.ai_bots.values():
            bot.set_thinking_enabled(enabled)
    
    def get_player_info(self, player_id: int) -> dict:
        """Get information about a player."""
        config = None
        for c in self.player_configs:
            if c.player_id == player_id:
                config = c
                break
        
        if not config:
            return {"type": "unknown", "id": player_id}
        
        player_type_str = str(config.player_type).split('.')[-1].lower() if hasattr(config.player_type, 'name') else str(config.player_type)
        
        info = {
            "id": player_id,
            "type": player_type_str,
            "thinking_time": config.thinking_time,
        }
        
        if 'ai' in player_type_str:
            info.update({
                "model_path": config.model_path,
                "ai_strength": config.ai_strength,
                "has_custom_model": bool(config.model_path and os.path.exists(config.model_path))
            })
        
        return info


def create_ai_player_manager_from_config(game, game_config):
    """
    Create an AI player manager from a GamePlayerConfig.
    
    Args:
        game: The pyspiel game object
        game_config: GamePlayerConfig object
        
    Returns:
        AIPlayerManager instance
    """
    if not _imports_successful:
        print("Warning: Cannot create AI player manager due to missing dependencies")
        return None
        
    return AIPlayerManager(game, game_config.players)


# Example usage and testing
if __name__ == "__main__":
    print("Testing configurable AI bot imports...")
    print(f"Imports successful: {_imports_successful}")