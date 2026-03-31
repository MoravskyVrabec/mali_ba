# player_config.py
"""
Player Configuration System for Mali-Ba Game
Handles configuration of human vs AI players with customizable AI models and thinking times.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import argparse


class PlayerType(Enum):
    HUMAN = "human"
    AI = "ai"
    HEURISTIC = "heuristic"


@dataclass
class PlayerConfig:
    """Configuration for a single player"""
    player_id: int
    player_type: PlayerType
    model_path: Optional[str] = None
    thinking_time: float = 1.0  # seconds
    ai_strength: float = 1.0  # multiplier for MCTS simulations


@dataclass
class GamePlayerConfig:
    """Complete configuration for all players in a game"""
    players: List[PlayerConfig]
    
    def to_player_types_string(self) -> str:
        """Convert to the player_types string format expected by GameInterface"""
        return ",".join([p.player_type.value for p in self.players])
    
    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        data = {
            'players': [
                {
                    'player_id': p.player_id,
                    'player_type': p.player_type.value,
                    'model_path': p.model_path,
                    'thinking_time': p.thinking_time,
                    'ai_strength': p.ai_strength
                }
                for p in self.players
            ]
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'GamePlayerConfig':
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        players = []
        for p_data in data['players']:
            players.append(PlayerConfig(
                player_id=p_data['player_id'],
                player_type=PlayerType(p_data['player_type']),
                model_path=p_data.get('model_path'),
                thinking_time=p_data.get('thinking_time', 1.0),
                ai_strength=p_data.get('ai_strength', 1.0)
            ))
        
        return cls(players=players)


class PlayerConfigDialog:
    """GUI dialog for configuring players"""
    
    def __init__(self, parent=None, num_players=3, default_config=None):
        self.num_players = num_players
        self.result = None
        
        # Create main window
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Mali-Ba Player Configuration")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Make dialog modal
        if parent:
            self.root.transient(parent)
            self.root.grab_set()
        
        self.setup_ui()
        self.load_default_config(default_config)
        
        # Center the window
        self.center_window()
    
    def center_window(self):
        """Center the dialog on screen"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (self.root.winfo_width() // 2)
        y = (self.root.winfo_screenheight() // 2) - (self.root.winfo_height() // 2)
        self.root.geometry(f"+{x}+{y}")
    
    def setup_ui(self):
        """Create the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Player Configuration", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # Player configuration frame
        config_frame = ttk.LabelFrame(main_frame, text="Player Settings", padding="10")
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        config_frame.columnconfigure(1, weight=1)
        config_frame.columnconfigure(3, weight=1)
        
        # Headers
        headers = ["Player", "Type", "AI Model", "Think Time (s)", "AI Strength"]
        for i, header in enumerate(headers):
            label = ttk.Label(config_frame, text=header, font=("Arial", 10, "bold"))
            label.grid(row=0, column=i, padx=5, pady=5, sticky=tk.W)
        
        # Create player configuration widgets
        self.player_widgets = []
        for player_id in range(self.num_players):
            self.create_player_row(config_frame, player_id)
        
        # Buttons frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # Preset buttons
        ttk.Button(button_frame, text="All Human", 
                  command=self.set_all_human).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="All AI", 
                  command=self.set_all_ai).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Human vs AI", 
                  command=self.set_human_vs_ai).pack(side=tk.LEFT, padx=5)
        
        # Load/Save buttons
        ttk.Button(button_frame, text="Load Config", 
                  command=self.load_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Config", 
                  command=self.save_config).pack(side=tk.LEFT, padx=5)
        
        # OK/Cancel buttons
        ttk.Button(button_frame, text="OK", 
                  command=self.ok_clicked).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", 
                  command=self.cancel_clicked).pack(side=tk.RIGHT, padx=5)
    
    def create_player_row(self, parent, player_id):
        """Create widgets for configuring a single player"""
        row = player_id + 1
        
        # Player label
        player_label = ttk.Label(parent, text=f"Player {player_id + 1}")
        player_label.grid(row=row, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Player type combobox
        type_var = tk.StringVar(value="human")
        type_combo = ttk.Combobox(parent, textvariable=type_var, 
                                 values=["human", "ai", "heuristic"],
                                 state="readonly", width=10)
        type_combo.grid(row=row, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        
        # Model path frame
        model_frame = ttk.Frame(parent)
        model_frame.grid(row=row, column=2, padx=5, pady=5, sticky=(tk.W, tk.E))
        model_frame.columnconfigure(0, weight=1)
        
        model_var = tk.StringVar()
        model_entry = ttk.Entry(model_frame, textvariable=model_var, state="disabled")
        model_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        model_button = ttk.Button(model_frame, text="Browse...", 
                                 command=lambda: self.browse_model(model_var),
                                 state="disabled", width=8)
        model_button.grid(row=0, column=1, padx=(5, 0))
        
        # Thinking time spinbox
        think_var = tk.DoubleVar(value=1.0)
        think_spin = ttk.Spinbox(parent, from_=0.1, to=10.0, increment=0.1,
                                textvariable=think_var, width=8, state="disabled")
        think_spin.grid(row=row, column=3, padx=5, pady=5)
        
        # AI strength spinbox
        strength_var = tk.DoubleVar(value=1.0)
        strength_spin = ttk.Spinbox(parent, from_=0.1, to=3.0, increment=0.1,
                                   textvariable=strength_var, width=8, state="disabled")
        strength_spin.grid(row=row, column=4, padx=5, pady=5)
        
        # Store widgets for this player
        widgets = {
            'type_var': type_var,
            'type_combo': type_combo,
            'model_var': model_var,
            'model_entry': model_entry,
            'model_button': model_button,
            'think_var': think_var,
            'think_spin': think_spin,
            'strength_var': strength_var,
            'strength_spin': strength_spin
        }
        
        self.player_widgets.append(widgets)
        
        # Bind type change event
        type_combo.bind('<<ComboboxSelected>>', 
                       lambda e, pid=player_id: self.on_type_changed(pid))
    
    def on_type_changed(self, player_id):
        """Handle player type change"""
        widgets = self.player_widgets[player_id]
        player_type = widgets['type_var'].get()
        
        is_ai = player_type == "ai"
        
        # Enable/disable AI-specific widgets
        state = "normal" if is_ai else "disabled"
        widgets['model_entry'].config(state=state)
        widgets['model_button'].config(state=state)
        widgets['think_spin'].config(state=state)
        widgets['strength_spin'].config(state=state)
        
        if not is_ai:
            # Clear AI-specific values when not AI
            widgets['model_var'].set("")
    
    def browse_model(self, model_var):
        """Browse for AI model file"""
        filetypes = [
            ("Model files", "*.h5 *.weights.h5"),
            ("All files", "*.*")
        ]
        filename = filedialog.askopenfilename(
            title="Select AI Model File",
            filetypes=filetypes
        )
        if filename:
            model_var.set(filename)
    
    def set_all_human(self):
        """Set all players to human"""
        for widgets in self.player_widgets:
            widgets['type_var'].set("human")
            self.on_type_changed(self.player_widgets.index(widgets))
    
    def set_all_ai(self):
        """Set all players to AI"""
        for widgets in self.player_widgets:
            widgets['type_var'].set("ai")
            self.on_type_changed(self.player_widgets.index(widgets))
    
    def set_human_vs_ai(self):
        """Set first player to human, rest to AI"""
        for i, widgets in enumerate(self.player_widgets):
            if i == 0:
                widgets['type_var'].set("human")
            else:
                widgets['type_var'].set("ai")
            self.on_type_changed(i)
    
    def load_config(self):
        """Load configuration from file"""
        filetypes = [("Config files", "*.json"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(
            title="Load Player Configuration",
            filetypes=filetypes
        )
        if filename:
            try:
                config = GamePlayerConfig.load_from_file(filename)
                self.load_default_config(config)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{e}")
    
    def save_config(self):
        """Save current configuration to file"""
        filetypes = [("Config files", "*.json"), ("All files", "*.*")]
        filename = filedialog.asksaveasfilename(
            title="Save Player Configuration",
            filetypes=filetypes,
            defaultextension=".json"
        )
        if filename:
            try:
                config = self.get_config()
                config.save_to_file(filename)
                messagebox.showinfo("Success", "Configuration saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
    
    def load_default_config(self, config):
        """Load configuration into the dialog"""
        if not config:
            return
        
        for i, player_config in enumerate(config.players):
            if i >= len(self.player_widgets):
                break
            
            widgets = self.player_widgets[i]
            widgets['type_var'].set(player_config.player_type.value)
            
            if player_config.model_path:
                widgets['model_var'].set(player_config.model_path)
            
            widgets['think_var'].set(player_config.thinking_time)
            widgets['strength_var'].set(player_config.ai_strength)
            
            self.on_type_changed(i)
    
    def get_config(self) -> GamePlayerConfig:
        """Get current configuration from dialog"""
        players = []
        
        for i, widgets in enumerate(self.player_widgets):
            player_type = PlayerType(widgets['type_var'].get())
            model_path = widgets['model_var'].get() if player_type == PlayerType.AI else None
            thinking_time = widgets['think_var'].get()
            ai_strength = widgets['strength_var'].get()
            
            players.append(PlayerConfig(
                player_id=i,
                player_type=player_type,
                model_path=model_path,
                thinking_time=thinking_time,
                ai_strength=ai_strength
            ))
        
        return GamePlayerConfig(players=players)
    
    def validate_config(self) -> Tuple[bool, str]:
        """Validate the current configuration"""
        for i, widgets in enumerate(self.player_widgets):
            player_type = widgets['type_var'].get()
            
            if player_type == "ai":
                model_path = widgets['model_var'].get()
                if not model_path:
                    return False, f"Player {i+1}: AI model path is required"
                if not os.path.exists(model_path):
                    return False, f"Player {i+1}: AI model file does not exist: {model_path}"
        
        return True, ""
    
    def ok_clicked(self):
        """Handle OK button click"""
        valid, error_msg = self.validate_config()
        if not valid:
            messagebox.showerror("Invalid Configuration", error_msg)
            return
        
        self.result = self.get_config()
        self.root.destroy()
    
    def cancel_clicked(self):
        """Handle Cancel button click"""
        self.result = None
        self.root.destroy()
    
    def show(self) -> Optional[GamePlayerConfig]:
        """Show the dialog and return the result"""
        self.root.mainloop()
        return self.result


def parse_player_config_args(parser: argparse.ArgumentParser):
    """Add player configuration arguments to argument parser"""
    player_group = parser.add_argument_group('Player Configuration')
    
    player_group.add_argument(
        '--player_config',
        type=str,
        help='Path to JSON file containing player configuration'
    )
    
    player_group.add_argument(
        '--player_types',
        type=str,
        default='human,ai',
        help='Comma-separated list of player types (human, ai, heuristic). E.g., "human,ai,ai"'
    )
    
    player_group.add_argument(
        '--ai_models',
        type=str,
        help='Comma-separated list of AI model paths (for AI players only)'
    )
    
    player_group.add_argument(
        '--ai_thinking_times',
        type=str,
        default='1.0',
        help='Comma-separated list of AI thinking times in seconds'
    )
    
    player_group.add_argument(
        '--ai_strengths',
        type=str,
        default='1.0',
        help='Comma-separated list of AI strength multipliers'
    )
    
    player_group.add_argument(
        '--no_gui_config',
        action='store_true',
        help='Skip GUI configuration dialog even if no config is provided'
    )


def get_player_config_from_args(args, num_players: int) -> Optional[GamePlayerConfig]:
    """Parse player configuration from command line arguments"""
    
    # Try to load from config file first
    if hasattr(args, 'player_config') and args.player_config:
        try:
            return GamePlayerConfig.load_from_file(args.player_config)
        except Exception as e:
            print(f"Warning: Failed to load player config from {args.player_config}: {e}")
    
    # Parse from individual arguments
    player_types = args.player_types.split(',') if hasattr(args, 'player_types') else ['human'] * num_players
    ai_models = args.ai_models.split(',') if hasattr(args, 'ai_models') and args.ai_models else []
    thinking_times = [float(x) for x in args.ai_thinking_times.split(',')] if hasattr(args, 'ai_thinking_times') else [1.0]
    ai_strengths = [float(x) for x in args.ai_strengths.split(',')] if hasattr(args, 'ai_strengths') else [1.0]
    
    # Extend lists to match number of players
    player_types = (player_types + ['human'] * num_players)[:num_players]
    thinking_times = (thinking_times + [1.0] * num_players)[:num_players]
    ai_strengths = (ai_strengths + [1.0] * num_players)[:num_players]
    
    players = []
    ai_model_index = 0
    
    for i in range(num_players):
        player_type = PlayerType(player_types[i])
        model_path = None
        
        if player_type == PlayerType.AI and ai_model_index < len(ai_models):
            model_path = ai_models[ai_model_index]
            ai_model_index += 1
        
        players.append(PlayerConfig(
            player_id=i,
            player_type=player_type,
            model_path=model_path,
            thinking_time=thinking_times[i],
            ai_strength=ai_strengths[i]
        ))
    
    return GamePlayerConfig(players=players)


def get_player_config(num_players: int, args=None) -> GamePlayerConfig:
    """
    Get player configuration either from command line args or GUI dialog.
    
    Args:
        num_players: Number of players in the game
        args: Parsed command line arguments (optional)
    
    Returns:
        GamePlayerConfig object with player settings
    """
    
    # Try to get config from command line arguments
    if args:
        config = get_player_config_from_args(args, num_players)
        if config:
            return config
        
        # If no_gui_config is set, return default config
        if hasattr(args, 'no_gui_config') and args.no_gui_config:
            return create_default_config(num_players)
    
    # Show GUI dialog to get configuration
    try:
        import tkinter as tk
        
        # Create temporary root window (hidden)
        root = tk.Tk()
        root.withdraw()
        
        dialog = PlayerConfigDialog(parent=root, num_players=num_players)
        config = dialog.show()
        
        root.destroy()
        
        if config:
            return config
        else:
            print("Configuration cancelled, using default settings.")
            return create_default_config(num_players)
            
    except ImportError:
        print("Tkinter not available, using default player configuration.")
        return create_default_config(num_players)
    except Exception as e:
        print(f"Error showing configuration dialog: {e}")
        return create_default_config(num_players)


def create_default_config(num_players: int) -> GamePlayerConfig:
    """Create default player configuration (first player human, rest AI)"""
    players = []
    for i in range(num_players):
        player_type = PlayerType.HUMAN if i == 0 else PlayerType.AI
        players.append(PlayerConfig(
            player_id=i,
            player_type=player_type,
            thinking_time=1.0,
            ai_strength=1.0
        ))
    
    return GamePlayerConfig(players=players)


# Example usage and testing
if __name__ == "__main__":
    # Test the dialog
    config = get_player_config(num_players=3)
    if config:
        print("Player Configuration:")
        for player in config.players:
            print(f"  Player {player.player_id + 1}: {player.player_type.value}")
            if player.player_type == PlayerType.AI:
                print(f"    Model: {player.model_path}")
                print(f"    Think Time: {player.thinking_time}s")
                print(f"    Strength: {player.ai_strength}")