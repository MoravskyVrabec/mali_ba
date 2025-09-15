import sys
sys.path.append("/media/robp/UD/Projects/mali_ba/open_spiel/python/games") # allow debugging in vs code
from dataclasses import dataclass
import os
import json
import collections
import random
from typing import List, Dict, Tuple, NamedTuple, Optional, Set
from mali_ba.config import PlayerColor, TradePostType

class HexCoord(NamedTuple):
    x: int # Corresponds to 'x' in cube systems
    y: int # Corresponds to 'y' in cube systems
    z: int # Corresponds to 'z' in cube systems (x+y+z=0)

    def __str__(self):
        return f"{self.x},{self.y},{self.z}" # Compact string for moves

    @classmethod
    def from_string(cls, coord_str):
        if not coord_str: return None
        parts = coord_str.strip("()").split(',')
        if len(parts) != 3:
            return None
        try:
            x, y, z = int(parts[0]), int(parts[1]), int(parts[2])
            if x + y + z != 0:
                 # print(f"Warning: Invalid cube coordinates (sum!=0): {coord_str}")
                 return None
            return cls(x, y, z)
        except ValueError:
            return None

    def distance(self, other: 'HexCoord') -> int:
        return (abs(self.x - other.x) + abs(self.y - other.y) + abs(self.z - other.z)) // 2

class TradePost:
    def __init__(self, owner: PlayerColor, type_: TradePostType):
        self.owner = owner
        self.type = type_

class City:
    def __init__(self, id_: int, name: str, culture: str, location: HexCoord,
                 common_good: str, rare_good: str):
        self.id = id_
        self.name = name
        self.culture = culture
        self.location = location
        self.common_good = common_good
        self.rare_good = rare_good

class GameRules:
    """Python representation of the C++ GameRules struct."""
    def __init__(self):
        self.free_action_trade_routes: bool = True
        self.endgm_cond_numroutes: int = 4           
        self.endgm_cond_numrare_goods: int = 4       
        self.upgrade_cost_common: int = 3
        self.upgrade_cost_rare: int = 1
        self.posts_per_player: int = 6               
        #self.mancala_capture_rule: str = "none"

# class for trade routes
class TradeRoute:
    def __init__(self, id_: int, owner: PlayerColor, hexes: List[HexCoord], goods: Dict[str, int] = None):
        self.id = id_
        self.owner = owner  # Player who owns this route
        self.hexes = hexes  # List of hexes in the route
        self.goods = goods or {}  # Resources gained from this route
        self.active = True  # Whether the route is currently valid/active
    
    def add_hex(self, hex_coord: HexCoord) -> bool:
        """Add a hex to the route if it's not already there"""
        if hex_coord not in self.hexes:
            self.hexes.append(hex_coord)
            return True
        return False
    
    def remove_hex(self, hex_coord: HexCoord) -> bool:
        """Remove a hex from the route if it exists"""
        if hex_coord in self.hexes:
            self.hexes.remove(hex_coord)
            return True
        return False
    
    def is_valid(self, player_posts: Dict[HexCoord, List[TradePost]]) -> bool:
        """Check if this route is valid (has player's posts/centers at all locations)"""
        if len(self.hexes) < 2:
            return False
            
        for hex_coord in self.hexes:
            # Check if player has a post at this hex
            posts = player_posts.get(hex_coord, [])
            player_has_post = any(post.owner == self.owner for post in posts)
            if not player_has_post:
                return False
                
        return True
    
    def calculate_value(self, cities: List[City]) -> Dict[str, int]:
        """Calculate the value of this trade route based on connected cities"""
        # Find cities connected by this route
        connected_cities = []
        for city in cities:
            if city.location in self.hexes:
                connected_cities.append(city)
        
        # Calculate value based on city connections
        goods = {}
        if len(connected_cities) >= 2:
            # Example rule: Each pair of connected cities grants 1 common good of each city
            for i in range(len(connected_cities)):
                for j in range(i+1, len(connected_cities)):
                    city_i = connected_cities[i]
                    city_j = connected_cities[j]
                    
                    # Add common goods from each city
                    if city_i.common_good not in goods:
                        goods[city_i.common_good] = 0
                    goods[city_i.common_good] += 1
                    
                    if city_j.common_good not in goods:
                        goods[city_j.common_good] = 0
                    goods[city_j.common_good] += 1
        
        self.goods = goods
        return goods

# Used to replay a log of states produced by c++
class SimpleReplayManager:
    """Simple replay manager that can be integrated into the application."""
    
    def __init__(self):
        self.setup_data: Optional[Dict] = None
        self.moves: List[Tuple[str, Dict]] = []  # List of (action, state) tuples
        self.current_move_index: int = -1  # -1 means showing the initial state before any moves

    def load_replay_file(self, filepath: Optional[str]) -> bool:
        """Load replay data from a log file."""
        if not filepath or not os.path.exists(filepath):
            print(f"❌ Replay file not found or not specified: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            self.setup_data = None
            self.moves = []
            current_section_content = ""
            current_section_type = None # 'setup' or 'move'

            for line in lines:
                stripped_line = line.strip()
                if not stripped_line:
                    continue

                if stripped_line.startswith('[') and stripped_line.endswith(']'):
                    # --- End of the previous section ---
                    if current_section_type and current_section_content:
                        if current_section_type == 'setup':
                            self.setup_data = json.loads(current_section_content)
                        elif current_section_type == 'move':
                            # Move sections have action=... on a separate line
                            action_line = ""
                            state_json_str = ""
                            move_lines = current_section_content.strip().split('\n')
                            for move_line in move_lines:
                                if move_line.startswith("action="):
                                    action_line = move_line.split('=', 1)[1]
                                elif move_line.startswith("state="):
                                    state_json_str = move_line.split('=', 1)[1]
                            
                            if action_line and state_json_str:
                                self.moves.append((action_line, json.loads(state_json_str)))

                    # --- Start of a new section ---
                    section_name = stripped_line[1:-1]
                    if section_name == 'setup':
                        current_section_type = 'setup'
                    elif section_name.startswith('move'):
                        current_section_type = 'move'
                    else:
                        current_section_type = None
                    current_section_content = ""
                
                elif current_section_type:
                    # Append line to the current section's content
                    current_section_content += line

            # --- Process the very last section in the file ---
            if current_section_type and current_section_content:
                if current_section_type == 'setup':
                    self.setup_data = json.loads(current_section_content)
                elif current_section_type == 'move':
                    action_line = ""
                    state_json_str = ""
                    move_lines = current_section_content.strip().split('\n')
                    for move_line in move_lines:
                        if move_line.startswith("action="):
                            action_line = move_line.split('=', 1)[1]
                        elif move_line.startswith("state="):
                            state_json_str = move_line.split('=', 1)[1]
                    
                    if action_line and state_json_str:
                        self.moves.append((action_line, json.loads(state_json_str)))

            self.current_move_index = 0
            print(f"✅ Loaded replay: {len(self.moves)} moves from {filepath}")
            if not self.setup_data:
                 print("⚠️ Warning: No [setup] section found in replay file.")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load replay file {filepath}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def get_num_players(self) -> Optional[int]:
        if self.setup_data:
            return self.setup_data.get("num_players")
        return None
    
    def get_grid_radius(self) -> int:
        return self.setup_data.get("grid_radius", 5) if self.setup_data else 5
        
    def get_cities(self) -> List[City]:
        if not self.setup_data or "cities" not in self.setup_data: return []
        cities = []
        for city_data in self.setup_data["cities"]:
            loc_parts = city_data["location"].split(',')
            location = HexCoord(int(loc_parts[0]), int(loc_parts[1]), int(loc_parts[2]))
            cities.append(City(city_data["id"], city_data["name"], city_data["cultural_group"], location, city_data["common_good"], city_data["rare_good"]))
        return cities

    def get_valid_hexes(self) -> Set[HexCoord]:
        if not self.setup_data or "valid_hexes" not in self.setup_data: return set()
        hexes = set()
        for hex_str in self.setup_data["valid_hexes"]:
            loc_parts = hex_str.split(',')
            hexes.add(HexCoord(int(loc_parts[0]), int(loc_parts[1]), int(loc_parts[2])))
        return hexes
    
    def get_current_state_json(self) -> str:
        """Get the current state as a JSON string."""
        if self.moves and 0 <= self.current_move_index < len(self.moves):
            return json.dumps(self.moves[self.current_move_index][1])
        return "{}"

    def get_move_info(self) -> str:
        """Get info string about the current position."""
        if self.moves:
            action = self.moves[self.current_move_index][0]
            return f"Move {self.current_move_index + 1}/{len(self.moves)}: {action}"
        return "No moves in replay."

    def try_go_forward(self) -> bool:
        if self.current_move_index < len(self.moves) - 1:
            self.current_move_index += 1
            return True
        return False

    def try_go_backward(self) -> bool:
        if self.current_move_index > 0:
            self.current_move_index -= 1
            return True
        return False

class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = collections.deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)