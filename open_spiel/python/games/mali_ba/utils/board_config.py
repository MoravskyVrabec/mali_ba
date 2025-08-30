"""
Configuration parser for Mali-Ba board setup via ini files and command line.
Handles custom board definitions including city placements and board geometry.
"""

import configparser
import argparse
from typing import List, Dict, Optional, Set, Any
from mali_ba.classes.classes_other import City, HexCoord, GameRules
from mali_ba.config import CITY_DATA

class BoardConfig:
    """Handles loading and parsing board configuration from ini files."""
    
    def __init__(self):
        self.grid_radius: int = 3
        self.custom_cities: List[City] = []
        self.custom_board_hexes: Set[HexCoord] = set()
        self.use_custom_board: bool = False
        self.use_custom_cities: bool = False

        self.game_rules: GameRules = GameRules()
        self.rules_loaded_from_ini: bool = False
       
        # Background map settings
        self.background_map_path: Optional[str] = None
        self.hex_transparency: int = 128  # Default 50% transparency (0-255)
        
    def load(self, grid_radius, custom_cities, custom_board_hexes, use_custom_board, use_custom_cities):
        self.grid_radius = grid_radius
        self.custom_cities = custom_cities
        self.custom_board_hexes = custom_board_hexes
        self.use_custom_board = use_custom_board
        self.use_custom_cities = use_custom_cities
       
    def load_from_ini(self, ini_path: str) -> bool:
        """
        Load board configuration from ini file.
        Returns True if custom configuration was loaded, False if using defaults.
        """
        try:
            config = configparser.ConfigParser()
            config.read(ini_path)
            
            # Load board settings
            if 'Board' in config:
                board_section = config['Board']
                self.grid_radius = board_section.getint('grid_radius', fallback=3)
                
                # Check if custom board hexes are defined
                if 'custom_hexes' in board_section:
                    custom_hexes_str = board_section['custom_hexes']
                    self.custom_board_hexes = self._parse_hex_list(custom_hexes_str)
                    self.use_custom_board = len(self.custom_board_hexes) > 0
            
            # Load UI settings including background map
            if 'PythonUISettings' in config:
                ui_section = config['PythonUISettings']
                
                # Background map path
                if 'BackgroundMap' in ui_section:
                    map_path = ui_section['BackgroundMap'].strip()
                    if map_path and not map_path.startswith('#'):
                        self.background_map_path = map_path
                        print(f"Background map configured: {map_path}")
                
                # Hex transparency (0-255)
                if 'HexTransparency' in ui_section:
                    transparency = ui_section.getint('HexTransparency', fallback=128)
                    self.hex_transparency = max(0, min(255, transparency))  # Clamp to valid range
                    print(f"Hex transparency set to: {self.hex_transparency}")

            # Load cities
            if 'Cities' in config:
                self._load_cities_from_config(config['Cities'])
                
            # Load Rules
            if 'Rules' in config:
                self._load_rules_from_config(config['Rules'])
                self.rules_loaded_from_ini = True
                print(f"✅ Game rules loaded from {ini_path}")
            else:
                print(f"ℹ️  No [Rules] section found, using default rules")

            return self.use_custom_board or self.use_custom_cities

            
        except Exception as e:
            print(f"Warning: Could not load config from {ini_path}: {e}")
            print("Using default board configuration")
            return False
    
    def load_from_args(self, args: argparse.Namespace) -> None:
        """Load configuration from command line arguments."""
        if hasattr(args, 'grid_radius') and args.grid_radius:
            self.grid_radius = args.grid_radius
            
        # Check for custom board hex specification via command line
        if hasattr(args, 'custom_hexes') and args.custom_hexes:
            self.custom_board_hexes = self._parse_hex_list(args.custom_hexes)
            self.use_custom_board = len(self.custom_board_hexes) > 0
    
    def get_cities(self) -> List[City]:
        """
        Get the final city list to use for the game.
        Returns custom cities if defined, otherwise default cities with Timbuktu at (0,0,0).
        """
        if self.use_custom_cities:
            return self.custom_cities
        else:
            return self._get_default_cities_with_timbuktu()
    
    def get_board_hexes(self) -> Set[HexCoord]:
        """
        Get the final board hex set to use for the game.
        Returns custom hexes if defined, otherwise regular board.
        """
        if self.use_custom_board:
            return self.custom_board_hexes
        else:
            return self._generate_regular_board(self.grid_radius)
    
    def get_rules_dict(self) -> Dict[str, Any]:
        """Get rules as dictionary for passing to game interface."""
        return {
            'free_action_trade_routes': self.game_rules.free_action_trade_routes,
            'endgm_cond_numroutes': self.game_rules.endgm_cond_numroutes,
            'endgm_cond_numrare_goods': self.game_rules.endgm_cond_numrare_goods,
            'upgrade_cost_common': self.game_rules.upgrade_cost_common,
            'upgrade_cost_rare': self.game_rules.upgrade_cost_rare,
            'posts_per_player': self.game_rules.posts_per_player  #,
            #'mancala_capture_rule': self.game_rules.mancala_capture_rule
        }

    
    def has_custom_rules(self) -> bool:
        """Check if custom rules were loaded from INI."""
        return self.rules_loaded_from_ini


    def _load_rules_from_config(self, rules_section) -> None:
        """Load game rules from the [Rules] section of the INI file."""
        try:
            self.game_rules.free_action_trade_routes = rules_section.getboolean(
                'free_action_trade_routes', fallback=True)
            self.game_rules.endgm_cond_numroutes = rules_section.getint(
                'endgm_cond_numroutes', fallback=4)
            self.game_rules.endgm_cond_numrare_goods = rules_section.getint(
                'endgm_cond_numrare_goods', fallback=4)
            self.game_rules.upgrade_cost_common = rules_section.getint(
                'upgrade_cost_common', fallback=3)
            self.game_rules.upgrade_cost_rare = rules_section.getint(
                'upgrade_cost_rare', fallback=1)
            self.game_rules.posts_per_player = rules_section.getint(
                'posts_per_player', fallback=6)
            # self.game_rules.mancala_capture_rule = rules_section.get(
            #     'mancala_capture_rule', fallback='none').strip()
                
        except Exception as e:
            print(f"⚠️  Error loading rules: {e}")


    def _load_cities_from_config(self, cities_section: configparser.SectionProxy) -> None:
        """Parse cities from ini file configuration - only requires name and coordinates."""
        self.custom_cities = []
        
        # Cities can now be defined as:
        # city1 = Timbuktu,0,0,0
        # city2 = Ségou,1,-1,0
        # city3 = Ouagadougou,-1,1,0
        
        for key, value in cities_section.items():
            if key.startswith('city'):
                try:
                    parts = [part.strip() for part in value.split(',')]
                    if len(parts) >= 4:  # Only need name and coordinates
                        name = parts[0]
                        x, y, z = int(parts[1]), int(parts[2]), int(parts[3])
                        
                        # Validate hex coordinate constraint
                        if x + y + z != 0:
                            print(f"Warning: Invalid hex coordinate for {name} ({x},{y},{z}) - x+y+z must equal 0")
                            continue
                        
                        # Find city in database by name
                        city_id = self._find_city_id_by_name(name)
                        if city_id is not None:
                            # Get city details from CITY_DATA
                            city_details = None
                            for id_, city_name, culture, common_good, rare_good in CITY_DATA:
                                if id_ == city_id:
                                    city_details = (id_, city_name, culture, common_good, rare_good)
                                    break
                            
                            if city_details:
                                id_, city_name, culture, common_good, rare_good = city_details
                                city = City(id_, city_name, culture, HexCoord(x, y, z), common_good, rare_good)
                                self.custom_cities.append(city)
                                print(f"  Added city: {city_name} at ({x},{y},{z}) with goods {common_good}/{rare_good}")
                            else:
                                print(f"Warning: City data not found for ID {city_id}")
                        else:
                            print(f"Warning: City '{name}' not found in city database. Available cities:")
                            for id_, city_name, _, _, _ in CITY_DATA:
                                print(f"  - {city_name}")
                    else:
                        print(f"Warning: Invalid city format for {key}: {value} (expected: name,x,y,z)")
                        
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse city '{key}': {value} - {e}")
        
        if self.custom_cities:
            self.use_custom_cities = True
            print(f"Loaded {len(self.custom_cities)} custom cities from config")
    
    def _get_default_cities_with_timbuktu(self) -> List[City]:
        """
        Get default cities with Timbuktu placed at (0,0,0) and other cities randomly placed.
        """
        from mali_ba.utils.cpp_interface import generate_regular_board
        import random
        
        # Always place Timbuktu first at (0,0,0)
        cities = [City(15, "Timbuktu", "Songhai", HexCoord(0, 0, 0), "Salt", "Gold")]
        
        # Generate valid board positions (excluding 0,0,0)
        valid_hexes = generate_regular_board(self.grid_radius)
        valid_hexes.discard(HexCoord(0, 0, 0))  # Remove Timbuktu's position
        
        # Select 3 more cities randomly from CITY_DATA (excluding Timbuktu)
        available_cities = [city_data for city_data in CITY_DATA if city_data[1] != "Timbuktu"]
        selected_cities_data = random.sample(available_cities, min(3, len(available_cities)))
        
        # Place them at random positions
        city_positions = random.sample(list(valid_hexes), min(3, len(valid_hexes)))
        
        for i, (id_, name, culture, common, rare) in enumerate(selected_cities_data):
            if i < len(city_positions):
                cities.append(City(id_, name, culture, city_positions[i], common, rare))
        
        print(f"Created default cities with Timbuktu at (0,0,0)")
        for city in cities:
            print(f"  - {city.name} at {city.location}")
        
        return cities
    
    def _parse_hex_list(self, hex_string: str) -> Set[HexCoord]:
        """Parse a string of hex coordinates into a set of HexCoord objects."""
        hexes = set()
        
        print(f"DEBUG: Parsing hex string: {hex_string[:100]}...")  # Show first 100 chars
        found_5_1_neg6 = False # DEBUG

        # Format: "0,0,0;1,-1,0;-1,1,0" or "0,0,0 1,-1,0 -1,1,0"
        coords_list = hex_string.replace(';', ' ').split()
        
        for coord_str in coords_list:
            coord_str = coord_str.strip()
            if not coord_str:
                continue
                
            try:
                parts = coord_str.split(',')
                if len(parts) == 3:
                    x, y, z = int(parts[0]), int(parts[1]), int(parts[2])

                    # DEBUG: Check for our specific hex
                    # if x == 5 and y == 1 and z == -6:
                    #     found_5_1_neg6 = True
                    #     print(f"DEBUG: Found 5,1,-6 in parsing!")

                    # Validate hex coordinate constraint: x + y + z = 0
                    if x + y + z == 0:
                        hexes.add(HexCoord(x, y, z))
                        # if x == 5 and y == 1 and z == -6:
                        #     print(f"DEBUG: Successfully added 5,1,-6 to hexes set!")
 
                    else:
                        print(f"Warning: Invalid hex coordinate {coord_str} (x+y+z != 0)")
            except ValueError:
                print(f"Warning: Could not parse hex coordinate: {coord_str}")
        
        # print(f"DEBUG: Finished parsing. Found 5,1,-6: {found_5_1_neg6}")
        # print(f"DEBUG: Total hexes parsed: {len(hexes)}")

        return hexes
    
    def _find_city_id_by_name(self, name: str) -> Optional[int]:
        """Find city ID from CITY_DATA by name."""
        for id_, city_name, _, _, _ in CITY_DATA:
            if city_name.lower() == name.lower():
                return id_
        return None
    
    def _generate_regular_board(self, radius: int) -> Set[HexCoord]:
        """Generate a regular hexagonal grid centered at origin."""
        hexes = set()
        for q in range(-radius, radius + 1):
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                s = -q - r
                hexes.add(HexCoord(q, r, s))
        return hexes

    
    def get_background_map_path(self) -> Optional[str]:
        """Get the configured background map path."""
        return self.background_map_path
    
    def get_hex_transparency(self) -> int:
        """Get the configured hex transparency (0-255)."""
        return self.hex_transparency
    
    def has_background_map(self) -> bool:
        """Check if a background map is configured."""
        return self.background_map_path is not None

    def create_sample_ini(self, filename: str = "mali_ba.ini") -> None:
        """Create a sample ini file showing the configuration format with background map support."""
        sample_content = """[GameSettings]
NumPlayers = 3
RngSeed = -1 ; Use -1 for random seed based on time, or a specific integer
PlayerColors = Red,Green,Blue ; Comma-separated, no spaces around commas
LogFile = /tmp/mali_ba_run.log
LoggingEnabled = true

[PythonUISettings]
ScreenWidth = 1280
ScreenHeight = 800
# Background map settings (optional)
# BackgroundMap = mali_ba_map.jpg  ; Path to your map image (jpg, png, etc.)
# HexTransparency = 128            ; Hex transparency when map is loaded (0-255, default 128)

[Board]
# Grid radius for regular hexagonal board (if not using custom_hexes)
grid_radius = 3

# Custom board hexes (optional) - format: x,y,z;x,y,z;...
# If specified, overrides regular board generation
# custom_hexes = 0,0,0;1,-1,0;-1,1,0;0,1,-1;1,0,-1;-1,0,1;0,-1,1

[Cities]
# Simplified city placement - only specify name and coordinates
# Format: name,x,y,z
# All other information (culture, goods) is automatically pulled from the city database

city1 = Timbuktu,0,0,0
city2 = Segou,1,-1,0
city3 = Ouagadougou,-1,1,0
city4 = Agadez,0,1,-1

# Available cities in database:
# Agadez (Tuareg) - Iron work/Silver cross
# Bandiagara (Dogon) - Onions/tobacco/Dogon Mask
# Dinguiraye (Fulani) - Cattle/Wedding blanket
# Dosso (Songhai-Zarma) - Cotton/Silver headdress
# Hemang (Akan) - Kente cloth/Gold weight
# Katsina (Housa) - Kola nuts/Holy book
# Linguère (Wolof) - Casava/peanut/Gold necklace
# Ouagadougou (Dagbani-Mossi) - Horses/Bronze bracelet
# Oudane (Arab) - Camel/Bronze incense burner
# Oyo (Yoruba) - Ivory/Ivory bracelet
# Segou (Mande/Bambara) - Millet/Chiwara
# Sikasso (Senoufo) - Brass jewelry/Kora
# Tabou (Kru) - Pepper/Kru boat
# Warri (Idjo) - Palm Oil/Coral necklace
# Timbuktu (Songhai) - Salt/Gold Coins

# Note: 
# - City names must match exactly those listed above
# - Hex coordinates must satisfy x + y + z = 0
# - All culture and goods information is pulled automatically from the database

# Background Map Usage:
# 1. Place your map image (mali_ba_map.jpg or mali_ba_map.png) in the game directory
# 2. Uncomment the BackgroundMap line above and set the correct path
# 3. Adjust HexTransparency if needed (0=invisible hexes, 255=opaque hexes)
# 4. The map will be automatically scaled and centered behind your hexes
"""
        
        with open(filename, 'w') as f:
            f.write(sample_content)
        
        print(f"Created sample configuration file: {filename}")
        print("Background map support added! Place your map image and update the BackgroundMap setting.")


def add_board_config_args(parser: argparse.ArgumentParser) -> None:
    """Add board configuration arguments to argument parser."""
    parser.add_argument('--config_file', type=str, default="mali_ba.ini", 
                       help='Path to mali_ba.ini config file.')
    parser.add_argument('--custom_hexes', type=str, default=None,
                       help='Custom board hexes as comma/semicolon separated coordinates (e.g., "0,0,0;1,-1,0")')
    parser.add_argument('--create_sample_ini', action='store_true',
                       help='Create a sample mali_ba.ini file and exit')


def load_board_config(args: argparse.Namespace) -> BoardConfig:
    """
    Load board configuration from ini file and command line arguments.
    Command line arguments override ini file settings.
    """
    config = BoardConfig()
    
    # Handle sample ini creation
    if hasattr(args, 'create_sample_ini') and args.create_sample_ini:
        config.create_sample_ini()
        return config
    
    # Try to load from ini file first
    if hasattr(args, 'config_file') and args.config_file:
        import os
        if os.path.exists(args.config_file):
            config.load_from_ini(args.config_file)
        else:
            print(f"Config file {args.config_file} not found, using defaults")
    
    # Apply command line overrides
    config.load_from_args(args)
    
    return config
