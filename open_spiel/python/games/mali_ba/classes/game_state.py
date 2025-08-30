# --- START OF FILE game_state.py ---

import sys
sys.path.append("/media/robp/UD/Projects/open_spiel/open_spiel/python/games") # allow debugging in vs code
from typing import Dict, List, Set, Optional, Tuple
from mali_ba.config import PlayerColor, MeepleColor, Phase
from mali_ba.classes.classes_other import TradePost, City, HexCoord, TradeRoute

# Simplified Python GameState - acts as a cache for visualization
class GameStateCache:
    def __init__(self, num_players=3): # Default player num if not determined
        self.version: int = 1
        self.current_player_id: int = -1 # Start with chance/invalid
        self.current_player_color: PlayerColor = PlayerColor.EMPTY
        self.setup_phase: bool = True # Flag from C++ state
        self.current_phase: Phase = Phase.START
        self.is_terminal: bool = False

        self.player_token_locations: Dict[HexCoord, List[PlayerColor]] = {}
        self.hex_meeples: Dict[HexCoord, List[MeepleColor]] = {}
        self.player_posts_supply: List[int] = []
        self.trade_posts_locations: Dict[HexCoord, List[TradePost]] = {}
        self.cities: List[City] = []
        self.common_goods: List[Dict[str, int]] = [{} for _ in range(num_players)]
        self.rare_goods: List[Dict[str, int]] = [{} for _ in range(num_players)]
        self.valid_hexes: Set[HexCoord] = set()
        self.grid_radius: int = 3 # Default
        self.game_player_colors: List[PlayerColor] = list(PlayerColor)[1:num_players+1] # Exclude EMPTY
        self.trade_routes: List[TradeRoute] = []
        self.next_route_id: int = 1


    def update_num_players(self, num_players):
        if not (2 <= num_players <= 5): return
        self.game_player_colors = list(PlayerColor)[1:num_players+1]
        self.common_goods = [{} for _ in range(num_players)]
        self.rare_goods = [{} for _ in range(num_players)]
        self.player_posts_supply = [6] * num_players    # 6 is just the default

    def initialize_default_board(self, radius=3):
        print(f"DEBUG: Initializing default board with radius {radius}")
        self.grid_radius = radius
        self.valid_hexes = self.generate_regular_board(self.grid_radius)
        print(f"DEBUG: Created {len(self.valid_hexes)} hexes in default board")

    def generate_regular_board(self, radius: int) -> Set[HexCoord]:
        """Generate a regular hexagonal grid centered at origin with given radius."""
        hexes = set()
        # In cube coordinates, we need to ensure q + r + s = 0
        for q in range(-radius, radius + 1):
            # Calculate the range for r based on q to maintain a proper hexagon
            r_min = max(-radius, -q - radius)
            r_max = min(radius, -q + radius)
            for r in range(r_min, r_max + 1):
                s = -q - r  # This ensures q + r + s = 0
                # No need for additional checks since the ranges ensure we're within the hexagon
                hexes.add(HexCoord(q, r, s))

        print(f"DEBUG: generate_regular_board created {len(hexes)} hexes with radius {radius}")
        if len(hexes) < 20:
            print("DEBUG: Generated hexes:", [str(h) for h in hexes])

        return hexes


    def load_cities_from_config(self, city_config: List[Tuple[int, str, str, HexCoord, str, str]]):
        self.cities = [City(*city_data) for city_data in city_config]

    def create_trade_route(self, player_color: PlayerColor, hexes: List[HexCoord] = None) -> TradeRoute:
        """Create a new trade route for a player"""
        if hexes is None:
            hexes = []
            
        route = TradeRoute(self.next_route_id, player_color, hexes)
        self.next_route_id += 1
        self.trade_routes.append(route)
        return route
        
    def get_player_trade_routes(self, player_color: PlayerColor) -> List[TradeRoute]:
        """Get all trade routes for a specific player."""
        return [route for route in self.trade_routes if route.owner == player_color]
        
    def get_hex_trade_routes(self, hex_coord: HexCoord) -> List[TradeRoute]:
        """Get all trade routes that include a specific hex."""
        return [route for route in self.trade_routes if hex_coord in route.hexes]
        
    def remove_trade_route(self, route_id: int) -> bool:
        """Remove a trade route by ID."""
        for i, route in enumerate(self.trade_routes):
            if route.id == route_id:
                self.trade_routes.pop(i)
                return True
        return False
        
    def validate_trade_routes(self):
        """Check all trade routes for validity and update status."""
        for route in self.trade_routes:
            # Check if all hexes in the route have the player's trading posts/centers
            valid = True
            for hex_coord in route.hexes:
                has_player_post = False
                posts = self.trade_posts_locations.get(hex_coord, [])
                for post in posts:
                    if post.owner == route.owner:
                        has_player_post = True
                        break
                if not has_player_post:
                    valid = False
                    break
                    
            # Update the route's active status
            route.active = valid
            
            # Calculate route value if it's valid
            if valid:
                # Find connected cities
                connected_cities = []
                for city in self.cities:
                    if city.location in route.hexes:
                        connected_cities.append(city)
                        
                # Calculate value (example: 1 common good per connected city)
                goods = {}
                for city in connected_cities:
                    if city.common_good not in goods:
                        goods[city.common_good] = 0
                    goods[city.common_good] += 1
                    
                route.goods = goods