# --- START OF FILE visualizer.py ---
# Import necessary components
from mali_ba.config import *
from mali_ba.classes.classes_other import TradePost, City, HexCoord, TradePostType, SimpleReplayManager
from mali_ba.classes.game_state import GameStateCache
from mali_ba.ui.gui_other import InteractiveObject, InteractiveObjectManager, ControlPanel, Sidebar, DialogBox
from mali_ba.utils.cpp_interface import GameInterface

# Import the new drawing and parsing modules
from mali_ba.ui.visualizer_drawing import draw_board_state, load_background_map
# THIS IS THE CORRECTED IMPORT:
from mali_ba.ui.visualizer_other import BoardVisualizerHelpers, parse_and_update_state_from_json, can_start_mancala_at, is_valid_mancala_step, can_select_for_upgrade, can_add_to_trade_route

import json
import math
import os
import traceback
from typing import List, Optional, Tuple


# Import pyspiel if needed (handle potential ImportError)
try:
    import pyspiel
except ImportError:
    print("Warning: pyspiel module not found. C++ integration might fail if not bypassed.")
    pyspiel = None


# --- Main Board Visualizer Class ---
class BoardVisualizer:
    """
    Visualizes the Mali-Ba board state and handles user interaction,
    communicating with the C++ OpenSpiel backend.
    (Core logic: Event handling, state management, UI orchestration)
    """
    def __init__(self, game_interface: Optional[GameInterface], game_player_colors: List[PlayerColor],
                 initialize_pygame=True, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
                 replay_manager: Optional[SimpleReplayManager] = None):
        """Initializes the visualizer."""

        print("Python: BoardVisualizer.__init__ ENTERED", flush=True)
        if initialize_pygame:
            pygame.init()

        self.width = screen_width
        self.height = screen_height
        self.width = max(self.width, SIDEBAR_WIDTH + 300)
        self.height = max(self.height, CONTROLS_HEIGHT + 300)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("Mali-Ba Board Visualizer")
        self.show_trade_routes = True
        self.clock = pygame.time.Clock()

        self.font_size, self.small_font_size, self.large_font_size = 20, 16, 24
        self.font = pygame.font.Font(None, self.font_size)
        self.small_font = pygame.font.Font(None, self.small_font_size)
        self.large_font = pygame.font.Font(None, self.large_font_size)
        
        self.fonts = {'font': self.font, 'small_font': self.small_font, 'large_font': self.large_font}
        self.font_sizes = {'font': self.font_size, 'small_font': self.small_font_size, 'large_font': self.large_font_size}
        
        self.helpers = BoardVisualizerHelpers(self)
        self.game_interface = game_interface
        
        # --- Determine mode and set up initial state cache ---
        self.replay_manager = replay_manager
        self.is_replay_mode = self.replay_manager is not None
        
        # Initialize an empty cache first
        num_players = len(game_player_colors) if game_player_colors else DEFAULT_PLAYERS
        self.state_cache = GameStateCache(num_players)
        self.state_cache.game_player_colors = game_player_colors

        # --- Initial Board Structure Setup ---
        if self.is_replay_mode:
            # In replay, get board info from the replay manager's setup data
            print("Visualizer running in Replay Mode.")
            self.state_cache.grid_radius = self.replay_manager.get_grid_radius()
            self.state_cache.valid_hexes = self.replay_manager.get_valid_hexes()
            self.state_cache.cities = self.replay_manager.get_cities()
        elif self.game_interface:
            # In live modes, get from the C++ engine
            print("Visualizer running in Live C++ Mode.")
            valid_hexes, cities, grid_radius = self.game_interface.get_board_config_data()
            self.state_cache.valid_hexes = valid_hexes
            self.state_cache.cities = cities
            self.state_cache.grid_radius = grid_radius
        else:
            raise RuntimeError("Visualizer must be initialized with either a GameInterface or a ReplayManager.")
        
        self.highlight_hexes: List[HexCoord] = []
        self.selected_start_hex: Optional[HexCoord] = None
        self.is_input_mode: bool = False
        self.input_mode_type: Optional[str] = None
        self.wants_post_on_mancala: bool = False

        self.zoom = 1.7
        self.min_zoom = 0.3
        self.max_zoom = 5
        self.board_center_offset = [0, 0]
        self.board_center = (0, 0)

        self.sidebar_rect = pygame.Rect(self.width - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, self.height)
        self.controls_rect = pygame.Rect(0, self.height - CONTROLS_HEIGHT, self.width - SIDEBAR_WIDTH, CONTROLS_HEIGHT)
        self.sidebar = Sidebar(self.sidebar_rect, self.fonts)
        self.control_panel = ControlPanel(self.controls_rect, self.font)
        self.interactive_objects = InteractiveObjectManager()
        dialog_width, dialog_height = 400, 200
        dialog_rect = pygame.Rect((self.width - dialog_width) // 2, (self.height - dialog_height) // 2, dialog_width, dialog_height)
        self.dialog_box = DialogBox(dialog_rect, self.font)
        self.selected_mancala_payment = None
        self.mancala_payment_options = []

        self.background_map_enabled = False
        self._try_load_background_map()
        
        if not self.state_cache.valid_hexes:
            raise RuntimeError("Failed to obtain valid_hexes. Cannot start visualizer.")
        
        self.is_resizing = False  
        self.last_resize_time = 0

        self.update_zoom_limits()
        self.auto_fit_board() # This sets zoom and pan
        self.update_layout()  # This applies the changes

        # Load initial state depending on the mode
        if self.is_replay_mode:
            initial_state_json = self.replay_manager.get_current_state_json()
        else:
            initial_state_json = self.game_interface.get_current_state_string()

        if not self.parse_and_update_state(initial_state_json):
            raise RuntimeError("FATAL: Could not parse initial state.")


    def parse_and_update_state(self, state_string: str) -> bool:
        """Parses the authoritative state string and updates the cache."""
        # This now calls the correctly imported function
        success = parse_and_update_state_from_json(state_string, self.state_cache)
        if success:
            self.helpers.update_status_from_cache()
            # IF WE have machine players, play their turn automatically
            self._check_for_non_human_turn()
        else:
            self.control_panel.update_status("Error: State parsing failed.")
        return success


    def _check_for_non_human_turn(self):
        """
        If it's a non-human's turn in a GUI-driven game, automatically trigger their move.
        """
        if self.is_replay_mode or self.is_input_mode or self.state_cache.is_terminal:
            return

        player_id = self.state_cache.current_player_id
        if player_id < 0: # Chance or terminal
            return
            
        player_types = self.game_interface.get_player_types()
        if player_id >= len(player_types):
            return # Avoid index error if state is inconsistent

        current_player_type = player_types[player_id]

        if current_player_type in ["ai", "heuristic"]:
            pygame.time.set_timer(pygame.USEREVENT + 1, 500, loops=1) # 500ms delay for visual feedback


    def _trigger_non_human_move(self):
        """Plays one move for an AI or Heuristic player."""
        if self.is_replay_mode or self.state_cache.is_terminal:
            return
        
        player_id = self.state_cache.current_player_id
        self.control_panel.update_status(f"Player {player_id + 1} (AI/Heuristic) is thinking...")
        self.draw() # Redraw to show the "thinking" message

        success, msg, new_state_json = self.game_interface.play_heuristic_move()

        if success:
            # This will parse the state and recursively call _check_for_non_human_turn
            self.parse_and_update_state(new_state_json)
        else:
            self.control_panel.update_status(f"Error during AI move: {msg}")


    def create_hex_objects(self):
        """Creates interactive objects for all valid hexes FROM THE CACHE."""
        self.interactive_objects.objects = [obj for obj in self.interactive_objects.objects if obj.name != "hex"]
        if not self.state_cache.valid_hexes:
             return
        for hex_coord in self.state_cache.valid_hexes:
            center_x, center_y = self.hex_to_pixel(hex_coord)
            size = HEX_SIZE * self.zoom
            w = max(10, int(size * 1.8))
            h = max(10, int(size * math.sqrt(3) * 0.9))
            hex_rect = pygame.Rect(center_x - w // 2, center_y - h // 2, w, h)
            self.interactive_objects.add_object(InteractiveObject(hex_rect, "hex", data=hex_coord))


    # --- Update Methods ---
    def update_layout(self):
        """Recalculates layout based on current screen size and zoom/pan."""
        board_area_width = self.width - SIDEBAR_WIDTH
        board_area_height = self.height - CONTROLS_HEIGHT
        self.board_center = (board_area_width // 2 + self.board_center_offset[0], board_area_height // 2 + self.board_center_offset[1] + 12)
        self.sidebar_rect = pygame.Rect(self.width - SIDEBAR_WIDTH, 0, SIDEBAR_WIDTH, self.height)
        self.controls_rect = pygame.Rect(0, self.height - CONTROLS_HEIGHT, board_area_width, CONTROLS_HEIGHT)
        self.sidebar.update_rect(self.sidebar_rect)
        self.control_panel.update_rect(self.controls_rect)
        self.create_hex_objects()


    def hex_to_pixel(self, hex_coord: HexCoord) -> Tuple[int, int]:
        """Converts CUBE hex coordinates to screen pixels (FLAT TOP)."""
        radius = (HEX_SIZE / 2.0) * self.zoom
        size = radius
        if size < 0.5:
            board_area_width = self.width - SIDEBAR_WIDTH
            board_area_height = self.height - CONTROLS_HEIGHT
            return (round(board_area_width / 2 + self.board_center_offset[0]), round(board_area_height / 2 + self.board_center_offset[1]))
        origin_x, origin_y = self.board_center
        pixel_x = size * (3.0 / 2.0) * hex_coord.x
        pixel_y = size * math.sqrt(3) * ((hex_coord.y - hex_coord.z)/2)
        return round(origin_x + pixel_x), round(origin_y + pixel_y)


    def pixel_to_hex(self, x: int, y: int) -> Optional[HexCoord]:
        """Converts screen pixels back to CUBE hex coordinates (FLAT TOP)."""
        origin_x, origin_y = self.board_center
        adj_x, adj_y = x - origin_x, y - origin_y
        size = (HEX_SIZE / 2.0) * self.zoom
        if size <= 1e-6: return None
        x_frac = (2.0 / 3.0 * adj_x) / size
        z_frac = (-1.0 / 3.0 * adj_x + math.sqrt(3) / 3.0 * adj_y) / size
        y_frac = -x_frac - z_frac
        rx, ry, rz = round(x_frac), round(y_frac), round(z_frac)
        x_diff, y_diff, z_diff = abs(rx - x_frac), abs(ry - y_frac), abs(rz - z_frac)
        if x_diff > y_diff and x_diff > z_diff: rx = -ry - rz
        elif y_diff > z_diff: ry = -rx - rz
        else: rz = -rx - ry
        final_hex = HexCoord(int(rx), int(rz), int(ry))
        return final_hex if final_hex in self.state_cache.valid_hexes else None


    def handle_click(self, pos):
        """Handles mouse clicks, routing to UI elements or the board."""
        if self.dialog_box.active:
            result = self.dialog_box.handle_click(pos)
            if result is not None:
                self.helpers.handle_dialog_result(result)
            return

        if self.sidebar_rect.collidepoint(pos) and hasattr(self, 'sidebar_handled_click') and self.sidebar_handled_click:
            self.sidebar_handled_click = False
            return

        clicked_button_id = None
        if self.controls_rect.collidepoint(pos):
            for checkbox_id, (checkbox_rect, _) in self.control_panel.checkboxes.items():
                if checkbox_rect.collidepoint(pos):
                    self.handle_checkbox_click(checkbox_id, self.control_panel.checkboxes[checkbox_id][1])
                    return
            for button_id, rect in self.control_panel.buttons.items():
                if rect.collidepoint(pos):
                    clicked_button_id = button_id
                    break
            if clicked_button_id:
                self.handle_control_button_click(clicked_button_id)
            return
        elif self.sidebar_rect.collidepoint(pos):
            return

        board_area_rect = pygame.Rect(0, 0, self.width - SIDEBAR_WIDTH, self.height - CONTROLS_HEIGHT)
        if board_area_rect.collidepoint(pos):
            hex_coord = self.pixel_to_hex(pos[0], pos[1])
            if hex_coord:
                if self.is_input_mode:
                    self.handle_input_hex_click(hex_coord)
                else:
                    token_info = self.state_cache.player_token_locations.get(hex_coord, [])
                    token_str = ", ".join([color.name for color in token_info]) if token_info else "None"
                    meeple_count = len(self.state_cache.hex_meeples.get(hex_coord, []))
                    posts_info = self.state_cache.trade_posts_locations.get(hex_coord, [])
                    post_str = ", ".join([f"{p.owner.name[0]}{p.type.name[0]}" for p in posts_info]) if posts_info else "None"
                    city_name = next((c.name for c in self.state_cache.cities if c.location == hex_coord), None)
                    city_str = f", City={city_name}" if city_name else ""
                    self.control_panel.update_status(f"Hex {hex_coord}: Tokens=[{token_str}], Meeples={meeple_count}, Posts=[{post_str}]{city_str}")
            elif self.is_input_mode:
                self.cancel_input_mode()


    def handle_control_button_click(self, button_id: str):
        if button_id == "submit" and self.is_input_mode:
            self.submit_move()
        elif button_id == "cancel" and self.is_input_mode:
            self.cancel_input_mode()
        elif button_id == "pass" and not self.is_input_mode and self.state_cache.current_phase == Phase.PLAY and not self.state_cache.is_terminal:
            self.attempt_apply_action("pass")
        elif button_id == "trade_route" and not self.is_input_mode and self.state_cache.current_phase == Phase.PLAY and not self.state_cache.is_terminal:
            self.helpers.start_trade_route_mode()
        elif button_id == "take_income" and not self.is_input_mode and self.state_cache.current_phase == Phase.PLAY and not self.state_cache.is_terminal:
            self.helpers.process_income_collection()
        elif not self.is_input_mode and not self.state_cache.is_terminal:
            phase = self.state_cache.current_phase
            allowed = (button_id == "place_token" and phase == Phase.PLACE_TOKEN) or \
                      (button_id in ["mancala", "upgrade"] and phase == Phase.PLAY)
            if allowed:
                self.start_input_mode(button_id)
            else:
                self.control_panel.update_status(f"Cannot enter '{button_id}' mode in phase {phase.name if phase else 'Unknown'}")
        elif self.state_cache.is_terminal:
            self.control_panel.update_status("Game is over.")


    def handle_input_hex_click(self, hex_coord: HexCoord):
        if not self.is_input_mode: return
        mode = self.input_mode_type
        player_color = self.state_cache.current_player_color

        if mode == "place_token":
            is_city = any(city.location == hex_coord for city in self.state_cache.cities)
            has_token = hex_coord in self.state_cache.player_token_locations
            if not is_city and not has_token:
                self.highlight_hexes = [hex_coord]
                self.control_panel.update_status(f"Place token at {hex_coord}? Submit or Cancel.")
            else:
                self.control_panel.update_status(f"Cannot place token: Hex {'is a city' if is_city else 'already occupied'}.")
                self.highlight_hexes = []

        elif mode == "mancala":
            if not self.selected_start_hex:
                if can_start_mancala_at(hex_coord, player_color, self.state_cache):
                    self.selected_start_hex = hex_coord
                    self.highlight_hexes = [hex_coord]
                    meeple_count = len(self.state_cache.hex_meeples.get(hex_coord, []))
                    self.control_panel.update_status(f"Start Mancala at {hex_coord} ({meeple_count} meeples). Click next hex.")
                else:
                    self.control_panel.update_status("Invalid start: Select a hex with your token.")
            else:
                if is_valid_mancala_step(self.highlight_hexes, hex_coord):
                    self.highlight_hexes.append(hex_coord)
                    # ... update status message ...
                else:
                    self.control_panel.update_status("Invalid step: Not adjacent or already in path.")
        
        elif mode == "upgrade":
            if can_select_for_upgrade(hex_coord, player_color, self.state_cache):
                self.helpers.handle_upgrade_hex_click(hex_coord)
            else:
                self.control_panel.update_status("Cannot upgrade: No upgradeable post found.")

        elif mode == "trade_route":
            if can_add_to_trade_route(hex_coord, player_color, self.state_cache):
                if hex_coord in self.highlight_hexes:
                    self.highlight_hexes.remove(hex_coord)
                else:
                    self.highlight_hexes.append(hex_coord)
            else:
                self.control_panel.update_status("You must have a post or center here.")


    def handle_checkbox_click(self, checkbox_id, is_checked):
        if checkbox_id == "show_trade_routes":
            self.show_trade_routes = not is_checked


    def handle_window_resize(self, new_width: int, new_height: int):
        self.width = max(new_width, SIDEBAR_WIDTH + 300)
        self.height = max(new_height, CONTROLS_HEIGHT + 300)
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        new_max_zoom = self.calculate_optimal_zoom(padding=10)
        self.update_zoom_limits()
        self.max_zoom = new_max_zoom
        self.zoom = new_max_zoom
        center_offset = self.calculate_optimal_board_center_offset(self.zoom)
        self.board_center_offset = [center_offset[0], center_offset[1]]
        self.last_resize_time = pygame.time.get_ticks()
        self.update_layout()
        dialog_width, dialog_height = 400, 200
        self.dialog_box.rect = pygame.Rect((self.width - dialog_width) // 2, (self.height - dialog_height) // 2, dialog_width, dialog_height)


    def start_input_mode(self, mode_type: str):
        if self.state_cache.is_terminal or self.state_cache.current_player_id < 0:
            self.control_panel.update_status("Cannot enter input mode now.")
            return
        self.is_input_mode = True
        self.input_mode_type = mode_type
        self.highlight_hexes = []
        self.selected_start_hex = None
        self.wants_post_on_mancala = False
        self.control_panel.update_status(f"Mode: {mode_type}. Select target hex(es).")


    def cancel_input_mode(self):
        if not self.is_input_mode: return
        self.is_input_mode = False
        self.input_mode_type = None
        self.highlight_hexes = []
        self.selected_start_hex = None
        self.wants_post_on_mancala = False
        self.helpers.update_status_from_cache()


    def submit_move(self):
        if not self.is_input_mode: return
        mode = self.input_mode_type
        move_string = None
        if mode == "place_token":
            if len(self.highlight_hexes) == 1:
                move_string = f"place {self.highlight_hexes[0]}"
        elif mode == "mancala":
            if self.selected_start_hex and len(self.highlight_hexes) > 1:
                path_str = ":".join(map(str, self.highlight_hexes[1:]))
                move_string = f"mancala {self.selected_start_hex}:{path_str}"
                if self.wants_post_on_mancala:
                    move_string += " post"
        elif mode == "upgrade":
            self.helpers.submit_upgrade_action()
            return
        elif mode == "trade_route":
            self.helpers.submit_trade_route()
            return
        if move_string:
            self.attempt_apply_action(move_string)
        else:
             self.control_panel.update_status("Submit failed: Incomplete move.")


    def attempt_apply_action(self, action_string: str):
        print(f"Attempting action: {action_string}")
        self.control_panel.update_status("Processing move...")
        success, message, new_state_json = self.game_interface.apply_action(action_string)
        if success:
            if self.parse_and_update_state(new_state_json):
                self.cancel_input_mode()
                return True
            else:
                self.control_panel.update_status("Error: Applied move, but failed to parse new state!")
                return False
        else:
            self.control_panel.update_status(f"Move failed: {message}")
            return False


    def draw(self):
        if self.is_resizing: return
        self.screen.fill(WHITE)
        draw_board_state(
            screen=self.screen, state_cache=self.state_cache,
            hex_to_pixel_func=self.hex_to_pixel, zoom=self.zoom,
            fonts=self.fonts, font_sizes=self.font_sizes,
            highlight_hexes=self.highlight_hexes, selected_start_hex=self.selected_start_hex,
            show_trade_routes=self.show_trade_routes
        )
        self.sidebar.draw(self.screen, self.state_cache, self.game_interface)
        self.control_panel.draw(self.screen, self.zoom, self.is_input_mode, self.input_mode_type, self.state_cache, self.show_trade_routes)
        self.dialog_box.draw(self.screen)
        pygame.display.flip()


    # --- Map displaying
    def _try_load_background_map(self):
        """Try to load a background map from common locations."""
        map_paths = [
            "mali_ba_map.jpg",
            "mali_ba_map.png", 
            "assets/mali_ba_map.jpg",
            "assets/mali_ba_map.png",
            os.path.join(os.path.dirname(__file__), "mali_ba_map.jpg"),
            os.path.join(os.path.dirname(__file__), "mali_ba_map.png"),
            os.path.join(os.path.dirname(__file__), "assets", "mali_ba_map.jpg"),
            os.path.join(os.path.dirname(__file__), "assets", "mali_ba_map.png"),
        ]
        
        for map_path in map_paths:
            if os.path.exists(map_path):
                load_background_map(map_path)
                self.background_map_enabled = True
                print(f"✅ Background map loaded from: {map_path}")
                return True
        
        print("ℹ️  No background map found.")
        return False

    
    def load_custom_background_map(self, map_file_path: str) -> bool:
        """Load a specific background map file."""
        if load_background_map(map_file_path):
            self.background_map_enabled = True
            print(f"✅ Custom background map loaded from: {map_file_path}")
            return True
        else:
            print(f"❌ Failed to load background map from: {map_file_path}")
            return False


    def calculate_optimal_zoom(self, padding: int = 10) -> float:
        """Calculates the optimal zoom level to fit the entire hex grid within the board area."""
        if not self.state_cache.valid_hexes:
            return 1.0  # Default zoom if no hexes exist

        board_area_width = self.width - SIDEBAR_WIDTH
        board_area_height = self.height - CONTROLS_HEIGHT

        # --- Find the pixel bounding box of the grid at a reference zoom of 1.0 ---
        min_px, max_px = float('inf'), float('-inf')
        min_py, max_py = float('inf'), float('-inf')
        
        # Use a reference size (HEX_SIZE/2) which is the radius at zoom=1.0
        size = HEX_SIZE / 2.0

        for h in self.state_cache.valid_hexes:
            # Calculate pixel center relative to a (0,0) origin
            px = size * (3.0 / 2.0) * h.x
            py = size * math.sqrt(3) * ((h.y - h.z) / 2.0)
            
            # Update min/max pixel coordinates
            min_px, max_px = min(min_px, px), max(max_px, px)
            min_py, max_py = min(min_py, py), max(max_py, py)

        # Calculate the total width and height of the grid at zoom=1.0,
        # including the radius of the hexes on the edges.
        grid_unzoomed_width = (max_px - min_px) + (size * 2)
        grid_unzoomed_height = (max_py - min_py) + (size * math.sqrt(3))

        if grid_unzoomed_width <= 0 or grid_unzoomed_height <= 0:
            return 1.0 # Avoid division by zero

        # --- Calculate the zoom required to fit width and height separately ---
        target_width = board_area_width - (2 * padding)
        target_height = board_area_height - (2 * padding)

        zoom_for_width = target_width / grid_unzoomed_width
        zoom_for_height = target_height / grid_unzoomed_height

        # The final zoom must be the smaller of the two to ensure everything fits
        return min(zoom_for_width, zoom_for_height)


    def calculate_optimal_board_center_offset(self, target_zoom: float) -> tuple:
        """Calculates the offset needed to center the grid within the board area."""
        if not self.state_cache.valid_hexes:
            return (0, 0)

        # --- Find the geometric center of the hex grid in cube coordinates ---
        sum_x = sum(h.x for h in self.state_cache.valid_hexes)
        sum_y = sum(h.y for h in self.state_cache.valid_hexes)
        sum_z = sum(h.z for h in self.state_cache.valid_hexes)
        count = len(self.state_cache.valid_hexes)
        
        avg_x, avg_y, avg_z = sum_x / count, sum_y / count, sum_z / count
        
        # --- Convert this average cube coordinate to a pixel offset ---
        # This tells us where the grid's center is relative to the grid's origin (0,0,0)
        size = (HEX_SIZE / 2.0) * target_zoom
        center_pixel_x = size * (3.0 / 2.0) * avg_x
        center_pixel_y = size * math.sqrt(3) * ((avg_y - avg_z) / 2.0)
        
        # The offset should be the negative of this pixel position to counteract it
        # and move the grid's center to the screen's center.
        # For some reason the y needs to be nudged higher 
        center_pixel_x -= 30
        center_pixel_y += 15
        return (-center_pixel_x, -center_pixel_y)


    def auto_fit_board(self):
        """Automatically sets the zoom and pan to fit the grid optimally in the view."""
        print("Auto-fitting board to view...")
        # Calculate the zoom needed with a 10px padding
        optimal_zoom = self.calculate_optimal_zoom(padding=10)
        self.zoom = max(self.min_zoom, min(self.max_zoom, optimal_zoom)) # Clamp to limits
        print(f"  - Optimal zoom calculated: {self.zoom:.2f}")
        
        # Based on the new zoom, calculate the centering offset
        center_offset = self.calculate_optimal_board_center_offset(self.zoom)
        self.board_center_offset = [center_offset[0], center_offset[1]]
        print(f"  - Optimal center offset: ({center_offset[0]:.1f}, {center_offset[1]:.1f})")
        
        # Apply all changes by updating the layout
        self.update_layout()


    def update_zoom_limits(self):
        """Updates zoom limits based on the board size."""
        self.min_zoom, self.max_zoom = (0.1, 10.0) # Simplified for now


    def calculate_dynamic_zoom_limits(self) -> tuple:
        """
        Calculate dynamic zoom limits based on effective board area (constrained by background map).
        Returns (min_zoom, max_zoom) tuple.
        """
        if not self.state_cache.valid_hexes:
            return (0.1, 10.0)  # Default fallback
        
        # Get effective dimensions considering background map
        effective_width, effective_height = self.get_effective_board_dimensions()
        
        if effective_width <= 0 or effective_height <= 0:
            return (0.1, 10.0)
        
        # Calculate hex grid bounds at zoom 1.0
        test_zoom = 1.0
        test_radius = (HEX_SIZE / 2.0) * test_zoom
        
        min_pixel_x = float('inf')
        max_pixel_x = float('-inf')
        min_pixel_y = float('inf')
        max_pixel_y = float('-inf')
        
        for hex_coord in self.state_cache.valid_hexes:
            sqrt3 = math.sqrt(3)
            pixel_x = test_radius * (3.0 / 2.0) * hex_coord.x
            pixel_y = test_radius * sqrt3 * ((hex_coord.y - hex_coord.z) / 2)
            
            min_pixel_x = min(min_pixel_x, pixel_x - test_radius)
            max_pixel_x = max(max_pixel_x, pixel_x + test_radius)
            min_pixel_y = min(min_pixel_y, pixel_y - test_radius)
            max_pixel_y = max(max_pixel_y, pixel_y + test_radius)
        
        total_pixel_width = max_pixel_x - min_pixel_x
        total_pixel_height = max_pixel_y - min_pixel_y
        
        if total_pixel_width <= 0 or total_pixel_height <= 0:
            return (0.1, 10.0)
        
        # Minimum zoom: smallest zoom where the entire grid fits within effective area
        min_zoom_width = effective_width / total_pixel_width
        min_zoom_height = effective_height / total_pixel_height
        calculated_min_zoom = min(min_zoom_width, min_zoom_height) * 0.8  # 80% margin
        
        # Maximum zoom: largest zoom where at least one hex is fully visible
        max_single_hex_width = 2 * test_radius
        max_single_hex_height = 2 * test_radius
        max_zoom_width = effective_width / max_single_hex_width
        max_zoom_height = effective_height / max_single_hex_height
        calculated_max_zoom = min(max_zoom_width, max_zoom_height) * 0.9  # 90% margin
        
        # Ensure reasonable bounds
        final_min_zoom = max(0.05, calculated_min_zoom)  # Never smaller than 0.05
        final_max_zoom = max(calculated_max_zoom, final_min_zoom * 2)  # At least 2x range
        final_max_zoom = min(final_max_zoom, 20.0)  # Cap at 20x for sanity
        
        print(f"Dynamic zoom limits (map-constrained): min={final_min_zoom:.3f}, max={final_max_zoom:.3f}")
        print(f"  Effective area: {effective_width}x{effective_height}")
        
        return (final_min_zoom, final_max_zoom)
    

    def get_effective_board_dimensions(self) -> tuple:
        """
        Get the effective board dimensions, constrained by background map if present.
        Returns (effective_width, effective_height) tuple.
        """
        board_area_width = self.width - SIDEBAR_WIDTH
        board_area_height = self.height - CONTROLS_HEIGHT
        
        # Import here to avoid circular imports
        from .visualizer_drawing import BACKGROUND_MAP, BACKGROUND_MAP_RECT
        
        if BACKGROUND_MAP is not None and BACKGROUND_MAP_RECT is not None:
            # Calculate how the background map would be scaled to fit the board area
            map_scale_x = board_area_width / BACKGROUND_MAP_RECT.width
            map_scale_y = board_area_height / BACKGROUND_MAP_RECT.height
            map_fit_scale = min(map_scale_x, map_scale_y)  # "fit" mode scaling
            
            # Calculate actual map dimensions when fitted to board
            scaled_map_width = int(BACKGROUND_MAP_RECT.width * map_fit_scale)
            scaled_map_height = int(BACKGROUND_MAP_RECT.height * map_fit_scale)
            
            # Use the smaller of board area or scaled map dimensions
            effective_width = min(board_area_width, scaled_map_width)
            effective_height = min(board_area_height, scaled_map_height)
            
            return (effective_width, effective_height)
        else:
            return (board_area_width, board_area_height)    

    # --- Draw overlay if we're in replay mode ---
    def draw_replay_overlay(self):
        """Draw replay mode information overlay in top left corner."""
        if not self.is_replay_mode or not hasattr(self, 'replay_manager'):
            return
        
        # Create semi-transparent background
        overlay_width = 300
        overlay_height = 80
        overlay = pygame.Surface((overlay_width, overlay_height))
        overlay.set_alpha(180)
        overlay.fill((0, 0, 50))  # Dark blue
        
        # Position in top-LEFT corner instead of top-right
        x = 10  # CHANGED FROM: self.width - overlay_width - 10
        y = 10
        
        self.screen.blit(overlay, (x, y))
        
        # Draw border
        pygame.draw.rect(self.screen, (100, 100, 255), (x, y, overlay_width, overlay_height), 2)
        
        # Draw text
        font = pygame.font.Font(None, 20)
        
        # Title
        title_text = font.render("REPLAY MODE", True, (255, 255, 255))
        self.screen.blit(title_text, (x + 10, y + 10))
        
        # Move info
        move_info = self.replay_manager.get_move_info()
        info_text = font.render(move_info, True, (200, 200, 255))
        self.screen.blit(info_text, (x + 10, y + 30))
        
        # Controls
        controls_text = font.render("↑/↓: Navigate  ESC: Exit", True, (150, 150, 255))
        self.screen.blit(controls_text, (x + 10, y + 50))

    # --- Utility Methods --- (see visualizer_other.py)

    # --- Main Loop ---
    def run(self):
        running = True
        is_dragging, drag_start_pos, last_mouse_pos = False, None, None
        self.helpers.update_status_from_cache()

        try:
            while running:
                current_time = pygame.time.get_ticks()
                if self.is_resizing and current_time - self.last_resize_time > RESIZE_WAIT:
                    self.is_resizing = False
                    self.handle_window_resize(self.screen.get_width(), self.screen.get_height())

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    # This gets triggered when it's time for a non-human player to play
                    elif event.type == pygame.USEREVENT + 1:
                        pygame.time.set_timer(pygame.USEREVENT + 1, 0) # Stop the timer
                        self._trigger_non_human_move()
                    elif event.type == pygame.VIDEORESIZE and not self.is_resizing:
                        self.is_resizing = True
                        self.last_resize_time = current_time
                    elif self.sidebar.handle_event(event):
                        continue
                    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.handle_click(event.pos)
                    elif event.type == pygame.KEYDOWN:
                        # --- ADD REPLAY NAVIGATION ---
                        if self.is_replay_mode:
                            print("Name of key pressed")
                            print(pygame.key.name(event.key))
                            if event.key in (pygame.K_RIGHT, pygame.K_DOWN):
                                if self.replay_manager.try_go_forward():
                                    self.parse_and_update_state(self.replay_manager.get_current_state_json())
                                    self.control_panel.update_status(self.replay_manager.get_move_info())
                            elif event.key in (pygame.K_PAGEDOWN, pygame.K_KP3):
                                for i in range(10):
                                    if not self.replay_manager.try_go_forward():
                                        print("Skipped forward less than 10 moves.")
                                        break
                                    else:
                                        print("Skipped forward 10 moves.")
                                        self.parse_and_update_state(self.replay_manager.get_current_state_json())
                                        self.control_panel.update_status(self.replay_manager.get_move_info())
                            elif event.key in (pygame.K_LEFT, pygame.K_UP):
                                if self.replay_manager.try_go_backward():
                                    self.parse_and_update_state(self.replay_manager.get_current_state_json())
                                    self.control_panel.update_status(self.replay_manager.get_move_info())
                            elif event.key in (pygame.K_PAGEUP, pygame.K_KP9):
                                for i in range(10):
                                    if not self.replay_manager.try_go_backward():
                                        print("Skipped backward less than 10 moves.")
                                        break
                                    else:
                                        print("Skipped backward 10 moves.")
                                        self.parse_and_update_state(self.replay_manager.get_current_state_json())
                                        self.control_panel.update_status(self.replay_manager.get_move_info())
                            elif event.key == pygame.K_ESCAPE:
                                running = False
                            continue # Don't process other key events in replay mode
                
                self.draw()
                self.clock.tick(60)
        except Exception as e:
            print(f"\n--- Error in Main Visualizer Loop ---\nError: {e}")
            traceback.print_exc()
        finally:
            self.helpers.cleanup()

    def debug_current_state(self):
        """Print current state for debugging."""
        print("\n🔍 === CURRENT VISUALIZER STATE DEBUG ===")
        print(f"Current Player ID: {self.state_cache.current_player_id}")
        print(f"Current Player Color: {self.state_cache.current_player_color}")
        print(f"Current Phase: {self.state_cache.current_phase}")
        print(f"Is Terminal: {self.state_cache.is_terminal}")
        
        print(f"Player Token Locations ({len(self.state_cache.player_token_locations)}):")
        for hex_coord, players in self.state_cache.player_token_locations.items():
            print(f"  {hex_coord}: {[p.name for p in players]}")
        
        # Also get the current state from the interface
        current_json = self.game_interface.get_current_state_string()
        if current_json:
            print(f"Interface state JSON length: {len(current_json)}")
            # Parse just the relevant parts
            try:
                data = json.loads(current_json)
                print(f"Interface currentPlayerId: {data.get('currentPlayerId', 'NOT_FOUND')}")
                print(f"Interface playerTokens: {data.get('playerTokens', 'NOT_FOUND')}")
                print(f"Interface playerTokenLocations: {data.get('playerTokenLocations', 'NOT_FOUND')}")
            except:
                print("Could not parse interface JSON")
        print("🔍 === END DEBUG ===\n")

# --- End BoardVisualizer Class ---
# --- END OF FILE visualizer_core.py ---