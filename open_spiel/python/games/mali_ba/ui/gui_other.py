import sys
sys.path.append("/media/robp/UD/Projects/mali_ba/open_spiel/python/games") # allow debugging in vs code
import pygame
from typing import List, Dict, Optional, Tuple
from mali_ba.classes.game_state import GameStateCache
from mali_ba.config import YELLOW, VIOLET, DARK_BLUE, INFO_PANEL_HEIGHT, PLAYER_COLOR_DICT, PlayerColor, Phase
from mali_ba.config import LIGHT_GRAY, DARK_GRAY, GRAY, GREEN, RED, BLUE, WHITE, BLACK


# InteractiveObject and InteractiveObjectManager remain the same as previous version
class InteractiveObject:
    """Represents a clickable object in the game."""
    def __init__(self, rect, name, is_active=True, is_visible=True, data=None):
        self.rect = rect
        self.name = name
        self.is_active = is_active
        self.is_visible = is_visible
        self.data = data # e.g., HexCoord for hexes, button ID for buttons

    def contains_point(self, pos):
        """Check if this object contains the given point."""
        if not self.is_active or not self.is_visible:
            return False
        return self.rect.collidepoint(pos)

    def __str__(self):
        return f"InteractiveObject(name={self.name}, active={self.is_active}, visible={self.is_visible}, data={self.data})"

class InteractiveObjectManager:
    """Manages interactive objects in the game."""
    def __init__(self):
        self.objects: List[InteractiveObject] = []

    def add_object(self, obj: InteractiveObject):
        """Add an interactive object to be managed."""
        self.objects.append(obj)
        return obj

    def remove_object(self, obj: InteractiveObject):
        """Remove an interactive object from management."""
        if obj in self.objects:
            self.objects.remove(obj)

    def clear_objects(self):
        """Remove all objects."""
        self.objects.clear()

    def find_object_at(self, pos: Tuple[int, int]) -> Optional[InteractiveObject]:
        """Find the topmost active and visible object containing the given point."""
        # Prioritize control panel buttons if they exist
        # Note: This assumes control panel objects are added *after* hex objects.
        # A more robust way might involve layers or checking specific managers.
        for obj in reversed(self.objects): # Check topmost first
             # Quick check: If click is in control panel area, only check control buttons
             # This needs access to the control panel rect, which isn't ideal here.
             # We'll rely on the drawing order for now (Interactive hexes added first).
            if obj.contains_point(pos):
                return obj
        return None

    def find_objects_with_name(self, name: str) -> List[InteractiveObject]:
        """Find all objects with the given name."""
        return [obj for obj in self.objects if obj.name == name]

# --- UI Components ---
# Sidebar is below ----------------------------------
# class Sidebar:
#     def __init__(self, rect, fonts):
#         self.rect = rect
#         self.font = fonts['font']
#         self.small_font = fonts['small_font']
#         self.large_font = fonts['large_font']
#         self.info_panel_height = INFO_PANEL_HEIGHT

#     def update_rect(self, new_rect):
#         self.rect = new_rect

#     def draw(self, surface, state_cache: GameStateCache):
#         pygame.draw.rect(surface, LIGHT_GRAY, self.rect)
#         pygame.draw.line(surface, DARK_GRAY, (self.rect.left, 0), (self.rect.left, self.rect.height), 2)

#         y_offset = 10

#         # Title
#         title_text = self.large_font.render("Mali-Ba Game Info", True, BLACK)
#         surface.blit(title_text, (self.rect.x + 10, y_offset))
#         y_offset += title_text.get_height() + 10

#         # Current Player Info
#         player_id = state_cache.current_player_id
#         phase = state_cache.current_phase
#         player_str = "Chance" if phase == Phase.START else ( "Terminal" if state_cache.is_terminal else f"Player {player_id + 1}" )
#         player_color_enum = state_cache.current_player_color
#         player_color_name = player_color_enum.name if player_color_enum != PlayerColor.EMPTY else "N/A"
#         player_color_rgb = PLAYER_COLOR_DICT.get(player_color_enum, GRAY)

#         current_player_text = self.font.render(f"Current: {player_str} ({player_color_name})", True, BLACK)
#         surface.blit(current_player_text, (self.rect.x + 10, y_offset))
#         # Draw color indicator next to text
#         pygame.draw.circle(surface, player_color_rgb, (self.rect.x + current_player_text.get_width() + 25, y_offset + current_player_text.get_height()//2), 8)
#         pygame.draw.circle(surface, BLACK, (self.rect.x + current_player_text.get_width() + 25, y_offset + current_player_text.get_height()//2), 8, 1)
#         y_offset += current_player_text.get_height() + 5

#         # Game Phase
#         phase_name = phase.name if phase else "Unknown"
#         phase_text = self.font.render(f"Phase: {phase_name}", True, BLACK)
#         surface.blit(phase_text, (self.rect.x + 10, y_offset))
#         y_offset += phase_text.get_height() + 15

#         # Draw Player Info Panels
#         num_players = len(state_cache.game_player_colors)
#         for p_id in range(num_players):
#             p_color_enum = state_cache.game_player_colors[p_id]
#             color_rgb = PLAYER_COLOR_DICT.get(p_color_enum, GRAY)

#             panel_rect = pygame.Rect(self.rect.x + 5, y_offset, self.rect.width - 10, self.info_panel_height)
#             # Highlight current player's panel
#             border_color = YELLOW if p_id == player_id and phase != Phase.START and not state_cache.is_terminal else BLACK
#             border_width = 3 if p_id == player_id and phase != Phase.START and not state_cache.is_terminal else 1

#             pygame.draw.rect(surface, color_rgb, panel_rect, border_radius=5)
#             pygame.draw.rect(surface, border_color, panel_rect, border_width, border_radius=5)

#             # Player Name
#             name_text = self.font.render(f"Player {p_id + 1} ({p_color_enum.name})", True, BLACK)
#             surface.blit(name_text, (panel_rect.x + 10, panel_rect.y + 5))

#             # Player Resources
#             if p_id < len(state_cache.common_goods):
#                 common_goods = state_cache.common_goods[p_id]
#                 rare_goods = state_cache.rare_goods[p_id]

#                 # Common Goods
#                 common_total = sum(common_goods.values())
#                 # common_str = ", ".join([f"{name[:4]}:{c}" for name, c in common_goods.items()]) if common_goods else "None"
#                 common_text = self.small_font.render(f"Common Goods: {common_total}", True, BLACK)
#                 surface.blit(common_text, (panel_rect.x + 10, panel_rect.y + 25))

#                 # Rare Goods
#                 rare_total = sum(rare_goods.values())
#                 # rare_str = ", ".join([f"{name[:4]}:{c}" for name, c in rare_goods.items()]) if rare_goods else "None"
#                 rare_text = self.small_font.render(f"Rare Goods: {rare_total}", True, BLACK)
#                 surface.blit(rare_text, (panel_rect.x + 10, panel_rect.y + 45))

#             y_offset += self.info_panel_height + 10

class Sidebar:
    def __init__(self, rect, fonts):
        self.rect = rect
        self.font = fonts['font']
        self.small_font = fonts['small_font']
        self.large_font = fonts['large_font']
        self.info_panel_height = INFO_PANEL_HEIGHT
        
        # Scrolling properties
        self.scroll_offset = 0
        self.max_scroll = 0
        self.scroll_bar_width = 15
        self.scroll_bar_rect = pygame.Rect(0, 0, 0, 0)  # Will be updated in draw
        self.is_dragging_scrollbar = False
        self.scrollbar_drag_start = 0
        
        # Create a surface to draw the content to
        self.content_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)

    def update_rect(self, new_rect):
        self.rect = new_rect
        self.content_surface = pygame.Surface((new_rect.width, new_rect.height), pygame.SRCALPHA)

    def handle_event(self, event):
        """Handle mouse events for scrolling."""
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll up
                self.scroll_offset = max(0, self.scroll_offset - 20)
                return True
            elif event.button == 5:  # Scroll down
                self.scroll_offset = min(self.max_scroll, self.scroll_offset + 20)
                return True
            elif event.button == 1 and self.scroll_bar_rect.collidepoint(event.pos):  # Left click on scrollbar
                self.is_dragging_scrollbar = True
                self.scrollbar_drag_start = event.pos[1] - self.scroll_bar_rect.y
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 and self.is_dragging_scrollbar:
                self.is_dragging_scrollbar = False
                return True
        elif event.type == pygame.MOUSEMOTION:
            if self.is_dragging_scrollbar:
                # Calculate new scroll position based on drag
                content_height = self.content_height
                visible_height = self.rect.height
                
                if content_height > visible_height:
                    # Calculate the ratio of content that's visible
                    visible_ratio = visible_height / content_height
                    
                    # Calculate scrollbar height based on visible ratio
                    scrollbar_height = max(20, int(visible_height * visible_ratio))
                    
                    # Calculate the scrollable area
                    scrollable_height = visible_height - scrollbar_height
                    
                    # Calculate new position with drag offset
                    new_y = event.pos[1] - self.scrollbar_drag_start
                    
                    # Constrain to scrollable area
                    new_y = max(self.rect.y, min(self.rect.y + scrollable_height, new_y))
                    
                    # Convert scrollbar position to content offset
                    scroll_ratio = (new_y - self.rect.y) / scrollable_height
                    self.scroll_offset = int(scroll_ratio * (content_height - visible_height))
                    return True
        
        return False  # Event not handled

    def draw(self, surface, state_cache: GameStateCache, game_interface=None):
        # Clear the content surface
        self.content_surface.fill((0, 0, 0, 0))
        
        # Draw background
        pygame.draw.rect(self.content_surface, LIGHT_GRAY, pygame.Rect(0, 0, self.rect.width, self.rect.height))
        pygame.draw.line(self.content_surface, DARK_GRAY, (0, 0), (0, self.rect.height), 2)

        y_offset = 10

        # Title
        title_text = self.large_font.render("Mali-Ba Game Info", True, BLACK)
        self.content_surface.blit(title_text, (10, y_offset))
        y_offset += title_text.get_height() + 10

        # Current Player Info
        player_id = state_cache.current_player_id
        phase = state_cache.current_phase
        player_str = "Chance" if phase == Phase.START else ("Terminal" if state_cache.is_terminal else f"Player {player_id + 1}")
        player_color_enum = state_cache.current_player_color
        player_color_name = player_color_enum.name if player_color_enum != PlayerColor.EMPTY else "N/A"
        player_color_rgb = PLAYER_COLOR_DICT.get(player_color_enum, GRAY)

        current_player_text = self.font.render(f"Current: {player_str} ({player_color_name})", True, BLACK)
        self.content_surface.blit(current_player_text, (10, y_offset))
        # Draw color indicator next to text
        pygame.draw.circle(self.content_surface, player_color_rgb, 
                          (current_player_text.get_width() + 25, y_offset + current_player_text.get_height()//2), 8)
        pygame.draw.circle(self.content_surface, BLACK, 
                          (current_player_text.get_width() + 25, y_offset + current_player_text.get_height()//2), 8, 1)
        y_offset += current_player_text.get_height() + 5

        # Game Phase
        phase_name = phase.name if phase else "Unknown"
        phase_text = self.font.render(f"Phase: {phase_name}", True, BLACK)
        self.content_surface.blit(phase_text, (10, y_offset))
        y_offset += phase_text.get_height() + 15

        # Draw Player Info Panels
        num_players = len(state_cache.game_player_colors)
        for p_id in range(num_players):
            p_color_enum = state_cache.game_player_colors[p_id]
            color_rgb = PLAYER_COLOR_DICT.get(p_color_enum, GRAY)
            
            # Calculate what will be displayed first to determine height
            
            # Start with base panel contents
            panel_y_offset = 30  # Height for player name and basic info
            
            # Calculate space needed for goods
            if p_id < len(state_cache.common_goods):
                common_goods = state_cache.common_goods[p_id]
                rare_goods = state_cache.rare_goods[p_id]
                
                # Space for common goods header
                panel_y_offset += 20
                
                # Space for each common good
                for good_name, count in common_goods.items():
                    if count > 0:
                        panel_y_offset += 15
                
                # Space between common and rare goods
                panel_y_offset += 5
                
                # Space for rare goods header
                panel_y_offset += 20
                
                # Space for each rare good
                for good_name, count in rare_goods.items():
                    if count > 0:
                        panel_y_offset += 15
            
            # Ensure minimum height
            panel_height = max(self.info_panel_height, panel_y_offset)
            
            # Create panel rectangle with calculated height
            panel_rect = pygame.Rect(5, y_offset, self.rect.width - 20, panel_height)
            
            # Highlight current player's panel
            border_color = YELLOW if p_id == player_id and phase != Phase.START and not state_cache.is_terminal else BLACK
            border_width = 3 if p_id == player_id and phase != Phase.START and not state_cache.is_terminal else 1
            
            # Now draw the correctly sized panel
            pygame.draw.rect(self.content_surface, color_rgb, panel_rect, border_radius=5)
            pygame.draw.rect(self.content_surface, border_color, panel_rect, border_width, border_radius=5)
            
            # Player Name - only color a small area behind the name
            name_text = self.large_font.render(f"Player {p_id + 1} ({p_color_enum.name})", True, BLACK)
            name_rect = pygame.Rect(panel_rect.x + 5, panel_rect.y + 5, name_text.get_width() + 10, name_text.get_height() + 2)
            
            # Optional: Draw a darker background behind just the name 
            # pygame.draw.rect(self.content_surface, darken_color(color_rgb), name_rect, border_radius=3)
            
            # Draw the name
            self.content_surface.blit(name_text, (panel_rect.x + 10, panel_rect.y + 5))

            if hasattr(state_cache, 'player_posts_supply') and p_id < len(state_cache.player_posts_supply):
                # Check if posts are unlimited based on game rules
                posts_are_unlimited = self._are_posts_unlimited(game_interface)
                
                if posts_are_unlimited:
                    post_text = "Trading Posts: Unlimited"
                else:
                    post_supply = state_cache.player_posts_supply[p_id]
                    post_text = f"Trading Posts: {post_supply}"
                
                supply_render = self.small_font.render(post_text, True, DARK_BLUE)
                # Position in top right of player panel
                self.content_surface.blit(supply_render, 
                (panel_rect.right - supply_render.get_width() - 10, panel_rect.y + 5))            
    
            # Start drawing goods information
            panel_y = panel_rect.y + 30
            
            # Player Resources
            if p_id < len(state_cache.common_goods):
                common_goods = state_cache.common_goods[p_id]
                rare_goods = state_cache.rare_goods[p_id]
                
                # Common Goods header
                common_total = sum(common_goods.values())
                common_header = self.font.render(f"Common Goods: {common_total}", True, BLACK)
                self.content_surface.blit(common_header, (panel_rect.x + 10, panel_y))
                panel_y += common_header.get_height() + 2
                
                # List each common good type
                if common_goods:
                    for good_name, count in sorted(common_goods.items()):
                        if count > 0:  # Only show non-zero quantities
                            good_text = self.font.render(f"  • {good_name}: {count}", True, BLACK)
                            self.content_surface.blit(good_text, (panel_rect.x + 15, panel_y))
                            panel_y += good_text.get_height()
                
                panel_y += 5  # Add spacing between common and rare goods
                
                # Rare Goods header
                rare_total = sum(rare_goods.values())
                rare_header = self.font.render(f"Rare Goods: {rare_total}", True, BLACK)
                self.content_surface.blit(rare_header, (panel_rect.x + 10, panel_y))
                panel_y += rare_header.get_height() + 2
                
                # List each rare good type
                if rare_goods:
                    for good_name, count in sorted(rare_goods.items()):
                        if count > 0:  # Only show non-zero quantities
                            good_text = self.font.render(f"  • {good_name}: {count}", True, BLACK)
                            self.content_surface.blit(good_text, (panel_rect.x + 15, panel_y))
                            panel_y += good_text.get_height()
            
            # Update y_offset for next panel
            y_offset = panel_rect.y + panel_rect.height + 10
        
        # Store the total content height for scrolling calculations
        self.content_height = y_offset
        self.max_scroll = max(0, self.content_height - self.rect.height)
        
        # Draw the visible portion of the content to the main surface
        visible_rect = pygame.Rect(0, self.scroll_offset, self.rect.width, self.rect.height)
        surface.blit(self.content_surface, self.rect.topleft, visible_rect)
        
        # Draw scrollbar if content exceeds visible area
        if self.content_height > self.rect.height:
            # Calculate scrollbar dimensions
            visible_ratio = self.rect.height / self.content_height
            scrollbar_height = max(20, int(self.rect.height * visible_ratio))
            
            # Calculate scrollbar position
            scroll_ratio = 0 if self.max_scroll == 0 else self.scroll_offset / self.max_scroll
            scrollbar_y = self.rect.y + int(scroll_ratio * (self.rect.height - scrollbar_height))
            
            # Create and store scrollbar rect
            self.scroll_bar_rect = pygame.Rect(
                self.rect.right - self.scroll_bar_width, 
                scrollbar_y,
                self.scroll_bar_width, 
                scrollbar_height
            )
            
            # Draw scrollbar
            pygame.draw.rect(surface, DARK_GRAY, self.scroll_bar_rect, border_radius=5)
            pygame.draw.rect(surface, BLACK, self.scroll_bar_rect, 1, border_radius=5)
    
    def _are_posts_unlimited(self, game_interface) -> bool:
        """Determine if trading posts are unlimited based on game rules."""
        if not game_interface:
            return False  # Conservative default
            
        try:
            # Check if game_interface has board_config with rules
            if hasattr(game_interface, 'board_config') and game_interface.board_config:
                board_config = game_interface.board_config
                
                # Check the posts_per_player rule from game_rules
                if hasattr(board_config, 'game_rules'):
                    posts_per_player = board_config.game_rules.posts_per_player
                    # -1 or very large number = unlimited
                    return posts_per_player == -1 or posts_per_player >= 1000
                    
                # Alternative: check game_params if that's where you stored it
                elif hasattr(board_config, 'game_params') and board_config.game_params:
                    posts_per_player = board_config.game_params.get('posts_per_player', 6)
                    return posts_per_player == -1 or posts_per_player >= 1000
                    
        except Exception as e:
            print(f"⚠️ Error checking posts rule: {e}")
        
        return False  # Conservative default: assume limited

class ControlPanel:
    def __init__(self, rect, font):
        self.rect = rect
        self.font = font
        self.status_message = "Game Started. Parsing state..."
        self.buttons: Dict[str, pygame.Rect] = {} # Store button name -> rect mapping

    def update_rect(self, new_rect):
        self.rect = new_rect

    def update_status(self, message: str):
        self.status_message = message

    def __init__(self, rect, font):
        self.rect = rect
        self.font = font
        self.status_message = "Game Started. Parsing state..."
        self.buttons: Dict[str, pygame.Rect] = {}  # Store button name -> rect mapping
        self.checkboxes: Dict[str, Tuple[pygame.Rect, bool]] = {}  # Store checkbox name -> (rect, checked) mapping

    def update_rect(self, new_rect):
        self.rect = new_rect

    def update_status(self, message: str):
        self.status_message = message

    def draw(self, surface, zoom, is_input_mode, input_mode_type, state_cache: GameStateCache, show_trade_routes: bool = True):
        self.buttons.clear()  # Clear old buttons before drawing new ones
        self.checkboxes.clear()  # Clear old checkboxes before drawing new ones
        pygame.draw.rect(surface, LIGHT_GRAY, self.rect)
        pygame.draw.line(surface, DARK_GRAY, (0, self.rect.top), (self.rect.width, self.rect.top), 2)

        # --- Top Row: Controls ---
        top_row_y = self.rect.y + 5
        button_height = 30
        button_padding = 10
        current_x = 10

        # Zoom Info & Buttons
        # --- ZOOM CONTROLS HIDDEN ---
        # Skip the zoom text and buttons entirely, just add equivalent spacing
        current_x += 150  # Add equivalent space where zoom controls were (roughly zoom text + 2 buttons + padding)

        # zoom_text = self.font.render(f"Zoom: {zoom:.1f}x", True, BLACK)
        # surface.blit(zoom_text, (current_x, top_row_y + (button_height - zoom_text.get_height()) // 2))
        # current_x += zoom_text.get_width() + button_padding

        # # Draw '-' button first
        # zoom_out_rect = pygame.Rect(current_x, top_row_y, 30, button_height)
        # pygame.draw.rect(surface, GRAY, zoom_out_rect, border_radius=3)
        # pygame.draw.rect(surface, DARK_GRAY, zoom_out_rect, 1, border_radius=3)
        # zoom_out_text = self.font.render("-", True, BLACK)
        # surface.blit(zoom_out_text, (zoom_out_rect.centerx - zoom_out_text.get_width() // 2, zoom_out_rect.centery - zoom_out_text.get_height() // 2))
        # self.buttons["zoom_out"] = zoom_out_rect  # Assign rect to correct ID
        # current_x += zoom_out_rect.width + 5

        # # Draw '+' button second
        # zoom_in_rect = pygame.Rect(current_x, top_row_y, 30, button_height)
        # pygame.draw.rect(surface, GRAY, zoom_in_rect, border_radius=3)
        # pygame.draw.rect(surface, DARK_GRAY, zoom_in_rect, 1, border_radius=3)
        # zoom_in_text = self.font.render("+", True, BLACK)
        # surface.blit(zoom_in_text, (zoom_in_rect.centerx - zoom_in_text.get_width() // 2, zoom_in_rect.centery - zoom_in_text.get_height() // 2))
        # self.buttons["zoom_in"] = zoom_in_rect  # Assign rect to correct ID
        # current_x += zoom_in_rect.width + button_padding * 2  # More space before next group

        # Trade Routes Toggle Checkbox
        # Draw checkbox text
        checkbox_text = self.font.render("Display Trade Routes?", True, BLACK)
        surface.blit(checkbox_text, (current_x, top_row_y + (button_height - checkbox_text.get_height()) // 2))
        current_x += checkbox_text.get_width() + 8  # Space between text and checkbox

        # Draw checkbox
        checkbox_size = button_height - 10  # Smaller than button height
        checkbox_rect = pygame.Rect(current_x, top_row_y + 5, checkbox_size, checkbox_size)
        pygame.draw.rect(surface, WHITE, checkbox_rect)  # Checkbox background
        pygame.draw.rect(surface, BLACK, checkbox_rect, 1)  # Checkbox border

        # Draw checkbox state (checked or unchecked)
        if show_trade_routes:  # Use the passed parameter
            # Draw checkmark or fill
            inset = 3  # Margin inside checkbox
            inner_rect = pygame.Rect(checkbox_rect.x + inset, checkbox_rect.y + inset, 
                                    checkbox_rect.width - 2*inset, checkbox_rect.height - 2*inset)
            pygame.draw.rect(surface, DARK_BLUE, inner_rect)

        # Store checkbox in dict with its current state
        self.checkboxes["show_trade_routes"] = (checkbox_rect, show_trade_routes)
        
        current_x += checkbox_rect.width + button_padding * 2  # More space before next group

        # Mode/Action Buttons (Same logic as before)
        if is_input_mode:
            # Submit Button
            submit_rect = pygame.Rect(current_x, top_row_y, 120, button_height)
            pygame.draw.rect(surface, GREEN, submit_rect, border_radius=3)
            pygame.draw.rect(surface, DARK_GRAY, submit_rect, 1, border_radius=3)
            submit_text = self.font.render("Submit Move", True, BLACK)
            surface.blit(submit_text, (submit_rect.centerx - submit_text.get_width() // 2, submit_rect.centery - submit_text.get_height() // 2))
            self.buttons["submit"] = submit_rect
            current_x += submit_rect.width + button_padding

            # Cancel Button
            cancel_rect = pygame.Rect(current_x, top_row_y, 120, button_height)
            pygame.draw.rect(surface, RED, cancel_rect, border_radius=3)
            pygame.draw.rect(surface, DARK_GRAY, cancel_rect, 1, border_radius=3)
            cancel_text = self.font.render("Cancel", True, BLACK)
            surface.blit(cancel_text, (cancel_rect.centerx - cancel_text.get_width() // 2, cancel_rect.centery - cancel_text.get_height() // 2))
            self.buttons["cancel"] = cancel_rect
            current_x += cancel_rect.width + button_padding
        elif not state_cache.is_terminal:  # Only show mode buttons if not terminal
            # Buttons depend on the current game phase
            modes = []
            phase = state_cache.current_phase
            # For DEBUG - if we have updated the state, fix the phase
            if phase == Phase.PLACE_TOKEN and len(state_cache.trade_posts_locations) > 0:
                phase = Phase.PLAY
                self.status_message = "Continue."
            # end DEBUG
            if phase == Phase.PLACE_TOKEN:
                modes.append(("Place Token", "place_token"))
            elif phase == Phase.PLAY:
                modes.append(("Pass Turn", "pass"))
                modes.append(("Mancala Move", "mancala"))
                modes.append(("Upgrade Post", "upgrade"))
                modes.append(("Take Income", "take_income"))
                modes.append(("Trade Routes", "trade_route")) 

            for label, mode_id in modes:
                mode_rect = pygame.Rect(current_x, top_row_y, 140, button_height)
                pygame.draw.rect(surface, BLUE, mode_rect, border_radius=3)
                pygame.draw.rect(surface, DARK_GRAY, mode_rect, 1, border_radius=3)
                mode_text = self.font.render(label, True, WHITE)
                surface.blit(mode_text, (mode_rect.centerx - mode_text.get_width() // 2, mode_rect.centery - mode_text.get_height() // 2))
                self.buttons[mode_id] = mode_rect  # Use mode_id as key
                current_x += mode_rect.width + button_padding

        # --- Bottom Row: Status Message ---
        status_y = self.rect.y + button_height + 15  # Position below buttons
        status_text = self.font.render(self.status_message, True, BLACK)
        surface.blit(status_text, (10, status_y))


class DialogBox:
    """A dialog box for user prompts with flexible layout options."""
    def __init__(self, rect, font, title="Dialog", message="", options=None):
        self.rect = rect
        self.font = font
        self.title = title
        self.message = message
        self.options = options or ["Yes", "No", "Cancel"]
        self.buttons = {}  # Will store rects for each button
        self.active = False
        self.result = None  # Will store the result of the dialog
        self.dialog_type = None  # Store the type of dialog for context
        self.context_data = {}  # Store additional context data for the dialog
        self.layout = "horizontal"  # New: layout option
        
    def show(self, title=None, message=None, options=None, dialog_type=None, context_data=None, layout="horizontal"):
        """Show the dialog with optional new content and layout."""
        if title:
            self.title = title
        if message:
            self.message = message
        if options:
            self.options = options
        self.dialog_type = dialog_type or "generic"
        self.context_data = context_data or {}
        self.layout = layout
        self.active = True
        self.result = None
        return self  # Allow method chaining
        
    def hide(self):
        """Hide the dialog without setting a result."""
        self.active = False
        self.result = None
        return self  # Allow method chaining
        
    def _get_text_width(self, text):
        """Get the width of text when rendered."""
        return self.font.size(text)[0]
        
    def _should_use_vertical_layout(self):
        """Determine if vertical layout should be used based on text length."""
        if self.layout == "vertical":
            return True
        elif self.layout == "horizontal":
            return False
        else:  # "auto"
            # Auto-detect based on text length
            max_option_width = max(self._get_text_width(option) for option in self.options) if self.options else 0
            button_width = max(80, max_option_width + 20)  # Minimum 80px, plus padding
            total_horizontal_width = len(self.options) * button_width + (len(self.options) - 1) * 10
            available_width = self.rect.width - 40  # Account for margins
            
            return total_horizontal_width > available_width
        
    def draw(self, surface):
        """Draw the dialog box if active."""
        if not self.active:
            return
        
        # Semi-transparent overlay for the whole screen
        overlay = pygame.Surface((surface.get_width(), surface.get_height()))
        overlay.set_alpha(128)
        overlay.fill((0, 0, 0))
        surface.blit(overlay, (0, 0))
        
        # Determine layout
        use_vertical = self._should_use_vertical_layout()
        
        # Calculate dialog size based on content
        if use_vertical:
            # Make dialog taller for vertical buttons
            dialog_height = self.rect.height + (len(self.options) - 2) * 35  # Extra height for vertical buttons
            dialog_rect = pygame.Rect(
                self.rect.x, 
                self.rect.centery - dialog_height // 2,
                self.rect.width, 
                dialog_height
            )
        else:
            dialog_rect = self.rect
        
        # Draw dialog background
        pygame.draw.rect(surface, LIGHT_GRAY, dialog_rect)
        pygame.draw.rect(surface, DARK_GRAY, dialog_rect, 2)

        # Draw title
        title_text = self.font.render(self.title, True, BLACK)
        title_rect = title_text.get_rect(
            centerx=dialog_rect.centerx,
            top=dialog_rect.top + 10
        )
        surface.blit(title_text, title_rect)
        
        # Draw message (can be multi-line)
        message_lines = self.message.split('\n')
        line_height = self.font.get_linesize()
        message_y = title_rect.bottom + 15
        
        for line in message_lines:
            message_text = self.font.render(line, True, BLACK)
            message_rect = message_text.get_rect(
                centerx=dialog_rect.centerx,
                top=message_y
            )
            surface.blit(message_text, message_rect)
            message_y += line_height + 2
        
        # Draw buttons based on layout
        self.buttons.clear()
        
        if use_vertical:
            self._draw_vertical_buttons(surface, dialog_rect, message_y + 15)
        else:
            self._draw_horizontal_buttons(surface, dialog_rect)
            
    def _draw_horizontal_buttons(self, surface, dialog_rect):
        """Draw buttons in horizontal layout."""
        button_width = 80
        button_height = 30
        total_width = len(self.options) * button_width + (len(self.options) - 1) * 10
        start_x = dialog_rect.centerx - total_width // 2
        button_y = dialog_rect.bottom - button_height - 15
        
        for i, option in enumerate(self.options):
            button_x = start_x + i * (button_width + 10)
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
            
            self._draw_button(surface, button_rect, option)
            
    def _draw_vertical_buttons(self, surface, dialog_rect, start_y):
        """Draw buttons in vertical layout."""
        # Calculate button width based on longest text
        max_text_width = max(self._get_text_width(option) for option in self.options) if self.options else 0
        button_width = max(120, max_text_width + 30)  # Minimum width with padding
        button_height = 30
        button_spacing = 10
        
        # Center buttons horizontally
        button_x = dialog_rect.centerx - button_width // 2
        current_y = start_y
        
        for option in self.options:
            button_rect = pygame.Rect(button_x, current_y, button_width, button_height)
            self._draw_button(surface, button_rect, option)
            current_y += button_height + button_spacing
            
    def _draw_button(self, surface, button_rect, option):
        """Draw a single button with appropriate styling."""
        # Different colors for different buttons
        if option == "Yes":
            button_color = GREEN
        elif option == "No":
            button_color = BLUE
        elif option == "Cancel":
            button_color = RED
        else:
            button_color = BLUE
            
        pygame.draw.rect(surface, button_color, button_rect)
        pygame.draw.rect(surface, DARK_GRAY, button_rect, 1)
        
        # Handle text that might be too long for the button
        option_text = self.font.render(option, True, WHITE)
        text_width = option_text.get_width()
        
        if text_width > button_rect.width - 10:  # Text too wide, truncate
            # Simple truncation - you could implement ellipsis here
            truncated_option = option
            while self._get_text_width(truncated_option + "...") > button_rect.width - 10 and len(truncated_option) > 1:
                truncated_option = truncated_option[:-1]
            option_text = self.font.render(truncated_option + "...", True, WHITE)
        
        text_rect = option_text.get_rect(center=button_rect.center)
        surface.blit(option_text, text_rect)
        
        self.buttons[option] = button_rect
            
    def handle_click(self, pos):
        """Handle clicks on the dialog box. Returns the clicked option or None."""
        if not self.active:
            return None
            
        for option, button_rect in self.buttons.items():
            if button_rect.collidepoint(pos):
                self.result = option
                self.active = False
                return option
                
        return None