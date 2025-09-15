# --- START OF FILE visualizer_drawing.py ---
import sys
sys.path.append("/media/robp/UD/Projects/mali_ba/open_spiel/python/games") # allow debugging in vs code
import pygame
import os
import math
from typing import Dict, List, Tuple, Set, Optional, Callable

from mali_ba.config import *
from mali_ba.classes.classes_other import TradePost, City, HexCoord, TradePostType, TradeRoute
from mali_ba.classes.game_state import GameStateCache
from mali_ba.config import PlayerColor # Explicit import for clarity

BACKGROUND_MAP: Optional[pygame.Surface] = None
BACKGROUND_MAP_RECT: Optional[pygame.Rect] = None

# Type alias for the hex_to_pixel function signature
HexToPixelFunc = Callable[[HexCoord], Tuple[int, int]]

# --- Individual Drawing Functions ---

def draw_hex(screen: pygame.Surface, hex_coord: HexCoord, hex_to_pixel_func: HexToPixelFunc,
             zoom: float, fonts: Dict, highlight_hexes: List[HexCoord], selected_start_hex: Optional[HexCoord],
             hex_transparency: int = 128):
    """Enhanced hex drawing with transparency support for background maps."""
    center_x, center_y = hex_to_pixel_func(hex_coord)
    radius = (HEX_SIZE / 2.0) * zoom
    if radius < 1: return

    size = radius
    points = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.pi / 180 * angle_deg
        x = center_x + size * math.cos(angle_rad)
        y = center_y + size * math.sin(angle_rad)
        points.append((round(x), round(y)))

    # Determine base colors
    fill_color = LIGHT_GRAY
    is_selected_start = hex_coord == selected_start_hex
    is_highlighted = hex_coord in highlight_hexes

    if is_selected_start: 
        fill_color = YELLOW
    elif is_highlighted: 
        fill_color = (200, 200, 0) # Highlight color

    # If we have a background map, make hexes semi-transparent
    if BACKGROUND_MAP is not None:
        # Create a surface with per-pixel alpha for transparency
        hex_surface = pygame.Surface((int(size * 2), int(size * 2)), pygame.SRCALPHA)
        
        # Adjust points for the temporary surface (center them)
        temp_points = [(x - center_x + size, y - center_y + size) for x, y in points]
        
        # Make the fill color semi-transparent using the configured transparency
        transparent_fill = (*fill_color[:3], hex_transparency)  # RGBA with configured alpha
        
        # Draw on the temporary surface
        pygame.draw.polygon(hex_surface, transparent_fill, temp_points)
        
        # Draw border (fully opaque)
        line_width = 3 if is_selected_start else (2 if is_highlighted else 1)
        border_color = (*DARK_GRAY[:3], 255) if len(DARK_GRAY) >= 3 else (64, 64, 64, 255)
        pygame.draw.polygon(hex_surface, border_color, temp_points, line_width)
        
        # Blit the transparent hex to the main screen
        screen.blit(hex_surface, (center_x - size, center_y - size))
    else:
        # No background map - draw normally (opaque)
        pygame.draw.polygon(screen, fill_color, points)
        line_width = 3 if is_selected_start else (2 if is_highlighted else 1)
        pygame.draw.polygon(screen, DARK_GRAY, points, line_width)

    # Draw coordinates if zoomed in enough
    if zoom > 1.0:
         coord_font = fonts['small_font']
         try:
             coord_text = coord_font.render(f"{hex_coord.x},{hex_coord.y},{hex_coord.z}", True, DARK_GRAY)
             text_rect = coord_text.get_rect(center=(center_x, center_y - round(size*0.7)))
             screen.blit(coord_text, text_rect)
         except AttributeError: # Handle case where hex_coord might be None temporarily
             pass


def draw_city(screen: pygame.Surface, hex_coord: HexCoord, city: City, hex_to_pixel_func: HexToPixelFunc, zoom: float, fonts: Dict, font_sizes: Dict):
    """Draws city information - simplified version for debugging."""
    center_x, center_y = hex_to_pixel_func(hex_coord)
    
    # Draw a very basic visualization - a red circle - NO, this is too busy
    # pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), 10)
    
    # Use the default font for simplicity
    new_size = int(font_sizes['small_font'] * zoom * 0.7)
    use_font = pygame.font.Font(None, new_size)
    text = use_font.render(f"{city.name}", True, (0, 0, 0))
    text_rect = text.get_rect(center=(center_x, center_y))
    screen.blit(text, text_rect)
    

def draw_player_tokens(screen: pygame.Surface, hex_coord: HexCoord, 
                      player_colors: List[PlayerColor],
                      hex_to_pixel_func: HexToPixelFunc, zoom: float):
    """Draws multiple player tokens (FLAT TOP position)."""
    center_x, center_y = hex_to_pixel_func(hex_coord)
    radius = (HEX_SIZE / 2.0) * zoom
    if radius < 1: return

    size = radius
    token_radius_pixels = max(1, int(size * 0.18))
    
    # Determine how to space multiple tokens
    num_tokens = len(player_colors)
    if num_tokens == 0:
        return
        
    # Adjust base angle and spacing based on number of tokens
    base_angle_deg = 150  # Original position (between top-left and left)
    angle_spread = min(30, 120 / num_tokens)  # Limit spread to prevent tokens going all around
    
    for i, player_color in enumerate(player_colors):
        # Calculate angle for this token
        offset_angle = base_angle_deg + (i * angle_spread)
        angle_rad = math.pi / 180 * offset_angle
        
        # Calculate position with adjusted offset distance
        offset_dist = size * 0.60
        token_x = center_x + offset_dist * math.cos(angle_rad)
        token_y = center_y + offset_dist * math.sin(angle_rad)

        # Ensure we're using the correct color from the dictionary
        if player_color in PLAYER_COLOR_DICT:
            color = PLAYER_COLOR_DICT[player_color]
        else:
            print(f"Warning: Unknown player color: {player_color}")
            color = GRAY  # Fallback color

        pygame.draw.circle(screen, color, (round(token_x), round(token_y)), token_radius_pixels)
        pygame.draw.circle(screen, BLACK, (round(token_x), round(token_y)), token_radius_pixels, 1)

def draw_trade_posts(screen: pygame.Surface, hex_coord: HexCoord, posts: List[TradePost],
                      hex_to_pixel_func: HexToPixelFunc, zoom: float):
    """Draws multiple trading posts/centers (FLAT TOP position)."""
    center_x, center_y = hex_to_pixel_func(hex_coord)
    radius = (HEX_SIZE / 2.0) * zoom
    if radius < 1: return

    size = radius
    item_base_size = max(1, size * 0.15)
    
    # Base position is at 30 degrees (between top-right and right)
    base_angle_deg = 30
    
    # Adjust spacing based on number of posts
    num_posts = len(posts)
    if num_posts == 0:
        return
        
    # If many posts, increase the spread
    angle_spread = min(20, 60 / num_posts)  # Limit spread to prevent overlap with other elements
    
    # Draw each post with proper spacing
    for i, post in enumerate(posts):
        try:
            # Calculate angle for this post
            offset_angle = base_angle_deg + (i * angle_spread)
            angle_rad = math.pi / 180 * offset_angle
            
            # Calculate position with offset distance
            offset_dist = size * 0.65
            item_x = center_x + offset_dist * math.cos(angle_rad)
            item_y = center_y + offset_dist * math.sin(angle_rad)
            
            color = PLAYER_COLOR_DICT.get(post.owner, GRAY)
            
            if post.type == TradePostType.POST:
                tri_height = item_base_size * 1.5
                tri_base = item_base_size * 1.2
                points = [
                    (item_x, item_y - tri_height / 2),
                    (item_x - tri_base / 2, item_y + tri_height / 2),
                    (item_x + tri_base / 2, item_y + tri_height / 2),
                ]
                pygame.draw.polygon(screen, color, [(round(px), round(py)) for px,py in points])
                pygame.draw.polygon(screen, BLACK, [(round(px), round(py)) for px,py in points], 1)
            elif post.type == TradePostType.CENTER:
                rect_side = item_base_size * 1.2
                rect = pygame.Rect(round(item_x - rect_side / 2), round(item_y - rect_side / 2), 
                                  round(rect_side), round(rect_side))
                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(screen, BLACK, rect, 1)
        except AttributeError:  # Handle case where post might be None temporarily
            pass

def draw_single_meeple(screen: pygame.Surface, x: int, y: int, meeple_color: MeepleColor, radius_pixels: int):
    """Draws a single meeple circle."""
    if radius_pixels < 1: return
    color = MEEPLE_COLOR_DICT.get(meeple_color, WHITE)
    pygame.draw.circle(screen, color, (round(x), round(y)), round(radius_pixels))
    pygame.draw.circle(screen, BLACK, (round(x), round(y)), round(radius_pixels), 1)


def draw_meeple_stack(screen: pygame.Surface, hex_coord: HexCoord, meeples: List[MeepleColor],
                       hex_to_pixel_func: HexToPixelFunc, zoom: float):
    """Draws a stack of meeples (FLAT TOP position)."""
    total_meeples = len(meeples)
    if total_meeples == 0: return

    center_x, center_y = hex_to_pixel_func(hex_coord)
    radius = (HEX_SIZE / 2.0) * zoom
    if radius < 2.5: return # Need some minimum size to draw anything meaningful

    size = radius
    meeple_radius_pixels = max(1, int(size * 0.09))
    meeple_spacing_x = max(2, int(meeple_radius_pixels * 2.2))
    row_spacing_y = max(2, int(meeple_radius_pixels * 2.0))

    # Base position relative to center (near bottom)
    base_row_y = center_y + round(size * 0.8) - round(meeple_radius_pixels * 1.5)

    # Row capacity logic
    row_capacities = [3, 5, 7] # Adjust if needed
    rows_data = []
    remaining = total_meeples
    start_idx = 0
    # Fill rows bottom-up
    count_row1 = min(remaining, row_capacities[0])
    if count_row1 > 0: rows_data.append((count_row1, start_idx)); remaining -= count_row1; start_idx += count_row1
    if remaining > 0:
        count_row2 = min(remaining, row_capacities[1])
        if count_row2 > 0: rows_data.append((count_row2, start_idx)); remaining -= count_row2; start_idx += count_row2
    if remaining > 0:
        # If more rows are needed, extend capacity logic or put all remaining in the last row
        count_row3 = min(remaining, row_capacities[2])
        if count_row3 > 0: rows_data.append((count_row3, start_idx)); remaining -= count_row3; start_idx += count_row3
        # Add any further remaining to the last defined row implicitly or explicitly
        if remaining > 0: # Add remaining to the *last* row calculated
             if rows_data:
                 last_row_count, last_row_start = rows_data[-1]
                 rows_data[-1] = (last_row_count + remaining, last_row_start)
             else: # Should not happen if remaining > 0, but safeguard
                  rows_data.append((remaining, start_idx))
             remaining = 0


    # Draw rows bottom-up
    current_meeple_index = 0
    for i, (num_in_row, _) in enumerate(rows_data):
        row_y = base_row_y - i * row_spacing_y # Stack upwards
        row_width = (num_in_row - 1) * meeple_spacing_x
        start_x = center_x - row_width // 2

        for j in range(num_in_row):
            meeple_x = start_x + j * meeple_spacing_x
            if current_meeple_index < total_meeples:
                 try:
                    draw_single_meeple(screen, round(meeple_x), round(row_y),
                                       meeples[current_meeple_index], meeple_radius_pixels)
                    current_meeple_index += 1
                 except IndexError:
                     print(f"Warning: Meeple index {current_meeple_index} out of bounds for list length {total_meeples}.")
                     break # Stop drawing this row if index is bad
            else: break # Should not be needed if logic above is correct
        if current_meeple_index >= total_meeples: break

def draw_trade_route(screen: pygame.Surface, route: TradeRoute, 
                    hex_to_pixel_func: HexToPixelFunc, zoom: float):
    """Draws a trade route connecting hexes."""
    if len(route.hexes) < 2:
        return
        
    color = PLAYER_COLOR_DICT.get(route.owner, GRAY)
    
    # If route is inactive, use a faded/dashed version
    if not route.active:
        # Faded color for inactive routes
        route_color = (min(color[0] + 30, 255), min(color[1] + 30, 255), min(color[2] + 30, 255), 128)
        line_style = "dashed"
    else:
        # Brighter color for active routes
        route_color = (min(color[0] + 50, 255), min(color[1] + 50, 255), min(color[2] + 50, 255))
        line_style = "solid"
    
    # Draw lines connecting the hexes in order
    for i in range(len(route.hexes) - 1):
        start_hex = route.hexes[i]
        end_hex = route.hexes[i + 1]
        
        start_x, start_y = hex_to_pixel_func(start_hex)
        end_x, end_y = hex_to_pixel_func(end_hex)
        
        # Draw line based on style
        line_width = max(2, int(zoom * 2))
        if line_style == "solid":
            pygame.draw.line(screen, route_color, (start_x, start_y), (end_x, end_y), line_width)
        else:
            # Create a dashed line for inactive routes
            dash_length = max(4, int(zoom * 4))
            gap_length = max(3, int(zoom * 3))
            
            # Calculate line vector and length
            dx, dy = end_x - start_x, end_y - start_y
            line_length = math.sqrt(dx*dx + dy*dy)
            
            # Normalize the vector
            if line_length > 0:
                dx, dy = dx/line_length, dy/line_length
            
            # Draw dashed line
            current_pos = 0
            drawing = True
            while current_pos < line_length:
                segment_length = dash_length if drawing else gap_length
                
                # Calculate start and end points of this segment
                seg_start_x = start_x + dx * current_pos
                seg_start_y = start_y + dy * current_pos
                
                current_pos += segment_length
                current_pos = min(current_pos, line_length)  # Don't go beyond the end
                
                seg_end_x = start_x + dx * current_pos
                seg_end_y = start_y + dy * current_pos
                
                # Draw segment if it's a dash (not a gap)
                if drawing:
                    pygame.draw.line(
                        screen, 
                        route_color, 
                        (round(seg_start_x), round(seg_start_y)), 
                        (round(seg_end_x), round(seg_end_y)), 
                        line_width
                    )
                
                drawing = not drawing  # Toggle between dash and gap
        
    # Draw dots at each point for emphasis
    for hex_coord in route.hexes:
        x, y = hex_to_pixel_func(hex_coord)
        dot_radius = max(3, int(zoom * 3))
        
        # For inactive routes, draw hollow dots
        if not route.active:
            pygame.draw.circle(screen, route_color, (round(x), round(y)), dot_radius, 1)
        else:
            pygame.draw.circle(screen, route_color, (round(x), round(y)), dot_radius)
            
    # Draw route ID or goods info if zoomed in enough
    if zoom > 1.0 and route.active:
        # Find a position to display route info
        if len(route.hexes) > 0:
            # Place info near first hex in route
            info_x, info_y = hex_to_pixel_func(route.hexes[0])
            
            # Draw route ID
            route_id_text = f"Route #{route.id}"
            font_size = max(12, int(12 * zoom))
            font = pygame.font.Font(None, font_size)
            
            # Create text surface with route ID
            text_surface = font.render(route_id_text, True, color)
            text_rect = text_surface.get_rect(center=(info_x, info_y - max(15, int(15 * zoom))))
            
            # Draw with a light background for better visibility
            bg_rect = text_rect.inflate(10, 6)
            pygame.draw.rect(screen, (245, 245, 245, 180), bg_rect)
            pygame.draw.rect(screen, color, bg_rect, 1)
            screen.blit(text_surface, text_rect)

# --- Main Drawing Orchestration ---
# Enhanced draw_board_state function with background map support
def draw_board_state(screen: pygame.Surface, state_cache: GameStateCache, hex_to_pixel_func: HexToPixelFunc, 
                                    zoom: float, fonts: Dict, font_sizes: Dict, highlight_hexes: List[HexCoord], 
                                    selected_start_hex: Optional[HexCoord], show_trade_routes=True, 
                                    camera_x: float = 0, camera_y: float = 0, hex_transparency: int = 128):
    """Enhanced board drawing with background map support."""
    
    # FIRST: Draw background map (if loaded)
    draw_background_map(screen, camera_x, camera_y, zoom, state_cache, hex_to_pixel_func)
    
    # THEN: Draw hexes (now with transparency if background map exists)
    for hex_coord in state_cache.valid_hexes:
        draw_hex_with_transparency(screen, hex_coord, hex_to_pixel_func, zoom, fonts, highlight_hexes, selected_start_hex, hex_transparency)

    # Draw meeples
    for hex_coord, meeples in state_cache.hex_meeples.items():
        if meeples and hex_coord in state_cache.valid_hexes:
            draw_meeple_stack(screen, hex_coord, meeples, hex_to_pixel_func, zoom)

    # Draw trade posts
    for hex_coord, posts in state_cache.trade_posts_locations.items():
        if posts and hex_coord in state_cache.valid_hexes:
            draw_trade_posts(screen, hex_coord, posts, hex_to_pixel_func, zoom)

    # Draw cities
    for city in state_cache.cities:
        if city.location in state_cache.valid_hexes:
            draw_city(screen, city.location, city, hex_to_pixel_func, zoom, fonts, font_sizes)

    # Draw player tokens
    for hex_coord, player_colors in state_cache.player_token_locations.items():
        if player_colors and hex_coord in state_cache.valid_hexes:
            draw_player_tokens(screen, hex_coord, player_colors, hex_to_pixel_func, zoom)

    # Draw trade routes if enabled
    if show_trade_routes and hasattr(state_cache, 'trade_routes') and state_cache.trade_routes:
        for route in state_cache.trade_routes:
            if route and route.hexes:
                draw_trade_route(screen, route, hex_to_pixel_func, zoom)

# def draw_board_state(screen: pygame.Surface, state_cache: GameStateCache, hex_to_pixel_func: HexToPixelFunc, 
#                      zoom: float, fonts: Dict, font_sizes: Dict, highlight_hexes: List[HexCoord], 
#                      selected_start_hex: Optional[HexCoord], show_trade_routes=True, 
#                      camera_x: float = 0, camera_y: float = 0, hex_transparency: int = 128):
#     """
#     Draws the complete board state including background map, hexes, and game elements.
#     """
#     # FIRST: Draw background map (if loaded) - now with board-fitted scaling
#     draw_background_map(screen, camera_x, camera_y, zoom, state_cache, hex_to_pixel_func)
    
#     # THEN: Draw hexes (now with transparency if background map exists)
#     for hex_coord in state_cache.valid_hexes:
#         draw_hex(screen, hex_coord, hex_to_pixel_func, zoom, fonts, highlight_hexes, selected_start_hex, hex_transparency)

#     # Draw meeples (only on valid hexes)
#     for hex_coord, meeples in state_cache.hex_meeples.items():
#         if meeples and hex_coord in state_cache.valid_hexes:
#             draw_meeple_stack(screen, hex_coord, meeples, hex_to_pixel_func, zoom)

#     # Draw trade posts (only on valid hexes)
#     for hex_coord, posts in state_cache.trade_posts_locations.items():
#         if posts and hex_coord in state_cache.valid_hexes:
#             draw_trade_posts(screen, hex_coord, posts, hex_to_pixel_func, zoom)

#     # Draw cities (ALL cities, even if outside valid hexes - for debugging)
#     for city in state_cache.cities:
#         # Remove the valid_hexes check to show all cities
#         draw_city(screen, city.location, city, hex_to_pixel_func, zoom, fonts, font_sizes)
        
#         # Optionally, draw a warning indicator for cities outside valid hexes
#         if city.location not in state_cache.valid_hexes:
#             # Draw a red warning circle around cities outside the valid hex grid
#             center_x, center_y = hex_to_pixel_func(city.location)
#             radius = max(10, int((HEX_SIZE / 2.0) * zoom))
#             pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), radius, 3)

#     # Draw player tokens (only on valid hexes)
#     for hex_coord, player_colors in state_cache.player_token_locations.items():
#         if player_colors and hex_coord in state_cache.valid_hexes:
#             draw_player_tokens(screen, hex_coord, player_colors, hex_to_pixel_func, zoom)

#     # Draw trade routes if enabled
#     if show_trade_routes and hasattr(state_cache, 'trade_routes') and state_cache.trade_routes:
#         for route in state_cache.trade_routes:
#             if route and route.hexes:
#                 draw_trade_route(screen, route, hex_to_pixel_func, zoom)

# --- Functions for drawing background map ---
def load_background_map(map_file_path: str) -> bool:
    """
    Load a background map image (jpg, png, etc.) to display behind the hexes.
    Returns True if successful, False otherwise.
    """
    global BACKGROUND_MAP, BACKGROUND_MAP_RECT
    
    import os
    if not os.path.exists(map_file_path):
        print(f"Warning: Background map file not found: {map_file_path}")
        return False
    
    try:
        # Load the image
        BACKGROUND_MAP = pygame.image.load(map_file_path)
        print(f"Loaded background map: {map_file_path} (size: {BACKGROUND_MAP.get_size()})")
        
        # Store original rect for scaling calculations
        BACKGROUND_MAP_RECT = BACKGROUND_MAP.get_rect()
        return True
        
    except pygame.error as e:
        print(f"Error loading background map {map_file_path}: {e}")
        BACKGROUND_MAP = None
        BACKGROUND_MAP_RECT = None
        return False


def draw_background_map(screen: pygame.Surface, camera_x: float, camera_y: float, zoom: float, 
                       state_cache=None, hex_to_pixel_func=None, scaling_mode: str = "hex_relative"):
    """
    Enhanced background map drawing with proper board-constrained scaling.
    """
    global BACKGROUND_MAP, BACKGROUND_MAP_RECT
    
    if BACKGROUND_MAP is None:
        return
    
    map_zoom_factor = 1.0
    screen_rect = screen.get_rect()
    
    # Calculate the available board area (excluding UI)
    from mali_ba.config import SIDEBAR_WIDTH, CONTROLS_HEIGHT
    board_area_width = screen_rect.width - SIDEBAR_WIDTH
    board_area_height = screen_rect.height - CONTROLS_HEIGHT
    
    # Add padding to ensure map doesn't touch edges
    padding = 0
    available_width = board_area_width - (2 * padding)
    available_height = board_area_height - (2 * padding)
    
    if available_width <= 0 or available_height <= 0:
        return
    
    # Calculate scale based on mode
    if scaling_mode == "fit":
        # Scale to fit within available area (maintains aspect ratio)
        scale_x = available_width / BACKGROUND_MAP_RECT.width
        scale_y = available_height / BACKGROUND_MAP_RECT.height
        fit_scale = min(scale_x, scale_y) * map_zoom_factor
        
    elif scaling_mode == "fill":
        # Scale to fill available area completely (may crop edges)
        scale_x = available_width / BACKGROUND_MAP_RECT.width
        scale_y = available_height / BACKGROUND_MAP_RECT.height
        fit_scale = max(scale_x, scale_y) * map_zoom_factor
        
    elif scaling_mode == "hex_relative" and state_cache and hex_to_pixel_func:
        # Scale relative to the hex grid bounds with zoom responsiveness
        if state_cache.valid_hexes:
            min_x, min_y, max_x, max_y = calculate_hex_grid_bounds(state_cache.valid_hexes, hex_to_pixel_func)
            grid_width = max_x - min_x
            grid_height = max_y - min_y
            
            if grid_width > 0 and grid_height > 0:
                # Calculate what scale would make the hex grid fit nicely
                hex_scale_x = available_width / grid_width
                hex_scale_y = available_height / grid_height
                hex_fit_scale = min(hex_scale_x, hex_scale_y) * 0.8  # 0.8 for margin around hex grid
                
                # But also ensure the map itself doesn't exceed the board area
                map_scale_x = available_width / BACKGROUND_MAP_RECT.width
                map_scale_y = available_height / BACKGROUND_MAP_RECT.height
                map_fit_scale = min(map_scale_x, map_scale_y)
                
                # Use the smaller of the two scales to ensure map stays within bounds
                fit_scale = min(hex_fit_scale, map_fit_scale) * map_zoom_factor
            else:
                # Fallback: fit to screen
                scale_x = available_width / BACKGROUND_MAP_RECT.width
                scale_y = available_height / BACKGROUND_MAP_RECT.height
                fit_scale = min(scale_x, scale_y) * map_zoom_factor
        else:
            # Fallback: fit to screen
            scale_x = available_width / BACKGROUND_MAP_RECT.width
            scale_y = available_height / BACKGROUND_MAP_RECT.height
            fit_scale = min(scale_x, scale_y) * map_zoom_factor
            
    elif scaling_mode == "fixed":
        # Use fixed scale factor, but constrain to board area
        base_scale_x = available_width / BACKGROUND_MAP_RECT.width
        base_scale_y = available_height / BACKGROUND_MAP_RECT.height
        max_allowed_scale = min(base_scale_x, base_scale_y)
        
        desired_scale = map_zoom_factor * zoom
        fit_scale = min(desired_scale, max_allowed_scale)
        
    else:
        # Default fallback: fit to screen
        scale_x = available_width / BACKGROUND_MAP_RECT.width
        scale_y = available_height / BACKGROUND_MAP_RECT.height
        fit_scale = min(scale_x, scale_y) * map_zoom_factor
    
    # Ensure minimum scale
    fit_scale = max(0.1, fit_scale)
    
    # Apply the calculated scale
    scaled_width = int(BACKGROUND_MAP_RECT.width * fit_scale)
    scaled_height = int(BACKGROUND_MAP_RECT.height * fit_scale)
    
    # Double-check that scaled dimensions don't exceed board area
    if scaled_width > board_area_width:
        scale_factor = board_area_width / scaled_width
        scaled_width = int(scaled_width * scale_factor)
        scaled_height = int(scaled_height * scale_factor)
    
    if scaled_height > board_area_height:
        scale_factor = board_area_height / scaled_height
        scaled_width = int(scaled_width * scale_factor)
        scaled_height = int(scaled_height * scale_factor)
    
    # Prevent scaling to zero or negative sizes
    scaled_width = max(1, scaled_width)
    scaled_height = max(1, scaled_height)
    
    # Scale the background map
    scaled_map = pygame.transform.scale(BACKGROUND_MAP, (scaled_width, scaled_height))
    
    # Position the map in the center of the board area
    board_area_center_x = board_area_width // 2
    board_area_center_y = board_area_height // 2
    
    # Apply camera offset (but don't let camera move map outside board area)
    camera_scale_factor = 1.0
    
    map_x = board_area_center_x - scaled_width // 2 - int(camera_x * camera_scale_factor)
    map_y = board_area_center_y - scaled_height // 2 - int(camera_y * camera_scale_factor)
    
    # Clamp map position to ensure it doesn't go outside board area
    map_x = max(0, min(map_x, board_area_width - scaled_width))
    map_y = max(0, min(map_y, board_area_height - scaled_height))
    
    # Create a clipping rectangle for the board area
    board_rect = pygame.Rect(0, 0, board_area_width, board_area_height)
    
    # Draw the scaled and positioned map, clipped to board area
    screen.set_clip(board_rect)
    screen.blit(scaled_map, (map_x, map_y))
    screen.set_clip(None)  # Remove clipping
    
    # Optional: Print debug info
    # print(f"DEBUG: Board area={board_area_width}x{board_area_height}, Map scale={fit_scale:.3f}, "
    #       f"size={scaled_width}x{scaled_height}, pos=({map_x},{map_y})")


# Also update the calculate_hex_grid_bounds function to be zoom-aware:

def calculate_hex_grid_bounds(valid_hexes: Set[HexCoord], hex_to_pixel_func: HexToPixelFunc) -> Tuple[int, int, int, int]:
    """
    Calculate the bounding box of the hex grid in pixel coordinates.
    Returns (min_x, min_y, max_x, max_y) of the hex grid.
    Note: hex_to_pixel_func already includes current zoom scaling.
    """
    if not valid_hexes:
        return (0, 0, 0, 0)
    
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    # Get actual hex positions using the current zoom level (already included in hex_to_pixel_func)
    for hex_coord in valid_hexes:
        center_x, center_y = hex_to_pixel_func(hex_coord)
        
        # Since hex_to_pixel_func includes zoom, we just need to account for hex extent
        # We can't access zoom directly here, so we'll use the coordinate spread
        # as an indicator of current scale
        min_x = min(min_x, center_x)
        max_x = max(max_x, center_x)
        min_y = min(min_y, center_y)
        max_y = max(max_y, center_y)
    
    # Add some padding based on the coordinate spread
    width = max_x - min_x
    height = max_y - min_y
    
    # Use a percentage-based padding that scales with the grid size
    padding_x = max(50, width * 0.1)  # At least 50px or 10% of width
    padding_y = max(50, height * 0.1)  # At least 50px or 10% of height
    
    return (int(min_x - padding_x), int(min_y - padding_y), 
            int(max_x + padding_x), int(max_y + padding_y))


# a method to the visualizer class to change scaling modes
def set_background_map_mode(self, mode="fit", zoom_factor=1.0):
    """
    Set the background map scaling mode.
    
    Args:
        mode: "fit", "fill", "hex_relative", or "fixed"
        zoom_factor: Additional scaling factor
    """
    self.bg_map_scaling_mode = mode
    self.bg_map_zoom_factor = zoom_factor
    print(f"Background map mode set to: {mode} (factor: {zoom_factor})")


def draw_hex_with_transparency(screen: pygame.Surface, hex_coord: HexCoord, hex_to_pixel_func: HexToPixelFunc,
                              zoom: float, fonts: Dict, highlight_hexes: List[HexCoord], 
                              selected_start_hex: Optional[HexCoord], hex_transparency: int = 128):
    """Enhanced hex drawing with transparency support for background maps."""
    center_x, center_y = hex_to_pixel_func(hex_coord)
    radius = (HEX_SIZE / 2.0) * zoom
    if radius < 1: return

    size = radius
    points = []
    for i in range(6):
        angle_deg = 60 * i
        angle_rad = math.pi / 180 * angle_deg
        x = center_x + size * math.cos(angle_rad)
        y = center_y + size * math.sin(angle_rad)
        points.append((round(x), round(y)))

    # Determine base colors
    fill_color = LIGHT_GRAY
    is_selected_start = hex_coord == selected_start_hex
    is_highlighted = hex_coord in highlight_hexes

    if is_selected_start: 
        fill_color = YELLOW
    elif is_highlighted: 
        fill_color = (200, 200, 0) # Highlight color

    # If we have a background map, make hexes semi-transparent
    if BACKGROUND_MAP is not None:
        # Create a surface with per-pixel alpha for transparency
        hex_surface = pygame.Surface((int(size * 2), int(size * 2)), pygame.SRCALPHA)
        
        # Adjust points for the temporary surface (center them)
        temp_points = [(x - center_x + size, y - center_y + size) for x, y in points]
        
        # Make the fill color semi-transparent using the configured transparency
        transparent_fill = (*fill_color[:3], hex_transparency)  # RGBA with configured alpha
        
        # Draw on the temporary surface
        pygame.draw.polygon(hex_surface, transparent_fill, temp_points)
        
        # Draw border (fully opaque)
        line_width = 3 if is_selected_start else (2 if is_highlighted else 1)
        border_color = (*DARK_GRAY[:3], 255) if len(DARK_GRAY) >= 3 else (64, 64, 64, 255)
        pygame.draw.polygon(hex_surface, border_color, temp_points, line_width)
        
        # Blit the transparent hex to the main screen
        screen.blit(hex_surface, (center_x - size, center_y - size))
    else:
        # No background map - draw normally (opaque)
        pygame.draw.polygon(screen, fill_color, points)
        line_width = 3 if is_selected_start else (2 if is_highlighted else 1)
        pygame.draw.polygon(screen, DARK_GRAY, points, line_width)

    # Draw coordinates if zoomed in enough
    if zoom > 1.0:
         coord_font = fonts['small_font']
         try:
             coord_text = coord_font.render(f"{hex_coord.x},{hex_coord.y},{hex_coord.z}", True, DARK_GRAY)
             text_rect = coord_text.get_rect(center=(center_x, center_y - round(size*0.7)))
             screen.blit(coord_text, text_rect)
         except AttributeError: # Handle case where hex_coord might be None temporarily
             pass

# --- END OF FILE visualizer_drawing.py ---