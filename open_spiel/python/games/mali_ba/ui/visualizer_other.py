# --- START OF FILE visualizer_other.py ---
import sys
sys.path.append("/media/robp/UD/Projects/mali_ba/open_spiel/python/games") # allow debugging in vs code
import json
from typing import List
from mali_ba.config import PlayerColor, MeepleColor, TradePostType, Phase
from mali_ba.classes.game_state import GameStateCache
from mali_ba.classes.classes_other import TradePost, City, HexCoord, TradeRoute

# --- Client-Side Validation Helpers ---
# These functions read from the state cache to provide immediate feedback.
def can_start_mancala_at(hex_coord: HexCoord, player_color: PlayerColor, cache: GameStateCache) -> bool:
    """Checks if a player can start a mancala move from a hex."""
    token_owners = cache.player_token_locations.get(hex_coord, [])
    return player_color in token_owners

def is_valid_mancala_step(current_path: List[HexCoord], next_hex: HexCoord) -> bool:
    """Checks if the next step in a mancala path is valid."""
    if not current_path: return False
    last_hex = current_path[-1]
    is_adjacent = next_hex.distance(last_hex) == 1
    not_in_path = next_hex not in current_path
    return is_adjacent and not_in_path

def can_select_for_upgrade(hex_coord: HexCoord, player_color: PlayerColor, cache: GameStateCache) -> bool:
    """Checks if a player can select a hex to upgrade a post."""
    posts = cache.trade_posts_locations.get(hex_coord, [])
    for post in posts:
        if post.owner == player_color and post.type == TradePostType.POST:
            return True
    return False

def can_add_to_trade_route(hex_coord: HexCoord, player_color: PlayerColor, cache: GameStateCache) -> bool:
    """Checks if a player can add a specific hex to a trade route."""
    posts_at_hex = cache.trade_posts_locations.get(hex_coord, [])
    player_post_at_hex = None
    for post in posts_at_hex:
        if post.owner == player_color:
            player_post_at_hex = post
            break

    # Player must have a post or center at the hex
    if not player_post_at_hex:
        return False

    # Check if the hex is a city
    is_city = any(city.location == hex_coord for city in cache.cities)

    # If it's a city, any type of post is valid
    if is_city:
        return True

    # If it's NOT a city, it MUST be a trading center
    if not is_city and player_post_at_hex.type == TradePostType.CENTER:
        return True
        
    # All other cases are invalid
    return False


# --- State Parsing (Simplified) ---
def parse_and_update_state_from_json(state_str: str, cache: GameStateCache) -> bool:
    """
    Parses the authoritative C++ JSON state string and completely updates the cache.
    This is now the ONLY way the GameStateCache is modified.
    
    Args:
        state_str: The JSON string from C++'s `serialize()`.
        cache: The GameStateCache object to update.
        
    Returns:
        True if parsing was successful, False otherwise.
    """
    # print(f"\n--- DEBUG: Parsing New State JSON ---\n{state_str[:300]}...\n--------------------------")
    
    try:
        data = json.loads(state_str)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON received: {e}")
        return False

    try:
        # Clear all dynamic content from the cache.
        # Board structure (valid_hexes, cities, grid_radius) is static and not cleared.
        cache.player_token_locations.clear()
        cache.hex_meeples.clear()
        cache.trade_posts_locations.clear()
        cache.trade_routes.clear()
        
        num_players = len(cache.game_player_colors)
        cache.common_goods = [{} for _ in range(num_players)]
        cache.rare_goods = [{} for _ in range(num_players)]

        # Parse basic game state
        cache.current_player_id = data.get("currentPlayerId", -1)
        cache.current_phase = Phase.from_int(data.get("currentPhase", -1))
        cache.is_terminal = cache.current_player_id == -2 # pyspiel.kTerminalPlayerId
        
        if 0 <= cache.current_player_id < num_players:
            cache.current_player_color = cache.game_player_colors[cache.current_player_id]
        else:
            cache.current_player_color = PlayerColor.EMPTY

        # Player Post Supply
        cache.player_posts_supply = data.get("playerPostsSupply", [6] * num_players)

        # Player Tokens
        for hex_str, p_ids in data.get("playerTokens", {}).items():
            hex_coord = HexCoord.from_string(hex_str)
            if hex_coord and hex_coord in cache.valid_hexes:
                cache.player_token_locations[hex_coord] = [PlayerColor.from_int(pid) for pid in p_ids]

        # Hex Meeples
        for hex_str, m_ids in data.get("hexMeeples", {}).items():
            hex_coord = HexCoord.from_string(hex_str)
            if hex_coord and hex_coord in cache.valid_hexes:
                cache.hex_meeples[hex_coord] = [MeepleColor.from_int(mid) for mid in m_ids]

        # Trade Posts
        for hex_str, posts_json in data.get("tradePosts", {}).items():
            hex_coord = HexCoord.from_string(hex_str)
            if hex_coord and hex_coord in cache.valid_hexes:
                posts_obj = [TradePost(PlayerColor.from_int(p["owner"]), TradePostType.from_int(p["type"])) for p in posts_json]
                if posts_obj:
                    cache.trade_posts_locations[hex_coord] = posts_obj

        # Goods
        cache.common_goods = data.get("commonGoods", [{} for _ in range(num_players)])
        cache.rare_goods = data.get("rareGoods", [{} for _ in range(num_players)])

        # Trade Routes
        for route_data in data.get("tradeRoutes", []):
            owner = PlayerColor.from_int(route_data.get("owner", -1))
            hexes = [HexCoord.from_string(s) for s in route_data.get("hexes", [])]
            if owner != PlayerColor.EMPTY and all(hexes):
                route = TradeRoute(route_data["id"], owner, hexes, route_data.get("goods", {}))
                route.active = route_data.get("active", False)
                cache.trade_routes.append(route)
        
        print(f"✅ State cache successfully updated. Player: {cache.current_player_id}, Phase: {cache.current_phase.name}")
        return True

    except (KeyError, TypeError, ValueError, AttributeError) as e:
        print(f"❌ Error parsing state JSON content: {e}")
        import traceback
        traceback.print_exc()
        return False

# def _parse_cities_from_json(json_cities: list) -> List[City]:
#     """Parse cities from JSON array."""
#     cities = []
#     for city_data in json_cities:
#         try:
#             city_id = city_data.get("id", 0)
#             name = city_data.get("name", "Unknown")
#             culture = city_data.get("culture", "Unknown")
#             common_good = city_data.get("commonGood", "Unknown")
#             rare_good = city_data.get("rareGood", "Unknown")
            
#             # Handle location
#             location_data = city_data.get("location", "0,0,0")
#             if isinstance(location_data, str):
#                 parts = location_data.split(",")
#                 if len(parts) == 3:
#                     x, y, z = map(int, parts)
#                     location = HexCoord(x, y, z)
#                 else:
#                     continue
#             elif isinstance(location_data, dict):
#                 x = location_data.get("x", 0)
#                 y = location_data.get("y", 0)
#                 z = location_data.get("z", 0)
#                 location = HexCoord(x, y, z)
#             else:
#                 continue
                
#             city = City(city_id, name, culture, location, common_good, rare_good)
#             cities.append(city)
            
#         except Exception as e:
#             print(f"Error parsing city {city_data}: {e}")
            
#     return cities
# --- END OF parsing ---

class BoardVisualizerHelpers:
    """Helper methods for BoardVisualizer to handle dialogs, trade routes, and other complex interactions."""
    
    def __init__(self, visualizer):
        """Store a reference to the main visualizer."""
        self.visualizer = visualizer
    
    # --- Dialog Methods ---
    
    def show_dialog(self, dialog_type, message, options=None, context_data=None, layout="auto"):
        """Show a dialog of the specified type with custom context data and layout."""
        title = self.get_dialog_title(dialog_type)
        default_options = self.get_dialog_options(dialog_type)
        
        # Use provided options or fall back to defaults
        final_options = options if options is not None else default_options
        
        # Determine layout - use vertical for upgrade payments due to long text
        if dialog_type == "choose_upgrade_payment":
            layout = "vertical"
        elif layout == "auto":
            # Auto-detect based on text length
            max_option_length = max(len(option) for option in final_options) if final_options else 0
            layout = "vertical" if max_option_length > 15 else "horizontal"
        
        self.visualizer.dialog_box.show(
            title=title,
            message=message,
            options=final_options,
            dialog_type=dialog_type,
            context_data=context_data,
            layout=layout
        )

    def get_dialog_title(self, dialog_type):
        """Get the appropriate title for the dialog type."""
        titles = {
            "mancala_post": "Place Trading Post?",
            "choose_mancala_payment": "Choose Payment Method",
            "upgrade_resource": "Choose Resource Type",
            "choose_upgrade_payment": "Choose Payment Method",
            "empty_hex_post": "Place Trading Post on Empty Hex?",
            "choose_income_city": "Choose Income Source",
            "choose_specific_resource": "Choose Resource",
            "trade_route_options": "Trade Route Options",
            "trade_route_selection": "Select Trade Route"
        }
        return titles.get(dialog_type, "Question")

    def get_dialog_options(self, dialog_type):
        """Get the appropriate options for the dialog type."""
        options = {
            "mancala_post": ["Yes", "No"],
            "choose_mancala_payment": [],  # Will be populated dynamically
            "upgrade_resource": ["Use Common (3)", "Use Rare (1)", "Cancel"],
            "empty_hex_post": ["Yes (1 Common)", "Yes (1 Rare)", "No"],
            "choose_income_city": [],  # Will be populated with city names
            "choose_specific_resource": []  # Will be populated with resource names
        }
        return options.get(dialog_type, ["Yes", "No", "Cancel"])

    def handle_dialog_result(self, result):
        """Handle the result from a dialog box interaction."""
        dialog_type = self.visualizer.dialog_box.dialog_type
        context_data = self.visualizer.dialog_box.context_data
        
        # Handle different dialog types
        if dialog_type == "mancala_post":
            if result == "Yes":
                # Player wants to place a trading post
                # Check if we need to show payment selection dialog
                last_hex = self.visualizer.highlight_hexes[-1]
                has_meeples_at_dest = last_hex in self.visualizer.state_cache.hex_meeples and self.visualizer.state_cache.hex_meeples.get(last_hex, [])
                
                if has_meeples_at_dest:
                    # Can place using meeples - no payment dialog needed
                    self.visualizer.wants_post_on_mancala = True
                    self.visualizer.selected_mancala_payment = "meeple"
                    self.visualizer.control_panel.update_status("Trading post will be placed using meeple. Press Submit to confirm.")
                    # Continue the submission process
                    self.visualizer.submit_move()
                else:
                    # Need to show payment dialog for resources
                    payment_options = self.get_mancala_payment_options()
                    if len(payment_options) == 0:
                        # No resources available
                        self.visualizer.wants_post_on_mancala = False
                        self.visualizer.control_panel.update_status("Cannot place trading post: No resources available.")
                        self.visualizer.submit_move()
                    elif len(payment_options) == 1:
                        # Only one option available - auto-select
                        self.visualizer.selected_mancala_payment = payment_options[0].split(" (")[0]  # Extract "good:count" part
                        self.visualizer.wants_post_on_mancala = True
                        self.visualizer.control_panel.update_status(f"Trading post will be placed using {payment_options[0]}. Press Submit to confirm.")
                        self.visualizer.submit_move()
                    else:
                        # Multiple options - show payment selection dialog
                        self.show_mancala_payment_dialog(last_hex, payment_options)
            else:  # "No" or "Cancel"
                self.visualizer.wants_post_on_mancala = False
                self.visualizer.control_panel.update_status("No trading post will be placed. Press Submit to confirm.")
                # Continue the submission process
                self.visualizer.submit_move()

        elif dialog_type == "choose_mancala_payment":
            if result == "Cancel":
                self.visualizer.cancel_input_mode()
                return
            
            # Store the selected payment (extract just the "good:count" part)
            self.visualizer.selected_mancala_payment = result.split(" (")[0]  # Extract payment part
            self.visualizer.wants_post_on_mancala = True
            self.visualizer.control_panel.update_status(f"Trading post will be placed using {result}. Press Submit to confirm.")
            
            # Continue the submission process
            self.visualizer.submit_move()


        elif dialog_type == "choose_upgrade_payment":  # Add this new case
            if result == "Cancel":
                self.visualizer.cancel_input_mode()
                return
            
            hex_coord = context_data.get("hex_coord")
            
            # Parse the selected payment option
            # Format: "good_name:count,good_name:count (description)" or "good_name:count (description)"
            if " (" in result:
                payment_part = result.split(" (")[0]  # Get payment part before description
            else:
                payment_part = result
            
            # Store the selected payment
            self.visualizer.selected_upgrade_payment = payment_part
            self.visualizer.control_panel.update_status(f"Will upgrade post at {hex_coord} using {payment_part}. Press Submit to confirm.")


        elif dialog_type == "upgrade_resource":
            hex_coord = context_data.get("hex_coord")
            if result == "Use Common (3)":
                self.visualizer.selected_resource_type = "common"
                self.visualizer.control_panel.update_status(f"Upgrading post at {hex_coord} using 3 common goods. Press Submit to confirm.")
                self.visualizer.highlight_hexes = [hex_coord]
            elif result == "Use Rare (1)":
                self.visualizer.selected_resource_type = "rare"
                self.visualizer.control_panel.update_status(f"Upgrading post at {hex_coord} using 1 rare good. Press Submit to confirm.")
                self.visualizer.highlight_hexes = [hex_coord]
            else:  # "Cancel"
                self.visualizer.cancel_input_mode()
                return
                
            # Continue with submission if not canceled
            if result != "Cancel":
                self.visualizer.submit_move()
                
        elif dialog_type == "empty_hex_post":
            hex_coord = context_data.get("hex_coord")
            if result == "Yes (1 Common)":
                # Set the resource type and continue with place post
                self.visualizer.empty_hex_post_resource = "common"
                self.visualizer.control_panel.update_status(f"Will place post at {hex_coord} using 1 common good. Press Submit to confirm.")
                self.visualizer.highlight_hexes = [hex_coord]
            elif result == "Yes (1 Rare)":
                self.visualizer.empty_hex_post_resource = "rare"
                self.visualizer.control_panel.update_status(f"Will place post at {hex_coord} using 1 rare good. Press Submit to confirm.")
                self.visualizer.highlight_hexes = [hex_coord]
            else:  # "No"
                self.visualizer.empty_hex_post_resource = None
                self.visualizer.cancel_input_mode()
                return
                
        elif dialog_type == "choose_income_city":
            self.handle_income_city_selection(result, context_data)

        elif dialog_type == "choose_income_option":
            if result == "Cancel":
                return
            
            # Find the corresponding action string
            display_options = context_data.get("display_options", [])
            income_actions = context_data.get("income_actions", [])
            
            try:
                # Find which option was selected
                selected_index = display_options.index(result)
                if 0 <= selected_index < len(income_actions):
                    action_string = income_actions[selected_index]
                    self._display_income_summary(action_string)
                    self.visualizer.attempt_apply_action(action_string)
                else:
                    self.visualizer.control_panel.update_status("Invalid income selection.")
            except ValueError:
                self.visualizer.control_panel.update_status("Invalid income selection.")

        elif dialog_type == "choose_specific_resource":
            selected_resource = result
            resource_type = context_data.get("resource_type", "common")
            action_type = context_data.get("action_type", "")
            
            # Store the selected resource
            self.visualizer.selected_specific_resource = selected_resource
            self.visualizer.control_panel.update_status(f"Selected {selected_resource} resource. Press Submit to confirm.")
            
            # Continue with the specific action that needed resource selection
            if action_type == "upgrade":
                self.visualizer.submit_move()
            elif action_type == "place_post":
                self.process_post_placement()
                
        elif dialog_type == "trade_route_options":
            self.handle_trade_route_options_result(result, context_data)
            
        elif dialog_type == "trade_route_selection":
            self.handle_trade_route_selection_result(result, context_data)
            
        else:
            errmsg = f"ERROR: unknown dialog box {dialog_type}. Result: {result}"
            print(errmsg)
            self.visualizer.control_panel.update_status(errmsg)

    def handle_income_city_selection(self, result, context_data):
        """Handle selection of a city for income collection."""
        selected_city_name = result
        post_hex = context_data.get("post_hex")
        cities = context_data.get("cities", [])
        
        # Find the selected city object
        selected_city = next((city for city in cities if city.name == selected_city_name), None)
        
        if selected_city:
            # Add this city's common good to the collection
            if selected_city.common_good not in self.visualizer.collected_common_goods:
                self.visualizer.collected_common_goods[selected_city.common_good] = 0
            self.visualizer.collected_common_goods[selected_city.common_good] += 1
            
            # Add to income sources for UI feedback
            self.visualizer.income_sources["trading_posts"].append((post_hex, selected_city, selected_city.common_good))
            
            # Move to next post needing selection or submit if done
            self.visualizer.current_post_selection_index += 1
            if self.visualizer.current_post_selection_index < len(self.visualizer.posts_needing_selection):
                # Show dialog for the next post
                next_post_hex, next_cities = self.visualizer.posts_needing_selection[self.visualizer.current_post_selection_index]
                self.show_dialog(
                    dialog_type="choose_income_city",
                    message=f"Choose which city to collect goods from\nfor trading post at {next_post_hex}:",
                    options=[city.name for city in next_cities],
                    context_data={"post_hex": next_post_hex, "cities": next_cities}
                )
            else:
                # All selections made, submit the income action
                self.submit_income_action(
                    self.visualizer.collected_common_goods,
                    self.visualizer.collected_rare_goods,
                    self.visualizer.income_sources
                )
                
                # Clean up temporary state
                self.visualizer.posts_needing_selection = []
                self.visualizer.current_post_selection_index = 0
                self.visualizer.collected_common_goods = {}
                self.visualizer.collected_rare_goods = {}
                self.visualizer.income_sources = {}
        else:
            self.visualizer.control_panel.update_status("Invalid city selection. Please try again.")

    def handle_trade_route_options_result(self, result, context_data):
        """Handle selection from trade route options dialog."""
        route = context_data.get("route")
        if not route:
            self.visualizer.control_panel.update_status("Error: No route selected.")
            return
            
        if result.startswith("Edit Route"):
            # Start editing the existing route
            self.visualizer.is_input_mode = True
            self.visualizer.input_mode_type = "trade_route"
            self.visualizer.highlight_hexes = route.hexes.copy()
            self.visualizer.selected_trade_route = route
            self.visualizer.control_panel.update_status(f"Editing Trade Route #{route.id}. Add/remove hexes and Submit when done.")
        elif result.startswith("Delete Route"):
            # Confirm before deleting
            self.show_dialog(
                dialog_type="confirm_delete_route",
                message=f"Are you sure you want to delete Trade Route #{route.id}?",
                options=["Yes", "No"],
                context_data={"route_id": route.id}
            )
        # "Cancel" option handled automatically by dialog closing
    
    def handle_trade_route_selection_result(self, result, context_data):
        """Handle selection from multiple trade routes dialog."""
        if result == "Cancel":
            return
            
        routes = context_data.get("routes", [])
        if not routes:
            return
            
        # Parse the route number from the result string
        # Format: "Route #X (Y hexes)"
        route_id = None
        for route in routes:
            if f"Route #{route.id}" in result:
                route_id = route.id
                break
                
        if route_id is not None:
            self.show_trade_route_options(route_id)
        else:
            self.visualizer.control_panel.update_status("Error: Could not determine selected route.")

    def handle_empty_hex_post_placement(self, hex_coord):
        """Handle placing a trading post on an empty hex (no meeples)."""
        player_id = self.visualizer.state_cache.current_player_id
        player_color = self.visualizer.state_cache.current_player_color
        
        # Check resources
        common_goods_count = 0
        rare_goods_count = 0
        
        if 0 <= player_id < len(self.visualizer.state_cache.common_goods):
            common_goods_count = sum(self.visualizer.state_cache.common_goods[player_id].values())
        
        if 0 <= player_id < len(self.visualizer.state_cache.rare_goods):
            rare_goods_count = sum(self.visualizer.state_cache.rare_goods[player_id].values())
        
        # Check if player has enough resources
        has_common = common_goods_count >= 1
        has_rare = rare_goods_count >= 1
        
        if has_common or has_rare:
            # Configure dialog options based on available resources
            options = []
            if has_common:
                options.append("Yes (1 Common)")
            if has_rare:
                options.append("Yes (1 Rare)")
            options.append("No")
            
            self.show_dialog(
                dialog_type="empty_hex_post",
                message=f"There are no meeples on this hex. Do you want to pay one resource to place a trading post at {hex_coord}?",
                options=options,
                context_data={"hex_coord": hex_coord}
            )
        else:
            self.visualizer.control_panel.update_status(
                "Cannot place trading post: You need at least 1 common or 1 rare good."
            )

    def handle_upgrade_payment_selection(self, result, context_data):
        """Handle selection of payment method for upgrade."""
        if result == "Cancel":
            self.visualizer.cancel_input_mode()
            return
        
        hex_coord = context_data.get("hex_coord")
        
        # Parse the selected payment option
        # Format: "good_name:count (description)"
        if " (" in result:
            payment_part = result.split(" (")[0]  # Get "good_name:count" part
        else:
            payment_part = result
        
        # Store the selected payment
        self.visualizer.selected_upgrade_payment = payment_part
        self.visualizer.control_panel.update_status(f"Will upgrade post at {hex_coord} using {payment_part}. Press Submit to confirm.")


    def can_place_mancala_post(self, hex_coord, has_meeples):
        """Check if player can place a trading post via mancala at the given hex."""
        player_id = self.visualizer.state_cache.current_player_id
        
        if has_meeples:
            return True  # Can always place if meeples available
        
        # Check if player has any resources
        common_goods_count = 0
        rare_goods_count = 0
        
        if 0 <= player_id < len(self.visualizer.state_cache.common_goods):
            common_goods_count = sum(self.visualizer.state_cache.common_goods[player_id].values())
        
        if 0 <= player_id < len(self.visualizer.state_cache.rare_goods):
            rare_goods_count = sum(self.visualizer.state_cache.rare_goods[player_id].values())
        
        return common_goods_count > 0 or rare_goods_count > 0

    def get_mancala_payment_options(self):
        """Get available payment options for mancala trading post placement."""
        player_id = self.visualizer.state_cache.current_player_id
        options = []
        
        # Get player's resources
        common_goods = self.visualizer.state_cache.common_goods[player_id] if player_id < len(self.visualizer.state_cache.common_goods) else {}
        rare_goods = self.visualizer.state_cache.rare_goods[player_id] if player_id < len(self.visualizer.state_cache.rare_goods) else {}
        
        # Add common goods options (1 of each available)
        for good_name, count in common_goods.items():
            if count >= 1:
                options.append(f"{good_name}:1 (1 Common)")
        
        # Add rare goods options (1 of each available)
        for good_name, count in rare_goods.items():
            if count >= 1:
                options.append(f"{good_name}:1 (1 Rare)")
        
        return options

    def show_mancala_payment_dialog(self, hex_coord, payment_options):
        """Show dialog for selecting mancala trading post payment method."""
        self.show_dialog(
            dialog_type="choose_mancala_payment",
            message=f"Choose resource to pay for trading post at {hex_coord}:",
            options=payment_options + ["Cancel"],
            context_data={"hex_coord": hex_coord, "payment_options": payment_options}
        )

    # --- Income Methods ---

    def process_income_collection(self):
        """Process the income collection action for the current player."""
        player_id = self.visualizer.state_cache.current_player_id
        player_color = self.visualizer.state_cache.current_player_color
        
        if player_id < 0 or player_color == PlayerColor.EMPTY:
            self.visualizer.control_panel.update_status("Error: Invalid player for income collection.")
            return
        
        # Get legal actions from the game interface
        try:
            legal_actions = self.visualizer.game_interface.spiel_state.legal_actions()
            
            # Find income actions
            income_actions = []
            for action_id in legal_actions:
                action_str = self.visualizer.game_interface.spiel_state.action_to_string(action_id)
                if action_str.startswith("income "):
                    income_actions.append(action_str)
            
            if len(income_actions) == 0:
                self.visualizer.control_panel.update_status("No income actions available.")
                return
            elif len(income_actions) == 1:
                # Single income option - use it directly
                action_string = income_actions[0]
                self._display_income_summary(action_string)
                self.visualizer.attempt_apply_action(action_string)
            else:
                # Multiple income options - show dialog to choose
                # Create user-friendly descriptions for each option
                display_options = []
                for action in income_actions:
                    display_text = self._create_income_display_text(action)
                    display_options.append(display_text)
                
                self.show_dialog(
                    dialog_type="choose_income_option",
                    message="Multiple income options available.\nChoose which resources to collect:",
                    options=display_options + ["Cancel"],
                    context_data={"income_actions": income_actions, "display_options": display_options},
                    layout="vertical"  # Use vertical layout for potentially long action strings
                )
                
        except Exception as e:
            print(f"Error getting legal income actions: {e}")
            self.visualizer.control_panel.update_status("Error: Could not determine available income options.")
        # """Process the income collection action for the current player."""
        # player_id = self.visualizer.state_cache.current_player_id
        # player_color = self.visualizer.state_cache.current_player_color
        
        # if player_id < 0 or player_color == PlayerColor.EMPTY:
        #     self.visualizer.control_panel.update_status("Error: Invalid player for income collection.")
        #     return
        
        # # Track income sources for UI feedback
        # income_sources = {
        #     "city_centers": [],  # (city, rare_good)
        #     "connected_centers": [],  # (center_hex, city, common_good)
        #     "trading_posts": []  # (post_hex, city, common_good)
        # }
        
        # # Track goods to be collected
        # rare_goods = {}
        # common_goods = {}
        
        # # Track trading posts that need city selection
        # posts_needing_selection = []
        
        # # 1. Process cities with player's trading centers (rare goods)
        # for hex_coord, posts in self.visualizer.state_cache.trade_posts_locations.items():
        #     for post in posts:
        #         if post.owner == player_color and post.type == TradePostType.CENTER:
        #             # Check if this is in a city
        #             city = next((city for city in self.visualizer.state_cache.cities if city.location == hex_coord), None)
        #             if city:
        #                 # 1. Rare good from city with trading center
        #                 if city.rare_good not in rare_goods:
        #                     rare_goods[city.rare_good] = 0
        #                 rare_goods[city.rare_good] += 1
        #                 income_sources["city_centers"].append((city, city.rare_good))
                        
        #             # 2. Check if trading center is adjacent to any cities
        #             for city in self.visualizer.state_cache.cities:
        #                 if hex_coord.distance(city.location) == 1:
        #                     # Common good from connected city
        #                     if city.common_good not in common_goods:
        #                         common_goods[city.common_good] = 0
        #                     common_goods[city.common_good] += 1
        #                     income_sources["connected_centers"].append((hex_coord, city, city.common_good))
        
        # # 3. Process trading posts (common goods from closest city)
        # for hex_coord, posts in self.visualizer.state_cache.trade_posts_locations.items():
        #     for post in posts:
        #         if post.owner == player_color and post.type == TradePostType.POST:
        #             # Find closest city
        #             closest_cities = []
        #             min_distance = float('inf')
                    
        #             for city in self.visualizer.state_cache.cities:
        #                 dist = hex_coord.distance(city.location)
        #                 if dist < min_distance:
        #                     min_distance = dist
        #                     closest_cities = [city]
        #                 elif dist == min_distance:
        #                     closest_cities.append(city)
                    
        #             # If multiple cities are equidistant, player chooses
        #             if len(closest_cities) > 1:
        #                 posts_needing_selection.append((hex_coord, closest_cities))
        #             elif closest_cities:
        #                 city = closest_cities[0]
        #                 if city.common_good not in common_goods:
        #                     common_goods[city.common_good] = 0
        #                 common_goods[city.common_good] += 1
        #                 income_sources["trading_posts"].append((hex_coord, city, city.common_good))
        
        # # If any posts need city selection, we need to handle that first
        # if posts_needing_selection:
        #     self.visualizer.posts_needing_selection = posts_needing_selection
        #     self.visualizer.current_post_selection_index = 0
        #     self.visualizer.collected_common_goods = common_goods
        #     self.visualizer.collected_rare_goods = rare_goods
        #     self.visualizer.income_sources = income_sources
            
        #     # Show dialog for the first post
        #     post_hex, cities = posts_needing_selection[0]
        #     self.show_dialog(
        #         dialog_type="choose_income_city",
        #         message=f"Choose which city to collect goods from for trading post at {post_hex}:",
        #         options=[city.name for city in cities],
        #         context_data={"post_hex": post_hex, "cities": cities}
        #     )
        #     return
        
        # # If no selection needed, process the income immediately
        # self.submit_income_action(common_goods, rare_goods, income_sources)

    def _create_income_display_text(self, action_string):
        """Convert income action string to user-friendly display text."""
        # Parse action string like "income Camel:3,Horses:1|"
        if not action_string.startswith("income "):
            return action_string
        
        goods_part = action_string[7:]  # Remove "income "
        if goods_part.endswith("|"):
            goods_part = goods_part[:-1]  # Remove trailing |
        
        # Parse goods into common and rare
        common_goods = {}
        rare_goods = {}
        
        # Determine which goods are rare by checking city data
        rare_good_names = {city.rare_good for city in self.visualizer.state_cache.cities}
        
        for item in goods_part.split(','):
            if ':' in item:
                good_name, count_str = item.split(':', 1)
                good_name = good_name.strip()
                try:
                    count = int(count_str.strip())
                    if good_name in rare_good_names:
                        rare_goods[good_name] = count
                    else:
                        common_goods[good_name] = count
                except ValueError:
                    continue
        
        # Create display text
        parts = []
        if common_goods:
            common_items = [f"{count} {name}" for name, count in common_goods.items()]
            parts.append("Common: " + ", ".join(common_items))
        if rare_goods:
            rare_items = [f"{count} {name}" for name, count in rare_goods.items()]
            parts.append("Rare: " + ", ".join(rare_items))
        
        if parts:
            return " | ".join(parts)
        else:
            return "No goods"


    def _display_income_summary(self, action_string):
        """Display a summary of income being collected."""
        display_text = self._create_income_display_text(action_string)
        self.visualizer.control_panel.update_status(f"Collecting income: {display_text}")


    def create_income_action(self, common_goods, rare_goods):
        """Create income action string with standardized formatting."""
        collection_str = GoodsFormatter.format_goods_collection_compact(common_goods, rare_goods)
        return f"income {collection_str}"


    def submit_income_action(self, common_goods, rare_goods, income_sources=None):
        """Submit income action with standardized formatting."""
        
        # Create standardized income action string
        action_string = self.create_income_action(common_goods, rare_goods)
        
        # Display feedback about income collected
        total_common = sum(common_goods.values())
        total_rare = sum(rare_goods.values())
        self.visualizer.control_panel.update_status(
            f"Collecting income: {total_common} common goods, {total_rare} rare goods"
        )
        
        # Apply the action
        self.visualizer.attempt_apply_action(action_string)



    # # def submit_income_action(self, common_goods, rare_goods, income_sources=None):
    # #     """Submit the income action to apply the collected goods."""
    # #     # Format the action string
    # #     common_str = ",".join([f"{name}:{count}" for name, count in common_goods.items()])
    # #     rare_str = ",".join([f"{name}:{count}" for name, count in rare_goods.items()])
    # #     parts = ["income"]
    # #     if common_goods:
    # #         common_str = ",".join([f"{name}:{count}" for name, count in common_goods.items()])
    # #         parts.append(common_str)
    # #     if rare_goods:
    # #         rare_str = ",".join([f"{name}:{count}" for name, count in rare_goods.items()])
    # #         parts.append(rare_str)
    # #     action_string = " ".join(parts)
        
    # #     # Display feedback about income collected
    # #     total_common = sum(common_goods.values())
    # #     total_rare = sum(rare_goods.values())
    # #     self.visualizer.control_panel.update_status(f"Collecting income: {total_common} common goods, {total_rare} rare goods")
        
    # #     # Apply the action
    # #     self.visualizer.attempt_apply_action(action_string)

    # def create_income_action(self, common_goods, rare_goods):
    #     """Create the income action string to send to the backend."""
    #     parts = ["income"]
        
    #     # Add common goods if any
    #     if common_goods:
    #         common_str = ",".join([f"{name}:{count}" for name, count in common_goods.items()])
    #         parts.append(common_str)
        
    #     # Add rare goods if any  
    #     if rare_goods:
    #         rare_str = ",".join([f"{name}:{count}" for name, count in rare_goods.items()])
    #         parts.append(rare_str)
        
    #     return " ".join(parts)

    def normalize_income_action(action_string):
        """
        Normalize income action strings to ensure consistent ordering using standardized format.
        
        Args:
            action_string: String like "income Palm Oil:1,Salt:2|Gold:1"
        
        Returns:
            Normalized string with goods sorted alphabetically within each category
        """
        if not action_string.startswith("income"):
            return action_string
        
        parts = action_string.split(' ', 1)
        if len(parts) != 2:
            return action_string
        
        # Parse and re-normalize the goods collection
        common_goods, rare_goods = GoodsFormatter.parse_goods_collection(parts[1])
        
        # Create normalized action string
        collection_str = GoodsFormatter.format_goods_collection_compact(common_goods, rare_goods)
        return f"income {collection_str}"


    # --- Trade Route Methods ---

    def start_trade_route_mode(self):
        """Start the trade route creation/editing mode."""
        if self.visualizer.state_cache.is_terminal or self.visualizer.state_cache.current_player_id < 0:
            self.visualizer.control_panel.update_status("Cannot create trade routes at this time.")
            return
            
        self.visualizer.is_input_mode = True
        self.visualizer.input_mode_type = "trade_route"
        self.visualizer.highlight_hexes = []  # Will store hexes in the route
        self.visualizer.selected_trade_route = None  # Will store an existing route if editing
        
        self.visualizer.control_panel.update_status("Trade Route Mode: Select trading posts to connect. Submit when done.")


    def handle_trade_route_hex_click(self, hex_coord):
        """Handle hex clicks when creating/editing a trade route."""
        if not self.visualizer.is_input_mode or self.visualizer.input_mode_type != "trade_route":
            return
            
        player_color = self.visualizer.state_cache.current_player_color
        
        # Check if player has a trading post or center at this hex
        has_trading_post = False
        is_city = False
        posts = self.visualizer.state_cache.trade_posts_locations.get(hex_coord, [])
        
        # Check if this hex is a city
        for city in self.visualizer.state_cache.cities:
            if city.location == hex_coord:
                is_city = True
                break
        
        # Check if player has a trading post/center here
        for post in posts:
            if post.owner == player_color:
                has_trading_post = True
                break
        
        if has_trading_post:
            # Add/remove hex from the route
            if hex_coord in self.visualizer.highlight_hexes:
                self.visualizer.highlight_hexes.remove(hex_coord)
                self.visualizer.control_panel.update_status(f"Removed {hex_coord} from trade route. Select more or Submit.")
            else:
                self.visualizer.highlight_hexes.append(hex_coord)
                
                # Check if we connected to a city
                city_name = None
                if is_city:
                    for city in self.visualizer.state_cache.cities:
                        if city.location == hex_coord:
                            city_name = city.name
                            break
                        
                status_msg = f"Added {hex_coord} to trade route"
                if city_name:
                    status_msg += f" (City: {city_name})"
                status_msg += ". Trade routes require trading centers at non-city hexes."
                
                self.visualizer.control_panel.update_status(status_msg)
        else:
            self.visualizer.control_panel.update_status(f"You need a trading post or center at {hex_coord} to include it in a trade route.")


    def submit_trade_route(self):
        """Submit the trade route to the game state."""
        if not self.visualizer.is_input_mode or self.visualizer.input_mode_type != "trade_route":
            return False
            
        # Need at least 2 hexes for a valid route
        if len(self.visualizer.highlight_hexes) < 2:
            self.visualizer.control_panel.update_status("Trade route needs at least 2 connected hexes. Please select more.")
            return False
            
        player_color = self.visualizer.state_cache.current_player_color
        player_id = self.visualizer.state_cache.current_player_id
        
        # Validate that the hexes meet the requirements for a trade route
        # 1. Check if non-city hexes have trading centers
        # 2. Check if city hexes have trading posts or centers
        # 3. Check for duplicate route
        
        non_city_hexes_without_center = []
        city_hexes_with_post = []
        city_hexes_to_upgrade = []
        
        for hex_coord in self.visualizer.highlight_hexes:
            # Check if this is a city hex
            is_city = False
            for city in self.visualizer.state_cache.cities:
                if city.location == hex_coord:
                    is_city = True
                    break
            
            # Get the player's posts at this hex
            player_posts = []
            all_posts = self.visualizer.state_cache.trade_posts_locations.get(hex_coord, [])
            for post in all_posts:
                if post.owner == player_color:
                    player_posts.append(post)
            
            if is_city:
                # For city hexes, we need at least a trading post or center
                if not player_posts:
                    self.visualizer.control_panel.update_status(
                        f"You need at least a trading post at city hex {hex_coord} for a trade route."
                    )
                    return False
                
                # Check if the city has a trading post that needs upgrade
                for post in player_posts:
                    if post.type == TradePostType.POST:
                        city_hexes_with_post.append(hex_coord)
                        city_hexes_to_upgrade.append(hex_coord)
                        break
            else:
                # For non-city hexes, we need a trading center
                has_center = False
                for post in player_posts:
                    if post.type == TradePostType.CENTER:
                        has_center = True
                        break
                
                if not has_center:
                    non_city_hexes_without_center.append(hex_coord)
        
        # Check if any non-city hexes don't have trading centers
        if non_city_hexes_without_center:
            self.visualizer.control_panel.update_status(
                f"Trade routes require trading centers at non-city hexes. Upgrade the posts at: {', '.join(str(h) for h in non_city_hexes_without_center)}"
            )
            return False
        
        # Check for duplicate route
        player_routes = self.visualizer.state_cache.get_player_trade_routes(player_color)
        for route in player_routes:
            if set(route.hexes) == set(self.visualizer.highlight_hexes) and route != self.visualizer.selected_trade_route:
                self.visualizer.control_panel.update_status(
                    f"You already have a trade route (#{route.id}) with these exact hexes."
                )
                return False
        
        # Automatically upgrade city trading posts to centers if needed
        posts_upgraded = False
        if city_hexes_to_upgrade:
            for hex_coord in city_hexes_to_upgrade:
                posts = self.visualizer.state_cache.trade_posts_locations.get(hex_coord, [])
                for i, post in enumerate(posts):
                    if post.owner == player_color and post.type == TradePostType.POST:
                        # Upgrade the post to a center
                        self.visualizer.state_cache.trade_posts_locations[hex_coord][i].type = TradePostType.CENTER
                        posts_upgraded = True
            
            if posts_upgraded:
                self.visualizer.control_panel.update_status(
                    "Automatically upgraded city trading posts to centers."
                )
        
        # All validation passed, continue with creating/updating the route
        
        # Check if editing an existing route or creating a new one
        if self.visualizer.selected_trade_route is not None:
            # Editing existing route
            route_id = self.visualizer.selected_trade_route.id
            action_string = f"trade_route update {route_id} {':'.join(str(h) for h in self.visualizer.highlight_hexes)}"
        else:
            # Creating new route
            next_route_id = self.visualizer.state_cache.next_route_id
            action_string = f"trade_route create {next_route_id} {':'.join(str(h) for h in self.visualizer.highlight_hexes)}"
        
        # Apply the action
        success = self.visualizer.attempt_apply_action(action_string)
        
        if success:
            # Ensure routes are validated after changes
            self.visualizer.state_cache.validate_trade_routes()
            # Update the game state to reflect the changes to trading posts
            if posts_upgraded:
                self._update_trade_posts_in_game_state()
            self._update_trade_routes_in_game_state()
            self.visualizer.control_panel.update_status("Trade route updated successfully.")
        else:
            self.visualizer.control_panel.update_status("Failed to update trade route.")
            
        return success


    # helper method to update trade posts in game state
    def _update_trade_posts_in_game_state(self):
        """Update the game state with the current trade posts."""
        try:
            current_state_str = self.visualizer.game_interface.get_current_state_string()
            state_data = json.loads(current_state_str)
            
            # Convert trade_posts_locations to the format expected by the game state
            trade_posts_data = {}
            for hex_coord, posts in self.visualizer.state_cache.trade_posts_locations.items():
                post_entries = []
                for post in posts:
                    # Find player ID from color
                    owner_id = -1
                    for i, color in enumerate(self.visualizer.state_cache.game_player_colors):
                        if color == post.owner:
                            owner_id = i
                            break
                    
                    if owner_id >= 0:
                        post_entries.append({
                            "owner": owner_id,
                            "type": post.type.value
                        })
                
                if post_entries:
                    trade_posts_data[str(hex_coord)] = post_entries
            
            # Update state data
            state_data["tradePosts"] = trade_posts_data
            
            # Send updated state back to the game interface
            updated_state_str = json.dumps(state_data)
            self.visualizer.game_interface.load_state_from_string(updated_state_str)
        except Exception as e:
            print(f"Error updating trade posts in game state: {e}")
            self.visualizer.control_panel.update_status("Error saving trade post upgrades. Please try again.")

    def _update_trade_routes_in_game_state(self):
        """Update the game state with the current trade routes."""
        # Create a serializable representation of all trade routes
        routes_data = []
        
        for route in self.visualizer.state_cache.trade_routes:
            # Convert owner color to player ID
            owner_id = -1
            for i, color in enumerate(self.visualizer.state_cache.game_player_colors):
                if color == route.owner:
                    owner_id = i
                    break
            
            # Skip invalid routes
            if owner_id < 0: continue
                
            # Convert hexes to strings
            hex_strings = [str(h) for h in route.hexes]
            
            route_data = {
                "id": route.id,
                "owner": owner_id,
                "hexes": hex_strings,
                "goods": route.goods,
                "active": route.active
            }
            
            routes_data.append(route_data)
        
        # Get current state data, add trade routes, and send the updated state
        try:
            current_state_str = self.visualizer.game_interface.get_current_state_string()
            state_data = json.loads(current_state_str)
            
            # Update trade routes in the state
            state_data["tradeRoutes"] = routes_data
            
            # Send updated state back to the game interface
            updated_state_str = json.dumps(state_data)
            self.visualizer.game_interface.load_state_from_string(updated_state_str)
        except Exception as e:
            print(f"Error updating trade routes in game state: {e}")
            self.visualizer.control_panel.update_status("Error saving trade routes. Please try again.")

    def show_trade_route_options(self, route_id):
        """Show dialog with options for an existing trade route."""
        player_color = self.visualizer.state_cache.current_player_color
        
        # Find the route
        selected_route = None
        for route in self.visualizer.state_cache.trade_routes:
            if route.id == route_id and route.owner == player_color:
                selected_route = route
                break
                
        if not selected_route:
            self.visualizer.control_panel.update_status(f"Error: Trade route #{route_id} not found.")
            return
            
        # Create dialog options
        options = [
            f"Edit Route #{route_id}",
            f"Delete Route #{route_id}",
            "Cancel"
        ]
        
        # Determine active/inactive status text
        status_text = "active" if selected_route.active else "inactive"
        city_count = sum(1 for city in self.visualizer.state_cache.cities if city.location in selected_route.hexes)
        
        self.show_dialog(
            dialog_type="trade_route_options",
            message=f"Trade Route #{route_id} ({status_text})\n{len(selected_route.hexes)} hexes, {city_count} cities",
            options=options,
            context_data={"route": selected_route}
        )


    def get_trade_route_status_text(self, route):
        """Get a descriptive status text for the given trade route."""
        status = "active" if route.active else "inactive"
        
        # Count how many cities are in the route
        city_count = 0
        city_names = []
        for hex_coord in route.hexes:
            city = self.get_city_at_hex(hex_coord)
            if city:
                city_count += 1
                city_names.append(city.name)
        
        # Format the goods as a string
        goods_str = ", ".join([f"{name}: {count}" for name, count in route.goods.items()]) if route.goods else "none"
        
        # Build the status text
        text = f"Trade Route #{route.id} ({status})\n"
        text += f"{len(route.hexes)} hexes, {city_count} cities"
        if city_names:
            text += f" ({', '.join(city_names)})"
        text += f"\nGoods: {goods_str}"
        
        return text


    def delete_trade_route(self, route_id):
        """Delete a trade route with confirmation."""
        # Find the route
        player_color = self.visualizer.state_cache.current_player_color
        route_to_delete = None
        
        for route in self.visualizer.state_cache.trade_routes:
            if route.id == route_id and route.owner == player_color:
                route_to_delete = route
                break
                
        if not route_to_delete:
            self.visualizer.control_panel.update_status(f"Error: Trade route #{route_id} not found.")
            return
            
        # Create action string
        action_string = f"trade_route delete {route_id}"
        
        # Apply the action
        success = self.visualizer.attempt_apply_action(action_string)
        
        if success:
            self.visualizer.control_panel.update_status(f"Trade route #{route_id} deleted successfully.")
            # Ensure the route is removed from the local cache
            self.visualizer.state_cache.remove_trade_route(route_id)
        else:
            self.visualizer.control_panel.update_status(f"Failed to delete trade route #{route_id}.")
            
        # Exit input mode if we were in it
        self.visualizer.cancel_input_mode()

    def check_trade_route_click(self, pos):
        """Check if the click is on a trade route marker."""
        # If we're already in trade route input mode, don't process route clicks
        if self.visualizer.is_input_mode and self.visualizer.input_mode_type == "trade_route":
            return False
            
        # Get current player
        player_color = self.visualizer.state_cache.current_player_color
        
        # Only consider clicks near hexes for efficiency
        hex_coord = self.visualizer.pixel_to_hex(pos[0], pos[1])
        if not hex_coord:
            return False
            
        # Check if any routes have this hex
        routes_at_hex = self.visualizer.state_cache.get_hex_trade_routes(hex_coord)
        if not routes_at_hex:
            return False
            
        # Filter for player's routes only
        player_routes = [r for r in routes_at_hex if r.owner == player_color]
        if not player_routes:
            return False
            
        # If there's exactly one route here, select it
        if len(player_routes) == 1:
            route = player_routes[0]
            self.show_trade_route_options(route.id)
            return True
            
        # If there are multiple routes, show selection dialog
        options = [f"Route #{r.id} ({len(r.hexes)} hexes)" for r in player_routes]
        options.append("Cancel")
        
        self.show_dialog(
            dialog_type="trade_route_selection",
            message=f"Multiple trade routes at {hex_coord}. Select one:",
            options=options,
            context_data={"routes": player_routes, "hex": hex_coord}
        )
        return True
    
    # Additional method to get trading post/center type for a player at a hex 
    def get_player_post_type(self, hex_coord, player_color):
        """Get the type of the player's post at the given hex, if any."""
        posts = self.visualizer.state_cache.trade_posts_locations.get(hex_coord, [])
        for post in posts:
            if post.owner == player_color:
                return post.type
        return None

    def handle_upgrade_hex_click(self, hex_coord):
        """Handle hex clicks when in upgrade mode."""
        # Check if this hex has a trading post owned by the current player
        current_player = self.visualizer.state_cache.current_player_color
        
        trade_posts = self.visualizer.state_cache.trade_posts_locations.get(hex_coord, [])
        player_posts = [post for post in trade_posts if post.owner == current_player and post.type == TradePostType.POST]
        
        if not player_posts:
            self.visualizer.control_panel.update_status("No upgradeable trading post at this location.")
            return
        
        # Get player's resources
        player_id = self.visualizer.state_cache.current_player_id
        common_goods = self.visualizer.state_cache.common_goods[player_id] if player_id < len(self.visualizer.state_cache.common_goods) else {}
        rare_goods = self.visualizer.state_cache.rare_goods[player_id] if player_id < len(self.visualizer.state_cache.rare_goods) else {}
        
        # Calculate available payment options using standardized format
        payment_options = []

        # Option 1: Any 3 common goods (mix and match allowed)
        total_common_goods = sum(common_goods.values())
        if total_common_goods >= 3:
            # Generate all possible combinations of 3 common goods
            from itertools import combinations_with_replacement
            
            # Create a list of available goods with their counts
            available_goods = []
            for good_name, count in common_goods.items():
                available_goods.extend([good_name] * count)
            
            # Generate unique combinations of exactly 3 goods
            unique_combinations = set()
            for combo in combinations_with_replacement(set(common_goods.keys()), 3):
                # Check if this combination is actually possible with available goods
                combo_dict = {}
                for good in combo:
                    combo_dict[good] = combo_dict.get(good, 0) + 1
                
                # Verify we have enough of each good type
                is_valid = True
                for good, needed in combo_dict.items():
                    if common_goods.get(good, 0) < needed:
                        is_valid = False
                        break
                
                if is_valid:
                    # Format the combination for display
                    combo_parts = []
                    for good, count in sorted(combo_dict.items()):
                        combo_parts.append(f"{good}:{count}")
                    combo_str = ",".join(combo_parts)
                    unique_combinations.add(combo_str)
            
            # Add all valid combinations to payment options
            for combo_str in sorted(unique_combinations):
                payment_options.append(f"{combo_str} (3 Common)")

        # Option 2: 1 rare good
        for good_name, count in rare_goods.items():
            if count >= 1:
                payment_options.append(f"{good_name}:1 (1 Rare)")

        if not payment_options:
            self.visualizer.control_panel.update_status("You don't have enough goods to upgrade this post. Need 3 common goods (any combination) OR 1 rare good.")
            return
        
        # Store the hex for later use
        self.visualizer.selected_upgrade_hex = hex_coord
        self.visualizer.highlight_hexes = [hex_coord]
        
        # Show payment selection dialog
        self.show_dialog(
            dialog_type="choose_upgrade_payment",
            message=f"Choose payment method to upgrade trading post at {hex_coord}:",
            options=payment_options + ["Cancel"],
            context_data={"hex_coord": hex_coord, "payment_options": payment_options}
        )

    # Update a trading post to trading center method
    def submit_upgrade_action(self):
        """Submit the upgrade action with standardized payment format."""
        if not hasattr(self.visualizer, 'selected_upgrade_hex') or not hasattr(self.visualizer, 'selected_upgrade_payment'):
            self.visualizer.control_panel.update_status("Error: Upgrade information missing.")
            return
        
        hex_coord = self.visualizer.selected_upgrade_hex
        payment_str = self.visualizer.selected_upgrade_payment  # e.g., "Salt:3"
        
        # Parse the payment to determine if it's common or rare
        payment_dict = GoodsFormatter.parse_goods_string(payment_str)
        
        # Categorize payment based on city data
        cities = self.visualizer.state_cache.cities
        rare_good_names = {city.rare_good for city in cities}
        
        common_payment = {}
        rare_payment = {}
        
        for good_name, count in payment_dict.items():
            if good_name in rare_good_names:
                rare_payment[good_name] = count
            else:
                common_payment[good_name] = count
        
        # Create the upgrade action string with standardized format
        # Format: "upgrade hex|common_payment|rare_payment"
        hex_str = f"{hex_coord.x},{hex_coord.y},{hex_coord.z}"
        payment_collection = GoodsFormatter.format_goods_collection(common_payment, rare_payment)
        action_string = f"upgrade {hex_str}|{payment_collection}"
        
        # Apply the action
        self.visualizer.attempt_apply_action(action_string)
        
        # Clean up
        if hasattr(self.visualizer, 'selected_upgrade_hex'):
            delattr(self.visualizer, 'selected_upgrade_hex')
        if hasattr(self.visualizer, 'selected_upgrade_payment'):
            delattr(self.visualizer, 'selected_upgrade_payment')


    # --- Utility Methods ---
    def get_city_at_hex(self, hex_coord):
        """Get the city at the given hex, if any."""
        for city in self.visualizer.state_cache.cities:
            if city.location == hex_coord:
                return city
        return None


    # Additional method to check if a hex is a city
    def is_city_hex(self, hex_coord):
        """Check if the given hex contains a city."""
        return self.get_city_at_hex(hex_coord) is not None
    

    def update_status_from_cache(self):
        """Updates the control panel status based on the current state cache."""
        if self.visualizer.state_cache.is_terminal:
             self.visualizer.control_panel.update_status("Game Over.")
        elif self.visualizer.state_cache.current_phase == Phase.START:
             self.visualizer.control_panel.update_status("Ready for Chance Setup action.")
        else:
             player_id = self.visualizer.state_cache.current_player_id
             phase_name = self.visualizer.state_cache.current_phase.name
             if player_id >= 0:
                 player_disp = f"Player {player_id + 1}"
                 self.visualizer.control_panel.update_status(f"{player_disp}'s turn. Phase: {phase_name}. Select action or hex.")
             elif player_id == -1: # Should normally be START phase
                 self.visualizer.control_panel.update_status(f"Chance Node active. Phase: {phase_name}.")
             else: # Should be terminal, handled above
                 self.visualizer.control_panel.update_status(f"Unknown State (Player {player_id}, Phase {phase_name})")


    def load_state(self, state_string):
        """Load a state from the provided JSON string via the interface."""
        success = self.visualizer.game_interface.load_state_from_string(state_string)
        if success:
            # Fetch the state string *after* loading it in the interface
            # to ensure we parse what the backend now holds
            current_state_str = self.visualizer.game_interface.get_current_state_string()
            if current_state_str:
                return self.visualizer.parse_and_update_state(current_state_str)
            else:
                print("Error: GameInterface loaded state but returned no string.")
                self.visualizer.control_panel.update_status("Error: State loaded but couldn't fetch.")
                return False
        else:
            print(f"Error: GameInterface failed to load state string.")
            self.visualizer.control_panel.update_status("Error: Failed to load state in backend.")
            return False


    def cleanup(self):
        """Safely clean up resources."""
        print("Cleaning up visualizer helper resources...")
        # Potentially close files, network connections etc. if added later

    def print_state_cache(self):
        """Print a formatted representation of the state cache to the terminal."""
        cache = self.visualizer.state_cache
        print("===== Current state cache =====")
        print(f"Current Player ID: {cache.current_player_id}")
        print(f"Current Player Color: {cache.current_player_color}")
        print(f"Current Phase: {cache.current_phase}")
        print(f"Is Terminal: {cache.is_terminal}")
        print(f"Grid Radius: {cache.grid_radius}")
        print(f"Game Player Colors: {[color.name for color in cache.game_player_colors]}")
        
        # Print Player Token Locations
        print("\n--- Player Token Locations ---")
        for hex_coord, players in cache.player_token_locations.items():
            print(f"{hex_coord}: {[player.name for player in players]}")
        
        # Print Trade Posts Locations
        print("\n--- Trade Posts Locations ---")
        for hex_coord, posts in cache.trade_posts_locations.items():
            post_info = []
            for post in posts:
                post_info.append(f"{post.owner.name}:{post.type.name}")
            print(f"{hex_coord}: {post_info}")
        
        # Print Cities
        print("\n--- Cities ---")
        for city in cache.cities:
            print(f"{city.name} at {city.location} - Culture: {city.culture}, Common: {city.common_good}, Rare: {city.rare_good}")
        
        # Print Common Goods
        print("\n--- Common Goods ---")
        for player_id, goods in enumerate(cache.common_goods):
            print(f"Player {player_id}: {goods}")
        
        # Print Rare Goods
        print("\n--- Rare Goods ---")
        for player_id, goods in enumerate(cache.rare_goods):
            print(f"Player {player_id}: {goods}")
        
        # Print Trade Routes
        print("\n--- Trade Routes ---")
        for route in cache.trade_routes:
            hex_str = ", ".join(str(h) for h in route.hexes)
            print(f"Route #{route.id} - Owner: {route.owner.name}, Active: {route.active}")
            print(f"  Hexes: {hex_str}")
            print(f"  Goods: {route.goods}")
        
        print("============================\n")

class GoodsFormatter:
    """Utility class for standardized goods formatting across the application."""
    
    @staticmethod
    def parse_goods_string(goods_str):
        """Parse a goods string like 'Palm Oil:1,Salt:2' into a dictionary."""
        goods = {}
        if not goods_str:
            return goods
        
        items = goods_str.split(',')
        for item in items:
            item = item.strip()
            if not item:
                continue
            
            if ':' in item:
                good_name, count_str = item.split(':', 1)
                good_name = good_name.strip()
                try:
                    count = int(count_str.strip())
                    if count > 0:
                        goods[good_name] = count
                except ValueError:
                    print(f"Warning: Could not parse count for good '{good_name}': '{count_str}'")
        
        return goods
    
    @staticmethod
    def format_goods_string(goods_dict):
        """Format a goods dictionary into a string like 'Palm Oil:1,Salt:2'."""
        if not goods_dict:
            return ""
        
        items = []
        # Sort goods alphabetically for consistency
        for good_name in sorted(goods_dict.keys()):
            count = goods_dict[good_name]
            if count > 0:
                items.append(f"{good_name}:{count}")
        
        return ','.join(items)
    
    @staticmethod
    def parse_goods_collection(collection_str, cities=None):
        """Parse a collection string like 'common_goods|rare_goods' into separate dictionaries."""
        parts = collection_str.split('|')
        
        common_goods = {}
        rare_goods = {}
        
        # Parse common goods (part 0)
        if len(parts) >= 1 and parts[0]:
            common_goods = GoodsFormatter.parse_goods_string(parts[0])
        
        # Parse rare goods (part 1)  
        if len(parts) >= 2 and parts[1]:
            rare_goods = GoodsFormatter.parse_goods_string(parts[1])
        
        # Auto-categorize goods if they're mixed in one part and we have city data
        if len(parts) == 1 and parts[0] and cities:
            all_goods = GoodsFormatter.parse_goods_string(parts[0])
            common_goods = {}
            rare_goods = {}
            
            # Create rare goods lookup
            rare_good_names = {city.rare_good for city in cities}
            
            # Categorize goods
            for good_name, count in all_goods.items():
                if good_name in rare_good_names:
                    rare_goods[good_name] = count
                else:
                    common_goods[good_name] = count
        
        return common_goods, rare_goods
    
    @staticmethod
    def format_goods_collection(common_goods, rare_goods):
        """Format goods into a collection string like 'common_goods|rare_goods'."""
        common_str = GoodsFormatter.format_goods_string(common_goods)
        rare_str = GoodsFormatter.format_goods_string(rare_goods)
        return f"{common_str}|{rare_str}"
    
    @staticmethod
    def format_goods_collection_compact(common_goods, rare_goods):
        """Format goods collection with empty parts omitted when possible."""
        common_str = GoodsFormatter.format_goods_string(common_goods)
        rare_str = GoodsFormatter.format_goods_string(rare_goods)
        
        if not common_str and not rare_str:
            return ""
        elif not rare_str:
            return f"{common_str}|"
        elif not common_str:
            return f"|{rare_str}"
        else:
            return f"{common_str}|{rare_str}"

# --- END OF FILE visualizer_other.py ---