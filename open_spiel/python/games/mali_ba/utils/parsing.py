import sys
sys.path.append("/media/robp/UD/Projects/mali_ba/open_spiel/python/games") # allow debugging in vs code
from typing import Dict, List, Optional, Set
from mali_ba.config import PlayerColor, MeepleColor, TradePostType, Phase
from mali_ba.classes.game_state import GameStateCache 
from mali_ba.classes.classes_other import TradePost, City, HexCoord
import json

# --- State Parsing & Updating ---
def determine_num_players_from_state(state_str: str) -> Optional[int]:
    """Attempt to determine num players from JSON state string."""
    if not state_str: return None
    try:
        data = json.loads(state_str)
        max_player_id = -1

        # 1. Check length of goods arrays (most reliable if present)
        if "commonGoods" in data and isinstance(data["commonGoods"], list):
            return len(data["commonGoods"])
        if "rareGoods" in data and isinstance(data["rareGoods"], list):
            return len(data["rareGoods"])

        # 2. Check player token owners
        if "playerTokens" in data:
            for player_list in data["playerTokens"].values():
                if isinstance(player_list, list):
                    for player_id in player_list:
                        if isinstance(player_id, int):
                            max_player_id = max(max_player_id, player_id)

        # 3. Check trade post owners
        if "tradePosts" in data:
             for posts_list in data["tradePosts"].values():
                 if isinstance(posts_list, list):
                     for post_data in posts_list:
                          if isinstance(post_data, dict) and "owner" in post_data:
                               if isinstance(post_data["owner"], int):
                                    max_player_id = max(max_player_id, post_data["owner"])

        # 4. Check current/previous player ID (less reliable for *max*)
        for key in ["currentPlayerId", "previousPlayerId"]:
            if key in data and isinstance(data[key], int) and data[key] >= 0:
                max_player_id = max(max_player_id, data[key])

        if max_player_id >= 0:
            num_players = max_player_id + 1
            if 2 <= num_players <= 5:
                return num_players
            else:
                print(f"Warning: Determined player count {num_players} from JSON out of range (2-5).")
                return None # Indicate failure
        else:
            # Could not determine from any field
            return None

    except (json.JSONDecodeError, TypeError, KeyError) as e:
        print(f"Error determining number of players from JSON state: {e}")
        return None
