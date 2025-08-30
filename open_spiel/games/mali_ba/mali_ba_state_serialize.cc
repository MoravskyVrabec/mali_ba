// mali_ba_state_serialize.cc
// Serialization functionality -- Deserialization is at the Game level

#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/games/mali_ba/hex_grid.h"

#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
// Include the nlohmann/json header
#include "json.hpp"
// For convenience
using json = nlohmann::json;

namespace open_spiel {
namespace mali_ba {

// The State::Serialize() and Game::DeserializeState() functions in OpenSpiel serve several important purposes, 
// primarily related to persistence, communication, and reproducibility:
// Saving and Loading Game Progress:
//     This is the most intuitive use case. You can serialize the current State object into a string.
//     This string can be saved to a file.
//     Later, you can read the string from the file and use Game::DeserializeState(saved_string) to recreate the exact same game state, allowing players or systems to resume the game from where they left off.
// Checkpointing in Long Computations (e.g., AI Training):
//     Training AI agents (like using Reinforcement Learning with MCTS) can take a very long time.
//     If the training process crashes or needs to be interrupted, you don't want to lose all progress.
//     Serialization allows the training system to periodically save the current game states being processed (along with agent parameters, etc.).
//     Upon restarting, the system can deserialize these states and resume training, saving potentially hours or days of computation.
// Distributed Systems and Parallel Processing:
//     If you are running game simulations or training across multiple machines or processes, you need a way to send the current game situation from one node/process to another.
//     Serializing the State into a string provides a standard, self-contained way to transmit it over a network or through inter-process communication channels. The receiving end can then deserialize it.
// Debugging and Reproducibility:
//     If you encounter a bug or an interesting situation during gameplay or testing, you can serialize the state at that exact moment.
//     This serialized string becomes a perfect "snapshot" of the problematic state.
//     You (or someone else) can later deserialize this string to instantly recreate the exact scenario for debugging algorithms (like checking MCTS search behavior, evaluating policy outputs, etc.) without needing to replay the entire game sequence leading up to it.
// Creating Test Scenarios or Puzzles:
//     You might want to test specific game functionalities or create puzzle-like starting positions.
//     You could manually set up a State object to represent that scenario, serialize it, and store the string.
//     Your tests or puzzle loader can then simply deserialize these strings to set up the required starting positions quickly and reliably.
// Logging and Replay:
//     For analysis or creating replay systems, you could log the serialized state string after each move (or at key intervals). This allows for perfect reconstruction and analysis of past games.

// Serialize the state to a string
// Serialize() const: This method (implemented by you for your specific game) must gather all the essential 
// information that defines the current state (board positions, current player, resources, game phase, scores, potentially 
// relevant history, etc.) and encode it into a single std::string. The format of this string is up to the game implementer.
// Helper to convert HexCoord to string for JSON keys (can't use HexCoord directly)
std::string HexCoordToJsonString(const HexCoord& hex) {
    // Using a simple comma-separated format. Avoid spaces/parentheses for simplicity.
    return std::to_string(hex.x) + "," + std::to_string(hex.y) + "," + std::to_string(hex.z);
}

// Helper to convert JSON string back to HexCoord
absl::optional<HexCoord> JsonStringToHexCoord(const std::string& s) {
    std::vector<std::string> parts = absl::StrSplit(s, ',');
    if (parts.size() != 3) return absl::nullopt;
    try {
        int x = std::stoi(parts[0]);
        int y = std::stoi(parts[1]);
        int z = std::stoi(parts[2]);
        if (x + y + z != 0) return absl::nullopt;
        return HexCoord(x, y, z);
    } catch (const std::exception& e) {
        return absl::nullopt;
    }
}

std::string Mali_BaState::Serialize() const {
    /*
    Converts the entire current game state into a JSON string that can be stored, transmitted, or recreated later.
    Serialize() takes all the important game state data and packages it into a self-contained string representation.
    What it does:

    Captures the complete game state at the current moment:

    Current player and phase, All player token locations, All meeples on the board, 
    All trade posts and their owners, Player resources (common and rare goods), 
    Move history, Trade routes

    Converts it to JSON format for easy parsing and transmission
    Returns a string that contains everything needed to recreate this exact game state
    */
    constexpr int kJsonSerializationVersion = 2;
    int tpsz;   // DEBUG
    json j;

    try {
        // Part 1: Version
        j["version"] = kJsonSerializationVersion;

        // Part 2: Game Flow State
        j["currentPlayerId"] = current_player_id_;
        j["currentPhase"] = static_cast<int>(current_phase_); // Store enum as int
       
        // Part 3: Player Tokens (map<HexCoord, PlayerColor>) -> json object { "x,y,z": color_int }
        json j_tokens = json::object();
        for (const auto& [hex, colors] : player_token_locations_) {
            if (!colors.empty()) {
                json token_list = json::array();
                for (PlayerColor color : colors) {
                    token_list.push_back(static_cast<int>(color));
                }
                j_tokens[HexCoordToJsonString(hex)] = token_list;
            }
        }
        j["playerTokens"] = j_tokens;

        // Part 4: Meeples (map<HexCoord, vector<MeepleColor>>) -> json object { "x,y,z": [mc1_int, mc2_int] }
        json j_meeples = json::object();
        for (const auto& [hex, meeples] : hex_meeples_) {
            if (!meeples.empty()) {
                json j_meeple_list = json::array();
                for (MeepleColor mc : meeples) {
                    j_meeple_list.push_back(static_cast<int>(mc));
                }
                j_meeples[HexCoordToJsonString(hex)] = j_meeple_list;
            }
        }
        j["hexMeeples"] = j_meeples;

        // Part 5: Trade Posts (map<HexCoord, vector<TradePost>>) -> json object { "x,y,z": [ {owner: int, type: int}, ... ] }
        json j_posts = json::object();
        for (const auto& [hex, posts] : trade_posts_locations_) {
            if (!posts.empty()) {
                json j_post_list = json::array();
                for (const auto& post : posts) {
                    if (post.type != TradePostType::kNone) { // Only serialize actual posts/centers
                         j_post_list.push_back({
                            {"owner", static_cast<int>(post.owner)},
                            {"type", static_cast<int>(post.type)}
                         });
                    }
                }
                // Only add if there were actual posts/centers at this hex
                if (!j_post_list.empty()) {
                    j_posts[HexCoordToJsonString(hex)] = j_post_list;
                }
            }
        }
        j["tradePosts"] = j_posts;
        // tpsz = j_post_list.size();   // DEBUG
        // LOG_INFO("In Serialize(). Number of trade posts: ", tpsz);   // DEBUG
        // tpsz = trade_posts_locations_.size();   // DEBUG
        // LOG_INFO("In Serialize(). trade_posts_locations_ size: ", tpsz);   // DEBUG

        // Part 5a. Serialize the players' supply of trading posts.
        j["playerPostsSupply"] = player_posts_supply_;

       
        // Part 6: Cities (part of Game, not serialized in State)
        // No action needed here.

        // Part 7: History (vector<Move>) -> json array [ move_obj1, move_obj2, ... ]
        json j_history = json::array();
        for (const auto& move : moves_history_) {
            json j_move;
            j_move["player"] = static_cast<int>(move.player);
            j_move["startHex"] = HexCoordToJsonString(move.start_hex); // Use helper
            j_move["type"] = static_cast<int>(move.type);
            j_move["placePost"] = move.place_trading_post;
            //j_move["directionIndex"] = move.direction_index;  // removed

            json j_path = json::array();
            for (const auto& path_hex : move.path) {
                j_path.push_back(HexCoordToJsonString(path_hex)); // Use helper
            }
            j_move["path"] = j_path;
            j_history.push_back(j_move);
        }
        j["history"] = j_history;

        // Part 8: Resources - Common Goods (vector<map<string, int>>) -> json array [ player0_goods_obj, player1_goods_obj ]
        json j_common_goods = json::array();
        for (const auto& player_goods : common_goods_) {
            j_common_goods.push_back(player_goods); // nlohmann/json handles map<string, int> directly
        }
        j["commonGoods"] = j_common_goods;

        // Part 9: Resources - Rare Goods (vector<map<string, int>>) -> json array [ player0_goods_obj, player1_goods_obj ]
        json j_rare_goods = json::array();
        for (const auto& player_goods : rare_goods_) {
            j_rare_goods.push_back(player_goods); // nlohmann/json handles map<string, int> directly
        }
        j["rareGoods"] = j_rare_goods;

        // Part 10. Trade Routes
        json j_routes = json::array();
        for (const auto& route : trade_routes_) {
            json j_route;
            j_route["id"] = route.id;
            j_route["owner"] = static_cast<int>(route.owner);
            
            json j_hexes = json::array();
            for (const auto& hex : route.hexes) {
                j_hexes.push_back(HexCoordToJsonString(hex));
            }
            j_route["hexes"] = j_hexes;
            
            j_route["goods"] = route.goods;
            j_route["active"] = route.active;
            
            j_routes.push_back(j_route);
        }
        j["tradeRoutes"] = j_routes;

        // Convert the JSON object to a string (compact version)
        return j.dump();

    } catch (const json::exception& e) {
        SpielFatalError(absl::StrCat("JSON serialization error: ", e.what()));
        return ""; // Should be unreachable
    }
}

// Helpers for Token management
bool Mali_BaState::HasTokenAt(const HexCoord& hex, PlayerColor color) const {
    auto it = player_token_locations_.find(hex);
    if (it == player_token_locations_.end()) return false;
    
    const auto& tokens = it->second;
    return std::find(tokens.begin(), tokens.end(), color) != tokens.end();
}

int Mali_BaState::CountTokensAt(const HexCoord& hex, PlayerColor color) const {
    auto it = player_token_locations_.find(hex);
    if (it == player_token_locations_.end()) return 0;
    
    const auto& tokens = it->second;
    return std::count(tokens.begin(), tokens.end(), color);
}

int Mali_BaState::CountTotalTokensAt(const HexCoord& hex) const {
    auto it = player_token_locations_.find(hex);
    if (it == player_token_locations_.end()) return 0;
    return it->second.size();
}

std::vector<PlayerColor> Mali_BaState::GetTokensAt(const HexCoord& hex) const {
    auto it = player_token_locations_.find(hex);
    if (it == player_token_locations_.end()) return {};
    return it->second;
}

bool Mali_BaState::RemoveTokenAt(const HexCoord& hex, PlayerColor color) {
    auto it = player_token_locations_.find(hex);
    if (it == player_token_locations_.end()) return false;
    
    auto& tokens = it->second;
    auto token_it = std::find(tokens.begin(), tokens.end(), color);
    if (token_it == tokens.end()) return false;
    
    tokens.erase(token_it);
    
    // Clean up empty hex entries
    if (tokens.empty()) {
        player_token_locations_.erase(it);
    }
    
    return true;
}

void Mali_BaState::AddTokenAt(const HexCoord& hex, PlayerColor color) {
    player_token_locations_[hex].push_back(color);
}

PlayerColor Mali_BaState::GetFirstTokenAt(const HexCoord& hex) const {
    auto it = player_token_locations_.find(hex);
    if (it == player_token_locations_.end() || it->second.empty()) {
        return PlayerColor::kEmpty;
    }
    return it->second[0];
}

// Deserialize is not included here - it would be part of the Game class in mali_ba_game.cc
// as DeserializeState. This is because the Game object needs to create a new State
// with the correct parameters.

}  // namespace mali_ba
}  // namespace open_spiel