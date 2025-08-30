// mali_ba_state_setup.cc
// Setup and chance-related functionality

#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/hex_grid.h"

#include <random>
#include <vector>
#include <fstream>
#include <iostream>
// Include the nlohmann/json header
#include "json.hpp"
// For convenience
using json = nlohmann::json;


namespace open_spiel {
namespace mali_ba {

// Performs the random meeple setup distribution
void Mali_BaState::ApplyChanceSetup() {
    LOG_INFO("ApplyChanceSetup: START");
    
    for (auto const &[hex, meeples] : hex_meeples_) {
        SPIEL_CHECK_TRUE(meeples.empty());
    }

    std::vector<MeepleColor> all_meeple_colors = {
        MeepleColor::kSolidBlack, MeepleColor::kClearBlack,
        MeepleColor::kSolidSilver, MeepleColor::kClearSilver,
        MeepleColor::kClearWhite, MeepleColor::kSolidGold,
        MeepleColor::kClearGold, MeepleColor::kSolidBronze,
        MeepleColor::kClearBronze, MeepleColor::kClearTan
    };

    std::uniform_int_distribution<int> dist(0, all_meeple_colors.size() - 1);

    // Place 3 random meeples on EVERY valid hex, including cities.
    for (const auto &hex : GetGame()->GetValidHexes()) {

        // **ADD THIS DEBUG CODE:**
        // if (hex.x == 5 && hex.y == 1 && hex.z == -6) {
        //     LOG_INFO("DEBUG: Processing hex 5,1,-6 in ApplyChanceSetup");
        //     bool exists_in_meeples = hex_meeples_.count(hex) > 0;
        //     LOG_INFO("DEBUG: hex_meeples_.count(5,1,-6): ", exists_in_meeples);
        // }

        SPIEL_CHECK_TRUE(hex_meeples_.count(hex));
        hex_meeples_[hex].clear();

        for (int i = 0; i < 3; ++i) {
            int random_index = dist(rng_);
            hex_meeples_[hex].push_back(all_meeple_colors[random_index]);
        }

        // **ADD THIS DEBUG CODE:**
        // if (hex.x == 5 && hex.y == 1 && hex.z == -6) {
        //     LOG_INFO("DEBUG: Added ", hex_meeples_[hex].size(), " meeples to 5,1,-6");
        // }

    }
    
    LOG_INFO("ApplyChanceSetup: END");
}

 // String conversion functions (unchanged)
std::string PlayerColorToString(PlayerColor c) {
    switch (c) {
        case PlayerColor::kRed: return "Red";
        case PlayerColor::kGreen: return "Green";
        case PlayerColor::kBlue: return "Blue";
        case PlayerColor::kViolet: return "Violet";
        case PlayerColor::kPink: return "Pink";
        default: return "Empty";
    }
}

PlayerColor StringToPlayerColor(const std::string &s) {
    std::string lower_s = StrLower(s);
    if (lower_s == "red") return PlayerColor::kRed;
    if (lower_s == "green") return PlayerColor::kGreen;
    if (lower_s == "blue") return PlayerColor::kBlue;
    if (lower_s == "violet") return PlayerColor::kViolet;
    if (lower_s == "pink") return PlayerColor::kPink;
    return PlayerColor::kEmpty;
}

char PlayerColorToChar(PlayerColor pc) {
    if (pc == PlayerColor::kEmpty) return '.';
    return PlayerColorToString(pc)[0];
}

std::string MeepleColorToString(MeepleColor mc) {
    switch (mc) {
        case MeepleColor::kSolidBlack: return "sb";
        case MeepleColor::kClearBlack: return "cb";
        case MeepleColor::kSolidSilver: return "ss";
        case MeepleColor::kClearSilver: return "cs";
        case MeepleColor::kClearWhite: return "cw";
        case MeepleColor::kSolidGold: return "sg";
        case MeepleColor::kClearGold: return "cg";
        case MeepleColor::kSolidBronze: return "sz";
        case MeepleColor::kClearBronze: return "cz";
        case MeepleColor::kClearTan: return "ct";
        default: return "";
    }
}

// Player helpers 
Player Mali_BaState::GetPlayerId(PlayerColor color) const {
    if (color == PlayerColor::kEmpty) return kInvalidPlayer;
    const auto &player_colors = GetGame()->GetPlayerColors();
    for (Player i = 0; i < player_colors.size(); ++i) {
        if (player_colors[i] == color) return i;
    }
    return kInvalidPlayer;
}

PlayerColor Mali_BaState::GetPlayerColor(Player id) const {
    if (id < 0 || id >= game_->NumPlayers()) return PlayerColor::kEmpty;
    return GetGame()->GetPlayerColors()[id];
}

PlayerColor Mali_BaState::GetNextPlayerColor(PlayerColor current) const {
    const auto &player_colors = GetGame()->GetPlayerColors();
    for (size_t i = 0; i < player_colors.size(); ++i) {
        if (player_colors[i] == current) {
            return player_colors[(i + 1) % player_colors.size()];
        }
    }
    SpielFatalError("GetNextPlayerColor: Current color not found.");
    return PlayerColor::kEmpty;
}

// Entity getters 
PlayerColor Mali_BaState::GetPlayerTokenAt(const HexCoord& hex) const {
    auto it = player_token_locations_.find(hex);
    if (it != player_token_locations_.end() && !it->second.empty()) {
        return it->second[0]; // Return first token for backward compatibility
    }
    return PlayerColor::kEmpty;
}

const std::vector<MeepleColor>& Mali_BaState::GetMeeplesAt(const HexCoord& hex) const {
    static const std::vector<MeepleColor> empty_vector;
    auto it = hex_meeples_.find(hex);
    return (it != hex_meeples_.end()) ? it->second : empty_vector;
}


// // Helper method to generate regular hex board (if not already existing)
// std::set<HexCoord> Mali_BaState::GenerateRegularHexBoard(int radius) const {
//     std::set<HexCoord> hexes;
//     for (int q = -radius; q <= radius; ++q) {
//         int r_min = std::max(-radius, -q - radius);
//         int r_max = std::min(radius, -q + radius);
//         for (int r = r_min; r <= r_max; ++r) {
//             int s = -q - r;
//             hexes.emplace(q, r, s);
//         }
//     }
//     return hexes;
// }


// Helper method implementations
HexCoord Mali_BaState::ParseHexCoordFromString(const std::string& coord_str) const {
    std::vector<std::string> coords = absl::StrSplit(coord_str, ',');
    if (coords.size() != 3) {
        throw std::invalid_argument("Invalid coordinate format: " + coord_str);
    }
    
    int x, y, z;
    if (!absl::SimpleAtoi(absl::StripAsciiWhitespace(coords[0]), &x) ||
        !absl::SimpleAtoi(absl::StripAsciiWhitespace(coords[1]), &y) ||
        !absl::SimpleAtoi(absl::StripAsciiWhitespace(coords[2]), &z)) {
        throw std::invalid_argument("Failed to parse coordinates: " + coord_str);
    }
    
    return HexCoord(x, y, z);
}

std::vector<HexCoord> Mali_BaState::ParseHexListFromData(const std::vector<std::vector<int>>& hex_data) const {
    std::vector<HexCoord> hexes;
    
    for (const auto& coord_array : hex_data) {
        if (coord_array.size() != 3) {
            LOG_WARN("Invalid hex coordinate array size: ", coord_array.size());
            continue;
        }
        
        hexes.emplace_back(coord_array[0], coord_array[1], coord_array[2]);
    }
    
    return hexes;
}

bool Mali_BaState::SetStateFromJson(const std::string& json_str) {
    try {
        LOG_INFO("🧪 TESTING: Setting state from JSON (", json_str.length(), " chars)");
        
        // Parse and validate JSON
        json j = json::parse(json_str);
        LOG_INFO("✅ JSON parsing successful");
        
        // Clear current state
        ClearAllState();
        
        // Restore state from JSON (reuse existing deserialization logic)
        if (j.contains("version")) {
            int version = j.at("version").get<int>();
            LOG_INFO("State version: ", version);
        }
        
        // Game flow state
        if (j.contains("currentPlayerId")) {
            current_player_id_ = j.at("currentPlayerId").get<int>();
            current_player_color_ = GetPlayerColor(current_player_id_);
        }
        
        if (j.contains("currentPhase")) {
            current_phase_ = static_cast<Phase>(j.at("currentPhase").get<int>());
        }
        
        // Player tokens
        if (j.contains("playerTokens")) {
            for (const auto& [hex_str, color_array] : j.at("playerTokens").items()) {
                if (auto hex = JsonStringToHexCoord(hex_str)) {
                    if (color_array.is_array()) {
                        std::vector<PlayerColor> colors;
                        for (const auto& color_int : color_array) {
                            colors.push_back(static_cast<PlayerColor>(color_int.get<int>()));
                        }
                        if (!colors.empty()) {
                            player_token_locations_[*hex] = colors;
                        }
                    }
                }
            }
        }
        
        // Hex meeples
        if (j.contains("hexMeeples")) {
            for (const auto& [hex_str, meeple_array] : j.at("hexMeeples").items()) {
                if (auto hex = JsonStringToHexCoord(hex_str)) {
                    std::vector<MeepleColor> meeples;
                    for (const auto& mc_int : meeple_array) {
                        meeples.push_back(static_cast<MeepleColor>(mc_int.get<int>()));
                    }
                    hex_meeples_[*hex] = meeples;
                }
            }
        }
        
        // Trade posts
        if (j.contains("tradePosts")) {
            for (const auto& [hex_str, post_array] : j.at("tradePosts").items()) {
                if (auto hex = JsonStringToHexCoord(hex_str)) {
                    std::vector<TradePost> posts;
                    for (const auto& post_obj : post_array) {
                        TradePost post;
                        post.owner = static_cast<PlayerColor>(post_obj.at("owner").get<int>());
                        post.type = static_cast<TradePostType>(post_obj.at("type").get<int>());
                        posts.push_back(post);
                    }
                    trade_posts_locations_[*hex] = posts;
                }
            }
        }
        
        // Move history
        if (j.contains("history")) {
            for (const auto& j_move : j.at("history")) {
                Move move;
                move.player = static_cast<PlayerColor>(j_move.at("player").get<int>());
                move.type = static_cast<ActionType>(j_move.at("type").get<int>());
                move.place_trading_post = j_move.value("placePost", false);
                
                if (auto start_hex = JsonStringToHexCoord(j_move.at("startHex").get<std::string>())) {
                    move.start_hex = *start_hex;
                }
                
                if (j_move.contains("path")) {
                    for (const auto& path_hex_str : j_move.at("path")) {
                        if (auto path_hex = JsonStringToHexCoord(path_hex_str.get<std::string>())) {
                            move.path.push_back(*path_hex);
                        }
                    }
                }
                
                moves_history_.push_back(move);
            }
        }
        
        // Resources
        if (j.contains("commonGoods")) {
            SetCommonGoods(j.at("commonGoods"));
        }
        
        if (j.contains("rareGoods")) {
            SetRareGoods(j.at("rareGoods"));
        }
        
        // Trade routes
        if (j.contains("tradeRoutes")) {
            for (const auto& j_route : j.at("tradeRoutes")) {
                TradeRoute route;
                route.id = j_route.at("id").get<int>();
                route.owner = static_cast<PlayerColor>(j_route.at("owner").get<int>());
                
                if (j_route.contains("hexes")) {
                    for (const auto& h_str : j_route.at("hexes")) {
                        if (auto h = JsonStringToHexCoord(h_str)) {
                            route.hexes.push_back(*h);
                        }
                    }
                }
                
                if (j_route.contains("goods")) {
                    route.goods = j_route.at("goods").get<std::map<std::string, int>>();
                }
                
                route.active = j_route.value("active", true);
                route.hexes = GetCanonicalRoute(route.hexes); // make sure hexes are sorted properly
                trade_routes_.push_back(route);
                
                if (route.id >= next_route_id_) {
                    next_route_id_ = route.id + 1;
                }
            }
        }
        
        // Clean up and refresh
        ClearCaches();
        RefreshTerminalStatus();
        
        LOG_INFO("✅ State successfully set from JSON");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("❌ SetStateFromJson failed: ", e.what());
        return false;
    }
}

bool Mali_BaState::SaveStateToFile(const std::string& filename) const {
    try {
        std::string state_json = Serialize();
        std::ofstream file(filename);
        if (!file.is_open()) {
            LOG_ERROR("❌ Could not open file for writing: ", filename);
            return false;
        }
        
        file << state_json;
        file.close();
        
        LOG_INFO("💾 State saved to ", filename, " (", state_json.length(), " chars)");
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR("❌ SaveStateToFile failed: ", e.what());
        return false;
    }
}

bool Mali_BaState::LoadStateFromFile(const std::string& filename) {
    try {
        std::ifstream file(filename);
        if (!file.is_open()) {
            LOG_ERROR("❌ Could not open file for reading: ", filename);
            return false;
        }
        
        std::string json_content((std::istreambuf_iterator<char>(file)),
                                std::istreambuf_iterator<char>());
        file.close();
        
        LOG_INFO("📁 Loaded state from ", filename, " (", json_content.length(), " chars)");
        return SetStateFromJson(json_content);
        
    } catch (const std::exception& e) {
        LOG_ERROR("❌ LoadStateFromFile failed: ", e.what());
        return false;
    }
}

void Mali_BaState::ResetToInitialState() {
    LOG_INFO("🔄 Resetting to initial state");
    
    // Clear all state
    ClearAllState();
    
    // Reinitialize to initial state
    current_player_id_ = -1;  // Chance node initially
    current_player_color_ = PlayerColor::kEmpty;
    current_phase_ = Phase::kSetup;
    
    // Initialize hex_meeples_ with empty vectors for all valid hexes
    for (const auto& hex : GetGame()->GetValidHexes()) {
        hex_meeples_[hex] = std::vector<MeepleColor>();
    }
    
    // Regenerate initial meeple distribution using existing setup function
    ApplyChanceSetup();
    
    // Move to play phase after setup
    current_phase_ = Phase::kPlay;
    current_player_id_ = 0;  // Start with player 0
    current_player_color_ = GetPlayerColor(current_player_id_);
    
    ClearCaches();
    RefreshTerminalStatus();
    
    LOG_INFO("✅ Reset to initial state complete");
}

void Mali_BaState::ClearAllState() {
    player_token_locations_.clear();
    hex_meeples_.clear();
    trade_posts_locations_.clear();
    moves_history_.clear();
    trade_routes_.clear();
    
    // Clear resources
    for (auto& player_goods : common_goods_) {
        player_goods.clear();
    }
    for (auto& player_goods : rare_goods_) {
        player_goods.clear();
    }
    
    next_route_id_ = 1;
    ClearCaches();
}


}  // namespace mali_ba
}  // namespace open_spiel