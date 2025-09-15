// mali_ba_game.cc

#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/hex_grid.h"
#include "open_spiel/games/mali_ba/mali_ba_observer.h"

#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <stdexcept>
#include <map>
#include <algorithm>
#include <cctype>

#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "json.hpp"

using json = nlohmann::json;
static std::ios_base::Init g_ios_init;

namespace open_spiel
{
    namespace mali_ba
    {

        // Anonymous namespace for other helper functions
        namespace {
            // Helper function to trim whitespace from a string
            std::string trim(const std::string& str) {
                size_t first = str.find_first_not_of(" \t\n\r");
                if (std::string::npos == first) {
                    return str;
                }
                size_t last = str.find_last_not_of(" \t\n\r");
                return str.substr(first, (last - first + 1));
            }
           
            // Helper function to parse string values into typed GameParameters
            GameParameter ParseParameterValue(const std::string& value) {
                std::string lower_val = absl::AsciiStrToLower(value);
                if (lower_val == "true") return GameParameter(true);
                if (lower_val == "false") return GameParameter(false);

                try {
                    size_t pos;
                    int int_val = std::stoi(value, &pos);
                    if (pos == value.length()) return GameParameter(int_val);
                } catch (...) { /* Not an int, continue */ }

                // Try to parse as double
               try {
                    size_t pos;
                    double double_val = std::stod(value, &pos);
                    if (pos == value.length()) return GameParameter(double_val);
                } catch (...) { /* Not a double */ }

                // If all else fails, it's a string
                return GameParameter(value);
            }

            std::vector<PlayerType> ParsePlayerTypes(const std::string& player_types_str, int num_players) {
                std::vector<PlayerType> types;
                std::vector<std::string> parts = absl::StrSplit(player_types_str, ',');
                
                for (const auto& part : parts) {
                    std::string lower_part = absl::AsciiStrToLower(trim(part));
                    if (lower_part == "human") {
                        types.push_back(PlayerType::kHuman);
                    } else if (lower_part == "ai") {
                        types.push_back(PlayerType::kAI);
                    } else if (lower_part == "heuristic") {
                        types.push_back(PlayerType::kHeuristic);
                    } else {
                        SpielFatalError(absl::StrCat("Invalid player type: '", part, "'"));
                    }
                }
                
                if (types.size() != num_players) {
                    SpielFatalError(absl::StrCat(
                        "Number of player types (", types.size(), 
                        ") does not match number of players (", num_players, ")."));
                }
                
                return types;
            }

            // --- Definition of ParseIntVector is here, before it's used ---
            std::vector<int> ParseIntVector(const std::string& s) {
                std::vector<int> result;
                if (s.empty()) return result;
                std::vector<std::string> parts = absl::StrSplit(s, ',');
                for(const auto& part : parts) {
                    try {
                        result.push_back(std::stoi(trim(part)));
                    } catch (...) { /* ignore parsing errors for robustness */ }
                }
                return result;
            }

            // --- Helper to calculate the board radius needed to contain a particular
            // board made up of custom hexes ---
            int CalculateEffectiveRadius(const std::set<HexCoord>& hexes) {
                if (hexes.empty()) {
                    return 0; // Or a default radius like 3
                }
                int max_abs_coord = 0;
                for (const auto& hex : hexes) {
                    max_abs_coord = std::max({max_abs_coord, std::abs(hex.x), std::abs(hex.y), std::abs(hex.z)});
                }
                return max_abs_coord;
            }
        } // namespace

        // =====================================================================
        // --- GoodsManager Implementation ---
        // =====================================================================
        GoodsManager::GoodsManager() {
            // Use a set to gather unique good names, which also sorts them alphabetically
            std::set<std::string> common_goods_set;
            std::set<std::string> rare_goods_set;

            for (const auto& [id, details] : kCityDetailsMap) {
                common_goods_set.insert(details.common_good);
                rare_goods_set.insert(details.rare_good);
            }

            // Copy from sorted sets to the final vectors
            common_goods_list_.assign(common_goods_set.begin(), common_goods_set.end());
            rare_goods_list_.assign(rare_goods_set.begin(), rare_goods_set.end());

            // Create the fast lookup maps
            for (int i = 0; i < common_goods_list_.size(); ++i) {
                common_good_to_index_[common_goods_list_[i]] = i;
            }
            for (int i = 0; i < rare_goods_list_.size(); ++i) {
                rare_good_to_index_[rare_goods_list_[i]] = i;
            }

            // Sanity check
            SPIEL_CHECK_EQ(common_goods_list_.size(), 15);
            SPIEL_CHECK_EQ(rare_goods_list_.size(), 15);
        }

        int GoodsManager::GetCommonGoodIndex(const std::string& good_name) const {
            auto it = common_good_to_index_.find(good_name);
            return (it != common_good_to_index_.end()) ? it->second : -1;
        }

        int GoodsManager::GetRareGoodIndex(const std::string& good_name) const {
            auto it = rare_good_to_index_.find(good_name);
            return (it != rare_good_to_index_.end()) ? it->second : -1;
        }
        // =====================================================================

        const GameType kGameType{
            "mali_ba",
            "Mali-Ba Game",
            GameType::Dynamics::kSequential,
            GameType::ChanceMode::kExplicitStochastic,
            GameType::Information::kPerfectInformation,
            GameType::Utility::kGeneralSum,
            GameType::RewardModel::kTerminal,
            5,
            2,
            true,
            true,
            true,
            true,
            {
                {"players", GameParameter(3)},
                {"NumPlayers", GameParameter(3)}, // Add alias for INI file
                {"grid_radius", GameParameter(5)},
                {"tokens_per_player", GameParameter(3)},
                {"enable_move_logging", GameParameter(false)}, // Logging of each move state for replay mode
                {"LoggingEnabled", GameParameter(true)}, // Logging in general
                {"rng_seed", GameParameter(-1)},
                {"RngSeed", GameParameter(-1)}, // Add alias for INI file
                {"config_file", GameParameter(std::string(""))},
                {"posts_per_player", GameParameter(6)},
                {"free_action_trade_routes", GameParameter(true)},
                {"endgm_cond_numroutes", GameParameter(4)},
                {"endgm_cond_numrare_goods", GameParameter(4)},
                {"upgrade_cost_common", GameParameter(3)},
                {"upgrade_cost_rare", GameParameter(1)},
                {"city_free_upgrade", GameParameter(false)},
                {"endgm_req_numroutes", GameParameter(0)}, // Default to 0 (no requirement)
                //{"mancala_capture_rule", GameParameter(std::string("none"))},
                {"custom_hexes", GameParameter(std::string(""))}, // Add for INI
                {"custom_cities", GameParameter(std::string(""))}, // Add for INI
                {"prune_moves_for_ai", GameParameter(true)}, // Whether to reduce # legal moves for AI training
                {"player_types", GameParameter(std::string("ai,ai,ai"))}, // What type of player each is
            }
        };

        // =====================================================================
        // --- Factory ---
        // =====================================================================
        std::shared_ptr<const Game> Factory(const GameParameters &params) {
            return std::make_shared<Mali_BaGame>(params);
        }
        
        // =====================================================================
        // --- Register game ---
        // =====================================================================
        REGISTER_SPIEL_GAME(kGameType, Factory);

        // =====================================================================
        // --- Constructor ---
        // =====================================================================
        Mali_BaGame::Mali_BaGame(const GameParameters &params)
            : Game(kGameType, params)
        {
            LOG_INFO("Mali_BaGame: Initializing game...");

            GameParameters effective_params = params;
            std::string config_file_path = ParameterValue<std::string>("config_file");

            // Lambda function
            auto get_effective_param = [&](const std::string& key, auto default_value) {
                using T = decltype(default_value);
                auto it_effective = effective_params.find(key);
                if (it_effective != effective_params.end()) {
                    return it_effective->second.template value<T>();
                }
                return default_value;
            };
            // --- End of lambda ---

            // --- Board and City Parsing Logic ---
            bool custom_board_defined = false;
            std::map<std::string, std::string> board_section_params;
            std::map<std::string, std::string> heuristics_section_params;
            std::map<std::string, std::string> training_section_params; 

            if (!config_file_path.empty()) {
                LOG_DEBUG("Found 'config_file', attempting to parse: ", config_file_path);
                std::ifstream file(config_file_path);
                if (file.is_open()) {
                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    std::string file_contents = buffer.str();
                    file.close();

                    std::string current_section;
                    std::string custom_cities_raw_str;
                    // A map to temporarily hold region names from the INI ---
                    std::map<std::string, std::string> region_section_params;

                    std::istringstream content_stream(file_contents);
                    std::string line;
                    while (std::getline(content_stream, line)) {
                        line = trim(line);
                        if (line.empty() || line[0] == ';' || line[0] == '#') continue;

                        if (line[0] == '[' && line.back() == ']') {
                            current_section = trim(line.substr(1, line.length() - 2));
                            continue;
                        }
                        
                        auto eq_pos = line.find('=');
                        if (eq_pos != std::string::npos) {
                            std::string key = trim(line.substr(0, eq_pos));
                            std::string value_str = trim(line.substr(eq_pos + 1));
                            
                            if (current_section == "Board") {
                                board_section_params[key] = value_str;
                            } else if (current_section == "Cities" && absl::StartsWith(key, "city")) {
                                if (!custom_cities_raw_str.empty()) custom_cities_raw_str += ":";
                                custom_cities_raw_str += value_str;
                            // --- Handle the [Regions] section ---
                            } else if (current_section == "Regions") {
                                region_section_params[key] = value_str;
                            } else if (current_section == "Heuristics") { 
                                heuristics_section_params[key] = value_str;
                            } else if (current_section == "Training") {
                                training_section_params[key] = value_str;
                            } else {
                                effective_params[key] = ParseParameterValue(value_str);
                            }
                        }
                    }

                    // --- Process the parsed region names ---
                    region_id_to_name_map_.clear();
                    std::string region_prefix = "region";
                    for (const auto& [key, name] : region_section_params) {
                        if (absl::StartsWith(key, region_prefix)) {
                            std::string region_num_str = key.substr(region_prefix.length());
                            try {
                                int region_id = std::stoi(region_num_str);
                                region_id_to_name_map_[region_id] = name;
                                LOG_DEBUG("Loaded Region Name: ID ", region_id, " -> '", name, "'");
                            } catch (const std::exception& e) {
                                LOG_WARN("Could not parse region ID from key: ", key);
                            }
                        }
                    }                    
                    valid_hexes_.clear();
                    hex_to_region_map_.clear();
                    std::string prefix = "custom_hexes";

                    for (const auto& [key, value_str] : board_section_params) {
                        if (absl::StartsWith(key, prefix)) {
                            custom_board_defined = true;
                            int region_id = -1;
                            std::string region_num_str = key.substr(prefix.length());
                            if (!region_num_str.empty()) {
                                try {
                                    region_id = std::stoi(region_num_str);
                                } catch (const std::exception& e) {
                                    LOG_WARN("Could not parse region ID from key: ", key);
                                    continue;
                                }
                            } else {
                                region_id = 0;
                            }

                            LOG_DEBUG("Parsing hexes for region ", region_id, " from key '", key, "'");
                            std::set<HexCoord> region_hexes = ParseHexList(value_str);
                            
                            valid_hexes_.insert(region_hexes.begin(), region_hexes.end());

                            for (const auto& hex : region_hexes) {
                                if (hex_to_region_map_.count(hex)) {
                                    LOG_WARN("Hex ", hex.ToString(), " is defined in multiple regions. Overwriting with region ", region_id);
                                }
                                hex_to_region_map_[hex] = region_id;
                            }
                        } else if (key == "grid_radius") {
                            effective_params[key] = GameParameter(std::stoi(value_str));
                        }
                    }

                    if (!custom_cities_raw_str.empty()) {
                        effective_params["custom_cities"] = GameParameter(custom_cities_raw_str);
                    }
                    LOG_DEBUG("Successfully parsed INI file. Parameters have been overlaid.");
                } else {
                    LOG_WARN("Could not open INI file: ", config_file_path);
                }
            }

            // parse coastal_hexes
            if (board_section_params.count("coastal_hexes")) {
                std::string coastal_hexes_str = board_section_params["coastal_hexes"];
                coastal_hexes_ = ParseHexList(coastal_hexes_str);
                if (!coastal_hexes_.empty()) {
                    LOG_DEBUG("Loaded ", coastal_hexes_.size(), " coastal hexes from INI file.");
                }
            }

            // --- Initialization logic uses the defined lambda ---
            num_players_ = get_effective_param("NumPlayers", get_effective_param("players", 3));
            grid_radius_ = get_effective_param("grid_radius", 5);
            tokens_per_player_ = get_effective_param("tokens_per_player", 3);
            LoggingEnabled_ = get_effective_param("LoggingEnabled", get_effective_param("LoggingEnabled", true));
            //g_mali_ba_logging_enabled = LoggingEnabled_;
            enable_move_logging_ = get_effective_param("enable_move_logging_", get_effective_param("enable_move_logging", false));
            int seed_val = get_effective_param("RngSeed", get_effective_param("rng_seed", -1));
            prune_moves_for_ai_ = get_effective_param("prune_moves_for_ai", get_effective_param("prune_moves_for_ai", true));
            std::string player_types_str = get_effective_param("player_types", std::string("ai,ai,ai"));
            player_types_ = ParsePlayerTypes(player_types_str, num_players_);

            rng_seed_ = (seed_val == -1)
                            ? static_cast<std::mt19937::result_type>(
                                std::chrono::high_resolution_clock::now().time_since_epoch().count())
                            : static_cast<std::mt19937::result_type>(seed_val);
            
            LOG_DEBUG("DEBUG: rng_seed_ = ", rng_seed_, "seed_val = ", seed_val);
            LOG_DEBUG("Mali_BaGame: Configuring game from effective parameters...");
            
            // Check if the INI file included hexes to create a custom board.  If not, make a hexagonal
            // board.  If there are hexes for a custom board, make that board and also calculate the
            // radius of a regular hexagonal board that would contain the custom board. This info will 
            // be used when building the observation tensor.
            if (!custom_board_defined) {
                LOG_WARN("No regional 'custom_hexesX' found. Generating regular board with radius: ", grid_radius_);
                valid_hexes_ = GenerateRegularBoard(grid_radius_);
            } else {
                // If a custom board was defined, we calculate its true radius.
                int effective_radius = CalculateEffectiveRadius(valid_hexes_);
                
                LOG_INFO("Mali_BaGame: Custom board detected. Calculating effective radius...");
                LOG_INFO("Mali_BaGame:   INI grid_radius (will be ignored): ", grid_radius_);
                LOG_INFO("Mali_BaGame:   Calculated effective radius: ", effective_radius);
                
                // Overwrite the member variable with the calculated radius.
                // This is now the single source of truth for the board's dimensions.
                grid_radius_ = effective_radius;

                LOG_DEBUG("Constructed board from ", hex_to_region_map_.size(), " hexes across custom regions.");
            }

            std::string custom_cities_str = get_effective_param("custom_cities", std::string(""));
            if (!custom_cities_str.empty()) {
                LOG_DEBUG("Found 'custom_cities' parameter. Parsing custom cities.");
                ParseCustomCitiesFromString(custom_cities_str);
            } else {
                LOG_DEBUG("No 'custom_cities' found. Generating default cities.");
                cities_ = GetDefaultCitiesWithTimbuktu();
            }

            LOG_DEBUG("Player types: ", player_types_str); // Add logging for verification
            
            // --- Rule Loading ---
            // rules_.declare_trade_route_is_free_action = get_effective_param("declare_trade_route_is_free_action", true);
            rules_.mancala_post_requires_meeple = get_effective_param("mancala_post_requires_meeple", true);

            rules_.income_center_in_city_rare = get_effective_param("income_center_in_city_rare", 1);
            rules_.income_center_connected_common = get_effective_param("income_center_connected_common", 2);
            rules_.income_center_connected_rare = get_effective_param("income_center_connected_rare", 1);
            rules_.income_center_isolated_common = get_effective_param("income_center_isolated_common", 2);
            rules_.income_post_common = get_effective_param("income_post_common", 1);
            
            rules_.upgrade_cost_common = get_effective_param("upgrade_cost_common", 3);
            rules_.upgrade_cost_rare = get_effective_param("upgrade_cost_rare", 1);
            rules_.remove_meeple_on_upgrade = get_effective_param("remove_meeple_on_upgrade", true);
            rules_.remove_meeple_on_trade_route = get_effective_param("remove_meeple_on_trade_route", true);
            rules_.city_free_upgrade = get_effective_param("city_free_upgrade", false);
            rules_.posts_per_player = get_effective_param("posts_per_player", 6);
            rules_.free_action_trade_routes = get_effective_param("free_action_trade_routes", true);
            rules_.non_city_center_limit_divisor = get_effective_param("non_city_center_limit_divisor", 1);
            rules_.min_hexes_for_trade_route = get_effective_param("min_hexes_for_trade_route", 3);
            rules_.max_shared_centers_between_routes = get_effective_param("max_shared_centers_between_routes", 2);

            rules_.end_game_cond_num_routes = get_effective_param("end_game_cond_num_routes", -1);
            rules_.end_game_cond_num_rare_goods = get_effective_param("end_game_cond_num_rare_goods", 5);
            rules_.end_game_req_num_routes = get_effective_param("end_game_req_num_routes", 2);
            rules_.end_game_cond_timbuktu_to_coast = get_effective_param("end_game_cond_timbuktu_to_coast", true);
            rules_.end_game_cond_rare_good_each_region = get_effective_param("end_game_cond_rare_good_each_region", false);
            rules_.end_game_cond_rare_good_num_regions = get_effective_param("end_game_cond_rare_good_num_regions", 5);
            // But if we found no coastal cities, switch to false
            if (coastal_hexes_.empty()) { rules_.end_game_cond_timbuktu_to_coast = false; }
            // rules_.timbuktu_city_id = get_effective_param("timbuktu_city_id", 15);

            // Scoring rules with defaults
            rules_.score_longest_routes = ParseIntVector(get_effective_param("score_longest_routes", std::string("11,7,4")));
            rules_.score_region_control = ParseIntVector(get_effective_param("score_region_control", std::string("11,7,4")));

            LOG_DEBUG("Mali_BaGame: Rules fully loaded from INI / defaults.");
            LOG_DEBUG("Mali_BaGame: Rules loaded:");
            // LOG_DEBUG("  - Free Action Trade Routes: ", rules_.declare_trade_route_is_free_action);
            LOG_DEBUG("  - Free Action Trade Routes: ", rules_.free_action_trade_routes);
            LOG_DEBUG("  - Mancala Post Requires Meeple: ", rules_.mancala_post_requires_meeple);
            LOG_DEBUG("  - Income Center in City (Rare): ", rules_.income_center_in_city_rare);
            LOG_DEBUG("  - Income Center Connected (Common): ", rules_.income_center_connected_common);
            LOG_DEBUG("  - Income Center Connected (Rare): ", rules_.income_center_connected_rare);
            LOG_DEBUG("  - Income Center Isolated (Common): ", rules_.income_center_isolated_common);
            LOG_DEBUG("  - Income Post (Common): ", rules_.income_post_common);
            LOG_DEBUG("  - Upgrade Cost (Common/Rare): ", rules_.upgrade_cost_common, "/", rules_.upgrade_cost_rare);
            LOG_DEBUG("  - Remove Meeple on Upgrade: ", rules_.remove_meeple_on_upgrade);
            LOG_DEBUG("  - Remove Meeple on Trade Route: ", rules_.remove_meeple_on_trade_route);
            LOG_DEBUG("  - City Free Upgrade on Route: ", rules_.city_free_upgrade);
            LOG_DEBUG("  - Posts Per Player: ", rules_.posts_per_player);
            LOG_DEBUG("  - Non-City Center Limit Divisor: ", rules_.non_city_center_limit_divisor);
            LOG_DEBUG("  - Min Hexes for Trade Route: ", rules_.min_hexes_for_trade_route);
            LOG_DEBUG("  - Max Shared Centers Between Routes: ", rules_.max_shared_centers_between_routes);
            LOG_DEBUG("  - Trade Routes Required to End Game: ", rules_.end_game_req_num_routes);
            LOG_DEBUG("  - Trade Routes that Trigger End-game: ", rules_.end_game_cond_num_routes);
            LOG_DEBUG("  - Rare Goods for End-game Trigger: ", rules_.end_game_cond_num_rare_goods);
            LOG_DEBUG("  - Timbuktu to Coast Ends Game: ", rules_.end_game_cond_timbuktu_to_coast);
            LOG_DEBUG("  - Rare Good Each Region Ends Game: ", rules_.end_game_cond_rare_good_each_region);
            LOG_DEBUG("  - Rare Good Num of Regions Ends Game: ", rules_.end_game_cond_rare_good_num_regions);
            // LOG_DEBUG("  - Timbuktu City ID: ", rules_.timbuktu_city_id);
            // LOG_DEBUG("  - Mancala Capture Rule: ", rules_.mancala_capture_rule);
            // Log scoring arrays (convert vectors to string representation)
            std::string longest_routes_str = "[";
            for (size_t i = 0; i < rules_.score_longest_routes.size(); ++i) {
                if (i > 0) longest_routes_str += ",";
                longest_routes_str += std::to_string(rules_.score_longest_routes[i]);
            }
            longest_routes_str += "]";
            LOG_DEBUG("  - Score Longest Routes: ", longest_routes_str);

            std::string region_control_str = "[";
            for (size_t i = 0; i < rules_.score_region_control.size(); ++i) {
                if (i > 0) region_control_str += ",";
                region_control_str += std::to_string(rules_.score_region_control[i]);
            }
            region_control_str += "]";
            LOG_DEBUG("  - Score Region Control: ", region_control_str);

            // --- HEURISTIC WEIGHT LOADING ---
            if (!heuristics_section_params.empty()) {
                LOG_DEBUG("Mali_BaGame: Loading heuristic weights from INI...");
                auto get_heuristic_param = [&](const std::string& key, double default_value) {
                    auto it = heuristics_section_params.find(key);
                    if (it != heuristics_section_params.end()) {
                        try {
                            return std::stod(it->second);
                        } catch(...) { return default_value; }
                    }
                    return default_value;
                };

                heuristic_weights_.weight_pass = get_heuristic_param("weight_pass", 0.1);
                heuristic_weights_.weight_mancala = get_heuristic_param("weight_mancala", 10.0);
                heuristic_weights_.weight_upgrade = get_heuristic_param("weight_upgrade", 15.0);
                heuristic_weights_.weight_income = get_heuristic_param("weight_income", 5.0);
                heuristic_weights_.weight_place_token = get_heuristic_param("weight_place_token", 5.0);
                heuristic_weights_.weight_trade_route_create = get_heuristic_param("weight_trade_route_create", 10.0);

                heuristic_weights_.bonus_mancala_city_end = get_heuristic_param("bonus_mancala_city_end", 30.0);
                heuristic_weights_.bonus_mancala_long_distance = get_heuristic_param("bonus_mancala_long_distance", 10.0);
                heuristic_weights_.bonus_mancala_meeple_density = get_heuristic_param("bonus_mancala_meeple_density", 15.0);
                heuristic_weights_.bonus_upgrade_diversity_factor = get_heuristic_param("bonus_upgrade_diversity_factor", 5.0);
                heuristic_weights_.bonus_upgrade_new_region = get_heuristic_param("bonus_upgrade_new_region", 20.0);

                heuristic_weights_.bonus1 = get_heuristic_param("bonus1", 25.0);
                heuristic_weights_.bonus2 = get_heuristic_param("bonus2", 0.0);
                heuristic_weights_.bonus3 = get_heuristic_param("bonus3", 0.0);
                heuristic_weights_.bonus4 = get_heuristic_param("bonus4", 0.0);
                LOG_DEBUG("Mali_BaGame: Heuristic weights loaded.");
            } else {
                LOG_DEBUG("Mali_BaGame: Using default heuristic weights.");
            }
            LOG_DEBUG("  - Mancala: ", heuristic_weights_.weight_mancala);
            LOG_DEBUG("  - Upgrade: ", heuristic_weights_.weight_upgrade);
            LOG_DEBUG("  - Income: ", heuristic_weights_.weight_income);
            LOG_DEBUG("  - Create Route: ", heuristic_weights_.weight_trade_route_create);
            LOG_DEBUG("  - Upgrade Diversity Factor: ", heuristic_weights_.bonus_upgrade_diversity_factor);
            LOG_DEBUG("  - Upgrade New Region Bonus: ", heuristic_weights_.bonus_upgrade_new_region);

            LOG_DEBUG("  - Bonus 1: ", heuristic_weights_.bonus1);
            LOG_DEBUG("  - Bonus 2: ", heuristic_weights_.bonus2);
            LOG_DEBUG("  - Bonus 3: ", heuristic_weights_.bonus3);
            LOG_DEBUG("  - Bonus 4: ", heuristic_weights_.bonus4);

            // --- Parse Training Parameters ---
            if (!training_section_params.empty()) {
                LOG_DEBUG("Parsing [Training] section from INI file...");
                
                auto parse_double = [&](const std::string& key, double default_val) {
                    auto it = training_section_params.find(key);
                    if (it != training_section_params.end()) {
                        try {
                            return std::stod(it->second);
                        } catch (...) {
                            LOG_WARN("Failed to parse training parameter: ", key, " = ", it->second, ". Using default: ", default_val);
                            return default_val;
                        }
                    }
                    return default_val;
                };
                
                auto parse_int = [&](const std::string& key, int default_val) {
                    auto it = training_section_params.find(key);
                    if (it != training_section_params.end()) {
                        try {
                            return std::stoi(it->second);
                        } catch (...) {
                            LOG_WARN("Failed to parse training parameter: ", key, " = ", it->second, ". Using default: ", default_val);
                            return default_val;
                        }
                    }
                    return default_val;
                };
                
                training_params_.time_penalty = parse_double("time_penalty", -0.0035);
                training_params_.draw_penalty = parse_double("draw_penalty", 0.0);
                training_params_.max_moves_penalty = parse_double("max_moves_penalty", -0.5);
                training_params_.loss_penalty = parse_double("loss_penalty", 0.0);
                training_params_.upgrade_reward = parse_double("upgrade_reward", 0.02);
                training_params_.trade_route_reward = parse_double("trade_route_reward", 0.04);
                training_params_.new_rare_region_reward = parse_double("new_rare_region_reward", 0.08);
                training_params_.new_common_good_reward = parse_double("new_common_good_reward", 0.02);
                training_params_.key_location_post_reward = parse_double("key_location_post_reward", 0.03);
                training_params_.quick_win_bonus = parse_double("quick_win_bonus", 0.2);
                training_params_.quick_win_threshold = parse_int("quick_win_threshold", 150);
                
                LOG_INFO("Training parameters loaded from INI:");
                LOG_INFO("  time_penalty: ", training_params_.time_penalty);
                LOG_INFO("  draw_penalty: ", training_params_.draw_penalty);
                LOG_INFO("  max_moves_penalty: ", training_params_.max_moves_penalty);
                LOG_INFO("  loss_penalty: ", training_params_.loss_penalty);
                LOG_INFO("  upgrade_reward: ", training_params_.upgrade_reward);
                LOG_INFO("  trade_route_reward: ", training_params_.trade_route_reward);
                LOG_INFO("  new_rare_region_reward: ", training_params_.new_rare_region_reward);
                LOG_INFO("  new_common_good_reward: ", training_params_.new_common_good_reward);
                LOG_INFO("  key_location_post_reward: ", training_params_.key_location_post_reward);
                LOG_INFO("  quick_win_bonus: ", training_params_.quick_win_bonus);
                LOG_INFO("  quick_win_threshold: ", training_params_.quick_win_threshold);
            }


            // Assign player colors based on number of players
            if (num_players_ > 0) player_colors_.push_back(PlayerColor::kRed);
            if (num_players_ > 1) player_colors_.push_back(PlayerColor::kGreen);
            if (num_players_ > 2) player_colors_.push_back(PlayerColor::kBlue);
            if (num_players_ > 3) player_colors_.push_back(PlayerColor::kViolet);
            if (num_players_ > 4) player_colors_.push_back(PlayerColor::kPink);

            // Dynamically build the observation tensor
            int dimension = grid_radius_ * 2 + 1;
            constexpr int kNumPlanes = 77; // see mali_ba_observer.cc for info
            observation_tensor_shape_ = {kNumPlanes, dimension, dimension};

            LOG_INFO("Mali_BaGame: Dynamically configured observation tensor shape to: {",
                    observation_tensor_shape_[0], ", ",
                    observation_tensor_shape_[1], ", ",
                    observation_tensor_shape_[2], "} based on effective grid radius of ",
                    grid_radius_);


            LOG_INFO("Mali_BaGame: Final configuration complete.");
            InitializeLookups();
        }

        // =======================================================================
        // Other game object functions
        // =======================================================================
        std::unique_ptr<State> Mali_BaGame::NewInitialStateForPopulation(int population) const {
            return NewInitialState();
        }

        int Mali_BaGame::MaxChanceOutcomes() const { return 1; }

        std::unique_ptr<State> Mali_BaGame::NewInitialState() const {
            auto state = std::make_unique<Mali_BaState>(shared_from_this());
            if (enable_move_logging_) {
                state->InitializeMoveLogging();
            }
            return state;
        }

        std::unique_ptr<State> Mali_BaGame::NewInitialState(const std::string &str) const {
            return DeserializeState(str);
        }
        
        std::shared_ptr<Observer> Mali_BaGame::MakeObserver(
            absl::optional<IIGObservationType> iig_obs_type,
            const GameParameters &params) const
        {
            return MakeMaliBaObserver(iig_obs_type.value_or(IIGObservationType{}));
        }

        const City *Mali_BaGame::GetCityAt(const HexCoord &location) const {
            for (const auto& city : cities_) if (city.location == location) return &city;
            return nullptr;
        }

        void Mali_BaGame::InitializeLookups() { 
            coord_to_index_map_.clear();
            index_to_coord_vec_.clear();
            int current_index = 0;
            for (const HexCoord &hex : valid_hexes_) {
                coord_to_index_map_[hex] = current_index;
                index_to_coord_vec_.push_back(hex);
                current_index++;
            }
            num_hexes_ = current_index;
        }

        int Mali_BaGame::CoordToIndex(const HexCoord &hex) const { 
            auto it = coord_to_index_map_.find(hex);
            return (it != coord_to_index_map_.end()) ? it->second : -1;
        }

        HexCoord Mali_BaGame::IndexToCoord(int index) const {
            SPIEL_CHECK_GE(index, 0);
            SPIEL_CHECK_LT(index, num_hexes_);
            return index_to_coord_vec_[index];
        }

        std::set<HexCoord> Mali_BaGame::GenerateRegularBoard(int radius) const {
            std::set<HexCoord> hexes;
            for (int x = -radius; x <= radius; ++x) {
                for (int y = -radius; y <= radius; ++y) {
                    int z = -x - y;
                    if (std::max({std::abs(x), std::abs(y), std::abs(z)}) <= radius) {
                        hexes.insert(HexCoord(x, y, z));
                    }
                }
            }
            return hexes;
        }

        // --- Parsing helpers moved from BoardConfig ---
        std::string Mali_BaGame::StripWhitespace(const std::string& str) const {
            size_t start = str.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) return "";
            size_t end = str.find_last_not_of(" \t\r\n");
            return str.substr(start, end - start + 1);
        }

        std::vector<std::string> Mali_BaGame::SplitString(const std::string& str, char delimiter) const {
            std::vector<std::string> result;
            std::stringstream ss(str);
            std::string item;
            while (std::getline(ss, item, delimiter)) {
                if (!item.empty()) result.push_back(item);
            }
            return result;
        }

        HexCoord Mali_BaGame::ParseHexCoord(const std::string& coord_str) const {
            std::vector<std::string> parts = SplitString(coord_str, ',');
            if (parts.size() != 3) {
                throw std::invalid_argument("Invalid coordinate format: " + coord_str);
            }
            int x = std::stoi(StripWhitespace(parts[0]));
            int y = std::stoi(StripWhitespace(parts[1]));
            int z = std::stoi(StripWhitespace(parts[2]));
            return HexCoord(x, y, z);
        }
        
        std::set<HexCoord> Mali_BaGame::ParseHexList(const std::string& hex_string) const {
            std::set<HexCoord> hexes;
            // Use absl::StrSplit with a Delimiter that skips empty parts.
            std::vector<std::string> coords_list = absl::StrSplit(hex_string, ':', absl::SkipEmpty());
            
            for (const std::string& coord_str : coords_list) {
                // `coord_str` will already be trimmed if there was no extra whitespace.
                try {
                    HexCoord hex = ParseHexCoord(coord_str);
                    if (hex.x + hex.y + hex.z == 0) {
                        hexes.insert(hex);
                    } else {
                        LOG_WARN("Invalid hex coordinate ", coord_str, " (x+y+z != 0)");
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("Could not parse hex coordinate: ", coord_str);
                }
            }
            return hexes;
        }


        int Mali_BaGame::FindCityIdByName(const std::string& name) const {
            for (const auto& [id, details] : kCityDetailsMap) {
                if (StrLower(details.name) == StrLower(name)) {
                    return id;
                }
            }
            return -1;
        }
        
        void Mali_BaGame::ParseCustomCitiesFromString(const std::string& cities_str) {
            cities_.clear();
            // Use absl::StrSplit for robustness
            std::vector<std::string> city_entries = absl::StrSplit(cities_str, ':', absl::SkipEmpty());

            for (const auto& entry : city_entries) {
                if (entry.empty()) continue;
                // Parse "name,x,y,z"
                std::vector<std::string> parts = absl::StrSplit(entry, ',');
                if (parts.size() >= 4) {
                    try {
                        std::string name = trim(parts[0]);
                        int city_id = FindCityIdByName(name);
                        if (city_id != -1) {
                            int x = std::stoi(trim(parts[1]));
                            int y = std::stoi(trim(parts[2]));
                            int z = std::stoi(trim(parts[3]));
                            const auto& details = kCityDetailsMap.at(city_id);
                            cities_.emplace_back(details.id, details.name, details.culture, 
                                                HexCoord(x, y, z), details.common_good, details.rare_good);
                        } else {
                            LOG_WARN("City '", name, "' not found in custom_cities string.");
                        }
                    } catch (const std::exception& e) {
                        LOG_ERROR("Could not parse city entry '", entry, "': ", e.what());
                    }
                }
            }
        }
        
        std::vector<City> Mali_BaGame::GetDefaultCitiesWithTimbuktu() const {
            std::vector<City> cities;
            auto timbuktu_it = kCityDetailsMap.find(15);
            if (timbuktu_it != kCityDetailsMap.end()) {
                const auto& details = timbuktu_it->second;
                cities.emplace_back(details.id, details.name, details.culture, 
                                   HexCoord(0, 0, 0), details.common_good, details.rare_good);
            }
            
            std::vector<int> available_ids;
            for (const auto& [id, details] : kCityDetailsMap) if (id != 15) available_ids.push_back(id);
            
            std::set<HexCoord> valid_hexes = GenerateRegularBoard(grid_radius_);
            valid_hexes.erase(HexCoord(0, 0, 0));
            
            std::mt19937 gen(rng_seed_);
            std::shuffle(available_ids.begin(), available_ids.end(), gen);
            
            std::vector<HexCoord> positions(valid_hexes.begin(), valid_hexes.end());
            std::shuffle(positions.begin(), positions.end(), gen);
            
            int count = std::min({3, (int)available_ids.size(), (int)positions.size()});
            for (int i = 0; i < count; ++i) {
                auto city_it = kCityDetailsMap.find(available_ids[i]);
                if (city_it != kCityDetailsMap.end()) {
                    const auto& details = city_it->second;
                    cities.emplace_back(details.id, details.name, details.culture, 
                                       positions[i], details.common_good, details.rare_good);
                }
            }
            return cities;
        }

        // --- Getter Implementation for what region a hex is in ---
        int Mali_BaGame::GetRegionForHex(const HexCoord& hex) const {
            auto it = hex_to_region_map_.find(hex);
            if (it != hex_to_region_map_.end()) {
                return it->second;
            }
            return -1; // Return -1 if the hex is not found in any region
        }

        // --- Implementation of the region name getter ---
        std::string Mali_BaGame::GetRegionName(int region_id) const {
            auto it = region_id_to_name_map_.find(region_id);
            if (it != region_id_to_name_map_.end()) {
                return it->second;
            }
            // Return a default/error string if the ID is not found
            return "Unknown Region " + std::to_string(region_id);
        }

        // --- Implementation of the region IDs getter ---
        std::vector<int> Mali_BaGame::GetValidRegionIds() const {
            std::vector<int> region_ids;
            region_ids.reserve(region_id_to_name_map_.size());
            
            for (const auto& [region_id, region_name] : region_id_to_name_map_) {
                region_ids.push_back(region_id);
            }
            // Sort to ensure consistent iteration order
            std::sort(region_ids.begin(), region_ids.end());
            return region_ids;
        }

        std::unique_ptr<State> Mali_BaGame::DeserializeState(const std::string &str) const
        {
            // The deserialization logic remains largely the same, as it only deals with
            // the dynamic parts of the state. The state object itself will get the static
            // board info from the Game object it's constructed with.
            if (str.empty()) return NewInitialState();
            std::unique_ptr<Mali_BaState> state = std::make_unique<Mali_BaState>(shared_from_this());
            try {
                json j = json::parse(str);
                
                state->current_player_id_ = j.at("currentPlayerId").get<int>();
                state->SetCurrentPhase(static_cast<Phase>(j.at("currentPhase").get<int>()));

                if (state->CurrentPhase() == Phase::kSetup) {
                    state->current_player_id_ = kChancePlayerId;
                    state->current_player_color_ = PlayerColor::kEmpty;
                } else {
                    SPIEL_CHECK_GE(state->current_player_id_, 0);
                    SPIEL_CHECK_LT(state->current_player_id_, NumPlayers());
                    state->current_player_color_ = state->GetPlayerColor(state->current_player_id_);
                }
                
                if (j.contains("playerTokens")) {
                    for (auto const &[hex_str, color_val] : j.at("playerTokens").items()) {
                        if (auto hex = JsonStringToHexCoord(hex_str)) {
                            if (color_val.is_array()) {
                                std::vector<PlayerColor> colors;
                                for (const auto& color_int : color_val) {
                                    colors.push_back(static_cast<PlayerColor>(color_int.get<int>()));
                                }
                                if (!colors.empty()) {
                                    state->player_token_locations_[*hex] = colors;
                                }
                            }
                        }
                    }
                }

                if (j.contains("hexMeeples")) {
                    for (auto const &[hex_str, j_list] : j.at("hexMeeples").items()) {
                         if (auto hex = JsonStringToHexCoord(hex_str)) {
                             for (const auto &mc_val : j_list) state->hex_meeples_[*hex].push_back(static_cast<MeepleColor>(mc_val.get<int>()));
                         }
                    }
                }
                if (j.contains("tradePosts")) {
                    for (auto const &[hex_str, j_list] : j.at("tradePosts").items()) {
                         if (auto hex = JsonStringToHexCoord(hex_str)) {
                             for (const auto &j_post : j_list) {
                                 PlayerColor owner = static_cast<PlayerColor>(j_post.at("owner").get<int>());
                                 TradePostType type = static_cast<TradePostType>(j_post.at("type").get<int>());
                                 state->trade_posts_locations_[*hex].push_back({owner, type});
                             }
                         }
                    }
                }
                if (j.contains("playerPostsSupply")) {
                    state->player_posts_supply_ = j.at("playerPostsSupply").get<std::vector<int>>();
                }

                if (j.contains("history")) {
                    for (const auto &j_move : j.at("history")) {
                        Move move;
                        move.player = static_cast<PlayerColor>(j_move.at("player").get<int>());
                        if (auto hex = JsonStringToHexCoord(j_move.at("startHex").get<std::string>())) move.start_hex = *hex;
                        move.type = static_cast<ActionType>(j_move.at("type").get<int>());
                        move.place_trading_post = j_move.at("placePost").get<bool>();
                        if (j_move.contains("path")) {
                            for (const auto &hex_str_val : j_move.at("path")) {
                                if (auto hex = JsonStringToHexCoord(hex_str_val.get<std::string>())) move.path.push_back(*hex);
                            }
                        }
                        state->moves_history_.push_back(move);
                    }
                }
                if (j.contains("commonGoods")) state->SetCommonGoods(j.at("commonGoods"));
                if (j.contains("rareGoods")) state->SetRareGoods(j.at("rareGoods"));
                if (j.contains("tradeRoutes")) {
                     for (const auto& j_route : j.at("tradeRoutes")) {
                        TradeRoute route;
                        route.id = j_route.at("id").get<int>();
                        route.owner = static_cast<PlayerColor>(j_route.at("owner").get<int>());
                        if(j_route.contains("hexes")) for(const auto& h_str : j_route.at("hexes")) if(auto h=JsonStringToHexCoord(h_str)) route.hexes.push_back(*h);
                        if(j_route.contains("goods")) route.goods = j_route.at("goods").get<std::map<std::string, int>>();
                        route.active = j_route.value("active", true);
                        route.hexes = state->GetCanonicalRoute(route.hexes); // make sure hexes are sorted properly
                        state->trade_routes_.push_back(route);
                        if (route.id >= state->next_route_id_) state->next_route_id_ = route.id + 1;
                    }
                }

            } catch (const std::exception &e) {
                LOG_ERROR(absl::StrCat("Deserialize ERROR: ", e.what()));
                return NewInitialState();
            }

            state->ClearCaches();
            state->RefreshTerminalStatus();
            return state;
        }


        // =======================================================================
        // definitions for global logging variables and functions
        // =======================================================================
        // bool g_mali_ba_logging_enabled = true;
        // LogLevel g_mali_ba_log_level = LogLevel::kInfo;
        // std::string datetime = GetCurrentDateTime();
        // std::string g_log_file_path = "/tmp/mali_ba." + datetime + ".log";

        // std::string LogLevelToString(LogLevel level) {
        //     switch (level) {
        //         case LogLevel::kDebug:   return "DEBUG";
        //         case LogLevel::kInfo:    return "INFO ";
        //         case LogLevel::kWarning: return "WARN ";
        //         case LogLevel::kError:   return "ERROR";
        //         default:                 return "UNKWN";
        //     }
        // }
        
        // void LogMBCore(LogLevel level, const std::string& message, bool print_to_terminal,
        //             const char* file, int line) {
        //     if (!g_mali_ba_logging_enabled || level < g_mali_ba_log_level) {
        //         return;
        //     }

        //     auto now = std::chrono::system_clock::now();
        //     auto in_time_t = std::chrono::system_clock::to_time_t(now);
            
        //     std::string file_str(file);
        //     size_t last_slash = file_str.find_last_of('/');
        //     if (last_slash != std::string::npos) {
        //         file_str = file_str.substr(last_slash + 1);
        //     }

        //     std::stringstream log_stream;
        //     log_stream << std::put_time(std::localtime(&in_time_t), "%H:%M:%S") << " "
        //             << "[" << LogLevelToString(level) << "] "
        //             << "[" << file_str << ":" << line << "] "
        //             << message;

        //     // Print the message to the terminal
        //     if (print_to_terminal) {
        //         fprintf(stderr, "%s\n", log_stream.str().c_str());
        //     }
        //     // Print the message to the log file
        //     std::ofstream log_file(g_log_file_path, std::ios_base::app);
        //     if (log_file.is_open()) {
        //         log_file << log_stream.str() << std::endl;
        //     }
        // }

        // =======================================================================
        // ostream operator definitions
        // =======================================================================
        std::ostream& operator<<(std::ostream& os, const Phase& phase) {
            switch (phase) {
                case Phase::kEmpty:     os << "Empty"; break;
                case Phase::kSetup:     os << "Setup"; break;
                case Phase::kPlaceToken:os << "PlaceToken"; break;
                case Phase::kPlay:      os << "Play"; break;
                case Phase::kEndRound:  os << "EndRound"; break;
                case Phase::kGameOver:  os << "GameOver"; break;
                default: os << "UnknownPhase(" << static_cast<int>(phase) << ")";
            }
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const ActionType& move_type) {
            switch (move_type) {
                case ActionType::kInvalid:          os << "Invalid"; break;
                case ActionType::kPass:             os << "Pass"; break;
                case ActionType::kChanceSetup:      os << "ChanceSetup"; break;
                case ActionType::kPlaceToken:       os << "PlaceToken"; break;
                case ActionType::kMancala:          os << "Mancala"; break;
                case ActionType::kPlaceTCenter:     os << "PlaceTCenter"; break;
                case ActionType::kIncome:           os << "Income"; break;
                case ActionType::kTradeRouteCreate: os << "TradeRouteCreate"; break;
                case ActionType::kTradeRouteUpdate: os << "TradeRouteUpdate"; break;
                case ActionType::kTradeRouteDelete: os << "TradeRouteDelete"; break;
                default: os << "UnknownActionType(" << static_cast<int>(move_type) << ")";
            }
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const PlayerColor& pc) {
            os << PlayerColorToString(pc);
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const MeepleColor& mc) {
            os << MeepleColorToString(mc);
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const TradePostType& tpt) {
            switch (tpt) {
                case TradePostType::kNone:   os << "None"; break;
                case TradePostType::kPost:   os << "Post"; break;
                case TradePostType::kCenter: os << "Center"; break;
                default: os << "UnknownTradePostType";
            }
            return os;
        }

    } // namespace mali_ba
} // namespace open_spiel