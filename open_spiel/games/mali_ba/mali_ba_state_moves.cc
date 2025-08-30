// mali_ba_state_moves.cc
// Mancala move generation and pathfinding

#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/games/mali_ba/hex_grid.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/numbers.h"

#include <set>
#include <vector>
#include <iostream>
#include <queue>
// Include the nlohmann/json header
#include "json.hpp"
// For convenience
using json = nlohmann::json;

namespace open_spiel
{
    namespace mali_ba
    {
        // Static member definitions for move logging
        std::unique_ptr<std::ofstream> Mali_BaState::move_log_file_ = nullptr;
        std::string Mali_BaState::move_log_filename_ = "";
        int Mali_BaState::move_count_ = 0;
        bool Mali_BaState::move_logging_enabled_ = false;
        bool Mali_BaState::move_logging_initialized_ = false;

        Action Mali_BaState::MoveToAction(const Move& move) const {
            int num_hexes = GetGame()->NumHexes();
            Action action = kInvalidAction;

            switch (move.type) {
                case ActionType::kPass:
                    action = kPassAction;
                    break;
                case ActionType::kIncome:
                    action = kIncomeAction;
                    break;
                case ActionType::kPlaceTCenter: { // Upgrade
                    int hex_index = GetGame()->CoordToIndex(move.start_hex);
                    if (hex_index != -1) {
                        action = kUpgradeActionBase + hex_index;
                    }
                    break;
                }
                case ActionType::kMancala: {
                    if (move.path.empty()) break;
                    int start_idx = GetGame()->CoordToIndex(move.start_hex);
                    int end_idx = GetGame()->CoordToIndex(move.path[0]);
                    if (start_idx != -1 && end_idx != -1) {
                        action = kMancalaActionBase + start_idx * kMaxHexes + end_idx;
                    }
                    break;
                }
                case ActionType::kTradeRouteCreate: {
                    if (move.route_id >= 0) {
                        action = kTradeRouteCreateBase + move.route_id;
                    }
                    break;
                }
                default:
                    return kInvalidAction;
            }

            // If a base action was successfully created, add flags.
            // This correctly handles adding multiple flags for super-compound moves.
            if (action != kInvalidAction) {
                if (move.place_trading_post) {
                    action += kPlacePostFlag;
                }
                if (move.declares_trade_route) {
                    action += kDeclareRouteFlag;
                }
            }

            return action;
        }

        Move Mali_BaState::ActionToMove(Action action) const {
            Move move;
            move.player = current_player_color_;
            int num_hexes = GetGame()->NumHexes();

            if (action >= kDeclareRouteFlag) {
                move.declares_trade_route = true;
                action -= kDeclareRouteFlag;
            }
            if (action >= kPlacePostFlag) {
                move.place_trading_post = true;
                action -= kPlacePostFlag;
            }

            if (action >= kPlaceTokenActionBase && action < kUpgradeActionBase) {
                move.type = ActionType::kPlaceToken;
                int hex_index = action - kPlaceTokenActionBase;
                
                // Safety check
                if (hex_index >= 0 && hex_index < GetGame()->NumHexes()) {
                    move.start_hex = GetGame()->IndexToCoord(hex_index);
                } else {
                    move.type = ActionType::kInvalid;
                }
                return move;
            }

            if (action == kPassAction) {
                move.type = ActionType::kPass;
                return move;
            }
            if (action == kIncomeAction) {
                move.type = ActionType::kIncome;
                move.action_string = "income";
                return move;
            }

            // decode place_token actions
            if (action >= kPlaceTokenActionBase && action < kUpgradeActionBase) {
                move.type = ActionType::kPlaceToken;
                int hex_index = action - kPlaceTokenActionBase;
                if (hex_index >= 0 && hex_index < num_hexes) {
                    move.start_hex = GetGame()->IndexToCoord(hex_index);
                } else {
                    move.type = ActionType::kInvalid;
                }
                return move;
            }

            if (action >= kTradeRouteCreateBase) {
                move.type = ActionType::kTradeRouteCreate;
                int route_index = action - kTradeRouteCreateBase;
                auto heuristic_routes = GenerateTradeRouteMoves();
                if (route_index >= 0 && route_index < heuristic_routes.size()) {
                    move.path = heuristic_routes[route_index].path;
                } else {
                    move.type = ActionType::kInvalid;
                }
                return move;
            }

            if (action >= kMancalaActionBase) {
                int relative_action = action - kMancalaActionBase;
                int start_hex_index = relative_action / kMaxHexes;
                int end_hex_index = relative_action % kMaxHexes;

                if (start_hex_index >= 0 && start_hex_index < num_hexes && end_hex_index >= 0 && end_hex_index < num_hexes) {
                    move.type = ActionType::kMancala;
                    move.start_hex = GetGame()->IndexToCoord(start_hex_index);
                    move.path = { GetGame()->IndexToCoord(end_hex_index) };
                    
                    // --- THIS IS THE CORRECTED MANCALA RECONSTRUCTION ---
                    if (move.declares_trade_route) {
                        Mali_BaState temp_state = *this;
                        
                        // Simulate the primary action using the actual game function.
                        // This ensures player_posts_supply_ is correctly decremented.
                        temp_state.AddTradingPost(move.path[0], move.player, TradePostType::kPost);
                        
                        // Now find the route in this correctly simulated future state.
                        auto routes = temp_state.FindPossibleTradeRoutes(move.player, true, move.path[0], 5);
                        if (!routes.empty()) {
                            std::sort(routes.begin(), routes.end(), [](const auto& a, const auto& b){ return a.size() > b.size(); });
                            move.trade_route_path = routes[0];
                        }
                    }
                    // --- END CORRECTION ---

                } else {
                    move.type = ActionType::kInvalid;
                }
                return move;
            }

            if (action >= kUpgradeActionBase) {
                int hex_index = action - kUpgradeActionBase;
                if (hex_index >= 0 && hex_index < num_hexes) {
                    move.type = ActionType::kPlaceTCenter;
                    move.start_hex = GetGame()->IndexToCoord(hex_index);
                    move.action_string = absl::StrCat("upgrade ", move.start_hex.ToString(), "|generic_payment");

                    if (move.declares_trade_route) {
                        Mali_BaState temp_state = *this;
                        
                        // Use the actual UpgradeTradingPost method on the temporary state.
                        temp_state.UpgradeTradingPost(move.start_hex, move.player);

                        // Now that the temp state is correct, find the route.
                        auto routes = temp_state.FindPossibleTradeRoutes(move.player, true, move.start_hex, 5);
                        if (!routes.empty()) {
                            std::sort(routes.begin(), routes.end(), [](const auto& a, const auto& b){ return a.size() > b.size(); });
                            move.trade_route_path = routes[0];
                        }
                    }
                } else {
                    move.type = ActionType::kInvalid;
                }
                return move;
            }
            
            move.type = ActionType::kInvalid;
            return move;
        }

        Action Mali_BaState::ParseMoveStringToAction(const std::string &move_str) const
        {
            // This is the efficient, O(1) parser.
            std::vector<std::string> parts = absl::StrSplit(move_str, ' ');
            if (parts.empty()) return kInvalidAction;

            if (parts[0] == "pass") {
                return kPassAction;
            } else if (parts[0] == "income") {
                return kIncomeAction;
            } else if (parts[0] == "upgrade" && parts.size() >= 2) {
                // Simplified parsing for now. A full implementation would handle the payment string.
                std::vector<std::string> hex_parts = absl::StrSplit(parts[1], '|');
                try {
                    HexCoord hex = ParseHexCoordFromString(hex_parts[0]);
                    int hex_index = GetGame()->CoordToIndex(hex);
                    if (hex_index != -1) return kUpgradeActionBase + hex_index;
                } catch(...) { /* Fall through */ }
            } else if (parts[0] == "mancala" && parts.size() >= 2) {
                std::vector<std::string> hex_parts = absl::StrSplit(parts[1], "->");
                if (hex_parts.size() == 2) {
                    try {
                        HexCoord start_hex = ParseHexCoordFromString(hex_parts[0]);
                        HexCoord end_hex = ParseHexCoordFromString(hex_parts[1]);
                        int start_idx = GetGame()->CoordToIndex(start_hex);
                        int end_idx = GetGame()->CoordToIndex(end_hex);
                        if (start_idx != -1 && end_idx != -1) {
                            return kMancalaActionBase + start_idx * kMaxHexes + end_idx;
                        }
                    } catch(...) { /* Fall through */ }
                }
            }
            
            // If direct parsing fails, fall back to the slower, brute-force method for robustness.
            return State::StringToAction(current_player_id_, move_str);
        }

        // -----------------------------------------------------------
        // Heuristic functions that assign weights and then pick a 'good' move to play
        // -----------------------------------------------------------
        // Private helper to pre-calculate context for the heuristic.
        Mali_BaState::HeuristicContext Mali_BaState::CreateHeuristicContext() const {
            HeuristicContext context;
            context.posts_in_supply = player_posts_supply_[current_player_id_];

            for (const auto& [hex, posts] : trade_posts_locations_) {
                for (const auto& post : posts) {
                    if (post.owner == current_player_color_ && post.type == TradePostType::kCenter) {
                        context.existing_centers.push_back(hex);
                        int region_id = GetGame()->GetRegionForHex(hex);
                        if (region_id != -1) {
                            context.existing_center_regions.insert(region_id);
                        }
                        break; // Only need to find one center per hex
                    }
                }
            }
            return context;
        }

        // Private helper containing the core, once-duplicated logic for weighting a single move.
        double Mali_BaState::CalculateHeuristicWeightForAction(
            const Move& move,
            const LegalActionsResult& legal_actions_result,
            const HeuristicContext& context) const {

            const GameRules& rules = GetGame()->GetRules();
            const auto& weights = GetGame()->GetHeuristicWeights();
            double current_weight = 1.0;

            // Base weight from move type
            switch (move.type) {
                case ActionType::kPass:             current_weight = weights.weight_pass; break;
                case ActionType::kMancala:          current_weight = weights.weight_mancala; break;
                case ActionType::kPlaceTCenter:     current_weight = weights.weight_upgrade; break;
                case ActionType::kIncome:           current_weight = weights.weight_income; break;
                case ActionType::kPlaceToken:       current_weight = weights.weight_place_token; break;
                case ActionType::kTradeRouteCreate: current_weight = weights.weight_trade_route_create; break;
                default: break;
            }

            // Apply bonuses and normalizations
            if (move.type == ActionType::kMancala && !move.path.empty()) {
                const HexCoord& start_hex = move.start_hex;
                const HexCoord& final_hex = move.path.back();
                if (start_hex.Distance(final_hex) > 3) current_weight += weights.bonus_mancala_long_distance;
                if (GetMeeplesAt(final_hex).size() > 3 || GetMeeplesAt(start_hex).size() > 5) 
                    current_weight += weights.bonus_mancala_meeple_density;
                if (move.place_trading_post) {
                    current_weight += weights.bonus4;
                    if (GetGame()->GetCityAt(final_hex) != nullptr) current_weight += weights.bonus_mancala_city_end;
                }
            } else if (move.type == ActionType::kPlaceTCenter) {
                if (legal_actions_result.counts.upgrade_moves > 0) {
                    current_weight *= static_cast<double>(legal_actions_result.counts.mancala_moves) / legal_actions_result.counts.upgrade_moves;
                }
                if (rules.posts_per_player != kUnlimitedPosts && context.posts_in_supply < 2) current_weight += weights.bonus3;

                const HexCoord& upgrade_hex = move.start_hex;
                if (!context.existing_centers.empty()) {
                    int min_dist = 999;
                    for (const auto& center_hex : context.existing_centers) {
                        min_dist = std::min(min_dist, upgrade_hex.Distance(center_hex));
                    }
                    current_weight += min_dist * weights.bonus_upgrade_diversity_factor;
                } else {
                    current_weight += 5 * weights.bonus_upgrade_diversity_factor;
                }

                int upgrade_region = GetGame()->GetRegionForHex(upgrade_hex);
                if (upgrade_region != -1 && context.existing_center_regions.find(upgrade_region) == context.existing_center_regions.end()) {
                    current_weight += weights.bonus_upgrade_new_region;
                }
            } else if (move.type == ActionType::kIncome) {
                if (legal_actions_result.counts.income_moves > 0) {
                    current_weight *= static_cast<double>(legal_actions_result.counts.mancala_moves) / legal_actions_result.counts.income_moves;
                }
            }

            return std::max(0.0, current_weight);
        }


        // Public method to get all weights associated with heuristic move choice
        std::map<Action, double> Mali_BaState::GetHeuristicActionWeights() const {
            std::map<Action, double> action_weights;
            if (IsTerminal() || IsChanceNode() || current_phase_ != Phase::kPlay) {
                return action_weights;
            }

            const LegalActionsResult result = GetLegalActionsAndCounts();
            const HeuristicContext context = CreateHeuristicContext();

            for (Action action : result.actions) {
                Move move = ActionToMove(action);
                action_weights[action] = CalculateHeuristicWeightForAction(move, result, context);
            }

            return action_weights;
        }

        // Public method to select one action using the heuristic
        Action Mali_BaState::SelectHeuristicRandomAction() const {
            if (current_phase_ != Phase::kPlay) {
                std::vector<Action> actions = LegalActions();
                if (actions.empty()) return kInvalidAction;
                std::uniform_int_distribution<> dist(0, actions.size() - 1);
                return actions[dist(rng_)];
            }

            // Delegate the complex calculation to the other public method
            std::map<Action, double> action_weights_map = GetHeuristicActionWeights();

            if (action_weights_map.empty()) {
                LOG_WARN("SelectHeuristicRandomAction: No legal actions found.");
                return kInvalidAction;
            }

            // Convert map to parallel vectors for discrete_distribution
            std::vector<Action> actions;
            std::vector<double> weights;
            actions.reserve(action_weights_map.size());
            weights.reserve(action_weights_map.size());

            for (const auto& [action, weight] : action_weights_map) {
                actions.push_back(action);
                weights.push_back(weight);
            }
            
            // Fallback if all weights are zero
            if (std::all_of(weights.begin(), weights.end(), [](double w) { return w <= 1e-6; })) {
                std::uniform_int_distribution<> dist(0, actions.size() - 1);
                return actions[dist(rng_)];
            }

            // Select an action based on the calculated weights
            std::discrete_distribution<> dist(weights.begin(), weights.end());
            int chosen_index = dist(rng_);

            LOG_DEBUG("Player ", current_player_id_, " chooses action ", actions[chosen_index], ", ", ActionToString(current_player_id_, actions[chosen_index]));
            return actions[chosen_index];
        }

        std::vector<Move> Mali_BaState::GeneratePlaceTokenMoves() const
        {
            // This function is now only used for reference and is not part of the main LegalActions path.
            // It can be removed if not needed elsewhere.
            std::vector<Move> moves;
            for (const auto &hex : GetGame()->GetValidHexes())
            {
                if (player_token_locations_.count(hex))
                    continue;
                bool is_city = false;
                for (const auto &city : GetGame()->GetCities())
                {
                    if (city.location == hex)
                    {
                        is_city = true;
                        break;
                    }
                }
                if (is_city)
                    continue;

                moves.push_back({.player = current_player_color_,
                                       .type = ActionType::kPlaceToken,
                                       .start_hex = hex});
            }
            return moves;
        }

        std::vector<Move> Mali_BaState::GenerateTradePostUpgradeMoves() const {
            LOG_DEBUG("Entering GenerateTradePostUpgradeMoves()");
            std::vector<Move> moves;
            Player player_id = current_player_id_;
            PlayerColor player_color = GetPlayerColor(player_id);

            const GameRules &rules = GetGame()->GetRules();

            if (!HasSufficientResourcesForUpgrade(player_id)) {
                return moves;
            }

            for (const auto &hex_to_upgrade : GetGame()->GetValidHexes()) {
                const auto &posts = GetTradePostsAt(hex_to_upgrade);
                bool player_has_post_here = false;
                for (const auto &post : posts) {
                    if (post.owner == player_color && post.type == TradePostType::kPost) {
                        player_has_post_here = true;
                        break;
                    }
                }

                if (player_has_post_here) {
                    Move basic_upgrade_move;
                    basic_upgrade_move.type = ActionType::kPlaceTCenter;
                    basic_upgrade_move.start_hex = hex_to_upgrade;
                    basic_upgrade_move.player = player_color;
                    basic_upgrade_move.action_string = absl::StrCat("upgrade ", hex_to_upgrade.ToString(), "|generic_payment");
                    moves.push_back(basic_upgrade_move);

                    if (rules.free_action_trade_routes) {
                        // If we're training the AI, only get a subset of moves for efficiency
                        if (GetGame()->GetPruneMovesForAI()) {
                            // --- AI/TRAINING MODE: Heuristic Pruning (Single Best Route) ---
                            Mali_BaState temp_state = *this;
                            temp_state.UpgradeTradingPost(hex_to_upgrade, player_color);
                            auto potential_routes = temp_state.FindPossibleTradeRoutes(player_color, true, hex_to_upgrade, 5);
                            if (!potential_routes.empty()) {
                                std::sort(potential_routes.begin(), potential_routes.end(), 
                                    [](const auto& a, const auto& b){ return a.size() > b.size(); });
                                Move compound_move = basic_upgrade_move;
                                compound_move.declares_trade_route = true;
                                compound_move.trade_route_path = potential_routes[0];
                                moves.push_back(compound_move);
                            }
                        } else {
                            // --- GUI MODE: Exhaustive Generation (All Routes) ---
                            std::vector<HexCoord> available_centers;
                            for (const auto& [hex, current_posts] : trade_posts_locations_) {
                                for (const auto& current_post : current_posts) {
                                    if (current_post.owner == player_color && current_post.type == TradePostType::kCenter) {
                                        available_centers.push_back(hex);
                                        break;
                                    }
                                }
                            }
                            available_centers.push_back(hex_to_upgrade);
                            std::sort(available_centers.begin(), available_centers.end());
                            available_centers.erase(std::unique(available_centers.begin(), available_centers.end()), available_centers.end());

                            if (available_centers.size() >= rules.min_hexes_for_trade_route) {
                                int min_len = rules.min_hexes_for_trade_route;
                                int max_len = std::min({(int)available_centers.size(), 5});
                                for (int k = min_len; k <= max_len; ++k) {
                                    std::vector<bool> v(available_centers.size());
                                    std::fill(v.begin() + v.size() - k, v.end(), true);
                                    do {
                                        std::vector<HexCoord> route_combo;
                                        for (int i = 0; i < available_centers.size(); ++i) {
                                            if (v[i]) route_combo.push_back(available_centers[i]);
                                        }
                                        if (std::find(route_combo.begin(), route_combo.end(), hex_to_upgrade) != route_combo.end()) {
                                            if (IsValidCompoundUpgradeAndRoute(hex_to_upgrade, route_combo, player_color)) {
                                                Move compound_move = basic_upgrade_move;
                                                compound_move.declares_trade_route = true;
                                                compound_move.trade_route_path = GetCanonicalRoute(route_combo);
                                                moves.push_back(compound_move);
                                            }
                                        }
                                    } while (std::next_permutation(v.begin(), v.end()));
                                }
                            }
                        }
                    }
                }
            }
            return moves;
        }

        bool Mali_BaState::HasSufficientResourcesForUpgrade(Player player_id) const {
            const GameRules &rules = GetGame()->GetRules();
            const int common_cost = rules.upgrade_cost_common;
            const int rare_cost = rules.upgrade_cost_rare;
            
            if (player_id < 0 || player_id >= common_goods_.size()) {
                return false;
            }
            
            // Check rare goods first (easier payment option)
            if (player_id < rare_goods_.size()) {
                for (const auto &[good, count] : rare_goods_[player_id]) {
                    if (count >= rare_cost) {
                        return true;
                    }
                }
            }
            
            // Check common goods
            int total_common = 0;
            for (const auto &[good, count] : common_goods_[player_id]) {
                total_common += count;
            }
            
            if (total_common >= common_cost) {
                LOG_DEBUG("Player: ", current_player_id_, "; goods count: ", total_common, "; cost: ", common_cost);
                return true;
            } else return false;
        }

        std::vector<Move> Mali_BaState::GenerateMancalaMoves() const {
            std::vector<Move> legal_moves;
            if (IsChanceNode() || IsTerminal()) return legal_moves;

            const GameRules &rules = GetGame()->GetRules();
            const auto& valid_hexes_set = GetGame()->GetValidHexes();
            PlayerColor p_color = GetCurrentPlayerColor();

            for (const auto& [start_hex, token_colors] : player_token_locations_) {
                // Find a token belonging to the current player at this hex
                if (std::find(token_colors.begin(), token_colors.end(), p_color) == token_colors.end()) {
                    continue;
                }

                int num_meeples = GetMeeplesAt(start_hex).size();
                int max_dist = num_meeples + 1;

                // --- START: Corrected BFS Implementation ---
                std::queue<std::pair<HexCoord, int>> q;
                q.push({start_hex, 0});
                std::set<HexCoord> reachable_hexes; // All hexes that can be reached
                std::set<HexCoord> visited_for_bfs = {start_hex}; // Hexes already added to the queue

                while (!q.empty()) {
                    auto [current_hex, distance] = q.front();
                    q.pop();

                    // The current hex itself is reachable.
                    reachable_hexes.insert(current_hex);

                    // If we can still move further, explore neighbors.
                    if (distance < max_dist) {
                        for (const auto& dir : kHexDirections) {
                            HexCoord neighbor = current_hex + dir;
                            // Explore neighbor if it's on the board and not yet queued for visit.
                            if (valid_hexes_set.count(neighbor) && visited_for_bfs.find(neighbor) == visited_for_bfs.end()) {
                                visited_for_bfs.insert(neighbor);
                                q.push({neighbor, distance + 1});
                            }
                        }
                    }
                }
                // --- END: Corrected BFS Implementation ---

                // Now, iterate through all reachable hexes to find valid landing spots.
                for (const auto& final_hex : reachable_hexes) {
                    // A landing spot cannot be the start hex itself.
                    if (final_hex == start_hex) {
                        continue;
                    }

                    // A landing spot cannot contain another of the player's tokens.
                    if (HasTokenAt(final_hex, p_color)) {
                        continue;
                    }

                    // If we've passed all checks, this is a valid landing spot.
                    Move base_move;
                    base_move.player = p_color;
                    base_move.type = ActionType::kMancala;
                    base_move.start_hex = start_hex;
                    base_move.path = {final_hex}; // Path just stores the destination

                    legal_moves.push_back(base_move);

                    // Check for compound moves (placing a post)
                    if (CanPlaceTradingPostAt(final_hex, p_color)) {
                        Move move_with_post = base_move;
                        move_with_post.place_trading_post = true;
                        legal_moves.push_back(move_with_post);

                        if (rules.free_action_trade_routes) {
                            Mali_BaState temp_state = *this;
                            temp_state.AddTradingPost(final_hex, p_color, TradePostType::kPost);
                            auto potential_routes = temp_state.FindPossibleTradeRoutes(p_color, true, final_hex, 5);
                            if (!potential_routes.empty()) {
                                std::sort(potential_routes.begin(), potential_routes.end(), 
                                    [](const auto& a, const auto& b){ return a.size() > b.size(); });

                                Move super_compound_move = move_with_post;
                                super_compound_move.declares_trade_route = true;
                                super_compound_move.trade_route_path = potential_routes[0];
                                legal_moves.push_back(super_compound_move);
                            }
                        }
                    }
                }
            }

            // Deduplicate moves
            std::sort(legal_moves.begin(), legal_moves.end());
            legal_moves.erase(std::unique(legal_moves.begin(), legal_moves.end()), legal_moves.end());

            return legal_moves;
        }

        std::vector<HexCoord> Mali_BaState::FindShortestPath(
            const HexCoord& start, const HexCoord& end, int num_meeples) const {
            
            if (start == end) return {};
            
            if (num_meeples == 0) {
                if (start.Distance(end) == 1) return {end};
                else return {};
            }
            
            if (start.Distance(end) > num_meeples + 1) return {};
            
            std::vector<HexCoord> path;
            HexCoord current = start;
            std::set<HexCoord> used;
            used.insert(start);
            
            for (int step = 0; step < num_meeples; ++step) {
                HexCoord best_next = current;
                int best_distance = current.Distance(end);
                
                if (best_distance == 1 && used.find(end) == used.end()) {
                    path.push_back(end);
                    return path;
                }
                
                for (const auto& dir : kHexDirections) {
                    HexCoord candidate = current + dir;
                    if (IsValidHex(candidate) && used.find(candidate) == used.end()) {
                        int new_distance = candidate.Distance(end);
                        if (candidate == end) {
                            if (step == num_meeples - 1) {
                                best_next = candidate;
                                best_distance = new_distance;
                            }
                        } else if (new_distance <= best_distance) {
                            best_next = candidate;
                            best_distance = new_distance;
                        }
                    }
                }
                
                if (best_next == current) {
                    for (const auto& dir : kHexDirections) {
                        HexCoord candidate = current + dir;
                        if (IsValidHex(candidate) && used.find(candidate) == used.end()) {
                            if (candidate == end && step != num_meeples - 1) continue;
                            best_next = candidate;
                            break;
                        }
                    }
                }
                
                if (best_next == current) return {};
                
                path.push_back(best_next);
                used.insert(best_next);
                current = best_next;
                
                if (current == end) return path;
            }
            
            if (current.Distance(end) == 1 && used.find(end) == used.end()) {
                path.push_back(end);
                return path;
            }
            
            return {};
        }

        void Mali_BaState::InitializeMoveLogging()
        {
            if (move_logging_initialized_) return;
            std::string datetime = GetCurrentDateTime();
            pid_t pid = getpid(); // Get the current Process ID
            move_log_filename_ = "/tmp/mali_ba.states." + datetime + ".pid-" + std::to_string(pid) + ".log";
            move_log_file_ = std::make_unique<std::ofstream>(move_log_filename_);
            if (!move_log_file_->is_open()) {
                LOG_WARN("Failed to open move log file: ", move_log_filename_);
                return;
            }
            std::string setup_json = CreateSetupJson();
            *move_log_file_ << "[setup]\n";
            *move_log_file_ << setup_json << "\n\n";
            move_log_file_->flush();
            move_count_ = 0;
            move_logging_initialized_ = true;
            LOG_INFO("Move logger initialized: ", move_log_filename_);
            move_logging_enabled_ = true;
        }

        void Mali_BaState::LogMove(const std::string &action_string, const std::string &state_json)
        {
            if (!move_logging_initialized_ && move_logging_enabled_) InitializeMoveLogging();
            if (!move_logging_enabled_ || !move_log_file_ || !move_log_file_->is_open()) return;

            move_count_++;
            *move_log_file_ << "[move" << move_count_ << "]\n";
            *move_log_file_ << "action=" << action_string << "\n";
            *move_log_file_ << "state=" << state_json << "\n\n";
            move_log_file_->flush();
        }

        std::string Mali_BaState::CreateSetupJson() const
        {
            json setup;
            setup["num_players"] = game_->NumPlayers();
            setup["grid_radius"] = GetGame()->GetGridRadius();
            setup["tokens_per_player"] = GetGame()->GetTokensPerPlayer();
            json valid_hexes_json = json::array();
            for (const auto &hex : GetGame()->GetValidHexes()) {
                valid_hexes_json.push_back(HexCoordToJsonString(hex));
            }
            setup["valid_hexes"] = valid_hexes_json;
            json cities_json = json::array();
            for (const auto &city : GetGame()->GetCities()) {
                json city_json;
                city_json["id"] = city.id;
                city_json["name"] = city.name;
                city_json["cultural_group"] = city.culture;
                city_json["location"] = HexCoordToJsonString(city.location);
                city_json["common_good"] = city.common_good;
                city_json["rare_good"] = city.rare_good;
                cities_json.push_back(city_json);
            }
            setup["cities"] = cities_json;
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            std::stringstream ss;
            ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
            setup["timestamp"] = ss.str();
            return setup.dump(2);
        }

    } // namespace mali_ba
} // namespace open_spiel