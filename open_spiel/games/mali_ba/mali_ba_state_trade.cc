// mali_ba_state_trade.cc
// Trading post and resource functionality

#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"

#include <string>
#include <vector>
#include <map>

namespace open_spiel {
namespace mali_ba {

// =====================================================================
// Trading Post & Center Methods
// =====================================================================
void Mali_BaState::AddTradingPost(const HexCoord& hex, PlayerColor player, TradePostType type) {
    auto& posts = trade_posts_locations_[hex];  // Add to the board
    posts.push_back({player, type});            // Track post/center for the player

    // If the number of posts per player is not unlimited then update the player's post inventory
    Player player_id = GetPlayerId(player);
    const GameRules& rules = GetGame()->GetRules();  // get the game rules
    if (rules.posts_per_player != kUnlimitedPosts) { // Only decrement if posts are limited
        // If they're adding a trading *post* (not center)
        if (type == TradePostType::kPost) {
            // Decrement the player's post supply if posts per player is not unlimited
            SPIEL_CHECK_GT(player_posts_supply_[player_id], 0);
            player_posts_supply_[player_id]--;
            //LOG_INFO("Add: Player ", player_id, " placed a post. Supply now: ", 
            //    player_posts_supply_[player_id]);
        }
        else {
            // They're placing a trading *center* (i.e. upgrading) so increment the 
            // player's number of posts as it comes off the board back into inventory
            player_posts_supply_[player_id]++;
            //LOG_INFO("Add: Player ", player_id, " placed a center. Post supply now: ", 
            //    player_posts_supply_[player_id]);
        }
    }
}


void Mali_BaState::UpgradeTradingPost(const HexCoord& hex, PlayerColor player) {
    auto& posts = trade_posts_locations_[hex];
    for (auto& post : posts) {
        if (post.owner == player && post.type == TradePostType::kPost) {
            post.type = TradePostType::kCenter;
            
            // If the number of posts per player is not unlimited then update the player's post inventory
            Player player_id = GetPlayerId(player);
            const GameRules& rules = GetGame()->GetRules();  // get the game rules
            if (rules.posts_per_player != kUnlimitedPosts) { // Only decrement if posts are limited
                // They're placing a trading *center* (i.e. upgrading) so increment the 
                // player's number of posts as it comes off the board back into inventory
                player_posts_supply_[player_id]++;
                //LOG_INFO("Upgrade: Player ", player_id, " placed a center. Post supply now: ", 
                //    player_posts_supply_[player_id]);
            }
            
            // If the rule is enabled and meeples are present: Remove one meeple 
            if (rules.remove_meeple_on_upgrade) {
                const auto& meeples = GetMeeplesAt(hex);
                if (!meeples.empty()) {
                    // Remove the first meeple from the hex
                    RemoveMeepleAt(hex, 0);  // Remove meeple at index 0
                    LOG_DEBUG("Removed one meeple from hex ", hex.ToString(), " due to trading post upgrade");
                }
            }
            
            break;  // Only upgrade one post per call
        }
    }
}

const std::vector<TradePost>& Mali_BaState::GetTradePostsAt(const HexCoord& hex) const {
    static const std::vector<TradePost> empty_vector;
    auto it = trade_posts_locations_.find(hex);
    return (it != trade_posts_locations_.end()) ? it->second : empty_vector;
}

bool Mali_BaState::HasPlayerPostOrCenterAt(const HexCoord& hex, PlayerColor player) const {
    const auto& posts = GetTradePostsAt(hex);
    for (const auto& post : posts) {
        if (post.owner == player) {
            return true; // Player already has a post or center here
        }
    }
    return false;
}


int Mali_BaState::CountTradingCentersAt(const HexCoord& hex) const {
    const auto& posts = GetTradePostsAt(hex);
    int count = 0;
    for (const auto& post : posts) {
        if (post.type == TradePostType::kCenter) {
            count++;
        }
    }
    return count;
}

bool Mali_BaState::CanPlaceTradingPostAt(const HexCoord& hex, PlayerColor player) const {
    // 1. Check if player already has a post or center here
    if (HasPlayerPostOrCenterAt(hex, player)) {
        return false; // Player already has something here
    }

    // 2. Check if they have enough trading posts in their supply (may be unlimited)
    const GameRules& rules = GetGame()->GetRules(); // Get game rules
    int posts_in_supply = player_posts_supply_[current_player_id_];
    // Check if player has posts available (or if they are unlimited)
    if (rules.posts_per_player != kUnlimitedPosts && posts_in_supply < 1) {
        return false;
    }
    
    // 3. Check if this is a city - cities vs. non-cities have special rules for trading centers
    bool is_city = false;
    for (const auto& city : GetGame()->GetCities()) {
        if (city.location == hex) {
            is_city = true;
            break;
        }
    }
    // For non-city hexes, check the (n-1) trading centers limit
    if (!is_city) {
        int center_count = CountTradingCentersAt(hex);
        int player_count = game_->NumPlayers();
        
        if (center_count >= player_count - 1) {
            return false; // Already at max trading centers for this hex
        }
    }
    
    // 4. Check if there's at least one meeple here or if player has resources
    const auto& meeples = GetMeeplesAt(hex);
    if (!meeples.empty()) {
        return true; // Has at least one meeple to support a trading post
    }
    
    // 5. Check if player has resources to place a post without meeples
    Player player_id = GetPlayerId(player);
    if (player_id >= 0 && player_id < common_goods_.size()) {
        int total_common = 0;
        for (const auto& [good_name, count] : common_goods_[player_id]) {
            total_common += count;
        }
        if (total_common > 0) {
            return true; // Player has at least one common good to spend
        }
    }
    
    return false; // No meeples and no resources to place a post or other rule fails    
}

// =====================================================================
// Trading Route Methods
// =====================================================================
// Function to create a trade route
bool Mali_BaState::CreateTradeRoute(const std::vector<HexCoord>& hexes, PlayerColor player) {
    const GameRules& rules = GetGame()->GetRules();

    // 1. Check min hexes requirement
    if (hexes.size() < rules.min_hexes_for_trade_route) {
        LOG_WARN("CreateTradeRoute ERROR: Route requires at least ", rules.min_hexes_for_trade_route, " hexes.");
        return false;
    }

    // 2. Check for max shared centers
    for (const auto& existing_route : trade_routes_) {
        if (existing_route.owner == player) {
            int shared_centers = 0;
            std::set<HexCoord> new_route_hexes(hexes.begin(), hexes.end());
            for (const auto& hex : existing_route.hexes) {
                if (new_route_hexes.count(hex)) {
                    shared_centers++;
                }
            }
            if (shared_centers > rules.max_shared_centers_between_routes) {
                LOG_WARN("CreateTradeRoute ERROR: Route shares too many centers with an existing route. Max allowed = ",
                    rules.max_shared_centers_between_routes, ". Shared = ", shared_centers);
                return false;
            }
        }
    }
    
    // 3. If the rules allow, automatically upgrade trading posts in cities to centers
    if (rules.city_free_upgrade) {
        for (const HexCoord& hex : hexes) {
            bool is_city = false;
            for (const auto& city : GetGame()->GetCities()) {
                if (city.location == hex) {
                    is_city = true;
                    break;
                }
            }
            
            if (is_city) {
                // Upgrade this player's post at this city hex to a center.
                // Note: UpgradeTradingPost is idempotent if it's already a center.
                UpgradeTradingPost(hex, player);
            }
        }
    }
    
    // If the rule is enabled and meeples are present: Remove one meeple from each hex
    if (rules.remove_meeple_on_trade_route) {
        for (const HexCoord& hex : hexes) {
            const auto& meeples = GetMeeplesAt(hex);
            if (!meeples.empty()) {
                // Remove the first meeple from the hex
                RemoveMeepleAt(hex, 0);  // Remove meeple at index 0
                LOG_DEBUG("Removed one meeple from hex ", hex.ToString(), " due to trade route creation");
            }
        }
    }
    
    // Create the trade route
    TradeRoute route;
    route.id = next_route_id_++;
    route.owner = player;
    route.hexes = hexes;
    route.active = true;

    // ... (rest of the function is the same) ... TODO: find out what code was here
    
    route.goods = {}; // Goods calculation can be done on income, not creation.
    route.hexes = GetCanonicalRoute(route.hexes); // make sure hexes are sorted properly
    trade_routes_.push_back(route);
    
    LOG_DEBUG("Moves: ",history_.size(), "| Trade route created successfully! Total routes now: ", trade_routes_.size());
    return true;
}


// Function to delete a trade route
bool Mali_BaState::DeleteTradeRoute(int route_id) {
    auto it = std::find_if(trade_routes_.begin(), trade_routes_.end(),
                           [route_id](const TradeRoute& r) { return r.id == route_id; });
    
    if (it == trade_routes_.end()) {
        LOG_WARN("DeleteTradeRoute ERROR: Route not found");
        return false;
    }
    
    trade_routes_.erase(it);
    LOG_INFO("Trade route deleted successfully");
    return true;
}

// Function to validate all trade routes
void Mali_BaState::ValidateTradeRoutes() {
    for (auto& route : trade_routes_) {
        bool valid = true;
        
        // Check if player has trading post/center at each hex
        for (const auto& hex : route.hexes) {
            bool has_player_post = false;
            const auto& posts = GetTradePostsAt(hex);
            
            for (const auto& post : posts) {
                if (post.owner == route.owner) {
                    has_player_post = true;
                    break;
                }
            }
            
            if (!has_player_post) {
                valid = false;
                break;
            }
        }
        
        route.active = valid;
    }
}

// Helper method to find cities connected to a trading center via trade routes
std::vector<const City*> Mali_BaState::GetConnectedCities(const HexCoord& center_hex, PlayerColor player) const {
    std::vector<const City*> connected_cities;
    
    // Find all trade routes owned by this player that include this hex
    for (const auto& route : trade_routes_) {
        if (route.owner == player && route.active && 
            std::find(route.hexes.begin(), route.hexes.end(), center_hex) != route.hexes.end()) {
            
            // Check which cities are in this route
            for (const auto& route_hex : route.hexes) {
                for (const auto& city : GetGame()->GetCities()) {
                    if (city.location == route_hex) {
                        // Make sure we don't add duplicates
                        if (std::find(connected_cities.begin(), connected_cities.end(), &city) == connected_cities.end()) {
                            connected_cities.push_back(&city);
                        }
                    }
                }
            }
        }
    }
    
    return connected_cities;
}

// Helper method to find closest cities to a hex
std::vector<const City*> Mali_BaState::FindClosestCities(const HexCoord& hex) const {
    std::vector<const City*> closest_cities;
    int min_distance = std::numeric_limits<int>::max();
    
    for (const auto& city : GetGame()->GetCities()) {
        int distance = hex.Distance(city.location);
        if (distance < min_distance) {
            min_distance = distance;
            closest_cities.clear();
            closest_cities.push_back(&city);
        } else if (distance == min_distance) {
            closest_cities.push_back(&city);
        }
    }
    
    return closest_cities;
}


// =====================================================================
// Functions to generate & apply legal moves for creating a trading route
// =====================================================================
std::vector<Move> Mali_BaState::GenerateTradeRouteMoves() const {
    std::vector<Move> moves;
    const GameRules& rules = GetGame()->GetRules();

    // Do not generate these moves if they are free actions, as they will be
    // generated as part of compound moves instead.
    if (rules.free_action_trade_routes) {
        return moves;
    }

    PlayerColor p_color = GetCurrentPlayerColor();
    if (p_color == PlayerColor::kEmpty) return moves;

    // Find all possible valid routes.
    // We limit the search to routes up to 5 hexes to keep it fast.
    auto all_possible_routes = FindPossibleTradeRoutes(p_color, true, nullptr, 5);
    
    if (all_possible_routes.empty()) {
        return moves;
    }

    // --- Heuristic Selection ---
    // Instead of adding all routes, we select a few "best" ones.
    // For this example, we'll just sort by length and take the top 5.
    std::sort(all_possible_routes.begin(), all_possible_routes.end(),
        [](const std::vector<HexCoord>& a, const std::vector<HexCoord>& b) {
            return a.size() > b.size();
        });

    int routes_to_add = std::min((int)all_possible_routes.size(), 5);
    for (int i = 0; i < routes_to_add; ++i) {
        Move move;
        move.type = ActionType::kTradeRouteCreate;
        move.player = p_color;
        move.path = all_possible_routes[i];
        move.route_id = i; // The index becomes the ID for MoveToAction encoding
        moves.push_back(move);
    }
    
    return moves;
}

// Add this helper function to perform additional validation
// This gets called directly from MaybeGenerateLegalActions()
bool Mali_BaState::IsValidTradeRouteForMoveGeneration(
    const std::vector<HexCoord>& route_hexes, PlayerColor player) const {
    
    const GameRules& rules = GetGame()->GetRules();
    
    // 1. Check minimum length
    if (route_hexes.size() < rules.min_hexes_for_trade_route) {
        return false;
    }

    // Create sorted copy of the hexes in the route for canonical comparison
    std::vector<HexCoord> sorted_route = route_hexes;
    std::sort(sorted_route.begin(), sorted_route.end());

    
    // 2. Check that all hexes have the player's trading centers
    // (Since we're only considering player_centers, this should always pass, but double-check)
    for (const auto& hex : sorted_route) {
        bool has_center = false;
        auto posts_it = trade_posts_locations_.find(hex);
        if (posts_it != trade_posts_locations_.end()) {
            for (const auto& post : posts_it->second) {
                if (post.owner == player && post.type == TradePostType::kCenter) {
                    has_center = true;
                    break;
                }
            }
        }
        if (!has_center) {
            return false;
        }
    }
    
    // 3. Check shared trading centers rule 
    if (rules.max_shared_centers_between_routes >= 0) {
        std::set<HexCoord> new_hex_set(sorted_route.begin(), sorted_route.end());
        
        for (const auto& existing_route : trade_routes_) {
            if (existing_route.owner == player && existing_route.active) {
                int shared_centers = 0;
                for (const auto& existing_hex : existing_route.hexes) {
                    if (new_hex_set.count(existing_hex)) {
                        shared_centers++;
                    }
                }
                if (shared_centers > rules.max_shared_centers_between_routes) {
                    LOG_DEBUG("Rejecting route - would share ", shared_centers, 
                             " centers with existing route (max allowed: ", 
                             rules.max_shared_centers_between_routes, ")");
                    return false; // This route would violate the shared centers rule
                }
            }
        }
    }
    
    // 4. Check for exact duplicate trade routes
    for (const auto& existing_route : trade_routes_) {
        if (existing_route.owner == player && existing_route.active) {
            std::vector<HexCoord> sorted_existing = existing_route.hexes;
            std::sort(sorted_existing.begin(), sorted_existing.end());
            if (sorted_existing == sorted_route) {
                return false; // Duplicate found
            }
        }
    }
    
    return true;
}


// =====================================================================
// Functions to generate & apply legal moves for taking income
// =====================================================================
std::vector<Move> Mali_BaState::GenerateIncomeMoves() const {
    std::vector<Move> moves;
    Player player_id = current_player_id_;
    if (player_id < 0) return moves;

    PlayerColor player_color = GetPlayerColor(player_id);

    // Check if there is any potential source of income.
    bool has_any_income_source = false;
    for (const auto& [hex, posts] : trade_posts_locations_) {
        for (const auto& post : posts) {
            if (post.owner == player_color) {
                has_any_income_source = true;
                break;
            }
        }
        if (has_any_income_source) break;
    }
    if (!has_any_income_source) return moves;

    // Define strategic profiles
    GoodsCollection profile_new_rare;
    GoodsCollection profile_new_common;
    GoodsCollection profile_max_total;
    GoodsCollection profile_hoard_rare;

    // --- Iterate through all income sources and apply heuristics for each profile ---
    for (const auto& [hex, posts] : trade_posts_locations_) {
        for (const auto& post : posts) {
            if (post.owner != player_color) continue;

            const City* city_at_hex = GetGame()->GetCityAt(hex);

            // Guaranteed income from centers in cities
            if (post.type == TradePostType::kCenter && city_at_hex) {
                profile_new_rare.rare_goods[city_at_hex->rare_good]++;
                profile_new_common.rare_goods[city_at_hex->rare_good]++;
                profile_max_total.rare_goods[city_at_hex->rare_good]++;
                profile_hoard_rare.rare_goods[city_at_hex->rare_good]++;
                continue; // This source is handled, move to next post
            }

            // Income from trading posts
            if (post.type == TradePostType::kPost) {
                auto closest = FindClosestCities(hex);
                if (!closest.empty()) {
                    const std::string& good = closest[0]->common_good;
                    profile_new_rare.common_goods[good]++;
                    profile_new_common.common_goods[good]++;
                    profile_max_total.common_goods[good]++;
                    profile_hoard_rare.common_goods[good]++;
                }
                continue;
            }

            // Choices for centers not in cities
            if (post.type == TradePostType::kCenter && !city_at_hex) {
                auto connected = GetConnectedCities(hex, player_color);
                const auto& choice_cities = connected.empty() ? FindClosestCities(hex) : connected;
                if (choice_cities.empty()) continue;

                // --- Apply heuristics for this choice point ---
                
                // Profile: Maximize New Rare Goods
                const City* best_new_rare_city = nullptr;
                if (!connected.empty()) { // Can only take rare goods if connected
                    for (const auto* city : choice_cities) {
                        if (GetRareGoodCount(player_id, city->rare_good) == 0) {
                            best_new_rare_city = city;
                            break;
                        }
                    }
                }
                if (best_new_rare_city) {
                    profile_new_rare.rare_goods[best_new_rare_city->rare_good]++;
                } else { // No new rare goods available, take 2 common instead
                    profile_new_rare.common_goods[choice_cities[0]->common_good]++;
                    profile_new_rare.common_goods[choice_cities.size() > 1 ? choice_cities[1]->common_good : choice_cities[0]->common_good]++;
                }

                // Profile: Hoard Rare Goods (take any rare good if possible)
                if (!connected.empty()) {
                    profile_hoard_rare.rare_goods[choice_cities[0]->rare_good]++;
                } else { // Isolated, must take common
                    profile_hoard_rare.common_goods[choice_cities[0]->common_good]+=2;
                }

                // Profile: Maximize New Common Goods
                // (This heuristic is complex, for now we just take the first two distinct goods)
                const City* best_new_common1 = choice_cities[0];
                const City* best_new_common2 = choice_cities.size() > 1 ? choice_cities[1] : choice_cities[0];
                profile_new_common.common_goods[best_new_common1->common_good]++;
                profile_new_common.common_goods[best_new_common2->common_good]++;

                // Profile: Maximize Total Goods (2 common is generally better than 1 rare)
                profile_max_total.common_goods[choice_cities[0]->common_good]++;
                profile_max_total.common_goods[choice_cities.size() > 1 ? choice_cities[1]->common_good : choice_cities[0]->common_good]++;
            }
        }
    }

    // --- De-duplicate and create final moves ---
    std::set<std::string> unique_actions;
    std::vector<GoodsCollection> profiles = {profile_new_rare, profile_new_common, profile_max_total, profile_hoard_rare};

    for (const auto& profile_outcome : profiles) {
        if (profile_outcome.IsEmpty()) continue;

        std::string action_str = "income " + FormatGoodsCollectionCompact(profile_outcome);
        std::string normalized_action = NormalizeIncomeAction(action_str);

        if (unique_actions.find(normalized_action) == unique_actions.end()) {
            unique_actions.insert(normalized_action);
            Move move;
            move.type = ActionType::kIncome;
            move.player = player_color;
            move.action_string = normalized_action;
            moves.push_back(move);
        }
    }
    return moves;
}


bool Mali_BaState::IsValidCompoundUpgradeAndRoute(const HexCoord& upgrade_hex,
                                                 const std::vector<HexCoord>& route_path,
                                                 PlayerColor player) const {
    const GameRules& rules = GetGame()->GetRules();

    // 1. Check minimum route length
    if (route_path.size() < rules.min_hexes_for_trade_route) {
        return false;
    }

    // --- THIS IS THE CRITICAL FIX ---
    // Create a temporary copy of the state's trade posts to simulate the upgrade.
    Mali_BaState temp_state = *this;
    
    // Simulate the upgrade in the temporary state
    bool found_post_to_upgrade = false;
    auto it = temp_state.trade_posts_locations_.find(upgrade_hex);
    if (it != temp_state.trade_posts_locations_.end()) {
        for (auto& post : it->second) {
            if (post.owner == player && post.type == TradePostType::kPost) {
                post.type = TradePostType::kCenter;
                found_post_to_upgrade = true;
                break;
            }
        }
    }
    // If for some reason the post to upgrade wasn't found, it's an invalid move.
    if (!found_post_to_upgrade) return false;
    
    // NOW, use the standard validation function on the temporary state, which has the upgraded center.
    // This correctly checks for duplicates and shared center violations.
    return temp_state.IsValidTradeRouteForMoveGeneration(route_path, player);
}


// Flexible function to find possible trade routes with optional filtering
std::vector<std::vector<HexCoord>> Mali_BaState::FindPossibleTradeRoutes(
    PlayerColor player,
    bool is_valid_per_rules,
    const HexCoord* includes_hex,
    int max_hexes,
    int min_hexes) const {
    
    const GameRules &rules = GetGame()->GetRules(); 
    std::vector<std::vector<HexCoord>> valid_routes;
    int max_len_allowed = 8;    
    
    if (player == PlayerColor::kEmpty) return valid_routes;
    
    if (min_hexes == -1 || min_hexes < 2 || min_hexes < rules.min_hexes_for_trade_route) {
        min_hexes = std::max(2, rules.min_hexes_for_trade_route);
    }
    
    if (max_hexes == -1) max_hexes = max_len_allowed;
    if (max_hexes < min_hexes) max_hexes = min_hexes;
    
    std::vector<HexCoord> available_centers;
    for (const auto& [hex, posts] : trade_posts_locations_) {
        for (const auto& post : posts) {
            if (post.owner == player && post.type == TradePostType::kCenter) {
                available_centers.push_back(hex);
                break;
            }
        }
    }
    
    if (includes_hex) {
        bool includes_hex_is_center = false;
        for (const auto& center_hex : available_centers) {
            if (center_hex == *includes_hex) {
                includes_hex_is_center = true;
                break;
            }
        }
        if (!includes_hex_is_center) available_centers.push_back(*includes_hex);
    }
    
    if (available_centers.size() < min_hexes) return valid_routes;
    max_hexes = std::min(max_hexes, static_cast<int>(available_centers.size()));
    
    for (int route_length = min_hexes; route_length <= max_hexes; ++route_length) {
        std::vector<bool> selector(available_centers.size(), false);
        std::fill(selector.begin() + selector.size() - route_length, selector.end(), true);
        
        do {
            std::vector<HexCoord> route_combination;
            for (size_t i = 0; i < available_centers.size(); ++i) {
                if (selector[i]) route_combination.push_back(available_centers[i]);
            }
            
            route_combination = GetCanonicalRoute(route_combination);
            
            if (includes_hex) {
                if (std::find(route_combination.begin(), route_combination.end(), *includes_hex) == route_combination.end()) continue;
            }
            
            if (is_valid_per_rules) {
                if (!IsValidTradeRouteForMoveGeneration(route_combination, player)) continue;
            }
            
            valid_routes.push_back(route_combination);
            
        } while (std::next_permutation(selector.begin(), selector.end()));
    }
    
    return valid_routes;
}

// Convenience overloads for common use cases
std::vector<std::vector<HexCoord>> Mali_BaState::FindPossibleTradeRoutes(
    PlayerColor player,
    bool is_valid_per_rules) const {
    return FindPossibleTradeRoutes(player, is_valid_per_rules, nullptr, -1, -1);
}

std::vector<std::vector<HexCoord>> Mali_BaState::FindPossibleTradeRoutes(
    PlayerColor player,
    bool is_valid_per_rules,
    const HexCoord& includes_hex) const {
    return FindPossibleTradeRoutes(player, is_valid_per_rules, &includes_hex, -1, -1);
}

std::vector<std::vector<HexCoord>> Mali_BaState::FindPossibleTradeRoutes(
    PlayerColor player,
    bool is_valid_per_rules,
    const HexCoord& includes_hex,
    int max_hexes) const {
    return FindPossibleTradeRoutes(player, is_valid_per_rules, &includes_hex, max_hexes, -1);
}


// Helper that returns the canonical (properly sorted) representation of a trade 
// route made up of the passed hexes
std::vector<HexCoord> Mali_BaState::GetCanonicalRoute(const std::vector<HexCoord>& route_combination) const {
    std::vector<HexCoord> canonical_route = route_combination;
    std::sort(canonical_route.begin(), canonical_route.end());
    return canonical_route;
}


// =====================================================================
// Helper Functions related to trade
// =====================================================================
std::map<std::string, int> ParseGoodsString(const std::string& goods_str) {
    std::map<std::string, int> goods;
    
    if (goods_str.empty()) {
        return goods;
    }
    
    std::vector<std::string> items = absl::StrSplit(goods_str, ',');
    for (const auto& item : items) {
        if (item.empty()) continue;
        
        std::vector<std::string> parts = absl::StrSplit(item, ':');
        if (parts.size() == 2) {
            std::string good_name = parts[0];
            // Trim whitespace
            good_name.erase(0, good_name.find_first_not_of(" \t"));
            good_name.erase(good_name.find_last_not_of(" \t") + 1);
            
            try {
                int count = std::stoi(parts[1]);
                if (count > 0) {
                    goods[good_name] = count;
                }
            } catch (const std::exception& e) {
                LOG_WARN("Failed to parse count for good ", good_name, ": ", e.what());
            }
        }
    }
    
    return goods;
}

std::string FormatGoodsString(const std::map<std::string, int>& goods) {
    std::vector<std::string> items;
    
    for (const auto& [good_name, count] : goods) {
        if (count > 0) {
            items.push_back(good_name + ":" + std::to_string(count));
        }
    }
    
    return absl::StrJoin(items, ",");
}

GoodsCollection ParseGoodsCollection(const std::string& collection_str, 
                                   const std::vector<City>& cities) {
    GoodsCollection collection;
    
    std::vector<std::string> parts = absl::StrSplit(collection_str, '|');
    
    if (parts.size() >= 1 && !parts[0].empty()) {
        collection.common_goods = ParseGoodsString(parts[0]);
    }
    
    if (parts.size() >= 2 && !parts[1].empty()) {
        collection.rare_goods = ParseGoodsString(parts[1]);
    }
    
    if (parts.size() == 1 && !parts[0].empty()) {
        auto all_goods = ParseGoodsString(parts[0]);
        collection.common_goods.clear();
        collection.rare_goods.clear();
        
        std::set<std::string> rare_good_names;
        for (const auto& city : cities) {
            rare_good_names.insert(city.rare_good);
        }
        
        for (const auto& [good_name, count] : all_goods) {
            if (rare_good_names.count(good_name)) {
                collection.rare_goods[good_name] = count;
            } else {
                collection.common_goods[good_name] = count;
            }
        }
    }
    
    return collection;
}

std::string FormatGoodsCollection(const GoodsCollection& collection) {
    std::string common_str = FormatGoodsString(collection.common_goods);
    std::string rare_str = FormatGoodsString(collection.rare_goods);
    
    return common_str + "|" + rare_str;
}

std::string FormatGoodsCollectionCompact(const GoodsCollection& collection) {
    std::string common_str = FormatGoodsString(collection.common_goods);
    std::string rare_str = FormatGoodsString(collection.rare_goods);
    
    if (common_str.empty() && rare_str.empty()) {
        return "";
    } else if (rare_str.empty()) {
        return common_str + "|";
    } else if (common_str.empty()) {
        return "|" + rare_str;
    } else {
        return common_str + "|" + rare_str;
    }
}

// Resource Management Methods
int Mali_BaState::GetCommonGoodCount(Player player, const std::string& good_name) const {
    if (player < 0 || player >= common_goods_.size())
        return 0; // Bounds check
    const auto& player_goods = common_goods_[player];
    auto it = player_goods.find(good_name);
    return (it != player_goods.end()) ? it->second : 0;
}

int Mali_BaState::GetRareGoodCount(Player player, const std::string& good_name) const {
    if (player < 0 || player >= rare_goods_.size())
        return 0; // Bounds check
    const auto& player_goods = rare_goods_[player];
    auto it = player_goods.find(good_name);
    return (it != player_goods.end()) ? it->second : 0;
}

const std::map<std::string, int>& Mali_BaState::GetPlayerCommonGoods(Player player) const {
    static const std::map<std::string, int> empty_map;
    if (player < 0 || player >= common_goods_.size())
        return empty_map; // Bounds check
    return common_goods_[player];
}

const std::map<std::string, int>& Mali_BaState::GetPlayerRareGoods(Player player) const {
    static const std::map<std::string, int> empty_map;
    if (player < 0 || player >= rare_goods_.size())
        return empty_map; // Bounds check
    return rare_goods_[player];
}

std::string Mali_BaState::NormalizeIncomeAction(const std::string& action_string) const {
    if (action_string.find("income") != 0) {
        return action_string;
    }
    
    std::vector<std::string> main_parts = absl::StrSplit(action_string, ' ', absl::SkipEmpty());
    if (main_parts.size() != 2) {
        return action_string;
    }
    
    GoodsCollection collection = ParseGoodsCollection(main_parts[1], GetGame()->GetCities());
    
    std::map<std::string, int> sorted_common = collection.common_goods;
    std::map<std::string, int> sorted_rare = collection.rare_goods;
    
    GoodsCollection normalized_collection;
    normalized_collection.common_goods = sorted_common;
    normalized_collection.rare_goods = sorted_rare;
    
    return "income " + FormatGoodsCollectionCompact(normalized_collection);
}

std::string Mali_BaState::CreateIncomeActionString(
    const std::map<std::string, int>& common_goods,
    const std::map<std::string, int>& rare_goods) const {
    
    std::string income_str = "income";
    
    std::vector<std::string> common_parts;
    for (const auto& [good, count] : common_goods) {
        if (count > 0) {
            common_parts.push_back(good + ":" + std::to_string(count));
        }
    }
    if (!common_parts.empty()) {
        income_str += " " + absl::StrJoin(common_parts, ",");
    }
    
    std::vector<std::string> rare_parts;
    for (const auto& [good, count] : rare_goods) {
        if (count > 0) {
            rare_parts.push_back(good + ":" + std::to_string(count));
        }
    }
    if (!rare_parts.empty()) {
        income_str += " " + absl::StrJoin(rare_parts, ",");
    }
    
    return income_str;
}
// =============================================================================
// Public Setters for Testing
// =============================================================================
void Mali_BaState::TestOnly_SetCurrentPlayer(Player player) {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, game_->NumPlayers());
    current_player_id_ = player;
    current_player_color_ = GetPlayerColor(player);
}

void Mali_BaState::TestOnly_SetTradePost(const HexCoord& hex, PlayerColor owner, TradePostType type) {
    // Clear existing posts at this hex for this player to avoid duplicates
    auto& posts = trade_posts_locations_[hex];
    posts.erase(std::remove_if(posts.begin(), posts.end(), 
                               [owner](const TradePost& p){ return p.owner == owner; }), 
                posts.end());
    // Add the new post
    posts.push_back({owner, type});
}

void Mali_BaState::TestOnly_SetPlayerToken(const HexCoord& hex, PlayerColor owner) {
    AddTokenAt(hex, owner); // Use new helper method
}
void Mali_BaState::TestOnly_SetPlayerTokens(const HexCoord& hex, const std::vector<PlayerColor>& owners) {
    player_token_locations_[hex] = owners;
}

void Mali_BaState::TestOnly_SetMeeples(const HexCoord& hex, const std::vector<MeepleColor>& meeples) {
    hex_meeples_[hex] = meeples;
}

void Mali_BaState::TestOnly_SetCommonGood(Player player, const std::string& good_name, int count) {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, common_goods_.size());
    common_goods_[player][good_name] = count;
}

void Mali_BaState::TestOnly_SetRareGood(Player player, const std::string& good_name, int count) {
    SPIEL_CHECK_GE(player, 0);
    SPIEL_CHECK_LT(player, rare_goods_.size());
    rare_goods_[player][good_name] = count;
}

void Mali_BaState::TestOnly_ClearPlayerTokens() {
    player_token_locations_.clear();
}

void Mali_BaState::TestOnly_ClearMeeples() {
    for (auto& [hex, meeples] : hex_meeples_) {
        meeples.clear();
    }
}

}  // namespace mali_ba
}  // namespace open_spiel