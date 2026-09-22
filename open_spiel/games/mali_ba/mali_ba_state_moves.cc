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
#include <fstream>
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
            return kInvalidAction; // Obsolete in sequential phase architecture
        }

        Move Mali_BaState::ActionToMove(Action action) const {
            Move move;
            move.type = ActionType::kInvalid; // Obsolete in sequential phase architecture
            return move;
        }

        Action Mali_BaState::ParseMoveStringToAction(const std::string &move_str) const {
            // Simply delegate to OpenSpiel's default string matcher which uses our updated ActionToString
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


        // File-scope helper: distance from a hex to the nearest city.
        static int MinCityDistance(const HexCoord& hex, const std::vector<City>& cities) {
            int min_dist = 999;
            for (const auto& city : cities) {
                min_dist = std::min(min_dist, hex.Distance(city.location));
            }
            return min_dist;
        }

        // File-scope helper: count how many hex neighbours are on the board (centrality proxy).
        static int ValidNeighbourCount(const HexCoord& hex,
                                       const std::set<HexCoord>& valid_hexes) {
            int count = 0;
            for (const auto& dir : kHexDirections) {
                if (valid_hexes.count(hex + dir)) count++;
            }
            return count;
        }

        // Reads /tmp/mali_ba_heuristic_params.txt once per process (cached).
        // File format: two floats on one line, e.g. "0.42 1.37"
        //   mult_add_in: added to distance multipliers (range 0..1)
        //   add_add_in:  added to flat bonuses        (range 0.5..2)
        // Returns (0.0, 0.0) if the file is absent or malformed.
        static std::pair<double,double> LoadHeuristicParams() {
            static bool loaded = false;
            static double mult = 0.7, add_val = 1.5;
            if (!loaded) {
                loaded = true;
                std::ifstream f("/tmp/mali_ba_heuristic_params.txt");
                if (f.is_open()) f >> mult >> add_val;
            }
            return {mult, add_val};
        }

        // Returns weighted probability for every legal action in the current phase.
        // Covers all sequential phases so SelectHeuristicRandomAction() can delegate here.
        std::map<Action, double> Mali_BaState::GetHeuristicActionWeights() const {
            std::map<Action, double> action_weights;
            if (IsTerminal() || IsChanceNode()) return action_weights;

            const std::vector<Action> actions = LegalActions();
            if (actions.empty()) return action_weights;

            const auto& cities      = GetGame()->GetCities();
            const auto& valid_hexes = GetGame()->GetValidHexes();
            const GameRules& rules  = GetGame()->GetRules();

            // ----------------------------------------------------------------
            // Pre-compute key hex locations for the Timbuktu-to-coast win
            // condition (the only active win condition by default).
            // We identify Timbuktu and the desert cities by name so the
            // heuristic can steer toward them at every stage.
            // ----------------------------------------------------------------
            const std::set<HexCoord>& coastal_hexes = GetGame()->GetCoastalHexes();

            HexCoord timbuktu_hex{0,0,0};
            std::vector<HexCoord> desert_city_hexes; // Agadez, Oudane
            bool timbuktu_found_in_cities = false;
            for (const auto& city : cities) {
                if (city.name == "Timbuktu") {
                    timbuktu_hex = city.location;
                    timbuktu_found_in_cities = true;
                } else if (city.name == "Agadez" || city.name == "Oudane") {
                    desert_city_hexes.push_back(city.location);
                }
            }

            // Minimum hex distance from h to the nearest coastal hex.
            auto MinCoastDist = [&](const HexCoord& h) -> int {
                int d = 999;
                for (const auto& ch : coastal_hexes) d = std::min(d, h.Distance(ch));
                return d;
            };

            // Bonus for a hex that is part of the winning route geography.
            // Returns 0 for ordinary hexes, larger values for key hexes.
            auto WinRouteBonus = [&](const HexCoord& h) -> double {
                double bonus = 0.0;
                if (timbuktu_found_in_cities)
                    bonus += std::max(0.0, 20.0 - timbuktu_hex.Distance(h) * 3.0);
                for (const auto& dh : desert_city_hexes)
                    bonus += std::max(0.0, 14.0 - dh.Distance(h) * 3.0);
                int coast_d = MinCoastDist(h);
                if (coast_d < 999)
                    bonus += std::max(0.0, 14.0 - coast_d * 3.0);
                return bonus;
            };

            // Hard bonus when landing exactly ON a key hex.
            auto ExactKeyBonus = [&](const HexCoord& h) -> double {
                double bonus = 0.0;
                if (timbuktu_found_in_cities && h == timbuktu_hex) bonus += 30.0;
                for (const auto& dh : desert_city_hexes) if (h == dh) bonus += 20.0;
                if (coastal_hexes.count(h)) bonus += 20.0;
                return bonus;
            };

            // Does the current player already own a center at h?
            auto PlayerHasCenterAt = [&](const HexCoord& h) -> bool {
                auto it = trade_posts_locations_.find(h);
                if (it == trade_posts_locations_.end()) return false;
                for (const auto& p : it->second)
                    if (p.owner == current_player_color_ && p.type == TradePostType::kCenter)
                        return true;
                return false;
            };

            // Build sets of covered / uncovered city hexes for the current player.
            // "Covered" = player already has a trading CENTER there (collecting that city's
            // rare good via income). "Uncovered" = no center yet → priority expansion target.
            std::set<HexCoord> uncovered_city_hexes;
            std::set<HexCoord> covered_city_hexes;
            for (const auto& city : cities) {
                if (PlayerHasCenterAt(city.location))
                    covered_city_hexes.insert(city.location);
                else
                    uncovered_city_hexes.insert(city.location);
            }

            // Distance from h to the nearest uncovered city.
            auto MinUncoveredCityDist = [&](const HexCoord& h) -> int {
                int min_dist = 999;
                for (const auto& ch : uncovered_city_hexes)
                    min_dist = std::min(min_dist, h.Distance(ch));
                return min_dist;
            };

            // Large bonus when h IS an uncovered city (new rare good source).
            auto NewCityBonus = [&](const HexCoord& h) -> double {
                return uncovered_city_hexes.count(h) ? 35.0 : 0.0;
            };

            // ----------------------------------------------------------------
            // Region-spread helpers for the "rare good from N regions" win.
            // A city in a region the player has NO center in is worth far more
            // than a city in an already-covered region — it's the only path to
            // a new rare-good type.
            // ----------------------------------------------------------------
            const HeuristicContext h_ctx = CreateHeuristicContext();

            // Cities in regions the player hasn't planted a center in yet.
            std::set<HexCoord> new_region_city_hexes;
            for (const auto& city : cities) {
                if (PlayerHasCenterAt(city.location)) continue; // already a center here
                int region_id = GetGame()->GetRegionForHex(city.location);
                if (region_id == -1 ||
                    h_ctx.existing_center_regions.find(region_id) ==
                        h_ctx.existing_center_regions.end()) {
                    new_region_city_hexes.insert(city.location);
                }
            }

            // Distance to nearest city in a new region.
            auto MinNewRegionCityDist = [&](const HexCoord& h) -> int {
                int min_dist = 999;
                for (const auto& ch : new_region_city_hexes)
                    min_dist = std::min(min_dist, h.Distance(ch));
                return min_dist;
            };

            // Hard bonus for landing exactly on a new-region city hex.
            auto NewRegionCityBonus = [&](const HexCoord& h) -> double {
                return new_region_city_hexes.count(h) ? 45.0 : 0.0;
            };

            // ----------------------------------------------------------------
            // PRIMARY win-condition targeting: cities in board regions from
            // which the player hasn't yet collected any rare good.
            // The active win condition (end_game_cond_rare_good_each_region)
            // checks region coverage, not type-count, so we must target
            // regions rather than individual rare-good types.
            // ----------------------------------------------------------------
            const auto& my_rare_goods = GetPlayerRareGoods(current_player_id_);
            // Step 1: which regions already have at least one rare good?
            std::set<int> covered_rare_regions;
            for (const auto& [good_name, count] : my_rare_goods) {
                if (count > 0) {
                    for (const auto& city : cities) {
                        if (city.rare_good == good_name) {
                            int region = GetGame()->GetRegionForHex(city.location);
                            if (region != -1) covered_rare_regions.insert(region);
                            break;
                        }
                    }
                }
            }
            // Step 2: cities in regions we haven't covered yet (and no center there).
            std::set<HexCoord> missing_rare_city_hexes;
            for (const auto& city : cities) {
                if (PlayerHasCenterAt(city.location)) continue;
                int region = GetGame()->GetRegionForHex(city.location);
                if (region != -1 && covered_rare_regions.find(region) == covered_rare_regions.end()) {
                    missing_rare_city_hexes.insert(city.location);
                }
            }
            // How many regions the player has covered so far.
            const int num_covered_regions = static_cast<int>(covered_rare_regions.size());
            auto MinMissingRareCityDist = [&](const HexCoord& h) -> int {
                int d = 999;
                for (const auto& ch : missing_rare_city_hexes)
                    d = std::min(d, h.Distance(ch));
                return d;
            };
            auto MissingRareCityBonus = [&](const HexCoord& h) -> double {
                return missing_rare_city_hexes.count(h) ? 55.0 : 0.0;
            };

            // ----------------------------------------------------------------
            // INLAND priority set: Desert West/Center + Sahel West/Center are
            // the hardest-to-reach regions (one city each, far from Coastal).
            // These get stronger bonuses than the general missing-rare set.
            // ----------------------------------------------------------------
            std::set<int> inland_region_ids;
            for (int i = 1; i <= 6; ++i) {
                const std::string rname = GetGame()->GetRegionName(i);
                if (rname.find("Desert") != std::string::npos ||
                    rname.find("Sahel")  != std::string::npos)
                    inland_region_ids.insert(i);
            }
            // Only activate inland targeting once the player has ≥2 regions covered.
            // Before that, Agadez/Oudane/Linguère are too isolated to be good early targets —
            // steering players there hurts infrastructure build in the accessible Coastal area.
            std::set<HexCoord> inland_missing_city_hexes;
            if (num_covered_regions >= 1) {
                for (const auto& city : cities) {
                    if (!missing_rare_city_hexes.count(city.location)) continue;
                    int region = GetGame()->GetRegionForHex(city.location);
                    if (inland_region_ids.count(region))
                        inland_missing_city_hexes.insert(city.location);
                }
            }
            // True when the player has a center at an inland city but hasn't
            // yet collected the rare good (income urgently needed there).
            bool has_center_in_uncovered_inland = false;
            for (const auto& city : cities) {
                if (!PlayerHasCenterAt(city.location)) continue;
                int region = GetGame()->GetRegionForHex(city.location);
                if (inland_region_ids.count(region) &&
                    !covered_rare_regions.count(region)) {
                    has_center_in_uncovered_inland = true;
                    break;
                }
            }
            auto MinInlandMissingCityDist = [&](const HexCoord& h) -> int {
                int d = 999;
                for (const auto& ch : inland_missing_city_hexes)
                    d = std::min(d, h.Distance(ch));
                return d;
            };
            auto InlandMissingCityBonus = [&](const HexCoord& h) -> double {
                return inland_missing_city_hexes.count(h) ? 70.0 : 0.0;
            };

            switch (current_phase_) {

                // ----------------------------------------------------------------
                // kPlay: choose what to do this turn
                //
                // Progression the heuristic tries to enforce:
                //   1. Mancala to city → place post  (generate income)
                //   2. Take income regularly         (accumulate goods)
                //   3. Upgrade post → center         (unlock trade routes)
                //   4. Declare trade route           (win condition)
                // ----------------------------------------------------------------
                case Phase::kPlay: {
                    // h_ctx already computed before the switch.
                    const HeuristicContext& context = h_ctx;

                    // --- Inventory snapshot ---
                    int post_count = 0, center_count = 0, active_routes = 0;
                    int total_common = 0, total_rare = 0;
                    for (const auto& [hex, posts] : trade_posts_locations_) {
                        for (const auto& post : posts) {
                            if (post.owner == current_player_color_) {
                                if (post.type == TradePostType::kPost)   post_count++;
                                else                                     center_count++;
                            }
                        }
                    }
                    for (const auto& [name, cnt] : GetPlayerCommonGoods(current_player_id_))
                        total_common += cnt;
                    for (const auto& [name, cnt] : GetPlayerRareGoods(current_player_id_))
                        total_rare += cnt;
                    for (const auto& route : trade_routes_) {
                        if (route.owner == current_player_color_ && route.active) active_routes++;
                    }
                    int infrastructure   = post_count + center_count;
                    int total_goods      = total_common + total_rare;
                    bool can_upgrade_soon = (total_common >= rules.upgrade_cost_common - 1) ||
                                           (total_rare   >= rules.upgrade_cost_rare);

                    for (Action action : actions) {
                        double w = 1.0;

                        if (action == kPassAction) {
                            // Human-only safety valve — never choose voluntarily.
                            w = 0.0;

                        } else if (action == kIncomeAction) {
                            // Income is the critical enabler for upgrades and routes.
                            // Any post at all generates goods → income is urgent immediately.
                            w = 2.0 + infrastructure * 6.0;
                            // Urgency: goods are scarce and we need more to upgrade
                            if (infrastructure > 0 && total_goods < rules.upgrade_cost_common)
                                w += 15.0;
                            // Even more urgent when almost enough to upgrade
                            if (can_upgrade_soon) w += 10.0;
                            // Scale back once routes are flowing and goods are plentiful
                            if (active_routes > 0 && total_goods >= rules.upgrade_cost_common * 2)
                                w -= 8.0;
                            // Critical: center exists in inland region (Desert/Sahel) but
                            // rare good not yet collected — income is the unlock.
                            if (has_center_in_uncovered_inland) w += 30.0;
                            // Near-mandatory: supply is critically low (<=1) AND can't afford
                            // upgrade. Only income -> goods -> upgrade can free a post slot.
                            // Without this, player arrives at an isolated city and does nothing.
                            {
                                bool cannot_afford_upgrade =
                                    (total_common < rules.upgrade_cost_common) &&
                                    (total_rare   < rules.upgrade_cost_rare);
                                if (infrastructure > 0 &&
                                    context.posts_in_supply <= 1 &&
                                    cannot_afford_upgrade) {
                                    w = std::max(w, 150.0);
                                }
                            }

                        } else if (action >= kMancalaStartBase && action < kHexSelectionBase) {
                            // Mancala: the main action, especially before first post.
                            int hex_index   = action - kMancalaStartBase;
                            HexCoord hex    = GetGame()->IndexToCoord(hex_index);
                            int num_meeples = static_cast<int>(GetMeeplesAt(hex).size());
                            w = 8.0 + num_meeples * 2.0;
                            // Bonus when any city is reachable within this trip
                            int city_dist = MinCityDistance(hex, cities);
                            if (city_dist <= num_meeples + 1) w += 10.0;
                            // Extra pull when an UNCOVERED city is within reach
                            // (reaching an uncovered city = path to a new rare good)
                            int uncov_dist = MinUncoveredCityDist(hex);
                            if (uncov_dist <= num_meeples + 1) w += 15.0;
                            else if (uncov_dist <= num_meeples + 3) w += 5.0;
                            // Strongest pull: city in a NEW REGION (new rare-good type → win)
                            int new_region_dist = MinNewRegionCityDist(hex);
                            if (new_region_dist <= num_meeples + 1) w += 22.0;
                            else if (new_region_dist <= num_meeples + 3) w += 9.0;
                            // Primary win-condition: city producing a MISSING rare good type
                            int missing_rare_dist = MinMissingRareCityDist(hex);
                            if (missing_rare_dist <= num_meeples + 1) w += 28.0;
                            else if (missing_rare_dist <= num_meeples + 3) w += 12.0;
                            // Inland priority (Desert/Sahel): far isolated single-city regions.
                            // Use a pure distance gradient (independent of meeple count) so
                            // there is consistent northward pull even when the city is many
                            // moves away and the reachability condition never fires.
                            if (!inland_missing_city_hexes.empty()) {
                                int inland_dist = MinInlandMissingCityDist(hex);
                                w += std::max(0.0, 80.0 - inland_dist * 3.5);
                                if (inland_dist <= num_meeples + 1) w += 40.0;
                                else if (inland_dist <= num_meeples + 3) w += 18.0;
                            }
                            // Extra bonus when a KEY city (Timbuktu, desert, coast) is reachable
                            w += WinRouteBonus(hex) * 0.5;
                            // Long-range trips are especially powerful
                            if (num_meeples >= 4) w += 8.0;
                            // Urgency boost when the player still has no posts at all
                            if (infrastructure == 0) w += 10.0;

                        } else if (action >= kUpgradeBase && action < kPaymentBase) {
                            // Upgrade a post to a center — unlocks trade routes.
                            // CRITICAL: only city hexes produce rare goods via income.
                            // Non-city centers generate common goods only and do NOT
                            // contribute to the rare-good-from-N-regions win condition.
                            // We compute weights in a two-pass approach: first check
                            // if ANY city upgrade is available; if not, non-city
                            // upgrades get a reasonable fallback weight.
                            int hex_index = action - kUpgradeBase;
                            HexCoord hex  = GetGame()->IndexToCoord(hex_index);
                            const City* city_at_hex = GetGame()->GetCityAt(hex);
                            if (city_at_hex == nullptr) {
                                // Non-city upgrade. Check if the player has any city
                                // posts they could upgrade instead.
                                bool has_city_upgrade_option = false;
                                for (Action a2 : actions) {
                                    if (a2 >= kUpgradeBase && a2 < kPaymentBase) {
                                        HexCoord h2 = GetGame()->IndexToCoord(a2 - kUpgradeBase);
                                        if (GetGame()->GetCityAt(h2) != nullptr) {
                                            has_city_upgrade_option = true;
                                            break;
                                        }
                                    }
                                }
                                // If a city option exists, strongly deprioritize non-city.
                                // If no city option, use moderate weight to avoid deadlock.
                                w = has_city_upgrade_option ? 2.0 : 20.0;
                            } else {
                                // City upgrade: strong base, tiered by region novelty.
                                int region = GetGame()->GetRegionForHex(hex);
                                bool new_region = (region != -1 &&
                                    context.existing_center_regions.find(region) ==
                                        context.existing_center_regions.end());
                                if (missing_rare_city_hexes.count(hex)) {
                                    // Direct win-condition: city produces a MISSING rare good type
                                    w = 90.0;
                                } else if (new_region_city_hexes.count(hex)) {
                                    // New region city (already have this rare type) → direct win-condition progress
                                    w = 75.0;
                                } else if (new_region) {
                                    // City in a new region (already have another center there)
                                    w = 55.0;
                                } else if (uncovered_city_hexes.count(hex)) {
                                    // Same-region city, not yet covered
                                    w = 40.0;
                                } else {
                                    // City we already have a center at (least useful)
                                    w = 10.0;
                                }
                                // Extra pull for Timbuktu/desert/coast (trade-route win)
                                w += ExactKeyBonus(hex) + WinRouteBonus(hex) * 0.3;
                            }

                            // Supply exhausted: upgrading is the ONLY way to free a post
                            // slot so the player can place again. Make it near-mandatory.
                            // Even a non-city upgrade is better than being permanently stuck.
                            if (rules.posts_per_player != kUnlimitedPosts &&
                                context.posts_in_supply == 0) {
                                w = std::max(w, 200.0); // near-mandatory: mancala can reach 80+ with inland gradients
                            }
                            if (rules.posts_per_player != kUnlimitedPosts &&
                                context.posts_in_supply == 1) {
                                // Supply critically low: strongly prefer upgrading before stuck
                                w = std::max(w, 110.0);
                            }
                        }

                        action_weights[action] = std::max(0.0, w);
                    }
                    break;
                }

                // ----------------------------------------------------------------
                // kMancalaStep: drop one meeple per step.
                // Steer toward key win-condition hexes (Timbuktu, desert, coast)
                // and toward any city so the token can land there.
                // ----------------------------------------------------------------
                case Phase::kMancalaStep: {
                    int steps_left = static_cast<int>(meeples_in_hand_.size());
                    // NEXT TO-DO: at the beginning of a training run, have 
                    // train_mali_ba drop a file with a random number between
                    // 0 and 1 for mult_add_in, and between .5 and 2 for add_add_in.
                    // this process picks that set of numbers up and uses them below.
                    // We need a shell script that will loop 100 times. It should:
                    // 0. start in /media/robp/UD/Projects/open_spiel/open_spiel/python/games/mali_ba
                    // Remove previous log files: rm /tmp/diag*.log /tmp/temp.log /tmp/test_*.log /tmp/mali_ba*.log
                    // 1. run the training script (which will drop the random numbers)
                    // python train_mali_ba.py --heuristic_only --num_actors 2 --bootstrap_episodes 100 --num_episodes 100 --config_file mali_ba.ini    2>&1 | tee /tmp/test_region_fix.log
                    // 
                    // 2. when the training is done grep to see how many wins this set of random
                    // add_ins produced.  Save that information in a file. Use the grep statement:
                    // grep "HEURISTIC_DIAG" /tmp/mali_ba.*.log | grep "Game ended:" | grep -o "reason='[^']*'" | sort | uniq -c
                    // BUT! add a third line that totals all the win conditions for a total number
                    // of wins.
                    // 3. iterate the loop until we have 100 sets of random numbers and their 
                    // corresponding win totals. Then we can analyze the results to see if there 
                    // are any correlations between the random add_in values and the win totals, 
                    // which could inform future heuristic tuning.
                    // We should analyze if any particular ranges of add_in values correlate with higher win totals, 
                    // and if so, consider adjusting the heuristic to use those ranges more consistently.
                    auto [mult_add_in, add_add_in] = LoadHeuristicParams();

                    for (Action action : actions) {
                        if (action == kPassAction) {
                            action_weights[action] = 0.1;
                            continue;
                        }
                        int dir         = action - kMancalaDirectionBase;
                        HexCoord target = current_mancala_hex_ + kHexDirections[dir];
                        double w        = 1.0;

                        // Close distance to the nearest city (any city)
                        int city_dist = MinCityDistance(target, cities);
                        w += std::max(0.0, 8.0 - city_dist * (2.0+mult_add_in));
                        if (city_dist <= steps_left) w += (8.0+add_add_in);
                        if (city_dist == 0)          w += (12.0+add_add_in); // passing through a city hex

                        // Extra pull toward UNCOVERED cities (new rare good sources)
                        int uncov_dist = MinUncoveredCityDist(target);
                        w += std::max(0.0, 10.0 - uncov_dist * (2.5+mult_add_in));
                        if (uncov_dist <= steps_left) w += (10.0+add_add_in);
                        if (uncov_dist == 0)          w += (15.0+add_add_in); // stepping through uncovered city
                        // Stronger pull toward cities in NEW REGIONS (new rare-good type)
                        int new_region_dist = MinNewRegionCityDist(target);
                        w += std::max(0.0, 14.0 - new_region_dist * (2.5+mult_add_in));
                        if (new_region_dist <= steps_left) w += (14.0+add_add_in);
                        if (new_region_dist == 0)          w += (20.0+add_add_in); // stepping through new-region city
                        // Primary win-condition: city producing a MISSING rare good type
                        int missing_rare_dist_s = MinMissingRareCityDist(target);
                        w += std::max(0.0, 16.0 - missing_rare_dist_s * (2.5+mult_add_in));
                        if (missing_rare_dist_s <= steps_left) w += (16.0+add_add_in);
                        if (missing_rare_dist_s == 0)          w += (24.0+add_add_in); // stepping through missing-rare city
                        // Inland priority (Desert/Sahel): pure gradient + reachability
                        if (!inland_missing_city_hexes.empty()) {
                            int inland_dist_s = MinInlandMissingCityDist(target);
                            w += std::max(0.0, 70.0 - inland_dist_s * (3.0+mult_add_in));
                            if (inland_dist_s <= steps_left) w += (28.0+add_add_in);
                            if (inland_dist_s == 0)          w += (40.0+add_add_in);
                        }

                        // Additional bias toward key win-route hexes
                        w += WinRouteBonus(target) * (0.6+mult_add_in);
                        w += ExactKeyBonus(target) * (0.5+mult_add_in); // partial: full reward is at token step

                        // Meeple density: consolidate for high-range future mancalas
                        w += GetMeeplesAt(target).size() * (2.0+mult_add_in);

                        // Centrality: more neighbours = more options for subsequent steps
                        w += ValidNeighbourCount(target, valid_hexes) * (0.5+mult_add_in);

                        action_weights[action] = std::max(0.1, w);
                    }
                    break;
                }

                // ----------------------------------------------------------------
                // kMancalaTokenStep: final token placement.
                // Strongly prefer key win-route hexes (Timbuktu, desert, coast)
                // so a post can be placed there for the eventual winning route.
                // ----------------------------------------------------------------
                case Phase::kMancalaTokenStep: {
                    for (Action action : actions) {
                        if (action == kPassAction) {
                            action_weights[action] = 0.1;
                            continue;
                        }
                        int dir         = action - kMancalaDirectionBase;
                        HexCoord target = current_mancala_hex_ + kHexDirections[dir];
                        double w        = 1.0;

                        // Landing on any city is good; UNCOVERED cities are the top priority
                        // (no center there yet → landing sets up a post → new rare good)
                        int city_dist = MinCityDistance(target, cities);
                        if      (city_dist == 0) w += 25.0;
                        else if (city_dist == 1) w += 12.0;
                        else if (city_dist == 2) w +=  4.0;

                        // Strong extra bonus for landing on an uncovered city hex
                        w += NewCityBonus(target) * 1.5; // up to +52 on top of the city bonus
                        // Highest priority: landing on a NEW-REGION city (new rare-good type)
                        w += NewRegionCityBonus(target) * 1.5; // up to +67 — trumps same-region cities
                        // Primary: city with a MISSING rare good type (highest priority)
                        w += MissingRareCityBonus(target) * 1.5; // up to +82
                        // Inland priority: Desert/Sahel cities get a gradient + exact bonus
                        // to overcome the gravitational pull of nearby Coastal cities
                        if (!inland_missing_city_hexes.empty()) {
                            int idist = MinInlandMissingCityDist(target);
                            w += std::max(0.0, 80.0 - idist * 4.0); // +50 at city, fades over 12 hexes
                            w += InlandMissingCityBonus(target) * 1.5; // +105 on exact landing
                        }

                        // Hard bonus for landing exactly on a key win-route hex
                        w += ExactKeyBonus(target);

                        // Gradient bonus for proximity to key hexes
                        w += WinRouteBonus(target) * 0.7;

                        // Can we place a post immediately after landing?
                        if (CanPlaceTradingPostAt(target, current_player_color_)) w += 18.0;

                        // Meeple density → better range on next mancala
                        w += GetMeeplesAt(target).size() * 3.0;

                        // Centrality
                        w += ValidNeighbourCount(target, valid_hexes) * 1.0;

                        action_weights[action] = std::max(0.1, w);
                    }
                    break;
                }

                // ----------------------------------------------------------------
                // kOptionalPost: almost always place — posts generate income and
                // are prerequisites for centers and trade routes.
                // Key-city posts are especially critical for the win condition.
                // ----------------------------------------------------------------
                case Phase::kOptionalPost: {
                    int supply = player_posts_supply_[current_player_id_];
                    bool in_city = (GetGame()->GetCityAt(last_action_hex_) != nullptr);
                    for (Action action : actions) {
                        if (action == kPlacePostAction) {
                            double w;
                            if (in_city) {
                                // City posts produce rare goods on income — always valuable.
                                w = 15.0;
                                if (inland_missing_city_hexes.count(last_action_hex_))
                                    w += 90.0; // inland (Desert/Sahel) isolated city — highest priority
                                else if (missing_rare_city_hexes.count(last_action_hex_))
                                    w += 55.0; // city with MISSING rare good type → new rare-good type (win progress)
                                else if (new_region_city_hexes.count(last_action_hex_))
                                    w += 40.0; // new region city
                                else if (uncovered_city_hexes.count(last_action_hex_))
                                    w += 20.0; // same-region city, still uncovered
                                else
                                    w += 5.0;  // city already has our center
                            } else {
                                // Non-city hex: posts here don't generate income directly.
                                // Only place if it's a key strategic location.
                                w = 2.0;
                            }
                            // Strong extra bonus at key win-route locations (may elevate non-city)
                            w += ExactKeyBonus(last_action_hex_);
                            w += WinRouteBonus(last_action_hex_) * 0.5;
                            // Only hesitate when completely out of supply
                            if (supply == 0) w = 0.0;
                            action_weights[action] = w;
                        } else { // kPassAction for kOptionalPost
                            // In a city: almost never skip. Outside a city: prefer to skip.
                            action_weights[action] = in_city ? 0.05 : 4.0;
                        }
                    }
                    break;
                }

                // ----------------------------------------------------------------
                // kOptionalPostPayment: spend the most abundant good to
                // preserve scarce ones needed for upgrades.
                // ----------------------------------------------------------------
                case Phase::kOptionalPostPayment: {
                    const auto& common_goods = GetPlayerCommonGoods(current_player_id_);
                    const auto& goods_list   = GoodsManager::GetInstance().GetCommonGoodsList();
                    for (Action action : actions) {
                        int good_id = action - kPaymentBase;
                        int count   = 0;
                        if (good_id >= 0 && good_id < static_cast<int>(goods_list.size())) {
                            auto it = common_goods.find(goods_list[good_id]);
                            if (it != common_goods.end()) count = it->second;
                        }
                        action_weights[action] = std::max(1.0, static_cast<double>(count));
                    }
                    break;
                }

                // ----------------------------------------------------------------
                // kOptionalRoute: almost always create — trade routes satisfy the
                // prerequisite (need ≥2) and potentially the win condition.
                // Routes anchored at Timbuktu / desert / coastal hubs are best.
                // ----------------------------------------------------------------
                case Phase::kOptionalRoute: {
                    int active_routes = 0;
                    for (const auto& route : trade_routes_) {
                        if (route.owner == current_player_color_ && route.active) active_routes++;
                    }
                    double anchor_bonus = ExactKeyBonus(last_action_hex_)
                                       + WinRouteBonus(last_action_hex_) * 0.5;

                    bool has_timbuktu_center = timbuktu_found_in_cities &&
                                               PlayerHasCenterAt(timbuktu_hex);

                    // Build the set of ALL hexes (not just cities) in existing routes
                    // so we can compute per-candidate path overlap fractions.
                    std::set<HexCoord> hexes_in_existing_routes;
                    std::set<HexCoord> cities_in_existing_routes;
                    for (const auto& route : trade_routes_) {
                        if (route.owner == current_player_color_ && route.active) {
                            for (const auto& rhex : route.hexes) {
                                hexes_in_existing_routes.insert(rhex);
                                if (GetGame()->GetCityAt(rhex) != nullptr)
                                    cities_in_existing_routes.insert(rhex);
                            }
                        }
                    }

                    // Fetch the candidate paths so we can inspect each route's hexes and cities.
                    auto possible_routes = FindPossibleTradeRoutes(
                        current_player_color_, true, last_action_hex_, 5);

                    for (Action action : actions) {
                        if (action == kPassAction) {
                            // Never pass while still building toward 2-route prerequisite
                            action_weights[action] = (active_routes >= 4) ? 2.0 : 0.1;
                        } else {
                            int route_idx = action - kRouteBase;
                            bool has_new_city = false;
                            bool has_missing_region_city = false;

                            // Hex-level overlap: fraction of this route's hexes already
                            // used by another route. 100% overlap = pure corridor duplication.
                            int total_hexes = 0;
                            int overlap_hexes = 0;

                            if (route_idx < (int)possible_routes.size()) {
                                for (const auto& rhex : possible_routes[route_idx]) {
                                    total_hexes++;
                                    if (hexes_in_existing_routes.count(rhex))
                                        overlap_hexes++;
                                    if (GetGame()->GetCityAt(rhex) != nullptr &&
                                        cities_in_existing_routes.find(rhex) ==
                                            cities_in_existing_routes.end()) {
                                        has_new_city = true;
                                        if (missing_rare_city_hexes.count(rhex))
                                            has_missing_region_city = true;
                                    }
                                }
                            }

                            // overlap_fraction: 0.0 = entirely new path, 1.0 = pure duplicate
                            double overlap_fraction = (total_hexes > 1)
                                ? (double)overlap_hexes / total_hexes : 0.0;
                            // Strong path-overlap penalty: up to -50 for a fully duplicated route.
                            double overlap_penalty = overlap_fraction * 50.0;

                            double w;
                            if (has_missing_region_city) {
                                // Passes through a city in a missing rare region — top priority.
                                // Still penalise if the path is mostly a re-tread.
                                w = 60.0 + anchor_bonus - overlap_penalty * 0.5;
                            } else if (has_new_city) {
                                // Introduces a new city but may share corridor with others.
                                w = 25.0 + anchor_bonus - overlap_penalty;
                            } else {
                                // All cities already covered — no geographic value.
                                w = 0.5;
                            }
                            if (has_timbuktu_center) w += 10.0;
                            // Allow highly overlapping routes to drop below pass weight (2.0).
                            action_weights[action] = std::max(0.1, w);
                        }
                    }
                    break;
                }

                // ----------------------------------------------------------------
                // kPlaceToken: pick a starting hex close to cities in new regions.
                // Going later in turn order means cities get claimed; steer toward
                // hexes that are close to the most uncovered-region cities so every
                // player can spread geographically from turn one.
                // ----------------------------------------------------------------
                case Phase::kPlaceToken: {
                    for (Action action : actions) {
                        int hex_index = action - kHexSelectionBase;
                        HexCoord hex  = GetGame()->IndexToCoord(hex_index);
                        double w = 1.0;

                        // Sum proximity to every city in a new region.
                        // Sum proximity to every city in a new region.
                        // Each new-region city within 5 hexes contributes a bonus.
                        for (const auto& ch : new_region_city_hexes) {
                            int d = hex.Distance(ch);
                            w += std::max(0.0, 12.0 - d * 2.0);
                        }
                        // Proximity to cities with MISSING rare good types
                        for (const auto& ch : missing_rare_city_hexes) {
                            int d = hex.Distance(ch);
                            w += std::max(0.0, 15.0 - d * 2.0);
                        }
                        // Inland priority: use raw inland_region_ids (NOT inland_missing_city_hexes
                        // which is gated on num_covered_regions>=2 and is always empty here at
                        // game start). Directly iterate cities in Desert/Sahel regions.
                        for (const auto& city : cities) {
                            int region = GetGame()->GetRegionForHex(city.location);
                            if (inland_region_ids.count(region)) {
                                int d = hex.Distance(city.location);
                                w += std::max(0.0, 60.0 - d * 3.0); // range: 20 hexes, peak 60 to outcompete coastal pile-up
                            }
                        }
                        w += ValidNeighbourCount(hex, valid_hexes) * 1.5;

                        action_weights[action] = std::max(1.0, w);
                    }
                    break;
                }

                // ----------------------------------------------------------------
                // All other phases: uniform random
                // ----------------------------------------------------------------
                default: {
                    for (Action action : actions) {
                        action_weights[action] = 1.0;
                    }
                    break;
                }
            }

            return action_weights;
        }

        // Selects one action using the heuristic weights for every phase.
        Action Mali_BaState::SelectHeuristicRandomAction() const {
            std::map<Action, double> action_weights = GetHeuristicActionWeights();

            if (action_weights.empty()) {
                // Last-resort uniform fallback (should not normally occur)
                std::vector<Action> actions = LegalActions();
                if (actions.empty()) return kInvalidAction;
                std::uniform_int_distribution<> dist(0, actions.size() - 1);
                return actions[dist(rng_)];
            }

            std::vector<Action> actions;
            std::vector<double> weights;
            actions.reserve(action_weights.size());
            weights.reserve(action_weights.size());
            for (const auto& [action, weight] : action_weights) {
                actions.push_back(action);
                weights.push_back(weight);
            }

            // Fallback to uniform if all weights collapsed to zero
            if (std::all_of(weights.begin(), weights.end(),
                            [](double w) { return w <= 1e-6; })) {
                std::uniform_int_distribution<> dist(0, actions.size() - 1);
                return actions[dist(rng_)];
            }

            std::discrete_distribution<> dist(weights.begin(), weights.end());
            return actions[dist(rng_)];
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
            // Bug fix (2026-07-19): the actual [setup]-section writer used by the
            // self-play training pipeline (train_mali_ba.py) calls this method's
            // sibling State::serialize() directly, which -- correctly, for its own
            // per-move-log purpose -- only captures dynamic state (tokens, meeples,
            // trade posts, etc.), never the static board-layout config (valid_hexes/
            // cities/grid_radius/num_players). That left every produced replay file's
            // [setup] section without the data the GUI replay loader needs to
            // reconstruct the board (ui/visualizer.py's BoardVisualizer.__init__
            // raises "Failed to obtain valid_hexes"). Fix: build [setup] as dynamic
            // state (via Serialize(), reused rather than duplicated) merged with the
            // static config fields below, so a single call produces everything the
            // replay loader expects. See DISTRIBUTED_TRAINING.md and main.py's
            // MODE_GUI_REPLAY branch for the (now-removable) Python-side ini-file
            // fallback this was worked around with before this fix landed.
            json setup = json::parse(Serialize());
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

        void Mali_BaState::LogHeuristicEndGameDiagnostic() const {
            // Logs a one-line summary per player at game end so we can diagnose
            // where the win-condition chain breaks during bootstrap.
            const auto& cities = GetGame()->GetCities();
            int num_players = NumPlayers();

            for (Player p = 0; p < num_players; ++p) {
                PlayerColor pc = GetPlayerColor(p);

                // Count posts and centers; track center hexes and their regions.
                int post_count = 0, center_count = 0;
                std::set<int> center_regions;
                std::string center_hexes;
                for (const auto& [hex, posts] : trade_posts_locations_) {
                    for (const auto& post : posts) {
                        if (post.owner != pc) continue;
                        if (post.type == TradePostType::kPost) {
                            post_count++;
                        } else {
                            center_count++;
                            int region_id = GetGame()->GetRegionForHex(hex);
                            if (region_id != -1) center_regions.insert(region_id);
                            center_hexes += "(" + std::to_string(hex.x) + "," +
                                            std::to_string(hex.y) + ") ";
                        }
                    }
                }

                // Count active trade routes.
                int route_count = 0;
                for (const auto& route : trade_routes_) {
                    if (route.owner == pc && route.active) route_count++;
                }

                // Collect rare goods and which regions they come from.
                int unique_rare = 0;
                std::set<int> rare_good_regions;
                std::string rare_goods_str;
                for (const auto& [good_name, good_count] : rare_goods_[p]) {
                    if (good_count > 0) {
                        unique_rare++;
                        rare_goods_str += good_name + ":" + std::to_string(good_count) + " ";
                        for (const auto& city : cities) {
                            if (city.rare_good == good_name) {
                                int region_id = GetGame()->GetRegionForHex(city.location);
                                if (region_id != -1) rare_good_regions.insert(region_id);
                                break;
                            }
                        }
                    }
                }

                // Common goods total.
                int total_common = 0;
                for (const auto& [name, cnt] : GetPlayerCommonGoods(p)) total_common += cnt;

                LOG_INFO("[HEURISTIC_DIAG] Player ", p,
                    " | posts=", post_count,
                    " centers=", center_count,
                    " center_regions=", center_regions.size(),
                    " rare_regions=", rare_good_regions.size(),
                    " routes=", route_count,
                    " unique_rare=", unique_rare,
                    " common_goods=", total_common,
                    " | centers_at=[", center_hexes, "]",
                    " rare=[", rare_goods_str, "]");
            }
            LOG_INFO("[HEURISTIC_DIAG] Game ended: reason=\'", game_end_reason_, "\'",
                " moves=", history_.size());
        }


    } // namespace mali_ba
} // namespace open_spiel