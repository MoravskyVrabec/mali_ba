// mali_ba_state_core.cc
// Core state implementation (constructor, basic state API functions)

#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/mali_ba_observer.h"
#include "open_spiel/games/mali_ba/hex_grid.h"

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/observer.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/utils/tensor_view.h" // THIS INCLUDE for DataType

namespace open_spiel
{
    namespace mali_ba
    {
        namespace { // Use anonymous namespace for local constants/helpers
            // Max values needed for plane indexing (match observer)
            constexpr int kMaxPlayersObs = 5;
            constexpr int kNumMeepleColorsObs = 10;

            class VectorTensorAllocator : public Allocator {
            public:
                static int GetSize(const absl::InlinedVector<int, 4>& shape) { // Pass by const ref here is fine
                    if (shape.empty()) return 0;
                    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
                }

                SpanTensor Get(absl::string_view name,
                                const absl::InlinedVector<int, 4>& shape) override {
                    SPIEL_CHECK_EQ(name, "observation");
                    buffer_.resize(GetSize(shape));

                    SpanTensorInfo info(name, shape);

                    return SpanTensor(info, absl::MakeSpan(buffer_));
                }

                const std::vector<float>& GetBuffer() const { return buffer_; }

            private:
                std::vector<float> buffer_;
            };
        } // namespace

        // --- State Constructor ---
        Mali_BaState::Mali_BaState(std::shared_ptr<const Game> game)
            : State(game),
              current_phase_(Phase::kSetup),
              current_player_id_(kChancePlayerId),
              current_player_color_(PlayerColor::kEmpty),
              next_route_id_(1)
        {
            LOG_DEBUG("Mali_BaState::Constructor: ENTRY");
            rng_.seed(GetGame()->GetRNGSeed());
            
            int num_players = game_->NumPlayers();
            common_goods_.resize(num_players);
            rare_goods_.resize(num_players);
            player_posts_supply_.resize(num_players);
            // Initialize cumulative_returns_
            cumulative_returns_.resize(game_->NumPlayers(), 0.0);

            InitializeBoard();
            LOG_DEBUG("Mali_BaState::Constructor: EXIT");
        }

        // --- State Copy Constructor ---
        Mali_BaState::Mali_BaState(const Mali_BaState& other)
            : State(other),
              current_phase_(other.current_phase_),
              current_player_id_(other.current_player_id_),
              current_player_color_(other.current_player_color_),
              player_token_locations_(other.player_token_locations_),
              player_posts_supply_(other.player_posts_supply_),
              hex_meeples_(other.hex_meeples_),
              trade_posts_locations_(other.trade_posts_locations_),
              common_goods_(other.common_goods_),
              rare_goods_(other.rare_goods_),
              trade_routes_(other.trade_routes_),
              next_route_id_(other.next_route_id_),
              moves_history_(other.moves_history_),
              rng_(other.rng_),
              is_terminal_(other.is_terminal_),
              undo_stack_(other.undo_stack_),
              game_end_triggered_by_player_(other.game_end_triggered_by_player_),
              winning_player_(other.winning_player_),
              game_end_reason_(other.game_end_reason_)
        {
            // Caches are intentionally NOT copied. They will be regenerated on the clone when needed.
            // Copy the cumulative returns
            cumulative_returns_ = other.cumulative_returns_;
        }


        std::unique_ptr<State> Mali_BaState::Clone() const {
            return std::make_unique<Mali_BaState>(*this);
        }

        // --- Delegating Getters ---
        const Mali_BaGame *Mali_BaState::GetGame() const {
          // This performs the cast from the base Game pointer to the derived Mali_BaGame pointer.
          return static_cast<const Mali_BaGame *>(game_.get());
        }
        const std::set<HexCoord>& Mali_BaState::ValidHexes() const {
          return GetGame()->GetValidHexes();
        }
        const std::vector<City>& Mali_BaState::GetCities() const {
          return GetGame()->GetCities();
        }
        int Mali_BaState::GridRadius() const {
          return GetGame()->GetGridRadius();
        }
        bool Mali_BaState::IsValidHex(const HexCoord &hex) const {
          return GetGame()->GetValidHexes().count(hex) > 0;
        }


        void Mali_BaState::InitializeBoard() {
            const GameRules& rules = GetGame()->GetRules();
            player_token_locations_.clear();
            hex_meeples_.clear();
            trade_posts_locations_.clear();
            for (const auto &hex : GetGame()->GetValidHexes()) {
                hex_meeples_[hex] = {};
                trade_posts_locations_[hex] = {};
            }
            for (int i = 0; i < game_->NumPlayers(); ++i) {
                player_posts_supply_[i] = rules.posts_per_player;
            }
        }

        Player Mali_BaState::CurrentPlayer() const {
            if (is_terminal_) return kTerminalPlayerId;
            return current_player_id_;
        }

        bool Mali_BaState::IsChanceNode() const {
            return current_player_id_ == kChancePlayerId;
        }

        std::vector<std::pair<Action, double>> Mali_BaState::ChanceOutcomes() const {
            SPIEL_CHECK_TRUE(IsChanceNode());
            return {{kChanceSetupAction, 1.0}};
        }
        
        // =============================================================================
        // ACTION GENERATION & DECODING LOGIC
        // =============================================================================
        // This function is now the source of truth for legal action generation.
        LegalActionsResult Mali_BaState::GetLegalActionsAndCounts() const {
            if (cached_legal_actions_result_) {
                return *cached_legal_actions_result_;
            }

            LegalActionsResult result;
            if (IsTerminal()) {
                cached_legal_actions_result_ = result;
                return result;
            }

            // --- SETUP PHASE ---
            if (IsChanceNode()) {
                result.actions.push_back(kChanceSetupAction);
                cached_legal_actions_result_ = result;
                return result;
            }

            // --- PLACE TOKEN PHASE ---
            if (current_phase_ == Phase::kPlaceToken) {
                // The action ID for placing a token must be distinct from other actions.
                // We will use a simple mapping: Action = hex_index.
                // This is safe because kMancalaActionBase, kUpgradeActionBase etc. are large numbers.
                for (int i = 0; i < GetGame()->NumHexes(); ++i) {
                    HexCoord hex = GetGame()->IndexToCoord(i);
                    // Check if the hex is a valid placement target
                    if (player_token_locations_.find(hex) == player_token_locations_.end() &&
                        GetGame()->GetCityAt(hex) == nullptr) {

                        // The action IS the hex index added to the base. 
                        // This is fine as long as other actions have large IDs.
                        Action place_action = kPlaceTokenActionBase + i;
                        
                        SPIEL_CHECK_LT(place_action, kUpgradeActionBase); // Ensure no overlap
                        
                        result.actions.push_back(place_action);
                        result.counts.place_token_moves++;
                    }
                }
                cached_legal_actions_result_ = result;
                return result;
            }

            // --- PLAY PHASE ---
            // Now, we call the helper functions to generate each type of move.

            // Get the type of the CURRENT player for rule applications.
            const auto& player_types = GetGame()->GetPlayerTypes();
            SPIEL_CHECK_GE(current_player_id_, 0); // This must be a player decision node.
            SPIEL_CHECK_LT(current_player_id_, player_types.size());
            PlayerType current_player_type = player_types[current_player_id_];
            
            // 1. Pass Action - only for human players
            if (current_player_type == PlayerType::kHuman) {
                result.actions.push_back(kPassAction);
                result.counts.pass_moves++;
            }

            // 2. Mancala Moves
            std::vector<Move> mancala_moves = GenerateMancalaMoves();
            for (const auto& move : mancala_moves) {
                // This part needs a proper Action encoding. We will fix this in the next step.
                // For now, let's assume a function MoveToAction exists.
                Action action = MoveToAction(move); // We will create this function
                if (action != kInvalidAction) {
                    result.actions.push_back(action);
                    result.counts.mancala_moves++;
                }
            }

            // 3. Upgrade Moves (including compound moves with FREE trade routes)
            std::vector<Move> upgrade_moves = GenerateTradePostUpgradeMoves();
            for (const auto& move : upgrade_moves) {
                Action action = MoveToAction(move);
                if (action != kInvalidAction) {
                    result.actions.push_back(action);
                    result.counts.upgrade_moves++;
                }
            }
            
            // 4. Income Moves
            // Prevent AI/Heuristic players from choosing 'income' twice in a row.
            bool allow_income_move = true; // By default, income moves are allowed.

            // The rule only applies to non-human players.
            if (current_player_type != PlayerType::kHuman) {
                // Search backwards through history to find the last move by this player.
                // We use a reverse iterator for efficiency.
                for (auto it = moves_history_.rbegin(); it != moves_history_.rend(); ++it) {
                    const Move& previous_move = *it;
                    if (previous_move.player == current_player_color_) {
                        // This is the last move this player made. Was it an income action?
                        if (previous_move.type == ActionType::kIncome) {
                            allow_income_move = false; // Disallow income this turn.
                            // LOG_INFO("Player ", PlayerColorToString(current_player_color_), 
                            //         " (AI/Heuristic) previously played income. Disabling income action.");
                        }
                        // We found the last move by this player, so we can stop searching.
                        break; 
                    }
                }
            }
            // Only generate income moves if they are allowed for this player type and situation.
            if (allow_income_move) {
                std::vector<Move> income_moves = GenerateIncomeMoves();
                for (const auto& move : income_moves) {
                    Action action = MoveToAction(move);
                    if (action != kInvalidAction) {
                        result.actions.push_back(action);
                        result.counts.income_moves++;
                    }
                }
            }

            // 5. STANDALONE Trade Route Moves
            std::vector<Move> trade_route_moves = GenerateTradeRouteMoves();
            for (const auto& move : trade_route_moves) {
                Action action = MoveToAction(move);
                if (action != kInvalidAction) {
                    result.actions.push_back(action);
                    result.counts.trade_route_create_moves++;
                }
            }

            cached_legal_actions_result_ = result;
            return result;
        }

        std::vector<Action> Mali_BaState::LegalActions() const {
            if (cached_legal_actions_result_) {
                return cached_legal_actions_result_->actions;
            }
            // Temporarily cast away const to populate the cache
            const_cast<Mali_BaState*>(this)->GetLegalActionsAndCounts();
            return cached_legal_actions_result_->actions;
        }

        void Mali_BaState::ApplyPlaceTokenMove(const Move &move) {
            // ... (this function is correct and remains unchanged) ...
             SPIEL_CHECK_EQ(move.type, ActionType::kPlaceToken);
            SPIEL_CHECK_TRUE(IsValidHex(move.start_hex));

            AddTokenAt(move.start_hex, move.player);

            const int required_tokens_per_player = GetGame()->GetTokensPerPlayer();
            const int num_players = game_->NumPlayers();

            std::map<PlayerColor, int> player_token_counts;
            for (const auto& [hex, colors] : player_token_locations_) {
                for (PlayerColor color : colors) {
                    if (color != PlayerColor::kEmpty) {
                        player_token_counts[color]++;
                    }
                }
            }

            bool all_players_placed_tokens = true;
            if (player_token_counts.size() < num_players) {
                all_players_placed_tokens = false;
            } else {
                const auto& player_colors = GetGame()->GetPlayerColors();
                for (const auto& color : player_colors) {
                    int count = 0;
                    auto it = player_token_counts.find(color);
                    if (it != player_token_counts.end()) {
                        count = it->second;
                    }
                    if (count < required_tokens_per_player) {
                        all_players_placed_tokens = false;
                        break;
                    }
                }
            }

            if (all_players_placed_tokens) {
                LOG_INFO("✅ All tokens placed - transitioning to PLAY phase");
                SetCurrentPhase(Phase::kPlay);
                current_player_id_ = 0;
                current_player_color_ = GetPlayerColor(current_player_id_);
            }
        }

        // This function now ONLY handles the token and meeple movement.
        void Mali_BaState::ApplyMancalaMove(const Move& move) {
            SPIEL_CHECK_EQ(move.type, ActionType::kMancala);
            SPIEL_CHECK_FALSE(move.path.empty());
            SPIEL_CHECK_TRUE(HasTokenAt(move.start_hex, move.player));

            const std::vector<MeepleColor> meeples_to_distribute = GetMeeplesAt(move.start_hex);
            const int num_meeples = meeples_to_distribute.size();
            const HexCoord& end_hex = move.path[0];
            
            // Pathfinding logic 
            std::vector<HexCoord> actual_path;
            if (num_meeples == 0) {
                SPIEL_CHECK_EQ(move.start_hex.Distance(end_hex), 1);
                actual_path = {end_hex};
            } else {
                actual_path = FindShortestPath(move.start_hex, end_hex, num_meeples);
                if (actual_path.empty()) {
                    LOG_WARN("No valid Mancala path found for ", move.start_hex.ToString(), " -> ", end_hex.ToString(), ". Treating as a pass.");
                    return;
                }
            }

            // State changes for token/meeple movement 
            bool removed = RemoveTokenAt(move.start_hex, move.player);
            SPIEL_CHECK_TRUE(removed);
            hex_meeples_.erase(move.start_hex);
            
            AddTokenAt(end_hex, move.player);

            for (int i = 0; i < num_meeples && i < actual_path.size() - 1; ++i) {
                const HexCoord& dest_hex = actual_path[i];
                hex_meeples_[dest_hex].push_back(meeples_to_distribute[i]);
            }
            
        }

        // This function ONLY handles placing the post and paying for it.
        void Mali_BaState::ApplyPlacePostFromMancala(const Move &move) {
            const HexCoord& end_hex = move.path[0];
            SPIEL_CHECK_TRUE(CanPlaceTradingPostAt(end_hex, move.player));

            AddTradingPost(end_hex, move.player, TradePostType::kPost);
            
            // Payment logic
            auto& meeples_at_dest = hex_meeples_[end_hex];
            if (!meeples_at_dest.empty()) {
                meeples_at_dest.pop_back();
                //LOG_DEBUG("Paid for trading post with a meeple at ", end_hex.ToString());
            } else {
                Player player_id = GetPlayerId(move.player);
                bool paid = false;
                if (player_id != kInvalidPlayer) {
                    for (auto& [name, count] : common_goods_[player_id]) {
                        if (count > 0) { count--; paid = true; break; }
                    }
                    if (!paid) {
                        for (auto& [name, count] : rare_goods_[player_id]) {
                            if (count > 0) { count--; paid = true; break; }
                        }
                    }
                }
                if (!paid) {
                    SpielFatalError("ApplyPlacePostFromMancala: No meeple or resource to pay for post.");
                }
                LOG_DEBUG("Paid for trading post with a resource.");
            }
        }

        void Mali_BaState::ApplyTradingPostUpgrade(const Move& move) {
            SPIEL_CHECK_EQ(move.type, ActionType::kPlaceTCenter);
            
            // Check if the player actually has a post to upgrade.
            bool has_player_post = false;
            const auto& posts = GetTradePostsAt(move.start_hex);
            for (const auto& post : posts) {
                if (post.owner == move.player && post.type == TradePostType::kPost) {
                    has_player_post = true;
                    break;
                }
            }
            if (!has_player_post) {
                LOG_WARN("ApplyTradingPostUpgrade ERROR: No trading post to upgrade at ", move.start_hex.ToString());
                return;
            }

            Player player_id = GetPlayerId(move.player);
            const GameRules& rules = GetGame()->GetRules();
            const int common_cost = rules.upgrade_cost_common;
            const int rare_cost = rules.upgrade_cost_rare;
            bool paid = false;

            /* Payment Decision Tree:
                Do I have surplus rare goods (count > 1)?
                ├─ YES → Use 1 rare good ✅ DONE
                └─ NO → Do I have enough common goods total (≥ 3)?
                    ├─ YES → Use common goods with variety preservation ✅ DONE  
                    └─ NO → Do I have any rare goods at all?
                        ├─ YES → Use 1 rare good ✅ DONE
                        └─ NO → ❌ Cannot upgrade (should only reach if there's a bug)
            */

            // A. Is there a surplus rare good to pay with?
            std::string rare_good_to_spend = "";
            if (player_id < rare_goods_.size()) {
                for (const auto& [name, count] : rare_goods_[player_id]) {
                    if (count > rare_cost) { // Cost is 1, so count > 1 means surplus.
                        rare_good_to_spend = name;
                        break;
                    }
                }
            }

            if (!rare_good_to_spend.empty()) {
                rare_goods_[player_id][rare_good_to_spend] -= rare_cost;
                paid = true;
                LOG_DEBUG("Paid for upgrade with surplus rare good: ", rare_good_to_spend);
            }

            // B. If not, are there enough common goods to pay with?
            if (!paid) {
                int total_common = 0;
                for (const auto& [name, count] : common_goods_[player_id]) {
                    total_common += count;
                }

                if (total_common >= common_cost) {
                    // Create a vector of goods with their counts for easier manipulation
                    std::vector<std::pair<std::string, int>> goods_list;
                    for (const auto& [name, count] : common_goods_[player_id]) {
                        if (count > 0) {
                            goods_list.push_back({name, count});
                        }
                    }
                    
                    // Sort by count (descending) so we prefer taking from goods with more surplus
                    std::sort(goods_list.begin(), goods_list.end(),
                            [](const auto& a, const auto& b) { return a.second > b.second; });

                    int to_remove = common_cost;
                    std::map<std::string, int> payment_plan; // Track how much to take from each good

                    LOG_DEBUG("Planning payment for upgrade, player: ", player_id, ", cost: ", common_cost);
                    for (const auto& [name, count] : goods_list) {
                        LOG_DEBUG("Available: ", name, " x", count);
                    }

                    // STEP 1: Take surplus goods (anything above 1) from each type
                    for (const auto& [name, count] : goods_list) {
                        if (to_remove == 0) break;
                        
                        int surplus = count - 1; // How much we can take while leaving at least 1
                        if (surplus > 0) {
                            int amount_to_take = std::min(to_remove, surplus);
                            payment_plan[name] += amount_to_take;
                            to_remove -= amount_to_take;
                            LOG_DEBUG("Step 1: Plan to take ", amount_to_take, " ", name, " (surplus)");
                        }
                    }

                    // STEP 2: If we still need more, take goods down to 0, but distribute evenly
                    if (to_remove > 0) {
                        // Count how many different goods we have left to take from
                        int goods_with_remainder = 0;
                        for (const auto& [name, count] : goods_list) {
                            int after_surplus = count - payment_plan[name]; // What's left after taking surplus
                            if (after_surplus > 0) {
                                goods_with_remainder++;
                            }
                        }
                        
                        // Try to distribute the remaining cost evenly across available goods
                        while (to_remove > 0 && goods_with_remainder > 0) {
                            bool took_any_this_round = false;
                            
                            for (const auto& [name, original_count] : goods_list) {
                                if (to_remove == 0) break;
                                
                                int already_taking = payment_plan[name];
                                int remaining_available = original_count - already_taking;
                                
                                if (remaining_available > 0) {
                                    payment_plan[name]++;
                                    to_remove--;
                                    took_any_this_round = true;
                                    LOG_DEBUG("Step 2: Plan to take 1 more ", name, " (total taking: ", payment_plan[name], ")");
                                    
                                    // If we just took the last of this good, decrease the count
                                    if (original_count - payment_plan[name] == 0) {
                                        goods_with_remainder--;
                                    }
                                }
                            }
                            
                            // Safety check to prevent infinite loop
                            if (!took_any_this_round) {
                                LOG_WARN("Payment planning failed - couldn't distribute remaining cost");
                                break;
                            }
                        }
                    }

                    // STEP 3: Execute the payment plan
                    if (to_remove == 0) {
                        for (const auto& [name, amount] : payment_plan) {
                            common_goods_[player_id][name] -= amount;
                            LOG_DEBUG("Paid ", amount, " ", name, " for upgrade (", 
                                    common_goods_[player_id][name], " remaining)");
                        }
                        paid = true;
                        LOG_DEBUG("Successfully paid for upgrade with common goods, preserving variety where possible");
                    } else {
                        LOG_WARN("ApplyTradingPostUpgrade ERROR: Payment planning failed, still need ", to_remove, " more goods");
                    }
                }
            }

            // C. Final fallback: If we couldn't pay with common goods, try any available rare goods
            if (!paid && player_id < rare_goods_.size()) {
                for (auto& [name, count] : rare_goods_[player_id]) {
                    if (count > 0) {
                        count -= rare_cost;
                        paid = true;
                        LOG_DEBUG("Paid for upgrade with non-surplus rare good: ", name, " (fallback option)");
                        break;
                    }
                }
            }
            
            if (!paid) {
                // This should only be reached if HasSufficientResourcesForUpgrade has a bug.
                // It's good to keep the warning for catching future desyncs.
                LOG_WARN("Player: ", current_player_id_, "ApplyTradingPostUpgrade ERROR: Not enough resources to pay with new logic.");
                return;
            }
            
            // If payment was successful, perform the state change.
            UpgradeTradingPost(move.start_hex, move.player);
            LOG_DEBUG("Trading post upgraded to center successfully.");
        }

        void Mali_BaState::ApplyIncomeCollection(const std::string& action_str) {
            int total_common =0;
            int total_rare = 0;
            Player player_id = current_player_id_;
            PlayerColor player_color = GetPlayerColor(player_id);
            SPIEL_CHECK_GE(player_id, 0);

            for (const auto& [hex, posts] : trade_posts_locations_) {
                for (const auto& post : posts) {
                    if (post.owner == player_color && post.type == TradePostType::kCenter) {
                        const City* city = GetGame()->GetCityAt(hex);
                        if (city != nullptr) {
                            rare_goods_[player_id][city->rare_good]++;
                            total_rare++;
                        }
                    }
                }
            }

            for (const auto& [hex, posts] : trade_posts_locations_) {
                for (const auto& post : posts) {
                    if (post.owner == player_color && post.type == TradePostType::kCenter) {
                        if (GetGame()->GetCityAt(hex) == nullptr) { 
                            auto connected_cities = GetConnectedCities(hex, player_color);
                            if (!connected_cities.empty()) {
                                const City* chosen_city = connected_cities[0];
                                rare_goods_[player_id][chosen_city->rare_good]++;
                                total_rare++;
                            } else {
                                auto closest_cities = FindClosestCities(hex);
                                if (!closest_cities.empty()) {
                                    common_goods_[player_id][closest_cities[0]->common_good] += 2;
                                    total_common += 2;
                                }
                            }
                        }
                    }
                }
            }

            for (const auto& [hex, posts] : trade_posts_locations_) {
                for (const auto& post : posts) {
                    if (post.owner == player_color && post.type == TradePostType::kPost) {
                        auto closest_cities = FindClosestCities(hex);
                        if (!closest_cities.empty()) {
                            common_goods_[player_id][closest_cities[0]->common_good]++;
                            total_common ++;
                        }
                    }
                }
            }
            LOG_DEBUG("Player ", player_id, " collected income: ", total_common, " common goods, ", total_rare, " rare goods");
        }

        void Mali_BaState::ApplyTradeRouteCreate(const Move& move) {
            if (move.type != ActionType::kTradeRouteCreate) {
                LOG_WARN("ApplyTradeRouteCreate called with wrong move type");
                return;
            }
            
            for (const auto& hex : move.path) {
                bool has_center = false;
                const auto& posts = GetTradePostsAt(hex);
                for (const auto& post : posts) {
                    if (post.owner == move.player && post.type == TradePostType::kCenter) {
                        has_center = true;
                        break;
                    }
                }
                if (!has_center) {
                    LOG_WARN("ApplyTradeRouteCreate: Player ", static_cast<int>(move.player), 
                            " doesn't have a center at ", hex.ToString());
                    return;
                }
            }
            
            bool success = CreateTradeRoute(move.path, move.player);
            if (success) {
                LOG_DEBUG("Trade route created successfully with ", move.path.size(), " hexes");
            } else {
                LOG_WARN("Failed to create trade route");
            }
        }        

        void Mali_BaState::ApplyTradeRouteDelete(const Move& move) {
            if (move.type != ActionType::kTradeRouteDelete) {
                SpielFatalError("ApplyTradeRouteDelete called with incorrect move type.");
            }
            bool success = DeleteTradeRoute(move.route_id);
            if (!success) {
                LOG_WARN("ApplyTradeRouteDelete: Failed to delete route ", move.route_id);
            }
        }
        
        void Mali_BaState::DoApplyAction(Action action) {
            PushStateToUndoStack();
            is_terminal_ = false;

            Player player_who_moved = current_player_id_;
            Phase old_phase = current_phase_;

            if (IsChanceNode()) {
                SPIEL_CHECK_EQ(action, kChanceSetupAction);
                ApplyChanceSetup();
                SetCurrentPhase(Phase::kPlaceToken);
                current_player_id_ = 0;
                current_player_color_ = GetPlayerColor(current_player_id_);
            } 
            else if (current_phase_ == Phase::kPlaceToken) {
                // This block is ONLY for placing tokens.
                SPIEL_CHECK_GE(action, kPlaceTokenActionBase);
                SPIEL_CHECK_LT(action, kUpgradeActionBase); // Sanity check the action range
                
                int hex_index = action - kPlaceTokenActionBase;
                
                SPIEL_CHECK_LT(hex_index, GetGame()->NumHexes());
                
                HexCoord hex = GetGame()->IndexToCoord(hex_index);

                Move place_token_move;
                place_token_move.type = ActionType::kPlaceToken;
                place_token_move.player = current_player_color_;
                place_token_move.start_hex = hex;
                
                ApplyPlaceTokenMove(place_token_move);
                moves_history_.push_back(place_token_move);
            } 
            else if (current_phase_ == Phase::kPlay) {
                // This block is ONLY for playing the game.
                // It should not be possible for a PlaceToken action to get here.
                SPIEL_CHECK_FALSE(action >= kPlaceTokenActionBase && action < kUpgradeActionBase);

                Move move = ActionToMove(action);
                
            switch (move.type) {
                case ActionType::kPass:
                    // No state change needed.
                    break;
                
                case ActionType::kMancala:
                    // First, apply the core mancala move.
                    ApplyMancalaMove(move);
                    // THEN, check for the compound part.
                    if (move.place_trading_post) {
                        ApplyPlacePostFromMancala(move); // New, simplified helper

                        // If, after placing the post, we also declare a route.
                        if (move.declares_trade_route) {
                            // If city_free_upgrade is true, a post in a city that's part of
                            // the route needs to become a center. That logic is in CreateTradeRoute()

                            LOG_DEBUG("Applying compound mancala action: creating trade route.");
                            CreateTradeRoute(move.trade_route_path, move.player);
                        }
                    }
                    break;

                case ActionType::kPlaceTCenter: // This is an upgrade
                    // First, apply the core upgrade.
                    ApplyTradingPostUpgrade(move);
                    // THEN, check for the compound part.
                    if (move.declares_trade_route) {
                        LOG_DEBUG("Applying compound action: creating trade route.");
                        CreateTradeRoute(move.trade_route_path, move.player);
                    }
                    break;

                case ActionType::kIncome:
                    ApplyIncomeCollection(move.action_string);
                    break;

                case ActionType::kTradeRouteCreate:
                    ApplyTradeRouteCreate(move);
                    break;
                    
                default:
                    SpielFatalError(absl::StrCat("DoApplyAction: Unhandled or invalid move type ",
                                                static_cast<int>(move.type), " for action ", action));
                    break;
            }
                moves_history_.push_back(move);
                ValidateTradeRoutes();
            }
            
            // Player switching logic
            if (current_phase_ == old_phase) {
                current_player_color_ = GetNextPlayerColor(current_player_color_);
                current_player_id_ = GetPlayerId(current_player_color_);
            }

            // After the action is applied, get the immediate rewards for that action.
            std::vector<double> rewards = Rewards();
            for (int i = 0; i < game_->NumPlayers(); ++i) {
                cumulative_returns_[i] += rewards[i];
            }

            ClearCaches();
            RefreshTerminalStatus();
        }

        std::string Mali_BaState::ActionToString(Player player, Action action) const {
            if (IsChanceNode()) {
                if (action == kChanceSetupAction) return "ChanceSetup";
                return absl::StrCat("Unknown chance action: ", action);
            }

            // Handle PlaceToken phase directly, as its action encoding is simple.
            if (current_phase_ == Phase::kPlaceToken) {
                if (action >= kPlaceTokenActionBase && action < kUpgradeActionBase) {
                    int hex_index = action - kPlaceTokenActionBase;
                    SPIEL_CHECK_LT(hex_index, GetGame()->NumHexes());
                    HexCoord hex = GetGame()->IndexToCoord(hex_index);
                    return absl::StrCat("place_token ", hex.ToString());
                }
            }

            Move move = ActionToMove(action);

            switch (move.type) {
                case ActionType::kPass:
                    return "pass";
                case ActionType::kIncome:
                    return "income";
                case ActionType::kPlaceTCenter: {
                    return move.action_string;
                }
                case ActionType::kMancala: {
                    std::string action_str = absl::StrCat("mancala ", move.start_hex.ToString(), "->", move.path.front().ToString());
                    if (move.place_trading_post) {
                        absl::StrAppend(&action_str, " post");
                    }
                    return action_str;
                }
                case ActionType::kTradeRouteCreate: {
                    std::string path_str;
                    for (size_t i = 0; i < move.path.size(); ++i) {
                        if (i > 0) path_str += ":";
                        path_str += move.path[i].ToString();
                    }
                    return absl::StrCat("route_create ", path_str);
                }
                default:
                    return absl::StrCat("Unknown(action_id=", action, ",type=", static_cast<int>(move.type),")");
            }
        }
        
        void Mali_BaState::PushStateToUndoStack() {
            undo_stack_.emplace_back(StateSnapshot{
                current_player_id_,
                current_player_color_,
                current_phase_,
                player_token_locations_,
                hex_meeples_,
                trade_posts_locations_,
                common_goods_,
                rare_goods_,
                trade_routes_,
                next_route_id_,
                moves_history_,
                cumulative_returns_,
                is_terminal_
            });
        }

        void Mali_BaState::UndoAction(Player player, Action action) {
            SPIEL_CHECK_FALSE(undo_stack_.empty());
            const StateSnapshot& last_state = undo_stack_.back();

            current_player_id_ = last_state.current_player_id_;
            current_player_color_ = last_state.current_player_color_;
            current_phase_ = last_state.current_phase_;
            player_token_locations_ = last_state.player_token_locations_;
            hex_meeples_ = last_state.hex_meeples_;
            trade_posts_locations_ = last_state.trade_posts_locations_;
            common_goods_ = last_state.common_goods_;
            rare_goods_ = last_state.rare_goods_;
            trade_routes_ = last_state.trade_routes_;
            next_route_id_ = last_state.next_route_id_;
            moves_history_ = last_state.moves_history_;
            cumulative_returns_ = last_state.cumulative_returns_;

            undo_stack_.pop_back();

            if (!history_.empty()) {
                history_.pop_back();
            }
            if (move_number_ > 0) {
                move_number_--;
            }

            ClearCaches();
            is_terminal_ = false;
        }

        std::string Mali_BaState::InformationStateString(Player player) const { return ObservationString(player); }
        std::string Mali_BaState::ObservationString(Player player) const { return ToString(); }

        bool Mali_BaState::IsTerminal() const {
            if (is_terminal_) return true;

            // Check for game length limit BEFORE checking for win conditions.
            // This makes it a hard limit.
            if (history_.size() >= MaxGameLength()) {
                const_cast<Mali_BaState*>(this)->is_terminal_ = true;
                // Optionally set the reason for debugging
                const_cast<Mali_BaState*>(this)->game_end_reason_ = "Max game length reached";
                const_cast<Mali_BaState*>(this)->winning_player_ = -1; // -1 for draw
                return true;
            }

            // This checks for player-driven win conditions
            if (MaybeFinalReturns().has_value() || history_.size() >= MaxGameLength()) {
                const_cast<Mali_BaState*>(this)->is_terminal_ = true;
                return true;
            }
            return false;
        }

        absl::optional<std::vector<double>> Mali_BaState::MaybeFinalReturns() const {
            const GameRules& rules = GetGame()->GetRules();

            for (Player p = 0; p < game_->NumPlayers(); ++p) {
                PlayerColor player_color = GetPlayerColor(p);
                int player_route_count = 0;
                for (const auto& route : trade_routes_) {
                    if (route.owner == player_color && route.active) {
                        player_route_count++;
                    }
                }

                // =============================================================
                // First check end-game *requirements* - they don't trigger the end-game
                // but the game can't end withouth them. If these fail, don't bother
                // checking for potential victory *conditions*...they are irrelevant
                // =============================================================

                // === 1. Whether the player has at least 2 trade routes and that rule is active
                // If the rule is active...
                if (rules.end_game_req_num_routes > 0) {
                    if (player_route_count < rules.end_game_req_num_routes) {
                        // LOG_DEBUG("Player ", p, " needs ", rules.end_game_req_num_routes - player_route_count, " more routes for other victory conditions");
                        continue;
                    }
                    // LOG_DEBUG("Player ", p, " has ", player_route_count, " active trade routes");
                }

                // =============================================================
                // Now check end-game *conditions* - all requirements are met but
                // have they reached one of these conditions that will end the game?
                // =============================================================

                // === 1. Having X or more unique rare goods in inventory. If it's set to -1 in the rules,
                // === this end-game condition does not apply
                if (rules.end_game_cond_num_rare_goods > 0) {
                    int unique_rare_count = 0;
                    for (const auto& [good_name, good_count] : rare_goods_[p]) {
                        if (good_count > 0) {
                            unique_rare_count++;
                        }
                    }
                    //LOG_DEBUG("Player ", p, " has ", unique_rare_count, " unique rare goods: [", rare_goods_debug, "]");

                    if (unique_rare_count >= rules.end_game_cond_num_rare_goods) {
                        game_end_triggered_by_player_ = p;
                        game_end_reason_ = "Rare goods victory condition";
                        LOG_DEBUG("🎉 GAME END TRIGGER: Player ", p, " has reached the rare good victory condition (", 
                                unique_rare_count, "/", rules.end_game_cond_num_rare_goods, ")! 🎉");
                        LOG_DEBUG("Total moves in history: ", history_.size());
                        return std::vector<double>();
                    }
                }

                // === 2. Having declared X or more trade routes. If it's set to -1 in the rules,
                // === this end-game condition does not apply
                // Check trade route victory condition
                if (rules.end_game_cond_num_routes > 0 && player_route_count >= rules.end_game_cond_num_routes) {
                    game_end_triggered_by_player_ = p;
                    game_end_reason_ = "Trade route victory condition";
                    LOG_DEBUG("🎉 GAME END TRIGGER: Player ", p, " has reached the trade route victory condition (", 
                            player_route_count, "/", rules.end_game_cond_num_routes, ")! 🎉");
                    LOG_DEBUG("Total moves in history: ", history_.size());

                    return std::vector<double>();
                }

                // === 3. Having declared a trade route through Timbuktu that goes to the coast. 
                // === If it's set to false in the rules, this end-game condition does not apply
                // Check trade route Desert --> Timbuktu --> Coast condition
                const std::set<HexCoord>& coastal_hexes = GetGame()->GetCoastalHexes();
                if (rules.end_game_cond_timbuktu_to_coast && !coastal_hexes.empty()) {
                    for (const auto& route : trade_routes_) {
                        if (route.owner == GetPlayerColor(p) && route.active) {
                            bool timbuktu_found = false;
                            bool coast_found = false;
                            bool desert_found = false;
                            int other_cities_count = 0;

                            for (const auto& hex : route.hexes) {
                                const City* city = GetGame()->GetCityAt(hex);
                                if (city) {
                                    if (city->id == GetCityID("Timbuktu")) {
                                        timbuktu_found = true;
                                    // Oudane and Agadez (1 and 9) are the two desert cities
                                    } else if (city->id == GetCityID("Agadez") || city->id == GetCityID("Oudane")) {
                                        desert_found = true;
                                    } else {
                                        other_cities_count++;
                                    }
                                }

                                if (coastal_hexes.count(hex)) {
                                    coast_found = true;
                                }
                            }
                            // Now check the full condition Desert + Timbuktu + two other cities + the coast
                            if (timbuktu_found && coast_found && desert_found && other_cities_count >= 2) {
                                LOG_DEBUG("GAME END TRIGGER: Player ", p, " connected Timbuktu to the coast via a trade route with at least 3 other cities.");
                                LOG_DEBUG("Total moves in history: ", history_.size());
                                // Track who triggered the end and why
                                game_end_triggered_by_player_ = p;
                                game_end_reason_ = "Trade route Timbuktu to coast";

                                return std::vector<double>();
                            }
                        }
                    } 
                }

                // === 4: Having a rare good from each region (or N regions). If it's set to false
                // === in the rules,this end-game condition does not apply
                if (rules.end_game_cond_rare_good_each_region) {
                    // Rare goods from what number of regions is set in the INI file or defaults to 5 
                    int num_regions_needed = rules.end_game_cond_rare_good_num_regions;
                    // Track which regions we have rare goods from
                    std::set<int> regions_with_rare_goods;
                    
                    // Go through each rare good this player has
                    for (const auto& [good_name, good_count] : rare_goods_[p]) {
                        if (good_count > 0) {
                            // Find which city produces this rare good
                            for (const auto& city : GetGame()->GetCities()) {
                                if (city.rare_good == good_name) {
                                    // Get the region this city is in
                                    int region_id = GetGame()->GetRegionForHex(city.location);
                                    if (region_id != -1) {
                                        regions_with_rare_goods.insert(region_id);
                                    }
                                    break; // Found the city, no need to continue searching
                                }
                            }
                        }
                    }                    
                    // Count total number of regions in the game
                    int total_regions = 0;
                    for (int i = 1; i <= 6; ++i) { // Based on your ini file showing regions 1-6
                        if (!GetGame()->GetRegionName(i).empty() && 
                            GetGame()->GetRegionName(i).find("Unknown") == std::string::npos) {
                            total_regions++;
                        }
                    }                    
                    // LOG_DEBUG("DEBUG: Player ", p, " has rare goods from ", 
                    //     regions_with_rare_goods.size(), " regions! 🎉");
                    // LOG_DEBUG("DEBUG: Regions covered: ", regions_with_rare_goods.size(), "/", total_regions);
                    // Check if player has rare goods from all regions
                    if ((int)regions_with_rare_goods.size() >= std::min(total_regions, num_regions_needed)) {
                        LOG_DEBUG("🎉 GAME END TRIGGER: Player ", p, " has rare goods from ", 
                            regions_with_rare_goods.size(), " regions! 🎉");
                        LOG_DEBUG("Regions covered: ", regions_with_rare_goods.size(), "/", total_regions);
                        
                        // Debug info - show which rare goods from which regions
                        std::string debug_info = "Rare goods by region: ";
                        for (int region_id : regions_with_rare_goods) {
                            debug_info += GetGame()->GetRegionName(region_id) + ", ";
                        }
                        LOG_DEBUG(debug_info);
                        
                        game_end_triggered_by_player_ = p;
                        game_end_reason_ = "Rare good from N regions victory condition";
                        
                        return std::vector<double>();
                    }
                }                
            }

            // === 999. Player-independent end-game condition. 
            // Check to see if we've exceeded the maximum game length
            if (history_.size() >= MaxGameLength()) {
                LOG_DEBUG("GAME END TRIGGER: Maximum game length of ", MaxGameLength(), " moves has been reached.");
                return std::vector<double>();
            }

            return absl::nullopt;
        }


        std::vector<double> Mali_BaState::Returns() const {
            if (!IsTerminal()) {
                // For non-terminal states, it's just the sum of rewards so far.
                return cumulative_returns_;
            }

            const auto& training_params = GetGame()->GetTrainingParameters();

            // First, check if the game ended because of the max length rule.
            if (history_.size() >= MaxGameLength()) {
                LOG_DEBUG("======== GAME END: MAX LENGTH REACHED ========");
                LOG_DEBUG("Game ended due to reaching max length. Declaring a draw.");
                // Return max game penalty for all players, for stalling to a draw.
                return std::vector<double>(NumPlayers(), training_params.max_moves_penalty);
            }

            LOG_DEBUG("======== FINAL SCORE CALCULATION ========");
            LOG_DEBUG("Moves in history: ", history_.size());
            std::vector<double> scores(NumPlayers(), 0.0);
            const GameRules& rules = GetGame()->GetRules();

            std::vector<double> route_scores(NumPlayers(), 0.0);
            std::vector<double> rare_good_scores(NumPlayers(), 0.0);
            std::vector<double> center_scores(NumPlayers(), 0.0);
            std::vector<double> common_good_set_scores(NumPlayers(), 0.0);
            std::vector<double> longest_route_scores(NumPlayers(), 0.0);
            std::vector<double> region_control_scores(NumPlayers(), 0.0);
            std::vector<double> regions_crossed_scores(NumPlayers(), 0.0);

            // Calculate the scores received for longest route(s)
            std::vector<std::pair<int, Player>> player_route_lengths;
            for (Player p = 0; p < NumPlayers(); ++p) {
                int longest_single_route = 0;
                for (const auto& route : trade_routes_) {
                    if (route.owner == GetPlayerColor(p) && route.active) {
                        longest_single_route = std::max(longest_single_route, (int)route.hexes.size());
                    }
                }
                player_route_lengths.push_back({longest_single_route, p});
            }
            std::sort(player_route_lengths.rbegin(), player_route_lengths.rend());
            for(int i = 0; i < player_route_lengths.size() && i < rules.score_longest_routes.size(); ++i) {
                if (player_route_lengths[i].first > 0) {
                    longest_route_scores[player_route_lengths[i].second] += rules.score_longest_routes[i];
                    int j = i + 1;
                    while(j < player_route_lengths.size() && player_route_lengths[j].first == player_route_lengths[i].first) {
                        longest_route_scores[player_route_lengths[j].second] += rules.score_longest_routes[i];
                        j++;
                    }
                    i = j - 1;
                }
            }

            // Give out scores for region control (most Trad. Centers in the region)
            std::vector<int> valid_region_ids = GetGame()->GetValidRegionIds();
            for (int region_id : valid_region_ids) {
                std::vector<std::pair<int, Player>> region_control;
                for (Player p = 0; p < NumPlayers(); ++p) {
                    int centers_in_region = 0;
                    for (const auto& [hex, posts] : trade_posts_locations_) {
                        if (GetGame()->GetRegionForHex(hex) == region_id) {
                            for (const auto& post : posts) {
                                if (post.owner == GetPlayerColor(p) && post.type == TradePostType::kCenter) {
                                    centers_in_region++;
                                }
                            }
                        }
                    }
                    region_control.push_back({centers_in_region, p});
                }
                std::sort(region_control.rbegin(), region_control.rend());
                for(int i = 0; i < region_control.size() && i < rules.score_region_control.size(); ++i) {
                    if (region_control[i].first > 0) {
                        region_control_scores[region_control[i].second] += rules.score_region_control[i];
                        int j = i + 1;
                        while(j < region_control.size() && region_control[j].first == region_control[i].first) {
                            region_control_scores[region_control[j].second] += rules.score_region_control[i];
                            j++;
                        }
                        i = j - 1;
                    }
                }
            }

            // Give out score for having the most trading routes
            for (Player p = 0; p < NumPlayers(); ++p) {
                PlayerColor p_color = GetPlayerColor(p);

                int active_routes = 0;
                for (const auto& route : trade_routes_) {
                    if (route.owner == p_color && route.active) {
                        active_routes++;
                        route_scores[p] += route.hexes.size();
                    }
                }
                if (active_routes >= 3) route_scores[p] += 5;

                for (const auto& [name, count] : GetPlayerRareGoods(p)) {
                    rare_good_scores[p] += count;
                }

                for (const auto& [hex, posts] : trade_posts_locations_) {
                    for (const auto& post : posts) {
                        if (post.owner == p_color && post.type == TradePostType::kCenter) {
                            center_scores[p] += 2.0;
                        }
                    }
                }

                // Give out score for sets of unique common goods
                int unique_common_count = 0;
                for (const auto& [name, count] : GetPlayerCommonGoods(p)) {
                    if (count > 0) unique_common_count++;
                }
                if (unique_common_count > 0) {
                    if (unique_common_count >= 12) {
                        common_good_set_scores[p] += rules.score_unique_common_goods_bonus;
                        common_good_set_scores[p] += rules.score_unique_common_goods.rbegin()->second;
                    } else {
                        auto it = rules.score_unique_common_goods.find(unique_common_count);
                        if (it != rules.score_unique_common_goods.end()) {
                            common_good_set_scores[p] += it->second;
                        }
                    }
                }

                // Give out score for trading routes crossing regions
                for (const auto& route : trade_routes_) {
                    if (route.owner == p_color && route.active) {
                        std::set<int> regions_crossed;
                        for (const auto& hex : route.hexes) {
                            int region_id = GetGame()->GetRegionForHex(hex);
                            if (region_id != -1) regions_crossed.insert(region_id);
                        }
                        int num_regions = regions_crossed.size();
                        if (num_regions > 0) {
                            auto it = rules.score_regions_crossed.find(num_regions);
                            if (it != rules.score_regions_crossed.end()) {
                                regions_crossed_scores[p] += it->second;
                            }
                        }
                    }
                }
            }

            for (Player p = 0; p < NumPlayers(); ++p) {
                scores[p] = route_scores[p] +
                            rare_good_scores[p] +
                            center_scores[p] +
                            common_good_set_scores[p] +
                            longest_route_scores[p] +
                            region_control_scores[p] +
                            regions_crossed_scores[p];
                
                LOG_DEBUG("--- Player ", p, " (", PlayerColorToString(GetPlayerColor(p)), ") Score: ", scores[p], " ---");
                LOG_DEBUG("  - Route Hexes & Bonus: ", route_scores[p]);
                LOG_DEBUG("  - Rare Goods Total:    ", rare_good_scores[p]);
                LOG_DEBUG("  - Trading Centers:     ", center_scores[p]);
                LOG_DEBUG("  - Unique Common Sets:  ", common_good_set_scores[p]);
                LOG_DEBUG("  - Longest Route Bonus: ", longest_route_scores[p]);
                LOG_DEBUG("  - Region Control Bonus:", region_control_scores[p]);
                LOG_DEBUG("  - Regions Crossed Bonus: ", regions_crossed_scores[p]);
            }           
            LOG_DEBUG("========================================");
            
            double max_score = -1.0 * std::numeric_limits<double>::infinity();
            for (double score : scores) {
                if (score > max_score) {
                    max_score = score;
                }
            }

            if (max_score <= 0 && scores.size() > 0) {
                return std::vector<double>(NumPlayers(), 0.0);
            }

            std::vector<Player> winners;
            for (Player p = 0; p < NumPlayers(); ++p) {
                if (scores[p] >= max_score - 1e-6) { 
                    winners.push_back(p);
                }
            }

            std::vector<double> returns(NumPlayers(), 0.0);
            if (winners.size() == 1) {
                returns[winners[0]] = 1.0;
                // Add loss penalty to non-winners
                for (Player p = 0; p < NumPlayers(); ++p) {
                    LOG_DEBUG("winners[0]= ",winners[0],"; p= ",p,";");
                    if (p != winners[0] && training_params.loss_penalty != -1.0) {
                        returns[p] += training_params.loss_penalty;
                        LOG_DEBUG("returns[",p,"] = ",returns[p],";");
                    }
                }
            } else if (winners.size() > 1 && winners.size() < NumPlayers()) {
                for (Player p : winners) {
                    returns[p] = training_params.draw_penalty; // Tie for the win is a draw
                }
                // Add loss penalty to non-winners
                for (Player p = 0; p < NumPlayers(); ++p) {
                    if (std::find(winners.begin(), winners.end(), p) == winners.end() && training_params.loss_penalty != -1.0) {
                        returns[p] += training_params.loss_penalty;
                    }
                }
            } else { 
                for (Player p = 0; p < NumPlayers(); ++p) {
                    returns[p] = training_params.draw_penalty;
                }
            }

            LOG_DEBUG("Returns (win/loss/draw): ", absl::StrJoin(returns, ", "));
            return returns;
        }

        // =======================================================================
        // This function hands out intermediate rewards that are in addition to the 
        // game end rewards
        // =======================================================================
        std::vector<double> Mali_BaState::Rewards() const {
            int max_moves;
            std::vector<double> rewards(game_->NumPlayers(), 0.0);

            if (IsTerminal()) {
                return std::vector<double>(game_->NumPlayers(), 0.0);
            }

            // Get training parameters from game
            const auto& training_params = GetGame()->GetTrainingParameters();
            max_moves = game_->MaxGameLength();

            // --- Reward shaping strategy: Time Penalty ---
            // Add a small penalty to the current player for taking a turn.
            // This encourages finishing the game faster.
            // Increase the severity as time goes on.
            if (current_player_id_ >= 0 && current_player_id_ < rewards.size()) {
                int moves = history_.size();
                if (moves > 0 && moves <= std::trunc(1/3*max_moves)) {
                    // Start with the base penalty
                    rewards[current_player_id_] += training_params.time_penalty;
                }
                else if (moves > std::trunc(1/3*max_moves) && moves <= std::trunc(1.7/3*max_moves)) {
                    // Increase the penalty
                    rewards[current_player_id_] += (2*training_params.time_penalty);
                }
                else if (moves > std::trunc(1.7*max_moves) && moves <= std::trunc(2.4/3*max_moves)) {
                    // Increase the penalty
                    rewards[current_player_id_] += (4*training_params.time_penalty);
                }
                else if (moves > std::trunc(2.4/3*max_moves)) {
                    // Increase the penalty
                    rewards[current_player_id_] += (7*training_params.time_penalty);
                }
                // if (moves > training_params.quick_win_threshold) {
                //     // Do linear penalty increases after the threshold move
                //     rewards[current_player_id_] += training_params.time_penalty;
                // }
                // else {
                //     // Calculate the penalty using an exponential that crosses over
                //     // the linear amount at the quick_win_threshold 
                //     // (i.e. the penalty amounts are same when moves == threshold move)
                //     double pen_pos = -1.0 * training_params.time_penalty;
                //     double factor = std::log(training_params.quick_win_threshold) / training_params.quick_win_threshold;
                //     double penalty = -1.0 / training_params.quick_win_threshold * (pen_pos * std::exp(factor * moves));
                //     rewards[current_player_id_] += penalty;
                //     LOG_DEBUG("Player ", current_player_id_, ", Move ", moves, ", Penalty = ", penalty);
                // }
            }

            if (moves_history_.empty()) {
                return rewards;
            }

            // Define reward values for easy tuning
            // constexpr double kUpgradeReward = 0.02;
            // constexpr double kTradeRouteReward = 0.04;
            // constexpr double kNewRareRegionReward = 0.08;
            // constexpr double kNewCommonGoodReward = 0.02;
            // constexpr double kKeyLocationPostReward = 0.03;

            const Move& last_move = moves_history_.back();
            Player player_who_moved = GetPlayerId(last_move.player);

            if (player_who_moved == kInvalidPlayer) {
                return rewards;
            }
            
            switch (last_move.type) {
                case ActionType::kPlaceTCenter: {
                    rewards[player_who_moved] += training_params.upgrade_reward;
                    break;
                }
                case ActionType::kTradeRouteCreate: {
                    rewards[player_who_moved] += training_params.trade_route_reward;
                    break;
                }
                case ActionType::kMancala: {
                    if (last_move.place_trading_post) {
                        const HexCoord& dest_hex = last_move.path.back();
                        bool is_key_location = false;

                        if (GetGame()->GetCoastalHexes().count(dest_hex) > 0) {
                            is_key_location = true;
                        }

                        const City* city = GetGame()->GetCityAt(dest_hex);
                        if (city != nullptr) {
                            // Use our helper to find the key city IDs dynamically
                            static const int timbuktu_id = GetCityID("Timbuktu");
                            static const int agadez_id = GetCityID("Agadez");
                            static const int oudane_id = GetCityID("Oudane");

                            if (city->id == timbuktu_id || city->id == agadez_id || city->id == oudane_id) {
                                is_key_location = true;
                            }
                        }

                        if (is_key_location) {
                            rewards[player_who_moved] += training_params.key_location_post_reward;
                        }
                    }
                    break;
                }
                case ActionType::kIncome: {
                    SPIEL_CHECK_FALSE(undo_stack_.empty());
                    const StateSnapshot& prev_state = undo_stack_.back();

                    // Reward for acquiring a new unique common good
                    const auto& common_before = prev_state.common_goods_[player_who_moved];
                    const auto& common_after = common_goods_[player_who_moved];
                    for (const auto& [good, count_after] : common_after) {
                        int count_before = common_before.count(good) ? common_before.at(good) : 0;
                        if (count_before == 0 && count_after > 0) {
                            rewards[player_who_moved] += training_params.new_common_good_reward;
                        }
                    }

                    // Reward for acquiring a rare good from a NEW region
                    const auto& rare_before = prev_state.rare_goods_[player_who_moved];
                    const auto& rare_after = rare_goods_[player_who_moved];

                    auto get_regions_for_goods = [&](const std::map<std::string, int>& goods_map) {
                        std::set<int> regions;
                        for (const auto& [good, count] : goods_map) {
                            if (count > 0) {
                                for (const auto& city : GetGame()->GetCities()) {
                                    if (city.rare_good == good) {
                                        regions.insert(GetGame()->GetRegionForHex(city.location));
                                        break;
                                    }
                                }
                            }
                        }
                        return regions;
                    };

                    std::set<int> regions_before = get_regions_for_goods(rare_before);
                    std::set<int> processed_new_regions;

                    for (const auto& [good, count_after] : rare_after) {
                        int count_before = rare_before.count(good) ? rare_before.at(good) : 0;
                        if (count_after > count_before) {
                            int new_region = -1;
                            for (const auto& city : GetGame()->GetCities()) {
                                if (city.rare_good == good) {
                                    new_region = GetGame()->GetRegionForHex(city.location);
                                    break;
                                }
                            }
                            
                            if (new_region != -1 && regions_before.find(new_region) == regions_before.end() && processed_new_regions.find(new_region) == processed_new_regions.end()) {
                                rewards[player_who_moved] += training_params.new_rare_region_reward;
                                processed_new_regions.insert(new_region);
                            }
                        }
                    }
                    break;
                }
                default:
                    break;
            }

            return rewards;
        }


        void Mali_BaState::ObservationTensor(Player player, absl::Span<float> values) const {
            // 1. Get the game object.
            std::shared_ptr<const Game> game = State::GetGame();
            SPIEL_CHECK_TRUE(game != nullptr);
            
            // 2. Create the observer.
            std::shared_ptr<Observer> observer = game->MakeObserver(absl::nullopt, {});
            SPIEL_CHECK_TRUE(observer != nullptr);

            // 3. Use our local allocator to get the tensor from the observer.
            VectorTensorAllocator allocator;
            observer->WriteTensor(*this, player, &allocator);

            // 4. Copy the data from the allocator's buffer into the provided span.
            const std::vector<float>& tensor_data = allocator.GetBuffer();
            SPIEL_CHECK_EQ(tensor_data.size(), values.size());
            std::copy(tensor_data.begin(), tensor_data.end(), values.begin());
        }

        void Mali_BaState::ClearCaches() {
            cached_legal_actions_result_ = absl::nullopt;
            // cached_legal_actions_ = absl::nullopt;
            // cached_legal_move_structs_ = absl::nullopt;
        }

        void Mali_BaState::RemoveMeepleAt(const HexCoord& hex, int index) {
            auto it = hex_meeples_.find(hex);
            if (it == hex_meeples_.end()) {
                LOG_WARN("Attempted to remove meeple from hex ", hex.ToString(), " which has no meeples.");
                return;
            }
            auto& meeples = it->second;
            if (index >= 0 && index < meeples.size()) {
                meeples.erase(meeples.begin() + index);
            } else {
                LOG_WARN("Attempted to remove meeple at invalid index ", index, " from hex ", hex.ToString());
            }
        }

        std::string Mali_BaState::PlayRandomMoveAndSerialize() {
            if (IsTerminal()) {
                LOG_WARN("PlayRandomMoveAndSerialize: Called on a terminal state.");
                return Serialize();
            }

            Action chosen_action = kInvalidAction;

            if (IsChanceNode()) {
                auto outcomes = ChanceOutcomes();
                SPIEL_CHECK_EQ(outcomes.size(), 1);
                chosen_action = outcomes[0].first;
            } else {
                // *** THIS IS THE CORE LOGIC CHANGE ***
                Player current_player = CurrentPlayer();
                const auto& player_types = GetGame()->GetPlayerTypes();
                SPIEL_CHECK_LT(current_player, player_types.size());
                PlayerType type = player_types[current_player];

                switch (type) {
                    case PlayerType::kHuman:
                        // This mode should not have human players.
                        SpielFatalError("Human player type is not supported in cpp_sync_gui mode or no-GUI simulations.");
                        break;
                    
                    case PlayerType::kAI:
                        // For now, the AI will use the heuristic. This can be replaced
                        // with a call to a neural network model later.
                        chosen_action = SelectHeuristicRandomAction();
                        break;

                    case PlayerType::kHeuristic:
                        chosen_action = SelectHeuristicRandomAction();
                        break;
                }

                if (chosen_action == kInvalidAction) {
                    LOG_WARN("Action selection returned kInvalidAction, falling back to uniform random.");
                    std::vector<Action> legal_actions = LegalActions();
                    if (legal_actions.empty()) {
                        SpielFatalError("PlayRandomMoveAndSerialize: Fallback failed, no legal actions.");
                    }
                    std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
                    chosen_action = legal_actions[dist(rng_)];
                }
            }

            ApplyAction(chosen_action);
            return Serialize();
        }

    } // namespace mali_ba
} // namespace open_spiel