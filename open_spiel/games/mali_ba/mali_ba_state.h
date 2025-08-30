// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not be-abused except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_MALI_BA_STATE_H_
#define OPEN_SPIEL_GAMES_MALI_BA_STATE_H_

#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"

#include "open_spiel/games/mali_ba/mali_ba_common.h"
//#include "open_spiel/games/mali_ba/mali_ba_game.h"

namespace open_spiel
{
namespace mali_ba
{
    class Mali_BaGame;
    struct MaliBaTest;

    std::string HexCoordToJsonString(const HexCoord& hex);
    absl::optional<HexCoord> JsonStringToHexCoord(const std::string& s);

    struct TurnEvaluation {
        std::vector<Action> actions;
        double estimated_value;
        bool changes_game_outcome;
        int trade_route_count;
        int income_potential;
    };

    struct StateSnapshot {
        Player current_player_id_;
        PlayerColor current_player_color_;
        Phase current_phase_;
        std::map<HexCoord, std::vector<PlayerColor>> player_token_locations_;
        std::map<HexCoord, std::vector<MeepleColor>> hex_meeples_;
        std::map<HexCoord, std::vector<TradePost>> trade_posts_locations_;
        std::vector<std::map<std::string, int>> common_goods_;
        std::vector<std::map<std::string, int>> rare_goods_;
        std::vector<TradeRoute> trade_routes_; // Add for undo
        int next_route_id_;                   // Add for undo
        std::vector<Move> moves_history_;
        std::vector<double> cumulative_returns_;
        bool is_terminal_; 
    };

    class Mali_BaState : public State {
    public:
        friend class Mali_BaGame;
        friend struct MaliBaTest;

        explicit Mali_BaState(std::shared_ptr<const Game> game);
        Mali_BaState(const Mali_BaState& other);
        Mali_BaState(Mali_BaState&& other) = default;
        Mali_BaState& operator=(Mali_BaState&& other) = default;

        Action SelectHeuristicRandomAction() const;
        std::map<Action, double> GetHeuristicActionWeights() const;
        Player CurrentPlayer() const override;
        std::vector<Action> LegalActions() const override;
        LegalActionsResult GetLegalActionsAndCounts() const;
        std::string ActionToString(Player player, Action action) const override;
        std::string ToString() const override;
        bool IsTerminal() const override;
        std::vector<double> Returns() const override;
        std::vector<double> Rewards() const override;
        std::string InformationStateString(Player player) const override;
        std::string ObservationString(Player player) const override;
        void ObservationTensor(Player player, absl::Span<float> values) const override;
        void UndoAction(Player player, Action action) override;
        std::string Serialize() const override;
        bool IsChanceNode() const override;
        std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
        std::unique_ptr<State> Clone() const override;
                
        // other methods like Undo, PlayRandom, AI selection
        void UndoLastAction();
        void UndoToTurnStart();
        std::string PlayRandomMoveAndSerialize();
        std::vector<Action> SelectRandomTurnActions();
        std::string PlayRandomTurnAndSerialize();
        Action SelectTrainingAwareRandomAction();
        std::vector<Action> SelectEvaluatedTurnActions();
        TurnEvaluation GenerateTurnStrategy(int num_free_actions, PlayerColor player);
        int CalculateIncomeGeneration(const Mali_BaState& state, PlayerColor player) const;
        Action ParseMoveStringToAction(const std::string& move_str) const;
        Move ActionToMove(Action action) const;
        Action MoveToAction(const Move& move) const; // New helper
        void ClearCaches();
        void RefreshTerminalStatus() { is_terminal_ = IsTerminal(); }
        Phase CurrentPhase() const { return current_phase_; }
        std::string DebugString() const { return ToString(); }

        // Getters that now delegate to the Game object
        const Mali_BaGame *GetGame() const;
        const std::set<HexCoord>& ValidHexes() const;
        const std::vector<City>& GetCities() const;
        int GridRadius() const;
        
        // Getters for dynamic state 
        const std::vector<std::map<std::string, int>>& GetCommonGoods() const { return common_goods_; }
        const std::vector<std::map<std::string, int>>& GetRareGoods() const { return rare_goods_; }
        PlayerColor GetPlayerTokenAt(const HexCoord &hex) const;
        const std::vector<MeepleColor> &GetMeeplesAt(const HexCoord &hex) const;
        const std::vector<TradePost> &GetTradePostsAt(const HexCoord &hex) const;
        const std::vector<TradeRoute> &GetTradeRoutes() const { return trade_routes_; }
        bool IsValidHex(const HexCoord &hex) const;
        PlayerColor GetCurrentPlayerColor() const { return current_player_color_; }
        Player GetPlayerId(PlayerColor color) const;
        PlayerColor GetPlayerColor(Player id) const;
        PlayerColor GetNextPlayerColor(PlayerColor current) const;
        const std::map<std::string, int> &GetPlayerCommonGoods(Player player) const;
        const std::map<std::string, int> &GetPlayerRareGoods(Player player) const;
        int GetCommonGoodCount(Player player, const std::string &good_name) const;
        int GetRareGoodCount(Player player, const std::string &good_name) const;
        std::string GetGameEndReason() const { return game_end_reason_; }
        int GetWinningPlayer() const { return winning_player_; }
        int GetGameEndTriggeringPlayer() const { return game_end_triggered_by_player_; }

        
        // --- Make GetRNG() a const method ---
        std::mt19937& GetRNG() const { return rng_; }

        // TestOnly setters and other methods
        void TestOnly_SetCurrentPlayer(Player player);
        void TestOnly_SetTradePost(const HexCoord& hex, PlayerColor owner, TradePostType type);
        void TestOnly_SetPlayerToken(const HexCoord& hex, PlayerColor owner);
        void TestOnly_SetPlayerTokens(const HexCoord& hex, const std::vector<PlayerColor>& owners); 
        void TestOnly_SetMeeples(const HexCoord& hex, const std::vector<MeepleColor>& meeples);
        void TestOnly_SetCommonGood(Player player, const std::string& good_name, int count);
        void TestOnly_SetRareGood(Player player, const std::string& good_name, int count);
        void TestOnly_ClearPlayerTokens();
        void TestOnly_ClearMeeples();

        // Setters
        void SetMoveLoggingEnabled(bool answ) {
            move_logging_enabled_ = answ;
        }
        void SetCurrentPhase(Phase phase) {
            current_phase_ = phase;
        }
        void SetCommonGoods(const std::vector<std::map<std::string, int>>& goods) {
            common_goods_ = goods;
        }
        void SetRareGoods(const std::vector<std::map<std::string, int>>& goods) {
            rare_goods_ = goods;
        }        void ApplyIncomeCollection(const std::string& action_str);

        void AddTradingPost(const HexCoord &hex, PlayerColor player, TradePostType type);
        void UpgradeTradingPost(const HexCoord &hex, PlayerColor player);
        bool CanPlaceTradingPostAt(const HexCoord &hex, PlayerColor player) const;
        int CountTradingCentersAt(const HexCoord &hex) const;
        bool HasPlayerPostOrCenterAt(const HexCoord &hex, PlayerColor player) const;
        bool CreateTradeRoute(const std::vector<HexCoord>& hexes, PlayerColor player);
        // bool UpdateTradeRoute(int route_id, const std::vector<HexCoord>& hexes);
        bool DeleteTradeRoute(int route_id);

        // Public helper methods like HasTokenAt, CountTokensAt 
        bool HasTokenAt(const HexCoord& hex, PlayerColor color) const;
        int CountTokensAt(const HexCoord& hex, PlayerColor color) const;
        int CountTotalTokensAt(const HexCoord& hex) const;
        std::vector<PlayerColor> GetTokensAt(const HexCoord& hex) const;
        bool RemoveTokenAt(const HexCoord& hex, PlayerColor color);
        void AddTokenAt(const HexCoord& hex, PlayerColor color);
        PlayerColor GetFirstTokenAt(const HexCoord& hex) const;
        
        // Move logging and Python sync helpers
        void InitializeMoveLogging();
        void LogMove(const std::string& action_string, const std::string& state_json);
        std::string CreateSetupJson() const;
        static std::string GetMoveLogFilename() { return move_log_filename_; }
        HexCoord ParseHexCoordFromString(const std::string& coord_str) const;
        std::vector<HexCoord> ParseHexListFromData(const std::vector<std::vector<int>>& hex_data) const;
        void ValidateTradeRoutes();
        int CountAllMeeples() const;
        int GetMaxMeeplesOnHex() const;
        void DebugPrintStateDetails() const;
        bool SetStateFromJson(const std::string& json_str);
        std::string GetCurrentStateJson() const { return Serialize(); }
        bool SaveStateToFile(const std::string& filename) const;
        bool LoadStateFromFile(const std::string& filename);
        void ResetToInitialState();

    protected:
        void DoApplyAction(Action action) override;

    private:
        // Private Dynamic State Members 
        std::vector<double> cumulative_returns_;
        Phase current_phase_;
        Player current_player_id_;
        PlayerColor current_player_color_;
        std::map<HexCoord, std::vector<PlayerColor>> player_token_locations_;
        std::vector<int> player_posts_supply_;
        std::map<HexCoord, std::vector<MeepleColor>> hex_meeples_;
        std::map<HexCoord, std::vector<TradePost>> trade_posts_locations_;
        std::vector<std::map<std::string, int>> common_goods_;
        std::vector<std::map<std::string, int>> rare_goods_;
        std::vector<TradeRoute> trade_routes_;
        int next_route_id_ = 1;
        std::vector<Move> moves_history_;
        mutable std::mt19937 rng_;
        mutable bool is_terminal_ = false;
        mutable absl::optional<LegalActionsResult> cached_legal_actions_result_;
        std::vector<StateSnapshot> undo_stack_;
        mutable int game_end_triggered_by_player_ = -1;  // -1 means not set
        mutable int winning_player_ = -1;                // -1 means tie/not set
        mutable std::string game_end_reason_;            // Description of how game ended

        // Private Static Members
        static std::unique_ptr<std::ofstream> move_log_file_;
        static std::string move_log_filename_;
        static int move_count_;
        static bool move_logging_initialized_;
        static bool move_logging_enabled_;

        // Private income/move generation helpers 
        struct IncomeChoice {
            HexCoord center_hex;
            std::vector<const City*> connected_cities;
            int num_choices;
            bool is_isolated;
        };
        struct PostChoice {
            HexCoord post_hex;
            std::vector<const City*> equidistant_cities;
        };
        // A helper struct to hold pre-calculated context for the heuristic.
        // This avoids passing many parameters around.
        struct HeuristicContext {
            int posts_in_supply;
            std::vector<HexCoord> existing_centers;
            std::set<int> existing_center_regions;
        };
        // Creates the context needed by the heuristic calculation.
        HeuristicContext CreateHeuristicContext() const;
        // The core helper function that calculates the weight for a single move.
        double CalculateHeuristicWeightForAction(
            const Move& move,
            const LegalActionsResult& legal_actions_result,
            const HeuristicContext& context) const;

        bool IsValidTradeRouteForMoveGeneration(
            const std::vector<HexCoord>& route_hexes, PlayerColor player) const;
        bool IsValidCompoundUpgradeAndRoute(const HexCoord& upgrade_hex, 
            const std::vector<HexCoord>& route_path, 
            PlayerColor player) const;
        bool HasSufficientResourcesForUpgrade(Player player_id) const;
        std::vector<Move> GenerateIncomeMoves() const;
        std::vector<Move> GenerateTradeRouteMoves() const;
        std::vector<const City*> GetConnectedCities(const HexCoord& center_hex, PlayerColor player) const;
        std::vector<const City*> FindClosestCities(const HexCoord& hex) const;
        void RemoveMeepleAt(const HexCoord& hex, int index);
        std::vector<HexCoord> GetCanonicalRoute(const std::vector<HexCoord>& route_combination) const;
        std::vector<std::vector<HexCoord>> FindPossibleTradeRoutes(
            PlayerColor player,
            bool is_valid_per_rules,
            const HexCoord* includes_hex = nullptr,
            int max_hexes = -1,
            int min_hexes = -1) const;
        // Convenience overloads for common use cases
        std::vector<std::vector<HexCoord>> FindPossibleTradeRoutes(
            PlayerColor player,
            bool is_valid_per_rules) const;
        std::vector<std::vector<HexCoord>> FindPossibleTradeRoutes(
            PlayerColor player,
            bool is_valid_per_rules,
            const HexCoord& includes_hex) const;
        std::vector<std::vector<HexCoord>> FindPossibleTradeRoutes(
            PlayerColor player,
            bool is_valid_per_rules,
            const HexCoord& includes_hex,
            int max_hexes) const;

        
        std::string NormalizeIncomeAction(const std::string& action_string) const;
        std::string CreateIncomeActionString(
            const std::map<std::string, int>& common_goods,
            const std::map<std::string, int>& rare_goods) const;
        bool IsRareGood(const std::string& good_name) const;
        std::vector<Move> GeneratePlaceTokenMoves() const;
        std::vector<Move> GenerateMancalaMoves() const;
        std::vector<Move> GenerateTradePostUpgradeMoves() const;
        void FindMancalaPathsRecursive(const HexCoord &start_hex, const HexCoord &current_hex,
            int remaining_meeples, std::vector<HexCoord> &currentt_path,
            std::set<HexCoord> &visited_hexes, std::vector<Move> &all_valid_moves) const;
        void FindMancalaPathsConstrainedRecursive(const HexCoord &original_start_hex, const HexCoord &currentt_hex,
            int remaining_meeples, std::vector<HexCoord> &path_so_far,
            std::set<HexCoord> &visited_hexes, std::vector<Move> &all_valid_moves) const;

        void GenerateRandomMancalaPaths(
            const Mali_BaState& state,
            const HexCoord& start_hex,
            int num_meeples,
            int num_paths_to_generate,
            std::vector<Move>& legal_moves) const;

        void ApplyMancalaMove(const Move &move);
        void ApplyPlaceTokenMove(const Move &move);
        void ApplyTradingPostUpgrade(const Move &move);
        void ApplyTradeRouteCreate(const Move& move);
        void ApplyPlacePostFromMancala(const Move &move);
        void ApplyTradeRouteUpdate(const Move& move);
        void ApplyTradeRouteDelete(const Move& move);
        void InitializeBoard();
        void ApplyChanceSetup();
        void PushStateToUndoStack();
        absl::optional<std::vector<double>> MaybeFinalReturns() const;
        void ClearAllState();

        // Helper for autoregressive training approach
        std::vector<HexCoord> FindShortestPath(
            const HexCoord& start, const HexCoord& end, int max_length) const;
    };

    // =====================================================================
    // FUNCTION RELATIONSHIP SUMMARY
    // =====================================================================

    /*
    EARLIER vs NEW FUNCTIONS:

    1. UNDO FUNCTIONS:
    - UndoAction(Player, Action) [EARLIER] - OpenSpiel standard, parameter-based undo
    - UndoLastAction() [NEW] - Convenience method, automatically undoes last action
    - UndoToTurnStart() [NEW] - Multi-action undo for free action turns
    
    RELATIONSHIP: UndoLastAction() calls UndoAction() internally for OpenSpiel compliance
    USE CASE: Use UndoLastAction() for GUI/AI, UndoAction() for OpenSpiel framework

    2. RANDOM PLAY FUNCTIONS:
    - PlayRandomMoveAndSerialize() [EARLIER/NOW ENHANCED] - Single action, OpenSpiel compatible
    - PlayRandomTurnAndSerialize() [NEW] - Multiple actions, training-focused
    
    RELATIONSHIP: Both serve different purposes in the same codebase
    USE CASE: PlayRandomMoveAndSerialize() for algorithms, PlayRandomTurnAndSerialize() for training

    3. ACTION SELECTION:
    - SelectTrainingAwareRandomAction() [NEW] - Single action with training awareness
    - SelectRandomTurnActions() [NEW] - Complete turn planning
    
    INTEGRATION: These work together to provide both single-action and multi-action AI

    DESIGN PHILOSOPHY:
    - Maintain backward compatibility with existing OpenSpiel patterns
    - Add enhanced functionality for free actions and training
    - Provide multiple interfaces for different use cases
    - Ensure all new functions work with existing OpenSpiel framework
    */

    // =====================================================================
    // Integration with OpenSpiel Random Play
    // =====================================================================
    // For OpenSpiel algorithms that expect single actions, create a wrapper:
    class Mali_BaTurnBasedWrapper {
    public:
        static Action SelectSingleRandomAction(Mali_BaState* state) {
            if (state->IsTerminal() || state->IsChanceNode()) {
                std::vector<Action> legal_actions = state->LegalActions();
                if (legal_actions.empty()) return kInvalidAction;
                
                std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
                return legal_actions[dist(state->GetRNG())];  // Now GetRNG() exists
            }
            
            // For regular player turns, we need to decide:
            // 1. Take a free action (if available and sometimes)
            // 2. Take a regular action
            
            std::vector<Action> legal_actions = state->LegalActions();
            if (legal_actions.empty()) return kInvalidAction;
            
            // Randomly decide whether to take a free action (30% chance)
            std::uniform_real_distribution<double> free_action_prob(0.0, 1.0);
            bool take_free_action = free_action_prob(state->GetRNG()) < 0.3;
            
            if (take_free_action) {
                // Look for free actions
                std::vector<Action> free_actions;
                for (Action action : legal_actions) {
                    Move move = state->ActionToMove(action);
                    if (move.type == ActionType::kTradeRouteCreate ||
                        move.type == ActionType::kTradeRouteUpdate ||
                        move.type == ActionType::kTradeRouteDelete) {
                        free_actions.push_back(action);
                    }
                }
                
                if (!free_actions.empty()) {
                    std::uniform_int_distribution<int> dist(0, free_actions.size() - 1);
                    return free_actions[dist(state->GetRNG())];  // Now GetRNG() exists
                }
            }
            
            // Take a regular action
            std::vector<Action> regular_actions;
            for (Action action : legal_actions) {
                Move move = state->ActionToMove(action);
                if (move.type != ActionType::kTradeRouteCreate &&
                    move.type != ActionType::kTradeRouteUpdate &&
                    move.type != ActionType::kTradeRouteDelete) {
                    regular_actions.push_back(action);
                }
            }
            
            if (regular_actions.empty()) {
                // Fallback to any legal action
                std::uniform_int_distribution<int> dist(0, legal_actions.size() - 1);
                return legal_actions[dist(state->GetRNG())];
            }
            
            std::uniform_int_distribution<int> dist(0, regular_actions.size() - 1);
            return regular_actions[dist(state->GetRNG())];  // Now GetRNG() exists
        }
    };
}
}
#endif