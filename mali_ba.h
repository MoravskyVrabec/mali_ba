// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef OPEN_SPIEL_GAMES_MALI_BA_H_
#define OPEN_SPIEL_GAMES_MALI_BA_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/mali_ba/mali_ba_board.h"
#include "open_spiel/games/mali_ba/board_adapter.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/observer.h"
#include <string>
#include <vector>

// Mali-Ba game parameters:
//   "players"             int   number of players (2-4)  (default: 2)
//   "grid_radius"         int   radius of the hex grid board (default: 5)
//   "simplified"          bool  use simplified board for testing (default: true)
//   "use_flexible_board"  bool  use flexible board implementation (default: true)

namespace open_spiel {
namespace mali_ba {

// Constants.
inline constexpr int NumPlayers() { return 2; }  // Default, can be changed by parameter
inline constexpr double LossUtility() { return -1; }
inline constexpr double DrawUtility() { return 0; }
inline constexpr double WinUtility() { return 1; }

// Maximum number of distinct actions - this needs to be a large enough constant
// to cover all possible paths a player might take in a Mancala-style move
inline constexpr int NumDistinctActions() { return 10000; }
inline constexpr int kPassActionId = 0;  // Special pass action ID

// A rough upper bound on game length - can be refined
inline constexpr int MaxGameLength() { return 200; }

// Observation tensor shape - to be defined based on game state representation
inline const std::vector<int>& ObservationTensorShape() {
  static std::vector<int> shape{
      // Planes for piece types and colors, regions, etc.
      20,
      // Grid dimension - needs to match the representation of the hex grid
      11, 11
  };
  return shape;
}

// Helper to convert between Color and Player
inline int ColorToPlayer(Color c) {
  switch (c) {
    case Color::kBlack: return 0;
    case Color::kWhite: return 1;
    case Color::kRed: return 2;
    case Color::kBlue: return 3;
    default: SpielFatalError("Unknown color");
  }
}

inline Color PlayerToColor(Player p) {
  switch (p) {
    case 0: return Color::kBlack;
    case 1: return Color::kWhite;
    case 2: return Color::kRed;
    case 3: return Color::kBlue;
    default: SpielFatalError("Unknown player");
  }
}

// Conversion between Moves and Actions
Action MoveToAction(const Move& move, int grid_radius);
Move ActionToMove(Action action, const BoardAdapter& board);
Move ActionToMove(Action action, const Mali_BaBoard& board);

// State of an in-play game
class Mali_BaState : public State {
 public:
  // Constructs a Mali-Ba state at the standard start position
  Mali_BaState(std::shared_ptr<const Game> game);
  
  // Constructs a Mali-Ba state from a provided MBN string
  Mali_BaState(std::shared_ptr<const Game> game, const std::string& mbn);
  
  // Copy constructor
  Mali_BaState(const Mali_BaState&) = default;
  
  // Required methods from the State base class
  Player CurrentPlayer() const override;
  std::vector<Action> LegalActions() const override;
  std::string ActionToString(Player player, Action action) const override;
  std::string ToString() const override;
  bool IsTerminal() const override;
  std::vector<double> Returns() const override;
  std::string InformationStateString(Player player) const override;
  std::string ObservationString(Player player) const override;
  void ObservationTensor(Player player, absl::Span<float> values) const override;
  std::unique_ptr<State> Clone() const override;
  void UndoAction(Player player, Action action) override;
  
  // Access to the board via adapter
  BoardAdapter& Board() { return board_adapter_; }
  const BoardAdapter& Board() const { return board_adapter_; }
  
  // Move history
  std::vector<Move>& MovesHistory() { return moves_history_; }
  const std::vector<Move>& MovesHistory() const { return moves_history_; }
  
  // Helper methods
  std::string DebugString() const;
  Action ParseMoveToAction(const std::string& move_str) const;
  std::string Serialize() const override;
  
 protected:
  // Apply an action to the current state
  void DoApplyAction(Action action) override;
  
 private:
  // Generate legal actions and cache them
  void MaybeGenerateLegalActions() const;
  
  // Check if the game has ended and return final scores
  absl::optional<std::vector<double>> MaybeFinalReturns() const;
  
  // Initialize meeples on the board
  void InitializeMeeples();
  
  // Move history
  std::vector<Move> moves_history_;
  
  // Starting board (for undo and history)
  BoardAdapter start_board_;
  
  // Current board state
  BoardAdapter board_adapter_;
  
  // Flag for game end
  mutable bool is_terminal_ = false;
  
  // Cache for legal actions
  mutable absl::optional<std::vector<Action>> cached_legal_actions_;
  
  // Hash table to track repeated board positions (for draw detection)
  using RepetitionTable = absl::flat_hash_map<uint64_t, int>;
  RepetitionTable repetitions_;
};

// Game object for Mali-Ba
class Mali_BaGame : public Game {
 public:
  explicit Mali_BaGame(const GameParameters& params);
  
  // Required Game interface methods
  int NumDistinctActions() const override { return mali_ba::NumDistinctActions(); }
  
  std::unique_ptr<State> NewInitialState() const override {
    return absl::make_unique<Mali_BaState>(shared_from_this());
  }
  
  std::unique_ptr<State> NewInitialState(const std::string& mbn) const override {
    return absl::make_unique<Mali_BaState>(shared_from_this(), mbn);
  }
  
  int NumPlayers() const override { return num_players_; }
  double MinUtility() const override { return LossUtility(); }
  absl::optional<double> UtilitySum() const override { return DrawUtility(); }
  double MaxUtility() const override { return WinUtility(); }
  std::vector<int> ObservationTensorShape() const override {
    return mali_ba::ObservationTensorShape();
  }
  int MaxGameLength() const override { return mali_ba::MaxGameLength(); }
  
  std::unique_ptr<State> DeserializeState(const std::string& str) const override;
  
  // Override the base class MakeObserver method
  std::shared_ptr<Observer> MakeObserver(
      absl::optional<IIGObservationType> iig_obs_type,
      const GameParameters& params) const override;
      
  int GridRadius() const { return grid_radius_; }
  bool UseFlexibleBoard() const { return use_flexible_board_; }
  
 private:
  int num_players_;
  int grid_radius_;
  bool use_flexible_board_;
};

}  // namespace mali_ba
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MALI_BA_H_