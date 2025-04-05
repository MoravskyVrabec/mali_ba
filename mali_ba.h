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
#include <map>
#include <set>
#include <iomanip>  // For std::setw

#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/memory/memory.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/mali_ba/hex_grid.h"
#include "open_spiel/games/mali_ba/board_config.h"
// mali_ba_board.h removed from new implementation
// board_adapter.h removed from new implementation
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/observer.h"

// Mali-Ba game parameters:
//   "players"             int   number of players (2-4)  (default: 2)
//   "grid_radius"         int   radius of the hex grid board (default: 5)
//   "simplified"          bool  use simplified board for testing (default: true)
//   "use_flexible_board"  bool  use flexible board implementation (default: false)

namespace open_spiel {
namespace mali_ba {

// Forward declarations

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

// Enum for piece colors
enum class Color {
    kEmpty = -1,
    kBlack = 0,
    kWhite = 1,
    kRed = 2,
    kBlue = 3
};

// Enum for meeple types
enum class MeepleType {
    kEmpty = 0,
    kCommon = 1,
    kRare = 2,
    kYellow = 3,  // Vizier
    kWhite = 4,   // Elder
    kBlue = 5,    // Builder
    kRed = 6,     // Assassin
    kGreen = 7    // Merchant
};

// Enum for trading post types
enum class TradePostType {
    kNone = 0,
    kPost = 1,
    kCenter = 2
};

// Structure to represent a game piece
struct Piece {
    Color color = Color::kEmpty;
    MeepleType type = MeepleType::kEmpty;
    
    bool operator==(const Piece& other) const {
        return color == other.color && type == other.type;
    }
    
    bool operator!=(const Piece& other) const {
        return !(*this == other);
    }
};

// Structure to represent a trading post
struct TradePost {
    Color owner = Color::kEmpty;
    TradePostType type = TradePostType::kNone;
    
    bool operator==(const TradePost& other) const {
        return owner == other.owner && type == other.type;
    }
    
    bool operator!=(const TradePost& other) const {
        return !(*this == other);
    }
};

// Structure to represent a city
struct City {
    std::string name;
    std::string culture;
    HexCoord location;
    std::string common_good;
    std::string rare_good;
};

// Structure to represent a move
struct Move {
    Color player;
    HexCoord start_hex;
    std::vector<HexCoord> path;
    bool place_trading_post = false;
    
    bool is_pass() const {
        return path.empty();
    }
    
    std::string ToString() const {
        if (is_pass()) {
            return "Pass";
        }
        
        std::string result = start_hex.ToString();
        for (const auto& hex : path) {
            result += ":" + hex.ToString();
        }
        
        if (place_trading_post) {
            result += ":post";
        }
        
        return result;
    }
};

// Empty piece for default returns
const Piece kEmptyPiece{Color::kEmpty, MeepleType::kEmpty};

// Pass move constant
const Move kPassMove{Color::kEmpty, HexCoord(0, 0, 0), {}, false};

// Convert hex coordinate to index
inline int HexToIndex(const HexCoord& hex, int grid_radius) {
    // Convert cube coordinates to an index in the range [0, 3r(r+1)]
    // where r is the grid radius
    // This assumes all valid hexes are within the radius
    
    // First, shift coordinates to be non-negative
    int x = hex.x + grid_radius;
    int y = hex.y + grid_radius;
    
    // Convert to a unique index
    return x * (2 * grid_radius + 1) + y;
}

// Convert index to hex coordinate
inline HexCoord IndexToHex(int index, int grid_radius) {
    // Convert index back to cube coordinates
    int width = 2 * grid_radius + 1;
    int x = index / width - grid_radius;
    int y = index % width - grid_radius;
    int z = -x - y;
    
    return HexCoord(x, y, z);
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

// Helper function to convert color to string
inline std::string ColorToString(Color c) {
    switch (c) {
        case Color::kBlack: return "Black";
        case Color::kWhite: return "White";
        case Color::kRed: return "Red";
        case Color::kBlue: return "Blue";
        case Color::kEmpty: return "Empty";
        default: return "Unknown";
    }
}

// Forward declaration for function in mali_ba.cc
// uint64_t CalculateBoardHash(const std::map<HexCoord, Piece>& pieces,
//                            const std::map<HexCoord, TradePost>& posts,
//                            Color current_player);

// Conversion between Moves and Actions
Action MoveToAction(const Move& move, int grid_radius);
Move ActionToMove(Action action, int grid_radius);

// State of an in-play game
class Mali_BaState : public State {
 public:
   // Constructs a Mali-Ba state at the standard start position
   Mali_BaState(std::shared_ptr<const Game> game);
   
   // Constructs a Mali-Ba state from a provided string
   Mali_BaState(std::shared_ptr<const Game> game, const std::string& str);
   
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
   
   // Helper methods
   std::string DebugString() const { return ToString(); }
   Action ParseMoveToAction(const std::string& move_str) const;
   std::string Serialize() const override;
   
   // Get valid hexes
   const std::set<HexCoord>& ValidHexes() const { return valid_hexes_; }
   
   // Check if a hex is valid
   bool IsValidHex(const HexCoord& hex) const {
     return valid_hexes_.count(hex) > 0;
   }
   
   // Get piece at a specific hex
   const Piece& GetPiece(const HexCoord& hex) const {
     auto it = pieces_.find(hex);
     return (it != pieces_.end()) ? it->second : kEmptyPiece;
   }
   
   // Set piece at a specific hex
   void SetPiece(const HexCoord& hex, const Piece& piece) {
     if (IsValidHex(hex)) {
       pieces_[hex] = piece;
     }
   }
   
   // Access to the game board state
   const std::map<HexCoord, Piece>& GetPieces() const { return pieces_; }
   const std::map<HexCoord, TradePost>& GetTradePosts() const { return trade_posts_; }
   const std::vector<City>& GetCities() const { return cities_; }
   // Member variables that need to be accessible
   std::set<HexCoord> valid_hexes_;
   std::map<HexCoord, Piece> pieces_;
   std::map<HexCoord, TradePost> trade_posts_;
   std::vector<City> cities_;
   Color current_player_ = Color::kBlack;
   std::vector<Move> moves_history_;
   int grid_radius_ = 3;
   
   
   // Get current player
   Color GetCurrentPlayer() const { return current_player_; }
   
   // Count connected trading posts for a player - for win condition check
   int CountConnectedPosts(Color color) const;
   
  protected:
   // Apply an action to the current state
   void DoApplyAction(Action action) override;
   
  private:
   // Initialize the board with valid hexes
   void InitializeBoard();
   
   // Initialize meeples on the board
   void InitializeMeeples();
   
   // Store the initial state for undo operations
   void StoreInitialState();
   
   // Helper method to apply a Mancala-style move
   void ApplyMancalaMove(const Move& move);
   
   // Helper method to place a trading post
   void PlaceTradePost(const Move& move);
   
   // Helper to get the next player in turn
   Color GetNextPlayer(Color current) const;
   
   // Generate legal actions and cache them
   void MaybeGenerateLegalActions() const;
   
   // Check if the game has ended and return final scores
   absl::optional<std::vector<double>> MaybeFinalReturns() const;
   
   // Initial pieces and posts for undo operations
   std::map<HexCoord, Piece> start_pieces_;
   std::map<HexCoord, TradePost> start_trade_posts_;
   
   // Flag for game end
   mutable bool is_terminal_ = false;
   
   // Cache for legal actions
   mutable absl::optional<std::vector<Action>> cached_legal_actions_;
   
   // Hash table to track repeated board positions (for draw detection)
   using RepetitionTable = absl::flat_hash_map<uint64_t, int>;
   mutable RepetitionTable repetitions_;
 };

// Game object for Mali-Ba
class Mali_BaGame : public Game {
 public:
  explicit Mali_BaGame(const GameParameters& params);
  
  // Required Game interface methods
  int NumDistinctActions() const override { return mali_ba::NumDistinctActions(); }
  
  std::unique_ptr<State> NewInitialState() const override {
    return std::make_unique<Mali_BaState>(shared_from_this());
  }
  
  std::unique_ptr<State> NewInitialState(const std::string& str) const override {
    return std::make_unique<Mali_BaState>(shared_from_this(), str);
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
  bool use_flexible_board_ = false;  // Default to not using flexible board

 private:
  // Board configuration
  BoardConfig board_config_;
  
 public:
  // Accessor for board configuration
  const BoardConfig& GetBoardConfig() const;
};

}  // namespace mali_ba
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MALI_BA_H_