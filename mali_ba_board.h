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

#ifndef OPEN_SPIEL_GAMES_MALI_BA_BOARD_H_
#define OPEN_SPIEL_GAMES_MALI_BA_BOARD_H_

// Define this to use the simplified test board instead of the full game board
#define MALI_BA_SIMPLIFIED_BOARD

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <memory>
#include <random>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/mali_ba/hex_grid.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"

namespace open_spiel {
namespace mali_ba {

// Forward declaration of Mali_BaBoard
class Mali_BaBoard;

// Color representation for players
enum class Color : int8_t { kEmpty = -1, kBlack = 0, kWhite = 1, kRed = 2, kBlue = 3 };

inline std::string ColorToString(Color c) {
  switch (c) {
    case Color::kBlack: return "Black";
    case Color::kWhite: return "White";
    case Color::kRed: return "Red";
    case Color::kBlue: return "Blue";
    case Color::kEmpty: return "Empty";
    default: SpielFatalError("Unknown color");
  }
}

inline Color OppColor(Color color) {
  switch (color) {
    case Color::kBlack: return Color::kWhite;
    case Color::kWhite: return Color::kBlack;
    case Color::kRed: return Color::kBlue;
    case Color::kBlue: return Color::kRed;
    default: SpielFatalError("No opponent for this color");
  }
}

// Add this operator overload:
inline std::ostream& operator<<(std::ostream& stream, const Color& c) {
  // Reuse the existing ColorToString function for consistency
  stream << ColorToString(c);
  return stream;
}

// Cultural regions on the board
enum class Region : int8_t {
  kEmpty = 0,
  kTuareg,
  kDogon,
  kFulani,
  kSonghai,
  kAkan,
  kHousa,
  kWolof,
  kDagbani,
  kArab,
  kYoruba,
  kMande,
  kSenoufo,
  kKru,
  kIdjo
};

std::string RegionToString(Region r);

// Meeple types (representing traders, resources, etc.)
enum class MeepleType : int8_t {
  kEmpty = 0,
  kTrader,    // Generic trader meeple
  kGold,      // Resource meeples
  kSpice,
  kSalt,
  kIvory,
  kCotton
};

std::string MeepleTypeToString(MeepleType t);

// City representation
struct City {
  std::string name;
  Region region;
  std::string common_good;
  std::string rare_good;
  HexCoord location;
  
  City(const std::string& name, Region region, 
       const std::string& common_good, const std::string& rare_good,
       const HexCoord& location)
      : name(name), region(region), common_good(common_good), 
        rare_good(rare_good), location(location) {}
};

// Trading post/center representation
enum class TradePostType : int8_t {
  kNone = 0,
  kPost,       // Basic trading post
  kCenter      // Upgraded trading center
};

// Add this operator overload:
inline std::ostream& operator<<(std::ostream& stream, const TradePostType& tpt) {
  switch (tpt) {
    case TradePostType::kNone:
      stream << "None";
      break;
    case TradePostType::kPost:
      stream << "Post";
      break;
    case TradePostType::kCenter:
      stream << "Center";
      break;
    default:
      // Optional: Handle potential invalid enum values gracefully
      stream << "UnknownTradePostType(" << static_cast<int>(tpt) << ")";
      // Or call SpielFatalError for stricter checking:
      // SpielFatalError(absl::StrCat("Unknown TradePostType enum value: ", static_cast<int>(tpt)));
  }
  return stream; // Return the stream to allow chaining
}

// A piece on the board
struct Piece {
  Color color;
  MeepleType type;
  
  bool operator==(const Piece& other) const {
    return color == other.color && type == other.type;
  }
  
  bool operator!=(const Piece& other) const {
    return !(*this == other);
  }
  
  std::string ToString() const;
};

// This doesn't need constexpr since it's a simple struct initialization
static inline const Piece kEmptyPiece = {Color::kEmpty, MeepleType::kEmpty};

// A move representing a Mancala-style path
struct Move {
  Color player_color;                   // Which player is making the move
  HexCoord start_hex;                   // Starting hex with meeples
  std::vector<HexCoord> path;           // Path of hexes where meeples are dropped
  bool place_trading_post;              // Whether to place a trading post at the end
  
  // Indicates this is a pass move
  bool is_pass() const { return start_hex == kInvalidHex; }
  
  std::string ToString() const;
  
  // Convert to human-readable format
  std::string ToLAN() const;
  
  bool operator==(const Move& other) const {
    return player_color == other.player_color &&
           start_hex == other.start_hex &&
           path == other.path &&
           place_trading_post == other.place_trading_post;
  }
};

// Special pass move constant
// Initialized in the implementation file to avoid issues with kInvalidHex
extern const Move kPassMove;

// Trading post ownership
struct TradePost {
  Color owner;
  TradePostType type;
  
  bool operator==(const TradePost& other) const {
    return owner == other.owner && type == other.type;
  }
  
  bool operator!=(const TradePost& other) const {
    return !(*this == other);
  }
};

// Mali-Ba board implementation
class Mali_BaBoard {
 public:
  // Constructor for a new game with default setup
  Mali_BaBoard(int grid_radius = 5);
  
#ifdef MALI_BA_SIMPLIFIED_BOARD
  // For simplified board, use a smaller default radius
  static inline int DefaultGridRadius() { return 2; }
#else
  // For full board, use a larger default radius
  static inline int DefaultGridRadius() { return 5; }
#endif
  
  // Constructor from a Mancala Board Notation (MBN) string
  static absl::optional<Mali_BaBoard> BoardFromMBN(const std::string& mbn);
  
  // Get piece at a specific hex
  const Piece& at(const HexCoord& hex) const;
  
  // Set piece at a specific hex
  void set_piece(const HexCoord& hex, const Piece& piece);
  
  // Get trading post info at a specific hex
  const TradePost& post_at(const HexCoord& hex) const;
  
  // Set trading post at a specific hex
  void set_post(const HexCoord& hex, const TradePost& post);
  
  // Get current player's color
  Color ToPlay() const { return to_play_; }
  
  // Set current player
  void SetToPlay(Color c);
  
  // Get board radius
  int GridRadius() const { return grid_radius_; }
  
  // Apply a move to the board
  void ApplyMove(const Move& move);
  
  // Check if a hex is within the board area
  bool InBoardArea(const HexCoord& hex) const;
  
  // Check if a hex is empty
  bool IsEmpty(const HexCoord& hex) const;
  
  // Check if a hex belongs to the opponent
  bool IsEnemy(const HexCoord& hex, Color our_color) const;
  
  // Check if a hex has our piece
  bool IsFriendly(const HexCoord& hex, Color our_color) const;
  
  // Convert board to MBN string representation
  std::string ToMBN() const;
  
  // Debug string representation
  std::string DebugString() const;
  
  // Unicode string representation for pretty printing
  std::string ToUnicodeString() const;
  
  // Generate all legal moves for the current player
  using MoveYieldFn = std::function<bool(const Move&)>;
  void GenerateLegalMoves(const MoveYieldFn& yield) const;
  void GenerateLegalMoves(const MoveYieldFn& yield, Color color) const;
  
  // Get cities on the board
  const std::vector<City>& GetCities() const { return cities_; }
  
  // Find a city at a given location
  absl::optional<const City*> CityAt(const HexCoord& hex) const;
  
  // Get meeples count at a specific hex
  int MeeplesCount(const HexCoord& hex) const;
  
  // Get unique hash for board state (for repetition detection)
  uint64_t HashValue() const { return zobrist_hash_; }
  
  // Count number of connected trading posts for a player
  int CountConnectedPosts(Color color) const;
  
 private:
  // Helper method to generate paths for Mancala-style moves
  void GeneratePaths(const HexCoord& start, 
                     std::vector<HexCoord>& current_path,
                     std::vector<bool>& visited,
                     int remaining_meeples,
                     const MoveYieldFn& yield,
                     Color color) const;
  
  // Helper method to initialize the board with cities
  void InitializeCities();
  
  // Helper method to initialize the board with meeples
  void InitializeMeeples();
  
  // Helper method to check if a trading post can be placed
  bool CanPlaceTradingPost(const HexCoord& hex, Color color) const;
  
  // Maps hex coordinates to their index in the board array
  int HexToIndex(const HexCoord& hex) const {
    return ::open_spiel::mali_ba::HexToIndex(hex, grid_radius_);
  }
  
  // The size/radius of the hex grid
  int grid_radius_;
  
  // Pieces on the board
  std::vector<Piece> board_;
  
  // Trading posts on the board
  std::vector<TradePost> trading_posts_;
  
  // Cities on the board
  std::vector<City> cities_;
  
  // Current player
  Color to_play_;
  
  // Move number
  int move_number_;
  
  // Hash value for board state
  uint64_t zobrist_hash_;
};

// Create a default board with standard setup
Mali_BaBoard MakeDefaultBoard();

}  // namespace mali_ba
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MALI_BA_BOARD_H_