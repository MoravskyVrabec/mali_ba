#ifndef OPEN_SPIEL_GAMES_MALI_BA_FLEX_BOARD_H_
#define OPEN_SPIEL_GAMES_MALI_BA_FLEX_BOARD_H_

#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <memory>
#include <set>
#include <map>
#include <random>

#include "open_spiel/spiel_utils.h"
#include "open_spiel/games/mali_ba/hex_grid.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"

namespace open_spiel {
namespace mali_ba {

// Forward declarations (existing types from current implementation)
enum class Color : int8_t;
enum class MeepleType : int8_t;
enum class Region : int8_t;
enum class TradePostType : int8_t;
struct City;
struct Piece;
struct TradePost;
struct Move;

// The new flexible board representation will use a map-based approach
// instead of a linear array indexed by position
class FlexBoard {
 public:
  // Constructor for a new empty board
  FlexBoard(int max_radius = 5);
  
  // Constructor with a set of valid hexes
  FlexBoard(const std::set<HexCoord>& valid_hexes, int max_radius = 5);
  
  // Create a standard full board (no holes) with given radius
  static FlexBoard CreateStandardBoard(int radius);
  
  // Create a board with the simplified setup for testing
  static FlexBoard CreateSimplifiedBoard();
  
  // Create a simplified custom board with predefined layout
  static FlexBoard CreateSimplifiedCustomBoard();
  
  // Create a custom board with specific hex structure
  static FlexBoard CreateCustomBoard(const std::vector<HexCoord>& valid_hexes);
  
  // Add a valid hex to the board
  void AddHex(const HexCoord& hex);
  
  // Remove a hex from the board, making it invalid
  void RemoveHex(const HexCoord& hex);
  
  // Check if a hex is valid on the board
  bool IsValidHex(const HexCoord& hex) const;
  
  // Get piece at a specific hex
  Piece GetPiece(const HexCoord& hex) const;
  
  // Set piece at a specific hex
  void SetPiece(const HexCoord& hex, const Piece& piece);
  
  // Get trading post info at a specific hex
  TradePost GetTradePost(const HexCoord& hex) const;
  
  // Set trading post at a specific hex
  void SetTradePost(const HexCoord& hex, const TradePost& post);
  
  // Get current player's color
  Color ToPlay() const;
  
  // Set current player
  void SetToPlay(Color c);
  
  // Get board maximum radius
  int MaxRadius() const;
  
  // Apply a move to the board
  void ApplyMove(const Move& move);
  
  // Get all valid hexes on the board
  const std::set<HexCoord>& GetValidHexes() const;
  
  // Convert board to MBN string representation
  std::string ToMBN() const;
  
  // Debug string representation
  std::string DebugString() const;
  
  // Generate all legal moves for the current player
  using MoveYieldFn = std::function<bool(const Move&)>;
  void GenerateLegalMoves(const MoveYieldFn& yield) const;
  void GenerateLegalMoves(const MoveYieldFn& yield, Color color) const;
  
  // Get cities on the board
  const std::vector<City>& GetCities() const;
  
  // Clear all cities from the board
  void ClearCities();
  
  // Add a city to the board
  void AddCity(const std::string& name, Region region, 
               const std::string& common_good, const std::string& rare_good,
               const HexCoord& location);
  
  // Find a city at a given location
  absl::optional<const City*> CityAt(const HexCoord& hex) const;
  
  // Get meeples count at a specific hex
  int MeeplesCount(const HexCoord& hex) const;
  
  // Get unique hash for board state (for repetition detection)
  uint64_t HashValue() const;
  
  // Count number of connected trading posts for a player
  int CountConnectedPosts(Color color) const;
  
  // Get all valid adjacent hexes for a given hex
  std::vector<HexCoord> GetValidNeighbors(const HexCoord& hex) const;
  
  // Find all path-connected components on the board
  std::vector<std::set<HexCoord>> GetConnectedComponents() const;
  
  // Check if two hexes are connected by a valid path
  bool AreConnected(const HexCoord& a, const HexCoord& b) const;
  
 private:
  // Helper method to initialize the board with cities
  void InitializeCities();
  
  // Helper method to initialize the board with meeples
  void InitializeMeeples();
  
  // Helper method to check if a trading post can be placed
  bool CanPlaceTradingPost(const HexCoord& hex, Color color) const;
  
  // Helper method to generate paths for Mancala-style moves
  void GeneratePaths(const HexCoord& start, 
                     std::vector<HexCoord>& current_path,
                     std::set<HexCoord>& visited,
                     int remaining_meeples,
                     const MoveYieldFn& yield,
                     Color color) const;
  
  // Set of valid hexes on the board
  std::set<HexCoord> valid_hexes_;
  
  // Map from hex coordinates to pieces
  std::map<HexCoord, Piece> pieces_;
  
  // Map from hex coordinates to trading posts
  std::map<HexCoord, TradePost> trade_posts_;
  
  // Cities on the board
  std::vector<City> cities_;
  
  // Maximum radius of the board
  int max_radius_;
  
  // Current player
  Color to_play_;
  
  // Move number
  int move_number_;
  
  // Hash value for board state
  uint64_t zobrist_hash_;
};

} // namespace mali_ba
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_MALI_BA_FLEX_BOARD_H_