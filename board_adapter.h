#ifndef OPEN_SPIEL_GAMES_MALI_BA_BOARD_ADAPTER_H_
#define OPEN_SPIEL_GAMES_MALI_BA_BOARD_ADAPTER_H_

#include <memory>
#include <string>
#include <vector>
#include <set>

#include "open_spiel/games/mali_ba/hex_grid.h"

namespace open_spiel {
namespace mali_ba {

// Forward declarations
class Mali_BaBoard;
class FlexBoard;
struct Piece;
struct TradePost;
struct City;
struct Move;
enum class Color : int8_t;

// This adapter class provides a consistent interface for the game logic
// to work with either the original Mali_BaBoard or the new FlexBoard
class BoardAdapter {
 public:
  // Create an adapter for the original fixed board
  explicit BoardAdapter(const Mali_BaBoard& board);
  
  // Create an adapter for the new flexible board
  explicit BoardAdapter(const FlexBoard& board);
  
  // Constructor for a board from a board notation string
  static BoardAdapter FromMBN(const std::string& mbn, bool use_flexible_board = false);
  
  // Static factory functions for different board types
  static BoardAdapter CreateStandardBoard(int radius, bool use_flexible_board = false);
  static BoardAdapter CreateSimplifiedBoard(bool use_flexible_board = false);
  static BoardAdapter CreateCustomBoard(const std::vector<HexCoord>& valid_hexes);
  
  // Core board operations
  Piece GetPiece(const HexCoord& hex) const;
  void SetPiece(const HexCoord& hex, const Piece& piece);
  
  TradePost GetTradePost(const HexCoord& hex) const;
  void SetTradePost(const HexCoord& hex, const TradePost& post);
  
  bool IsValidHex(const HexCoord& hex) const;
  
  Color ToPlay() const;
  void SetToPlay(Color c);
  
  int GridRadius() const;
  
  void ApplyMove(const Move& move);
  
  // Game state operations
  std::string ToMBN() const;
  std::string DebugString() const;
  
  // Use FlexBoard's MoveYieldFn type here to avoid circular dependencies
  using MoveYieldFn = std::function<bool(const Move&)>;
  void GenerateLegalMoves(const MoveYieldFn& yield) const;
  void GenerateLegalMoves(const MoveYieldFn& yield, Color color) const;
  
  const std::vector<City>& GetCities() const;
  absl::optional<const City*> CityAt(const HexCoord& hex) const;
  
  int MeeplesCount(const HexCoord& hex) const;
  
  uint64_t HashValue() const;
  
  int CountConnectedPosts(Color color) const;
  
  bool IsUsingFlexBoard() const { return using_flexible_board_; }
  
  // Advanced flexible board operations
  std::vector<HexCoord> GetValidNeighbors(const HexCoord& hex) const;
  bool AreConnected(const HexCoord& a, const HexCoord& b) const;
  std::vector<std::set<HexCoord>> GetConnectedComponents() const;
  
  // Access to the underlying board implementations
  const Mali_BaBoard* GetFixedBoard() const;
  const FlexBoard* GetFlexBoard() const;
  
 private:
  bool using_flexible_board_;
  
  // We use unique_ptr to avoid object slicing and allow polymorphic behavior
  std::unique_ptr<Mali_BaBoard> fixed_board_;
  std::unique_ptr<FlexBoard> flexible_board_;
};

}  // namespace mali_ba
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MALI_BA_BOARD_ADAPTER_H_