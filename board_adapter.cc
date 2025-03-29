#include "open_spiel/games/mali_ba/board_adapter.h"

#include <memory>
#include <string>
#include <vector>

// Include full implementations after forward declarations
#include "open_spiel/games/mali_ba/mali_ba_board.h"
#include "open_spiel/games/mali_ba/flex_board.h"

namespace open_spiel {
namespace mali_ba {

// Default constructor - creates a standard simplified board
BoardAdapter::BoardAdapter()
    : using_flexible_board_(true),
      fixed_board_(nullptr),
      flexible_board_(std::make_unique<FlexBoard>(FlexBoard::CreateSimplifiedBoard())) {
}

// Create an adapter for the original fixed board
BoardAdapter::BoardAdapter(const Mali_BaBoard& board)
    : using_flexible_board_(false),
      fixed_board_(std::make_unique<Mali_BaBoard>(board)),
      flexible_board_(nullptr) {}

// Create an adapter for the new flexible board
BoardAdapter::BoardAdapter(const FlexBoard& board)
    : using_flexible_board_(true),
      fixed_board_(nullptr),
      flexible_board_(std::make_unique<FlexBoard>(board)) {}

// Copy constructor
BoardAdapter::BoardAdapter(const BoardAdapter& other)
    : using_flexible_board_(other.using_flexible_board_) {
  
  if (using_flexible_board_) {
    // Deep copy the flexible board
    if (other.flexible_board_) {
      flexible_board_ = std::make_unique<FlexBoard>(*other.flexible_board_);
    } else {
      flexible_board_ = nullptr;
    }
    fixed_board_ = nullptr;
  } else {
    // Deep copy the fixed board
    if (other.fixed_board_) {
      fixed_board_ = std::make_unique<Mali_BaBoard>(*other.fixed_board_);
    } else {
      fixed_board_ = nullptr;
    }
    flexible_board_ = nullptr;
  }
}

// Assignment operator
BoardAdapter& BoardAdapter::operator=(const BoardAdapter& other) {
  if (this != &other) {
    using_flexible_board_ = other.using_flexible_board_;
    
    if (using_flexible_board_) {
      // Deep copy the flexible board
      if (other.flexible_board_) {
        flexible_board_ = std::make_unique<FlexBoard>(*other.flexible_board_);
      } else {
        flexible_board_ = nullptr;
      }
      fixed_board_ = nullptr;
    } else {
      // Deep copy the fixed board
      if (other.fixed_board_) {
        fixed_board_ = std::make_unique<Mali_BaBoard>(*other.fixed_board_);
      } else {
        fixed_board_ = nullptr;
      }
      flexible_board_ = nullptr;
    }
  }
  return *this;
}

// Constructor for a board from a board notation string
BoardAdapter BoardAdapter::FromMBN(const std::string& mbn, bool use_flexible_board) {
  if (use_flexible_board) {
    // Create a flexible board from the MBN
    // This would need to parse the MBN format for the flexible board
    // For now, we'll just create a default board
    return BoardAdapter(FlexBoard::CreateStandardBoard(5));
  } else {
    // Use the existing Mali_BaBoard::BoardFromMBN function
    auto board_opt = Mali_BaBoard::BoardFromMBN(mbn);
    if (board_opt.has_value()) {
      return BoardAdapter(*board_opt);
    } else {
      // Fallback to default board
      return BoardAdapter(Mali_BaBoard(5));
    }
  }
}

// Static factory functions for different board types
BoardAdapter BoardAdapter::CreateStandardBoard(int radius, bool use_flexible_board) {
  if (use_flexible_board) {
    return BoardAdapter(FlexBoard::CreateStandardBoard(radius));
  } else {
    return BoardAdapter(Mali_BaBoard(radius));
  }
}

// Create a simplified board
BoardAdapter BoardAdapter::CreateSimplifiedBoard(bool use_flexible_board) {
  if (use_flexible_board) {
    return BoardAdapter(FlexBoard::CreateSimplifiedBoard());
  } else {
    return BoardAdapter(Mali_BaBoard(Mali_BaBoard::DefaultGridRadius()));
  }
}

BoardAdapter BoardAdapter::CreateCustomBoard(const std::vector<HexCoord>& valid_hexes) {
  // Custom boards are only supported with the flexible board implementation
  return BoardAdapter(FlexBoard::CreateCustomBoard(valid_hexes));
}

// Core board operations
Piece BoardAdapter::GetPiece(const HexCoord& hex) const {
  if (using_flexible_board_) {
    return flexible_board_->GetPiece(hex);
  } else {
    return fixed_board_->at(hex);
  }
}

void BoardAdapter::SetPiece(const HexCoord& hex, const Piece& piece) {
  if (using_flexible_board_) {
    flexible_board_->SetPiece(hex, piece);
  } else {
    fixed_board_->set_piece(hex, piece);
  }
}

TradePost BoardAdapter::GetTradePost(const HexCoord& hex) const {
  if (using_flexible_board_) {
    return flexible_board_->GetTradePost(hex);
  } else {
    return fixed_board_->post_at(hex);
  }
}

void BoardAdapter::SetTradePost(const HexCoord& hex, const TradePost& post) {
  if (using_flexible_board_) {
    flexible_board_->SetTradePost(hex, post);
  } else {
    fixed_board_->set_post(hex, post);
  }
}

bool BoardAdapter::IsValidHex(const HexCoord& hex) const {
  if (using_flexible_board_) {
    return flexible_board_->IsValidHex(hex);
  } else {
    return fixed_board_->InBoardArea(hex);
  }
}

Color BoardAdapter::ToPlay() const {
  if (using_flexible_board_) {
    return flexible_board_->ToPlay();
  } else {
    return fixed_board_->ToPlay();
  }
}

void BoardAdapter::SetToPlay(Color c) {
  if (using_flexible_board_) {
    flexible_board_->SetToPlay(c);
  } else {
    fixed_board_->SetToPlay(c);
  }
}

int BoardAdapter::GridRadius() const {
  if (using_flexible_board_) {
    return flexible_board_->MaxRadius();
  } else {
    return fixed_board_->GridRadius();
  }
}

void BoardAdapter::ApplyMove(const Move& move) {
  if (using_flexible_board_) {
    flexible_board_->ApplyMove(move);
  } else {
    fixed_board_->ApplyMove(move);
  }
}

// Game state operations
std::string BoardAdapter::ToMBN() const {
  if (using_flexible_board_) {
    return flexible_board_->ToMBN();
  } else {
    return fixed_board_->ToMBN();
  }
}

std::string BoardAdapter::DebugString() const {
  if (using_flexible_board_) {
    return flexible_board_->DebugString();
  } else {
    return fixed_board_->DebugString();
  }
}

void BoardAdapter::GenerateLegalMoves(const MoveYieldFn& yield) const {
  if (using_flexible_board_) {
    flexible_board_->GenerateLegalMoves(yield);
  } else {
    fixed_board_->GenerateLegalMoves(yield);
  }
}

void BoardAdapter::GenerateLegalMoves(const MoveYieldFn& yield, Color color) const {
  if (using_flexible_board_) {
    flexible_board_->GenerateLegalMoves(yield, color);
  } else {
    fixed_board_->GenerateLegalMoves(yield, color);
  }
}

const std::vector<City>& BoardAdapter::GetCities() const {
  if (using_flexible_board_) {
    return flexible_board_->GetCities();
  } else {
    return fixed_board_->GetCities();
  }
}

absl::optional<const City*> BoardAdapter::CityAt(const HexCoord& hex) const {
  if (using_flexible_board_) {
    return flexible_board_->CityAt(hex);
  } else {
    return fixed_board_->CityAt(hex);
  }
}

int BoardAdapter::MeeplesCount(const HexCoord& hex) const {
  if (using_flexible_board_) {
    return flexible_board_->MeeplesCount(hex);
  } else {
    return fixed_board_->MeeplesCount(hex);
  }
}

uint64_t BoardAdapter::HashValue() const {
  if (using_flexible_board_) {
    return flexible_board_->HashValue();
  } else {
    return fixed_board_->HashValue();
  }
}

int BoardAdapter::CountConnectedPosts(Color color) const {
  if (using_flexible_board_) {
    return flexible_board_->CountConnectedPosts(color);
  } else {
    return fixed_board_->CountConnectedPosts(color);
  }
}

// Advanced flexible board operations
std::vector<HexCoord> BoardAdapter::GetValidNeighbors(const HexCoord& hex) const {
  if (using_flexible_board_) {
    return flexible_board_->GetValidNeighbors(hex);
  } else {
    // Convert from fixed board - get all neighbors that are in the board area
    std::vector<HexCoord> neighbors;
    for (const auto& dir : kHexDirections) {
      HexCoord neighbor = hex + dir;
      if (fixed_board_->InBoardArea(neighbor)) {
        neighbors.push_back(neighbor);
      }
    }
    return neighbors;
  }
}

bool BoardAdapter::AreConnected(const HexCoord& a, const HexCoord& b) const {
  if (using_flexible_board_) {
    return flexible_board_->AreConnected(a, b);
  } else {
    // In the fixed board, all hexes within the board area are connected
    return fixed_board_->InBoardArea(a) && fixed_board_->InBoardArea(b);
  }
}

std::vector<std::set<HexCoord>> BoardAdapter::GetConnectedComponents() const {
  if (using_flexible_board_) {
    return flexible_board_->GetConnectedComponents();
  } else {
    // In the fixed board, there's only one connected component - all valid hexes
    std::vector<std::set<HexCoord>> components;
    std::set<HexCoord> all_hexes;
    
    // Add all hexes within the grid radius
    for (int x = -fixed_board_->GridRadius(); x <= fixed_board_->GridRadius(); ++x) {
      for (int y = -fixed_board_->GridRadius(); y <= fixed_board_->GridRadius(); ++y) {
        int z = -x - y;
        // Skip invalid hex coordinates
        if (std::abs(x) + std::abs(y) + std::abs(z) > 2 * fixed_board_->GridRadius()) continue;
        
        HexCoord hex(x, y, z);
        if (fixed_board_->InBoardArea(hex)) {
          all_hexes.insert(hex);
        }
      }
    }
    
    components.push_back(all_hexes);
    return components;
  }
}

// Access to the underlying board implementations
const Mali_BaBoard* BoardAdapter::GetFixedBoard() const {
  return fixed_board_.get();
}

const FlexBoard* BoardAdapter::GetFlexBoard() const {
  return flexible_board_.get();
}

}  // namespace mali_ba
}  // namespace open_spiel