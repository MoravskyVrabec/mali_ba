#include "open_spiel/games/mali_ba/board_adapter.h"

// Include full implementations after the forward declarations are resolved
#include "open_spiel/games/mali_ba/mali_ba_board.h"
#include "open_spiel/games/mali_ba/flex_board.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"

#include <memory>
#include <string>
#include <vector>
#include <iostream>

namespace open_spiel {
namespace mali_ba {

// Implementation class definition
class BoardAdapter::Impl {
 public:
  Impl() : using_flexible_board_(true),
          fixed_board_(nullptr),
          flexible_board_(std::make_unique<FlexBoard>(FlexBoard::CreateSimplifiedBoard())) {}
  
  explicit Impl(const Mali_BaBoard& board) 
      : using_flexible_board_(false),
        fixed_board_(std::make_unique<Mali_BaBoard>(board)),
        flexible_board_(nullptr) {}
  
  explicit Impl(const FlexBoard& board)
      : using_flexible_board_(true),
        fixed_board_(nullptr),
        flexible_board_(std::make_unique<FlexBoard>(board)) {}
  
  Impl(const Impl& other) : using_flexible_board_(other.using_flexible_board_) {
    if (using_flexible_board_) {
      if (other.flexible_board_) {
        flexible_board_ = std::make_unique<FlexBoard>(*other.flexible_board_);
      }
    } else {
      if (other.fixed_board_) {
        fixed_board_ = std::make_unique<Mali_BaBoard>(*other.fixed_board_);
      }
    }
  }
  
  bool using_flexible_board_;
  std::unique_ptr<Mali_BaBoard> fixed_board_;
  std::unique_ptr<FlexBoard> flexible_board_;
};

// Default constructor
BoardAdapter::BoardAdapter() : pimpl_(std::make_unique<Impl>()) {}

// Create an adapter for the original fixed board
BoardAdapter::BoardAdapter(const Mali_BaBoard& board)
    : pimpl_(std::make_unique<Impl>(board)) {}

// Create an adapter for the new flexible board
BoardAdapter::BoardAdapter(const FlexBoard& board)
    : pimpl_(std::make_unique<Impl>(board)) {}

// Copy constructor
BoardAdapter::BoardAdapter(const BoardAdapter& other)
    : pimpl_(std::make_unique<Impl>(*other.pimpl_)) {}

// Assignment operator
BoardAdapter& BoardAdapter::operator=(const BoardAdapter& other) {
  if (this != &other) {
    pimpl_ = std::make_unique<Impl>(*other.pimpl_);
  }
  return *this;
}

// Destructor
BoardAdapter::~BoardAdapter() = default;

// Constructor for a board from a board notation string
BoardAdapter BoardAdapter::FromMBN(const std::string& mbn, bool use_flexible_board) {
  if (mbn.empty()) {
    // Handle empty string gracefully by creating default board
    return CreateSimplifiedBoard(use_flexible_board);
  }
  
  if (use_flexible_board) {
    // Parse the MBN format for the flexible board
    std::vector<std::string> parts = absl::StrSplit(mbn, '/');
    if (parts.size() < 1) {
      return CreateSimplifiedBoard(true);
    }
    
    // Create a default board to start with
    auto board = FlexBoard::CreateSimplifiedBoard();
    
    // Set the current player based on the first part of the MBN
    if (parts[0] == "b") {
      board.SetToPlay(Color::kBlack);
    } else if (parts[0] == "w") {
      board.SetToPlay(Color::kWhite);
    } else if (parts[0] == "r") {
      board.SetToPlay(Color::kRed);
    } else if (parts[0] == "u") {
      board.SetToPlay(Color::kBlue);
    }
    
    // Set the move number if available
    if (parts.size() > 1) {
      try {
        int move_number = std::stoi(parts[1]);
        // Note: Would need to implement a method to set move number directly
        // For now, this is handled implicitly by the number of moves applied
      } catch (...) {
        // Ignore invalid move number
      }
    }
    
    // Add parsing for the rest of the MBN format here if needed
    
    return BoardAdapter(board);
  } else {
    // Use the existing Mali_BaBoard::BoardFromMBN function
    auto board_opt = Mali_BaBoard::BoardFromMBN(mbn);
    if (board_opt.has_value()) {
      return BoardAdapter(*board_opt);
    } else {
      // Handle invalid MBN gracefully
      std::cerr << "Error parsing MBN for fixed board, using default board" << std::endl;
      return BoardAdapter(Mali_BaBoard(Mali_BaBoard::DefaultGridRadius()));
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
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->GetPiece(hex);
  } else {
    return pimpl_->fixed_board_->at(hex);
  }
}

void BoardAdapter::SetPiece(const HexCoord& hex, const Piece& piece) {
  if (pimpl_->using_flexible_board_) {
    pimpl_->flexible_board_->SetPiece(hex, piece);
  } else {
    pimpl_->fixed_board_->set_piece(hex, piece);
  }
}

TradePost BoardAdapter::GetTradePost(const HexCoord& hex) const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->GetTradePost(hex);
  } else {
    return pimpl_->fixed_board_->post_at(hex);
  }
}

void BoardAdapter::SetTradePost(const HexCoord& hex, const TradePost& post) {
  if (pimpl_->using_flexible_board_) {
    pimpl_->flexible_board_->SetTradePost(hex, post);
  } else {
    pimpl_->fixed_board_->set_post(hex, post);
  }
}

bool BoardAdapter::IsValidHex(const HexCoord& hex) const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->IsValidHex(hex);
  } else {
    return pimpl_->fixed_board_->InBoardArea(hex);
  }
}

Color BoardAdapter::ToPlay() const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->ToPlay();
  } else {
    return pimpl_->fixed_board_->ToPlay();
  }
}

void BoardAdapter::SetToPlay(Color c) {
  if (pimpl_->using_flexible_board_) {
    pimpl_->flexible_board_->SetToPlay(c);
  } else {
    pimpl_->fixed_board_->SetToPlay(c);
  }
}

int BoardAdapter::GridRadius() const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->MaxRadius();
  } else {
    return pimpl_->fixed_board_->GridRadius();
  }
}

void BoardAdapter::ApplyMove(const Move& move) {
  if (pimpl_->using_flexible_board_) {
    pimpl_->flexible_board_->ApplyMove(move);
  } else {
    pimpl_->fixed_board_->ApplyMove(move);
  }
}

bool BoardAdapter::IsUsingFlexBoard() const {
  return pimpl_->using_flexible_board_;
}

// Game state operations
std::string BoardAdapter::ToMBN() const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->ToMBN();
  } else {
    return pimpl_->fixed_board_->ToMBN();
  }
}

std::string BoardAdapter::DebugString() const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->DebugString();
  } else {
    return pimpl_->fixed_board_->DebugString();
  }
}

void BoardAdapter::GenerateLegalMoves(const MoveYieldFn& yield) const {
  if (pimpl_->using_flexible_board_) {
    pimpl_->flexible_board_->GenerateLegalMoves(yield);
  } else {
    pimpl_->fixed_board_->GenerateLegalMoves(yield);
  }
}

void BoardAdapter::GenerateLegalMoves(const MoveYieldFn& yield, Color color) const {
  if (pimpl_->using_flexible_board_) {
    pimpl_->flexible_board_->GenerateLegalMoves(yield, color);
  } else {
    pimpl_->fixed_board_->GenerateLegalMoves(yield, color);
  }
}

const std::vector<City>& BoardAdapter::GetCities() const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->GetCities();
  } else {
    return pimpl_->fixed_board_->GetCities();
  }
}

absl::optional<const City*> BoardAdapter::CityAt(const HexCoord& hex) const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->CityAt(hex);
  } else {
    return pimpl_->fixed_board_->CityAt(hex);
  }
}

int BoardAdapter::MeeplesCount(const HexCoord& hex) const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->MeeplesCount(hex);
  } else {
    return pimpl_->fixed_board_->MeeplesCount(hex);
  }
}

uint64_t BoardAdapter::HashValue() const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->HashValue();
  } else {
    return pimpl_->fixed_board_->HashValue();
  }
}

int BoardAdapter::CountConnectedPosts(Color color) const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->CountConnectedPosts(color);
  } else {
    return pimpl_->fixed_board_->CountConnectedPosts(color);
  }
}

// Advanced flexible board operations
std::vector<HexCoord> BoardAdapter::GetValidNeighbors(const HexCoord& hex) const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->GetValidNeighbors(hex);
  } else {
    // Convert from fixed board - get all neighbors that are in the board area
    std::vector<HexCoord> neighbors;
    for (const auto& dir : kHexDirections) {
      HexCoord neighbor = hex + dir;
      if (pimpl_->fixed_board_->InBoardArea(neighbor)) {
        neighbors.push_back(neighbor);
      }
    }
    return neighbors;
  }
}

bool BoardAdapter::AreConnected(const HexCoord& a, const HexCoord& b) const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->AreConnected(a, b);
  } else {
    // In the fixed board, all hexes within the board area are connected
    return pimpl_->fixed_board_->InBoardArea(a) && pimpl_->fixed_board_->InBoardArea(b);
  }
}

std::vector<std::set<HexCoord>> BoardAdapter::GetConnectedComponents() const {
  if (pimpl_->using_flexible_board_) {
    return pimpl_->flexible_board_->GetConnectedComponents();
  } else {
    // In the fixed board, there's only one connected component - all valid hexes
    std::vector<std::set<HexCoord>> components;
    std::set<HexCoord> all_hexes;
    
    // Add all hexes within the grid radius
    for (int x = -pimpl_->fixed_board_->GridRadius(); x <= pimpl_->fixed_board_->GridRadius(); ++x) {
      for (int y = -pimpl_->fixed_board_->GridRadius(); y <= pimpl_->fixed_board_->GridRadius(); ++y) {
        int z = -x - y;
        // Skip invalid hex coordinates
        if (std::abs(x) + std::abs(y) + std::abs(z) > 2 * pimpl_->fixed_board_->GridRadius()) continue;
        
        HexCoord hex(x, y, z);
        if (pimpl_->fixed_board_->InBoardArea(hex)) {
          all_hexes.insert(hex);
        }
      }
    }
    
    components.push_back(all_hexes);
    return components;
  }
}

}  // namespace mali_ba
}  // namespace open_spiel