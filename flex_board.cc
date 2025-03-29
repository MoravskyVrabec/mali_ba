#include "open_spiel/games/mali_ba/flex_board.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/games/mali_ba/mali_ba_board.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace mali_ba {

// Constructor for an empty board
FlexBoard::FlexBoard(int max_radius)
    : max_radius_(max_radius),
      to_play_(Color::kBlack),
      move_number_(1),
      zobrist_hash_(0) {
  // Default constructor creates an empty board
}

// Constructor with explicitly provided valid hexes
FlexBoard::FlexBoard(const std::set<HexCoord>& valid_hexes, int max_radius)
    : valid_hexes_(valid_hexes),
      max_radius_(max_radius),
      to_play_(Color::kBlack),
      move_number_(1),
      zobrist_hash_(0) {
  
  // Initialize with empty pieces for all valid hexes
  for (const auto& hex : valid_hexes_) {
    pieces_[hex] = kEmptyPiece;
    trade_posts_[hex] = {Color::kEmpty, TradePostType::kNone};
  }
  
  // Initialize cities and meeples
  InitializeCities();
  InitializeMeeples();
}

// Create a standard full board with given radius
FlexBoard FlexBoard::CreateStandardBoard(int radius) {
  std::set<HexCoord> valid_hexes;
  
  // Add all hexes within radius
  for (int x = -radius; x <= radius; ++x) {
    for (int y = -radius; y <= radius; ++y) {
      int z = -x - y;
      // Skip invalid hex coordinates (not in a circle of radius radius)
      if (std::abs(x) + std::abs(y) + std::abs(z) > 2 * radius) continue;
      
      valid_hexes.insert(HexCoord(x, y, z));
    }
  }
  
  return FlexBoard(valid_hexes, radius);
}

// In flexible_board.cc:
FlexBoard FlexBoard::CreateSimplifiedBoard() {
  #ifdef MALI_BA_SIMPLIFIED_BOARD
    // Use our custom simplified board layout
    return CreateSimplifiedCustomBoard();
  #else
    // Use a smaller radius for the simplified board
    int radius = 4;
    return CreateStandardBoard(radius);
  #endif
  }
  

// Create a custom board with specific hex structure
FlexBoard FlexBoard::CreateCustomBoard(const std::vector<HexCoord>& valid_hexes) {
  std::set<HexCoord> hex_set(valid_hexes.begin(), valid_hexes.end());
  
  // Calculate max radius from the provided hexes
  int max_radius = 0;
  for (const auto& hex : hex_set) {
    max_radius = std::max(max_radius, std::max(std::abs(hex.x), 
                           std::max(std::abs(hex.y), std::abs(hex.z))));
  }
  
  return FlexBoard(hex_set, max_radius);
}

// Add these methods to flex_board.cc

// Clear all cities from the board
void FlexBoard::ClearCities() {
  cities_.clear();
}

// Add a city to the board
void FlexBoard::AddCity(const std::string& name, Region region, 
                       const std::string& common_good, const std::string& rare_good,
                       const HexCoord& location) {
  cities_.emplace_back(name, region, common_good, rare_good, location);
  
  // Ensure the city's location is a valid hex
  if (!IsValidHex(location)) {
    AddHex(location);
  }
}

// Implementation for CreateSimplifiedCustomBoard
FlexBoard FlexBoard::CreateSimplifiedCustomBoard() {
  // Create a board with the specific layout requested
  std::vector<HexCoord> valid_hexes;
  
  // Add all hexes to create the board shape shown in the example
  // Row 1
  valid_hexes.push_back(HexCoord(-5, 2, 3));  // C1: Warri
  valid_hexes.push_back(HexCoord(-4, 1, 3));
  valid_hexes.push_back(HexCoord(-3, 0, 3));
  valid_hexes.push_back(HexCoord(-2, -1, 3));
  valid_hexes.push_back(HexCoord(-1, -2, 3));
  valid_hexes.push_back(HexCoord(0, -3, 3));
  
  // Row 2
  valid_hexes.push_back(HexCoord(-4, 2, 2));
  valid_hexes.push_back(HexCoord(-3, 1, 2));
  valid_hexes.push_back(HexCoord(-2, 0, 2));
  valid_hexes.push_back(HexCoord(-1, -1, 2));
  valid_hexes.push_back(HexCoord(0, -2, 2));
  valid_hexes.push_back(HexCoord(1, -3, 2));
  
  // Row 3
  valid_hexes.push_back(HexCoord(-5, 3, 2));
  valid_hexes.push_back(HexCoord(-4, 2, 2));
  valid_hexes.push_back(HexCoord(-3, 1, 2));
  valid_hexes.push_back(HexCoord(-2, 0, 2));
  valid_hexes.push_back(HexCoord(-1, -1, 2));
  valid_hexes.push_back(HexCoord(0, -2, 2));  // C2: Dosso
  
  // Row 4
  valid_hexes.push_back(HexCoord(-4, 3, 1));
  valid_hexes.push_back(HexCoord(-3, 2, 1));
  valid_hexes.push_back(HexCoord(-2, 1, 1));
  valid_hexes.push_back(HexCoord(-1, 0, 1));
  valid_hexes.push_back(HexCoord(0, -1, 1));
  valid_hexes.push_back(HexCoord(1, -2, 1));
  
  // Row 5
  valid_hexes.push_back(HexCoord(-5, 4, 1));  // C3: Tabou
  valid_hexes.push_back(HexCoord(-4, 3, 1));
  valid_hexes.push_back(HexCoord(-3, 2, 1));
  valid_hexes.push_back(HexCoord(-2, 1, 1));
  valid_hexes.push_back(HexCoord(-1, 0, 1));  // C4: Oyo
  valid_hexes.push_back(HexCoord(0, -1, 1));
  
  // Row 6
  valid_hexes.push_back(HexCoord(-4, 4, 0));
  valid_hexes.push_back(HexCoord(-3, 3, 0));
  valid_hexes.push_back(HexCoord(-2, 2, 0));
  valid_hexes.push_back(HexCoord(-1, 1, 0));
  valid_hexes.push_back(HexCoord(0, 0, 0));
  valid_hexes.push_back(HexCoord(1, -1, 0));
  
  // Create the board with these hexes
  auto board = FlexBoard::CreateCustomBoard(valid_hexes);
  
  // We'll override the city initialization since we want specific city placements
  board.ClearCities();
  
  // Add the cities at the specified locations
  board.AddCity("Warri", Region::kIdjo, "Palm Oil", "Coral Necklace", HexCoord(-5, 2, 3));
  board.AddCity("Dosso", Region::kSonghai, "Cotton", "Silver Headdress", HexCoord(0, -2, 2));
  board.AddCity("Tabou", Region::kKru, "Pepper", "Canoe", HexCoord(-5, 4, 1));
  board.AddCity("Oyo", Region::kYoruba, "Ivory", "Ivory Bracelet", HexCoord(-1, 0, 1));
  
  return board;
}

// Add a valid hex to the board
void FlexBoard::AddHex(const HexCoord& hex) {
  if (valid_hexes_.insert(hex).second) {
    // Only initialize if this was a new hex
    pieces_[hex] = kEmptyPiece;
    trade_posts_[hex] = {Color::kEmpty, TradePostType::kNone};
  }
}

// Remove a hex from the board, making it invalid
void FlexBoard::RemoveHex(const HexCoord& hex) {
  valid_hexes_.erase(hex);
  pieces_.erase(hex);
  trade_posts_.erase(hex);
}

// Check if a hex is valid on the board
bool FlexBoard::IsValidHex(const HexCoord& hex) const {
  return valid_hexes_.find(hex) != valid_hexes_.end();
}

// Get piece at a specific hex
Piece FlexBoard::GetPiece(const HexCoord& hex) const {
  auto it = pieces_.find(hex);
  if (it != pieces_.end()) {
    return it->second;
  }
  return kEmptyPiece;
}

// Set piece at a specific hex
void FlexBoard::SetPiece(const HexCoord& hex, const Piece& piece) {
  if (IsValidHex(hex)) {
    pieces_[hex] = piece;
    
    // Update Zobrist hash (implementation would go here)
  }
}

// Get trading post info at a specific hex
TradePost FlexBoard::GetTradePost(const HexCoord& hex) const {
  static const TradePost kEmptyPost = {Color::kEmpty, TradePostType::kNone};
  
  auto it = trade_posts_.find(hex);
  if (it != trade_posts_.end()) {
    return it->second;
  }
  return kEmptyPost;
}

// Set trading post at a specific hex
void FlexBoard::SetTradePost(const HexCoord& hex, const TradePost& post) {
  if (IsValidHex(hex)) {
    trade_posts_[hex] = post;
    
    // Update Zobrist hash (implementation would go here)
  }
}

// Get current player's color
Color FlexBoard::ToPlay() const {
  return to_play_;
}

// Set current player
void FlexBoard::SetToPlay(Color c) {
  to_play_ = c;
  // Update Zobrist hash (implementation would go here)
}

// Get board maximum radius
int FlexBoard::MaxRadius() const {
  return max_radius_;
}

// Get all valid hexes on the board
const std::set<HexCoord>& FlexBoard::GetValidHexes() const {
  return valid_hexes_;
}

// Get valid neighbors for a hex
std::vector<HexCoord> FlexBoard::GetValidNeighbors(const HexCoord& hex) const {
  std::vector<HexCoord> valid_neighbors;
  
  for (const auto& dir : kHexDirections) {
    HexCoord neighbor = hex + dir;
    if (IsValidHex(neighbor)) {
      valid_neighbors.push_back(neighbor);
    }
  }
  
  return valid_neighbors;
}

// Apply a move to the board
void FlexBoard::ApplyMove(const Move& move) {
  if (move.is_pass()) {
    // Pass move - just change the player
    SetToPlay(OppColor(to_play_));
    if (to_play_ == Color::kBlack) {
      ++move_number_;  // Increment move number after a full round
    }
    return;
  }
  
  // Check that the move is valid
  SPIEL_CHECK_TRUE(IsValidHex(move.start_hex));
  SPIEL_CHECK_GE(move.path.size(), 0);
  
  // Get number of meeples at the starting hex
  int num_meeples = MeeplesCount(move.start_hex);
  
  // Check we have the right number of meeples for this path
  SPIEL_CHECK_EQ(num_meeples, move.path.size());
  
  // Distribute meeples along the path
  Piece starting_piece = GetPiece(move.start_hex);
  SetPiece(move.start_hex, kEmptyPiece); // Remove meeples from starting hex
  
  // Place one meeple on each hex in the path
  for (size_t i = 0; i < move.path.size(); ++i) {
    SPIEL_CHECK_TRUE(IsValidHex(move.path[i]));
    
    // Place a meeple on this hex - set color to current player
    SetPiece(move.path[i], {to_play_, starting_piece.type});
  }
  
  // Place a trading post if requested and legal
  if (move.place_trading_post && move.path.size() > 0) {
    const HexCoord& last_hex = move.path.back();
    if (CanPlaceTradingPost(last_hex, to_play_)) {
      SetTradePost(last_hex, {to_play_, TradePostType::kPost});
    }
  }
  
  // Change player
  SetToPlay(OppColor(to_play_));
  
  // Increment move number after full round
  if (to_play_ == Color::kBlack) {
    ++move_number_;
  }
}

// Generate all legal moves for the current player
void FlexBoard::GenerateLegalMoves(const MoveYieldFn& yield) const {
  GenerateLegalMoves(yield, to_play_);
}

void FlexBoard::GenerateLegalMoves(const MoveYieldFn& yield, Color color) const {
  bool generating = true;
  
  // Always provide option to pass
  if (!yield(kPassMove)) {
    return;
  }
  
  // Iterate through all valid hexes
  for (const auto& hex : valid_hexes_) {
    if (!generating) break;
    
    // Check if this hex has our meeples or neutral meeples we can use
    const Piece& piece = GetPiece(hex);
    if (piece.type != MeepleType::kEmpty && 
        (piece.color == color || piece.color == Color::kEmpty)) {
      
      // Get meeple count at this location
      int meeple_count = MeeplesCount(hex);
      
      // Skip if no meeples
      if (meeple_count <= 0) continue;
      
      // Generate all possible paths starting from this hex
      std::vector<HexCoord> current_path;
      std::set<HexCoord> visited;
      visited.insert(hex);  // Mark start as visited
      
      // Generate paths of exactly meeple_count length
      GeneratePaths(hex, current_path, visited, meeple_count, 
                    [&](const Move& move) {
                      return yield(move) && (generating = true);
                    }, color);
    }
  }
}

// Helper method to recursively generate all valid paths
void FlexBoard::GeneratePaths(
    const HexCoord& start,
    std::vector<HexCoord>& current_path,
    std::set<HexCoord>& visited,
    int remaining_meeples,
    const MoveYieldFn& yield,
    Color color) const {
  
  // Base case: if we've used all meeples, we have a valid path
  if (remaining_meeples == 0) {
    // Create a move object
    Move move = {color, start, current_path, true};
    
    // Yield this move
    if (!yield(move)) {
      return;  // Stop generating if requested
    }
    
    // Also yield the same move without trading post placement
    move.place_trading_post = false;
    yield(move);
    
    return;
  }
  
  // Get the last location in our path (or start if path is empty)
  HexCoord current = current_path.empty() ? start : current_path.back();
  
  // Try each possible neighbor
  auto neighbors = GetValidNeighbors(current);
  for (const auto& next : neighbors) {
    // Skip already visited hexes
    if (visited.find(next) != visited.end()) continue;
    
    // Mark as visited
    visited.insert(next);
    
    // Add to path
    current_path.push_back(next);
    
    // Recurse
    GeneratePaths(start, current_path, visited, remaining_meeples - 1, 
                 yield, color);
    
    // Backtrack
    current_path.pop_back();
    visited.erase(next);
  }
}

// Helper method to check if a trading post can be placed
bool FlexBoard::CanPlaceTradingPost(const HexCoord& hex, Color color) const {
  // Rules for trading post placement:
  // 1. Hex must be emptied of meeples by the current move
  // 2. Cannot place where there's already a post of same player
  // 3. Number of trading centers limited based on number of players
  
  // First, check if this hex is empty of meeples
  if (GetPiece(hex).type != MeepleType::kEmpty) return false;
  
  // Don't place on a city
  if (CityAt(hex).has_value()) return false;
  
  // Check if there's already a trading post or center here
  const TradePost& existing_post = GetTradePost(hex);
  if (existing_post.type != TradePostType::kNone && 
      existing_post.owner == color) {
    return false;  // Already have a post here
  }
  
  // Count existing trading centers at this location
  int center_count = 0;
  if (existing_post.type == TradePostType::kCenter) {
    center_count = 1;
  }
  
  // Allow placement if we don't exceed limit
  int player_count = 4;  // Adjust based on actual player count
  return center_count < (player_count - 1);
}

// Get meeples count at a specific hex
int FlexBoard::MeeplesCount(const HexCoord& hex) const {
  // In this implementation, we're tracking a single piece per hex location
  return GetPiece(hex).type != MeepleType::kEmpty ? 1 : 0;
}

// Find a city at a given location
absl::optional<const City*> FlexBoard::CityAt(const HexCoord& hex) const {
  for (const auto& city : cities_) {
    if (city.location == hex) {
      return &city;
    }
  }
  return absl::nullopt;
}

// Get cities on the board
const std::vector<City>& FlexBoard::GetCities() const {
  return cities_;
}

// Initialize cities (placeholder that would be properly implemented)
void FlexBoard::InitializeCities() {
#ifdef MALI_BA_SIMPLIFIED_BOARD
  // Simplified board with just 4 cities and 4 regions
  cities_.emplace_back("Warri", Region::kIdjo, "Palm Oil", "Coral Necklace", HexCoord(-3, -1, 4));
  cities_.emplace_back("Dosso", Region::kSonghai, "Cotton", "Silver Headdress", HexCoord(3, 0, -3));
  cities_.emplace_back("Tabou", Region::kKru, "Pepper", "Canoe", HexCoord(-3, -2, 5));
  cities_.emplace_back("Oyo", Region::kYoruba, "Ivory", "Ivory Bracelet", HexCoord(2, -4, 2));
#else
  // Full board with all cities and regions (implement fully as needed)
  cities_.emplace_back("Timbuktu", Region::kSonghai, "Cotton", "Silver Headdress", HexCoord(0, 0, 0));
  // Add more cities here...
#endif
  
  // Ensure all city locations are valid hexes
  for (const auto& city : cities_) {
    if (!IsValidHex(city.location)) {
      // Add the hex if it's not already valid
      const_cast<FlexBoard*>(this)->AddHex(city.location);
    }
  }
}

// Initialize meeples (placeholder that would be properly implemented)
void FlexBoard::InitializeMeeples() {
  // Random distribution of meeples across the board
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  
  // Define possible meeple types for distribution
  std::vector<MeepleType> meeple_types = {
    MeepleType::kTrader, MeepleType::kGold, MeepleType::kSpice,
    MeepleType::kSalt, MeepleType::kIvory, MeepleType::kCotton
  };
  
#ifdef MALI_BA_SIMPLIFIED_BOARD
  // For simplified board, use a more deterministic placement
  // (implementation would duplicate what's in the original code)
#else
  // For the full board, use random distribution
  // (implementation would duplicate what's in the original code)
#endif
}

// Debug string representation
std::string FlexBoard::DebugString() const {
  std::stringstream ss;
  
  ss << "Mali-Ba Flex Board (Max Radius: " << max_radius_ << ")\n";
  ss << "Current Player: " << ColorToString(to_play_) << "\n";
  ss << "Move Number: " << move_number_ << "\n";
  ss << "Valid Hexes: " << valid_hexes_.size() << "\n\n";
  
  // The hexes need to be sorted in some order to produce a readable output
  std::vector<std::pair<int, int>> row_col_pairs;
  for (const auto& hex : valid_hexes_) {
    auto [col, row] = CubeToOffset(hex);
    row_col_pairs.push_back({row, col});
  }
  
  // Sort by row, then column
  std::sort(row_col_pairs.begin(), row_col_pairs.end());
  
  // Find min/max rows and columns
  int min_row = std::numeric_limits<int>::max();
  int max_row = std::numeric_limits<int>::min();
  int min_col = std::numeric_limits<int>::max();
  int max_col = std::numeric_limits<int>::min();
  
  for (const auto& [row, col] : row_col_pairs) {
    min_row = std::min(min_row, row);
    max_row = std::max(max_row, row);
    min_col = std::min(min_col, col);
    max_col = std::max(max_col, col);
  }
  
  // Display the board in a hex grid layout
  // This is complex to display properly in text, but we'll do a simple version
  
  // Loop through rows
  for (int row = min_row; row <= max_row; ++row) {
    // Indent based on row (this would need to be adjusted based on the display style)
    for (int i = 0; i < row - min_row; ++i) {
      ss << "  ";
    }
    
    // Loop through columns
    for (int col = min_col; col <= max_col; ++col) {
      // Convert to cube coordinates
      HexCoord hex = OffsetToCube(col, row);
      
      // Skip if not a valid hex
      if (!IsValidHex(hex)) {
        ss << "     ";  // Space for invalid hex
        continue;
      }
      
      // Display piece and trading post
      const Piece& piece = GetPiece(hex);
      const TradePost& post = GetTradePost(hex);
      
      // Check if this is a city
      bool is_city = false;
      for (const auto& city : cities_) {
        if (city.location == hex) {
          is_city = true;
          ss << "[C] ";
          break;
        }
      }
      
      if (!is_city) {
        ss << "[" << piece.ToString();
        
        // Add trading post indicator
        if (post.type != TradePostType::kNone) {
          ss << (post.type == TradePostType::kPost ? "p" : "c");
        } else {
          ss << " ";
        }
        
        ss << "] ";
      }
    }
    
    ss << "\n";
  }
  
  // List cities
  ss << "\nCities:\n";
  for (const auto& city : cities_) {
    ss << "  " << city.name << " (" << RegionToString(city.region) << ")"
       << " at " << city.location.ToString() << "\n";
    ss << "    Common Good: " << city.common_good << "\n";
    ss << "    Rare Good: " << city.rare_good << "\n";
  }
  
  return ss.str();
}

// Convert board to MBN string representation
std::string FlexBoard::ToMBN() const {
  // A simple MBN format could look like:
  // <current_player>/<move_number>/<valid_hexes>/<board_pieces>/<trading_posts>
  
  std::stringstream ss;
  
  // Current player
  switch (to_play_) {
    case Color::kBlack: ss << "b"; break;
    case Color::kWhite: ss << "w"; break;
    case Color::kRed: ss << "r"; break;
    case Color::kBlue: ss << "u"; break;
    default: ss << "?";
  }
  
  ss << "/";
  
  // Move number
  ss << move_number_;
  
  ss << "/";
  
  // Valid hexes (new section specific to flexible board)
  std::vector<std::string> hex_strs;
  for (const auto& hex : valid_hexes_) {
    hex_strs.push_back(std::to_string(hex.x) + "," + 
                        std::to_string(hex.y) + "," + 
                        std::to_string(hex.z));
  }
  ss << absl::StrJoin(hex_strs, ";");
  
  ss << "/";
  
  // Board pieces (compact encoding)
  for (const auto& hex : valid_hexes_) {
    const Piece& piece = GetPiece(hex);
    if (piece.type != MeepleType::kEmpty) {
      ss << hex.x << "," << hex.y << "," << hex.z << ":";
      ss << piece.ToString() << ";";
    }
  }
  
  ss << "/";
  
  // Trading posts (compact encoding)
  for (const auto& hex : valid_hexes_) {
    const TradePost& post = GetTradePost(hex);
    if (post.type != TradePostType::kNone) {
      ss << hex.x << "," << hex.y << "," << hex.z << ":";
      
      // Owner
      switch (post.owner) {
        case Color::kBlack: ss << "b"; break;
        case Color::kWhite: ss << "w"; break;
        case Color::kRed: ss << "r"; break;
        case Color::kBlue: ss << "u"; break;
        default: ss << "?";
      }
      
      // Type
      ss << (post.type == TradePostType::kPost ? "p" : "c");
      
      ss << ";";
    }
  }
  
  return ss.str();
}

// Get hash value (placeholder)
uint64_t FlexBoard::HashValue() const {
  return zobrist_hash_;
}

// Check if two hexes are connected by a valid path
bool FlexBoard::AreConnected(const HexCoord& a, const HexCoord& b) const {
  if (!IsValidHex(a) || !IsValidHex(b)) return false;
  if (a == b) return true;
  
  // Use breadth-first search to find a path
  std::queue<HexCoord> queue;
  std::set<HexCoord> visited;
  
  queue.push(a);
  visited.insert(a);
  
  while (!queue.empty()) {
    HexCoord current = queue.front();
    queue.pop();
    
    // If we've reached the destination, we're done
    if (current == b) return true;
    
    // Add all unvisited valid neighbors to the queue
    for (const auto& neighbor : GetValidNeighbors(current)) {
      if (visited.find(neighbor) == visited.end()) {
        visited.insert(neighbor);
        queue.push(neighbor);
      }
    }
  }
  
  // If we've exhausted all reachable hexes without finding b, they're not connected
  return false;
}

// Find all path-connected components on the board
std::vector<std::set<HexCoord>> FlexBoard::GetConnectedComponents() const {
  std::vector<std::set<HexCoord>> components;
  std::set<HexCoord> unvisited = valid_hexes_; // Copy all valid hexes
  
  while (!unvisited.empty()) {
    // Start a new component with the first unvisited hex
    HexCoord start = *unvisited.begin();
    
    // Use BFS to find all connected hexes
    std::queue<HexCoord> queue;
    std::set<HexCoord> component;
    
    queue.push(start);
    component.insert(start);
    unvisited.erase(start);
    
    while (!queue.empty()) {
      HexCoord current = queue.front();
      queue.pop();
      
      // Add all unvisited valid neighbors to the queue
      for (const auto& neighbor : GetValidNeighbors(current)) {
        if (unvisited.find(neighbor) != unvisited.end()) {
          queue.push(neighbor);
          component.insert(neighbor);
          unvisited.erase(neighbor);
        }
      }
    }
    
    // Add the completed component to our list
    components.push_back(component);
  }
  
  return components;
}

// Count number of connected trading posts for a player
int FlexBoard::CountConnectedPosts(Color color) const {
  // Find all hexes with trading posts/centers owned by this player
  std::set<HexCoord> post_hexes;
  
  for (const auto& hex : valid_hexes_) {
    const auto& post = GetTradePost(hex);
    if (post.type != TradePostType::kNone && post.owner == color) {
      post_hexes.insert(hex);
    }
  }
  
  // No posts? Return 0
  if (post_hexes.empty()) return 0;
  
  // Find connected components among these post hexes
  std::vector<std::set<HexCoord>> components;
  std::set<HexCoord> unvisited = post_hexes;
  
  while (!unvisited.empty()) {
    // Start a new component with the first unvisited hex
    HexCoord start = *unvisited.begin();
    
    // Use BFS to find all connected hexes
    std::queue<HexCoord> queue;
    std::set<HexCoord> component;
    
    queue.push(start);
    component.insert(start);
    unvisited.erase(start);
    
    while (!queue.empty()) {
      HexCoord current = queue.front();
      queue.pop();
      
      // Find all valid neighbors that have posts and are unvisited
      for (const auto& neighbor : GetValidNeighbors(current)) {
        if (unvisited.find(neighbor) != unvisited.end()) {
          queue.push(neighbor);
          component.insert(neighbor);
          unvisited.erase(neighbor);
        }
      }
    }
    
    // Add the completed component to our list
    components.push_back(component);
  }
  
  // Return the number of connected components
  return components.size();
}