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

#include "open_spiel/games/mali_ba/mali_ba_board.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace mali_ba {

// Define the static pass move constant
const Move kPassMove = {Color::kEmpty, kInvalidHex, {}, false};

std::string RegionToString(Region r) {
  switch (r) {
    case Region::kTuareg: return "Tuareg";
    case Region::kDogon: return "Dogon";
    case Region::kFulani: return "Fulani";
    case Region::kSonghai: return "Songhai";
    case Region::kAkan: return "Akan";
    case Region::kHousa: return "Housa";
    case Region::kWolof: return "Wolof";
    case Region::kDagbani: return "Dagbani";
    case Region::kArab: return "Arab";
    case Region::kYoruba: return "Yoruba";
    case Region::kMande: return "Mande";
    case Region::kSenoufo: return "Senoufo";
    case Region::kKru: return "Kru";
    case Region::kIdjo: return "Idjo";
    case Region::kEmpty: return "Empty";
    default: SpielFatalError("Unknown region");
  }
}

std::string MeepleTypeToString(MeepleType t) {
  switch (t) {
    case MeepleType::kEmpty: return "Empty";
    case MeepleType::kTrader: return "Trader";
    case MeepleType::kGold: return "Gold";
    case MeepleType::kSpice: return "Spice";
    case MeepleType::kSalt: return "Salt";
    case MeepleType::kIvory: return "Ivory";
    case MeepleType::kCotton: return "Cotton";
    default: SpielFatalError("Unknown meeple type");
  }
}

std::string Piece::ToString() const {
  if (type == MeepleType::kEmpty) return " ";
  
  std::string result;
  switch (color) {
    case Color::kBlack:
      result = "B";
      break;
    case Color::kWhite:
      result = "W";
      break; 
    case Color::kRed:
      result = "R";
      break;
    case Color::kBlue:
      result = "U";  // U for blue to avoid confusion with black
      break;
    case Color::kEmpty:
      result = " ";
      break;
    default:
      SpielFatalError("Unknown color");
  }
  
  switch (type) {
    case MeepleType::kTrader:
      result += "T";
      break;
    case MeepleType::kGold:
      result += "G";
      break;
    case MeepleType::kSpice:
      result += "S";
      break;
    case MeepleType::kSalt:
      result += "A";  // A for salt
      break;
    case MeepleType::kIvory:
      result += "I";
      break;
    case MeepleType::kCotton:
      result += "C";
      break;
    case MeepleType::kEmpty:
      result += " ";
      break;
    default:
      SpielFatalError("Unknown meeple type");
  }
  
  return result;
}

std::string Move::ToString() const {
  if (is_pass()) return "Pass";
  
  std::stringstream ss;
  ss << ColorToString(player_color) << " starts from " << start_hex.ToString();
  ss << " and follows path: ";
  std::vector<std::string> hex_strs;
  for (const auto& hex : path) {
    hex_strs.push_back(hex.ToString());
  }
  ss << absl::StrJoin(hex_strs, " -> ");
  
  if (place_trading_post) {
    ss << " (places trading post)";
  }
  
  return ss.str();
}

std::string Move::ToLAN() const {
  if (is_pass()) return "pass";
  
  std::stringstream ss;
  ss << start_hex.x << "," << start_hex.y << "," << start_hex.z;
  
  for (const auto& hex : path) {
    ss << ":" << hex.x << "," << hex.y << "," << hex.z;
  }
  
  if (place_trading_post) {
    ss << ":post";
  }
  
  return ss.str();
}

// Constructor for Mali_BaBoard
Mali_BaBoard::Mali_BaBoard(int grid_radius)
    : grid_radius_(grid_radius),
      to_play_(Color::kBlack),
      move_number_(1),
      zobrist_hash_(0) {
  
  // Calculate the total number of hexes in the board
  // For a hex grid of radius r, we have 3r(r+1)+1 hexes total
  int num_hexes = 3 * grid_radius * (grid_radius + 1) + 1;
  
  // Initialize the board with empty pieces
  board_.resize(num_hexes, kEmptyPiece);
  
  // Initialize trading posts
  trading_posts_.resize(num_hexes, {Color::kEmpty, TradePostType::kNone});
  
  // Initialize cities and meeples for a standard setup
  InitializeCities();
  InitializeMeeples();
}

// Implement cities initialization
void Mali_BaBoard::InitializeCities() {
#ifdef MALI_BA_SIMPLIFIED_BOARD
  // Simplified board with just 4 cities and 4 regions
  cities_.emplace_back("Warri", Region::kIdjo, "Palm Oil", "Coral Necklace", HexCoord(-2, 0, 2));      // Northwest corner
  cities_.emplace_back("Dosso", Region::kSonghai, "Cotton", "Silver Headdress", HexCoord(2, -2, 0));   // Northeast corner
  cities_.emplace_back("Tabou", Region::kKru, "Pepper", "Canoe", HexCoord(-2, 2, 0));                  // Southwest corner
  cities_.emplace_back("Oyo", Region::kYoruba, "Ivory", "Ivory Bracelet", HexCoord(2, 0, -2));         // Southeast corner
#else
  // Full board with all cities and regions
  cities_.emplace_back("Timbuktu", Region::kSonghai, "Cotton", "Silver Headdress", HexCoord(0, 0, 0));
  cities_.emplace_back("Agadez", Region::kTuareg, "Iron", "Silver Cross", HexCoord(3, -1, -2));
  cities_.emplace_back("Bandiagara", Region::kDogon, "Onions", "Dogon Mask", HexCoord(-2, -1, 3));
  cities_.emplace_back("Dinguiraye", Region::kFulani, "Cattle", "Wedding Blanket", HexCoord(1, 2, -3));
  cities_.emplace_back("Hemang", Region::kAkan, "Gold", "Gold Weight", HexCoord(-3, 0, 3));
  cities_.emplace_back("Katsina", Region::kHousa, "Kola Nuts", "Holy Book", HexCoord(2, -3, 1));
  cities_.emplace_back("Linguère", Region::kWolof, "Peanuts", "Gold Necklace", HexCoord(-1, -2, 3));
  cities_.emplace_back("Ouagadougou", Region::kDagbani, "Horses", "Bronze Bracelet", HexCoord(2, 1, -3));
  cities_.emplace_back("Oudane", Region::kArab, "Camel", "Bronze Incense Burner", HexCoord(4, -2, -2));
  cities_.emplace_back("Oyo", Region::kYoruba, "Ivory", "Ivory Bracelet", HexCoord(-4, 2, 2));
  cities_.emplace_back("Segou", Region::kMande, "Millet", "Chiwara", HexCoord(0, 3, -3));
  cities_.emplace_back("Sikasso", Region::kSenoufo, "Brass", "Kora", HexCoord(-2, 4, -2));
  cities_.emplace_back("Tabou", Region::kKru, "Pepper", "Canoe", HexCoord(3, -4, 1));
  cities_.emplace_back("Warri", Region::kIdjo, "Palm Oil", "Coral Necklace", HexCoord(-3, -1, 4));
#endif
  
  // For each city, mark its location on the board with a special piece or property
  for (const auto& city : cities_) {
    int index = HexToIndex(city.location);
    // We could use a special piece type to represent cities
    // For now, just ensure the location is valid
    SPIEL_CHECK_TRUE(InBoardArea(city.location));
  }
}

// Implement initial meeple distribution
void Mali_BaBoard::InitializeMeeples() {
  // Random distribution of meeples across the board
  std::mt19937 rng(42);  // Fixed seed for reproducibility
  
  // Define possible meeple types for distribution
  std::vector<MeepleType> meeple_types = {
    MeepleType::kTrader, MeepleType::kGold, MeepleType::kSpice,
    MeepleType::kSalt, MeepleType::kIvory, MeepleType::kCotton
  };
  
#ifdef MALI_BA_SIMPLIFIED_BOARD
  // For simplified board, use a more deterministic placement
  // Place 3 meeples in predefined locations
  
  // Place some meeples around Warri
  set_piece(HexCoord(-4, -1, 5), {Color::kEmpty, MeepleType::kTrader});
  set_piece(HexCoord(-2, -1, 3), {Color::kEmpty, MeepleType::kGold});
  set_piece(HexCoord(-3, -2, 5), {Color::kEmpty, MeepleType::kIvory});
  
  // Place some meeples around Dosso
  set_piece(HexCoord(3, -1, -2), {Color::kEmpty, MeepleType::kSpice});
  set_piece(HexCoord(4, -1, -3), {Color::kEmpty, MeepleType::kSalt});
  set_piece(HexCoord(2, 0, -2), {Color::kEmpty, MeepleType::kCotton});
  
  // Place some meeples around Tabou
  set_piece(HexCoord(-4, -1, 5), {Color::kEmpty, MeepleType::kTrader});
  set_piece(HexCoord(-2, -3, 5), {Color::kEmpty, MeepleType::kGold});
  
  // Place some meeples around Oyo
  set_piece(HexCoord(1, -3, 2), {Color::kEmpty, MeepleType::kSpice});
  set_piece(HexCoord(2, -3, 1), {Color::kEmpty, MeepleType::kIvory});
#else
  // For the full board, use random distribution
  // For each hex (except cities, which we'll handle separately)
  for (int i = -grid_radius_; i <= grid_radius_; ++i) {
    for (int j = -grid_radius_; j <= grid_radius_; ++j) {
      int k = -i - j;
      
      // Skip invalid hex coordinates (not in a circle of radius grid_radius_)
      if (std::abs(i) + std::abs(j) + std::abs(k) > 2 * grid_radius_) continue;
      
      HexCoord hex(i, j, k);
      
      // Skip hexes with cities on them
      bool is_city = false;
      for (const auto& city : cities_) {
        if (city.location == hex) {
          is_city = true;
          break;
        }
      }
      if (is_city) continue;
      
      // Randomly decide if this hex gets meeples (70% chance)
      if (rng() % 100 < 70) {
        // Place a meeple of random type
        MeepleType type = meeple_types[rng() % meeple_types.size()];
        // Neutral meeples initially (will be claimed by players during gameplay)
        set_piece(hex, {Color::kEmpty, type});
      }
    }
  }
#endif
}

// Get piece at a specific hex
const Piece& Mali_BaBoard::at(const HexCoord& hex) const {
  if (!InBoardArea(hex)) return kEmptyPiece;
  return board_[HexToIndex(hex)];
}

// Set piece at a specific hex
void Mali_BaBoard::set_piece(const HexCoord& hex, const Piece& piece) {
  if (!InBoardArea(hex)) return;
  board_[HexToIndex(hex)] = piece;
  
  // Update Zobrist hash if implementing that for state tracking
  // (implementation omitted for simplicity)
}

// Get trading post info at a specific hex
const TradePost& Mali_BaBoard::post_at(const HexCoord& hex) const {
  static const TradePost kEmptyPost = {Color::kEmpty, TradePostType::kNone};
  if (!InBoardArea(hex)) return kEmptyPost;
  return trading_posts_[HexToIndex(hex)];
}

// Set trading post at a specific hex
void Mali_BaBoard::set_post(const HexCoord& hex, const TradePost& post) {
  if (!InBoardArea(hex)) return;
  trading_posts_[HexToIndex(hex)] = post;
  
  // Update Zobrist hash if needed
  // (implementation omitted for simplicity)
}

// Set current player
void Mali_BaBoard::SetToPlay(Color c) {
  to_play_ = c;
  // Update Zobrist hash if needed
}

// Check if a hex is within the board area
bool Mali_BaBoard::InBoardArea(const HexCoord& hex) const {
  // In cube coordinates, for a grid of radius r, all valid hexes satisfy:
  // |x| <= r, |y| <= r, |z| <= r
  return hex.IsValid() && 
         std::abs(hex.x) <= grid_radius_ && 
         std::abs(hex.y) <= grid_radius_ && 
         std::abs(hex.z) <= grid_radius_;
}

// Check if a hex is empty
bool Mali_BaBoard::IsEmpty(const HexCoord& hex) const {
  return at(hex).type == MeepleType::kEmpty;
}

// Check if a hex belongs to the opponent
bool Mali_BaBoard::IsEnemy(const HexCoord& hex, Color our_color) const {
  const Piece& piece = at(hex);
  return piece.type != MeepleType::kEmpty && piece.color != our_color &&
         piece.color != Color::kEmpty;
}

// Check if a hex has our piece
bool Mali_BaBoard::IsFriendly(const HexCoord& hex, Color our_color) const {
  const Piece& piece = at(hex);
  return piece.color == our_color;
}

// Get meeples count at a specific hex
int Mali_BaBoard::MeeplesCount(const HexCoord& hex) const {
  // In this implementation, we're tracking a single piece per hex location
  // But we could track multiple meeples per hex if needed
  return at(hex).type != MeepleType::kEmpty ? 1 : 0;
}

// Find a city at a given location
absl::optional<const City*> Mali_BaBoard::CityAt(const HexCoord& hex) const {
  for (const auto& city : cities_) {
    if (city.location == hex) {
      return &city;
    }
  }
  return absl::nullopt;
}

// Apply a move to the board
void Mali_BaBoard::ApplyMove(const Move& move) {
  if (move.is_pass()) {
    // Pass move - just change the player
    SetToPlay(OppColor(to_play_));
    if (to_play_ == Color::kBlack) {
      ++move_number_;  // Increment move number after a full round
    }
    return;
  }
  
  // Check that the move is valid
  SPIEL_CHECK_TRUE(InBoardArea(move.start_hex));
  SPIEL_CHECK_GE(move.path.size(), 0);
  
  // Get number of meeples at the starting hex
  int num_meeples = MeeplesCount(move.start_hex);
  
  // Check we have the right number of meeples for this path
  SPIEL_CHECK_EQ(num_meeples, move.path.size());
  
  // Distribute meeples along the path
  Piece starting_piece = at(move.start_hex);
  set_piece(move.start_hex, kEmptyPiece); // Remove meeples from starting hex
  
  // Place one meeple on each hex in the path
  for (size_t i = 0; i < move.path.size(); ++i) {
    SPIEL_CHECK_TRUE(InBoardArea(move.path[i]));
    
    // Place a meeple on this hex - set color to current player
    set_piece(move.path[i], {to_play_, starting_piece.type});
  }
  
  // Place a trading post if requested and legal
  if (move.place_trading_post && move.path.size() > 0) {
    const HexCoord& last_hex = move.path.back();
    if (CanPlaceTradingPost(last_hex, to_play_)) {
      set_post(last_hex, {to_play_, TradePostType::kPost});
    }
  }
  
  // Change player
  SetToPlay(OppColor(to_play_));
  
  // Increment move number after full round
  if (to_play_ == Color::kBlack) {
    ++move_number_;
  }
}

// Check if a trading post can be placed
bool Mali_BaBoard::CanPlaceTradingPost(const HexCoord& hex, Color color) const {
  // Rules for trading post placement:
  // 1. Hex must be emptied of meeples by the current move
  // 2. Cannot place where there's already a post of same player
  // 3. Number of trading centers limited based on number of players
  
  // First, check if this hex is empty of meeples
  if (!IsEmpty(hex)) return false;
  
  // Don't place on a city
  if (CityAt(hex).has_value()) return false;
  
  // Check if there's already a trading post or center here
  const TradePost& existing_post = post_at(hex);
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

// Generate all legal moves for the current player
void Mali_BaBoard::GenerateLegalMoves(const MoveYieldFn& yield) const {
  GenerateLegalMoves(yield, to_play_);
}

void Mali_BaBoard::GenerateLegalMoves(const MoveYieldFn& yield, Color color) const {
  bool generating = true;
  
  // Always provide option to pass
  if (!yield(kPassMove)) {
    return;
  }
  
  // Iterate through all hexes
  for (int i = -grid_radius_; i <= grid_radius_ && generating; ++i) {
    for (int j = -grid_radius_; j <= grid_radius_ && generating; ++j) {
      int k = -i - j;
      
      // Skip invalid hex coordinates
      if (std::abs(i) + std::abs(j) + std::abs(k) > 2 * grid_radius_) continue;
      
      HexCoord hex(i, j, k);
      
      // Check if this hex has our meeples or neutral meeples we can use
      const Piece& piece = at(hex);
      if (piece.type != MeepleType::kEmpty && 
          (piece.color == color || piece.color == Color::kEmpty)) {
        
        // Get meeple count at this location
        int meeple_count = MeeplesCount(hex);
        
        // Skip if no meeples
        if (meeple_count <= 0) continue;
        
        // Generate all possible paths starting from this hex
        std::vector<HexCoord> current_path;
        std::vector<bool> visited(board_.size(), false);
        visited[HexToIndex(hex)] = true;  // Mark start as visited
        
        // Generate paths of exactly meeple_count length
        GeneratePaths(hex, current_path, visited, meeple_count, 
                      [&](const Move& move) {
                        return yield(move) && (generating = true);
                      }, color);
      }
    }
  }
}

// Helper method to recursively generate all valid paths
void Mali_BaBoard::GeneratePaths(
    const HexCoord& start,
    std::vector<HexCoord>& current_path,
    std::vector<bool>& visited,
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
  auto neighbors = GetNeighbors(current);
  for (const auto& next : neighbors) {
    // Skip if not in board area
    if (!InBoardArea(next)) continue;
    
    // Skip already visited hexes
    if (visited[HexToIndex(next)]) continue;
    
    // Mark as visited
    visited[HexToIndex(next)] = true;
    
    // Add to path
    current_path.push_back(next);
    
    // Recurse
    GeneratePaths(start, current_path, visited, remaining_meeples - 1, 
                 yield, color);
    
    // Backtrack
    current_path.pop_back();
    visited[HexToIndex(next)] = false;
  }
}

// Debug string representation
std::string Mali_BaBoard::DebugString() const {
  std::stringstream ss;
  
  ss << "Mali-Ba Board (Grid Radius: " << grid_radius_ << ")\n";
  ss << "Current Player: " << ColorToString(to_play_) << "\n";
  ss << "Move Number: " << move_number_ << "\n\n";
  
  // Display the board in a hex grid layout
  // This is complex to display properly in text, but we'll do a simple version
  
  // Loop through rows (z coordinate)
  for (int z = -grid_radius_; z <= grid_radius_; ++z) {
    // Indent based on row
    for (int i = 0; i < grid_radius_ + z; ++i) {
      ss << "  ";
    }
    
    // Loop through hexes in this row
    for (int x = std::max(-grid_radius_, -z - grid_radius_); 
         x <= std::min(grid_radius_, -z + grid_radius_); 
         ++x) {
      int y = -x - z;  // Calculate y from x and z
      
      HexCoord hex(x, y, z);
      
      // Skip if not in board area
      if (!InBoardArea(hex)) continue;
      
      // Display piece and trading post
      const Piece& piece = at(hex);
      const TradePost& post = post_at(hex);
      
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
std::string Mali_BaBoard::ToMBN() const {
  // A simple MBN format could look like:
  // <current_player>/<move_number>/<board_pieces>/<trading_posts>
  
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
  
  // Board pieces (compact encoding)
  // We'll use run-length encoding for empty spaces
  int empty_count = 0;
  
  for (int i = -grid_radius_; i <= grid_radius_; ++i) {
    for (int j = -grid_radius_; j <= grid_radius_; ++j) {
      int k = -i - j;
      
      // Skip invalid hex coordinates
      if (std::abs(i) + std::abs(j) + std::abs(k) > 2 * grid_radius_) continue;
      
      HexCoord hex(i, j, k);
      const Piece& piece = at(hex);
      
      if (piece.type == MeepleType::kEmpty) {
        empty_count++;
      } else {
        // Output empty count if needed
        if (empty_count > 0) {
          ss << empty_count;
          empty_count = 0;
        }
        
        // Output the piece
        ss << piece.ToString();
      }
    }
  }
  
  // Output final empty count if needed
  if (empty_count > 0) {
    ss << empty_count;
  }
  
  ss << "/";
  
  // Trading posts (compact encoding)
  for (int i = -grid_radius_; i <= grid_radius_; ++i) {
    for (int j = -grid_radius_; j <= grid_radius_; ++j) {
      int k = -i - j;
      
      // Skip invalid hex coordinates
      if (std::abs(i) + std::abs(j) + std::abs(k) > 2 * grid_radius_) continue;
      
      HexCoord hex(i, j, k);
      const TradePost& post = post_at(hex);
      
      if (post.type != TradePostType::kNone) {
        // Output the hex coordinates and post info
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
  }
  
  return ss.str();
}

// Create a default board with standard setup
Mali_BaBoard MakeDefaultBoard() {
  return Mali_BaBoard(Mali_BaBoard::DefaultGridRadius());
}

// Static method to create a board from MBN notation
absl::optional<Mali_BaBoard> Mali_BaBoard::BoardFromMBN(const std::string& mbn) {
  // Parse MBN string and create a board
  // This is a placeholder implementation - would need to be expanded
  // based on the actual MBN format
  std::vector<std::string> parts = absl::StrSplit(mbn, '/');
  if (parts.size() != 4) {
    return absl::nullopt;  // Invalid format
  }
  
  // Create a default board with the correct radius
  Mali_BaBoard board(DefaultGridRadius());
  
  // Initialize with default layout
  board.InitializeCities();
  board.InitializeMeeples();
  
  // Set the current player
  if (parts[0] == "b") {
    board.SetToPlay(Color::kBlack);
  } else if (parts[0] == "w") {
    board.SetToPlay(Color::kWhite);
  } else if (parts[0] == "r") {
    board.SetToPlay(Color::kRed);
  } else if (parts[0] == "u") {
    board.SetToPlay(Color::kBlue);
  } else {
    return absl::nullopt;  // Invalid player
  }
  
  // Set move number
  try {
    board.move_number_ = std::stoi(parts[1]);
  } catch (...) {
    return absl::nullopt;  // Invalid move number
  }
  
  // For now, we just return the board with its initial setup
  // A complete implementation would parse parts[2] for pieces and parts[3] for trading posts
  
  return board;
}

// Count number of connected trading posts for a player
int Mali_BaBoard::CountConnectedPosts(Color color) const {
  // This would use a graph traversal algorithm to find connected components
  // of trading posts of the same color
  // Implementation placeholder
  return 0;
}

}  // namespace mali_ba

}  // namespace open_spiel