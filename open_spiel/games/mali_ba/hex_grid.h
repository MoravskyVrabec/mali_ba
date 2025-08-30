// hex_grid.h 
#ifndef OPEN_SPIEL_GAMES_MALI_BA_HEX_GRID_H_
#define OPEN_SPIEL_GAMES_MALI_BA_HEX_GRID_H_

#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <ostream>
#include <cmath>

namespace open_spiel {
namespace mali_ba {

// Representation of a hex using cube coordinates (x, y, z) where x + y + z = 0
struct HexCoord {
  int x;
  int y;
  int z;
  
  // Constructor with validation
  HexCoord(int x, int y, int z) : x(x), y(y), z(z) {
    // In production code, we'd check that x + y + z = 0
    // For simplicity in parsing, we'll skip the validation
  }
  
  // Shorthand constructor - computes z from x and y
  HexCoord(int x, int y) : x(x), y(y), z(-x - y) {}
  
  // Default constructor for an invalid hex
  HexCoord() : x(0), y(0), z(0) {}

  // Hashing function
  template <typename H>
  friend H AbslHashValue(H h, const HexCoord& c) {
      // Combine the hash values of the members.
      // The order matters but this specific combination is common.
      return H::combine(std::move(h), c.x, c.y, c.z);
  }

  bool operator==(const HexCoord& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  
  bool operator!=(const HexCoord& other) const {
    return !(*this == other);
  }
  
  // Required for use in STL containers like std::set
  bool operator<(const HexCoord& other) const {
    if (x != other.x) return x < other.x;
    if (y != other.y) return y < other.y;
    return z < other.z;
  }
  
  // Addition with another hex
  HexCoord operator+(const HexCoord& other) const {
    return HexCoord(x + other.x, y + other.y, z + other.z);
  }
  
  // Distance to another hex
  int Distance(const HexCoord& other) const {
    return (std::abs(x - other.x) + std::abs(y - other.y) + std::abs(z - other.z)) / 2;
  }
  
  std::string ToString() const {
    return "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
  }
};

// The six directions in cube coordinates (for finding neighbors)
const std::array<HexCoord, 6> kHexDirections = {{
  HexCoord(1, -1, 0),  // East
  HexCoord(1, 0, -1),  // Southeast
  HexCoord(0, 1, -1),  // Southwest
  HexCoord(-1, 1, 0),  // West
  HexCoord(-1, 0, 1),  // Northwest
  HexCoord(0, -1, 1)   // Northeast
}};

// Get all 6 neighbors of a hex
inline std::vector<HexCoord> GetNeighbors(const HexCoord& hex) {
  std::vector<HexCoord> neighbors;
  neighbors.reserve(6);
  for (const auto& dir : kHexDirections) {
    neighbors.push_back(hex + dir);
  }
  return neighbors;
}

// Check if two hexes are adjacent
inline bool AreAdjacent(const HexCoord& a, const HexCoord& b) {
  return a.Distance(b) == 1;
}

// Convert between offset coordinates (row, col) and cube coordinates
// This implementation assumes "odd-r" horizontal layout
inline HexCoord OffsetToCube(int col, int row) {
  int x = col;
  int z = row - (col - (col & 1)) / 2;
  int y = -x - z;
  return HexCoord(x, y, z);
}

// Convert from cube to offset coordinates
inline std::pair<int, int> CubeToOffset(const HexCoord& hex) {
  int col = hex.x;
  int row = hex.z + (hex.x - (hex.x & 1)) / 2;
  return {col, row};
}

// Print a hex to an output stream
inline std::ostream& operator<<(std::ostream& stream, const HexCoord& hex) {
  return stream << hex.ToString();
}

}  // namespace mali_ba
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MALI_BA_HEX_GRID_H_
