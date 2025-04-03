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

#ifndef OPEN_SPIEL_GAMES_MALI_BA_HEX_GRID_H_
#define OPEN_SPIEL_GAMES_MALI_BA_HEX_GRID_H_

#include <array>
#include <cstdint>
#include <vector>
#include <string>
#include <ostream>
#include <cmath>

#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace mali_ba {

// Representation of a hex using cube coordinates (x, y, z) where x + y + z = 0
struct HexCoord {
  int8_t x;
  int8_t y;
  int8_t z;
  
  // Constructor with validation
  HexCoord(int8_t x, int8_t y, int8_t z) : x(x), y(y), z(z) {
    // Check that coordinates satisfy x + y + z = 0
    int8_t sum = x + y + z;
    SPIEL_CHECK_EQ(sum, 0);
  }
  
  // Shorthand constructor - computes z from x and y
  HexCoord(int8_t x, int8_t y) : x(x), y(y), z(-x - y) {}
  
  // Constructor for invalid or uninitialized hex
  HexCoord() : x(-127), y(-127), z(-127) {}
  
  bool operator==(const HexCoord& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
  
  bool operator!=(const HexCoord& other) const {
    return !(*this == other);
  }
  
  // Required for use in STL containers
  bool operator<(const HexCoord& other) const {
    if (x != other.x) return x < other.x;
    if (y != other.y) return y < other.y;
    return z < other.z;
  }
  
  // Addition with another hex
  HexCoord operator+(const HexCoord& other) const {
    return HexCoord(x + other.x, y + other.y, z + other.z);
  }
  
  // Check if this is a valid hex
  bool IsValid() const {
    return x != -127 && y != -127 && z != -127 && (x + y + z == 0);
  }
  
  // Distance to another hex
  int Distance(const HexCoord& other) const {
    return (std::abs(x - other.x) + std::abs(y - other.y) + std::abs(z - other.z)) / 2;
  }
  
  std::string ToString() const {
    return "(" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(z) + ")";
  }
};

// Invalid hex used as sentinel
// We can't use constexpr with HexCoord() constructor due to the SPIEL_CHECK_EQ call
const HexCoord kInvalidHex = HexCoord();

// The six directions in cube coordinates (for finding neighbors)
// Can't be constexpr because of the SPIEL_CHECK_EQ in the constructor
inline const std::array<HexCoord, 6> kHexDirections = {{
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
// This implementation assumes "odd-q" offset system where odd columns are shifted up
// Replace or fix the existing OffsetToCube function in hex_grid.h
inline HexCoord OffsetToCube(int col, int row) {
  // Using "odd-r" horizontal layout
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

// Utility function to get a linear index for a hex in a grid of given radius
inline int HexToIndex(const HexCoord& hex, int grid_radius) {
  // Calculate index in a spiral layout starting from center (0,0,0)
  // This is one possible mapping scheme; can be adjusted based on needs
  int N = 2 * grid_radius + 1;
  auto [col, row] = CubeToOffset(hex);
  col += grid_radius;  // Shift to 0-based indexing
  row += grid_radius;
  return row * N + col;
}

// Convert a linear index back to a hex coordinate
inline HexCoord IndexToHex(int index, int grid_radius) {
  int N = 2 * grid_radius + 1;
  int row = index / N;
  int col = index % N;
  row -= grid_radius;  // Shift back to centered coordinates
  col -= grid_radius;
  return OffsetToCube(col, row);
}

// Print a hex to an output stream
inline std::ostream& operator<<(std::ostream& stream, const HexCoord& hex) {
  return stream << hex.ToString();
}

}  // namespace mali_ba
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MALI_BA_HEX_GRID_H_