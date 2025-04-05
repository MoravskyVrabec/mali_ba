// board_config.h
#ifndef OPEN_SPIEL_GAMES_MALI_BA_BOARD_CONFIG_H_
#define OPEN_SPIEL_GAMES_MALI_BA_BOARD_CONFIG_H_

#include <string>
#include <vector>
#include <set>
#include "open_spiel/games/mali_ba/hex_grid.h"

namespace open_spiel {
namespace mali_ba {

// Configuration for Mali-Ba board loaded from .ini file
struct BoardConfig {
  bool regular_board = true;
  int board_radius = 3;
  std::set<HexCoord> valid_hexes;

  // Load configuration from an .ini file
  static BoardConfig LoadFromFile(const std::string& filename);
  
  // Parse a string of valid hexes in the format "-1,2,-1;0,0,0;1,-1,0"
  static std::set<HexCoord> ParseValidHexes(const std::string& hex_str);
  
  // Generate valid hexes for a regular board with the given radius
  static std::set<HexCoord> GenerateRegularBoard(int radius);
};

} // namespace mali_ba
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_MALI_BA_BOARD_CONFIG_H_