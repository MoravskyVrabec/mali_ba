// board_config.cc
#include "open_spiel/games/mali_ba/board_config.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace mali_ba {

// Simple INI file parsing function to extract a value by key
std::string GetIniValue(const std::string& content, const std::string& key) {
  std::istringstream stream(content);
  std::string line;
  
  while (std::getline(stream, line)) {
    // Skip comments and empty lines
    if (line.empty() || line[0] == ';' || line[0] == '#') continue;
    
    // Look for the key followed by = or :
    size_t pos = line.find(key);
    if (pos != std::string::npos) {
      // Find the separator (= or :)
      size_t sep_pos = line.find('=', pos + key.length());
      if (sep_pos == std::string::npos) {
        sep_pos = line.find(':', pos + key.length());
      }
      
      if (sep_pos != std::string::npos) {
        // Extract value, trimming whitespace
        std::string value = line.substr(sep_pos + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);
        return value;
      }
    }
  }
  
  return "";
}

BoardConfig BoardConfig::LoadFromFile(const std::string& filename) {
  BoardConfig config;
  
  // Try to open the INI file
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Warning: Could not open board configuration file: " << filename 
              << ". Using default regular board with radius 3." << std::endl;
    return config; // Return default config
  }
  
  // Read the entire file content
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string content = buffer.str();
  
  // Parse regular_board parameter
  std::string regular_board_str = GetIniValue(content, "regular_board");
  if (regular_board_str.empty()) {
    std::cerr << "Warning: Missing required parameter 'regular_board'. "
              << "Using default: Y" << std::endl;
  } else {
    // Convert to uppercase for comparison
    for (char& c : regular_board_str) c = std::toupper(c);
    if (regular_board_str == "Y" || regular_board_str == "YES" || regular_board_str == "TRUE") {
      config.regular_board = true;
    } else if (regular_board_str == "N" || regular_board_str == "NO" || regular_board_str == "FALSE") {
      config.regular_board = false;
    } else {
      std::cerr << "Warning: Invalid value for 'regular_board': " << regular_board_str 
                << ". Using default: Y" << std::endl;
    }
  }
  
  // Parse board_radius parameter
  std::string radius_str = GetIniValue(content, "board_radius");
  if (radius_str.empty()) {
    std::cerr << "Warning: Missing required parameter 'board_radius'. "
              << "Using default: 3" << std::endl;
  } else {
    try {
      config.board_radius = std::stoi(radius_str);
    } catch (const std::exception& e) {
      std::cerr << "Warning: Invalid value for 'board_radius': " << radius_str 
                << ". Using default: 3" << std::endl;
    }
  }
  
  // Parse valid_hexes if needed
  if (!config.regular_board) {
    std::string hexes_str = GetIniValue(content, "board_valid_hexes");
    if (hexes_str.empty()) {
      std::cerr << "Warning: Missing required parameter 'board_valid_hexes' for irregular board. "
                << "Falling back to regular board." << std::endl;
      config.regular_board = true;
    } else {
      config.valid_hexes = ParseValidHexes(hexes_str);
      if (config.valid_hexes.empty()) {
        std::cerr << "Warning: Failed to parse 'board_valid_hexes'. "
                  << "Falling back to regular board." << std::endl;
        config.regular_board = true;
      }
    }
  }
  
  // Generate valid hexes for regular board
  if (config.regular_board) {
    config.valid_hexes = GenerateRegularBoard(config.board_radius);
  }
  
  return config;
}

std::set<HexCoord> BoardConfig::ParseValidHexes(const std::string& hex_str) {
  std::set<HexCoord> hexes;
  
  // Split by semicolons to get individual hex coordinates
  std::vector<std::string> hex_coords = absl::StrSplit(hex_str, ';');
  
  for (const auto& coord_str : hex_coords) {
    // Split by commas to get x,y,z components
    std::vector<std::string> components = absl::StrSplit(coord_str, ',');
    
    if (components.size() != 3) {
      continue; // Skip invalid format
    }
    
    try {
      int x = std::stoi(components[0]);
      int y = std::stoi(components[1]);
      int z = std::stoi(components[2]);
      
      // Validate that x + y + z = 0
      if (x + y + z != 0) {
        std::cerr << "Warning: Invalid hex coordinate (x+y+z!=0): " 
                  << coord_str << std::endl;
        continue;
      }
      
      hexes.insert(HexCoord(x, y, z));
    } catch (const std::exception& e) {
      std::cerr << "Warning: Failed to parse hex coordinate: " << coord_str << std::endl;
    }
  }
  
  return hexes;
}

std::set<HexCoord> BoardConfig::GenerateRegularBoard(int radius) {
  std::set<HexCoord> hexes;
  
  // Add all hexes within radius
  for (int x = -radius; x <= radius; ++x) {
    for (int y = -radius; y <= radius; ++y) {
      int z = -x - y;
      // Skip invalid hex coordinates (not in a circle of radius)
      if (std::abs(x) + std::abs(y) + std::abs(z) > 2 * radius) continue;
      
      hexes.insert(HexCoord(x, y, z));
    }
  }
  
  return hexes;
}

} // namespace mali_ba
} // namespace open_spiel