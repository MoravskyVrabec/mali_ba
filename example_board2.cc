#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <random>
#include <cmath>

#include "open_spiel/games/mali_ba/mali_ba.h"
#include "open_spiel/games/mali_ba/flex_board.h"
#include "open_spiel/games/mali_ba/board_adapter.h"
#include "open_spiel/spiel.h"

using namespace open_spiel;
using namespace open_spiel::mali_ba;

// Helper function to create a spiral pattern of hexes
std::vector<HexCoord> CreateSpiralBoard(int max_radius, int num_rings) {
  std::vector<HexCoord> valid_hexes;
  
  // Start with the center hex
  valid_hexes.push_back(HexCoord(0, 0, 0));
  
  // Create a spiral pattern
  int direction = 0; // Start moving east
  int length = 1;    // Length of the current segment
  int x = 0, y = 0, z = 0;
  
  for (int ring = 1; ring <= num_rings; ++ring) {
    // For each ring, we need to create 6 segments, one for each direction
    for (int d = 0; d < 6; ++d) {
      direction = (direction + 1) % 6; // Turn 60 degrees
      
      // If this is the second segment of the ring or beyond, increase the length
      if (d >= 2) {
        length = ring;
      }
      
      // Create segment by adding hexes in the current direction
      for (int step = 0; step < length; ++step) {
        // Move in the current direction
        const HexCoord& dir = kHexDirections[direction];
        x += dir.x;
        y += dir.y;
        z += dir.z;
        
        // Add the new hex if it's within the maximum radius
        if (std::abs(x) <= max_radius && std::abs(y) <= max_radius && std::abs(z) <= max_radius) {
          valid_hexes.push_back(HexCoord(x, y, z));
        }
      }
    }
  }
  
  return valid_hexes;
}

// Helper function to create an archipelago (islands) board
std::vector<HexCoord> CreateArchipelagoBoard(int max_radius, int num_islands, int seed = 42) {
  std::vector<HexCoord> valid_hexes;
  std::mt19937 rng(seed);
  
  // Create a distribution for island centers
  std::uniform_int_distribution<int> coord_dist(-max_radius + 2, max_radius - 2);
  
  for (int i = 0; i < num_islands; ++i) {
    // Create a random island center
    int center_x = coord_dist(rng);
    int center_y = coord_dist(rng);
    int center_z = -center_x - center_y;
    
    // If z is out of range, adjust x and y
    if (std::abs(center_z) > max_radius - 2) {
      center_x = coord_dist(rng);
      center_y = -center_x - (max_radius - 2) * (center_z < 0 ? -1 : 1);
      center_z = -center_x - center_y;
    }
    
    HexCoord island_center(center_x, center_y, center_z);
    
    // Add the center hex
    valid_hexes.push_back(island_center);
    
    // Determine island size (random between 3 and 8 hexes)
    std::uniform_int_distribution<int> size_dist(3, 8);
    int island_size = size_dist(rng);
    
    // Add surrounding hexes to form the island
    std::vector<HexCoord> candidates = GetNeighbors(island_center);
    
    for (int j = 0; j < island_size - 1 && !candidates.empty(); ++j) {
      // Pick a random candidate
      std::uniform_int_distribution<int> cand_dist(0, candidates.size() - 1);
      int idx = cand_dist(rng);
      HexCoord next = candidates[idx];
      
      // Add it to the island
      valid_hexes.push_back(next);
      
      // Remove it from candidates
      candidates.erase(candidates.begin() + idx);
      
      // Add its neighbors to candidates, if they're within bounds
      for (const auto& neighbor : GetNeighbors(next)) {
        if (std::abs(neighbor.x) <= max_radius && 
            std::abs(neighbor.y) <= max_radius && 
            std::abs(neighbor.z) <= max_radius) {
            
          // Check if it's already in our island or candidates
          bool already_exists = false;
          for (const auto& hex : valid_hexes) {
            if (hex == neighbor) {
              already_exists = true;
              break;
            }
          }
          
          if (!already_exists) {
            for (const auto& hex : candidates) {
              if (hex == neighbor) {
                already_exists = true;
                break;
              }
            }
          }
          
          if (!already_exists) {
            candidates.push_back(neighbor);
          }
        }
      }
    }
  }
  
  return valid_hexes;
}

// Helper function to create a river board with trade outposts
std::vector<HexCoord> CreateRiverBoard(int max_radius) {
  std::vector<HexCoord> valid_hexes;
  
  // Add all hexes within radius
  for (int x = -max_radius; x <= max_radius; ++x) {
    for (int y = -max_radius; y <= max_radius; ++y) {
      int z = -x - y;
      // Skip invalid hex coordinates
      if (std::abs(x) + std::abs(y) + std::abs(z) > 2 * max_radius) continue;
      
      // Create a winding river pattern - hexes along the "riverbed" are invalid
      // Using a sine wave pattern for the river
      double river_center = 2.0 * sin(0.5 * static_cast<double>(x));
      double distance_from_river = std::abs(y - river_center);
      
      if (distance_from_river > 1.0) {
        valid_hexes.push_back(HexCoord(x, y, z));
      }
    }
  }
  
  // Add bridges at specific points
  valid_hexes.push_back(HexCoord(0, 0, 0));   // Center bridge
  valid_hexes.push_back(HexCoord(4, -1, -3)); // East bridge
  valid_hexes.push_back(HexCoord(-4, 0, 4));  // West bridge
  
  return valid_hexes;
}

// Demo function to create and display various creative board designs
void CreativeBoardDesignExamples() {
  std::cout << "=== Creative Board Design Examples ===\n\n";
  
  // Example 1: Spiral Board
  std::cout << "1. Spiral Board\n";
  std::vector<HexCoord> spiral_hexes = CreateSpiralBoard(7, 3);
  FlexBoard spiral_board = FlexBoard::CreateCustomBoard(spiral_hexes);
  
  std::cout << "Created a spiral board with " << spiral_hexes.size() << " hexes.\n";
  std::cout << "This design creates winding paths that limit movement options\n";
  std::cout << "and focuses gameplay along the spiral path.\n\n";
  
  // Example 2: Archipelago Board
  std::cout << "2. Archipelago Board\n";
  std::vector<HexCoord> archipelago_hexes = CreateArchipelagoBoard(7, 5);
  FlexBoard archipelago_board = FlexBoard::CreateCustomBoard(archipelago_hexes);
  
  std::cout << "Created an archipelago board with " << archipelago_hexes.size() << " hexes\n";
  std::cout << "distributed across 5 disconnected islands.\n";
  std::cout << "This design creates isolated gameplay zones and strategic\n";
  std::cout << "implications for resource and city placement.\n\n";
  
  // Example 3: River Board
  std::cout << "3. River Board\n";
  std::vector<HexCoord> river_hexes = CreateRiverBoard(7);
  FlexBoard river_board = FlexBoard::CreateCustomBoard(river_hexes);
  
  std::cout << "Created a river board with " << river_hexes.size() << " hexes\n";
  std::cout << "featuring a winding river that divides the board with limited crossing points.\n";
  std::cout << "This design creates natural barriers and strategic chokepoints,\n";
  std::cout << "requiring players to control key bridge locations.\n\n";
  
  // Check connectivity in the archipelago board
  std::cout << "Analyzing the archipelago board for connectivity...\n";
  auto components = archipelago_board.GetConnectedComponents();
  std::cout << "Found " << components.size() << " disconnected regions.\n";
  
  // Count hexes in each component
  for (size_t i = 0; i < components.size(); ++i) {
    std::cout << "  Island " << (i+1) << " has " << components[i].size() << " hexes.\n";
  }
  
  std::cout << "\nThese examples demonstrate the flexibility of the new board system,\n";
  std::cout << "allowing for creative game designs that would be impossible with\n";
  std::cout << "the traditional fixed-radius hex grid.\n\n";
}