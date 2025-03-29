#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include "open_spiel/games/mali_ba/mali_ba.h"
#include "open_spiel/games/mali_ba/flex_board.h"
#include "open_spiel/games/mali_ba/board_adapter.h"
#include "open_spiel/spiel.h"

using namespace open_spiel;
using namespace open_spiel::mali_ba;

// Example of creating a custom Mali-Ba board with disconnected regions
void CustomBoardExample() {
  std::cout << "Creating a custom Mali-Ba board\n";
  
  // Create a "figure eight" shaped board with two regions
  std::vector<HexCoord> valid_hexes;
  
  // First region (top circle)
  valid_hexes.push_back(HexCoord(0, 0, 0));  // Center
  valid_hexes.push_back(HexCoord(1, -1, 0));
  valid_hexes.push_back(HexCoord(1, 0, -1));
  valid_hexes.push_back(HexCoord(0, 1, -1));
  valid_hexes.push_back(HexCoord(-1, 1, 0));
  valid_hexes.push_back(HexCoord(-1, 0, 1));
  valid_hexes.push_back(HexCoord(0, -1, 1));
  
  // Second region (bottom circle) - disconnected from the first
  valid_hexes.push_back(HexCoord(0, -3, 3));  // Center
  valid_hexes.push_back(HexCoord(1, -4, 3));
  valid_hexes.push_back(HexCoord(1, -3, 2));
  valid_hexes.push_back(HexCoord(0, -2, 2));
  valid_hexes.push_back(HexCoord(-1, -2, 3));
  valid_hexes.push_back(HexCoord(-1, -3, 4));
  valid_hexes.push_back(HexCoord(0, -4, 4));
  
  // Create the flexible board with these hexes
  FlexBoard board = FlexBoard::CreateCustomBoard(valid_hexes);
  
  // Initialize with some cities
  // In a real implementation, we'd customize the InitializeCities method
  // Here we'll manually add some cities
  
  // Add a city to the first region
  City city1("Timbuktu", Region::kSonghai, "Cotton", "Silver Headdress", HexCoord(0, 0, 0));
  
  // Add a city to the second region
  City city2("Warri", Region::kIdjo, "Palm Oil", "Coral Necklace", HexCoord(0, -3, 3));
  
  // Print the board
  std::cout << board.DebugString() << std::endl;
  
  // Find connected components
  auto components = board.GetConnectedComponents();
  
  // Print information about the components
  std::cout << "Found " << components.size() << " disconnected regions on the board.\n";
  for (size_t i = 0; i < components.size(); ++i) {
    std::cout << "Region " << (i+1) << " contains " << components[i].size() 
              << " hexes.\n";
  }
  
  // Test connectivity between hexes
  HexCoord hex1(0, 0, 0);  // Center of first region
  HexCoord hex2(0, -3, 3); // Center of second region
  
  if (board.AreConnected(hex1, hex2)) {
    std::cout << "Hexes " << hex1.ToString() << " and " << hex2.ToString() 
              << " are connected.\n";
  } else {
    std::cout << "Hexes " << hex1.ToString() << " and " << hex2.ToString() 
              << " are NOT connected.\n";
  }
  
  // Create a board adapter to use with the game engine
  BoardAdapter adapter(board);
  
  // Initialize some pieces
  adapter.SetPiece(HexCoord(1, -1, 0), {Color::kBlack, MeepleType::kTrader});
  adapter.SetPiece(HexCoord(-1, 1, 0), {Color::kWhite, MeepleType::kGold});
  adapter.SetPiece(HexCoord(0, -4, 4), {Color::kBlack, MeepleType::kSalt});
  
  // Generate legal moves
  std::vector<Move> legal_moves;
  adapter.GenerateLegalMoves([&legal_moves](const Move& move) {
    legal_moves.push_back(move);
    return true;
  });
  
  std::cout << "Found " << legal_moves.size() << " legal moves.\n";
  
  // Print a few sample moves
  for (size_t i = 0; i < std::min(5ul, legal_moves.size()); ++i) {
    std::cout << "Move " << (i+1) << ": " << legal_moves[i].ToString() << "\n";
  }
  
  // Note the importance of connectivity for game mechanics
  std::cout << "\nImportant Note: With disconnected regions, the Mancala-style movement\n"
            << "is constrained within each region. Players cannot move pieces between\n"
            << "disconnected regions, which creates strategic implications.\n";
  
  // Serialize the board state
  std::string mbn = adapter.ToMBN();
  std::cout << "\nBoard MBN representation: " << mbn << "\n";
}