#include "open_spiel/games/mali_ba/mali_ba.h"

#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/strings/match.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/games/mali_ba/mali_ba_observer.h"
#include "open_spiel/games/mali_ba/board_adapter.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace mali_ba {

namespace {

// Game type information
const GameType kGameType{
    /*short_name=*/"mali_ba",
    /*long_name=*/"Mali-Ba: Trade Routes of West Africa",
    GameType::Dynamics::kSequential,
    GameType::ChanceMode::kDeterministic,
    GameType::Information::kPerfectInformation,
    GameType::Utility::kZeroSum,
    GameType::RewardModel::kTerminal,
    /*max_num_players=*/4,
    /*min_num_players=*/2,
    /*provides_information_state_string=*/true,
    /*provides_information_state_tensor=*/false,
    /*provides_observation_string=*/true,
    /*provides_observation_tensor=*/true,
    /*parameter_specification=*/
    {
        {"players", GameParameter(2)},
        {"grid_radius", GameParameter(5)},
        {"simplified", GameParameter(true)},
        {"use_flexible_board", GameParameter(true)}
    }};

// Use the global pass action ID
// No need to redefine it here since it's already defined in mali_ba.h

// Register the observer
RegisterSingleTensorObserver single_tensor(kGameType.short_name);

// Register the game type with the OpenSpiel system
// Define the factory function for creating the game
std::shared_ptr<const Game> Factory(const GameParameters& params) {
  return std::shared_ptr<const Game>(new Mali_BaGame(params));
}

REGISTER_SPIEL_GAME(kGameType, Factory);

// Constants
constexpr int kNumRepetitionsToDraw = 3;  // Draw after position repeats 3 times

// Utility values for win/loss/draw
constexpr double kWinUtility = 1.0;
constexpr double kLossUtility = -1.0;
constexpr double kDrawUtility = 0.0;

}  // namespace

// Mali_BaGame implementation

// Constructor from parameters
Mali_BaGame::Mali_BaGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      grid_radius_(ParameterValue<int>("grid_radius")),
      use_flexible_board_(ParameterValue<bool>("use_flexible_board")) {
  // Validate parameters
  SPIEL_CHECK_GE(num_players_, 2);
  SPIEL_CHECK_LE(num_players_, 4);
      
#ifdef MALI_BA_SIMPLIFIED_BOARD
  // Override grid radius for simplified board
  grid_radius_ = Mali_BaBoard::DefaultGridRadius();
  // Note: The actual board layout is defined in FlexBoard::CreateSimplifiedCustomBoard
#else
  SPIEL_CHECK_GE(grid_radius_, 2);
  SPIEL_CHECK_LE(grid_radius_, 10); // Upper bound for reasonable grid size
#endif
}

std::shared_ptr<Observer> Mali_BaGame::MakeObserver(
  absl::optional<IIGObservationType> iig_obs_type,
  const GameParameters& params) const {
  if (params.empty()) {
    if (!iig_obs_type) {
      // Default observer
      return MakeMaliBaObserver(
          IIGObservationType{/*public_info=*/true,
                            /*perfect_recall=*/false,
                            /*private_info=*/PrivateInfoType::kAllPlayers});
    } else {
      return MakeMaliBaObserver(*iig_obs_type);
    }
  } else {
    // If we have parameters, use the parent class's implementation
    return Game::MakeObserver(iig_obs_type, params);
  }
}
// Deserialize a state from a string
std::unique_ptr<State> Mali_BaGame::DeserializeState(const std::string& str) const {
  // Check if the string is empty
  if (str.empty()) {
    return NewInitialState(); // Return a fresh state
  }
  
  std::unique_ptr<Mali_BaState> state = absl::make_unique<Mali_BaState>(shared_from_this(), str);
  
  // Verify the current player is correctly restored
  std::vector<std::string> parts = absl::StrSplit(str, '/');
  if (!parts.empty()) {
    Color expected_color = Color::kBlack; // Default
    if (parts[0] == "b") {
      expected_color = Color::kBlack;
    } else if (parts[0] == "w") {
      expected_color = Color::kWhite;
    } else if (parts[0] == "r") {
      expected_color = Color::kRed;
    } else if (parts[0] == "u") {
      expected_color = Color::kBlue;
    }
    
    if (state->Board().ToPlay() != expected_color) {
      // Fix the state if needed
      std::cerr << "Warning: Fixing deserialized state current player from "
                << ColorToString(state->Board().ToPlay()) << " to " 
                << ColorToString(expected_color) << std::endl;
      state->Board().SetToPlay(expected_color);
    }
  }
  
  return state;
}

// Mali_BaState implementation

// Constructor for a new game
Mali_BaState::Mali_BaState(std::shared_ptr<const Game> game)
    : State(game),
      is_terminal_(false) {
  
  auto mali_ba_game = static_cast<const Mali_BaGame*>(game.get());
  
  // Initialize board adapter based on game parameters
  bool use_flexible_board = mali_ba_game->UseFlexibleBoard();
  int grid_radius = mali_ba_game->GridRadius();
  
  // Create the board
  board_adapter_ = BoardAdapter::CreateSimplifiedBoard(use_flexible_board);
  
  // Initialize with meeples
  InitializeMeeples();
}

// Constructor from an MBN string
Mali_BaState::Mali_BaState(std::shared_ptr<const Game> game,
                         const std::string& mbn)
    : State(game),
      is_terminal_(false) {
  
  auto mali_ba_game = static_cast<const Mali_BaGame*>(game.get());
  bool use_flexible_board = mali_ba_game->UseFlexibleBoard();
  
  // Try to load from MBN
  board_adapter_ = BoardAdapter::FromMBN(mbn, use_flexible_board);
  
  // Record the starting board state
  start_board_ = board_adapter_;
}

// Initialize meeples on the board
void Mali_BaState::InitializeMeeples() {
  // Record the starting board state
  start_board_ = board_adapter_;
  
  // Note: The actual meeple initialization is done in the board implementation,
  // so we don't need to do anything here
}

// Get current player
Player Mali_BaState::CurrentPlayer() const {
  if (is_terminal_) {
    return kTerminalPlayerId;
  }
  return ColorToPlayer(board_adapter_.ToPlay());
}

// Get legal actions for current player
std::vector<Action> Mali_BaState::LegalActions() const {
  if (is_terminal_) {
    return {};
  }
  
  // Use cached legal actions if available
  if (cached_legal_actions_.has_value()) {
    return *cached_legal_actions_;
  }
  
  // Generate legal actions
  MaybeGenerateLegalActions();
  return *cached_legal_actions_;
}

// Helper to generate legal actions
void Mali_BaState::MaybeGenerateLegalActions() const {
  if (cached_legal_actions_.has_value()) {
    return;
  }
  
  std::vector<Action> actions;
  
  // Add Pass action first
  actions.push_back(kPassActionId);
  
  // Then add all other actions
  board_adapter_.GenerateLegalMoves([&](const Move& move) {
    // Skip pass moves since we already added kPassActionId
    if (!move.is_pass()) {
      actions.push_back(MoveToAction(move, board_adapter_.GridRadius()));
    }
    return true;
  });
  
  // Sort the actions in ascending order
  std::sort(actions.begin(), actions.end());
  
  // Remove duplicates - important to prevent the error
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
  
  cached_legal_actions_ = actions;
}

// Convert action to string
std::string Mali_BaState::ActionToString(Player player, Action action) const {
  if (action == kPassActionId) {
    return "Pass";
  }
  
  // Convert action to move and then to string
  Move move = ActionToMove(action, board_adapter_);
  return move.ToString();
}

// Convert board state to string
std::string Mali_BaState::ToString() const {
  return board_adapter_.DebugString();
}

// Check if game is over
bool Mali_BaState::IsTerminal() const {
  if (is_terminal_) {
    return true;
  }
  
  // Check win conditions
  auto returns = MaybeFinalReturns();
  is_terminal_ = returns.has_value();
  return is_terminal_;
}

// Get score for each player
std::vector<double> Mali_BaState::Returns() const {
  auto returns = MaybeFinalReturns();
  if (returns.has_value()) {
    return *returns;
  }
  
  // If game is not over, return zeros
  std::vector<double> result(NumPlayers(), 0);
  return result;
}

// Helper to check for game end
absl::optional<std::vector<double>> Mali_BaState::MaybeFinalReturns() const {
  // Game is over if either:
  // 1. A position is repeated 3 times (draw)
  // 2. A player has created 4 trading routes
  // 3. A player has 4 different rare goods
  // 4. A player connects Timbuktu with the sea via other cities
  
  // Check for repetitions
  for (const auto& [hash, count] : repetitions_) {
    if (count >= kNumRepetitionsToDraw) {
      // Draw
      std::vector<double> result(NumPlayers(), kDrawUtility);
      return result;
    }
  }
  
  // Check trading routes (placeholder implementation)
  for (int player = 0; player < NumPlayers(); player++) {
    Color color = PlayerToColor(player);
    int routes = board_adapter_.CountConnectedPosts(color);
    if (routes >= 4) {
      // This player wins
      std::vector<double> result(NumPlayers(), kLossUtility);
      result[player] = kWinUtility;
      return result;
    }
    
    // TODO: Implement checks for other win conditions:
    // - 4 different rare goods
    // - Connect Timbuktu with the sea via other cities
  }
  
  // No end condition met yet
  return absl::nullopt;
}

// Get game state from player's perspective
std::string Mali_BaState::InformationStateString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  
  // Mali-Ba is a perfect information game, so the information state
  // is the same as the observation state
  return ObservationString(player);
}

// Get observation of the game state
std::string Mali_BaState::ObservationString(Player player) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  
  // For perfect information games, the observation is just the board state
  return ToString();
}

// Get tensor representation of the observation
void Mali_BaState::ObservationTensor(Player player,
                                    absl::Span<float> values) const {
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, NumPlayers());
  
  // Zero out the values first
  std::fill(values.begin(), values.end(), 0);
  
  // This would be filled in properly by the observer defined in mali_ba_observer.cc
  // For now, we provide a very basic implementation
  
  const auto& shape = ObservationTensorShape();
  if (shape.size() != 3 || values.size() != shape[0] * shape[1] * shape[2]) {
    SpielFatalError("Observation tensor has wrong size");
    return;
  }
  
  // Write current player plane
  int current_player_plane = 19;
  int current_player = CurrentPlayer();
  int height = shape[1];
  int width = shape[2];
  
  if (current_player >= 0 && current_player < NumPlayers()) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        values[current_player_plane * height * width + i * width + j] = 
            (current_player == player) ? 1.0 : 0.0;
      }
    }
  }
}

// Create a copy of the state
std::unique_ptr<State> Mali_BaState::Clone() const {
  return std::unique_ptr<State>(new Mali_BaState(*this));
}

// Apply an action to the current state
void Mali_BaState::DoApplyAction(Action action) {
  // Clear the cached legal actions
  cached_legal_actions_ = absl::nullopt;
  
  // Convert action to move
  Move move;
  if (action == kPassActionId) {
    move = kPassMove;
  } else {
    move = ActionToMove(action, board_adapter_);
  }
  
  // Record the current board hash for repetition detection
  uint64_t hash = board_adapter_.HashValue();
  repetitions_[hash]++;
  
  // Apply the move to the board
  board_adapter_.ApplyMove(move);
  
  // Store the move in history
  moves_history_.push_back(move);
}

// Undo the last action
void Mali_BaState::UndoAction(Player player, Action action) {
  // Clear the cached legal actions
  cached_legal_actions_ = absl::nullopt;
  
  if (moves_history_.empty()) {
    // Nothing to undo
    return;
  }
  
  // Remove the last move from history
  moves_history_.pop_back();
  
  // Reset the current board to the starting board
  board_adapter_ = start_board_;
  
  // Clear repetition table
  repetitions_.clear();
  
  // Replay all moves except the last one
  for (const auto& move : moves_history_) {
    uint64_t hash = board_adapter_.HashValue();
    repetitions_[hash]++;
    board_adapter_.ApplyMove(move);
  }
  
  // Reset terminal state flag
  is_terminal_ = false;
}

// Debug string representation
std::string Mali_BaState::Serialize() const {
  std::string mbn = board_adapter_.ToMBN();
  
  // For debugging, check that deserialization works:
  BoardAdapter test_board = BoardAdapter::FromMBN(mbn, board_adapter_.IsUsingFlexBoard());
  if (test_board.ToPlay() != board_adapter_.ToPlay()) {
    // Log an error - mismatch between serialized and deserialized state
    std::cerr << "Warning: Serialization/deserialization mismatch for current player!" << std::endl;
    std::cerr << "Original: " << ColorToString(board_adapter_.ToPlay()) 
              << ", Deserialized: " << ColorToString(test_board.ToPlay()) << std::endl;
  }
  
  return mbn;
}

// Parse a move string to an action
Action Mali_BaState::ParseMoveToAction(const std::string& move_str) const {
  if (move_str == "pass" || move_str == "Pass") {
    return kPassActionId;
  }
  
  // Parse the move LAN format
  // e.g., "1,2,-3:0,1,-1:2,0,-2:post" for a move with trading post
  std::vector<std::string> parts = absl::StrSplit(move_str, ':');
  if (parts.empty()) {
    return kPassActionId;
  }
  
  // Parse the start hex
  std::vector<std::string> start_coords = absl::StrSplit(parts[0], ',');
  if (start_coords.size() != 3) {
    // Invalid format
    return kPassActionId;
  }
  
  int x = std::stoi(start_coords[0]);
  int y = std::stoi(start_coords[1]);
  int z = std::stoi(start_coords[2]);
  HexCoord start_hex(x, y, z);
  
  // Parse the path
  std::vector<HexCoord> path;
  for (size_t i = 1; i < parts.size(); ++i) {
    if (parts[i] == "post") {
      continue;  // Skip the "post" marker for now
    }
    
    std::vector<std::string> coords = absl::StrSplit(parts[i], ',');
    if (coords.size() != 3) {
      continue;  // Skip invalid hex
    }
    
    int px = std::stoi(coords[0]);
    int py = std::stoi(coords[1]);
    int pz = std::stoi(coords[2]);
    path.emplace_back(px, py, pz);
  }
  
  // Check if we should place a trading post
  bool place_post = (parts.size() > 1 && parts.back() == "post");
  
  // Create the move
  Move move{board_adapter_.ToPlay(), start_hex, path, place_post};
  
  // Convert to action
  return MoveToAction(move, board_adapter_.GridRadius());
}

// Converting a move to an action ID
Action MoveToAction(const Move& move, int grid_radius) {
  if (move.is_pass()) {
    return kPassActionId;
  }
  
  // Calculate unique ID for the starting hex
  int num_hexes = 3 * grid_radius * (grid_radius + 1) + 1;
  int start_index = HexToIndex(move.start_hex, grid_radius);
  
  // Hash the path
  size_t path_hash = 0;
  for (const auto& hex : move.path) {
    int hex_index = HexToIndex(hex, grid_radius);
    path_hash = path_hash * 31 + hex_index;
  }
  
  // Create a unique action ID
  int trading_post_bit = move.place_trading_post ? 1 : 0;
  return 1 + start_index + (path_hash % 1000) * num_hexes + 
         trading_post_bit * num_hexes * 1000;
}

// Converting an action ID back to a move
Move ActionToMove(Action action, const BoardAdapter& board) {
  if (action == kPassActionId) {
    return kPassMove;
  }
  
  // Reverse the encoding
  int grid_radius = board.GridRadius();
  action -= 1;
  int num_hexes = 3 * grid_radius * (grid_radius + 1) + 1;
  
  int trading_post_bit = action / (num_hexes * 1000);
  action %= (num_hexes * 1000);
  
  int path_hash = action / num_hexes;
  int start_index = action % num_hexes;
  
  HexCoord start_hex = IndexToHex(start_index, grid_radius);
  
  // Find matching move from legal moves
  std::vector<Move> matching_moves;
  board.GenerateLegalMoves([&](const Move& move) {
    if (move.start_hex == start_hex && 
        move.place_trading_post == (trading_post_bit == 1)) {
      // Calculate path hash
      size_t move_path_hash = 0;
      for (const auto& hex : move.path) {
        int hex_index = HexToIndex(hex, grid_radius);
        move_path_hash = move_path_hash * 31 + hex_index;
      }
      
      if ((move_path_hash % 1000) == path_hash) {
        matching_moves.push_back(move);
      }
    }
    return true;
  });
  
  if (matching_moves.empty()) {
    // Fallback to pass move if no match found (shouldn't happen)
    return kPassMove;
  }
  
  return matching_moves[0];
}

// Overload for backward compatibility
Move ActionToMove(Action action, const Mali_BaBoard& board) {
  if (action == kPassActionId) {
    return kPassMove;
  }
  
  // Create a temporary adapter to use the more generic implementation
  BoardAdapter adapter(board);
  return ActionToMove(action, adapter);
}

}  // namespace mali_ba
}  // namespace open_spiel