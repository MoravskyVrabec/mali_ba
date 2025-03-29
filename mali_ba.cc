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
        {"simplified", GameParameter(true)}
    }};

// Define special action ID for pass moves
constexpr Action kPassActionId = 0;

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

// Player/Color conversions
Player ColorToPlayer(Color color) {
  return static_cast<Player>(color);
}

Color PlayerToColor(Player player) {
  return static_cast<Color>(player);
}

// Mali_BaGame implementation

// Constructor from parameters
Mali_BaGame::Mali_BaGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      grid_radius_(ParameterValue<int>("grid_radius")) {
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

// Implementation of pure virtual methods from Game
int Mali_BaGame::NumDistinctActions() const {
  // Calculate maximum number of possible actions
  // This is an upper bound estimate - can be refined with more analysis
  int num_hexes = 3 * grid_radius_ * (grid_radius_ + 1) + 1;
  return 1 + num_hexes * 1001; // Pass + hex combinations with path hashes
}

int Mali_BaGame::MaxGameLength() const {
  // Estimate based on board size and typical game length
  return 100 + 10 * grid_radius_; 
}

int Mali_BaGame::NumPlayers() const {
  return num_players_;
}

double Mali_BaGame::MinUtility() const {
  return kLossUtility;
}

double Mali_BaGame::MaxUtility() const {
  return kWinUtility;
}

std::vector<int> Mali_BaGame::ObservationTensorShape() const {
  // 20 channels, hexagonal board with radius
  int board_size = 2 * grid_radius_ + 1;
  return {20, board_size, board_size};
}

// Implementation of NewInitialState
std::unique_ptr<State> Mali_BaGame::NewInitialState() const {
  return std::unique_ptr<State>(new Mali_BaState(shared_from_this()));
}

// Deserialize a state from a string
std::unique_ptr<State> Mali_BaGame::DeserializeState(
    const std::string& str) const {
  // For now, assume the serialized string is an MBN notation
  return std::unique_ptr<State>(new Mali_BaState(shared_from_this(), str));
}

// Create the observer for the game
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

// Mali_BaState implementation

// Constructor for a new game
Mali_BaState::Mali_BaState(std::shared_ptr<const Game> game)
    : State(game),
      is_terminal_(false) {
  
  // Initialize board
  start_board_ = BoardAdapter::CreateSimplifiedBoard(true);
  current_board_ = start_board_;
  
  // Initialize with meeples
  InitializeMeeples();
}

// Constructor from an MBN string
Mali_BaState::Mali_BaState(std::shared_ptr<const Game> game,
                         const std::string& mbn)
    : State(game),
      is_terminal_(false) {
  
  // Initialize default board
  start_board_ = BoardAdapter::CreateSimplifiedBoard(true);
  
  // Try to load from MBN
  auto board_opt = BoardAdapter::FromMBN(mbn, true);
  current_board_ = board_opt;  // Use assignment operator
}

// Placeholder implementation for InitializeMeeples
void Mali_BaState::InitializeMeeples() {
  // This would be implemented to set up initial meeples on the board
}

// Get current player
Player Mali_BaState::CurrentPlayer() const {
  if (is_terminal_) {
    return kTerminalPlayerId;
  }
  return ColorToPlayer(current_board_.ToPlay());
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
  current_board_.GenerateLegalMoves([&](const Move& move) {
    actions.push_back(MoveToAction(move, GridRadius()));
    return true;
  });
  
  // Sort the actions in ascending order to meet OpenSpiel requirements
  std::sort(actions.begin(), actions.end());
  
  cached_legal_actions_ = actions;
}

// Convert action to string
std::string Mali_BaState::ActionToString(Player player, Action action) const {
  if (action == kPassActionId) {
    return "Pass";
  }
  
  // Convert action to move and then to string
  Move move = ActionToMove(action, current_board_);
  return move.ToString();
}

// Convert board state to string
std::string Mali_BaState::ToString() const {
  return current_board_.DebugString();
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
    int routes = current_board_.CountConnectedPosts(color);
    if (routes >= 4) {
      // This player wins
      std::vector<double> result(NumPlayers(), kLossUtility);
      result[player] = kWinUtility;
      return result;
    }
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
  
  // Placeholder implementation - in practice, this would be filled in properly
  // by the observer defined in mali_ba_observer.cc
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
    move = ActionToMove(action, current_board_);
  }
  
  // Record the current board hash for repetition detection
  uint64_t hash = current_board_.HashValue();
  repetitions_[hash]++;
  
  // Apply the move to the board
  current_board_.ApplyMove(move);
  
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
  current_board_ = start_board_;
  
  // Clear repetition table
  repetitions_.clear();
  
  // Replay all moves except the last one
  for (const auto& move : moves_history_) {
    uint64_t hash = current_board_.HashValue();
    repetitions_[hash]++;
    current_board_.ApplyMove(move);
  }
  
  // Reset terminal state flag
  is_terminal_ = false;
}

// Serialize the state
std::string Mali_BaState::Serialize() const {
  return current_board_.ToMBN();
}

// Debug string representation
std::string Mali_BaState::DebugString() const {
  return current_board_.DebugString();
}

// Get the grid radius from the game
int Mali_BaState::GridRadius() const {
  auto game = static_cast<const Mali_BaGame*>(GetGame().get());
  return game->GridRadius();
}

// Get number of players
int Mali_BaState::NumPlayers() const {
  auto game = static_cast<const Mali_BaGame*>(GetGame().get());
  return game->NumPlayers();
}

// Utility helper functions
double Mali_BaState::WinUtility() const {
  return kWinUtility;
}

double Mali_BaState::LossUtility() const {
  return kLossUtility;
}

double Mali_BaState::DrawUtility() const {
  return kDrawUtility;
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
  Move move{current_board_.ToPlay(), start_hex, path, place_post};
  
  // Convert to action
  return MoveToAction(move, GridRadius());
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