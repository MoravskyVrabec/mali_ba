#include <iomanip>  // For std::setw
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
#include "open_spiel/games/mali_ba/board_config.h"
#include "open_spiel/games/mali_ba/hex_grid.h"
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
      {"board_config_file", GameParameter("")}
  }
};

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

// Calculate a hash value for the board state
uint64_t CalculateBoardHash(const std::map<HexCoord, Piece>& pieces,
  const std::map<HexCoord, TradePost>& posts,
  Color current_player) {
uint64_t hash = 0;
  
  // Hash pieces
  for (const auto& [hex, piece] : pieces) {
    // Simple hash function
    hash ^= (static_cast<uint64_t>(hex.x) * 73856093) ^
            (static_cast<uint64_t>(hex.y) * 19349663) ^
            (static_cast<uint64_t>(hex.z) * 83492791) ^
            (static_cast<uint64_t>(piece.color) * 12582917) ^
            (static_cast<uint64_t>(piece.type) * 38495729);
  }
  
  // Hash posts
  for (const auto& [hex, post] : posts) {
    hash ^= (static_cast<uint64_t>(hex.x) * 73856093) ^
            (static_cast<uint64_t>(hex.y) * 19349663) ^
            (static_cast<uint64_t>(hex.z) * 83492791) ^
            (static_cast<uint64_t>(post.owner) * 12582917) ^
            (static_cast<uint64_t>(post.type) * 38495729);
  }
  
  // Hash current player
  hash ^= static_cast<uint64_t>(current_player) * 12345678901234567;
  
  return hash;
}

}  // namespace

// Mali_BaGame implementation

// Constructor from parameters
Mali_BaGame::Mali_BaGame(const GameParameters& params)
    : Game(kGameType, params),
      num_players_(ParameterValue<int>("players")),
      grid_radius_(ParameterValue<int>("grid_radius")) {
  
  // Validate parameters
  SPIEL_CHECK_GE(num_players_, 2);
  SPIEL_CHECK_LE(num_players_, 4);
  
  // Load the board configuration from an INI file if specified
  std::string board_config_file = ParameterValue<std::string>("board_config_file", "");
  if (!board_config_file.empty()) {
    board_config_ = BoardConfig::LoadFromFile(board_config_file);
    // Update grid_radius from the config
    grid_radius_ = board_config_.board_radius;
  } else {
    // Use default configuration (regular board)
    board_config_.regular_board = true;
    board_config_.board_radius = grid_radius_;
    board_config_.valid_hexes = BoardConfig::GenerateRegularBoard(grid_radius_);
  }
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
  
  std::unique_ptr<Mali_BaState> state = std::make_unique<Mali_BaState>(shared_from_this());
  
  // Parse the serialized string
  std::vector<std::string> parts = absl::StrSplit(str, '/');
  if (parts.empty()) {
    return state; // Return default state if no parts
  }
  
  // Set current player from first part
  if (parts[0] == "b") {
    state->current_player_ = Color::kBlack;
  } else if (parts[0] == "w") {
    state->current_player_ = Color::kWhite;
  } else if (parts[0] == "r") {
    state->current_player_ = Color::kRed;
  } else if (parts[0] == "u") {
    state->current_player_ = Color::kBlue;
  }
  
  // Parse pieces (format: "x,y,z:color,type")
  if (parts.size() > 1) {
    std::vector<std::string> piece_strs = absl::StrSplit(parts[1], ';');
    for (const std::string& piece_str : piece_strs) {
      if (piece_str.empty()) continue;
      
      std::vector<std::string> piece_parts = absl::StrSplit(piece_str, ':');
      if (piece_parts.size() != 2) continue;
      
      // Parse hex
      std::vector<std::string> hex_parts = absl::StrSplit(piece_parts[0], ',');
      if (hex_parts.size() != 3) continue;
      
      int x = std::stoi(hex_parts[0]);
      int y = std::stoi(hex_parts[1]);
      int z = std::stoi(hex_parts[2]);
      HexCoord hex(x, y, z);
      
      // Parse piece
      std::vector<std::string> piece_data = absl::StrSplit(piece_parts[1], ',');
      if (piece_data.size() != 2) continue;
      
      int color_value = std::stoi(piece_data[0]);
      int type_value = std::stoi(piece_data[1]);
      
      Piece piece;
      piece.color = static_cast<Color>(color_value);
      piece.type = static_cast<MeepleType>(type_value);
      
      state->pieces_[hex] = piece;
    }
  }
  
  // Parse trade posts (format: "x,y,z:owner,type")
  if (parts.size() > 2) {
    std::vector<std::string> post_strs = absl::StrSplit(parts[2], ';');
    for (const std::string& post_str : post_strs) {
      if (post_str.empty()) continue;
      
      std::vector<std::string> post_parts = absl::StrSplit(post_str, ':');
      if (post_parts.size() != 2) continue;
      
      // Parse hex
      std::vector<std::string> hex_parts = absl::StrSplit(post_parts[0], ',');
      if (hex_parts.size() != 3) continue;
      
      int x = std::stoi(hex_parts[0]);
      int y = std::stoi(hex_parts[1]);
      int z = std::stoi(hex_parts[2]);
      HexCoord hex(x, y, z);
      
      // Parse post
      std::vector<std::string> post_data = absl::StrSplit(post_parts[1], ',');
      if (post_data.size() != 2) continue;
      
      int owner_value = std::stoi(post_data[0]);
      int type_value = std::stoi(post_data[1]);
      
      TradePost post;
      post.owner = static_cast<Color>(owner_value);
      post.type = static_cast<TradePostType>(type_value);
      
      state->trade_posts_[hex] = post;
    }
  }
  
  // Parse cities (format: "x,y,z:name:culture:common:rare")
  if (parts.size() > 3) {
    std::vector<std::string> city_strs = absl::StrSplit(parts[3], ';');
    for (const std::string& city_str : city_strs) {
      if (city_str.empty()) continue;
      
      std::vector<std::string> city_parts = absl::StrSplit(city_str, ':');
      if (city_parts.size() != 5) continue;
      
      // Parse hex
      std::vector<std::string> hex_parts = absl::StrSplit(city_parts[0], ',');
      if (hex_parts.size() != 3) continue;
      
      int x = std::stoi(hex_parts[0]);
      int y = std::stoi(hex_parts[1]);
      int z = std::stoi(hex_parts[2]);
      
      City city;
      city.location = HexCoord(x, y, z);
      city.name = city_parts[1];
      city.culture = city_parts[2];
      city.common_good = city_parts[3];
      city.rare_good = city_parts[4];
      
      state->cities_.push_back(city);
    }
  }
  
  // Parse move history (format: "move1;move2;move3")
  if (parts.size() > 4) {
    std::vector<std::string> move_strs = absl::StrSplit(parts[4], ';');
    for (const std::string& move_str : move_strs) {
      if (move_str.empty()) continue;
      
      Action action = state->ParseMoveToAction(move_str);
      state->moves_history_.push_back(ActionToMove(action, state->grid_radius_));
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
  grid_radius_ = mali_ba_game->GridRadius();
  
  // Get the board configuration from the game
  const BoardConfig& config = mali_ba_game->GetBoardConfig();
  
  // Initialize the board using the valid hexes from the configuration
  valid_hexes_ = config.valid_hexes;
  InitializeBoard();
  
  // Initialize with meeples
  InitializeMeeples();
  
  // Store the starting state for undo operations
  StoreInitialState();
}

// Constructor from a serialized string
Mali_BaState::Mali_BaState(std::shared_ptr<const Game> game,
                          const std::string& str)
    : State(game),
      is_terminal_(false) {
  
  auto mali_ba_game = static_cast<const Mali_BaGame*>(game.get());
  grid_radius_ = mali_ba_game->GridRadius();
  
  // Get the board configuration from the game
  const BoardConfig& config = mali_ba_game->GetBoardConfig();
  
  // Initialize the board using the valid hexes from the configuration
  valid_hexes_ = config.valid_hexes;
  
  // The actual parsing happens in DeserializeState
}

// Method to initialize the board
void Mali_BaState::InitializeBoard() {
  // Initialize the board with empty pieces
  for (const auto& hex : valid_hexes_) {
    pieces_[hex] = kEmptyPiece;
    trade_posts_[hex] = {Color::kEmpty, TradePostType::kNone};
  }
  
  // Set the current player to the starting player
  current_player_ = Color::kBlack;
  
  // Initialize cities
  // This is just an example, real cities would be defined based on game rules
  std::vector<City> default_cities = {
    {"Timbuktu", "Songhai", HexCoord(0, 0, 0), "Salt", "Gold"},
    {"Segou", "Bambara", HexCoord(1, -1, 0), "Millet", "Chiwara"},
    {"Ouagadougou", "Mossi", HexCoord(-1, 1, 0), "Horses", "Bronze bracelet"}
  };
  
  cities_ = default_cities;
}

// Method to initialize meeples on the board
void Mali_BaState::InitializeMeeples() {
  // Place meeples randomly or according to game rules
  // Example: Place some basic meeples for testing
  for (const auto& hex : valid_hexes_) {
    // Skip cities or some hexes
    bool is_city = false;
    for (const auto& city : cities_) {
      if (city.location == hex) {
        is_city = true;
        break;
      }
    }
    
    if (!is_city && (rand() % 3 == 0)) {  // Place meeples on 1/3 of non-city hexes
      MeepleType type = static_cast<MeepleType>(1 + rand() % 7);  // Random meeple type 1-7
      Color color = Color::kEmpty;  // Meeples start with no owner
      
      pieces_[hex] = {color, type};
    }
  }
}

// Store the initial state for undo operations
void Mali_BaState::StoreInitialState() {
  start_pieces_ = pieces_;
  start_trade_posts_ = trade_posts_;
}

// Get current player
Player Mali_BaState::CurrentPlayer() const {
  if (is_terminal_) {
    return kTerminalPlayerId;
  }
  return ColorToPlayer(current_player_);
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
  
  // Generate all possible Mancala-style moves
  for (const auto& [start_hex, piece] : pieces_) {
    // Skip hexes that don't contain pieces with the player's color
    // In the future, implement this based on game rules
    
    // For simplicity, we'll generate some basic moves for each hex
    // In a full implementation, you'd use the Mali-Ba rules
    std::vector<HexCoord> neighbors = GetNeighbors(start_hex);
    
    for (const auto& neighbor : neighbors) {
      if (valid_hexes_.count(neighbor) > 0) {
        // Create a move where we pick up from start_hex and drop on neighbor
        Move simple_move{current_player_, start_hex, {neighbor}, false};
        actions.push_back(MoveToAction(simple_move, grid_radius_));
        
        // Also consider placing a trading post
        Move post_move{current_player_, start_hex, {neighbor}, true};
        actions.push_back(MoveToAction(post_move, grid_radius_));
      }
    }
  }
  
  // Sort the actions in ascending order
  std::sort(actions.begin(), actions.end());
  
  // Remove duplicates
  actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
  
  cached_legal_actions_ = actions;
}

// Convert action to string
std::string Mali_BaState::ActionToString(Player player, Action action) const {
  if (action == kPassActionId) {
    return "Pass";
  }
  
  // Convert action to move and then to string
  Move move = ActionToMove(action, grid_radius_);
  return move.ToString();
}

// Convert board state to string
std::string Mali_BaState::ToString() const {
  std::stringstream ss;
  
  ss << "Current Player: " << ColorToString(current_player_) << "\n";
  
  // Print board grid
  int min_x = grid_radius_;
  int max_x = -grid_radius_;
  int min_y = grid_radius_;
  int max_y = -grid_radius_;
  
  // Find board bounds
  for (const auto& hex : valid_hexes_) {
    min_x = std::min(min_x, hex.x);
    max_x = std::max(max_x, hex.x);
    min_y = std::min(min_y, hex.y);
    max_y = std::max(max_y, hex.y);
  }
  
  // Print header
  ss << "   ";
  for (int x = min_x; x <= max_x; ++x) {
    ss << std::setw(3) << x;
  }
  ss << "\n";
  
  // Print rows
  for (int y = min_y; y <= max_y; ++y) {
    ss << std::setw(3) << y;
    
    // Offset for the row
    for (int i = 0; i < y - min_y; ++i) {
      ss << " ";
    }
    
    // Print cells
    for (int x = min_x; x <= max_x; ++x) {
      HexCoord hex(x, y, -x-y);
      if (valid_hexes_.count(hex) > 0) {
        const Piece& piece = pieces_.at(hex);
        const TradePost& post = trade_posts_.at(hex);
        
        if (piece.type != MeepleType::kEmpty) {
          // Show piece type
          char type_char = '.';
          switch (piece.type) {
            case MeepleType::kCommon: type_char = 'c'; break;
            case MeepleType::kRare: type_char = 'r'; break;
            case MeepleType::kYellow: type_char = 'Y'; break;
            case MeepleType::kWhite: type_char = 'W'; break;
            case MeepleType::kBlue: type_char = 'B'; break;
            case MeepleType::kRed: type_char = 'R'; break;
            case MeepleType::kGreen: type_char = 'G'; break;
            default: type_char = '.'; break;
          }
          ss << " " << type_char << " ";
        } else if (post.type != TradePostType::kNone) {
          // Show trading post
          char post_char = post.type == TradePostType::kPost ? 'p' : 'C';
          char owner_char = '.';
          switch (post.owner) {
            case Color::kBlack: owner_char = 'b'; break;
            case Color::kWhite: owner_char = 'w'; break;
            case Color::kRed: owner_char = 'r'; break;
            case Color::kBlue: owner_char = 'u'; break;
            default: owner_char = '.'; break;
          }
          ss << owner_char << post_char << " ";
        } else {
          // Check if it's a city
          bool is_city = false;
          for (const auto& city : cities_) {
            if (city.location == hex) {
              ss << " C ";
              is_city = true;
              break;
            }
          }
          
          if (!is_city) {
            ss << " . ";
          }
        }
      } else {
        ss << "   ";
      }
    }
    ss << "\n";
  }
  
  return ss.str();
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
  
  // Check for win conditions (simplified implementation)
  // In a full implementation, these would follow the specific Mali-Ba rules
  
  // Count trading posts by player
  std::map<Color, int> post_counts;
  for (const auto& [hex, post] : trade_posts_) {
    if (post.type != TradePostType::kNone) {
      post_counts[post.owner]++;
    }
  }
  
  // Check if any player has enough trading posts
  for (const auto& [color, count] : post_counts) {
    if (count >= 4 && color != Color::kEmpty) {
      int player = ColorToPlayer(color);
      std::vector<double> result(NumPlayers(), kLossUtility);
      result[player] = kWinUtility;
      return result;
    }
  }
  
  // No end condition met yet
  return absl::nullopt;
}

// Helper method to apply a Mancala-style move
void Mali_BaState::ApplyMancalaMove(const Move& move) {
  if (move.is_pass()) return;
  
  // Get the starting piece
  Piece starting_piece = pieces_[move.start_hex];
  
  // In Mali-Ba, we pick up all meeples at the starting location
  // and distribute them along the path
  
  // First, clear the starting hex
  pieces_[move.start_hex] = kEmptyPiece;
  
  // If there's no path, nothing more to do
  if (move.path.empty()) return;
  
  // In a real implementation, you would distribute pieces along the path
  // according to Mali-Ba rules. This is a simplified version.
  
  // Place a piece at the last hex in the path
  HexCoord last_hex = move.path.back();
  if (valid_hexes_.count(last_hex) > 0) {
    // In a real game, this would follow Mali-Ba's rules
    // This is a simplified implementation
    pieces_[last_hex] = starting_piece;
  }
}

// Helper method to place a trading post
void Mali_BaState::PlaceTradePost(const Move& move) {
  if (move.is_pass() || move.path.empty()) return;
  
  // Place a trading post at the last hex in the path
  HexCoord last_hex = move.path.back();
  if (valid_hexes_.count(last_hex) > 0) {
    // Check if the hex is empty or already has a post
    TradePost& post = trade_posts_[last_hex];
    
    // In a real game, you would follow Mali-Ba's rules for placing posts
    // This is a simplified implementation
    if (post.type == TradePostType::kNone) {
      post.owner = move.player;
      post.type = TradePostType::kPost;
    } else if (post.type == TradePostType::kPost && post.owner == move.player) {
      // Upgrade to a center if it's already a post owned by the player
      post.type = TradePostType::kCenter;
    }
  }
}

// Helper to get the next player in turn
Color Mali_BaState::GetNextPlayer(Color current) const {
  // Simple implementation for 2-4 players
  switch (current) {
    case Color::kBlack: return Color::kWhite;
    case Color::kWhite: 
      return (NumPlayers() > 2) ? Color::kRed : Color::kBlack;
    case Color::kRed:
      return (NumPlayers() > 3) ? Color::kBlue : Color::kBlack;
    case Color::kBlue:
      return Color::kBlack;
    default:
      return Color::kBlack;
  }
}

// Count connected trading posts for a player
int Mali_BaState::CountConnectedPosts(Color color) const {
  // Find all posts for this player
  std::vector<HexCoord> player_posts;
  for (const auto& [hex, post] : trade_posts_) {
    if (post.owner == color && post.type != TradePostType::kNone) {
      player_posts.push_back(hex);
    }
  }
  
  // Simple implementation: Count each post as a separate route
  // In a full implementation, you would identify connected networks
  return player_posts.size();
}

Move ActionToMove(Action action, int grid_radius) {
  if (action == kPassActionId) {
    return kPassMove;
  }
  
  action -= 1;  // Subtract 1 to reverse the addition in MoveToAction
  
  // Extract the place_trading_post flag
  bool place_post = (action % 2) == 1;
  action /= 2;
  
  // Extract the end hex (simplified)
  int end_index = action % 500;
  action /= 500;
  
  // Extract the start hex
  int start_index = action;
  
  // Convert indices back to hex coordinates
  HexCoord start_hex = IndexToHex(start_index, grid_radius);
  HexCoord end_hex = IndexToHex(end_index, grid_radius);
  
  // Create a move with a path containing just the end hex
  return Move{Color::kEmpty, start_hex, {end_hex}, place_post};
}

// Implementation of MoveToAction function
Action MoveToAction(const Move& move, int grid_radius) {
  if (move.is_pass()) {
    return kPassActionId;
  }
  
  // Encode a move as a unique action ID
  // We use a simple scheme where:
  // - The start hex is encoded in the high bits
  // - The path is encoded in the middle bits
  // - The place_trading_post flag is encoded in the low bit
  
  // First, encode the start hex
  int start_index = HexToIndex(move.start_hex, grid_radius);
  Action action = start_index * 1000;  // Leave room for path encoding
  
  // Encode the path (simplified - for a full implementation, use a better encoding)
  if (!move.path.empty()) {
    int end_index = HexToIndex(move.path.back(), grid_radius);
    action += end_index;
  }
  
  // Encode the place_trading_post flag
  if (move.place_trading_post) {
    action += 1;
  }
  
  return action + 1;  // Add 1 to avoid conflict with kPassActionId
}

// Implementation of UndoAction method
void Mali_BaState::UndoAction(Player player, Action action) {
  // Reset the board to the starting state and replay all actions except the last one
  pieces_ = start_pieces_;
  trade_posts_ = start_trade_posts_;
  current_player_ = Color::kBlack;
  
  // Reset repetitions
  repetitions_.clear();
  
  // Pop the last move
  if (!moves_history_.empty()) {
    moves_history_.pop_back();
  }
  
  // Replay all moves in the history
  for (const auto& move : moves_history_) {
    if (!move.is_pass()) {
      ApplyMancalaMove(move);
      
      if (move.place_trading_post) {
        PlaceTradePost(move);
      }
      
      current_player_ = GetNextPlayer(current_player_);
    } else {
      current_player_ = GetNextPlayer(current_player_);
    }
  }
  
  // Clear cached data
  cached_legal_actions_ = absl::nullopt;
  is_terminal_ = false;
}

// Implementation of Serialize method
std::string Mali_BaState::Serialize() const {
  // Format: player / pieces / posts / cities / moves
  std::stringstream ss;
  
  // Current player
  switch (current_player_) {
    case Color::kBlack: ss << "b"; break;
    case Color::kWhite: ss << "w"; break;
    case Color::kRed: ss << "r"; break;
    case Color::kBlue: ss << "u"; break;
    default: ss << "b"; break;
  }
  ss << "/";
  
  // Pieces (format: "x,y,z:color,type;")
  bool first_piece = true;
  for (const auto& [hex, piece] : pieces_) {
    if (piece.type != MeepleType::kEmpty) {
      if (!first_piece) ss << ";";
      ss << hex.x << "," << hex.y << "," << hex.z << ":"
         << static_cast<int>(piece.color) << ","
         << static_cast<int>(piece.type);
      first_piece = false;
    }
  }
  ss << "/";
  
  // Trade posts (format: "x,y,z:owner,type;")
  bool first_post = true;
  for (const auto& [hex, post] : trade_posts_) {
    if (post.type != TradePostType::kNone) {
      if (!first_post) ss << ";";
      ss << hex.x << "," << hex.y << "," << hex.z << ":"
         << static_cast<int>(post.owner) << ","
         << static_cast<int>(post.type);
      first_post = false;
    }
  }
  ss << "/";
  
  // Cities (format: "x,y,z:name:culture:common:rare;")
  bool first_city = true;
  for (const auto& city : cities_) {
    if (!first_city) ss << ";";
    ss << city.location.x << "," << city.location.y << "," << city.location.z << ":"
       << city.name << ":" << city.culture << ":" << city.common_good << ":" << city.rare_good;
    first_city = false;
  }
  ss << "/";
  
  // Moves (format: "move1;move2;move3")
  bool first_move = true;
  for (const auto& move : moves_history_) {
    if (!first_move) ss << ";";
    ss << move.ToString();
    first_move = false;
  }
  
  return ss.str();
}

// Implementation of ParseMoveToAction method
Action Mali_BaState::ParseMoveToAction(const std::string& move_str) const {
  if (move_str == "Pass") {
    return kPassActionId;
  }
  
  // Split the move string into components
  std::vector<std::string> parts = absl::StrSplit(move_str, ':');
  if (parts.empty()) {
    return kPassActionId;
  }
  
  // Parse the starting hex
  HexCoord start_hex;
  std::string start_str = parts[0];
  // Remove ( and ) from the hex string
  start_str = start_str.substr(1, start_str.length() - 2);
  std::vector<std::string> start_parts = absl::StrSplit(start_str, ',');
  if (start_parts.size() == 3) {
    start_hex = HexCoord(std::stoi(start_parts[0]), std::stoi(start_parts[1]), std::stoi(start_parts[2]));
  }
  
  // Parse the path
  std::vector<HexCoord> path;
  bool place_post = false;
  
  for (size_t i = 1; i < parts.size(); ++i) {
    if (parts[i] == "post") {
      place_post = true;
    } else {
      // Parse hex
      std::string hex_str = parts[i];
      // Remove ( and ) from the hex string
      hex_str = hex_str.substr(1, hex_str.length() - 2);
      std::vector<std::string> hex_parts = absl::StrSplit(hex_str, ',');
      if (hex_parts.size() == 3) {
        path.push_back(HexCoord(std::stoi(hex_parts[0]), std::stoi(hex_parts[1]), std::stoi(hex_parts[2])));
      }
    }
  }
  
  // Create the move and convert to action
  Move move{current_player_, start_hex, path, place_post};
  return MoveToAction(move, grid_radius_);
}

// Implementation of GetBoardConfig method for Mali_BaGame
const BoardConfig& Mali_BaGame::GetBoardConfig() const {
  return board_config_;
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
  
  // Get the tensor shapes from the game
  const auto& shape = ObservationTensorShape();
  int num_planes = shape[0];
  int height = shape[1];
  int width = shape[2];
  
  // Calculate the offset for different planes
  // Planes could be organized as follows:
  // 0-3: Player pieces (4 colors)
  // 4-9: Meeple types (6 types)
  // 10-13: Trading posts (4 players)
  // 14-17: Trading centers (4 players)
  // 18: Cities
  // 19: Current player indicator
  
  // Example: Mark the current player
  int current_player_plane = 19;
  int current_player = CurrentPlayer();
  
  if (current_player >= 0 && current_player < NumPlayers()) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        values[current_player_plane * height * width + i * width + j] = 
            (current_player == player) ? 1.0 : 0.0;
      }
    }
  }
  
  // Map pieces, posts, and cities to the tensor
  // This is a simplified implementation
  // In a full implementation, you would map all game state to the tensor planes
  
  // Calculate tensor indices for each hex
  std::map<HexCoord, std::pair<int, int>> hex_indices;
  for (const auto& hex : valid_hexes_) {
    // Convert hex to offset coordinates
    auto [col, row] = CubeToOffset(hex);
    
    // Adjust to tensor indices (centered)
    int adjusted_col = col + grid_radius_;
    int adjusted_row = row + grid_radius_;
    
    // Skip if out of bounds of the tensor
    if (adjusted_row >= 0 && adjusted_row < height && 
        adjusted_col >= 0 && adjusted_col < width) {
      hex_indices[hex] = {adjusted_row, adjusted_col};
    }
  }
  
  // Fill in pieces
  for (const auto& [hex, piece] : pieces_) {
    if (hex_indices.count(hex) > 0 && piece.type != MeepleType::kEmpty) {
      auto [row, col] = hex_indices[hex];
      
      // Set piece type plane
      int type_plane = static_cast<int>(piece.type) + 3;  // Adjust offset as needed
      values[type_plane * height * width + row * width + col] = 1.0;
      
      // Set piece color plane if it has an owner
      if (piece.color != Color::kEmpty) {
        int color_plane = static_cast<int>(piece.color);
        values[color_plane * height * width + row * width + col] = 1.0;
      }
    }
  }
  
  // Fill in trading posts
  for (const auto& [hex, post] : trade_posts_) {
    if (hex_indices.count(hex) > 0 && post.type != TradePostType::kNone) {
      auto [row, col] = hex_indices[hex];
      
      // Set post type plane
      int post_plane = (post.type == TradePostType::kPost) ? 10 : 14;
      post_plane += static_cast<int>(post.owner);
      values[post_plane * height * width + row * width + col] = 1.0;
    }
  }
  
  // Fill in cities
  for (const auto& city : cities_) {
    if (hex_indices.count(city.location) > 0) {
      auto [row, col] = hex_indices[city.location];
      
      // Set city plane
      int city_plane = 18;
      values[city_plane * height * width + row * width + col] = 1.0;
    }
  }
}

// Create a copy of the state
std::unique_ptr<State> Mali_BaState::Clone() const {
  return std::make_unique<Mali_BaState>(*this);
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
    move = ActionToMove(action, grid_radius_);
  }
  
  // Record the current board hash for repetition detection
  uint64_t hash = CalculateBoardHash(pieces_, trade_posts_, current_player_);
  repetitions_[hash]++;
  
  // Apply the move
  if (!move.is_pass()) {
    // Move pieces according to Mali-Ba's Mancala-style movement
    ApplyMancalaMove(move);
    
    // Place a trading post if required
    if (move.place_trading_post) {
      PlaceTradePost(move);
    }
    
    // Switch to the next player
    current_player_ = GetNextPlayer(current_player_);
  } else {
    // Just switch to the next player for pass moves
    current_player_ = GetNextPlayer(current_player_);
  }
  
  // Store the move in history
  moves_history_.push_back(move);
}
}  // namespace mali_ba
}  // namespace open_spiel