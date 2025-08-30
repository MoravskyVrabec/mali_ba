Excellent question. You're right, I did mention that. Refactoring the `BoardConfig` logic entirely into the `Mali_BaGame` class is a significant architectural improvement that aligns better with the OpenSpiel design philosophy.

Here is a high-level outline of what that refactoring would look like, focusing on the concepts and data flow rather than specific code.

### The Core Philosophy

The fundamental principle of OpenSpiel's design is:

*   **`Game` Object:** Represents the *rules and constants* of a specific game instance. It is the factory for states. It is **static** and **shared** across all states. Things like the board shape, number of players, and victory conditions belong here.
*   **`State` Object:** Represents a *specific moment* in a single playthrough of the game. It is **dynamic** and changes with every move. It should be as lightweight as possible because it gets cloned frequently by algorithms like MCTS.

Your current design has the `State` holding the `BoardConfig`, which contains static information. This means every time a `State` is cloned, you're unnecessarily copying the entire board and city configuration.

The refactor moves all this static configuration into the `Game` object, making it the single source of truth. The `State` then simply asks its parent `Game` object for this information when needed.

---

### What Moves Where?

**Responsibilities moving FROM `Mali_BaState` TO `Mali_BaGame`:**

1.  **Board Shape:**
    *   `valid_hexes_` (the `std::set<HexCoord>`)
    *   `grid_radius_`
    *   Logic for parsing/generating the board shape.
2.  **City Configuration:**
    *   `cities_` (the `std::vector<City>`)
    *   Logic for parsing/generating city layouts.
3.  **Lookup Tables (LUTs):**
    *   The maps for converting `HexCoord` to a sequential index and back. This is derived from the static board shape and should only be calculated once per game instance.
4.  **Configuration Loading:**
    *   All file I/O and parsing logic for the `.ini` file. The `BoardConfig` class itself would be completely eliminated, its logic absorbed by the `Mali_BaGame` constructor.

**What remains in `Mali_BaState`:**

*   Dynamic game data:
    *   `current_phase_`
    *   `current_player_id_`
    *   `player_token_locations_`
    *   `hex_meeples_`
    *   `trade_posts_locations_`
    *   `common_goods_` and `rare_goods_`
    *   `trade_routes_`
    *   `moves_history_` and `undo_stack_`

---

### The Refactoring Steps (An Outline)

**Step 1: Enhance `Mali_BaGame`**

*   **Add Data Members:** Add `valid_hexes_`, `cities_`, `grid_radius_`, `rules_`, and the coordinate-to-index lookup maps directly to the private members of the `Mali_BaGame` class.
*   **Move Loading Logic:** Move all the `.ini` parsing logic from `BoardConfig::LoadFromIniFile` into the `Mali_BaGame` constructor. The constructor will now be responsible for:
    *   Reading the `config_file` parameter.
    *   Opening and parsing the `.ini` file.
    *   Populating its own `valid_hexes_`, `cities_`, and `rules_` members based on the file.
    *   If no file is provided, it generates the default board and cities.
    *   Initializing the coordinate lookup tables once all hexes are determined.
*   **Add Public Getters:** Create public const getters in `Mali_BaGame` like `GetValidHexes()`, `GetCities()`, `GetGridRadius()`, `GetRules()`, `CoordToIndex()`, etc., so states and observers can access this data.

**Step 2: Simplify `Mali_BaState`**

*   **Remove Data Members:** Delete `board_config_`, `grid_radius_`, `valid_hexes_`, and `cities_` from `Mali_BaState`.
*   **Simplify Constructor:** The state's constructor becomes much simpler. It no longer needs to load or apply any board configuration. It just initializes its dynamic members (e.g., resizing goods vectors).

**Step 3: Connect the `State` back to the `Game`**

*   The `State` class already has a `std::shared_ptr<const Game> game_` member. We need a convenient, type-safe way to access the specific `Mali_BaGame` methods.
*   **Create a Helper:** Add a private helper method in `Mali_BaState`:
    ```cpp
    const Mali_BaGame* Mali_BaState::GetGame() const {
      return static_cast<const Mali_BaGame*>(game_.get());
    }
    ```

**Step 4: Update the Logic in `Mali_BaState`**

*   Go through all methods in `mali_ba_state_*.cc` files.
*   Wherever you see a reference to a member that was removed (like `valid_hexes_` or `cities_`), replace it with a call to the parent `Game` object via the new helper.
    *   `for (const auto& hex : valid_hexes_)` becomes `for (const auto& hex : GetGame()->GetValidHexes())`
    *   `if (city.location == hex)` becomes `if (GetGame()->GetCityAt(hex))`
    *   `grid_radius_` becomes `GetGame()->GetGridRadius()`
*   The `ApplyChanceSetup` method, for example, would now get the list of hexes from `GetGame()->GetValidHexes()` to populate its local `hex_meeples_` map.

**Step 5: Eliminate `BoardConfig`**

*   Once all its logic and data have been moved into `Mali_BaGame`, you can delete `mali_ba_board_config.h` and `mali_ba_board_config.cc` entirely.

**Step 6: Update Tests**

*   Your test fixture `MaliBaTest` and other tests might directly access members like `mali_ba_state->ValidHexes()`. These calls will now fail.
*   Update them to access the data through the game object, which the test fixture already holds: `static_cast<const Mali_BaGame*>(game.get())->GetValidHexes()`.
*   Alternatively, and more cleanly, you can keep public methods like `ValidHexes()` on the `Mali_BaState` class, but have them simply delegate the call:
    ```cpp
    // In mali_ba_state.h
    const std::set<HexCoord>& ValidHexes() const { return GetGame()->GetValidHexes(); }
    ```
    This way, the public API for tests and observers remains largely the same.

### Benefits of This Refactor

1.  **Efficiency:** `State` objects become much smaller. `state->Clone()` is now significantly faster, which is critical for algorithms that do thousands or millions of clones (like MCTS).
2.  **Correctness:** This design properly separates static game-instance data from dynamic turn-by-turn data, adhering to OpenSpiel's intended architecture.
3.  **Maintainability:** The logic for setting up a game is consolidated in one place (`Mali_BaGame` constructor) instead of being split across multiple classes. This makes debugging and future changes much easier.

===================================================================================================
Of course. This is an excellent refactoring that will significantly improve the design and efficiency of your game by aligning it with OpenSpiel's architecture.

The core idea is to move all static configuration (board shape, cities, rules) from the `State` object into the `Game` object. The `State` then becomes much lighter and simply queries its parent `Game` object for this constant information.

Here are the complete code changes for this refactoring.

### Summary of Changes:

1.  **`mali_ba_board_config.h` & `mali_ba_board_config.cc`**: These files will be **deleted**. Their logic is being moved into `Mali_BaGame`.
2.  **`mali_ba_game.h`**: Will be expanded to hold the board/city configuration and the INI parsing logic.
3.  **`mali_ba_game.cc`**: The constructor will now perform the INI loading.
4.  **`mali_ba_state.h`**: Will be simplified, removing the board configuration members.
5.  **`mali_ba_state_*.cc`**: All methods will be updated to get configuration from `GetGame()->...` instead of local members.
6.  **`mali_ba_test.cc`**: The `IniFileConfigTest` will be updated to verify the new architecture.

---

### Step 1: Delete Obsolete Files

First, delete these two files from your project and build system:

*   `mali_ba_board_config.h`
*   `mali_ba_board_config.cc`

---

### Step 2: Updated `mali_ba_game.h`

This file now declares the configuration members and parsing helpers.

```cpp
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

#ifndef OPEN_SPIEL_GAMES_MALI_BA_GAME_H_
#define OPEN_SPIEL_GAMES_MALI_BA_GAME_H_

#include <memory>
#include <vector>
#include <string>
#include <random>
#include <set>
#include <map>

#include "open_spiel/spiel.h"
#include "open_spiel/observer.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/container/flat_hash_map.h"
#include "open_spiel/games/mali_ba/mali_ba_common.h"

namespace open_spiel
{
  namespace mali_ba
  {

    // Forward declaration
    class Mali_BaState;

    /**
     * Game class for Mali-Ba - a mancala-style game about West African trade routes.
     * This class handles game creation, parameters, and initial state setup.
     */
    class Mali_BaGame : public Game
    {
    public:
      // Constructor requires GameParameters
      explicit Mali_BaGame(const GameParameters &params);

      // DEBUG -- ADD THIS OVERRIDE DECLARATION:
      std::unique_ptr<State> NewInitialStateForPopulation(int population) const override;

      // --- OpenSpiel Game API Overrides ---
      int MaxChanceOutcomes() const override; 
      std::unique_ptr<open_spiel::State> NewInitialState() const override;
      std::unique_ptr<State> NewInitialState(const std::string &str) const override;
      std::vector<int> ObservationTensorShape() const override;
      std::unique_ptr<State> DeserializeState(const std::string &str) const override;
      std::shared_ptr<Observer> MakeObserver(
          absl::optional<IIGObservationType> iig_obs_type,
          const GameParameters &params) const override;

      int NumDistinctActions() const override { return mali_ba::NumDistinctActions(); }
      int NumPlayers() const override { return num_players_; }
      double MinUtility() const override { return LossUtility(); }
      absl::optional<double> UtilitySum() const override { return absl::nullopt; }
      double MaxUtility() const override { return WinUtility(); }
      int MaxGameLength() const override { return mali_ba::MaxGameLength(); }
      // --- End OpenSpiel API ---

      // --- Mali-Ba Specific Accessors ---
      const std::vector<PlayerColor> &GetPlayerColors() const { return player_colors_; }
      uint_fast64_t GetRNGSeed() const { return rng_seed_; }
      int GetTokensPerPlayer() const { return tokens_per_player_; }
      const GameRules& GetRules() const { return rules_; }

      // Board configuration accessors
      const std::set<HexCoord>& GetValidHexes() const { return valid_hexes_; }
      const std::vector<City>& GetCities() const { return cities_; }
      const City* GetCityAt(const HexCoord &location) const;
      int GetGridRadius() const { return grid_radius_; }
      
      // Board lookup-table (LUT) Get the total number of valid hexes on the board.
      int NumHexes() const { return num_hexes_; }
      // Board lookup-table (LUT) Get the sequential index (0 to NumHexes-1) for a valid HexCoord.
      // Returns -1 if the hex is not a valid hex on the board.
      int CoordToIndex(const HexCoord &hex) const;
      // Board lookup-table (LUT) Get the HexCoord corresponding to a sequential index.
      // Asserts that the index is valid (0 <= index < NumHexes).
      HexCoord IndexToCoord(int index) const;

    private:
      // --- Helper methods for INI parsing (moved from BoardConfig) ---
      void ParseCitiesFromIni(const std::map<std::string, std::string>& cities_section);
      std::set<HexCoord> ParseHexList(const std::string& hex_string) const;
      HexCoord ParseHexCoord(const std::string& coord_str) const;
      int FindCityIdByName(const std::string& name) const;
      std::string StripWhitespace(const std::string& str) const;
      std::vector<std::string> SplitString(const std::string& str, char delimiter) const;

      // --- Helper methods for board generation (moved from BoardConfig) ---
      std::set<HexCoord> GenerateRegularBoard(int radius) const;
      std::vector<City> GetDefaultCitiesWithTimbuktu() const;

      // Helper to initialize LUTs during construction
      void InitializeLookups();

      // --- Game Configuration (read from parameters and INI) ---
      GameRules rules_;
      int num_players_;
      int grid_radius_;
      int tokens_per_player_;
      bool enable_move_logging_;
      std::mt19937::result_type rng_seed_;

      // --- Player Configuration ---
      std::vector<PlayerColor> player_colors_; // Order matters (player 0, 1, ...)

      // --- Board Configuration (now lives here) ---
      std::set<HexCoord> valid_hexes_;
      std::vector<City> cities_;
      int num_hexes_ = 0; // Total number of valid hexes on the board
      
      // --- Board Hexes Lookup Tables (LUTs) ---
      absl::flat_hash_map<HexCoord, int> coord_to_index_map_;
      std::vector<HexCoord> index_to_coord_vec_;
    };

  } // namespace mali_ba
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_MALI_BA_GAME_H_

```

### Step 3: Updated `mali_ba_game.cc`

This file absorbs all the INI parsing logic into its constructor.

```cpp
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

#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/hex_grid.h"
#include "open_spiel/games/mali_ba/mali_ba_observer.h"
// #include "open_spiel/games/mali_ba/mali_ba_board_config.h" // REMOVED

#include <memory>
#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <chrono>
#include <string>
#include <stdexcept>
#include <map>
#include <algorithm> // For std::transform
#include <cctype>    // For std::tolower

#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/str_join.h"
#include "open_spiel/abseil-cpp/absl/strings/ascii.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/game_parameters.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "json.hpp"

using json = nlohmann::json;
static std::ios_base::Init g_ios_init;

namespace open_spiel
{
    namespace mali_ba
    {
        // ... (logging and ostream definitions remain the same) ...
        bool g_mali_ba_logging_enabled = true;
        LogLevel g_mali_ba_log_level = LogLevel::kInfo;
        std::string g_log_file_path = "/tmp/mali_ba.log";

        std::string LogLevelToString(LogLevel level) {
            switch (level) {
                case LogLevel::kDebug:   return "DEBUG";
                case LogLevel::kInfo:    return "INFO ";
                case LogLevel::kWarning: return "WARN ";
                case LogLevel::kError:   return "ERROR";
                default:                 return "UNKWN";
            }
        }
        
        void LogMBCore(LogLevel level, const std::string& message, bool print_to_terminal,
                    const char* file, int line) {
            if (!g_mali_ba_logging_enabled || level < g_mali_ba_log_level) {
                return;
            }

            auto now = std::chrono::system_clock::now();
            auto in_time_t = std::chrono::system_clock::to_time_t(now);
            
            std::string file_str(file);
            size_t last_slash = file_str.find_last_of('/');
            if (last_slash != std::string::npos) {
                file_str = file_str.substr(last_slash + 1);
            }

            std::stringstream log_stream;
            log_stream << std::put_time(std::localtime(&in_time_t), "%H:%M:%S") << " "
                    << "[" << LogLevelToString(level) << "] "
                    << "[" << file_str << ":" << line << "] "
                    << message;

            if (print_to_terminal) {
                fprintf(stderr, "%s\n", log_stream.str().c_str());
            }
            std::ofstream log_file(g_log_file_path, std::ios_base::app);
            if (log_file.is_open()) {
                log_file << log_stream.str() << std::endl;
            }
        }

        std::ostream& operator<<(std::ostream& os, const Phase& phase) {
            switch (phase) {
                case Phase::kEmpty:     os << "Empty"; break;
                case Phase::kSetup:     os << "Setup"; break;
                case Phase::kPlaceToken:os << "PlaceToken"; break;
                case Phase::kPlay:      os << "Play"; break;
                case Phase::kEndRound:  os << "EndRound"; break;
                case Phase::kGameOver:  os << "GameOver"; break;
                default: os << "UnknownPhase(" << static_cast<int>(phase) << ")";
            }
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const ActionType& move_type) {
            switch (move_type) {
                case ActionType::kInvalid:          os << "Invalid"; break;
                case ActionType::kPass:             os << "Pass"; break;
                case ActionType::kChanceSetup:      os << "ChanceSetup"; break;
                case ActionType::kPlaceToken:       os << "PlaceToken"; break;
                case ActionType::kMancala:          os << "Mancala"; break;
                case ActionType::kPlaceTCenter:     os << "PlaceTCenter"; break;
                case ActionType::kIncome:           os << "Income"; break;
                case ActionType::kTradeRouteCreate: os << "TradeRouteCreate"; break;
                case ActionType::kTradeRouteUpdate: os << "TradeRouteUpdate"; break;
                case ActionType::kTradeRouteDelete: os << "TradeRouteDelete"; break;
                default: os << "UnknownActionType(" << static_cast<int>(move_type) << ")";
            }
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const PlayerColor& pc) { os << PlayerColorToString(pc); return os; }
        std::ostream& operator<<(std::ostream& os, const MeepleColor& mc) { os << MeepleColorToString(mc); return os; }
        std::ostream& operator<<(std::ostream& os, const TradePostType& tpt) {
            switch (tpt) {
                case TradePostType::kNone:   os << "None"; break;
                case TradePostType::kPost:   os << "Post"; break;
                case TradePostType::kCenter: os << "Center"; break;
                default: os << "UnknownTradePostType";
            }
            return os;
        }

        const GameType kGameType{
            /*short_name=*/"mali_ba",
            /*long_name=*/"Mali-Ba Game",
            GameType::Dynamics::kSequential,
            GameType::ChanceMode::kExplicitStochastic,
            GameType::Information::kPerfectInformation,
            GameType::Utility::kGeneralSum,
            GameType::RewardModel::kTerminal,
            /*max_num_players=*/5,
            /*min_num_players=*/2,
            /*provides_information_state_string=*/true,
            /*provides_information_state_tensor=*/false,
            /*provides_observation_string=*/true,
            /*provides_observation_tensor=*/true,
            /*parameter_specification=*/{
                {"players", GameParameter(3)},
                {"grid_radius", GameParameter(5)},
                {"tokens_per_player", GameParameter(3)},
                {"enable_move_logging", GameParameter(false)},
                {"rng_seed", GameParameter(-1)},
                {"config_file", GameParameter(std::string(""))},
                // Rules parameters
                {"posts_per_player", GameParameter(6)},
                {"free_action_trade_routes", GameParameter(true)},
                {"endgm_cond_numroutes", GameParameter(4)},
                {"endgm_cond_numrare_goods", GameParameter(4)},
                {"upgrade_cost_common", GameParameter(3)},
                {"upgrade_cost_rare", GameParameter(1)},
                {"mancala_capture_rule", GameParameter(std::string("none"))}
            }
        };

        std::shared_ptr<const Game> Factory(const GameParameters &params) {
            return std::make_shared<Mali_BaGame>(params);
        }
        
        REGISTER_SPIEL_GAME(kGameType, Factory);

        Mali_BaGame::Mali_BaGame(const GameParameters &params)
            : Game(kGameType, params),
              rules_(), // Initialize with defaults
              num_players_(ParameterValue<int>("players")),
              grid_radius_(ParameterValue<int>("grid_radius")),              
              tokens_per_player_(ParameterValue<int>("tokens_per_player")),
              enable_move_logging_(ParameterValue<bool>("enable_move_logging")),
              rng_seed_(ParameterValue<int>("rng_seed", -1) == -1
                            ? static_cast<std::mt19937::result_type>(
                                std::chrono::high_resolution_clock::now().time_since_epoch().count())
                            : static_cast<std::mt19937::result_type>(ParameterValue<int>("rng_seed")))
        {
            // === 1. Load configuration from INI file if provided ===
            std::string config_file = ParameterValue<std::string>("config_file");
            bool custom_board = false;
            bool custom_cities = false;

            if (!config_file.empty()) {
                LOG_INFO("Mali_BaGame: Attempting to load config from: ", config_file);
                std::ifstream file(config_file);
                if (file.is_open()) {
                    std::string line;
                    std::string current_section;
                    std::map<std::string, std::string> board_section, cities_section, rules_section;

                    while (std::getline(file, line)) {
                        line = StripWhitespace(line);
                        if (line.empty() || line[0] == '#' || line[0] == ';') continue;
                        if (line[0] == '[' && line.back() == ']') {
                            current_section = line.substr(1, line.length() - 2);
                            continue;
                        }
                        auto eq_pos = line.find('=');
                        if (eq_pos != std::string::npos) {
                            std::string key = StripWhitespace(line.substr(0, eq_pos));
                            std::string value = StripWhitespace(line.substr(eq_pos + 1));
                            if (current_section == "Board") board_section[key] = value;
                            else if (current_section == "Cities") cities_section[key] = value;
                            else if (current_section == "Rules") rules_section[key] = value;
                        }
                    }

                    // Process Board section
                    if (board_section.count("grid_radius")) grid_radius_ = std::stoi(board_section["grid_radius"]);
                    if (board_section.count("custom_hexes")) {
                        valid_hexes_ = ParseHexList(board_section["custom_hexes"]);
                        custom_board = !valid_hexes_.empty();
                    }

                    // Process Cities section
                    if (!cities_section.empty()) {
                        ParseCitiesFromIni(cities_section);
                        custom_cities = !cities_.empty();
                    }

                    // Process Rules section
                    if (!rules_section.empty()) {
                        try {
                            if (rules_section.count("free_action_trade_routes")) {
                                std::string val = rules_section.at("free_action_trade_routes");
                                std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                                rules_.free_action_trade_routes = (val == "true" || val == "1");
                            }
                            if (rules_section.count("endgm_cond_numroutes")) rules_.endgm_cond_numroutes = std::stoi(rules_section.at("endgm_cond_numroutes"));
                            if (rules_section.count("endgm_cond_numrare_goods")) rules_.endgm_cond_numrare_goods = std::stoi(rules_section.at("endgm_cond_numrare_goods"));
                            if (rules_section.count("upgrade_cost_common")) rules_.upgrade_cost_common = std::stoi(rules_section.at("upgrade_cost_common"));
                            if (rules_section.count("upgrade_cost_rare")) rules_.upgrade_cost_rare = std::stoi(rules_section.at("upgrade_cost_rare"));
                            if (rules_section.count("posts_per_player")) rules_.posts_per_player = std::stoi(rules_section.at("posts_per_player"));
                            if (rules_section.count("mancala_capture_rule")) rules_.mancala_capture_rule = rules_section.at("mancala_capture_rule");
                        } catch (const std::exception& e) {
                             LOG_ERROR("Could not parse a rule from [Rules] section: ", e.what());
                        }
                    }
                } else {
                     LOG_WARN("Could not open config file: ", config_file);
                }
            }

            // === 2. Apply Defaults if INI did not provide custom configs ===
            if (!custom_board) {
                valid_hexes_ = GenerateRegularBoard(grid_radius_);
            }
            if (!custom_cities) {
                cities_ = GetDefaultCitiesWithTimbuktu();
            }

            // === 3. Apply parameter overrides (highest priority) ===
            rules_.free_action_trade_routes = ParameterValue<bool>("free_action_trade_routes", rules_.free_action_trade_routes);
            rules_.posts_per_player = ParameterValue<int>("posts_per_player", rules_.posts_per_player);
            rules_.upgrade_cost_common = ParameterValue<int>("upgrade_cost_common", rules_.upgrade_cost_common);
            rules_.upgrade_cost_rare = ParameterValue<int>("upgrade_cost_rare", rules_.upgrade_cost_rare);
            rules_.endgm_cond_numroutes = ParameterValue<int>("endgm_cond_numroutes", rules_.endgm_cond_numroutes);
            rules_.endgm_cond_numrare_goods = ParameterValue<int>("endgm_cond_numrare_goods", rules_.endgm_cond_numrare_goods);
            rules_.mancala_capture_rule = ParameterValue<std::string>("mancala_capture_rule", rules_.mancala_capture_rule);
            
            // === 4. Set player colors ===
            if (num_players_ > 0) player_colors_.push_back(PlayerColor::kRed);
            if (num_players_ > 1) player_colors_.push_back(PlayerColor::kGreen);
            if (num_players_ > 2) player_colors_.push_back(PlayerColor::kBlue);
            if (num_players_ > 3) player_colors_.push_back(PlayerColor::kViolet);
            if (num_players_ > 4) player_colors_.push_back(PlayerColor::kPink);

            LOG_INFO("Mali_BaGame: Final configuration loaded:");
            LOG_INFO("  - Grid Radius: ", grid_radius_);
            LOG_INFO("  - Valid Hexes: ", valid_hexes_.size());
            LOG_INFO("  - Cities: ", cities_.size());
            LOG_INFO("  - Rules loaded (see below)");
            LOG_INFO("    - Free Action Trade Routes: ", rules_.free_action_trade_routes);
            LOG_INFO("    - End Game Num Routes: ", rules_.endgm_cond_numroutes);
            LOG_INFO("    - End Game Num Rare Goods: ", rules_.endgm_cond_numrare_goods);
            LOG_INFO("    - Upgrade Cost (Common/Rare): ", rules_.upgrade_cost_common, "/", rules_.upgrade_cost_rare);
            LOG_INFO("    - Posts Per Player: ", rules_.posts_per_player);
            LOG_INFO("    - Mancala Capture Rule: ", rules_.mancala_capture_rule);

            // === 5. Initialize final lookups ===
            InitializeLookups(); 
        }

        // ... (rest of mali_ba_game.cc methods: NewInitialState, etc. remain the same) ...
        std::unique_ptr<State> Mali_BaGame::NewInitialStateForPopulation(int population) const {
            return NewInitialState();
        }

        int Mali_BaGame::MaxChanceOutcomes() const { return 1; }

        std::unique_ptr<State> Mali_BaGame::NewInitialState() const {
            auto state = std::make_unique<Mali_BaState>(shared_from_this());
            if (enable_move_logging_) {
                state->InitializeMoveLogging();
            }
            return state;
        }

        std::unique_ptr<State> Mali_BaGame::NewInitialState(const std::string &str) const {
            return DeserializeState(str);
        }
        std::vector<int> Mali_BaGame::ObservationTensorShape() const {
            return mali_ba::ObservationTensorShape();
        }
        
        std::shared_ptr<Observer> Mali_BaGame::MakeObserver(
            absl::optional<IIGObservationType> iig_obs_type,
            const GameParameters &params) const
        {
            return MakeMaliBaObserver(iig_obs_type.value_or(IIGObservationType{}));
        }

        const City *Mali_BaGame::GetCityAt(const HexCoord &location) const {
            for (const auto& city : cities_) if (city.location == location) return &city;
            return nullptr;
        }

        void Mali_BaGame::InitializeLookups() { 
            coord_to_index_map_.clear();
            index_to_coord_vec_.clear();
            int current_index = 0;
            for (const HexCoord &hex : valid_hexes_) {
                coord_to_index_map_[hex] = current_index;
                index_to_coord_vec_.push_back(hex);
                current_index++;
            }
            num_hexes_ = current_index;
        }

        int Mali_BaGame::CoordToIndex(const HexCoord &hex) const { 
            auto it = coord_to_index_map_.find(hex);
            return (it != coord_to_index_map_.end()) ? it->second : -1;
        }

        HexCoord Mali_BaGame::IndexToCoord(int index) const {
            SPIEL_CHECK_GE(index, 0);
            SPIEL_CHECK_LT(index, num_hexes_);
            return index_to_coord_vec_[index];
        }

        // --- Parsing helpers moved from BoardConfig ---
        std::string Mali_BaGame::StripWhitespace(const std::string& str) const {
            size_t start = str.find_first_not_of(" \t\r\n");
            if (start == std::string::npos) return "";
            size_t end = str.find_last_not_of(" \t\r\n");
            return str.substr(start, end - start + 1);
        }

        std::vector<std::string> Mali_BaGame::SplitString(const std::string& str, char delimiter) const {
            std::vector<std::string> result;
            std::stringstream ss(str);
            std::string item;
            while (std::getline(ss, item, delimiter)) {
                if (!item.empty()) result.push_back(item);
            }
            return result;
        }

        HexCoord Mali_BaGame::ParseHexCoord(const std::string& coord_str) const {
            std::vector<std::string> parts = SplitString(coord_str, ',');
            if (parts.size() != 3) {
                throw std::invalid_argument("Invalid coordinate format: " + coord_str);
            }
            int x = std::stoi(StripWhitespace(parts[0]));
            int y = std::stoi(StripWhitespace(parts[1]));
            int z = std::stoi(StripWhitespace(parts[2]));
            return HexCoord(x, y, z);
        }
        
        std::set<HexCoord> Mali_BaGame::ParseHexList(const std::string& hex_string) const {
            std::set<HexCoord> hexes;
            std::string processed = hex_string;
            std::replace(processed.begin(), processed.end(), ';', ' ');
            std::vector<std::string> coords_list = SplitString(processed, ' ');
            for (const auto& coord_str : coords_list) {
                std::string trimmed = StripWhitespace(coord_str);
                if (trimmed.empty()) continue;
                try {
                    HexCoord hex = ParseHexCoord(trimmed);
                    if (hex.x + hex.y + hex.z == 0) {
                        hexes.insert(hex);
                    } else {
                        LOG_WARN("Invalid hex coordinate ", trimmed, " (x+y+z != 0)");
                    }
                } catch (const std::exception& e) {
                    LOG_ERROR("Could not parse hex coordinate: ", trimmed);
                }
            }
            return hexes;
        }

        int Mali_BaGame::FindCityIdByName(const std::string& name) const {
            for (const auto& [id, details] : kCityDetailsMap) {
                if (StrLower(details.name) == StrLower(name)) {
                    return id;
                }
            }
            return -1;
        }
        
        void Mali_BaGame::ParseCitiesFromIni(const std::map<std::string, std::string>& cities_section) {
            cities_.clear();
            for (const auto& [key, value] : cities_section) {
                if (key.rfind("city", 0) == 0) {
                    try {
                        std::vector<std::string> parts = SplitString(value, ',');
                        if (parts.size() >= 4) {
                            std::string name = StripWhitespace(parts[0]);
                            int city_id = FindCityIdByName(name);
                            if (city_id != -1) {
                                int x = std::stoi(StripWhitespace(parts[1]));
                                int y = std::stoi(StripWhitespace(parts[2]));
                                int z = std::stoi(StripWhitespace(parts[3]));
                                const auto& details = kCityDetailsMap.at(city_id);
                                cities_.emplace_back(details.id, details.name, details.culture, 
                                                       HexCoord(x, y, z), details.common_good, details.rare_good);
                            } else {
                                LOG_WARN("City '", name, "' not found in city database.");
                            }
                        }
                    } catch (const std::exception& e) {
                        LOG_ERROR("Could not parse city ", key, ": ", value, " - ", e.what());
                    }
                }
            }
        }
        
        // --- Generation helpers moved from BoardConfig ---
        std::set<HexCoord> Mali_BaGame::GenerateRegularBoard(int radius) const {
            std::set<HexCoord> hexes;
            for (int q = -radius; q <= radius; ++q) {
                int r_min = std::max(-radius, -q - radius);
                int r_max = std::min(radius, -q + radius);
                for (int r = r_min; r <= r_max; ++r) {
                    hexes.emplace(q, r, -q - r);
                }
            }
            return hexes;
        }
        
        std::vector<City> Mali_BaGame::GetDefaultCitiesWithTimbuktu() const {
            std::vector<City> cities;
            auto timbuktu_it = kCityDetailsMap.find(15);
            if (timbuktu_it != kCityDetailsMap.end()) {
                const auto& details = timbuktu_it->second;
                cities.emplace_back(details.id, details.name, details.culture, 
                                   HexCoord(0, 0, 0), details.common_good, details.rare_good);
            }
            
            std::vector<int> available_ids;
            for (const auto& [id, details] : kCityDetailsMap) if (id != 15) available_ids.push_back(id);
            
            std::set<HexCoord> valid_hexes = GenerateRegularBoard(grid_radius_);
            valid_hexes.erase(HexCoord(0, 0, 0));
            
            std::mt19937 gen(rng_seed_);
            std::shuffle(available_ids.begin(), available_ids.end(), gen);
            
            std::vector<HexCoord> positions(valid_hexes.begin(), valid_hexes.end());
            std::shuffle(positions.begin(), positions.end(), gen);
            
            int count = std::min({3, (int)available_ids.size(), (int)positions.size()});
            for (int i = 0; i < count; ++i) {
                auto city_it = kCityDetailsMap.find(available_ids[i]);
                if (city_it != kCityDetailsMap.end()) {
                    const auto& details = city_it->second;
                    cities.emplace_back(details.id, details.name, details.culture, 
                                       positions[i], details.common_good, details.rare_good);
                }
            }
            return cities;
        }

        std::unique_ptr<State> Mali_BaGame::DeserializeState(const std::string &str) const
        {
            // The deserialization logic remains largely the same, as it only deals with
            // the dynamic parts of the state. The state object itself will get the static
            // board info from the Game object it's constructed with.
            if (str.empty()) return NewInitialState();
            std::unique_ptr<Mali_BaState> state = std::make_unique<Mali_BaState>(shared_from_this());
            try {
                json j = json::parse(str);
                
                state->current_player_id_ = j.at("currentPlayerId").get<int>();
                state->SetCurrentPhase(static_cast<Phase>(j.at("currentPhase").get<int>()));

                if (state->CurrentPhase() == Phase::kSetup) {
                    state->current_player_id_ = kChancePlayerId;
                    state->current_player_color_ = PlayerColor::kEmpty;
                } else {
                    SPIEL_CHECK_GE(state->current_player_id_, 0);
                    SPIEL_CHECK_LT(state->current_player_id_, NumPlayers());
                    state->current_player_color_ = state->GetPlayerColor(state->current_player_id_);
                }
                
                if (j.contains("playerTokens")) {
                    for (auto const &[hex_str, color_val] : j.at("playerTokens").items()) {
                        if (auto hex = JsonStringToHexCoord(hex_str)) {
                            if (color_val.is_array()) {
                                std::vector<PlayerColor> colors;
                                for (const auto& color_int : color_val) {
                                    colors.push_back(static_cast<PlayerColor>(color_int.get<int>()));
                                }
                                if (!colors.empty()) {
                                    state->player_token_locations_[*hex] = colors;
                                }
                            }
                        }
                    }
                }
                if (j.contains("hexMeeples")) {
                    for (auto const &[hex_str, j_list] : j.at("hexMeeples").items()) {
                         if (auto hex = JsonStringToHexCoord(hex_str)) {
                             for (const auto &mc_val : j_list) state->hex_meeples_[*hex].push_back(static_cast<MeepleColor>(mc_val.get<int>()));
                         }
                    }
                }
                if (j.contains("tradePosts")) {
                    for (auto const &[hex_str, j_list] : j.at("tradePosts").items()) {
                         if (auto hex = JsonStringToHexCoord(hex_str)) {
                             for (const auto &j_post : j_list) {
                                 PlayerColor owner = static_cast<PlayerColor>(j_post.at("owner").get<int>());
                                 TradePostType type = static_cast<TradePostType>(j_post.at("type").get<int>());
                                 state->trade_posts_locations_[*hex].push_back({owner, type});
                             }
                         }
                    }
                }
                if (j.contains("history")) {
                    for (const auto &j_move : j.at("history")) {
                        Move move;
                        move.player = static_cast<PlayerColor>(j_move.at("player").get<int>());
                        if (auto hex = JsonStringToHexCoord(j_move.at("startHex").get<std::string>())) move.start_hex = *hex;
                        move.type = static_cast<ActionType>(j_move.at("type").get<int>());
                        move.place_trading_post = j_move.at("placePost").get<bool>();
                        if (j_move.contains("path")) {
                            for (const auto &hex_str_val : j_move.at("path")) {
                                if (auto hex = JsonStringToHexCoord(hex_str_val.get<std::string>())) move.path.push_back(*hex);
                            }
                        }
                        state->moves_history_.push_back(move);
                    }
                }
                if (j.contains("commonGoods")) state->SetCommonGoods(j.at("commonGoods"));
                if (j.contains("rareGoods")) state->SetRareGoods(j.at("rareGoods"));
                if (j.contains("tradeRoutes")) {
                     for (const auto& j_route : j.at("tradeRoutes")) {
                        TradeRoute route;
                        route.id = j_route.at("id").get<int>();
                        route.owner = static_cast<PlayerColor>(j_route.at("owner").get<int>());
                        if(j_route.contains("hexes")) for(const auto& h_str : j_route.at("hexes")) if(auto h=JsonStringToHexCoord(h_str)) route.hexes.push_back(*h);
                        if(j_route.contains("goods")) route.goods = j_route.at("goods").get<std::map<std::string, int>>();
                        route.active = j_route.value("active", true);
                        state->trade_routes_.push_back(route);
                        if (route.id >= state->next_route_id_) state->next_route_id_ = route.id + 1;
                    }
                }

            } catch (const std::exception &e) {
                LOG_ERROR(absl::StrCat("Deserialize ERROR: ", e.what()));
                return NewInitialState();
            }

            state->ClearCaches();
            state->RefreshTerminalStatus();
            return state;
        }

    } // namespace mali_ba
} // namespace open_spiel
```

### Step 4: Updated `mali_ba_state.h`

The state header becomes much lighter.

```cpp
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

#ifndef OPEN_SPIEL_GAMES_MALI_BA_STATE_H_
#define OPEN_SPIEL_GAMES_MALI_BA_STATE_H_

#include <vector>
#include <map>
#include <set>
#include <string>
#include <memory>
#include <random>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sstream>

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"

#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"
// #include "open_spiel/games/mali_ba/mali_ba_board_config.h" // REMOVED

namespace open_spiel
{
namespace mali_ba
{
    class Mali_BaGame;
    struct MaliBaTest;

    std::string HexCoordToJsonString(const HexCoord& hex);
    absl::optional<HexCoord> JsonStringToHexCoord(const std::string& s);

    // ... (TurnEvaluation, StateSnapshot structs remain the same) ...
    struct TurnEvaluation {
        std::vector<Action> actions;
        double estimated_value;
        bool changes_game_outcome;
        int trade_route_count;
        int income_potential;
    };

    struct StateSnapshot {
        Player current_player_id_;
        PlayerColor current_player_color_;
        Phase current_phase_;
        std::map<HexCoord, std::vector<PlayerColor>> player_token_locations_;
        std::map<HexCoord, std::vector<MeepleColor>> hex_meeples_;
        std::map<HexCoord, std::vector<TradePost>> trade_posts_locations_;
        std::vector<std::map<std::string, int>> common_goods_;
        std::vector<std::map<std::string, int>> rare_goods_;
        std::vector<TradeRoute> trade_routes_;
        int next_route_id_;
        std::vector<Move> moves_history_;
        bool is_terminal_;
    };


    class Mali_BaState : public State {
    public:
        friend class Mali_BaGame;
        friend struct MaliBaTest;

        explicit Mali_BaState(std::shared_ptr<const Game> game);
        Mali_BaState(const Mali_BaState& other);
        Mali_BaState(Mali_BaState&& other) = default;
        Mali_BaState& operator=(Mali_BaState&& other) = default;

        Player CurrentPlayer() const override;
        std::vector<Action> LegalActions() const override;
        std::string ActionToString(Player player, Action action) const override;
        std::string ToString() const override;
        bool IsTerminal() const override;
        std::vector<double> Returns() const override;
        std::string InformationStateString(Player player) const override;
        std::string ObservationString(Player player) const override;
        void ObservationTensor(Player player, absl::Span<float> values) const override;
        void UndoAction(Player player, Action action) override;
        std::string Serialize() const override;
        bool IsChanceNode() const override;
        std::vector<std::pair<Action, double>> ChanceOutcomes() const override;
        std::unique_ptr<State> Clone() const override;
        
        // Removed Board Configuration methods
        
        // ... (other methods like Undo, PlayRandom, AI selection remain) ...
        void UndoLastAction();
        void UndoToTurnStart();
        std::string PlayRandomMoveAndSerialize();
        std::vector<Action> SelectRandomTurnActions();
        std::string PlayRandomTurnAndSerialize();
        Action SelectTrainingAwareRandomAction();
        std::vector<Action> SelectEvaluatedTurnActions();
        TurnEvaluation GenerateTurnStrategy(int num_free_actions, PlayerColor player);
        int CalculateIncomeGeneration(const Mali_BaState& state, PlayerColor player) const;
        Action ParseMoveStringToAction(const std::string& move_str) const;
        Move ActionToMove(Action action) const;
        void ClearCaches();
        void RefreshTerminalStatus() { is_terminal_ = IsTerminal(); }
        Phase CurrentPhase() const { return current_phase_; }
        std::string DebugString() const { return ToString(); }

        // --- Getters that now delegate to the Game object ---
        const Mali_BaGame *GetGame() const;
        const std::set<HexCoord>& ValidHexes() const;
        const std::vector<City>& GetCities() const;
        int GridRadius() const;
        
        // --- Getters for dynamic state (remain the same) ---
        const std::vector<std::map<std::string, int>>& GetCommonGoods() const { return common_goods_; }
        const std::vector<std::map<std::string, int>>& GetRareGoods() const { return rare_goods_; }
        PlayerColor GetPlayerTokenAt(const HexCoord &hex) const;
        const std::vector<MeepleColor> &GetMeeplesAt(const HexCoord &hex) const;
        const std::vector<TradePost> &GetTradePostsAt(const HexCoord &hex) const;
        const std::vector<TradeRoute> &GetTradeRoutes() const { return trade_routes_; }
        bool IsValidHex(const HexCoord &hex) const;
        PlayerColor GetCurrentPlayerColor() const { return current_player_color_; }
        Player GetPlayerId(PlayerColor color) const;
        PlayerColor GetPlayerColor(Player id) const;
        PlayerColor GetNextPlayerColor(PlayerColor current) const;
        const std::map<std::string, int> &GetPlayerCommonGoods(Player player) const;
        const std::map<std::string, int> &GetPlayerRareGoods(Player player) const;
        int GetCommonGoodCount(Player player, const std::string &good_name) const;
        int GetRareGoodCount(Player player, const std::string &good_name) const;
        std::mt19937& GetRNG() { return rng_; }

        // ... (TestOnly setters and other methods remain) ...
        void TestOnly_SetCurrentPlayer(Player player);
        void TestOnly_SetTradePost(const HexCoord& hex, PlayerColor owner, TradePostType type);
        void TestOnly_SetPlayerToken(const HexCoord& hex, PlayerColor owner);
        void TestOnly_SetPlayerTokens(const HexCoord& hex, const std::vector<PlayerColor>& owners); 
        void TestOnly_SetMeeples(const HexCoord& hex, const std::vector<MeepleColor>& meeples);
        void TestOnly_SetCommonGood(Player player, const std::string& good_name, int count);
        void TestOnly_SetRareGood(Player player, const std::string& good_name, int count);
        void TestOnly_ClearPlayerTokens();
        void TestOnly_ClearMeeples();
        void SetMoveLoggingEnabled(bool answ);
        void SetCurrentPhase(Phase phase);
        void SetCommonGoods(const std::vector<std::map<std::string, int>>& goods);
        void SetRareGoods(const std::vector<std::map<std::string, int>>& goods);
        void ApplyIncomeCollection(const std::string& action_str);
        void AddTradingPost(const HexCoord &hex, PlayerColor player, TradePostType type);
        void UpgradeTradingPost(const HexCoord &hex, PlayerColor player);
        bool CanPlaceTradingPostAt(const HexCoord &hex, PlayerColor player) const;
        int CountTradingCentersAt(const HexCoord &hex) const;
        bool HasPlayerPostOrCenterAt(const HexCoord &hex, PlayerColor player) const;
        bool CreateTradeRoute(const std::vector<HexCoord>& hexes, PlayerColor player);
        bool UpdateTradeRoute(int route_id, const std::vector<HexCoord>& hexes);
        bool DeleteTradeRoute(int route_id);

        // ... (Public helper methods like HasTokenAt, CountTokensAt remain) ...
        bool HasTokenAt(const HexCoord& hex, PlayerColor color) const;
        int CountTokensAt(const HexCoord& hex, PlayerColor color) const;
        int CountTotalTokensAt(const HexCoord& hex) const;
        std::vector<PlayerColor> GetTokensAt(const HexCoord& hex) const;
        bool RemoveTokenAt(const HexCoord& hex, PlayerColor color);
        void AddTokenAt(const HexCoord& hex, PlayerColor color);
        PlayerColor GetFirstTokenAt(const HexCoord& hex) const;
        
        // ... (Move logging and Python sync helpers remain) ...
        void InitializeMoveLogging();
        void LogMove(const std::string& action_string, const std::string& state_json);
        std::string CreateSetupJson() const;
        static std::string GetCurrentDateTime();
        static std::string GetMoveLogFilename() { return move_log_filename_; }
        void SetCitiesFromPython(const std::vector<std::map<std::string, std::string>>& cities_data);
        void SetMeeplesFromPython(const std::map<std::string, std::vector<int>>& meeples_data);
        void SetPlayerTokensFromPython(const std::map<std::string, std::vector<int>>& tokens_data);
        void SetBoardConfigFromPython(const std::map<std::string, std::variant<int, std::vector<std::vector<int>>>>& config_data);
        HexCoord ParseHexCoordFromString(const std::string& coord_str) const;
        std::vector<HexCoord> ParseHexListFromData(const std::vector<std::vector<int>>& hex_data) const;
        void ValidateTradeRoutes();
        int CountAllMeeples() const;
        int GetMaxMeeplesOnHex() const;
        void DebugPrintStateDetails() const;
        bool SetStateFromJson(const std::string& json_str);
        std::string GetCurrentStateJson() const { return Serialize(); }
        bool SaveStateToFile(const std::string& filename) const;
        bool LoadStateFromFile(const std::string& filename);
        void ResetToInitialState();

    protected:
        void DoApplyAction(Action action) override;

    private:
        // --- Private Dynamic State Members (Unchanged) ---
        Phase current_phase_;
        Player current_player_id_;
        PlayerColor current_player_color_;
        std::map<HexCoord, std::vector<PlayerColor>> player_token_locations_;
        std::vector<int> player_posts_supply_;
        std::map<HexCoord, std::vector<MeepleColor>> hex_meeples_;
        std::map<HexCoord, std::vector<TradePost>> trade_posts_locations_;
        std::vector<std::map<std::string, int>> common_goods_;
        std::vector<std::map<std::string, int>> rare_goods_;
        std::vector<TradeRoute> trade_routes_;
        int next_route_id_ = 1;
        std::vector<Move> moves_history_;
        std::mt19937 rng_; 
        mutable bool is_terminal_ = false;
        mutable absl::optional<std::vector<Action>> cached_legal_actions_;
        mutable absl::optional<std::vector<Move>> cached_legal_move_structs_;
        std::vector<StateSnapshot> undo_stack_;

        // --- Private Static Members (Unchanged) ---
        static std::unique_ptr<std::ofstream> move_log_file_;
        static std::string move_log_filename_;
        static int move_count_;
        static bool move_logging_initialized_;
        static bool move_logging_enabled_;

        // ... (Private income/move generation helpers remain) ...
        struct IncomeChoice {
            HexCoord center_hex;
            std::vector<const City*> connected_cities;
            int num_choices;
            bool is_isolated;
        };
        struct PostChoice {
            HexCoord post_hex;
            std::vector<const City*> equidistant_cities;
        };
        void MaybeGenerateLegalActions() const;
        std::vector<Move> GenerateIncomeMoves() const;
        std::vector<const City*> GetConnectedCities(const HexCoord& center_hex, PlayerColor player) const;
        std::vector<const City*> FindClosestCities(const HexCoord& hex) const;
        void GenerateIncomeMoveCombinations(
            std::vector<Move>& moves,
            std::map<std::string, int> current_common,
            std::map<std::string, int> current_rare,
            const std::vector<IncomeChoice>& center_choices,
            const std::vector<PostChoice>& post_choices,
            int center_index,
            int post_index) const;
        void GenerateIsolatedCenterCombinations(
            std::vector<Move>& moves,
            std::map<std::string, int> current_common,
            std::map<std::string, int> current_rare,
            const std::vector<IncomeChoice>& center_choices,
            const std::vector<PostChoice>& post_choices,
            int center_index,
            int post_index,
            const IncomeChoice& choice) const;
        void GenerateConnectedCenterCommonCombinations(
            std::vector<Move>& moves,
            std::map<std::string, int> current_common,
            std::map<std::string, int> current_rare,
            const std::vector<IncomeChoice>& center_choices,
            const std::vector<PostChoice>& post_choices,
            int center_index,
            int post_index,
            const IncomeChoice& choice) const;
        void GenerateConnectedCenterRareCombinations(
            std::vector<Move>& moves,
            std::map<std::string, int> current_common,
            std::map<std::string, int> current_rare,
            const std::vector<IncomeChoice>& center_choices,
            const std::vector<PostChoice>& post_choices,
            int center_index,
            int post_index,
            const IncomeChoice& choice) const;
        std::string NormalizeIncomeAction(const std::string& action_string) const;
        std::string CreateIncomeActionString(
            const std::map<std::string, int>& common_goods,
            const std::map<std::string, int>& rare_goods) const;
        bool IsRareGood(const std::string& good_name) const;
        std::vector<Move> GeneratePlaceTokenMoves() const;
        std::vector<Move> GenerateMancalaMoves() const;
        std::vector<Move> GenerateTradePostUpgradeMoves() const;
        std::vector<Move> GenerateTradeRouteMoves() const;
        void FindMancalaPathsRecursive(const HexCoord &start_hex, const HexCoord &current_hex,
            int remaining_meeples, std::vector<HexCoord> &current_path,
            std::set<HexCoord> &visited_hexes, std::vector<Move> &all_valid_moves) const;
        void ApplyMancalaMove(const Move &move);
        void ApplyPlaceTokenMove(const Move &move);
        void ApplyTradingPostUpgrade(const Move &move);
        void ApplyTradeRouteCreate(const Move& move);
        void ApplyTradeRouteUpdate(const Move& move);
        void ApplyTradeRouteDelete(const Move& move);
        void InitializeBoard();
        void ApplyChanceSetup();
        void PushStateToUndoStack();
        absl::optional<std::vector<double>> MaybeFinalReturns() const;
        void ClearAllState();
    };
    // ... (rest of the file remains the same) ...
}
}
#endif
```

### Step 5: Updated `mali_ba_state_core.cc`

This file is a great example of the changes. The constructor is much simpler, and other methods now use `GetGame()->...`

```cpp
// mali_ba_state_core.cc
// Core state implementation (constructor, basic state API functions)

#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/hex_grid.h"

#include <algorithm>
#include <map>
#include <memory>
#include <sstream>
#include <fstream>
#include <filesystem>

#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel
{
    namespace mali_ba
    {
        namespace { // Use anonymous namespace for local constants/helpers

            // Max values needed for plane indexing (match observer)
            constexpr int kMaxPlayersObs = 5;
            constexpr int kNumMeepleColorsObs = 10;
        } // namespace

        // --- State Constructor ---
        Mali_BaState::Mali_BaState(std::shared_ptr<const Game> game)
            : State(game),
              current_phase_(Phase::kSetup),
              current_player_id_(kChancePlayerId),
              current_player_color_(PlayerColor::kEmpty),
              next_route_id_(1)
        {
            LOG_INFO("Mali_BaState::Constructor: ENTRY");
            rng_.seed(GetGame()->GetRNGSeed());
            
            int num_players = game_->NumPlayers();
            common_goods_.resize(num_players);
            rare_goods_.resize(num_players);
            player_posts_supply_.resize(num_players);

            // Initialize board state based on the already configured Game object.
            InitializeBoard();
            LOG_INFO("Mali_BaState::Constructor: EXIT");
        }

        // --- State Copy Constructor ---
        Mali_BaState::Mali_BaState(const Mali_BaState& other)
            : State(other), // Base class copy constructor
              // Core state members are copied
              current_phase_(other.current_phase_),
              current_player_id_(other.current_player_id_),
              current_player_color_(other.current_player_color_),
              player_token_locations_(other.player_token_locations_),
              player_posts_supply_(other.player_posts_supply_),
              hex_meeples_(other.hex_meeples_),
              trade_posts_locations_(other.trade_posts_locations_),
              common_goods_(other.common_goods_),
              rare_goods_(other.rare_goods_),
              trade_routes_(other.trade_routes_),
              next_route_id_(other.next_route_id_),
              moves_history_(other.moves_history_),
              rng_(other.rng_), // Copy RNG state for determinism
              is_terminal_(other.is_terminal_),
              undo_stack_(other.undo_stack_)
              // No need to copy board config, it's in the shared Game object.
        {
            LOG_DEBUG("Mali_BaState::CopyConstructor: State copied.");
        }


        std::unique_ptr<State> Mali_BaState::Clone() const {
            return std::make_unique<Mali_BaState>(*this);
        }

        // --- Delegating Getters ---
        const Mali_BaGame *Mali_BaState::GetGame() const {
          return static_cast<const Mali_BaGame *>(game_.get());
        }
        const std::set<HexCoord>& Mali_BaState::ValidHexes() const {
          return GetGame()->GetValidHexes();
        }
        const std::vector<City>& Mali_BaState::GetCities() const {
          return GetGame()->GetCities();
        }
        int Mali_BaState::GridRadius() const {
          return GetGame()->GetGridRadius();
        }
        bool Mali_BaState::IsValidHex(const HexCoord &hex) const {
          return GetGame()->GetValidHexes().count(hex) > 0;
        }


        void Mali_BaState::InitializeBoard() {
            const GameRules& rules = GetGame()->GetRules();
            player_token_locations_.clear();
            hex_meeples_.clear();
            trade_posts_locations_.clear();
            // Get valid hexes from the Game object now
            for (const auto &hex : GetGame()->GetValidHexes()) {
                hex_meeples_[hex] = {};
                trade_posts_locations_[hex] = {};
            }
            for (int i = 0; i < game_->NumPlayers(); ++i) {
                player_posts_supply_[i] = rules.posts_per_player;
            }
        }

        Player Mali_BaState::CurrentPlayer() const {
            if (is_terminal_) return kTerminalPlayerId;
            return current_player_id_;
        }

        bool Mali_BaState::IsChanceNode() const {
            return current_player_id_ == kChancePlayerId;
        }

        std::vector<std::pair<Action, double>> Mali_BaState::ChanceOutcomes() const {
            SPIEL_CHECK_TRUE(IsChanceNode());
            return {{kChanceSetupAction, 1.0}};
        }

        void Mali_BaState::MaybeGenerateLegalActions() const {
            if (cached_legal_actions_) return;

            // ... (rest of the method is the same) ...
            if (current_phase_ == Phase::kPlay) {
                LOG_INFO("Total meeples: ", CountAllMeeples(), ", Max on a single hex: ", GetMaxMeeplesOnHex());
            }
            std::vector<Action> actions;
            std::vector<Move> moves;
            if (IsTerminal()) {
            } else if (IsChanceNode()) {
                actions.push_back(kChanceSetupAction);
                moves.push_back({.type = ActionType::kChanceSetup});
            } else { // Player's turn
                switch (current_phase_) {
                    case Phase::kPlaceToken: {
                        moves = GeneratePlaceTokenMoves();
                        break;
                    }
                    case Phase::kPlay: {
                        moves.push_back(kPassMove);
                        std::vector<Move> income_moves = GenerateIncomeMoves();
                        moves.insert(moves.end(), income_moves.begin(), income_moves.end());
                        std::vector<Move> mancala_moves = GenerateMancalaMoves();
                        moves.insert(moves.end(), mancala_moves.begin(), mancala_moves.end());
                        std::vector<Move> upgrade_moves = GenerateTradePostUpgradeMoves();
                        moves.insert(moves.end(), upgrade_moves.begin(), upgrade_moves.end());
                        std::vector<Move> trade_route_moves = GenerateTradeRouteMoves();
                        moves.insert(moves.end(), trade_route_moves.begin(), trade_route_moves.end());
                        break;
                    }
                    default:
                        LOG_WARN(absl::StrCat("MaybeGenerateLegalActions: No actions for phase ", current_phase_));
                        break;
                }
                for (int i = 0; i < moves.size(); ++i) {
                    actions.push_back(i);
                }
            }
            cached_legal_actions_ = actions;
            cached_legal_move_structs_ = moves;
            LOG_INFO(absl::StrCat("MaybeGenerateLegalActions: FINISHED generating ",
                cached_legal_actions_->size(), " actions for phase ", current_phase_));
        }

        // ... (DoApplyAction, ActionToString, Undo, etc. will now use GetGame()->... where needed) ...
        // ... (for example, in ActionToString, `cities_` becomes `GetGame()->GetCities()`) ...

        // Example change in ObservationTensor:
        void Mali_BaState::ObservationTensor(Player player, absl::Span<float> values) const
        {
            // ... (setup remains the same) ...
            const Mali_BaGame *mali_ba_game = GetGame();
            
            // ... (plane definitions remain the same) ...
            
            int grid_radius = mali_ba_game->GetGridRadius();

            // The loop now gets hexes from the game object.
            for (const auto &hex : mali_ba_game->GetValidHexes()) {
                // ... (logic inside loop is the same, but calls like `GetCityAt` must use the game object)
                if (mali_ba_game->GetCityAt(hex) != nullptr) {
                    //...
                }
            }
            // ... (rest of the method is the same) ...
        }

        // ... (The rest of mali_ba_state_core.cc methods are adapted similarly) ...
        // The implementation of DoApplyAction, ActionToString, PushStateToUndoStack, UndoAction,
        // InformationStateString, ObservationString, IsTerminal, MaybeFinalReturns, Returns,
        // ClearCaches, CountAllMeeples, GetMaxMeeplesOnHex, DebugPrintStateDetails, and the
        // training-related functions do not need significant changes beyond using GetGame()
        // for accessing cities, valid hexes, or rules. The provided implementations should
        // adapt well.
    }
}
```
*Note: I've only shown the key changes in `mali_ba_state_core.cc` for brevity. You would apply the same principle (`valid_hexes_` -> `GetGame()->GetValidHexes()`, `cities_` -> `GetGame()->GetCities()`, etc.) to all other methods in the `mali_ba_state_*.cc` files.*

### Step 6: Updated `mali_ba_test.cc`

The test for the INI file must be updated to check the `Game` object instead of the `State` object for the configuration.

```cpp
// ... includes ...

namespace open_spiel {
namespace mali_ba {
namespace {

namespace testing = open_spiel::testing;

// ... (MaliBaTest fixture remains mostly the same) ...
struct MaliBaTest {
    // ...
    // No changes needed here, as it already holds both game and state
};

// ... (Other tests like APITest_StateCreationAndClone remain the same) ...


void IniFileConfigTest() {
    LOG_INFO("--- IniFileConfigTest ---");

    // Create a temporary config file for the test
    std::string config_content = 
        "[Board]\n"
        "grid_radius = 2\n"
        "custom_hexes = 0,0,0; 1,0,-1; -1,0,1; 0,1,-1; 0,-1,1; 1,-1,0; -1,1,0\n" // 7 hexes
        "[Cities]\n"
        "city1 = Timbuktu,0,0,0\n"
        "city2 = Segou,1,-1,0\n";
    std::string config_path = "/tmp/mali_ba_test.ini";
    std::ofstream out(config_path);
    out << config_content;
    out.close();

    open_spiel::GameParameters params;
    params["config_file"] = open_spiel::GameParameter(config_path);

    // Load the game with the config file parameter.
    std::shared_ptr<const open_spiel::Game> game = open_spiel::LoadGame("mali_ba", params);
    SPIEL_CHECK_TRUE(game != nullptr);
    
    // Cast to the specific game type to access its members
    const auto* mali_ba_game = static_cast<const Mali_BaGame*>(game.get());

    // Check if the GAME object was configured correctly.
    SPIEL_CHECK_EQ(mali_ba_game->GetGridRadius(), 2);
    SPIEL_CHECK_EQ(mali_ba_game->GetValidHexes().size(), 7);
    SPIEL_CHECK_EQ(mali_ba_game->GetCities().size(), 2);

    // The game object is now configured. Create a state FROM it.
    std::unique_ptr<open_spiel::State> state = game->NewInitialState();
    auto mali_ba_state = static_cast<Mali_BaState*>(state.get());

    // Check if the STATE correctly reflects the game's configuration.
    SPIEL_CHECK_EQ(mali_ba_state->GridRadius(), 2);
    SPIEL_CHECK_EQ(mali_ba_state->ValidHexes().size(), 7);
    SPIEL_CHECK_EQ(mali_ba_state->GetCities().size(), 2);

    // Print the state to see the result
    std::cout << "State from INI file config:\n" << state->ToString() << std::endl;
    
    LOG_INFO("IniFileConfigTest passed.");
}

// ... (rest of main() and other tests) ...

}
}
}

int main(int argc, char** argv) {
    // ...
    open_spiel::mali_ba::IniFileConfigTest();
    // ...
}
```

After applying these changes and removing the `mali_ba_board_config.*` files from your build, your game will be more efficient and better aligned with the OpenSpiel architecture.