// Copyright 2025 DeepMind Technologies Limited
//queue size
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

#ifndef OPEN_SPIEL_GAMES_MALI_BA_COMMON_H_
#define OPEN_SPIEL_GAMES_MALI_BA_COMMON_H_

#include <array>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iomanip>
#include <sstream>  // for stringstream
#include <ostream>
#include <ctime>    // for std::tm and std::time
#include <algorithm> // For std::transform
#include <cctype>    // For std::tolower

#include "open_spiel/spiel.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/mali_ba/hex_grid.h"

namespace open_spiel
{
  namespace mali_ba
  {

    // Constants - open_spiel requires an 'Action' to be an integer.
    constexpr Action kInvalidAction = -1;
    // 0 is fine for chance setup because it will only be returned by LegalActions() on a chance node,
    // where it won't conflict with player action 0.
    constexpr Action kChanceSetupAction = 0;
    // Some constants for readability
    constexpr int kUnlimitedPosts = -1;

    // Enum to describe the type of a Move struct.
    // This is now independent of the Action's integer value.
    enum class ActionType
    {
      kInvalid = -1,
      kPass = 0,
      kChanceSetup = 1,
      kPlaceToken = 2,
      kMancala = 3,
      kPlaceTCenter = 4, // This type is for the "upgrade" move.
      kIncome = 5,
      kTradeRouteCreate = 6,
      kTradeRouteUpdate = 7,
      kTradeRouteDelete = 8,
    };
    std::ostream &operator<<(std::ostream &os, const ActionType &move_type);

    // Other constants
    // A reasonable upper bound on the number of moves from any single state.
    inline constexpr int MaxGameLength() { return 310; }
    inline constexpr double LossUtility() { return -1; }
    inline constexpr double DrawUtility() { return 0; }
    inline constexpr double WinUtility() { return 1; }
    // Constants to implement autoregressive training approach
    constexpr int kMaxHexes = 100; // A safe upper bound for board size

    constexpr int kPassAction = 0;
    constexpr int kIncomeAction = 1;
    // Reserve a block of actions for placing tokens.
    // It must be large enough to hold all possible hex indices.
    constexpr int kPlaceTokenActionBase = 10; // Start after Pass and Income
    
    // All other action bases must now be offset by this.
    // kMaxHexes is a safe upper bound for board size
    constexpr int kUpgradeActionBase = kPlaceTokenActionBase + kMaxHexes;
    constexpr int kMancalaActionBase = kUpgradeActionBase + kMaxHexes;
    
    // Flags remain the same
    constexpr int kPlacePostFlag = 20000; // A large, safe number
    constexpr int kDeclareRouteFlag = 40000; // Must be > kPlacePostFlag
    
    // Trade Route Base is now relative to the new Mancala base
    constexpr int kTradeRouteCreateBase = kMancalaActionBase + (kMaxHexes * kMaxHexes);
    // kMaxActions must be larger than any possible action ID, including flags.
    // The largest possible action is a compound upgrade: kDeclareRouteFlag + upgrade_action.
    // Let's set it to a safe, round number above the highest possible flag.
    constexpr int kMaxActions = 50000;
    // // A large, fixed number for the policy head
    // constexpr int kMaxActions = kTradeRouteCreateBase + 300; // Allow for 300 heuristic trade routes

    constexpr int kLegacyMaxActions = kMancalaActionBase + (kMaxHexes * kMaxHexes); // Old constant

    inline constexpr int NumDistinctActions() { return kMaxActions; }

    // Observation tensor shape
    /* -----------------------------------------------------
    Mali-Ba Observation Tensor Planes (Total: 47)

    The tensor shape is (47, 11, 11). Each cell (plane, row, col) corresponds to a specific piece of information about a hex on the board.
    Plane Index(es)	Count	Purpose	Value Represents	Notes
    0 - 4	5	Player Tokens	1.0 if a token of the corresponding player is present on the hex, 0.0 otherwise.	Plane 0 = Player 0 (Red), Plane 1 = Player 1 (Green), etc. A hex can only have one token, so only one of these planes will be non-zero at any hex.
    5 - 14	10	Meeple Counts	The number of meeples of a specific color on the hex (e.g., 3.0 if there are 3 Solid Black meeples).	Plane 5 = Solid Black, Plane 6 = Clear Black, ..., Plane 14 = Clear Tan. Allows the network to see resource density.
    15 - 19	5	Trading Posts	1.0 if a Trading Post of the corresponding player is present, 0.0 otherwise.	Plane 15 = Player 0's Posts, Plane 16 = Player 1's Posts, etc.
    20 - 24	5	Trading Centers	1.0 if a Trading Center of the corresponding player is present, 0.0 otherwise.	Plane 20 = Player 0's Centers, Plane 21 = Player 1's Centers, etc. Separated from posts for strategic importance.
    25	1	Cities	1.0 if the hex contains a City, 0.0 otherwise.	Highlights strategically critical locations on the board.
    26	1	Current Player's Turn	1.0 everywhere if it is this player's turn to act, 0.0 otherwise.	This is a "uniform" plane. It tells the network "this observation is from your perspective."
    27 - 31	5	Player's Total Common Goods	The total number of common goods owned by the corresponding player (e.g., 7.0).	Plane 27 = Player 0's total common goods. A uniform plane providing non-spatial info about a player's economic state.
    32 - 36	5	Player's Total Rare Goods	The total number of rare goods owned by the corresponding player (e.g., 2.0).	Plane 32 = Player 0's total rare goods. A uniform plane providing non-spatial info about a player's victory progress.
    37 - 41	5	Potential Trade Routes (Posts & Centers)	1.0 if the hex has a Post, 2.0 if it has a Center, for the corresponding player.	Plane 37 = Player 0's potential routes. Lets the CNN "see" the shape of a player's network and potential connections.
    42 - 46	5	Active Trade Routes	1.0 if the hex is part of an active trade route for the corresponding player.	Plane 42 = Player 0's active routes. Shows currently scoring/valid routes, a direct indicator of strategic success.
    -------------------------------------------------------- */
    inline const std::vector<int> &ObservationTensorShape()
    {
      static std::vector<int> shape = {
        47,    // See above for information
        11, 11 // Board dimensions
      };
      return shape;
    }

    // --- Enums ---
    enum class Phase
    {
      kEmpty = -1,
      kSetup = 0,
      kPlaceToken = 1,
      kPlay = 2,
      kEndRound = 3,
      kGameOver = 9
    };
    std::ostream &operator<<(std::ostream &os, const Phase &phase);

    enum class PlayerColor: int
    {
      kEmpty = -1,
      kRed = 0,
      kGreen = 1,
      kBlue = 2,
      kViolet = 3,
      kPink = 4
    };

    enum class PlayerType {
      kHuman = 0,
      kAI = 1,
      kHeuristic = 2
    };

    enum class MeepleColor
    {
      kEmpty = -1,
      kSolidBlack = 0,
      kClearBlack = 1,
      kSolidSilver = 2,
      kClearSilver = 3,
      kClearWhite = 4,
      kSolidGold = 5,
      kClearGold = 6,
      kSolidBronze = 7,
      kClearBronze = 8,
      kClearTan = 9
    };

    enum class TradePostType
    {
      kNone = 0,
      kPost = 1,
      kCenter = 2
    };

    // --- Structs ---
    struct TradeRoute
    {
      int id;
      PlayerColor owner;
      std::vector<HexCoord> hexes;
      std::map<std::string, int> goods;
      bool active;
    };

    struct TradePost
    {
      PlayerColor owner = PlayerColor::kEmpty;
      TradePostType type = TradePostType::kNone;

      bool operator==(const TradePost &other) const
      {
        return owner == other.owner && type == other.type;
      }

      bool operator!=(const TradePost &other) const
      {
        return !(*this == other);
      }
    };

    struct City
    {
      int id;
      std::string name;
      std::string culture;
      HexCoord location;
      std::string common_good;
      std::string rare_good;

      City() : id(-1) {}

      City(int _id, const std::string &n, const std::string &c, const HexCoord &l,
           const std::string &cg, const std::string &rg)
          : id(_id), name(n), culture(c), location(l), common_good(cg), rare_good(rg) {}
    };

    struct Move {
        PlayerColor player = PlayerColor::kEmpty;
        ActionType type = ActionType::kPass;
        HexCoord start_hex = {0, 0, 0};
        std::vector<HexCoord> path;
        bool place_trading_post = false;
        std::string action_string; // For complex actions like income/upgrades
        int route_id = -1; // Kept for deleting existing routes

        // --- MEMBERS for compound actions ---
        bool declares_trade_route = false;
        std::vector<HexCoord> trade_route_path;
        
        // Comparison operators for sorting and deduplication (updated)
        bool operator<(const Move& other) const {
            if (player != other.player) return player < other.player;
            if (type != other.type) return type < other.type;
            if (start_hex != other.start_hex) return start_hex < other.start_hex;
            if (path != other.path) return path < other.path;
            if (place_trading_post != other.place_trading_post) return place_trading_post < other.place_trading_post;
            if (declares_trade_route != other.declares_trade_route) return declares_trade_route < other.declares_trade_route;
            if (trade_route_path != other.trade_route_path) return trade_route_path < other.trade_route_path;
            if (action_string != other.action_string) return action_string < other.action_string;
            return route_id < other.route_id;
        }
        
        bool operator==(const Move& other) const {
            return player == other.player && 
                  type == other.type && 
                  start_hex == other.start_hex && 
                  path == other.path && 
                  place_trading_post == other.place_trading_post &&
                  declares_trade_route == other.declares_trade_route &&
                  trade_route_path == other.trade_route_path &&
                  action_string == other.action_string &&
                  route_id == other.route_id;
        }
    };

    // City type details for city creation and lookup
    struct CityTypeDetails
    {
      int id;
      std::string name;
      std::string culture;
      std::string common_good;
      std::string rare_good;
    };

    // Standardized structure for a collection of goods (income, upgrade)
    // Structure to hold parsed goods
    struct GoodsCollection {
        std::map<std::string, int> common_goods;
        std::map<std::string, int> rare_goods;
        
        bool IsEmpty() const {
            return common_goods.empty() && rare_goods.empty();
        }
        
        int TotalCommon() const {
            int total = 0;
            for (const auto& [name, count] : common_goods) {
                total += count;
            }
            return total;
        }
        
        int TotalRare() const {
            int total = 0;
            for (const auto& [name, count] : rare_goods) {
                total += count;
            }
            return total;
        }
    };

    // Goods formatting and parsing functions
    std::map<std::string, int> ParseGoodsString(const std::string& goods_str);
    std::string FormatGoodsString(const std::map<std::string, int>& goods);
    GoodsCollection ParseGoodsCollection(const std::string& collection_str, 
                                      const std::vector<City>& cities);
    std::string FormatGoodsCollection(const GoodsCollection& collection);
    std::string FormatGoodsCollectionCompact(const GoodsCollection& collection);

    struct GameRules {
      // Turn structure
      // bool declare_trade_route_is_free_action = true;  // Deprecated

      // Mancala Post Placement
      bool mancala_post_requires_meeple = false; // NEW: If true, must have meeple; if false, can use common good
      
      // Income Rules
      int income_center_in_city_rare = 1;
      int income_center_connected_common = 2;
      int income_center_connected_rare = 1;
      int income_center_isolated_common = 2;
      int income_post_common = 1;

      // Upgrade Rules
      int upgrade_cost_common = 3;
      int upgrade_cost_rare = 1;
      bool remove_meeple_on_upgrade;
      bool remove_meeple_on_trade_route;
      bool city_free_upgrade = true;

      // Trade Post & Route Rules
      int posts_per_player = 6;
      int non_city_center_limit_divisor = 1; // n-1 limit: set to 1. No limit: set to 0.
      int min_hexes_for_trade_route = 3;
      int max_shared_centers_between_routes = 2;
      bool free_action_trade_routes = false;  // Default to declare route = standalone action

      // End Game Trigger Rules
      int end_game_req_num_routes = 2;          // The REQUIREMENT
      int end_game_cond_num_routes = -1;        // The victory CONDITION (e.g., win with 6 routes)
      int end_game_cond_num_rare_goods = -1;
      bool end_game_cond_timbuktu_to_coast = true;
      bool end_game_cond_rare_good_each_region = false;
      int end_game_cond_rare_good_num_regions = 5;

      // Scoring Rules
      // Points for longest routes (1st, 2nd, 3rd)
      std::vector<int> score_longest_routes = {11, 7, 4}; 
      // Map for unique common good set points
      std::map<int, int> score_unique_common_goods = {
        {1,1}, {2,3}, {3,6}, {4,11}, {5,19}, {6,30}, {7,45}, 
        {8,60}, {9,75}, {10,90}, {11,110}
      };
      int score_unique_common_goods_bonus = 20; // For 12+
      // Map for regions crossed points
      std::map<int, int> score_regions_crossed = {
        {1,4}, {2,8}, {3,12}, {4,17}, {5,23}, {6,30}
      };
      // Points for region control (1st, 2nd, 3rd)
      std::vector<int> score_region_control = {11, 7, 4};
      // Example other rule
      //std::string mancala_capture_rule = "none"; // This was also missing
    };

    struct HeuristicWeights {
      double weight_pass = 0.1;
      double weight_mancala = 10.0;
      double weight_upgrade = 15.0;
      double weight_income = 5.0;
      double weight_place_token = 5.0;
      double weight_trade_route_create = 50.0;
      // Specific Mancala Bonuses
      double bonus_mancala_city_end = 30.0;
      double bonus_mancala_long_distance = 10.0;
      double bonus_mancala_meeple_density = 15.0;
      // Bonus for upgrading a post that is far from other centers
      double bonus_upgrade_diversity_factor = 5.0; 
      // Bonus for upgrading a post in a new region
      double bonus_upgrade_new_region = 20.0;

      // Generic Bonuses
      double bonus1 = 0.0;
      double bonus2 = 0.0;
      double bonus3 = 0.0;
      double bonus4 = 0.0;
    };

    struct TrainingParameters {
      double time_penalty = -0.0035;
      double max_moves_penalty = -0.5;    // penalty for hitting max move limit
      double draw_penalty = 0.0;    // penalty for a draw (mult players same score)
      double loss_penalty = 0.0;     // penalty for losing
      
      // Intermediate reward values
      double upgrade_reward = 0.02;
      double trade_route_reward = 0.04;
      double new_rare_region_reward = 0.08;
      double new_common_good_reward = 0.02;
      double key_location_post_reward = 0.03;
      
      // Bonuses for quick wins
      double quick_win_bonus = 0.2;
      int quick_win_threshold = 150; // Games under this many moves get bonus
    };


    // Details about cities (Name, Culture, Common good, Rare good)
    const std::map<int, CityTypeDetails> kCityDetailsMap = {
        {1, {1, "Agadez", "Tuareg", "Iron work", "Silver cross"}},
        {2, {2, "Bandiagara", "Dogon", "Onions/tobacco", "Dogon mask"}},
        {3, {3, "Dinguiraye", "Fulani", "Cattle", "Wedding blanket"}},
        {4, {4, "Dosso", "Songhai-Zarma", "Cotton", "Silver headdress"}},
        {5, {5, "Hemang", "Akan", "Kente cloth", "Gold weight"}},
        {6, {6, "Katsina", "Housa", "Kola nuts", "Holy book"}},
        {7, {7, "Linguère", "Wolof", "Casava/peanut", "Gold necklace"}},
        {8, {8, "Ouagadougou", "Dagbani-Mossi", "Horses", "Bronze bracelet"}},
        {9, {9, "Oudane", "Arab", "Camel", "Bronze incense burner"}},
        {10, {10, "Oyo", "Yoruba", "Ivory", "Ivory bracelet"}},
        {11, {11, "Ségou", "Mande/Bambara", "Millet", "Chiwara"}},
        {12, {12, "Sikasso", "Senoufo", "Brass jewelry", "Kora"}},
        {13, {13, "Tabou", "Kru", "Pepper", "Kru boat"}},
        {14, {14, "Warri", "Idjo", "Palm Oil", "Coral necklace"}},
        {15, {15, "Timbuktu", "Songhai", "Salt", "Gold crown"}} // ID 15 for Timbuktu
    };
    // Function to get the ID of a City from the map above
    // std::string toLower(const std::string& input) {
    //   std::string result = input;
    //   std::transform(result.begin(), result.end(), result.begin(),
    //                 [](unsigned char c) { return std::tolower(c); });
    //   return result;
    // }
    inline int GetCityID(const std::string& name_to_find) {
      for (const auto& [id, details] : kCityDetailsMap) {
        if (absl::AsciiStrToLower(details.name) == absl::AsciiStrToLower(name_to_find)) { 
            return id;
        }
      }
      return -1; // Not found
    };


    // Logging. Define the log levels
    enum class LogLevel {
      kDebug,
      kInfo,
      kWarning,
      kError
    };

    extern bool g_mali_ba_logging_enabled;
    extern LogLevel g_mali_ba_log_level;
    extern std::string g_log_file_path;
    void SetLogLevel(LogLevel level);
    void LogMBCore(LogLevel, const std::string&, bool, const char*, int);

    #define LOG_DEBUG(...)   do { if (g_mali_ba_logging_enabled && g_mali_ba_log_level <= LogLevel::kDebug)   { LogMBCore(LogLevel::kDebug,   absl::StrCat(__VA_ARGS__), true, __FILE__, __LINE__); } } while(0)
    #define LOG_INFO(...)    do { if (g_mali_ba_logging_enabled && g_mali_ba_log_level <= LogLevel::kInfo)    { LogMBCore(LogLevel::kInfo,    absl::StrCat(__VA_ARGS__), true, __FILE__, __LINE__); } } while(0)
    #define LOG_WARN(...)    do { if (g_mali_ba_logging_enabled && g_mali_ba_log_level <= LogLevel::kWarning) { LogMBCore(LogLevel::kWarning, absl::StrCat(__VA_ARGS__), true, __FILE__, __LINE__); } } while(0)
    #define LOG_ERROR(...)   do { if (g_mali_ba_logging_enabled && g_mali_ba_log_level <= LogLevel::kError)   { LogMBCore(LogLevel::kError,   absl::StrCat(__VA_ARGS__), true, __FILE__, __LINE__); } } while(0)

    const TradePost kEmptyTradePost{PlayerColor::kEmpty, TradePostType::kNone};
    const Move kPassMove{.type = ActionType::kPass};

    std::string PlayerColorToString(PlayerColor c);
    PlayerColor StringToPlayerColor(const std::string &s);
    std::string MeepleColorToString(MeepleColor mc);
    char PlayerColorToChar(PlayerColor pc);
    void LogFromPython(LogLevel level, const std::string& message);

    std::ostream& operator<<(std::ostream& os, const PlayerColor& pc);
    std::ostream& operator<<(std::ostream& os, const MeepleColor& mc);
    std::ostream& operator<<(std::ostream& os, const TradePostType& tpt);
    std::ostream &operator<<(std::ostream &os, const Phase &phase);
    std::ostream &operator<<(std::ostream &os, const ActionType &move_type);

    inline std::string GetCurrentDateTime() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y%m%d_%H%M%S");
        return ss.str();
    }  
    inline std::string StrLower(const std::string &s)
    {
      std::string result = s;
      std::transform(result.begin(), result.end(), result.begin(),
                     [](unsigned char c){ return std::tolower(c); });
      return result;
    }

    // Structure to hold the counts of different legal move types.
    struct LegalActionCounts {
        int pass_moves = 0;
        int place_token_moves = 0;
        int mancala_moves = 0;
        int upgrade_moves = 0;
        int income_moves = 0;
        int trade_route_create_moves = 0;
    };

    // Structure to hold the complete result of legal action generation.
    struct LegalActionsResult {
        std::vector<Action> actions;
        LegalActionCounts counts;
    };

    class GoodsManager {
    public:
        // Singleton access pattern
        static const GoodsManager& GetInstance() {
            static GoodsManager instance;
            return instance;
        }

        int GetCommonGoodIndex(const std::string& good_name) const;
        int GetRareGoodIndex(const std::string& good_name) const;
        const std::vector<std::string>& GetCommonGoodsList() const { return common_goods_list_; }
        const std::vector<std::string>& GetRareGoodsList() const { return rare_goods_list_; }

    private:
        // Private constructor for singleton pattern
        GoodsManager();

        std::vector<std::string> common_goods_list_;
        std::vector<std::string> rare_goods_list_;
        std::map<std::string, int> common_good_to_index_;
        std::map<std::string, int> rare_good_to_index_;
    };

  } // namespace mali_ba
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_MALI_BA_COMMON_H_