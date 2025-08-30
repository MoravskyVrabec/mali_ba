// mali_ba_game.h

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

    class Mali_BaGame : public Game
    {
    public:
      explicit Mali_BaGame(const GameParameters &params);

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
      int GetTokensPerPlayer() const { return tokens_per_player_; }
      uint_fast64_t GetRNGSeed() const { return rng_seed_; }
      bool GetPruneMovesForAI() const { return prune_moves_for_ai_; }
      const GameRules& GetRules() const { return rules_; }
      const HeuristicWeights& GetHeuristicWeights() const { return heuristic_weights_; }
      const TrainingParameters& GetTrainingParameters() const { return training_params_; }
      const std::vector<PlayerType>& GetPlayerTypes() const { return player_types_; }
      int GetMaxGameLength() { return MaxGameLength(); }

      // --- Board Configuration Accessors ---
      const std::set<HexCoord>& GetValidHexes() const { return valid_hexes_; }
      const std::vector<City>& GetCities() const { return cities_; }
      const std::set<HexCoord>& GetCoastalHexes() const { return coastal_hexes_; } 
      const City* GetCityAt(const HexCoord &location) const;
      int GetGridRadius() const { return grid_radius_; }
      // Returns the region ID for a given hex, or -1 if the hex is not in a region.
      int GetRegionForHex(const HexCoord& hex) const;
      std::string GetRegionName(int region_id) const;
      std::vector<int> GetValidRegionIds() const;

      // --- Board LUT Accessors ---
      int NumHexes() const { return num_hexes_; }
      int CoordToIndex(const HexCoord &hex) const;
      HexCoord IndexToCoord(int index) const;


    private:
      // --- Private Helper methods for INI parsing ---
      void ParseCustomCitiesFromString(const std::string& cities_str);
      std::set<HexCoord> ParseHexList(const std::string& hex_string) const;
      HexCoord ParseHexCoord(const std::string& coord_str) const;
      int FindCityIdByName(const std::string& name) const;
      std::string StripWhitespace(const std::string& str) const;
      std::vector<std::string> SplitString(const std::string& str, char delimiter) const;

      // --- Private Helper methods for board generation ---
      std::set<HexCoord> GenerateRegularBoard(int radius) const;
      std::vector<City> GetDefaultCitiesWithTimbuktu() const;

      // Helper to initialize LUTs during construction
      void InitializeLookups();

      // --- Game Configuration (read from parameters and INI) ---
      GameRules rules_;
      HeuristicWeights heuristic_weights_;
      TrainingParameters training_params_;
      int num_players_;
      int grid_radius_;
      int tokens_per_player_;
      bool LoggingEnabled_;
      bool enable_move_logging_;
      std::mt19937::result_type rng_seed_;
      bool prune_moves_for_ai_;
      std::vector<PlayerType> player_types_;

      // --- Player Configuration ---
      std::vector<PlayerColor> player_colors_;

      // --- Board Configuration ---
      std::set<HexCoord> valid_hexes_;
      std::set<HexCoord> coastal_hexes_;
      std::vector<City> cities_;
      int num_hexes_ = 0;

      // --- Board Hexes Lookup Tables (LUTs) ---
      absl::flat_hash_map<HexCoord, int> coord_to_index_map_;
      std::vector<HexCoord> index_to_coord_vec_;
      absl::flat_hash_map<HexCoord, int> hex_to_region_map_; // Member to store region data
      absl::flat_hash_map<int, std::string> region_id_to_name_map_; // For region names

    };

  } // namespace mali_ba
} // namespace open_spiel

#endif // OPEN_SPIEL_GAMES_MALI_BA_GAME_H_