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

#include "open_spiel/games/mali_ba/mali_ba.h"

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/random/uniform_int_distribution.h"
#include "open_spiel/abseil-cpp/absl/strings/str_split.h"
#include "open_spiel/abseil-cpp/absl/types/optional.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/games/mali_ba/mali_ba_board.h"
#include "open_spiel/spiel.h"
#include "open_spiel/spiel_globals.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/tests/basic_tests.h"

namespace open_spiel {
namespace mali_ba {
namespace {

namespace testing = open_spiel::testing;

void BasicMaliBaTests() {
  testing::LoadGameTest("mali_ba");
  testing::NoChanceOutcomesTest(*LoadGame("mali_ba"));
  testing::RandomSimTest(*LoadGame("mali_ba"), 10);
  testing::RandomSimTestWithUndo(*LoadGame("mali_ba"), 10);
}

void MoveGenerationTests() {
  // Create a default board
  Mali_BaBoard board = MakeDefaultBoard();
  
  // Test move generation
  int move_count = 0;
  board.GenerateLegalMoves([&move_count](const Move& move) {
    ++move_count;
    return true;
  });
  
  // A non-empty board should always have at least the pass move available
  SPIEL_CHECK_GE(move_count, 1);
}

void ActionConversionTests() {
  std::shared_ptr<const Game> game = LoadGame("mali_ba");
  
  // Create initial state
  std::unique_ptr<State> state = game->NewInitialState();
  const Mali_BaState* mali_ba_state = static_cast<const Mali_BaState*>(state.get());
  
  // Get legal actions
  std::vector<Action> legal_actions = state->LegalActions();
  
  // Test that all actions can be converted to moves and back
  for (Action action : legal_actions) {
    // Skip pass action for this test
    if (action == kPassActionId) continue;
    
    Move move = ActionToMove(action, mali_ba_state->Board());
    Action action_from_move = MoveToAction(move, mali_ba_state->Board().GridRadius());
    SPIEL_CHECK_EQ(action, action_from_move);
  }
}

void HexCoordinateTests() {
  // Test basic hex coordinate properties
  HexCoord hex1(1, -1, 0);
  HexCoord hex2(1, 0, -1);
  
  // Test equality
  SPIEL_CHECK_NE(hex1, hex2);
  
  // Test conversion to/from index
  int grid_radius = 5;
  int index1 = HexToIndex(hex1, grid_radius);
  int index2 = HexToIndex(hex2, grid_radius);
  
  SPIEL_CHECK_NE(index1, index2);
  
  HexCoord hex1_back = IndexToHex(index1, grid_radius);
  HexCoord hex2_back = IndexToHex(index2, grid_radius);
  
  SPIEL_CHECK_EQ(hex1, hex1_back);
  SPIEL_CHECK_EQ(hex2, hex2_back);
  
  // Test adjacency
  SPIEL_CHECK_TRUE(AreAdjacent(hex1, hex1 + kHexDirections[0]));
  SPIEL_CHECK_FALSE(AreAdjacent(hex1, hex2));
  
  // Test getting neighbors
  auto neighbors = GetNeighbors(hex1);
  SPIEL_CHECK_EQ(neighbors.size(), 6);
  
  // All neighbors should be adjacent to the original hex
  for (const auto& neighbor : neighbors) {
    SPIEL_CHECK_TRUE(AreAdjacent(hex1, neighbor));
  }
}

void TradePostTests() {
  Mali_BaBoard board = MakeDefaultBoard();
  
  // Test trading post placement
  HexCoord test_hex(1, -1, 0);
  
  // Initially there should be no trading post
  SPIEL_CHECK_EQ(board.post_at(test_hex).type, TradePostType::kNone);
  
  // Place a trading post
  board.set_post(test_hex, {Color::kBlack, TradePostType::kPost});
  
  // Check that the trading post was placed correctly
  SPIEL_CHECK_EQ(board.post_at(test_hex).owner, Color::kBlack);
  SPIEL_CHECK_EQ(board.post_at(test_hex).type, TradePostType::kPost);
  
  // Upgrade to a trading center
  board.set_post(test_hex, {Color::kBlack, TradePostType::kCenter});
  
  // Check that the upgrade worked
  SPIEL_CHECK_EQ(board.post_at(test_hex).type, TradePostType::kCenter);
}

void CityRegionTests() {
  Mali_BaBoard board = MakeDefaultBoard();
  
  // Check that cities were initialized
  const auto& cities = board.GetCities();
  SPIEL_CHECK_FALSE(cities.empty());
  
  // Test finding a city by location
  for (const City& city : cities) {
    auto found_city = board.CityAt(city.location);
    SPIEL_CHECK_TRUE(found_city.has_value());
    SPIEL_CHECK_EQ((*found_city)->name, city.name);
  }
  
  // Test with a location that doesn't have a city
  HexCoord non_city_hex(0, 1, -1);
  auto found_city = board.CityAt(non_city_hex);
  SPIEL_CHECK_FALSE(found_city.has_value());
}

void SerializationTests() {
  std::shared_ptr<const Game> game = LoadGame("mali_ba");
  
  // Test serialization of initial state
  std::unique_ptr<State> state = game->NewInitialState();
  std::string serialized = state->Serialize();
  
  // Deserialize and check equality
  std::unique_ptr<State> deserialized = game->DeserializeState(serialized);
  SPIEL_CHECK_EQ(state->ToString(), deserialized->ToString());
  
  // Apply some actions and test again
  const std::vector<Action>& legal_actions = state->LegalActions();
  if (!legal_actions.empty() && legal_actions[0] != kPassActionId) {
    state->ApplyAction(legal_actions[0]);
    serialized = state->Serialize();
    deserialized = game->DeserializeState(serialized);
    SPIEL_CHECK_EQ(state->ToString(), deserialized->ToString());
  }
}

}  // namespace
}  // namespace mali_ba
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::mali_ba::BasicMaliBaTests();
  open_spiel::mali_ba::MoveGenerationTests();
  open_spiel::mali_ba::ActionConversionTests();
  open_spiel::mali_ba::HexCoordinateTests();
  open_spiel::mali_ba::TradePostTests();
  open_spiel::mali_ba::CityRegionTests();
  open_spiel::mali_ba::SerializationTests();
  
  return 0;
}