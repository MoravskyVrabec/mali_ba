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

void BoardConfigurationTests() {
  std::shared_ptr<const Game> game = LoadGame("mali_ba(grid_radius=3)");
  auto mali_ba_game = static_cast<const Mali_BaGame*>(game.get());
  
  const auto& config = mali_ba_game->GetBoardConfig();
  
  // Test that the board configuration has the expected properties
  SPIEL_CHECK_EQ(config.board_radius, 3);
  SPIEL_CHECK_TRUE(config.regular_board);
  SPIEL_CHECK_FALSE(config.valid_hexes.empty());
  
  // Create an initial state
  std::unique_ptr<State> state = game->NewInitialState();
  auto mali_ba_state = static_cast<Mali_BaState*>(state.get());
  
  // Check that the state has the valid hexes from the configuration
  SPIEL_CHECK_FALSE(mali_ba_state->valid_hexes_.empty());
  SPIEL_CHECK_EQ(mali_ba_state->valid_hexes_.size(), config.valid_hexes.size());
  
  // Test IsValidHex method
  SPIEL_CHECK_TRUE(mali_ba_state->IsValidHex(HexCoord(0, 0, 0)));  // Origin should be valid
  SPIEL_CHECK_FALSE(mali_ba_state->IsValidHex(HexCoord(100, -100, 0)));  // Far outside should be invalid
}

void MoveAndActionConversionTests() {
  std::shared_ptr<const Game> game = LoadGame("mali_ba");
  
  // Create a test move
  Move move{Color::kBlack, HexCoord(0, 0, 0), {HexCoord(1, -1, 0)}, true};
  
  // Convert to action
  Action action = MoveToAction(move, 5);
  
  // Convert back to move
  Move back_move = ActionToMove(action, 5);
  
  // Check that round-trip conversion works
  SPIEL_CHECK_EQ(back_move.start_hex.x, move.start_hex.x);
  SPIEL_CHECK_EQ(back_move.start_hex.y, move.start_hex.y);
  SPIEL_CHECK_EQ(back_move.start_hex.z, move.start_hex.z);
  SPIEL_CHECK_EQ(back_move.path.size(), move.path.size());
  SPIEL_CHECK_EQ(back_move.path[0].x, move.path[0].x);
  SPIEL_CHECK_EQ(back_move.path[0].y, move.path[0].y);
  SPIEL_CHECK_EQ(back_move.path[0].z, move.path[0].z);
  SPIEL_CHECK_EQ(back_move.place_trading_post, move.place_trading_post);
  
  // Test pass move
  Move pass_move = kPassMove;
  action = MoveToAction(pass_move, 5);
  SPIEL_CHECK_EQ(action, kPassActionId);
  back_move = ActionToMove(action, 5);
  SPIEL_CHECK_TRUE(back_move.is_pass());
}

void GameStateTests() {
  std::shared_ptr<const Game> game = LoadGame("mali_ba");
  std::unique_ptr<State> state = game->NewInitialState();
  auto mali_ba_state = static_cast<Mali_BaState*>(state.get());
  
  // Check initial state
  SPIEL_CHECK_EQ(mali_ba_state->CurrentPlayer(), 0);  // Black starts
  SPIEL_CHECK_FALSE(mali_ba_state->IsTerminal());
  
  // Get legal actions
  std::vector<Action> legal_actions = state->LegalActions();
  SPIEL_CHECK_FALSE(legal_actions.empty());  // Should always have at least the pass action
  
  // Apply the pass action
  state->ApplyAction(kPassActionId);
  SPIEL_CHECK_EQ(mali_ba_state->CurrentPlayer(), 1);  // White's turn after Black passes
  
  // Apply another pass action
  state->ApplyAction(kPassActionId);
  SPIEL_CHECK_EQ(mali_ba_state->CurrentPlayer(), 0);  // Back to Black's turn
  
  // Test moving pieces
  if (legal_actions.size() > 1) {
    // Get a non-pass action
    Action move_action = legal_actions[1];  // Skip the pass action
    
    // Create a fresh state
    state = game->NewInitialState();
    
    // Apply the move action
    state->ApplyAction(move_action);
    
    // Check that the state changed
    SPIEL_CHECK_EQ(mali_ba_state->CurrentPlayer(), 1);  // Should be White's turn
  }
}

void SerializationTests() {
  std::shared_ptr<const Game> game = LoadGame("mali_ba");
  
  // Create an initial state
  std::unique_ptr<State> state = game->NewInitialState();
  
  // Serialize the state
  std::string serialized = state->Serialize();
  SPIEL_CHECK_FALSE(serialized.empty());
  
  // Deserialize the state
  std::unique_ptr<State> deserialized = game->DeserializeState(serialized);
  SPIEL_CHECK_TRUE(deserialized != nullptr);
  
  // Compare serialized representations
  SPIEL_CHECK_EQ(deserialized->Serialize(), serialized);
  
  // Apply some actions to create a non-trivial state
  std::vector<Action> legal_actions = state->LegalActions();
  if (!legal_actions.empty()) {
    state->ApplyAction(legal_actions[0]);
    serialized = state->Serialize();
    deserialized = game->DeserializeState(serialized);
    SPIEL_CHECK_EQ(deserialized->Serialize(), serialized);
  }
}

void ObserverTests() {
  std::shared_ptr<const Game> game = LoadGame("mali_ba");
  std::unique_ptr<State> state = game->NewInitialState();
  
  // Get a default observer
  auto observer = game->MakeObserver(
      IIGObservationType{/*public_info=*/true,
                        /*perfect_recall=*/false,
                        /*private_info=*/PrivateInfoType::kAllPlayers},
      GameParameters());
  
  // Create an observation tensor
  std::vector<float> tensor(game->ObservationTensorSize());
  
  // Create an allocator that will write into our tensor
  open_spiel::DimensionedSpan<float> span(tensor, 
      game->ObservationTensorShape());
  std::unordered_map<std::string, open_spiel::DimensionedSpan<float>> tensors;
  tensors["observation"] = span;
  open_spiel::TensorDict tensor_dict(tensors);
  
  // Write the observation
  observer->WriteTensor(*state, 0, &tensor_dict);
  
  // Check that the tensor has the expected size
  SPIEL_CHECK_EQ(tensor.size(), game->ObservationTensorSize());
  
  // Check that values are in the range [0, 1]
  for (float value : tensor) {
    SPIEL_CHECK_GE(value, 0.0);
    SPIEL_CHECK_LE(value, 1.0);
  }
}

}  // namespace
}  // namespace mali_ba
}  // namespace open_spiel

int main(int argc, char** argv) {
  open_spiel::mali_ba::BasicMaliBaTests();
  open_spiel::mali_ba::BoardConfigurationTests();
  open_spiel::mali_ba::MoveAndActionConversionTests();
  open_spiel::mali_ba::GameStateTests();
  open_spiel::mali_ba::SerializationTests();
  open_spiel::mali_ba::ObserverTests();
  
  return 0;
}