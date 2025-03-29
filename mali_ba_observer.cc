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

#include "open_spiel/games/mali_ba/mali_ba_observer.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/games/mali_ba/mali_ba.h"
#include "open_spiel/games/mali_ba/mali_ba_board.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace mali_ba {

MaliBaObserver::MaliBaObserver(IIGObservationType iig_obs_type)
    : Observer(/*has_string=*/true, /*has_tensor=*/true),
      iig_obs_type_(iig_obs_type) {}

// Implement the WriteTensor method required by the Observer base class
void MaliBaObserver::WriteTensor(const State& state, int player,
                                Allocator* allocator) const {
  auto* mali_ba_state = static_cast<const Mali_BaState*>(&state);
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, mali_ba_state->NumPlayers());

  // Get the tensor shapes from the game
  const auto& shape = ObservationTensorShape();
  int num_planes = shape[0];
  int height = shape[1];
  int width = shape[2];
  
  // Get a tensor from the allocator
  SpanTensor tensor = allocator->Get("observation", {num_planes, height, width});
  auto values = tensor.data();
  
  // Fill with zeros
  std::fill(values.begin(), values.end(), 0.0);
  
  // Build the observation tensor using the helpers
  WritePublicInfo(*mali_ba_state, player, values);
  
  if (iig_obs_type_.private_info == PrivateInfoType::kSinglePlayer ||
      iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
    WritePrivateInfo(*mali_ba_state, player, values);
  }
  
  if (iig_obs_type_.public_info &&
      iig_obs_type_.private_info == PrivateInfoType::kAllPlayers) {
    WritePerfectInfo(*mali_ba_state, player, values);
  }
}

// Implement the StringFrom method required by the Observer base class
std::string MaliBaObserver::StringFrom(const State& state, int player) const {
  auto* mali_ba_state = static_cast<const Mali_BaState*>(&state);
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, mali_ba_state->NumPlayers());

  // Delegate to the state's ObservationString method
  return mali_ba_state->ObservationString(player);
}

// Helper method to write public information to the tensor
void MaliBaObserver::WritePublicInfo(const State& state, int player,
                                   absl::Span<float> values) const {
  auto* mali_ba_state = static_cast<const Mali_BaState*>(&state);
  const auto& board = mali_ba_state->Board();
  
  // Get the tensor shapes from the game
  const auto& shape = ObservationTensorShape();
  int num_planes = shape[0];
  int height = shape[1];
  int width = shape[2];
  
  // Fill in the basic board state, which is all public
  int grid_radius = board.GridRadius();
  
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
  int current_player = mali_ba_state->CurrentPlayer();
  
  // Set all values in current player plane to 1.0 if it matches
  if (current_player >= 0 && current_player < mali_ba_state->NumPlayers()) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        values[current_player_plane * height * width + i * width + j] = 
            (current_player == player) ? 1.0 : 0.0;
      }
    }
  }
  
  // Populate piece information
  for (int x = -grid_radius; x <= grid_radius; ++x) {
    for (int y = -grid_radius; y <= grid_radius; ++y) {
      int z = -x - y;
      
      // Skip invalid hex coordinates
      if (std::abs(x) + std::abs(y) + std::abs(z) > 2 * grid_radius) continue;
      
      HexCoord hex(x, y, z);
      if (!board.InBoardArea(hex)) continue;
      
      // Convert hex to board coordinates
      auto [col, row] = CubeToOffset(hex);
      // Adjust to tensor indices
      col += grid_radius;
      row += grid_radius;
      
      // Skip if out of bounds of the tensor
      if (row < 0 || row >= height || col < 0 || col >= width) continue;
      
      // Get piece at this hex
      const Piece& piece = board.at(hex);
      if (piece.type != MeepleType::kEmpty) {
        // Set the appropriate player plane
        if (piece.color != Color::kEmpty) {
          int player_plane = static_cast<int>(piece.color);
          values[player_plane * height * width + row * width + col] = 1.0;
        }
        
        // Set the appropriate meeple type plane
        int type_plane = 4 + static_cast<int>(piece.type) - 1; // -1 to adjust for kEmpty=0
        values[type_plane * height * width + row * width + col] = 1.0;
      }
      
      // Mark trading posts and centers
      const TradePost& post = board.post_at(hex);
      if (post.type != TradePostType::kNone && post.owner != Color::kEmpty) {
        int owner_plane = static_cast<int>(post.owner);
        int post_plane = (post.type == TradePostType::kPost) ? 
                          (10 + owner_plane) : (14 + owner_plane);
        values[post_plane * height * width + row * width + col] = 1.0;
      }
      
      // Mark cities
      auto city = board.CityAt(hex);
      if (city.has_value()) {
        int city_plane = 18;
        values[city_plane * height * width + row * width + col] = 1.0;
      }
    }
  }
}

// Helper method to write private information to the tensor
void MaliBaObserver::WritePrivateInfo(const State& state, int player,
                                    absl::Span<float> values) const {
  // Mali-Ba is a perfect information game, so there's no private info
  // This is just a placeholder
}

// Helper method to write perfect information to the tensor
void MaliBaObserver::WritePerfectInfo(const State& state, int player,
                                    absl::Span<float> values) const {
  // For Mali-Ba, all information is already public, 
  // so there's nothing extra to add here
}

// Factory function implementation to create Mali-Ba observer
std::shared_ptr<Observer> MakeMaliBaObserver(IIGObservationType iig_obs_type) {
  return std::make_shared<MaliBaObserver>(iig_obs_type);
}

}  // namespace mali_ba
}  // namespace open_spiel