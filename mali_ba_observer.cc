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
#include "open_spiel/games/mali_ba/hex_grid.h"
#include "open_spiel/spiel_utils.h"

namespace open_spiel {
namespace mali_ba {

MaliBaObserver::MaliBaObserver(IIGObservationType iig_obs_type)
    : Observer(/*has_string=*/true, /*has_tensor=*/true),
      iig_obs_type_(iig_obs_type) {}

// Helper method to calculate tensor coordinates from a hex
std::pair<int, int> MaliBaObserver::HexToTensorCoordinates(const HexCoord& hex, int grid_radius) const {
  // Convert hex to offset coordinates
  auto [col, row] = CubeToOffset(hex);
  
  // Adjust to tensor indices (centered on grid_radius)
  int adjusted_col = col + grid_radius;
  int adjusted_row = row + grid_radius;
  
  return {adjusted_row, adjusted_col};
}

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

// Helper method to write public information to the tensor
void MaliBaObserver::WritePublicInfo(const State& state, int player,
                                   absl::Span<float> values) const {
  auto* mali_ba_state = static_cast<const Mali_BaState*>(&state);
  
  // Get the tensor shapes from the game
  const auto& shape = ObservationTensorShape();
  int num_planes = shape[0];
  int height = shape[1];
  int width = shape[2];
  
  // Get grid radius from the state
  int grid_radius = mali_ba_state->grid_radius_;
  
  // Calculate the offset for different planes
  // Planes are organized as follows:
  // 0-3: Player pieces (4 colors)
  // 4-10: Meeple types (7 types)
  // 11-14: Trading posts (4 players)
  // 15-18: Trading centers (4 players)
  // 19: Cities
  
  // Mark the current player
  int current_player_plane = 0;
  int current_player = mali_ba_state->CurrentPlayer();
  
  if (current_player >= 0 && current_player < mali_ba_state->NumPlayers()) {
    for (int i = 0; i < height; ++i) {
      for (int j = 0; j < width; ++j) {
        values[current_player_plane * height * width + i * width + j] = 
            (current_player == player) ? 1.0 : 0.0;
      }
    }
  }
  
  // Populate piece information
  for (const auto& hex : mali_ba_state->valid_hexes_) {
    // Convert hex to tensor coordinates
    auto [row, col] = HexToTensorCoordinates(hex, grid_radius);
    
    // Skip if out of bounds of the tensor
    if (row < 0 || row >= height || col < 0 || col >= width) continue;
    
    // Get piece at this hex
    const Piece& piece = mali_ba_state->GetPiece(hex);
    if (piece.type != MeepleType::kEmpty) {
      // Set the appropriate player plane
      if (piece.color != Color::kEmpty) {
        int player_plane = static_cast<int>(piece.color);
        values[player_plane * height * width + row * width + col] = 1.0;
      }
      
      // Set the appropriate meeple type plane
      int type_plane = 4 + static_cast<int>(piece.type);
      values[type_plane * height * width + row * width + col] = 1.0;
    }
    
    // Get trading post at this hex
    auto trade_post_iter = mali_ba_state->trade_posts_.find(hex);
    if (trade_post_iter != mali_ba_state->trade_posts_.end()) {
      const TradePost& post = trade_post_iter->second;
      if (post.type != TradePostType::kNone && post.owner != Color::kEmpty) {
        int owner_int = static_cast<int>(post.owner);
        int post_plane = (post.type == TradePostType::kPost) ? 
                          (11 + owner_int) : (15 + owner_int);
        values[post_plane * height * width + row * width + col] = 1.0;
      }
    }
    
    // Mark cities
    bool is_city = false;
    for (const auto& city : mali_ba_state->cities_) {
      if (city.location == hex) {
        is_city = true;
        break;
      }
    }
    
    if (is_city) {
      int city_plane = 19;
      values[city_plane * height * width + row * width + col] = 1.0;
    }
  }
}

// Helper method to write private information to the tensor
void MaliBaObserver::WritePrivateInfo(const State& state, int player,
                                    absl::Span<float> values) const {
  // Mali-Ba is a perfect information game, so there's no private info
  // This is just a placeholder for potential future game variants
}

// Helper method to write perfect information to the tensor
void MaliBaObserver::WritePerfectInfo(const State& state, int player,
                                    absl::Span<float> values) const {
  // For Mali-Ba, all information is already public, 
  // so there's nothing extra to add here
  // This is just a placeholder for potential future game variants
}

// Implement the StringFrom method required by the Observer base class
std::string MaliBaObserver::StringFrom(const State& state, int player) const {
  auto* mali_ba_state = static_cast<const Mali_BaState*>(&state);
  SPIEL_CHECK_GE(player, 0);
  SPIEL_CHECK_LT(player, mali_ba_state->NumPlayers());

  // Delegate to the state's ObservationString method
  return mali_ba_state->ObservationString(player);
}

// Factory function implementation to create Mali-Ba observer
std::shared_ptr<Observer> MakeMaliBaObserver(IIGObservationType iig_obs_type) {
  return std::make_shared<MaliBaObserver>(iig_obs_type);
}

}  // namespace mali_ba
}  // namespace open_spiel