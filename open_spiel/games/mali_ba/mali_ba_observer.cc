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

#include "open_spiel/games/mali_ba/mali_ba_common.h"
#include "open_spiel/games/mali_ba/mali_ba_observer.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"
#include "open_spiel/games/mali_ba/mali_ba_state.h"
#include "open_spiel/games/mali_ba/hex_grid.h"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "open_spiel/abseil-cpp/absl/algorithm/container.h"
#include "open_spiel/abseil-cpp/absl/strings/str_cat.h"
#include "open_spiel/spiel_utils.h"
#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/observer.h"
#include "open_spiel/abseil-cpp/absl/container/inlined_vector.h"

namespace open_spiel
{
  namespace mali_ba
  {

    namespace { // ANONYMOUS NAMESPACE HexToTensorCoordinates() implemented here
      // Max values needed for plane indexing
      constexpr int kMaxPlayers = 5;
      constexpr int kNumMeepleColors = 10;

      // Helper method to calculate tensor coordinates from a hex
      std::pair<int, int> HexToTensorCoordinates(const HexCoord &hex, int grid_radius) {
        // Convert hex to offset coordinates
        auto [col, row] = CubeToOffset(hex);

        // Adjust to tensor indices (centered on grid_radius)
        int adjusted_col = col + grid_radius;
        int adjusted_row = row + grid_radius;

        return {adjusted_row, adjusted_col};
      }
    } // END ANONYMOUS NAMESPACE

    MaliBaObserver::MaliBaObserver(IIGObservationType iig_obs_type)
        : Observer(/*has_string=*/true, /*has_tensor=*/true),
          iig_obs_type_(iig_obs_type) {}


    // Implement the WriteTensor method required by the Observer base class
    void MaliBaObserver::WriteTensor(const State &state, int player,
                                     Allocator *allocator) const
    {
      // Cast state to derived type
      auto *mali_ba_state = static_cast<const Mali_BaState *>(&state);
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, state.NumPlayers());

      // 1. ---- Get the Game object from the State ----
      std::shared_ptr<const Game> game = state.GetGame();
      SPIEL_CHECK_TRUE(game != nullptr);
      // 2. ---- CAST to the derived game type ----
      const Mali_BaGame *mali_ba_game = static_cast<const Mali_BaGame *>(game.get());
      SPIEL_CHECK_TRUE(mali_ba_game != nullptr);

      // Get shape from game
      const std::vector<int> &shape_vec = game->ObservationTensorShape();
      absl::InlinedVector<int, 4> shape_inlined(shape_vec.begin(), shape_vec.end());

      // Get tensor span from allocator
      SpanTensor tensor = allocator->Get("observation", shape_inlined);
      absl::Span<float> values = tensor.data(); // Use .data()

      // Ensure correct size
      SPIEL_CHECK_EQ(shape_inlined.size(), 3);
      int num_planes = shape_inlined[0];
      int height = shape_inlined[1];
      int width = shape_inlined[2];
      int expected_size = num_planes * height * width;
      SPIEL_CHECK_EQ(values.size(), expected_size);

      // Fill with zeros
      std::fill(values.begin(), values.end(), 0.0f);

      // --- Define Plane Indices ---
      int plane_idx = 0;
      const int player_token_base = plane_idx; // Planes 0-4
      plane_idx += kMaxPlayers;
      const int meeple_color_base = plane_idx; // Planes 5-14
      plane_idx += kNumMeepleColors;
      const int post_base = plane_idx; // Planes 15-19
      plane_idx += kMaxPlayers;
      const int center_base = plane_idx; // Planes 20-24
      plane_idx += kMaxPlayers;
      const int city_plane = plane_idx++;            // Plane 25
      const int current_player_plane = plane_idx++;  // Plane 26
      const int common_goods_total_base = plane_idx;     // Planes 27-31
      plane_idx += kMaxPlayers;
      const int rare_goods_total_base = plane_idx;       // Planes 32-36
      plane_idx += kMaxPlayers;
      const int player_potential_route_base = plane_idx; // Planes 37-41
      plane_idx += kMaxPlayers;
      const int player_active_route_base = plane_idx;    // Planes 42-46
      plane_idx += kMaxPlayers;
      // --- INDIVIDUAL GOODS PLANES ---
      const int individual_common_good_base = plane_idx; // Planes 47-61 (15 planes)
      plane_idx += 15;
      const int individual_rare_good_base = plane_idx;   // Planes 62-76 (15 planes)
      plane_idx += 15;

      // Check if calculated planes match expected shape
      SPIEL_CHECK_LE(plane_idx, num_planes);
      int expected_total_planes = 5 + 10 + kMaxPlayers + kMaxPlayers + 1 + 1 + kMaxPlayers + kMaxPlayers + kMaxPlayers + kMaxPlayers + 15 + 15;
      SPIEL_CHECK_EQ(plane_idx, expected_total_planes);
      SPIEL_CHECK_EQ(expected_total_planes, num_planes); // Ensure matches shape definition

      int grid_radius = mali_ba_game->GetGridRadius();
      int HxW = height * width;                      // Calculate once

      // --- Fill Planes ---
      // Iterate through valid hexes
      for (const auto &hex : mali_ba_game->GetValidHexes())
      {
        int index = mali_ba_game->CoordToIndex(hex);
        if (index != -1)
        {

          auto [row, col] = HexToTensorCoordinates(hex, grid_radius);

          // Skip if hex is outside the tensor bounds (conservative check)
          if (row < 0 || row >= height || col < 0 || col >= width)
            continue;

          int offset_base = row * width + col;

          // 1. Player Tokens
          PlayerColor token_owner = mali_ba_state->GetPlayerTokenAt(hex);
          if (token_owner != PlayerColor::kEmpty)
          {
            int token_plane = player_token_base + static_cast<int>(token_owner);
            if (token_plane >= player_token_base && token_plane < meeple_color_base)
            {
              values[token_plane * HxW + offset_base] = 1.0f;
            }
            else
            {
              std::cerr << "WriteTensor WARNING: Invalid plane index for player token color "
                        << static_cast<int>(token_owner) << std::endl;
            }
          }

          // 2. Meeples (Counts per color)
          const auto &meeples = mali_ba_state->GetMeeplesAt(hex);
          if (!meeples.empty())
          {
            std::map<MeepleColor, int> counts;
            for (MeepleColor mc : meeples)
            {
              counts[mc]++;
            }
            for (const auto &[mc, count] : counts)
            {
              if (mc != MeepleColor::kEmpty && count > 0)
              {
                int meeple_plane = meeple_color_base + static_cast<int>(mc);
                if (meeple_plane >= meeple_color_base && meeple_plane < post_base)
                {
                  values[meeple_plane * HxW + offset_base] = static_cast<float>(count);
                }
                else
                {
                  std::cerr << "WriteTensor WARNING: Invalid plane index for meeple color "
                            << static_cast<int>(mc) << std::endl;
                }
              }
            }
          }

          // ==================================================================
          // 3. Trade Posts & Centers
          // ==================================================================
          const auto &posts_at_hex = mali_ba_state->GetTradePostsAt(hex); // Get the vector of posts

          // Iterate through ALL posts at this hex
          for (const auto &post : posts_at_hex)
          {
            // Check if the current post is valid (has owner and type)
            if (post.type != TradePostType::kNone && post.owner != PlayerColor::kEmpty)
            {
              // Get the player ID for the owner of this post
              Player owner_id = mali_ba_state->GetPlayerId(post.owner);

              // Check if the owner is a valid, active player
              if (owner_id != kInvalidPlayer)
              {
                // Handle Trade Posts
                if (post.type == TradePostType::kPost)
                {
                  int post_plane = post_base + owner_id;
                  // Bounds check the calculated plane index
                  if (post_plane >= post_base && post_plane < center_base)
                  {
                    // Set the value in the tensor. Use 1.0 for presence.
                    // If multiple posts of the same type/owner were possible (they aren't),
                    // you might increment instead (+= 1.0f).
                    values[post_plane * HxW + offset_base] = 1.0f;
                  }
                  else
                  {
                    // Log a warning if the plane index is out of bounds
                    std::cerr << "WriteTensor WARNING: Invalid plane index " << post_plane
                              << " for post owner ID " << owner_id << " at hex " << hex.ToString() << std::endl;
                  }
                }
                // Handle Trading Centers
                else
                { // post.type == TradePostType::kCenter
                  int center_plane = center_base + owner_id;
                  // Bounds check the calculated plane index
                  if (center_plane >= center_base && center_plane < city_plane)
                  {
                    // Set the value in the tensor
                    values[center_plane * HxW + offset_base] = 1.0f;
                  }
                  else
                  {
                    // Log a warning if the plane index is out of bounds
                    std::cerr << "WriteTensor WARNING: Invalid plane index " << center_plane
                              << " for center owner ID " << owner_id << " at hex " << hex.ToString() << std::endl;
                  }
                }
              } // end if (owner_id != kInvalidPlayer)
            } // end if (post is valid)
          } // end for loop over posts_at_hex
          // ==================================================================
          // End Trade Post / Center Logic
          // ==================================================================

          // 4. Cities
          bool is_city = false;
          for (const auto &city : mali_ba_game->GetCities())
          {
            if (city.location == hex)
            {
              is_city = true;
              break;
            }
          }
          if (is_city)
          {
            values[city_plane * HxW + offset_base] = 1.0f;
          }
        } // end if (index != -1)
      } // End hex loop

      // 5. Current Player Plane (Fill uniformly)
      Player current_player_id = mali_ba_state->CurrentPlayer();
      if (current_player_id >= 0 && current_player_id < state.NumPlayers())
      {
        float value = (current_player_id == player) ? 1.0f : 0.0f;
        int plane_offset = current_player_plane * HxW;
        for (int i = 0; i < HxW; ++i)
        {
          values[plane_offset + i] = value;
        }
      }
      else if (current_player_id == kChancePlayerId)
      {
        // Leave as 0.0
      }

      // --- Fill Resource Total Planes ---
      for (Player p = 0; p < state.NumPlayers(); ++p)
      {
        // Calculate totals for player p
        int common_total = 0;
        const auto &common_map = mali_ba_state->GetPlayerCommonGoods(p);
        for (const auto &[name, count] : common_map)
        {
          common_total += count;
        }

        int rare_total = 0;
        const auto &rare_map = mali_ba_state->GetPlayerRareGoods(p);
        for (const auto &[name, count] : rare_map)
        {
          rare_total += count;
        }

        // Fill the corresponding planes uniformly with the total count
        int common_plane = common_goods_total_base + p;
        if (common_plane >= common_goods_total_base && common_plane < rare_goods_total_base)
        {
          int plane_offset = common_plane * HxW;
          for (int i = 0; i < HxW; ++i)
          {
            values[plane_offset + i] = static_cast<float>(common_total);
          }
        }
        else
        {
          std::cerr << "WriteTensor WARNING: Invalid plane index for common goods total for player " << p << std::endl;
        }

        int rare_plane = rare_goods_total_base + p;
        if (rare_plane >= rare_goods_total_base && rare_plane < num_planes)
        {
          int plane_offset = rare_plane * HxW;
          for (int i = 0; i < HxW; ++i)
          {
            values[plane_offset + i] = static_cast<float>(rare_total);
          }
        }
        else
        {
          std::cerr << "WriteTensor WARNING: Invalid plane index for rare goods total for player " << p << std::endl;
        }
      }

      // Note: We only fill these planes for the player whose perspective this is ('player').
      // The network doesn't need to know the exact inventory of opponents, just their totals
      // (which we already provide in planes 27-36).

      // 7. Individual Common Goods
      const auto& common_goods_map = mali_ba_state->GetPlayerCommonGoods(player);
      for (const auto& [good_name, count] : common_goods_map) {
          int good_index = GoodsManager::GetInstance().GetCommonGoodIndex(good_name);
          if (good_index != -1) {
              int plane = individual_common_good_base + good_index;
              int plane_offset = plane * HxW;
              // Fill the entire plane with the count for this good
              for (int i = 0; i < HxW; ++i) {
                  values[plane_offset + i] = static_cast<float>(count);
              }
          }
      }
      
      // 8. Individual Rare Goods
      const auto& rare_goods_map = mali_ba_state->GetPlayerRareGoods(player);
      for (const auto& [good_name, count] : rare_goods_map) {
          int good_index = GoodsManager::GetInstance().GetRareGoodIndex(good_name);
          if (good_index != -1) {
              int plane = individual_rare_good_base + good_index;
              int plane_offset = plane * HxW;
              // Fill the entire plane with the count for this good
              for (int i = 0; i < HxW; ++i) {
                  values[plane_offset + i] = static_cast<float>(count);
              }
          }
      }      
    } // End WriteTensor

    // Implement the StringFrom method required by the Observer base class
    std::string MaliBaObserver::StringFrom(const State &state, int player) const
    {
      auto *mali_ba_state = static_cast<const Mali_BaState *>(&state);
      SPIEL_CHECK_GE(player, 0);
      SPIEL_CHECK_LT(player, state.NumPlayers());

      // Delegate to the state's ObservationString method
      return mali_ba_state->ObservationString(player);
    }

    // Factory function implementation to create Mali-Ba observer
    std::shared_ptr<Observer> MakeMaliBaObserver(IIGObservationType iig_obs_type)
    {
      return std::make_shared<MaliBaObserver>(iig_obs_type);
    }

  } // namespace mali_ba
} // namespace open_spiel