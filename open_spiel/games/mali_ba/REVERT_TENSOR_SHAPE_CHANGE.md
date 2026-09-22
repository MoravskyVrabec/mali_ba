# Revert: Per-Region Rare Good Observation Tensor Change

**Date applied:** 2026-05-14  
**Purpose of change:** Added 18 new planes to the observation tensor (3 players × 6 regions), one binary plane per player per region indicating whether that player holds at least one rare good from that region. This gives the neural network direct visibility into each player's progress toward the rare-good-each-region win condition, which was previously invisible (only total rare good count was encoded).

**Net effect on tensor shape:** 77 planes → 95 planes (for 3-player game with 6 regions).  
New shape: `[95, 15, 15]` (radius-7 custom board).

---

## Files Changed

### 1. `mali_ba_observer.cc`

Three separate changes in this file.

#### Change 1 — Add includes (near top of file, after existing `#include <algorithm>`)

**Add these two lines:**
```cpp
#include <set>
#include <unordered_map>
```

**To revert:** Remove those two lines.

---

#### Change 2 — Plane index block (around line 117–126 in the original)

**Original code:**
```cpp
      const int individual_rare_good_base = plane_idx;   // Planes 62-76 (15 planes)
      plane_idx += 15;

      // Check if calculated planes match expected shape
      SPIEL_CHECK_LE(plane_idx, num_planes);
      int expected_total_planes = 5 + 10 + kMaxPlayers + kMaxPlayers + 1 + 1 + kMaxPlayers + kMaxPlayers + kMaxPlayers + kMaxPlayers + 15 + 15;
      SPIEL_CHECK_EQ(plane_idx, expected_total_planes);
      SPIEL_CHECK_EQ(expected_total_planes, num_planes); // Ensure matches shape definition
```

**New code (what is currently in the file):**
```cpp
      const int individual_rare_good_base = plane_idx;   // Planes 62-76 (15 planes)
      plane_idx += 15;

      // Per-player, per-region rare good indicators.
      // One plane per (player, region) pair — 1.0 if that player holds at least one
      // rare good from that region, 0.0 otherwise. Regions are sorted by ID for
      // consistent ordering across game init and tensor writes.
      std::vector<int> sorted_region_ids = mali_ba_game->GetValidRegionIds();
      std::sort(sorted_region_ids.begin(), sorted_region_ids.end());
      const int num_regions_obs = static_cast<int>(sorted_region_ids.size());
      const int rare_good_region_base = plane_idx;  // Planes 77+ (num_players * num_regions)
      plane_idx += state.NumPlayers() * num_regions_obs;

      // Verify calculated plane count matches the shape declared at game init.
      SPIEL_CHECK_EQ(plane_idx, num_planes);
```

**To revert:** Replace the new code block with the original code block above.

---

#### Change 3 — Section 9 plane writing (at the end of WriteTensor, just before `} // End WriteTensor`)

**Original code** (section 8 was the last section; the closing brace followed immediately):
```cpp
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
```

**New code (what is currently in the file)** — section 9 block was inserted between section 8 and the closing brace:
```cpp
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

      // 9. Per-player, per-region rare good indicators.
      // Build a one-time lookup from rare good name -> region ID using city data,
      // then for each player write a binary plane per region (1 = covered, 0 = not).
      if (num_regions_obs > 0) {
          std::unordered_map<std::string, int> rare_good_to_region;
          for (const auto& city : mali_ba_game->GetCities()) {
              int region_id = mali_ba_game->GetRegionForHex(city.location);
              if (region_id != -1 && !city.rare_good.empty()) {
                  rare_good_to_region[city.rare_good] = region_id;
              }
          }

          for (Player p = 0; p < state.NumPlayers(); ++p) {
              std::set<int> covered;
              for (const auto& [good_name, count] : mali_ba_state->GetPlayerRareGoods(p)) {
                  if (count <= 0) continue;
                  auto it = rare_good_to_region.find(good_name);
                  if (it != rare_good_to_region.end()) {
                      covered.insert(it->second);
                  }
              }
              for (int ri = 0; ri < num_regions_obs; ++ri) {
                  float has_region = covered.count(sorted_region_ids[ri]) ? 1.0f : 0.0f;
                  int plane_offset = (rare_good_region_base + p * num_regions_obs + ri) * HxW;
                  for (int i = 0; i < HxW; ++i) {
                      values[plane_offset + i] = has_region;
                  }
              }
          }
      }
    } // End WriteTensor
```

**To revert:** Remove the entire section 9 block (everything between the end of section 8 and `} // End WriteTensor`).

---

### 2. `mali_ba_game.cc`

One change in this file, in the observation tensor shape initialization block inside the constructor.

**Original code:**
```cpp
            // Dynamically build the observation tensor
            int dimension = grid_radius_ * 2 + 1;
            constexpr int kNumPlanes = 77; // see mali_ba_observer.cc for info
            observation_tensor_shape_ = {kNumPlanes, dimension, dimension};
```

**New code (what is currently in the file):**
```cpp
            // Dynamically build the observation tensor shape.
            // Base 77 planes + one plane per (player, region) pair for the
            // per-region rare good indicator planes added in section 9 of WriteTensor.
            // Regions are sorted by ID to match the ordering used in the observer.
            int dimension = grid_radius_ * 2 + 1;
            std::vector<int> sorted_region_ids = GetValidRegionIds();
            std::sort(sorted_region_ids.begin(), sorted_region_ids.end());
            int kNumPlanes = 77 + num_players_ * static_cast<int>(sorted_region_ids.size());
            observation_tensor_shape_ = {kNumPlanes, dimension, dimension};
```

**To revert:** Replace the new block with the original 3-line block above.

---

## Rebuild After Reverting

```bash
conda activate mali_ba
touch open_spiel/games/mali_ba/mali_ba_observer.cc open_spiel/games/mali_ba/mali_ba_game.cc
cd build
make pyspiel -j$(nproc)
```

## Important Note

Reverting this change makes the model trained with the new tensor incompatible with the old tensor shape. Any checkpoint saved after this change was applied **cannot** be loaded after reverting. Start a new training run from scratch after reverting.
