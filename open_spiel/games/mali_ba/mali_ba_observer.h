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

#ifndef OPEN_SPIEL_GAMES_MALI_BA_OBSERVER_H_
#define OPEN_SPIEL_GAMES_MALI_BA_OBSERVER_H_

#include <string>
#include <vector>

#include "open_spiel/abseil-cpp/absl/types/span.h"
#include "open_spiel/observer.h"
#include "open_spiel/games/mali_ba/mali_ba_game.h"

namespace open_spiel {
namespace mali_ba {

// A concrete Observer class for Mali-Ba game that implements the required
// pure virtual methods from the Observer base class.
class MaliBaObserver : public Observer {
 public:
  // Constructor for Mali-Ba observer
  MaliBaObserver(IIGObservationType iig_obs_type);

  // Implementation of required Observer methods with signatures matching base class
  void WriteTensor(const State& state, int player,
    Allocator* allocator) const override;

  std::string StringFrom(const State& state, int player) const override;

 private:
  // Helper methods to write different parts of the observation tensor
  // These methods take a state, player, and a span to write the tensor values into
  // Helpers should still take Span
  void WritePublicInfo(const State& state, int player,
    absl::Span<float> values) const;
  void WritePrivateInfo(const State& state, int player,
    absl::Span<float> values) const;
  void WritePerfectInfo(const State& state, int player,
    absl::Span<float> values) const;

  // The type of observation we're providing
  IIGObservationType iig_obs_type_;
};

// Factory function to create a Mali-Ba observer
std::shared_ptr<Observer> MakeMaliBaObserver(IIGObservationType iig_obs_type);

}  // namespace mali_ba
}  // namespace open_spiel

#endif  // OPEN_SPIEL_GAMES_MALI_BA_OBSERVER_H_
