"""Bot players for a GameSession.

Every bot has .name and .choose(state) -> action id. Two kinds:

  HeuristicBot   the engine's heuristic (select_heuristic_random_action); instant.
  NetworkBot     MCTS guided by a trained policy/value network, set up the way ab_eval
                 evaluates models: PUCT, uct_c 0.5, no Dirichlet noise, heuristic
                 guidance 0, normalized blend, and the most-visited move (greedy).
                 300 simulations by default; fewer is faster and weaker.

Token placement is random in self-play, so the networks never learned it; NetworkBot
places tokens with the heuristic.

Weights are named as ab_eval takes them: "mali_ba_agent_vD039.weights.h5" stands for
mali_ba_agent_vD039._policy.weights.h5 and ._value.weights.h5. Networks trained with
mover_relative_obs (D029 on) need mover_relative=True, the default.
"""
import os
import threading

DEFAULT_SIMS = 300
DEFAULT_UCT_C = 0.5

_models = {}                 # (policy path, value path) -> (policy model, value model)
_models_lock = threading.Lock()


def weights_paths(spec):
    """'.../name.weights.h5' -> (policy path, value path). Raises if either is missing."""
    if not spec.endswith(".weights.h5"):
        spec = spec + ".weights.h5"
    pol = spec.replace("weights.h5", "_policy.weights.h5")
    val = spec.replace("weights.h5", "_value.weights.h5")
    for p in (pol, val):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing weights file: {p}")
    return pol, val


def model_name(spec):
    """Display name: 'mali_ba_agent_vD039.weights.h5' -> 'D039'."""
    base = os.path.basename(spec).replace(".weights.h5", "").rstrip(".")
    return base.split("_v")[-1] if "_agent_v" in base else base


def _load_models(spec, game):
    """Loads (and caches) a network pair, so seats sharing a model share one copy."""
    pol, val = weights_paths(spec)
    with _models_lock:
        if (pol, val) not in _models:
            from mali_ba.training_utils import (create_mali_ba_policy_network,
                                                create_mali_ba_value_network)
            shape = game.observation_tensor_shape()
            pm = create_mali_ba_policy_network(shape, game.num_distinct_actions())
            vm = create_mali_ba_value_network(shape, game.num_players())
            pm.load_weights(pol)
            vm.load_weights(val)
            _models[(pol, val)] = (pm, vm)
        return _models[(pol, val)]


class HeuristicBot:
    kind = "heuristic"

    def __init__(self, game=None):
        self.name = "heuristic"

    def choose(self, state):
        import pyspiel
        return pyspiel.mali_ba.downcast_state(state).select_heuristic_random_action()


class NetworkBot:
    kind = "network"

    def __init__(self, game, weights, sims=DEFAULT_SIMS, uct_c=DEFAULT_UCT_C,
                 mover_relative=True, heuristic_weight=0.0, cache_size=20000):
        from open_spiel.python.algorithms import mcts
        from mali_ba.training_utils import AlphaZeroEvaluator
        pm, vm = _load_models(weights, game)
        self.weights = weights
        self.sims = int(sims)
        self.name = f"{model_name(weights)} ({self.sims} sims)"
        self.mover_relative = mover_relative
        self._ev = AlphaZeroEvaluator(game, pm, vm, heuristic_guidance_weight=heuristic_weight,
                                      cache_size=cache_size, blend_normalized=True,
                                      mover_relative=mover_relative)
        self._bot = mcts.MCTSBot(game=game, uct_c=uct_c, max_simulations=self.sims,
                                 evaluator=self._ev, solve=False, dirichlet_noise=None,
                                 child_selection_fn=mcts.SearchNode.puct_value,
                                 verbose=False)
        self._heuristic = HeuristicBot()

    def choose(self, state):
        import pyspiel
        ms = pyspiel.mali_ba.downcast_state(state)
        if ms.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
            return self._heuristic.choose(state)
        legal = state.legal_actions()
        if len(legal) == 1:
            return legal[0]
        if self.mover_relative:
            # One view for the whole search: the searching seat is slot 0 at every node.
            self._ev.set_relabel_seat(state.current_player())
        root = self._bot.mcts_search(state)
        return max(root.children, key=lambda c: c.explore_count).action


def make_bot(game, spec):
    """spec: {"type": "heuristic"} or {"type": "network", "weights": path,
    "sims": 300, "uct_c": 0.5, "mover_relative": true}."""
    t = spec.get("type")
    if t == "heuristic":
        return HeuristicBot(game)
    if t == "network":
        return NetworkBot(game, spec["weights"], sims=spec.get("sims", DEFAULT_SIMS),
                          uct_c=spec.get("uct_c", DEFAULT_UCT_C),
                          mover_relative=spec.get("mover_relative", True))
    raise ValueError(f"unknown bot type: {t!r}")
