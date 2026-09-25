# Usage: python analyze_log.py /home/robp/Downloads/train_run5.log
 
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import collections
import os
import random
# Import pyspiel here so all classes and functions in this file can see it.
try:
    import pyspiel
except ImportError:
    # Handle case where this file might be imported in an environment
    # without OpenSpiel built, e.g., for documentation generation.
    # This prevents the program from crashing on import.
    pyspiel = None
from pyspiel.mali_ba import log, LogLevel


# Extra functions
def get_training_parameters_from_game(game):
    """Extract training parameters from the C++ game object."""
    try:
        # Access training parameters from the game
        mali_ba_game = pyspiel.mali_ba.downcast_game(game)
        training_params = mali_ba_game.get_training_parameters()
        return {
            'draw_penalty': training_params.draw_penalty,
            'max_moves_penalty': training_params.max_moves_penalty,
            'quick_win_bonus': training_params.quick_win_bonus,
            'quick_win_threshold': training_params.quick_win_threshold
        }
    except Exception as e:
        print(f"Warning: Could not load training parameters from game: {e}")
        return {
            'draw_penalty': 0.0,  
            'max_moves_penalty': -0.2, 
            'quick_win_bonus': 0.2,
            'quick_win_threshold': 150
        }

# EMERGENCY MEMORY CLEANUP FUNCTION
def emergency_cleanup():
    """Call this if memory usage gets too high."""
    import gc
    import tensorflow as tf
    
    # Force garbage collection
    gc.collect()
    
    # Clear TensorFlow session if possible
    try:
        tf.keras.backend.clear_session()
    except:
        pass
    
    log(LogLevel.WARN, "Emergency memory cleanup performed")

# 7. PROCESS MEMORY MONITORING DECORATOR
def monitor_memory(func):
    """Decorator to monitor memory usage of functions."""
    import psutil
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        before = process.memory_info().rss / 1024 / 1024
        
        result = func(*args, **kwargs)
        
        after = process.memory_info().rss / 1024 / 1024
        if after - before > 50:  # If function used more than 50MB
            log(LogLevel.WARN, f"{func.__name__} used {after - before:.1f} MB")
        
        return result
    return wrapper

# A simple replay buffer
class ReplayBuffer:
    """Replay buffer with four pools: bootstrap, MCTS-natural, MCTS-near-win, MCTS-rare-goods.

    mcts_buffer_fraction controls both the MCTS capacity share (vs bootstrap) and
    the target batch sampling ratio.  near_win_pool_fraction controls what fraction
    of the MCTS capacity is reserved for near-win timeout games.
    raregoods_pool_fraction controls what fraction is reserved for Rare goods natural wins
    (guaranteeing Rare goods exposure each batch regardless of self-play frequency).
    The natural pool gets the remaining MCTS fraction (Timbuktu wins).

    When any pool is short, the others compensate so the batch always reaches the
    full requested size.
    """
    def __init__(self, buffer_size, mcts_buffer_fraction: float = 0.8,
                 near_win_pool_fraction: float = 0.30,
                 raregoods_pool_fraction: float = 0.15,
                 replace_bootstrap_with_mcts: bool = False):
        mcts_cap         = max(1, int(buffer_size * mcts_buffer_fraction))
        bootstrap_cap    = max(1, buffer_size - mcts_cap)
        nearwin_cap      = max(1, int(mcts_cap * near_win_pool_fraction))
        raregoods_cap    = max(1, int(mcts_cap * raregoods_pool_fraction))
        natural_cap      = max(1, mcts_cap - nearwin_cap - raregoods_cap)

        self.bootstrap_buffer      = collections.deque(maxlen=bootstrap_cap)
        self.mcts_natural_buffer   = collections.deque(maxlen=natural_cap)
        self.mcts_nearwin_buffer   = collections.deque(maxlen=nearwin_cap)
        self.mcts_raregoods_buffer = collections.deque(maxlen=raregoods_cap)

        self.mcts_fraction               = mcts_buffer_fraction
        self.near_win_pool_fraction      = near_win_pool_fraction
        self.raregoods_pool_fraction     = raregoods_pool_fraction
        self.replace_bootstrap_with_mcts = replace_bootstrap_with_mcts

    def add(self, experience, is_bootstrap: bool = False, is_near_win: bool = False,
            is_rare_goods: bool = False):
        if is_bootstrap:
            self.bootstrap_buffer.append(experience)
        elif is_near_win:
            self.mcts_nearwin_buffer.append(experience)
        elif is_rare_goods:
            self.mcts_raregoods_buffer.append(experience)
        else:
            if (self.replace_bootstrap_with_mcts
                    and len(self.mcts_natural_buffer) == self.mcts_natural_buffer.maxlen):
                # Natural pool is full. Promote the oldest natural win into the bootstrap
                # pool before the deque evicts it — seeds bootstrap from scratch if empty.
                self.bootstrap_buffer.append(self.mcts_natural_buffer[0])
            self.mcts_natural_buffer.append(experience)

    def sample(self, batch_size):
        have_bootstrap  = len(self.bootstrap_buffer) > 0
        have_natural    = len(self.mcts_natural_buffer) > 0
        have_nearwin    = len(self.mcts_nearwin_buffer) > 0
        have_raregoods  = len(self.mcts_raregoods_buffer) > 0
        have_mcts       = have_natural or have_nearwin or have_raregoods

        # Target counts based on configured fractions.
        n_mcts_target      = max(1, round(batch_size * self.mcts_fraction))
        n_nearwin_target   = max(0, round(n_mcts_target * self.near_win_pool_fraction))
        n_raregoods_target = max(0, round(n_mcts_target * self.raregoods_pool_fraction))
        n_natural_target   = n_mcts_target - n_nearwin_target - n_raregoods_target

        samples = []

        if have_mcts and have_bootstrap:
            # Draw from each MCTS sub-pool; let bootstrap fill any shortfall.
            n_natural   = min(n_natural_target,   len(self.mcts_natural_buffer))   if have_natural   else 0
            n_nearwin   = min(n_nearwin_target,   len(self.mcts_nearwin_buffer))   if have_nearwin   else 0
            n_raregoods = min(n_raregoods_target, len(self.mcts_raregoods_buffer)) if have_raregoods else 0
            n_mcts      = n_natural + n_nearwin + n_raregoods
            n_bootstrap = min(batch_size - n_mcts, len(self.bootstrap_buffer))
            if n_natural   > 0: samples += random.sample(list(self.mcts_natural_buffer),   n_natural)
            if n_nearwin   > 0: samples += random.sample(list(self.mcts_nearwin_buffer),   n_nearwin)
            if n_raregoods > 0: samples += random.sample(list(self.mcts_raregoods_buffer), n_raregoods)
            if n_bootstrap > 0: samples += random.sample(list(self.bootstrap_buffer),      n_bootstrap)
        elif have_mcts:
            # Bootstrap is empty; MCTS pools must fill the full batch_size.
            n_natural   = min(n_natural_target,   len(self.mcts_natural_buffer))   if have_natural   else 0
            n_nearwin   = min(n_nearwin_target,   len(self.mcts_nearwin_buffer))   if have_nearwin   else 0
            n_raregoods = min(n_raregoods_target, len(self.mcts_raregoods_buffer)) if have_raregoods else 0
            shortfall   = batch_size - n_natural - n_nearwin - n_raregoods
            # Distribute shortfall to pools with remaining capacity, priority: natural → nearwin → raregoods.
            if shortfall > 0 and have_natural:
                extra = min(shortfall, len(self.mcts_natural_buffer) - n_natural)
                n_natural += extra; shortfall -= extra
            if shortfall > 0 and have_nearwin:
                extra = min(shortfall, len(self.mcts_nearwin_buffer) - n_nearwin)
                n_nearwin += extra; shortfall -= extra
            if shortfall > 0 and have_raregoods:
                extra = min(shortfall, len(self.mcts_raregoods_buffer) - n_raregoods)
                n_raregoods += extra; shortfall -= extra
            if n_natural   > 0: samples += random.sample(list(self.mcts_natural_buffer),   n_natural)
            if n_nearwin   > 0: samples += random.sample(list(self.mcts_nearwin_buffer),   n_nearwin)
            if n_raregoods > 0: samples += random.sample(list(self.mcts_raregoods_buffer), n_raregoods)
        else:
            n = min(batch_size, len(self.bootstrap_buffer))
            samples = random.sample(list(self.bootstrap_buffer), n)

        random.shuffle(samples)
        return np.array(samples, dtype=object)

    def __len__(self):
        return (len(self.bootstrap_buffer) +
                len(self.mcts_natural_buffer) +
                len(self.mcts_nearwin_buffer) +
                len(self.mcts_raregoods_buffer))

class SimpleAgent:
    # ** Accept num_players in constructor **
    def __init__(self, observation_shape, num_actions, num_players, learning_rate=0.001):
        # Create two separate models
        self.policy_model = create_mali_ba_policy_network(observation_shape, num_actions)
        self.value_model = create_mali_ba_value_network(observation_shape, num_players)
        
        # Create two separate optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.value_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.num_players = num_players
    
    def save_model(self, path):
        """Saves the policy and value model weights to separate files."""
        try:
            # Ensure the directory exists before trying to save
            save_dir = os.path.dirname(path)
            if save_dir and not os.path.exists(save_dir):
                log(LogLevel.INFO, f"Agent: Creating directory for model saving: {save_dir}")
                os.makedirs(save_dir)

            policy_path = path.replace("weights.h5", "_policy.weights.h5")
            value_path = path.replace("weights.h5", "_value.weights.h5")

            log(LogLevel.INFO, f"Agent: Saving policy model weights to {policy_path}")
            self.policy_model.save_weights(policy_path)
            
            log(LogLevel.INFO, f"Agent: Saving value model weights to {value_path}")
            self.value_model.save_weights(value_path)
            
            log(LogLevel.INFO, "Agent: Model weights saved successfully.")
            
        except Exception as e:
            log(LogLevel.ERROR, f"Agent: An unexpected error occurred during model saving to path '{path}': {e}")
            import traceback
            log(LogLevel.ERROR, f"Agent: Full traceback: {traceback.format_exc()}")

    def load_model(self, path):
        # Load both models' weights
        self.policy_model.load_weights(path.replace("weights.h5", "_policy.weights.h5"))
        self.value_model.load_weights(path.replace("weights.h5", "_value.weights.h5"))

    
    def train(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return None

        try:
            samples = replay_buffer.sample(batch_size)
            observations, policy_targets, value_data_list = zip(*samples)

            # --- Data Preparation ---
            observations_flat = np.array(observations)
            target_shape_3d = self.policy_model.input_shape[1:]
            observations_reshaped = observations_flat.reshape((-1, *target_shape_3d))

            policy_targets = np.array(policy_targets)
            
            full_value_targets = np.zeros((batch_size, self.num_players))
            for i, (player_id, player_value, all_discounted_returns) in enumerate(value_data_list):
                if i >= batch_size: break
                for p in range(self.num_players):
                    if p < len(all_discounted_returns):
                        full_value_targets[i, p] = all_discounted_returns[p]
            
            # --- Input Sanity Checks ---
            if np.any(np.isnan(observations_reshaped)) or np.any(np.isinf(observations_reshaped)):
                log(LogLevel.ERROR, "Trainer: NaN/Inf detected in observation data. Skipping batch.")
                return None
            if np.any(np.isnan(policy_targets)) or np.any(np.isinf(policy_targets)):
                log(LogLevel.ERROR, "Trainer: NaN/Inf detected in policy target data. Skipping batch.")
                return None
            if np.any(np.isnan(full_value_targets)) or np.any(np.isinf(full_value_targets)):
                log(LogLevel.ERROR, "Trainer: NaN/Inf detected in value target data. Skipping batch.")
                return None

            # --- Policy Model Training Step ---
            with tf.GradientTape() as tape:
                predicted_policy = self.policy_model(observations_reshaped, training=True)
                policy_loss = tf.keras.losses.CategoricalCrossentropy()(policy_targets, predicted_policy)
            
            if tf.math.is_nan(policy_loss) or tf.math.is_inf(policy_loss):
                log(LogLevel.ERROR, f"Trainer: Invalid policy loss detected: {policy_loss}. Skipping batch.")
                return None
            
            policy_grads = tape.gradient(policy_loss, self.policy_model.trainable_variables)
            if any(g is None for g in policy_grads):
                log(LogLevel.ERROR, "Trainer: None gradients detected for policy model. Skipping batch.")
                return None
            self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_model.trainable_variables))
            
            # --- Value Model Training Step ---
            with tf.GradientTape() as tape:
                predicted_value = self.value_model(observations_reshaped, training=True)
                value_loss = tf.keras.losses.MeanSquaredError()(full_value_targets, predicted_value)

            # DEBUG =====================================================================
            log(LogLevel.INFO, f"Training: Observation range: min={np.min(observations_reshaped):.6f}, max={np.max(observations_reshaped):.6f}")
            log(LogLevel.INFO, f"Training: Observation mean={np.mean(observations_reshaped):.6f}, std={np.std(observations_reshaped):.6f}")
            log(LogLevel.INFO, f"Training: Policy target range: min={np.min(policy_targets):.6f}, max={np.max(policy_targets):.6f}")
            log(LogLevel.INFO, f"Training: Policy target sum per sample: {np.sum(policy_targets, axis=1)[:5]}")  # Should be 1.0 for each
            # END DEBUG =====================================================================

            if tf.math.is_nan(value_loss) or tf.math.is_inf(value_loss):
                log(LogLevel.ERROR, f"Trainer: Invalid value loss detected: {value_loss}. Skipping batch.")
                return None

            value_grads = tape.gradient(value_loss, self.value_model.trainable_variables)
            if any(g is None for g in value_grads):
                log(LogLevel.ERROR, "Trainer: None gradients detected for value model. Skipping batch.")
                return None
            self.value_optimizer.apply_gradients(zip(value_grads, self.value_model.trainable_variables))

            # --- Logging and Return ---
            total_loss = policy_loss + value_loss
            log(LogLevel.INFO, f"Training: Total Loss={total_loss.numpy():.4f} (Policy={policy_loss.numpy():.4f}, Value={value_loss.numpy():.4f})")
            
            # DEBUG ==================================================================
            log(LogLevel.INFO, f"Training: Gradient norms: {[tf.norm(g).numpy() for g in value_grads[:3]]}")
            log(LogLevel.INFO, f"Training: Policy output sample: {predicted_policy[0][:10].numpy()}")
            log(LogLevel.INFO, f"Training: Value output sample: {predicted_value[0].numpy()}")
            log(LogLevel.INFO, f"Training: Total loss: {total_loss.numpy():.6f}, Policy loss: {policy_loss.numpy():.6f}, Value loss: {value_loss.numpy():.6f}")
            # END DEBUG ==================================================================
            
            return total_loss.numpy()

        except Exception as e:
            log(LogLevel.ERROR, f"Trainer: An unexpected error occurred during training: {e}")
            import traceback
            log(LogLevel.ERROR, f"Trainer: Full traceback: {traceback.format_exc()}")
            return None


def _get_shared_infer_fn(policy_model, value_model, obs_shape):
    """Return a traced tf.function that runs the policy and value nets in one call.

    Two reasons this exists:
      * A fixed input_signature stops TF from re-tracing and skips the eager
        dispatch that dominates batch-1 CPU inference for nets this small --
        the graph call is several times cheaper than `model(x, training=False)`.
      * Running both nets inside one traced function removes a second
        Python -> TF round trip per evaluated state.

    The traced function is cached on the policy model, not on the evaluator:
    AlphaZeroEvaluator is rebuilt for every game while the models live for the
    whole actor process, and tracing costs about a second. This stays correct
    across weight refreshes because `set_weights()` assigns into the existing
    variables rather than replacing them, so the traced graph always reads the
    current weights.
    """
    fn = getattr(policy_model, '_mali_ba_infer_fn', None)
    if fn is not None:
        return fn

    spec = tf.TensorSpec(shape=(1, *obs_shape), dtype=tf.float32)

    @tf.function(input_signature=[spec])
    def _infer(obs_batch):
        return (policy_model(obs_batch, training=False),
                value_model(obs_batch, training=False))

    policy_model._mali_ba_infer_fn = _infer
    return _infer


class AlphaZeroEvaluator:
    """An evaluator for MCTS that uses a trained neural network.

    prior() and evaluate() are served from a single cached forward pass per
    state. MCTS reaches a leaf and calls evaluate() on it, then only calls
    prior() on a later simulation that descends through that node again (see
    mcts.py _apply_tree_policy / mcts_search), so an uncached evaluator pays
    two full inferences for every node it expands. The cache also absorbs
    repeated descents and transpositions within a single search.

    The C++ heuristic prior is still computed lazily, in prior() only. Most
    leaves are evaluated once and never expanded, so folding it into the shared
    forward pass would add work rather than save it.
    """

    def __init__(self, game, policy_model, value_model, heuristic_guidance_weight=0.40,
                 cache_size=8192):
        self._game = game
        self._policy_model = policy_model
        self._value_model = value_model
        self._shape = game.observation_tensor_shape()
        self._num_actions = game.num_distinct_actions()
        self.heuristic_guidance_weight = heuristic_guidance_weight
        self._infer = _get_shared_infer_fn(policy_model, value_model, self._shape)
        # key -> [value_vector, raw_nn_policy or None, mixed legal prior or None]
        self._cache = collections.OrderedDict()
        self._cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

    def cache_stats(self):
        """Returns (hits, misses, hit_rate) since construction."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits, self.cache_misses, (self.cache_hits / total) if total else 0.0

    def clear_cache(self):
        self._cache.clear()

    def _entry(self, state):
        """Fetch or compute the cache entry for a state: [value, raw_policy, prior]."""
        key = tuple(state.history())
        entry = self._cache.get(key)
        if entry is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
            return entry

        self.cache_misses += 1
        obs = np.asarray(state.observation_tensor(), dtype=np.float32)
        policy_t, value_t = self._infer(obs.reshape((1, *self._shape)))
        entry = [value_t[0].numpy(), policy_t[0].numpy(), None]

        self._cache[key] = entry
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return entry

    def evaluate(self, state):
        if state.is_terminal():
            return np.array(state.returns(), dtype=np.float32)
        return self._entry(state)[0]

    def prior(self, state):
        if state.is_terminal():
            return []

        legal_actions = state.legal_actions()
        if not legal_actions:
            return []

        entry = self._entry(state)
        if entry[2] is not None:
            # Copy: MCTS shuffles the returned list in place (mcts.py), and this
            # one is shared with every later visit to the same node.
            return list(entry[2])

        policy_full = entry[1]

        # --- Mix the network policy with the C++ heuristic prior ---
        # final_policy = (1 - w) * P_nn + w * P_heuristic
        w = self.heuristic_guidance_weight
        if w > 0.0:
            mali_ba_state = pyspiel.mali_ba.downcast_state(state)
            action_weights_map = mali_ba_state.get_heuristic_action_weights()
            total_weight = sum(action_weights_map.values())
            if total_weight > 0:
                policy_heuristic_full = np.zeros(self._num_actions, dtype=np.float32)
                for act, weight in action_weights_map.items():
                    policy_heuristic_full[act] = weight / total_weight
                policy_full = (1.0 - w) * policy_full + w * policy_heuristic_full

        legal_policy = [(action, policy_full[action]) for action in legal_actions]
        total_prob = sum(p for _, p in legal_policy)
        if total_prob > 0:
            legal_policy = [(action, p / total_prob) for action, p in legal_policy]
        else:
            # Fallback to uniform if all mixed probabilities are zero
            uniform_prob = 1.0 / len(legal_actions)
            legal_policy = [(action, uniform_prob) for action in legal_actions]

        # Keep only the mixed legal prior. The full-width NN vector is no longer
        # needed for this state and is what would dominate cache memory.
        entry[1] = None
        entry[2] = legal_policy
        return list(legal_policy)


# ** Accept num_players to build the correct output shape **
def create_mali_ba_policy_network(observation_shape, num_actions):
    """Creates the policy network."""
    inputs = layers.Input(shape=observation_shape)
    # Use a slightly simpler body for the policy net
    x = layers.Conv2D(128, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    for _ in range(5): # Fewer residual blocks
        residual = x
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(128, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
    
    policy_head = layers.Conv2D(4, 1, padding='same')(x)
    policy_head = layers.BatchNormalization()(policy_head)
    policy_head = layers.Activation('relu')(policy_head)
    policy_head = layers.Flatten()(policy_head)
    policy_head = layers.Dense(num_actions, activation='softmax', name='policy')(policy_head)
    
    return models.Model(inputs=inputs, outputs=policy_head)


def create_mali_ba_value_network(observation_shape, num_players):
    """Creates the value network."""
    inputs = layers.Input(shape=observation_shape)
    # Use a slightly simpler body for the value net as well
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    for _ in range(3): # Fewer residual blocks
        residual = x
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.add([x, residual])
        x = layers.Activation('relu')(x)
        
    value_head = layers.Conv2D(1, 1, padding='same')(x)
    value_head = layers.BatchNormalization()(value_head)
    value_head = layers.Activation('relu')(value_head)
    value_head = layers.Flatten()(value_head)
    value_head = layers.Dense(64, activation='relu')(value_head)
    value_head = layers.Dense(num_players, activation='tanh', name='value')(value_head)
    
    return models.Model(inputs=inputs, outputs=value_head)
