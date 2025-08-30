import os
import sys
import argparse
import random
import time
import collections

# Path setup should be done early
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
build_dir = os.path.join(project_root, "build", "python")
python_games_dir = os.path.join(project_root, "open_spiel", "python", "games")
if build_dir not in sys.path: sys.path.insert(0, build_dir)
if python_games_dir not in sys.path: sys.path.insert(0, python_games_dir)

# We need to import the MP components at the top level for the __main__ guard
import multiprocessing as mp
from multiprocessing import Process, Queue

# Constant for gathering updated weights
WEIGHTS_UPDATE_INTERVAL_SECONDS = 15

# --- Child Process Functions ---

def trainer_process(args, initial_game_params, replay_buffer_queue, weights_queue, stats_queue):
    """A dedicated process for training the model."""
    # --- IMPORTS ARE THE VERY FIRST THING ---
    import tensorflow as tf
    import pyspiel
    from mali_ba.training_utils import SimpleAgent
    from pyspiel.mali_ba import log, LogLevel
    # This import can be removed, as the ReplayBuffer is local now
    # from mali_ba.classes.classes_other import ReplayBuffer

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Trainer GPU setup error: {e}")

    log(LogLevel.INFO, "Trainer process started.")

    temp_game = pyspiel.load_game(args.game_name, initial_game_params)
    agent = SimpleAgent(
        temp_game.observation_tensor_shape(),
        temp_game.num_distinct_actions(),
        temp_game.num_players(),
        args.learning_rate
    )
    del temp_game

    if args.load_model_path and os.path.exists(args.load_model_path.replace("weights.h5", "_policy.weights.h5")):
        try:
            agent.load_model(args.load_model_path)
            log(LogLevel.INFO, "Trainer loaded initial model weights.")
        except Exception as e:
            log(LogLevel.WARN, f"Could not load model weights: {e}. Starting from scratch.")
    else:
        log(LogLevel.WARN, f"Can't find load model file. Starting from scratch.")
    
    # ### <<< CORRECTION 1: Put a TUPLE of weights on the queue.
    weights_queue.put((agent.policy_model.get_weights(), agent.value_model.get_weights()))

    # Re-importing here is fine, or just remove the one at the top.
    from mali_ba.classes.classes_other import ReplayBuffer

    local_replay_buffer = ReplayBuffer(args.replay_buffer_size)
    training_counter = 0
    last_save_time = time.time()
    
    while True:
        # Process all available experiences without blocking
        experiences_processed = 0
        max_experiences_per_cycle = args.batch_size * 4  # Process in chunks
        
        try:
            while experiences_processed < max_experiences_per_cycle:
                experience = replay_buffer_queue.get_nowait()  # Non-blocking get
                if experience is None:
                    log(LogLevel.INFO, "Trainer received shutdown signal. Saving final model.")
                    agent.save_model(args.save_model_path)
                    return
                local_replay_buffer.add(experience)
                experiences_processed += 1
        except:  # Queue empty - this is normal and expected
            pass
        
        if experiences_processed > 0:
             # Log less frequently to reduce log spam
            if len(local_replay_buffer) % 100 < experiences_processed:
                log(LogLevel.INFO, f"Trainer processed {experiences_processed} new experiences. Buffer size: {len(local_replay_buffer)}")

        # Train in batches
        if len(local_replay_buffer) >= args.batch_size:
            # Check if we have enough new data to warrant a training step
            if experiences_processed > args.batch_size // 2 or len(local_replay_buffer) % (args.batch_size * 2) == 0 :
                log(LogLevel.INFO, f"Trainer: Starting training with buffer size {len(local_replay_buffer)}")
                try:
                    loss = agent.train(local_replay_buffer, args.batch_size) # Use the configured batch size
                    if loss is not None:
                        log(LogLevel.INFO, f"Trainer: Training successful, loss = {loss:.4f}")
                        stats_queue.put({"loss": loss})
                    else:
                        log(LogLevel.WARN, "Trainer: agent.train() returned None")
                except Exception as e:
                    log(LogLevel.ERROR, f"Trainer: Training failed with error: {e}")
                    import traceback
                    log(LogLevel.ERROR, f"Trainer: Full traceback: {traceback.format_exc()}")
                
                # Update weights for actors after a successful training step
                if weights_queue.empty():
                    # ### <<< CORRECTION 2: Put a TUPLE of weights on the queue here as well.
                    weights_queue.put((agent.policy_model.get_weights(), agent.value_model.get_weights()))

        # Periodic model saving
        current_time = time.time()
        if current_time - last_save_time > args.save_every * 60:
            log(LogLevel.INFO, f"Trainer: Save interval of {args.save_every} minutes reached. Attempting to save model.")
            try:
                agent.save_model(args.save_model_path)
                last_save_time = current_time # Update time ONLY on successful save attempt
            except Exception as e:
                # The agent's save_model will also log, but we add one here too.
                log(LogLevel.ERROR, f"Trainer: agent.save_model failed inside periodic save. Error: {e}")

        # Prune the buffer if it's almost full
        if len(local_replay_buffer) > args.replay_buffer_size * 0.95:
            log(LogLevel.INFO, "Pruning replay buffer...")
            remove_count = int(args.replay_buffer_size * 0.1)
            for _ in range(remove_count):
                if len(local_replay_buffer) > 0:
                    local_replay_buffer.buffer.popleft()

        # If there's nothing to do, sleep briefly to prevent busy-waiting
        if experiences_processed == 0 and len(local_replay_buffer) < args.batch_size:
            time.sleep(0.1)


def actor_process(actor_id, game_params, args, job_queue, result_queue, games_per_actor):
    # --- (Delayed imports are the same and correct) ---
    import numpy as np
    import random
    from open_spiel.python.algorithms import mcts
    import pyspiel
    from pyspiel import mali_ba
    from pyspiel.mali_ba import log, LogLevel
    import tensorflow as tf
    from mali_ba.training_utils import AlphaZeroEvaluator, create_mali_ba_policy_network, create_mali_ba_value_network

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')
    log(LogLevel.INFO, f"Actor {actor_id} started, configured for CPU-only execution.")

    # --- Create ONE game object and ONE model for the actor's lifetime ---
    log(LogLevel.INFO, f"Actor {actor_id}: Initializing its game instance and model.")
    game = pyspiel.load_game(args.game_name, game_params)
        # Get the max game length
    max_game_length = game.get_max_game_length()
    log(LogLevel.INFO, f"Using max game length: {max_game_length}")

    # Create instances of both networks
    policy_model = create_mali_ba_policy_network(game.observation_tensor_shape(), game.num_distinct_actions())
    value_model = create_mali_ba_value_network(game.observation_tensor_shape(), game.num_players())

    for _ in range(args.games_per_actor):
        job = job_queue.get()
        if job is None:  break

        episode_num, (policy_weights, value_weights), game_rng_seed = job
        
        # Set weights on the two separate models
        policy_model.set_weights(policy_weights)
        value_model.set_weights(value_weights)
        
        random.seed(game_rng_seed)
        np.random.seed(game_rng_seed)
        move_count = 0
        
        # --- Use the existing game object to create a new state ---
        # DO NOT RELOAD THE GAME.
        # seeded_game_params = game_params.copy()
        # seeded_game_params["rng_seed"] = game_rng_seed
        # game = pyspiel.load_game(args.game_name, seeded_game_params)
        # INSTEAD:
        state = game.new_initial_state()

        # NOTE: For MCTS, if you need the C++ RNG to be different for each game,
        # you would need a way to re-seed the game object's internal RNG.
        # A simple `game.set_rng_state(str(game_rng_seed))` exposed via pybind11
        # would be the proper way to handle this. For now, we proceed as the
        # MCTS bot's own randomness (dirichlet noise, policy sampling) will
        # provide sufficient exploration.

        # Pass both models to the evaluator
        evaluator = AlphaZeroEvaluator(game, policy_model, value_model)
        
        bot = mcts.MCTSBot(
            game=game, uct_c=args.uct_c, max_simulations=args.max_simulations,
            evaluator=evaluator, solve=False,
            # The first value (alpha) controls the shape of the noise. A lower
            # alpha creates more "spiky" noise, forcing exploration of
            # a few non-policy moves. The default is 0.3. Let's make it stronger.
            dirichlet_noise=(0.2, 0.25), 
            child_selection_fn=mcts.SearchNode.puct_value, verbose=False)

        state = game.new_initial_state()
        
        # Chance node startup
        episode_trajectory = []
        if state.is_chance_node():
            state.apply_action(state.legal_actions()[0])

        # Now do place tokens
        mali_ba_state = pyspiel.mali_ba.downcast_state(state)
        if mali_ba_state.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
            log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}: Starting token placement phase.")
            
            # --- START OF OPTIMIZATION ---
            while mali_ba_state.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
                if mali_ba_state.is_terminal(): break
                
                # For token placement, a simple uniform random choice is much faster.
                legal_actions = mali_ba_state.legal_actions()
                if not legal_actions:
                    log(LogLevel.WARN, f"Actor {actor_id}, Game {episode_num}: No legal actions in placement phase.")
                    break
                
                # Use Python's random for this, as it's already seeded.
                action = random.choice(legal_actions)
                
                # Optional: Log the placement move
                action_str = mali_ba_state.action_to_string(mali_ba_state.current_player(), action)
                log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}: Placing token with action '{action_str}'")

                mali_ba_state.apply_action(action)
            # --- END OF OPTIMIZATION ---

        log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}: Starting main play phase.")
        move_count = 0

        # Switch to PLAY mode
        while not state.is_terminal():
            observation = np.array(state.observation_tensor(), dtype=np.float32)
            # # DEBUG ======================================================================
            # log(LogLevel.INFO, f"Raw observation shape: {observation.shape}")
            # log(LogLevel.INFO, f"Raw observation range: min={np.min(observation)}, max={np.max(observation)}")
            # log(LogLevel.INFO, f"Raw observation sample: {observation[:20]}")  # First 20 values
            # # END DEBUG ======================================================================
            player = state.current_player()
            
            root = bot.mcts_search(state)
            
            temperature = 1.0 if move_count < 100 else 0.5

            # ** Create the action map for this state **
            legal_actions = state.legal_actions()
            if not legal_actions:
                log(LogLevel.WARN, f"Actor {actor_id} found no legal actions for a non-terminal state. Breaking game loop.")
                break
            #===============================================================
            # DEBUG to see what choices the bot has
            #===============================================================
            if move_count < 20 and player >= 0:  # Only debug first moves
                #legal_actions = state.legal_actions()
                log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}, Move {move_count}: DEBUG")
                log(LogLevel.INFO, f"  Legal actions ({len(legal_actions)}): {legal_actions[:10]}...")  # Show first 10
                
                # Get neural network predictions
                observation = np.array(state.observation_tensor(), dtype=np.float32)
                obs_reshaped = np.reshape(observation, game.observation_tensor_shape())
                obs_batch = np.expand_dims(obs_reshaped, 0)
                
                try:
                    # Get policy and value from their separate, dedicated models
                    policy_pred = policy_model(obs_batch, training=False)
                    value_pred = value_model(obs_batch, training=False)
                    
                    log(LogLevel.INFO, f"  Neural network value prediction: {value_pred[0].numpy()}")
                    
                    # START DEBUG =================================================================================
                    # Show policy values for legal actions
                    policy_flat = policy_pred[0].numpy()

                    # Categorize actions by type and find the best of each
                    action_categories = {
                        'income': [],
                        'mancala': [],
                        'upgrade': [],
                        'pass': [],
                        'other': []
                    }

                    # Categorize all legal actions
                    for action in legal_actions:
                        if 0 <= action < len(policy_flat):
                            action_str = state.action_to_string(player, action).lower()
                            policy_val = policy_flat[action]
                            
                            if "income" in action_str:
                                action_categories['income'].append((action, policy_val, action_str))
                            elif "mancala" in action_str:
                                action_categories['mancala'].append((action, policy_val, action_str))
                            elif "upgrade" in action_str:
                                action_categories['upgrade'].append((action, policy_val, action_str))
                            elif "pass" in action_str:
                                action_categories['pass'].append((action, policy_val, action_str))
                            else:
                                action_categories['other'].append((action, policy_val, action_str))

                    # Find and log the top action for each category
                    top_by_category = {}
                    for category, actions in action_categories.items():
                        if actions:
                            # Sort by policy value (descending) and take the top one
                            top_action = max(actions, key=lambda x: x[1])
                            top_by_category[category] = f"{category.upper()}: {top_action[2]}: {top_action[1]:.4f}"

                    if top_by_category:
                        log(LogLevel.INFO, f"   Top policy by type: {list(top_by_category.values())}")

                    # Also show overall top 5 actions across all types
                    all_legal_with_policy = [(action, policy_flat[action], state.action_to_string(player, action)) 
                                            for action in legal_actions if 0 <= action < len(policy_flat)]
                    top_5_overall = sorted(all_legal_with_policy, key=lambda x: x[1], reverse=True)[:5]
                    top_5_strings = [f"{action_str}: {policy_val:.4f}" for _, policy_val, action_str in top_5_overall]
                    log(LogLevel.INFO, f"   Top 5 overall policies: {top_5_strings}")

                    # Special check for income actions (keep your existing logic)
                    income_actions = [a for a in legal_actions if "income" in state.action_to_string(player, a).lower()]
                    if income_actions:
                        income_action = income_actions[0]
                        income_policy = policy_flat[income_action] if income_action < len(policy_flat) else 0
                        log(LogLevel.INFO, f"   Income action {income_action} policy value: {income_policy:.4f}")
            #===============================================================
            # END DEBUG to see what choices the bot has
            #===============================================================
                        
                except Exception as e:
                    log(LogLevel.ERROR, f"  Neural network prediction failed: {e}")
            
            action_map = {action: i for i, action in enumerate(legal_actions)}

            visit_counts = np.zeros(len(legal_actions))
            for child in root.children:
                if child.action in action_map:
                    visit_counts[action_map[child.action]] = child.explore_count

            if np.sum(visit_counts) > 0:
                powered_policy = np.power(visit_counts, 1.0 / temperature)
                mcts_policy_compact = powered_policy / np.sum(powered_policy)
                # Choose an action from the *compact* index space
                chosen_compact_index = np.random.choice(len(legal_actions), p=mcts_policy_compact)
                action = legal_actions[chosen_compact_index]
            else:
                action = random.choice(legal_actions)

            #===============================================================
            # DEBUG to see what choices the bot has
            #===============================================================
            if move_count < 200 and player >= 0:
                chosen_action_str = state.action_to_string(player, action)
                log(LogLevel.INFO, f"  MCTS chose: {chosen_action_str} (action {action})")
                
                # Show MCTS visit counts for top actions
                if hasattr(root, 'children') and len(root.children) > 0:
                    visit_counts = [(child.action, child.explore_count) for child in root.children]
                    visit_counts.sort(key=lambda x: x[1], reverse=True)
                    top_visits = []
                    for act, count in visit_counts[:3]:
                        act_str = state.action_to_string(player, act)
                        top_visits.append(f"{act_str}: {count} visits")
                    log(LogLevel.INFO, f"  MCTS top visits: {top_visits}")
            #===============================================================
            # DEBUG to see what choices the bot has
            #===============================================================

            # For the replay buffer, we need the policy over the FULL action space
            mcts_policy_full = np.zeros(game.num_distinct_actions())
            for child in root.children:
                 if 0 <= child.action < game.num_distinct_actions():
                    mcts_policy_full[child.action] = child.explore_count
            if np.sum(mcts_policy_full) > 0:
                mcts_policy_full /= np.sum(mcts_policy_full)
            else: # Fallback for states with no visits (should be rare)
                prob = 1.0 / len(legal_actions)
                for act in legal_actions:
                    mcts_policy_full[act] = prob
            
            if action != pyspiel.INVALID_ACTION:
                action_str = state.action_to_string(player, action)
                log(LogLevel.INFO, f"Actor {actor_id}, Game {episode_num}, Move {move_count}: Player {player} plays '{action_str}'")

            if action == pyspiel.INVALID_ACTION: break

            reward_vector = state.rewards() # Get intermediate rewards BEFORE applying the next action
            state.apply_action(action)
            episode_trajectory.append((observation, player, mcts_policy_full, reward_vector))
            move_count += 1

        returns = state.returns()
        if state.is_terminal():
            mali_ba_state_terminal = pyspiel.mali_ba.downcast_state(state)
            reason = mali_ba_state_terminal.get_game_end_reason()
            trigger_player = mali_ba_state_terminal.get_game_end_triggering_player()

            winner_str = "Tie/Draw" # Default
            if 1.0 in returns:
                winner_player_id = returns.index(1.0)
                winner_str = f"Player {winner_player_id}"
            elif reason == "Max game length reached":
                 winner_str = "Tie/Draw (Max Length)"

            # --- BEGIN intermediate rewards calculation ---
            try:
                # Calculate the sum of intermediate rewards for each player from the trajectory
                num_players = game.num_players()
                total_intermediate_rewards = [0.0] * num_players
                non_zero_reward_steps = 0

                # The trajectory is a list of (observation, player, policy, reward_vector)
                for _, _, _, reward_vector in episode_trajectory:
                    if any(r != 0 for r in reward_vector):
                        non_zero_reward_steps += 1
                    for i in range(num_players):
                        total_intermediate_rewards[i] += reward_vector[i]
                
                # Format the rewards for readable logging
                formatted_rewards = [f"{r:.4f}" for r in total_intermediate_rewards]

            except Exception as e:
                log(LogLevel.WARN, f"Actor {actor_id}, Game {episode_num}: Failed to generate reward summary. Error: {e}")
            # --- END intermediate rewards calculation ---
            
            log(LogLevel.INFO, 
                f"Actor {actor_id}, Game {episode_num}: FINISHED in {move_count} moves. "
                f"Winner: {winner_str}. Reason: '{reason}'. "
                f"Triggered by: Player {trigger_player}. Final Returns: {returns}"
                f"Intermediate Rewards: [{', '.join(formatted_rewards)}], "
                f"Rewarded Steps: {non_zero_reward_steps}"
                )
            
        result_queue.put((episode_trajectory, returns))

    log(LogLevel.INFO, f"Actor {actor_id} completed its quota of {games_per_actor} games and is terminating.")

def heuristic_actor_process(actor_id, game_params, args, job_queue, result_queue, games_per_actor):
    """
    An actor process that plays games using the built-in C++ heuristic.
    It generates trajectories with one-hot policies to bootstrap the initial model.
    """
    # --- Delayed Imports ---
    import numpy as np
    import random
    import pyspiel
    from pyspiel import mali_ba
    from pyspiel.mali_ba import log, LogLevel
    import tensorflow as tf

    # Ensure this actor runs on CPU to leave GPU for the trainer
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')
    log(LogLevel.INFO, f"Heuristic Actor {actor_id} started (CPU-only).")

    # --- Create ONE game object for the lifetime of the actor ---
    log(LogLevel.INFO, f"Heuristic Actor {actor_id}: Initializing its game instance.")
    game = pyspiel.load_game(args.game_name, game_params)
    # Get the max game length
    max_game_length = game.get_max_game_length()
    log(LogLevel.INFO, f"Using max game length: {max_game_length}")
    
    for _ in range(args.games_per_actor):
        job = job_queue.get()
        if job is None:
            break

        episode_num, _, game_rng_seed = job

        # Seed Python's random for deterministic choices if needed
        random.seed(game_rng_seed)
        np.random.seed(game_rng_seed)
        
        # --- FIX: Use the existing game object to create a new state ---
        # We do NOT need to reload the game. We create a fresh state from the template.
        state = game.new_initial_state()
        
        # The C++ state's internal RNG will be seeded by the game object's RNG state,
        # which is usually set once at creation. For true per-game randomness from C++,
        # you might need a game.set_rng_state() method if the initial seed isn't sufficient.
        # However, for the heuristic actor, this is less critical.

        # --- Play the full game to generate a trajectory ---
        
        # This list will store tuples of (observation, player, one_hot_policy)
        # The final game outcome will be added later.
        temp_trajectory = []

        # Handle initial chance and token placement phases randomly
        if state.is_chance_node():
            state.apply_action(state.legal_actions()[0])
        
        mali_ba_state = pyspiel.mali_ba.downcast_state(state)
        while mali_ba_state.current_phase() == pyspiel.mali_ba.Phase.PLACE_TOKEN:
            if mali_ba_state.is_terminal(): break
            legal_actions = mali_ba_state.legal_actions()
            if not legal_actions: break
            action = random.choice(legal_actions)
            mali_ba_state.apply_action(action)
        
        log(LogLevel.INFO, f"Heuristic Actor {actor_id}, Game {episode_num}: Starting main play phase.")
        move_count = 0

        # Main heuristic-driven play loop
        while not state.is_terminal():
            player = state.current_player()
            observation = np.array(state.observation_tensor(), dtype=np.float32)
            # # DEBUG ======================================================================
            # log(LogLevel.INFO, f"Raw observation shape: {observation.shape}")
            # log(LogLevel.INFO, f"Raw observation range: min={np.min(observation)}, max={np.max(observation)}")
            # log(LogLevel.INFO, f"Raw observation sample: {observation[:20]}")  # First 20 values
            # # END DEBUG ======================================================================

            # Get the action from the C++ heuristic
            action = mali_ba_state.select_heuristic_random_action()
            if action == pyspiel.INVALID_ACTION:
                log(LogLevel.WARN, f"Heuristic Actor {actor_id} received invalid action. Breaking.")
                break
                
            # Get the weights for ALL legal actions
            action_weights_map = mali_ba_state.get_heuristic_action_weights()
            
            # Create the policy target vector
            policy_target = np.zeros(game.num_distinct_actions(), dtype=np.float32)
            
            total_weight = sum(action_weights_map.values())
            
            if total_weight > 0:
                for act, weight in action_weights_map.items():
                    policy_target[act] = weight / total_weight # Normalize to a probability distribution
            else:
                # Fallback for states where all weights are zero (should be rare)
                legal_actions = state.legal_actions()
                if legal_actions:
                    prob = 1.0 / len(legal_actions)
                    for act in legal_actions:
                        policy_target[act] = prob

            # Store the observation and policy for the current state (S_t)
            # Then, apply the action to transition to the next state (S_{t+1})
            state.apply_action(action)
            
            # Now get the immediate reward (R_{t+1}) received for that transition
            reward_vector = state.rewards()

            # Store the complete 4-element tuple for this step
            temp_trajectory.append((observation, player, policy_target, reward_vector))

            move_count += 1
            # END Main heuristic-driven play loop - while not state.is_terminal():

        # Game is over, get the final returns
        returns = state.returns()

        # # DEBUG Check action diversity in this game
        # action_counts = {}
        # action_names = {}
        # for obs, player, policy_target in temp_trajectory:
        #     action = np.argmax(policy_target)
        #     action_counts[action] = action_counts.get(action, 0) + 1
        #     # Get the action name for readability
        #     if action not in action_names:
        #         action_names[action] = state.action_to_string(player, action) if hasattr(state, 'action_to_string') else f"action_{action}"
        
        # # DEBUG Log the action distribution with names
        # action_summary = {action_names.get(action, f"action_{action}"): count 
        #                  for action, count in action_counts.items()}
        # log(LogLevel.DEBUG, f"Heuristic Actor {actor_id}, Game {episode_num}: Action distribution: {action_summary}")


        log(LogLevel.INFO, 
            f"Heuristic Actor {actor_id}, Game {episode_num}: FINISHED in {move_count} moves. "
            f"Final Returns: {returns}")

        # The result queue expects a trajectory and the returns.
        # This format is identical to what the MCTS actor produces.
        result_queue.put((temp_trajectory, returns))

    log(LogLevel.INFO, f"Heuristic Actor {actor_id} completed its quota and is terminating.")


def spawn_actor(actor_id, initial_game_params, args, job_queue, result_queue, actor_pool, actor_function):
    """
    Creates, starts, and tracks a new actor process using the specified actor function.
    """
    p = mp.Process(target=actor_function, args=(
        actor_id, initial_game_params, args, job_queue, result_queue, args.games_per_actor))
    p.start()
    actor_pool[p] = actor_id # Associates the process object with its ID
    print(f"Main: Spawned new actor (type: {actor_function.__name__}) with ID {actor_id}.")

def log_game_outcome_debug(total_games_processed, returns, game_length, max_game_length):
    try:
        from pyspiel.mali_ba import log, LogLevel
    except ImportError:
        class LogLevel: INFO, WARN = 1, 2
        def log(level, msg): print(msg)

    """Debug logging for game outcomes."""
    if total_games_processed % 50 == 0 or total_games_processed <= 10:
        length_penalty_ratio = game_length / max_game_length
        
        if any(r > 0 for r in returns):  # Someone won
            winner = returns.index(max(returns))
            #discounted_win = 1.0 - (length_penalty_ratio ** 1.5)
            log(LogLevel.INFO, f"  DECISIVE GAME: Player {winner} won in {game_length} moves")
            log(LogLevel.INFO, f"    Return: {returns[winner]:.3f} ")
        elif any(r == 0 for r in returns):  # Draw
            log(LogLevel.INFO, f"  DRAW GAME: {game_length} moves, all players get penalty")
        else:
            log(LogLevel.INFO, f"  UNUSUAL GAME: Returns {returns}")

# --- Main Orchestrator (Updated for robustness) ---
def main(args):
    import pyspiel
    try:
        from pyspiel.mali_ba import log, LogLevel
    except ImportError:
        class LogLevel: INFO, WARN = 1, 2
        def log(level, msg): print(msg)
    import queue
    from training_utils import get_training_parameters_from_game

    log(LogLevel.INFO, "--- Starting Mali-Ba MULTIPROCESS AI Training Script ---")

    # --- 1. Setup ---
    initial_game_params = {"config_file": args.config_file or ""}
    if args.players: initial_game_params["NumPlayers"] = args.players
    if args.grid_radius: initial_game_params["grid_radius"] = args.grid_radius
    initial_game_params["player_types"] = "ai,ai,ai"
    # Get training parameters from the game
    # DEBUG: Not sure if this is needed or if it's all taken care of in C++
    temp_game = pyspiel.load_game(args.game_name, initial_game_params)
    training_params = get_training_parameters_from_game(temp_game)
    del temp_game
    DRAW_PENALTY = training_params['draw_penalty']
    MAX_MOVES_PENALTY = training_params['max_moves_penalty']
    QUICK_WIN_BONUS = training_params['quick_win_bonus'] 
    QUICK_WIN_THRESHOLD = training_params['quick_win_threshold']

    # log(LogLevel.INFO, f"Using training parameters from INI:")
    # log(LogLevel.INFO, f"  Draw penalty: {DRAW_PENALTY}")
    # log(LogLevel.INFO, f"  Draw penalty: {MAX_MOVES_PENALTY}")
    # log(LogLevel.INFO, f"  Quick win bonus: {QUICK_WIN_BONUS}")
    # log(LogLevel.INFO, f"  Quick win threshold: {QUICK_WIN_THRESHOLD}")

    # The job queue should have enough for all actors to have a backup job.
    max_jobs = args.num_actors * 3
    job_queue = mp.Queue(maxsize=max_jobs)

    # The result queue should be small; we want to process results quickly.
    max_results = args.num_actors 
    result_queue = mp.Queue(maxsize=max_results)

    # The replay buffer queue is the biggest memory consumer.
    # Limit it to a few batches worth of experiences.
    # batch_size * num_moves_per_game * num_games
    # A fixed large number is simpler. Let's allow 20 batches.
    max_replay_items = args.batch_size * 50
    replay_buffer_queue = mp.Queue(maxsize=max_replay_items)

    weights_queue = mp.Queue()
    stats_queue = mp.Queue()

    trainer = mp.Process(target=trainer_process, args=(
        args, initial_game_params, replay_buffer_queue, weights_queue, stats_queue))
    trainer.start()
    log(LogLevel.INFO, "Launched trainer process.")

    # --- 2. Learner State Init (Same) ---
    actor_pool = {}
    next_actor_id = 0
    total_games_processed = 0
    jobs_dispatched = 0
    start_time = time.time()
    last_weights_update_time = time.time()
    
    if args.random_seed is None:
        master_seed = int(time.time() * 1000) % (2**32 - 1)
    else:
        master_seed = args.random_seed
            
    import numpy as np
    seed_generator = np.random.RandomState(master_seed)
    log(LogLevel.INFO, f"Master seed generator initialized with seed: {master_seed}")

    log(LogLevel.INFO, "Learner waiting for initial weights from trainer...")
    current_weights = weights_queue.get()
    log(LogLevel.INFO, "Learner received initial weights.")

    # --- 3. UNIFIED Main Learner Loop ---
    bootstrap_transition_logged = False
    while total_games_processed < args.num_episodes:


        if total_games_processed % 10 == 0:
            log(LogLevel.INFO, f"Queue sizes - Jobs: {job_queue.qsize()}, "
                            f"Results: {result_queue.qsize()}, "
                            f"Replay: {replay_buffer_queue.qsize()}")

        # --- A. Actor & Job Management ---
        
        # Determine which actor function to use for any NEW spawns
        if total_games_processed < args.bootstrap_episodes:
            current_actor_function = heuristic_actor_process
        else:
            if not bootstrap_transition_logged and args.bootstrap_episodes > 0:
                log(LogLevel.INFO, f"--- Bootstrap phase complete. New actors will be MCTS type. ---")
                bootstrap_transition_logged = True
            current_actor_function = actor_process
            
        # --- Find, remove, and immediately replace dead actors ---
        dead_actors = [p for p in actor_pool if not p.is_alive()]
        if dead_actors:
            log(LogLevel.INFO, f"Found {len(dead_actors)} dead actor(s). Respawning...")
            
        for p in dead_actors:
            actor_crashed_id = actor_pool[p]
            log(LogLevel.WARN, f"Actor ID {actor_crashed_id} terminated. Spawning replacement of type {current_actor_function.__name__}.")
            
            # Remove the dead process from the pool
            del actor_pool[p]
            
            # Immediately spawn its replacement
            spawn_actor(next_actor_id, initial_game_params, args, job_queue, result_queue, actor_pool, current_actor_function)
            next_actor_id += 1
            
        # This separate loop is now only necessary for the initial startup,
        # but it's harmless to keep it for ensuring the pool is always full.
        while len(actor_pool) < args.num_actors:
            log(LogLevel.INFO, f"Actor pool below target ({len(actor_pool)}/{args.num_actors}). Spawning new actor.")
            spawn_actor(next_actor_id, initial_game_params, args, job_queue, result_queue, actor_pool, current_actor_function)
            next_actor_id += 1
            
        # --- Maintain a healthy job queue size ---
        # This logic remains the same and is correct.
        target_job_queue_size = args.num_actors * 2
        while job_queue.qsize() < target_job_queue_size and jobs_dispatched < args.num_episodes:
            unique_seed = seed_generator.randint(0, 2**31 - 1)
            job_queue.put((jobs_dispatched, current_weights, unique_seed)) # This passes the tuple
            jobs_dispatched += 1

        # --- B. Try to process a result ---
        try:
            trajectory, returns = result_queue.get(timeout=1.0)
            
            # Process the game result
            total_games_processed += 1
            phase_label = "Bootstrap" if total_games_processed <= args.bootstrap_episodes else "MCTS"
            log(LogLevel.INFO, f"LEARNER ({phase_label}) RECEIVED GAME #{total_games_processed}/{args.num_episodes}. "
                            f"Length: {len(trajectory)} moves. Returns: {returns}")

            GAMMA = 0.99  # Discount factor. Rewards further in the future are worth slightly less.

            # The 'returns' variable from the C++ state is the final terminal outcome.
            final_terminal_returns = returns 
            
            # This will store (obs, player, policy, calculated_value_vector) for each step.
            trajectory_with_values = []
            
            # We iterate backwards from the end of the game.
            # The value of the last state is just the final game outcome.
            next_state_discounted_returns = final_terminal_returns

            for i in range(len(trajectory) - 1, -1, -1):
                observation, player, policy_target, immediate_reward_vector = trajectory[i]
                
                # The value of a state is: (the immediate reward you get) + gamma * (the value of the state you land in).
                # Since we are iterating backwards, 'next_state_discounted_returns' holds the value of the next state.
                current_state_value_vector = [
                    r + GAMMA * next_r for r, next_r in zip(immediate_reward_vector, next_state_discounted_returns)
                ]
                
                # Add this step's data with the correctly calculated value to our list.
                trajectory_with_values.append((observation, player, policy_target, current_state_value_vector))
                
                # The value we just calculated becomes the "next state value" for the previous step in the next iteration.
                next_state_discounted_returns = current_state_value_vector

            # The list is currently in reverse order, so let's put it back chronologically.
            trajectory_with_values.reverse()

            # Now, add the correctly calculated experiences to the replay buffer.
            for observation, player, policy_target, value_target_vector in trajectory_with_values:
                # The value target for the network is the full vector of expected returns for all players.
                # The player_value is just one element of that vector, used for logging/debugging if needed.
                player_value = value_target_vector[player]
                value_data = (player, player_value, value_target_vector)
                
                if not replay_buffer_queue.full():
                    replay_buffer_queue.put((observation, policy_target, value_data))
                else:
                    log(LogLevel.WARN, "Replay buffer queue is full during data insertion.")
                    break

            # Add detailed outcome logging
            # log_game_outcome_debug(total_games_processed, discounted_returns, game_length, max_game_length)

            # Debug logging for first few games
            if total_games_processed <= 5:
                log(LogLevel.INFO, f"Game {total_games_processed} debug:")
                log(LogLevel.INFO, f"  Original returns: {returns}")
                log(LogLevel.INFO, f"  returns: {returns}")
                # log(LogLevel.INFO, f"  Length penalty ratio: {length_ratio:.3f}")

            # # Create training experiences with proper value targets
            # experiences_added = 0
            # for observation, player, policy_target in trajectory:
            #     player_value = returns[player]
            #     value_data = (player, player_value, returns)
            #     replay_buffer_queue.put((observation, policy_target, value_data))
            #     experiences_added += 1

            # # Debug log
            # if total_games_processed <= 10:  # Only for first 10 games
            #     log(LogLevel.INFO, f"MAIN: Added {experiences_added} experiences to queue. Queue size now: {replay_buffer_queue.qsize()}")


        except queue.Empty:  # Only catch timeout exceptions
            #log(LogLevel.INFO, "...")
            pass
        except Exception as e:  # Log any other exceptions
            log(LogLevel.ERROR, f"Error processing game result: {e}")
            import traceback
            log(LogLevel.ERROR, f"Traceback: {traceback.format_exc()}")


        # --- C. Weight & Stats Management (Periodic) ---
        if time.time() - last_weights_update_time > WEIGHTS_UPDATE_INTERVAL_SECONDS:
            latest_weights = None
            while not weights_queue.empty():
                try: latest_weights = weights_queue.get_nowait()
                except: break
            if latest_weights is not None:
                current_weights = latest_weights
                log(LogLevel.INFO, f"Learner updated to latest weights at game #{total_games_processed}. Distributing to {len(actor_pool)} actors.")

            
            while not stats_queue.empty():
                try:
                    stats = stats_queue.get_nowait()
                    if "loss" in stats:
                        log(LogLevel.INFO, f"Trainer reported loss: {stats['loss']:.4f} at game #{total_games_processed}")
                except: break
            
            last_weights_update_time = time.time()

    # --- 4. Final Shutdown (Same) ---
    log(LogLevel.INFO, "All episodes processed. Sending shutdown signals...")
    # ... (shutdown code is correct) ...
    replay_buffer_queue.put(None)
    
    while not job_queue.empty():
        try: job_queue.get_nowait()
        except: break
            
    for _ in range(len(actor_pool)):
        job_queue.put(None)

    log(LogLevel.INFO, "Waiting for trainer to terminate...")
    trainer.join(timeout=180)
    if trainer.is_alive():
        log(LogLevel.WARN, "Trainer did not terminate gracefully. Forcing.")
        trainer.terminate()

    log(LogLevel.INFO, "Waiting for actors to terminate...")
    for p in actor_pool:
        p.join(timeout=60)
        if p.is_alive(): p.terminate()

    log(LogLevel.INFO, "All processes terminated.")
    end_time = time.time()
    log(LogLevel.INFO, f"Total time: {end_time - start_time:.2f} seconds")
    log(LogLevel.INFO, f"Training complete. Final model saved to {args.save_model_path}")
    

# --- Main Execution Guard ---
if __name__ == "__main__":

    try:
        # Use the mp alias we defined at the top
        mp.set_start_method('spawn', force=True) 
    except RuntimeError:
        print("Note: multiprocessing start method already set.")
        pass
    
    parser = argparse.ArgumentParser(description='Mali-Ba Multiprocess AI Training Script')
    # ... (all args are the same) ...
    parser.add_argument('--num_actors', type=int, default=max(1, os.cpu_count() - 2), help="Number of actor processes to run in parallel.")
    parser.add_argument('--game_name', type=str, default="mali_ba")
    parser.add_argument('--config_file', type=str, default="/home/robp/Projects/open_spiel/open_spiel/games/mali_ba/mali_ba.ini")
    parser.add_argument('--players', type=int, default=None, choices=range(2, 6))
    parser.add_argument('--grid_radius', type=int, default=None)
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--save_model_path', type=str, default="mali_ba_agent_v2.weights.h5")
    parser.add_argument('--replay_buffer_size', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--save_every', type=int, default=30, help="Save a model checkpoint every N minutes in the trainer process.")
    parser.add_argument('--update_workers_every', type=int, default=50, help="This argument is now effectively unused as weights are updated via a queue.")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Initial learning rate for the schedule.")
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--uct_c', type=float, default=2.0)
    parser.add_argument('--max_simulations', type=int, default=50)
    parser.add_argument('--games_per_actor', type=int, default=10, 
                    help="Number of games each actor process plays before self-terminating to free memory.")
    parser.add_argument('--bootstrap_episodes', type=int, default=0,
                    help="Number of initial episodes to generate using the C++ heuristic for bootstrapping.")

    
    parsed_args = parser.parse_args()
    main(parsed_args)