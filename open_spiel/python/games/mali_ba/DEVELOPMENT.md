# Mali-Ba Development Guide

## Development Mode Architecture

The Mali-Ba project now features a development mode that allows you to work on the game UI without requiring the C++ OpenSpiel backend. This approach makes development faster and more flexible.

### Key Components

1. **Development Configuration (`dev_config.py`)**
   - Central location for all development settings
   - Toggle `BYPASS_CPP = True` to disable C++ integration
   - Configure default settings like number of players and grid radius

2. **Game Interface (`utils/cpp_interface.py`)**
   - Encapsulates all interactions with C++ code
   - Provides mock implementations when in development mode
   - Handles state serialization and action application

3. **Main Application Entry Point (`main.py`)**
   - Uses command-line flags to override development settings if needed
   - Creates the appropriate game interface based on settings
   - Passes the interface to the visualizer

4. **Visualizer (`ui/visualizer.py`)**
   - Uses the interface to interact with the game logic
   - Displays the board state and handles user interaction
   - Doesn't need to know if it's using real or mocked C++ code

## Using Development Mode

### Toggle C++ Integration

Set `BYPASS_CPP` in `dev_config.py`:

```python
# Set to True to bypass all C++ code integration
BYPASS_CPP = True
```

Or use command line flags:

```bash
# Force bypass C++ (overrides dev_config.py)
python main.py --bypass_cpp

# Force use C++ (overrides dev_config.py)
python main.py --use_cpp
```

### Development Workflow

1. **UI/Visualization Development**:
   - Set `BYPASS_CPP = True` and use sample state data
   - Work on visualization, UI components, etc. without C++ dependencies

2. **Game Logic Integration**:
   - Set `BYPASS_CPP = False` to test with actual C++ game logic
   - Verify that UI correctly interprets and displays game state

3. **Testing Different Scenarios**:
   - Use `--state` parameter to load specific game states
   - Use `--sample` to load the built-in sample state
   - Control player count with `--players`

## Implementation Details

### Adding New C++ Functionality

When adding new C++ functionality:

1. Add the function to `GameInterface` in `cpp_interface.py`
2. Implement both the real C++ version and a mock version
3. Use conditionals based on `BYPASS_CPP` to determine which to use

Example:

```python
def get_legal_moves(self, player_id):
    """Get legal moves for a player."""
    if BYPASS_CPP:
        # Mock implementation for development
        return ["place 0,0,0", "place 1,-1,0"]
    
    # Real implementation using C++
    return self.spiel_state.get_legal_moves(player_id)
```

### Adding New UI Features

When adding new UI features:

1. Use the `game_interface` instead of directly accessing C++ objects
2. Test with both development mode and C++
3. Use the state cache for all display logic rather than directly querying game objects
4. Make sure any new interactions are processed through the `game_interface`

### Debugging

The development mode makes debugging much easier:

1. **State Transitions**: When in bypass mode, you can add prints to the mock state generation to understand what state properties change with different actions.

2. **UI/Game State Synchronization**: The state parsing system maintains a clean separation between the display and the game logic, making it easier to identify issues.

3. **Tracing Actions**: All action attempts are logged, making it easy to track what's happening.

## Sample Data

### Sample Cities

In development mode, sample cities are created using the `create_sample_cities()` function in `cpp_interface.py`. You can modify this function to create different test scenarios by changing:

- City locations and connections
- Cultural groups and resources
- Number and distribution of cities

### Sample State Generation

The `load_sample_state_str()` function provides a complete game state for testing. This includes:

- Player token positions
- Meeple distributions
- Trading posts and centers
- Resource distribution

You can modify this function to create different test scenarios or add specialized test states for different features.

## Implementation Tips

### 1. Extending Mock Behavior

When extending the mock behavior in development mode, focus on creating the minimal implementation needed for testing. You don't need to fully replicate the C++ logic, just enough to test your UI components.

For example, if you're testing the mancala move UI, you might implement:

```python
def _get_updated_mock_state_for_mancala(self, action_string):
    # Basic mock implementation that just shows the path and changes player
    path_parts = action_string.split()
    if len(path_parts) > 1:
        # Extract hexes from path
        path = path_parts[1].split(":")
        
        # Update player to next player
        parts = self._get_mock_state_string().split('/')
        player_info = parts[1].split(',')
        player_id = (int(player_info[0]) + 1) % self.num_players
        parts[1] = f"{player_id},0"
        
        # Maybe add a trading post at the end
        if "post" in action_string:
            # Add mock trading post at the end of the path
            # ...
        
        return '/'.join(parts)
```

### 2. Testing Phase Transitions

To test phase transitions (like moving from PLACE_TOKEN to PLAY), add phase transition logic to your mock state generator:

```python
def _get_updated_mock_state(self, action_string):
    # ... existing code ...
    
    # Check if all players have placed their tokens
    tokens_per_player = {} 
    for token_entry in parts[2].split(';'):
        if '=' in token_entry:
            _, player_id_str = token_entry.split('=')
            player_id = int(player_id_str)
            tokens_per_player[player_id] = tokens_per_player.get(player_id, 0) + 1
    
    # If all players have 3 tokens, switch to PLAY phase
    all_placed = True
    for pid in range(self.num_players):
        if tokens_per_player.get(pid, 0) < 3:
            all_placed = False
            break
    
    if all_placed:
        # Update game phase indicator if needed
        # ...
    
    return '/'.join(parts)
```

### 3. Visualizer Integration

When modifying the visualizer to work with the game interface:

1. Replace all direct C++ calls with calls to the game interface
2. Update the state cache based on parsing results 
3. Use the state cache for all rendering
4. Process user actions through the game interface

## Common Patterns

### Action Processing Pattern

When processing user actions:

```python
def submit_move(self):
    if not self.is_input_mode:
        return
        
    move_string = self._construct_move_string()
    if not move_string:
        self.control_panel.update_status("Invalid move construction")
        return
        
    success = self.attempt_apply_action(move_string)
    if success:
        # Action succeeded, state already updated
        pass
    else:
        # Handle failure (UI already shows error)
        pass
```

### State Update Pattern

When updating the game state:

```python
def update_game_state(self):
    # Get current state string from interface
    state_string = self.game_interface.get_current_state_string()
    
    # Parse state to update cache
    if self.parse_state_string(state_string):
        # Update UI based on new state
        self.update_layout()
        self.update_status_message()
    else:
        self.control_panel.update_status("Failed to parse state")
```

## Performance Considerations

### Development Mode vs. C++ Mode

Development mode prioritizes ease of testing over performance. Some differences to be aware of:

1. **Move Validation**: In development mode, move validation is simplified. The C++ version may reject moves that the development mode accepts.

2. **State Complexity**: The mock state is simpler than real game states. Test with C++ integration before release.

3. **Game Rules**: Development mode implements a simplified subset of game rules. Some advanced rules might only work in C++ mode.

## Future Enhancements

### Planned Development Features

1. **Configurable Mock Rules**: Allow more detailed configuration of mock game rules.

2. **Scenario Generation**: Add functions to generate specific test scenarios.

3. **Replay System**: Record and replay sequences of moves for testing.

4. **State Editor**: Add a simple UI for editing the game state during development.

5. **Automated Testing**: Create automated UI tests using the development mode.

## Conclusion

The development mode architecture provides a flexible way to work on the Mali-Ba UI without requiring the C++ backend. By encapsulating all C++ interactions in the game interface, we've created a clean separation that allows for easier development and testing.

Remember to test with both development mode and C++ integration to ensure full compatibility before releasing new features.