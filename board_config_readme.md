# Mali-Ba Board Configuration

This is a simplified implementation for configuring the Mali-Ba game board using an INI file. The code has been streamlined to focus solely on board configuration.

## Files

- `hex_grid.h` - Contains basic hex grid functionality
- `board_config.h` / `board_config.cc` - Board configuration loading from INI files
- `board_config_test.cc` - Test program to demonstrate the configuration loading
- `sample_board.ini` - Example of a regular board configuration
- `irregular_board.ini` - Example of an irregular board configuration
- `Makefile` - Build instructions

## INI File Format

The board configuration is specified in an INI file with the following parameters:

### Required Parameters

- `regular_board` - Set to 'Y' for a regular hexagonal board or 'N' for a custom board
- `board_radius` - Integer specifying the radius from the center (0,0,0)

### Conditional Parameters

- `board_valid_hexes` - Required only if `regular_board` is 'N'. Specifies the valid hexes in the format: `x,y,z;x,y,z;...`

## Example INI Files

### Regular Board

```ini
# Mali-Ba board configuration
regular_board = Y
board_radius = 4
```

### Irregular Board

```ini
# Mali-Ba irregular board configuration
regular_board = N
board_radius = 5
board_valid_hexes = 0,0,0;1,-1,0;1,0,-1;0,1,-1;-1,1,0;-1,0,1;0,-1,1
```

## Default Values

If the INI file is missing or contains invalid values, the program will use these defaults:
- `regular_board` = Y (regular hexagonal board)
- `board_radius` = 3

## Building and Running

```bash
# Build the test program
make

# Run with the regular board configuration
make run

# Run with the irregular board configuration
make run_irregular

# Or specify a custom INI file
./board_config_test my_custom_config.ini
```

## Integration with Game

To integrate this board configuration with the Mali-Ba game:

1. Load the board configuration using `BoardConfig::LoadFromFile(filename)`
2. Use the resulting `valid_hexes` set to initialize your game board

## Hex Coordinate System

The program uses cube coordinates (x,y,z) for hexes where x + y + z = 0. This provides a consistent and easy-to-use coordinate system for hexagonal grids.
