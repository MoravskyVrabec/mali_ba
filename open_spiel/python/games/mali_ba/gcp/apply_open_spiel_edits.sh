#!/bin/bash
# Applies the 3 stock-OpenSpiel edits documented in OPEN_SPIEL_EDITS.txt so the
# mali_ba game (symlinked in separately -- see main_vm_setup.sh / worker_vm_setup.sh)
# compiles as part of the pyspiel target.
#
# Safe to re-run: each edit checks whether it's already present and skips if so.
#
# Usage: ./apply_open_spiel_edits.sh <path-to-open_spiel-clone>
set -euo pipefail

OPENSPIEL="${1:?Usage: $0 <path-to-open_spiel-clone>}"

GAMES_CMAKE="$OPENSPIEL/open_spiel/games/CMakeLists.txt"
PYBIND_CMAKE="$OPENSPIEL/open_spiel/python/CMakeLists.txt"
PYSPIEL_CC="$OPENSPIEL/open_spiel/python/pybind11/pyspiel.cc"

for f in "$GAMES_CMAKE" "$PYBIND_CMAKE" "$PYSPIEL_CC"; do
  [ -f "$f" ] || { echo "ERROR: expected stock OpenSpiel file not found: $f" >&2; exit 1; }
done

# 1. open_spiel/games/CMakeLists.txt -- list mali_ba's sources directly in the
#    top-level GAME_SOURCES set(), and register its test executable. This is
#    NOT add_subdirectory(mali_ba) -- the mali_ba/CMakeLists.txt that ships
#    with the mali_ba repo (using an open_spiel_game(...) macro) is a dead
#    file that was never actually part of the working build; the flat-list
#    approach below is what the local dev build actually uses.
#    (Relies on the symlinked open_spiel/games/mali_ba/ dir already being in
#    place before this is built.)
if grep -q 'mali_ba/mali_ba_common\.cc' "$GAMES_CMAKE"; then
  echo "[1/3] games/CMakeLists.txt already lists mali_ba sources -- skipping"
else
  MALIBA_SOURCES_TMP="$(mktemp)"
  cat > "$MALIBA_SOURCES_TMP" <<'EOF'
# Begin Mali-ba specific
  mali_ba/mali_ba_common.cc
  mali_ba/mali_ba_game.cc
  mali_ba/mali_ba_state_core.cc
  mali_ba/mali_ba_state_display.cc
  mali_ba/mali_ba_state_moves.cc
  mali_ba/mali_ba_state_serialize.cc
  mali_ba/mali_ba_state_setup.cc
  mali_ba/mali_ba_state_trade.cc
  mali_ba/mali_ba_observer.cc
  mali_ba/hex_grid.h
  mali_ba/mali_ba_common.h
#  mali_ba/mali_ba_constants.h
  mali_ba/mali_ba_game.h
  mali_ba/mali_ba_state.h
  mali_ba/mali_ba_observer.h
# End Mali-ba specific
EOF
  sed -i "1r $MALIBA_SOURCES_TMP" "$GAMES_CMAKE"
  rm -f "$MALIBA_SOURCES_TMP"
  cat >> "$GAMES_CMAKE" <<'EOF'

# Begin Mali-ba specific
add_executable(mali_ba_test mali_ba/mali_ba_test.cc ${OPEN_SPIEL_OBJECTS}
               $<TARGET_OBJECTS:tests>)
add_test(mali_ba_test mali_ba_test)
# End Mali-ba specific
EOF
  echo "[1/3] Added mali_ba sources to GAME_SOURCES and registered mali_ba_test in games/CMakeLists.txt"
fi

# 2. open_spiel/python/CMakeLists.txt -- add games_mali_ba.cc/.h to the
#    PYTHON_BINDINGS list, anchored after the sibling games_trade_comm.cc entry.
if grep -q 'games_mali_ba\.cc' "$PYBIND_CMAKE"; then
  echo "[2/3] python/CMakeLists.txt already lists games_mali_ba.cc -- skipping"
else
  sed -i '/pybind11\/games_trade_comm\.cc/a\  pybind11/games_mali_ba.cc\n  pybind11/games_mali_ba.h' "$PYBIND_CMAKE"
  echo "[2/3] Added games_mali_ba.cc/.h to python/CMakeLists.txt PYTHON_BINDINGS"
fi

# 3. open_spiel/python/pybind11/pyspiel.cc -- include the mali_ba pybind header
#    and register its init function, both anchored next to the trade_comm entries.
if grep -q 'games_mali_ba\.h' "$PYSPIEL_CC"; then
  echo "[3/3] pyspiel.cc already includes games_mali_ba.h -- skipping"
else
  sed -i '/#include "open_spiel\/python\/pybind11\/games_trade_comm\.h"/a #include "open_spiel/python/pybind11/games_mali_ba.h"' "$PYSPIEL_CC"
  sed -i '/init_pyspiel_games_trade_comm(m);/a\  init_pyspiel_games_mali_ba(m);' "$PYSPIEL_CC"
  echo "[3/3] Added #include and init_pyspiel_games_mali_ba(m) to pyspiel.cc"
fi

echo "Done. Stock OpenSpiel edits applied (or already present) in $OPENSPIEL"
