cmake_minimum_required(VERSION 3.5)
project(MaliBa)

set(CMAKE_CXX_STANDARD 17)

# Define the root directory when you're in open_spiel/open_spiel/games/mali_ba/build
set(OPEN_SPIEL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")

# Add your source files
add_executable(mali_ba_test
  ../mali_ba_test.cc
  ../mali_ba.cc
  ../mali_ba_board.cc
)

# Include directories
target_include_directories(mali_ba_test PRIVATE
  ${OPEN_SPIEL_ROOT}
)
