#!/bin/bash
# This just gets run one time to set up the symlinks of my mali_ba files into the cloned open_spiel folder
OPENSPIEL=/media/robp/UD/Projects/open_spiel
MALI_BA=/media/robp/UD/Projects/mali_ba

ln -s $MALI_BA/open_spiel/games/mali_ba $OPENSPIEL/open_spiel/games/mali_ba
ln -s $MALI_BA/open_spiel/python/games/mali_ba_py $OPENSPIEL/open_spiel/python/games/mali_ba_py
ln -s $MALI_BA/open_spiel/python/pybind11/games_mali_ba.cc $OPENSPIEL/open_spiel/python/pybind11/games_mali_ba.cc
ln -s $MALI_BA/open_spiel/python/pybind11/games_mali_ba.h $OPENSPIEL/open_spiel/python/pybind11/games_mali_ba.h
# ... one line per symlink needed
