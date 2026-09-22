#!/bin/bash
# Bootstraps the GCP GPU learner VM (main-vm) for Mali-Ba distributed training:
# NVIDIA driver, Miniconda + conda env, a vanilla OpenSpiel clone with the
# mali_ba edits applied, and a build of the pyspiel target.
#
# Run this AFTER the mali_ba repo has been pushed to this VM (see push_code.sh).
# Safe to re-run -- each step skips work that's already done.
#
# Usage: ./main_vm_setup.sh [path-to-mali_ba-repo] [path-for-open_spiel-clone]
set -euo pipefail

MALIBA_REPO="${1:-$HOME/mali_ba}"
OPENSPIEL="${2:-$HOME/open_spiel}"
CONDA_ENV=mali_ba
PYTHON_VERSION=3.11
GAME_DIR="$MALIBA_REPO/open_spiel/python/games/mali_ba"

[ -d "$MALIBA_REPO" ] || { echo "ERROR: mali_ba repo not found at $MALIBA_REPO -- run push_code.sh first" >&2; exit 1; }

echo "=== [1/9] Installing system build dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake clang git python3-dev curl tmux

echo "=== [2/9] Installing Tailscale ==="
# Required: workers reach this VM over Tailscale, not a public IP/port-forward
# (see DISTRIBUTED_TRAINING.md). Without this, no worker can connect at all.
if ! command -v tailscale &>/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sudo sh
else
  echo "tailscale already installed -- skipping"
fi
if ! sudo tailscale status &>/dev/null; then
  TS_AUTHKEY="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/attributes/tailscale-authkey)"
  if [ -n "$TS_AUTHKEY" ]; then
    sudo tailscale up --authkey="$TS_AUTHKEY"
  else
    echo "No tailscale-authkey project metadata set -- run 'sudo tailscale up' manually, or gcp/set_tailscale_authkey.sh to automate this."
  fi
fi

echo "=== [3/9] Installing NVIDIA driver ==="
if ! command -v nvidia-smi &>/dev/null; then
  curl -fsSL -O https://raw.githubusercontent.com/GoogleCloudPlatform/compute-gpu-installation/main/linux/install_gpu_driver.py
  sudo python3 install_gpu_driver.py
else
  echo "nvidia-smi already present -- skipping driver install"
fi
nvidia-smi

echo "=== [4/9] Installing Miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "=== [5/9] Creating conda env '$CONDA_ENV' ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
if ! conda env list | grep -q "^$CONDA_ENV "; then
  conda create -y -n "$CONDA_ENV" python=$PYTHON_VERSION
fi
conda activate "$CONDA_ENV"
pip install --upgrade pip -q
pip install -r "$GAME_DIR/requirements-gpu.txt"

echo "=== [6/9] Cloning vanilla OpenSpiel + applying mali_ba edits ==="
if [ ! -d "$OPENSPIEL" ]; then
  git clone --depth 1 https://github.com/google-deepmind/open_spiel.git "$OPENSPIEL"
fi
bash "$GAME_DIR/gcp/apply_open_spiel_edits.sh" "$OPENSPIEL"

echo "=== [7/9] Bootstrapping OpenSpiel third-party dependencies (install.sh) ==="
# Matches the local dev build's config: only the non-optional deps (pybind11,
# abseil-cpp, json, pybind11_json, pybind11_abseil) -- everything optional
# (hanabi, ACPC, xinxin, roshambo, libnop, libtorch, ortools) is OFF, same as
# CMakeLists.txt's own defaults, so skip cloning them here too.
(
  cd "$OPENSPIEL"
  OPEN_SPIEL_BUILD_WITH_HANABI=OFF \
  OPEN_SPIEL_BUILD_WITH_ACPC=OFF \
  OPEN_SPIEL_BUILD_WITH_XINXIN=OFF \
  OPEN_SPIEL_BUILD_WITH_ROSHAMBO=OFF \
  OPEN_SPIEL_BUILD_WITH_LIBNOP=OFF \
  OPEN_SPIEL_BUILD_WITH_LIBTORCH=OFF \
  OPEN_SPIEL_BUILD_WITH_ORTOOLS=OFF \
  ./install.sh "$(which python)"
)

echo "=== [8/9] Symlinking mali_ba code into the OpenSpiel clone ==="
ln -sf  "$MALIBA_REPO/open_spiel/python/pybind11/games_mali_ba.cc" "$OPENSPIEL/open_spiel/python/pybind11/games_mali_ba.cc"
ln -sf  "$MALIBA_REPO/open_spiel/python/pybind11/games_mali_ba.h"  "$OPENSPIEL/open_spiel/python/pybind11/games_mali_ba.h"
ln -sfn "$MALIBA_REPO/open_spiel/games/mali_ba"                    "$OPENSPIEL/open_spiel/games/mali_ba"
ln -sfn "$GAME_DIR"                                                "$OPENSPIEL/open_spiel/python/games/mali_ba"

echo "=== [9/9] Building pyspiel (first build takes a while) ==="
mkdir -p "$OPENSPIEL/build"
cd "$OPENSPIEL/build"
cmake "$OPENSPIEL/open_spiel" -DPython3_EXECUTABLE="$(which python)" -DCMAKE_CXX_FLAGS="-O2" -DCMAKE_C_FLAGS="-O2"
make -j"$(nproc)" pyspiel

echo
echo "=== Sanity check ==="
PYTHONPATH="$OPENSPIEL/build/python:$OPENSPIEL:$OPENSPIEL/open_spiel/python/games" \
  python -c "
import pyspiel
print('pyspiel loaded from:', pyspiel.__file__)
assert 'site-packages' not in pyspiel.__file__, 'pyspiel resolved to a pip package, not the custom build! Check for a stray pip install open_spiel / open-spiel in this env.'
game = pyspiel.load_game('mali_ba')
print('mali_ba game loaded OK:', game)
"

echo
if sudo tailscale status &>/dev/null; then
  TS_IP="$(sudo tailscale ip -4 2>/dev/null)"
  echo "Tailscale: connected ($TS_IP)"
  echo "IMPORTANT: workers only reach whichever host is current in project"
  echo "metadata 'trainer-tailscale-ip'. If moving the trainer here from"
  echo "elsewhere, run this from the desktop (in gcp/) BEFORE starting workers:"
  echo "  ./set_trainer_host.sh $TS_IP"
else
  echo "Tailscale: NOT authenticated. Either set gcp/env.sh's TS_AUTHKEY and run"
  echo "set_tailscale_authkey.sh, or authenticate this VM manually now:"
  echo "  sudo tailscale up"
fi
echo
echo "Done. Before running train_mali_ba.py, export:"
echo "  export PYTHONPATH=$OPENSPIEL/build/python:$OPENSPIEL:$OPENSPIEL/open_spiel/python/games"
echo "Then, inside tmux so it survives an SSH disconnect:"
echo "  tmux new -s trainer"
echo "  cd $GAME_DIR"
echo "  python train_mali_ba.py --distributed --bind_host 0.0.0.0 --queue_port 50000 \\"
echo "      --remote_actors <N> --config_file $GAME_DIR/mali_ba.ini \\"
echo "      2>&1 | tee ~/train_run.log"
echo "  # then detach with Ctrl-b d -- it keeps running after you disconnect"
echo "(omit --authkey to use train_mali_ba.py's built-in default, malibatraining2024,"
echo " matching what workers already use -- pass one explicitly only if you want"
echo " to change the shared secret on both sides)"
