#!/bin/bash
# Bootstraps a GCP spot CPU worker VM for Mali-Ba distributed training:
# Miniconda + conda env, a vanilla OpenSpiel clone with the mali_ba edits
# applied, and a CPU-only build of the pyspiel target. No GPU/NVIDIA driver
# involved -- actors run MCTS + TF inference on CPU (see --cpu_only in
# remote_actors.py).
#
# Run this AFTER the mali_ba repo has been pushed to this VM (see push_code.sh).
# Safe to re-run -- each step skips work that's already done.
#
# Usage: ./worker_vm_setup.sh [path-to-mali_ba-repo] [path-for-open_spiel-clone] [actor_id_start] [trainer_host]
#   actor_id_start: if omitted, self-derived from this VM's GCP instance ID
#   (globally unique, matches worker_startup_script.sh's approach). Pass the
#   value create_worker_vm.sh printed (its GCP-label-based actor-id-block) to
#   use that instead -- either is collision-free, just pick one consistently.
#   On non-GCP hosts (no instance-id metadata available), this arg is
#   required -- pick any value that doesn't collide with another worker's.
#   trainer_host: if omitted, read from GCP project metadata (set via
#   set_trainer_host.sh) -- GCP-only. On non-GCP hosts (Hetzner, etc.) this
#   arg is required since there's no metadata server to read it from.
set -euo pipefail

MALIBA_REPO="${1:-$HOME/mali_ba}"
OPENSPIEL="${2:-$HOME/open_spiel}"
ACTOR_ID_START_OVERRIDE="${3:-}"
TRAINER_HOST_OVERRIDE="${4:-}"
CONDA_ENV=mali_ba
PYTHON_VERSION=3.11
GAME_DIR="$MALIBA_REPO/open_spiel/python/games/mali_ba"

[ -d "$MALIBA_REPO" ] || { echo "ERROR: mali_ba repo not found at $MALIBA_REPO -- run push_code.sh first" >&2; exit 1; }

echo "=== [1/8] Installing system build dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake clang git python3-dev curl tmux

echo "=== [2/8] Installing Tailscale ==="
# Lets this worker reach a queue server that isn't on the GCP VPC (e.g. your
# desktop). If run on create_worker_image.sh's temp builder VM, its own
# later step explicitly clears tailscaled's state before imaging (baking an
# authenticated node identity into a golden image would make every VM cloned
# from it fight over the same Tailscale node key) -- so it's safe to
# auto-authenticate here unconditionally.
if ! command -v tailscale &>/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sudo sh
else
  echo "tailscale already installed -- skipping"
fi

# Auto-authenticate with a reusable auth key from project metadata (see
# set_tailscale_authkey.sh), if not already connected.
if ! sudo tailscale status &>/dev/null; then
  TS_AUTHKEY="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/attributes/tailscale-authkey)"
  if [ -n "$TS_AUTHKEY" ]; then
    sudo tailscale up --authkey="$TS_AUTHKEY"
  else
    echo "No tailscale-authkey project metadata set -- run 'sudo tailscale up' manually, or gcp/set_tailscale_authkey.sh to automate this."
  fi
fi

echo "=== [3/8] Installing Miniconda ==="
if [ ! -d "$HOME/miniconda3" ]; then
  curl -fsSL -o /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
fi
# shellcheck disable=SC1091
source "$HOME/miniconda3/etc/profile.d/conda.sh"

echo "=== [4/8] Creating conda env '$CONDA_ENV' ==="
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
if ! conda env list | grep -q "^$CONDA_ENV "; then
  conda create -y -n "$CONDA_ENV" python=$PYTHON_VERSION
fi
conda activate "$CONDA_ENV"
pip install --upgrade pip -q
pip install -r "$GAME_DIR/requirements-cpu.txt"

echo "=== [5/8] Cloning vanilla OpenSpiel + applying mali_ba edits ==="
if [ ! -d "$OPENSPIEL" ]; then
  git clone --depth 1 https://github.com/google-deepmind/open_spiel.git "$OPENSPIEL"
fi
bash "$GAME_DIR/gcp/apply_open_spiel_edits.sh" "$OPENSPIEL"

echo "=== [6/8] Bootstrapping OpenSpiel third-party dependencies (install.sh) ==="
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

echo "=== [7/8] Symlinking mali_ba code into the OpenSpiel clone ==="
ln -sf  "$MALIBA_REPO/open_spiel/python/pybind11/games_mali_ba.cc" "$OPENSPIEL/open_spiel/python/pybind11/games_mali_ba.cc"
ln -sf  "$MALIBA_REPO/open_spiel/python/pybind11/games_mali_ba.h"  "$OPENSPIEL/open_spiel/python/pybind11/games_mali_ba.h"
ln -sfn "$MALIBA_REPO/open_spiel/games/mali_ba"                    "$OPENSPIEL/open_spiel/games/mali_ba"
ln -sfn "$GAME_DIR"                                                "$OPENSPIEL/open_spiel/python/games/mali_ba"

echo "=== [8/8] Building pyspiel (first build takes a while) ==="
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

CORES="$(nproc)"
# remote_actors.py has no CPU-aware default of its own (unlike train_mali_ba.py's
# local --num_actors, which uses max(1, cpu_count - 2) to leave headroom for its
# own trainer/learner process). A dedicated actor-only worker has no such
# competing process, so recommend leaving just 1 core for the OS/tailscaled/
# remote_actors.py's own orchestration overhead rather than 2.
RECOMMENDED_ACTORS=$((CORES > 1 ? CORES - 1 : 1))

if [ -n "$ACTOR_ID_START_OVERRIDE" ]; then
  ACTOR_ID_START="$ACTOR_ID_START_OVERRIDE"
else
  INSTANCE_ID="$(curl -s -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/id || true)"
  if [ -z "$INSTANCE_ID" ]; then
    echo "ERROR: no GCP instance-id metadata available (not a GCP host?) and no actor_id_start arg given." >&2
    echo "       Pass one explicitly: ./worker_vm_setup.sh '' '' <actor_id_start> [trainer_host]" >&2
    exit 1
  fi
  ACTOR_ID_START=$((100000 + INSTANCE_ID % 900000))
fi
if [ -n "$TRAINER_HOST_OVERRIDE" ]; then
  TRAINER_HOST="$TRAINER_HOST_OVERRIDE"
else
  TRAINER_HOST="$(curl -s -m 3 -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/attributes/trainer-tailscale-ip || true)"
  if [ -z "$TRAINER_HOST" ]; then
    echo "ERROR: no GCP trainer-tailscale-ip metadata available (not a GCP host?) and no trainer_host arg given." >&2
    echo "       Pass one explicitly: ./worker_vm_setup.sh '' '' <actor_id_start> <trainer_host>" >&2
    exit 1
  fi
fi
PYTHONPATH_VAL="$OPENSPIEL/build/python:$OPENSPIEL:$OPENSPIEL/open_spiel/python/games"

# Wrapper script: activates the env and re-runs remote_actors.py in a loop if
# it ever exits/crashes, appending to the same log each time. Run this INSIDE
# a tmux session (see DISTRIBUTED_TRAINING.md's tmux section) so it survives
# an SSH disconnect too -- an unattended overnight run needs both.
cat > "$HOME/run_actors.sh" <<RUNNER_EOF
#!/bin/bash
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate mali_ba
export PYTHONPATH=$PYTHONPATH_VAL
cd $GAME_DIR
while true; do
  # Rotate the log if it's grown past 200MB. This loop is the only place a
  # long-running worker ever comes back through (crash restarts, and
  # preemption-restarts since the log lives on the persistent boot disk and
  # tee -a just keeps appending across those) -- with no rotation the file
  # grows unbounded for as long as the worker stays alive.
  if [ -f ~/remote_actors.log ] && [ "\$(stat -c%s ~/remote_actors.log 2>/dev/null || echo 0)" -gt 209715200 ]; then
    mv -f ~/remote_actors.log ~/remote_actors.log.1
    echo "\$(date): rotated remote_actors.log (was over 200MB) -> remote_actors.log.1" | tee -a ~/remote_actors.log
  fi
  python remote_actors.py --server_host $TRAINER_HOST --server_port 50000 \\
    --num_actors $RECOMMENDED_ACTORS --authkey malibatraining2024 \\
    --actor_id_start $ACTOR_ID_START --cpu_only \\
    2>&1 | tee -a ~/remote_actors.log
  echo "\$(date): remote_actors.py exited -- restarting in 10s..." | tee -a ~/remote_actors.log
  sleep 10
done
RUNNER_EOF
chmod +x "$HOME/run_actors.sh"

echo
echo "=== Machine report ==="
echo "CPU cores available: $CORES"
echo "Recommended --num_actors for remote_actors.py on this machine: $RECOMMENDED_ACTORS"
echo "(leaves 1 core for the OS / tailscaled / remote_actors.py's own overhead;"
echo " each actor is a separate MCTS + TF-inference process with no internal"
echo " thread-count limit set, so oversubscribing beyond core count will thrash)"

echo
if sudo tailscale status &>/dev/null; then
  echo "Tailscale: connected ($(sudo tailscale ip -4 2>/dev/null)) -- auto-authenticated."
else
  echo "Tailscale: NOT authenticated. Either set gcp/env.sh's TS_AUTHKEY and run"
  echo "set_tailscale_authkey.sh, or authenticate this worker manually now:"
  echo "  sudo tailscale up"
fi
echo
echo "Wrote ~/run_actors.sh -- auto-restarts remote_actors.py if it ever exits."
echo "actor_id_start for this worker: $ACTOR_ID_START"
echo "Trainer host (from project metadata): $TRAINER_HOST"
echo
echo "Run it inside tmux so it also survives an SSH disconnect"
echo "(see DISTRIBUTED_TRAINING.md's tmux section if you're new to tmux):"
echo "  tmux new -s actors"
echo "  ~/run_actors.sh"
echo "  # then detach with Ctrl-b d -- it keeps running after you disconnect"
