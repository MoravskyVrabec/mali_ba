#!/bin/bash
# GCP startup-script -- runs automatically as root on every boot of a worker
# VM created from the mali-ba-worker-template instance template (see
# create_worker_template.sh). The golden image (create_worker_image.sh)
# already has conda/TF/pyspiel fully built, so this does NOT redo setup --
# it just verifies the environment is healthy and writes a login banner
# (MOTD) with this VM's ready-to-paste remote_actors.py command, so opening
# the Console's "Open in browser window" SSH button is immediately
# actionable with no manual setup step.
#
# actor_id_start is derived from this VM's GCP instance ID (globally unique,
# assigned by GCP, never reused) rather than coordinated by an operator-side
# script, since Console-created VMs don't go through create_worker_vm.sh's
# label-based assignment.
#
# --server_host is read from this GCP project's "trainer-tailscale-ip"
# metadata (see set_trainer_host.sh) rather than hardcoded, so the trainer
# can move -- desktop, laptop, or a future GPU main-vm -- without editing
# this script or rebuilding the instance template. Tailscale is a full mesh:
# any two authenticated nodes on the tailnet reach each other directly
# regardless of which one is "the trainer" at any given time. This is only
# read once, at boot -- if the trainer moves while a worker is already
# running, that worker's banner (and any in-flight remote_actors.py) still
# points at the old host until the worker reboots or you restart
# remote_actors.py with a new --server_host by hand.
set -uo pipefail  # no -e: a partial failure here shouldn't block login, just gets reported in the MOTD

RUN_USER="robp"
MALIBA_REPO="/home/$RUN_USER/mali_ba"
OPENSPIEL="/home/$RUN_USER/open_spiel"
GAME_DIR="$MALIBA_REPO/open_spiel/python/games/mali_ba"
PYTHONPATH_VAL="$OPENSPIEL/build/python:$OPENSPIEL:$OPENSPIEL/open_spiel/python/games"
TRAINER_HOST="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/attributes/trainer-tailscale-ip)"
MOTD_SCRIPT="/etc/update-motd.d/99-mali-ba-worker"
LOG_FILE="/var/log/mali-ba-startup.log"

INSTANCE_ID="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/id)"
INSTANCE_NAME="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/instance/name)"
ACTOR_ID_START=$((100000 + INSTANCE_ID % 900000))
CORES="$(nproc)"
RECOMMENDED_ACTORS=$((CORES > 1 ? CORES - 1 : 1))

command -v tmux &>/dev/null || apt-get install -y -qq tmux

SANITY_OK=1
SANITY_OUTPUT="$(sudo -u "$RUN_USER" bash -c "
  source /home/$RUN_USER/miniconda3/etc/profile.d/conda.sh && conda activate mali_ba
  export PYTHONPATH='$PYTHONPATH_VAL'
  python -c \"import pyspiel; pyspiel.load_game('mali_ba')\" 2>&1
")" || SANITY_OK=0
echo "$SANITY_OUTPUT" > "$LOG_FILE"

# Auto-authenticate Tailscale using a reusable auth key from project
# metadata (see set_tailscale_authkey.sh), if not already connected. Falls
# back to a manual-auth message in the banner if the key is missing/expired/
# not reusable -- doesn't block boot either way (set -e is off).
if ! sudo tailscale status &>/dev/null; then
  TS_AUTHKEY="$(curl -s -H 'Metadata-Flavor: Google' http://metadata.google.internal/computeMetadata/v1/project/attributes/tailscale-authkey)"
  if [ -n "$TS_AUTHKEY" ]; then
    sudo tailscale up --authkey="$TS_AUTHKEY" &>> "$LOG_FILE"
  fi
fi

TS_STATUS="NOT authenticated -- run: sudo tailscale up (auto-auth failed or no key set -- see $LOG_FILE)"
if sudo tailscale status &>/dev/null; then
  TS_IP="$(sudo tailscale ip -4 2>/dev/null)"
  TS_STATUS="connected ($TS_IP)"
fi

# Wrapper script: re-runs remote_actors.py in a loop if it ever exits/crashes,
# appending to the same log each time. Meant to be run inside a tmux session
# (see DISTRIBUTED_TRAINING.md's tmux section) so it also survives an SSH
# disconnect -- an unattended overnight run needs both.
RUNNER_SCRIPT="/home/$RUN_USER/run_actors.sh"
cat > "$RUNNER_SCRIPT" <<RUNNER_EOF
#!/bin/bash
source /home/$RUN_USER/miniconda3/etc/profile.d/conda.sh
conda activate mali_ba
export PYTHONPATH=$PYTHONPATH_VAL
cd $GAME_DIR
while true; do
  # Rotate the log if it's grown past 200MB. This loop is the only place a
  # long-running worker ever comes back through (crash restarts, and
  # preemption-restarts since the log lives on the persistent boot disk and
  # tee -a just keeps appending across those) -- with no rotation the file
  # grows unbounded for as long as the worker stays alive.
  if [ -f /home/$RUN_USER/remote_actors.log ] && [ "\$(stat -c%s /home/$RUN_USER/remote_actors.log 2>/dev/null || echo 0)" -gt 209715200 ]; then
    mv -f /home/$RUN_USER/remote_actors.log /home/$RUN_USER/remote_actors.log.1
    echo "\$(date): rotated remote_actors.log (was over 200MB) -> remote_actors.log.1" | tee -a /home/$RUN_USER/remote_actors.log
  fi
  python remote_actors.py --server_host $TRAINER_HOST --server_port 50000 \\
    --num_actors $RECOMMENDED_ACTORS --authkey malibatraining2024 \\
    --actor_id_start $ACTOR_ID_START --cpu_only \\
    2>&1 | tee -a /home/$RUN_USER/remote_actors.log
  echo "\$(date): remote_actors.py exited -- restarting in 10s..." | tee -a /home/$RUN_USER/remote_actors.log
  sleep 10
done
RUNNER_EOF
chown "$RUN_USER:$RUN_USER" "$RUNNER_SCRIPT"
chmod +x "$RUNNER_SCRIPT"

cat > "$MOTD_SCRIPT" <<BANNER_EOF
#!/bin/sh
cat <<BANNER

=== Mali-Ba worker: $INSTANCE_NAME ===
Environment sanity check: $([ "$SANITY_OK" = 1 ] && echo OK || echo "FAILED -- see $LOG_FILE")
Tailscale: $TS_STATUS
CPU cores: $CORES  (recommended --num_actors: $RECOMMENDED_ACTORS)
actor_id_start for this worker: $ACTOR_ID_START

Wrote ~/run_actors.sh -- auto-restarts remote_actors.py if it ever exits.
Run it inside tmux so it also survives an SSH disconnect (see
DISTRIBUTED_TRAINING.md's tmux section if you're new to tmux):
  tmux new -s actors
  ~/run_actors.sh
  # then detach with Ctrl-b d -- it keeps running after you disconnect

BANNER
BANNER_EOF
chmod +x "$MOTD_SCRIPT"
