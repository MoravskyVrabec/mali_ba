#!/bin/bash
# Updates the "trainer-tailscale-ip" project metadata that worker VMs read
# (via worker_startup_script.sh) at boot to know where train_mali_ba.py
# --distributed is currently running -- desktop, laptop, or a GPU main-vm.
#
# Only affects workers created AFTER this runs (or rebooted after). Workers
# already running against the old host need remote_actors.py restarted by
# hand with the new --server_host, or a reboot.
#
# Reads config from gcp/env.sh if present.
#
# Usage: ./set_trainer_host.sh <tailscale-ip>
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
NEW_HOST="${1:?Usage: $0 <tailscale-ip>}"

gcloud compute project-info add-metadata --project="$GCP_PROJECT_ID" \
  --metadata=trainer-tailscale-ip="$NEW_HOST"

echo "Trainer host set to $NEW_HOST for project $GCP_PROJECT_ID."
echo "Takes effect for workers created/rebooted from now on."
