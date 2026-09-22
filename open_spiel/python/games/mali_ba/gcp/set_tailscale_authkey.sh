#!/bin/bash
# Pushes TS_AUTHKEY (from gcp/env.sh) to this GCP project's
# "tailscale-authkey" metadata, which worker_startup_script.sh and
# worker_vm_setup.sh read at boot/setup time to authenticate Tailscale
# automatically -- no manual "sudo tailscale up" + browser click per worker.
#
# Must be a REUSABLE key (Tailscale admin console > Settings > Keys), or only
# the first worker to use it will succeed. Re-run this whenever you rotate
# the key (they expire, 90 days by default).
#
# Usage: ./set_tailscale_authkey.sh   (reads TS_AUTHKEY from env.sh)
#    or: ./set_tailscale_authkey.sh <key>   (overrides env.sh for this run)
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
KEY="${1:-${TS_AUTHKEY:-}}"
: "${KEY:?Set TS_AUTHKEY in env.sh, or pass the key as an argument}"

gcloud compute project-info add-metadata --project="$GCP_PROJECT_ID" \
  --metadata=tailscale-authkey="$KEY"

echo "Tailscale auth key set for project $GCP_PROJECT_ID."
echo "The key value itself is read fresh from project metadata at each boot,"
echo "so this takes effect for any worker created/rebooted from now on --"
echo "no need to re-run create_worker_template.sh just for a key rotation."
