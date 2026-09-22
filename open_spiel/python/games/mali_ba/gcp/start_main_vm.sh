#!/bin/bash
# Starts a previously-stopped main-vm and prints its internal IP (needed by
# worker-vms' remote_actors.py --server_host) once it's back up.
#
# Reads config from gcp/env.sh if present (copy env.sh.example -> env.sh first).
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
MAIN_VM_NAME="${MAIN_VM_NAME:-mali-ba-main}"

echo "Starting '$MAIN_VM_NAME' in $GCP_ZONE (project $GCP_PROJECT_ID)..."
gcloud compute instances start "$MAIN_VM_NAME" \
  --project="$GCP_PROJECT_ID" --zone="$GCP_ZONE"

echo
echo "Internal IP (for worker-vm --server_host):"
gcloud compute instances describe "$MAIN_VM_NAME" \
  --project="$GCP_PROJECT_ID" --zone="$GCP_ZONE" \
  --format='get(networkInterfaces[0].networkIP)'