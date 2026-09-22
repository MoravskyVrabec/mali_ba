#!/bin/bash
# Stops main-vm to halt CPU/GPU/RAM billing between training sessions.
# The boot disk (and anything on it -- conda env, build, checkpoints) is kept,
# so no need to re-run main_vm_setup.sh next time -- just start_main_vm.sh.
#
# Reads config from gcp/env.sh if present (copy env.sh.example -> env.sh first).
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
MAIN_VM_NAME="${MAIN_VM_NAME:-mali-ba-main}"

echo "Stopping '$MAIN_VM_NAME' in $GCP_ZONE (project $GCP_PROJECT_ID)..."
gcloud compute instances stop "$MAIN_VM_NAME" \
  --project="$GCP_PROJECT_ID" --zone="$GCP_ZONE"

echo "Stopped. Compute billing halted; boot disk storage billing continues."