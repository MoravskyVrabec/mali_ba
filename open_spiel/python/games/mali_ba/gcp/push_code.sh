#!/bin/bash
# Pushes the mali_ba repo to a GCP VM's home directory via gcloud compute scp,
# excluding large training artifacts (weights, replay buffers, build output).
# GitHub is intentionally not used yet for this -- see HANDOFF-GCP-SETUP.md.
#
# Run from a Linux shell with gcloud configured (gcloud init) and the target
# VM already created and RUNNING (see create_main_vm.sh / create_worker_vm.sh).
#
# Usage: ./push_code.sh <vm-name> [local-repo-path] [zone]
#   local-repo-path defaults to /mnt/d/Projects/mali_ba (WSL mount of the
#   Windows D: drive). Pass an explicit path if the repo lives elsewhere,
#   e.g. a native Linux checkout at ~/Projects/mali_ba.
#   zone overrides env.sh's GCP_ZONE -- needed when a VM landed in a
#   fallback zone via create_instance_with_retry (see lib_create_with_retry.sh).
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

VM_NAME="${1:?Usage: $0 <vm-name> [local-repo-path] [zone]}"
LOCAL_REPO="${2:-/mnt/d/Projects/mali_ba}"
GCP_ZONE="${3:-${GCP_ZONE:-us-central1-a}}"
: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"

[ -d "$LOCAL_REPO" ] || { echo "ERROR: local repo not found at $LOCAL_REPO" >&2; exit 1; }

echo "Packaging repo at $LOCAL_REPO (excluding weights/buffers/build artifacts)..."
TARBALL="$(mktemp /tmp/mali_ba_repo.XXXXXX.tar.gz)"
tar -czf "$TARBALL" -C "$(dirname "$LOCAL_REPO")" \
  --exclude='*.weights.h5' \
  --exclude='*.pkl.gz' \
  --exclude='*.pkl.gz.old' \
  --exclude='*.mali_ba_replay' \
  --exclude='build' \
  --exclude='.git' \
  --exclude='__pycache__' \
  "$(basename "$LOCAL_REPO")"

echo "Copying to $VM_NAME:~/mali_ba.tar.gz ..."
gcloud compute scp "$TARBALL" "$VM_NAME:~/mali_ba.tar.gz" \
  --zone="$GCP_ZONE" --project="$GCP_PROJECT_ID"

echo "Extracting on $VM_NAME (into ~/mali_ba) ..."
gcloud compute ssh "$VM_NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT_ID" \
  --command="rm -rf ~/mali_ba && mkdir -p ~/mali_ba && tar -xzf ~/mali_ba.tar.gz -C ~/mali_ba --strip-components=1 && rm ~/mali_ba.tar.gz"

rm -f "$TARBALL"
echo "Done. Code is at ~/mali_ba on $VM_NAME. Re-run this script any time to sync local changes."
