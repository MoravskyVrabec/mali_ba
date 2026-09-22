#!/bin/bash
# Builds a fully-provisioned worker VM from scratch and snapshots it as a
# reusable GCP image ("golden image"), so future create_worker_vm.sh calls
# skip the ~20-40 min OpenSpiel/abseil-cpp build entirely -- new workers boot
# with conda, TF, and a compiled pyspiel.so already in place.
#
# Re-run this whenever you want to refresh the baseline: dependency bumps
# (requirements-cpu.txt, install.sh's pinned abseil-cpp version), system
# security patches, or after adding/removing mali_ba C++ source files (that
# changes the GAME_SOURCES list in games/CMakeLists.txt, which needs a cmake
# reconfigure -- see note below).
#
# For routine mali_ba code edits that DON'T add/remove files, you do NOT need
# to rebuild the image: create_worker_vm.sh (from the existing image) +
# push_code.sh + an incremental `make pyspiel` on the worker picks up the
# change in seconds, since abseil-cpp/json/pybind11 are already baked in and
# only mali_ba's own object files need recompiling/relinking. If you did
# add/remove a mali_ba source file, either rebuild this image, or on the
# worker re-run: cd ~/open_spiel/build && cmake . && make -j$(nproc) pyspiel
#
# Reads config from gcp/env.sh if present (copy env.sh.example -> env.sh first).
#
# Usage: ./create_worker_image.sh [local-mali_ba-repo-path]
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh
source lib_create_with_retry.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
GCP_NETWORK="${GCP_NETWORK:-default}"
WORKER_VM_MACHINE_TYPE="${WORKER_VM_MACHINE_TYPE:-n2-standard-16}"
DISK_SIZE_GB="${WORKER_VM_DISK_SIZE_GB:-50}"
LOCAL_REPO="${1:-/media/robp/UD/Projects/mali_ba}"

IMAGE_FAMILY="mali-ba-worker-base"
IMAGE_NAME="${IMAGE_FAMILY}-$(date +%Y%m%d-%H%M%S)"
BUILD_VM_NAME="mali-ba-image-builder"

[ -d "$LOCAL_REPO" ] || { echo "ERROR: local repo not found at $LOCAL_REPO" >&2; exit 1; }

echo "=== [1/7] Creating temporary build VM '$BUILD_VM_NAME' ==="
# Standard (non-spot) provisioning -- don't want a preemption mid-build to
# waste a 20-40 min run. This VM is deleted at the end regardless of outcome.
create_instance_with_retry "$BUILD_VM_NAME" "$GCP_ZONE" "$WORKER_VM_MACHINE_TYPE" \
  --project="$GCP_PROJECT_ID" \
  --network="$GCP_NETWORK" \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_SIZE_GB}GB" \
  --boot-disk-type=pd-balanced \
  --tags=mali-ba-worker
echo "Created in zone $CREATED_ZONE with machine type $CREATED_MACHINE_TYPE."

cleanup() {
  echo "=== Cleaning up temporary build VM (best-effort) ==="
  gcloud compute instances delete "$BUILD_VM_NAME" \
    --project="$GCP_PROJECT_ID" --zone="${CREATED_ZONE:-$GCP_ZONE}" --quiet || true
}
trap cleanup EXIT

echo "=== [2/7] Waiting for SSH to become available ==="
sleep 20

echo "=== [3/7] Pushing mali_ba code ==="
./push_code.sh "$BUILD_VM_NAME" "$LOCAL_REPO" "$CREATED_ZONE"

echo "=== [4/7] Running full worker setup (conda, deps, OpenSpiel build) ==="
gcloud compute ssh "$BUILD_VM_NAME" --zone="$CREATED_ZONE" --project="$GCP_PROJECT_ID" \
  --command="cd ~/mali_ba/open_spiel/python/games/mali_ba && ./gcp/worker_vm_setup.sh"

echo "=== [5/7] Clearing Tailscale machine key ==="
# Merely installing the tailscale package starts tailscaled, which generates
# and persists a machine key immediately -- independent of whether
# `tailscale up` was ever run. Baking that key into the image means every VM
# cloned from it shares one Tailscale node identity and fights over it (seen
# firsthand: two workers from the same image showed a "duplicate node key"
# and only one was reachable). Deleting the state file forces a fresh,
# genuinely unique key on each new VM's first boot.
gcloud compute ssh "$BUILD_VM_NAME" --zone="$CREATED_ZONE" --project="$GCP_PROJECT_ID" \
  --command="sudo systemctl stop tailscaled && sudo rm -f /var/lib/tailscale/tailscaled.state && sudo systemctl start tailscaled"

echo "=== [6/7] Stopping VM and creating image '$IMAGE_NAME' (family: $IMAGE_FAMILY) ==="
gcloud compute instances stop "$BUILD_VM_NAME" --project="$GCP_PROJECT_ID" --zone="$CREATED_ZONE"
gcloud compute images create "$IMAGE_NAME" \
  --project="$GCP_PROJECT_ID" \
  --source-disk="$BUILD_VM_NAME" \
  --source-disk-zone="$CREATED_ZONE" \
  --family="$IMAGE_FAMILY"

echo "=== [7/7] Done -- temporary build VM will now be deleted ==="
echo
echo "New golden image: $IMAGE_NAME (family: $IMAGE_FAMILY)"
echo "Set this in env.sh to use it:"
echo "  export WORKER_IMAGE_FAMILY=\"$IMAGE_FAMILY\""
echo "create_worker_vm.sh always picks the latest image in the family automatically."
