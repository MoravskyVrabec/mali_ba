#!/bin/bash
# Creates the on-demand GPU learner VM (main-vm) for Mali-Ba distributed
# training: runs the trainer + learner + queue_server. Persistent -- holds the
# model and replay buffer, so this is deliberately NOT a spot instance.
#
# Reads config from gcp/env.sh if present (copy env.sh.example -> env.sh first).
#
# Tries nvidia-tesla-t4 (n1-standard-8) across a list of fallback zones first;
# if that's exhausted everywhere, automatically falls back to nvidia-l4
# (g2-standard-8, ~2x T4 hourly cost, quota already granted -- see
# DISTRIBUTED_TRAINING.md's capacity-exhaustion section). T4 was exhausted in
# every zone tried on 2026-07-16, so don't be surprised if this falls back.
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh
source lib_create_with_retry.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
GCP_NETWORK="${GCP_NETWORK:-default}"
MAIN_VM_NAME="${MAIN_VM_NAME:-mali-ba-main}"
MAIN_VM_MACHINE_TYPE="${MAIN_VM_MACHINE_TYPE:-n1-standard-8}"
MAIN_VM_GPU_TYPE="${MAIN_VM_GPU_TYPE:-nvidia-tesla-t4}"
MAIN_VM_FALLBACK_MACHINE_TYPE="${MAIN_VM_FALLBACK_MACHINE_TYPE:-g2-standard-8}"
MAIN_VM_FALLBACK_GPU_TYPE="${MAIN_VM_FALLBACK_GPU_TYPE:-nvidia-l4}"
DISK_SIZE_GB="${MAIN_VM_DISK_SIZE_GB:-150}"

COMMON_ARGS=(
  --project="$GCP_PROJECT_ID"
  --network="$GCP_NETWORK"
  --maintenance-policy=TERMINATE
  --provisioning-model=STANDARD
  --image-family=ubuntu-2204-lts
  --image-project=ubuntu-os-cloud
  --boot-disk-size="${DISK_SIZE_GB}GB"
  --boot-disk-type=pd-ssd
  --tags=mali-ba-main
)

# lib_create_with_retry.sh's own internal machine-type fallback (default
# e2-standard-16) must be disabled for GPU VMs: it would combine e2 (no GPU
# support at all) with the --accelerator flag already baked into extra_args
# below, producing an invalid-config error rather than a clean capacity
# signal, which would stop retrying prematurely instead of exhausting all
# zones first. Pin the library's fallback to match the primary type each
# call so it never adds its own second machine-type stage -- the T4-then-L4
# fallback here is handled explicitly by this script instead.
echo "Creating main-vm '$MAIN_VM_NAME' (trying $MAIN_VM_GPU_TYPE first, project $GCP_PROJECT_ID)..."
WORKER_VM_FALLBACK_MACHINE_TYPE="$MAIN_VM_MACHINE_TYPE"
if ! create_instance_with_retry "$MAIN_VM_NAME" "$GCP_ZONE" "$MAIN_VM_MACHINE_TYPE" \
    --accelerator="type=$MAIN_VM_GPU_TYPE,count=1" \
    "${COMMON_ARGS[@]}"; then
  echo
  echo "$MAIN_VM_GPU_TYPE exhausted everywhere tried -- falling back to $MAIN_VM_FALLBACK_GPU_TYPE ($MAIN_VM_FALLBACK_MACHINE_TYPE)..."
  WORKER_VM_FALLBACK_MACHINE_TYPE="$MAIN_VM_FALLBACK_MACHINE_TYPE"
  create_instance_with_retry "$MAIN_VM_NAME" "$GCP_ZONE" "$MAIN_VM_FALLBACK_MACHINE_TYPE" \
    --accelerator="type=$MAIN_VM_FALLBACK_GPU_TYPE,count=1" \
    "${COMMON_ARGS[@]}"
fi

echo
echo "Created in zone $CREATED_ZONE with machine type $CREATED_MACHINE_TYPE."
echo "Wait for it to reach RUNNING, then:"
echo "  Internal IP: gcloud compute instances describe $MAIN_VM_NAME --zone=$CREATED_ZONE --project=$GCP_PROJECT_ID --format='get(networkInterfaces[0].networkIP)'"
echo "  SSH:         gcloud compute ssh $MAIN_VM_NAME --zone=$CREATED_ZONE --project=$GCP_PROJECT_ID"
echo
echo "Push code and run setup with (note the zone!):"
echo "  ./push_code.sh $MAIN_VM_NAME <local-mali_ba-repo-path> $CREATED_ZONE"
echo "  gcloud compute ssh $MAIN_VM_NAME --zone=$CREATED_ZONE --project=$GCP_PROJECT_ID \\"
echo "    --command=\"cd ~/mali_ba/open_spiel/python/games/mali_ba && ./gcp/main_vm_setup.sh\""
