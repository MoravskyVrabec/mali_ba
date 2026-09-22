#!/bin/bash
# Creates a spot/preemptible CPU worker VM running MCTS actors
# (remote_actors.py) for Mali-Ba distributed training. A preempted worker only
# loses its in-flight games -- the main-vm's job_timeout_hours mechanism
# re-queues them (see mali_ba.ini). instance-termination-action=STOP (not
# DELETE) so preemption keeps the boot disk (conda env, built pyspiel) around
# instead of destroying setup work; restart with `gcloud compute instances
# start` and it picks back up where it left off.
#
# Reads config from gcp/env.sh if present (copy env.sh.example -> env.sh first).
#
# Usage: ./create_worker_vm.sh [name-suffix]
#   Default name: mali-ba-worker-YYYYMMDD-HHMM (creation timestamp). Pass an
#   explicit suffix (e.g. "1") to override, e.g. for a fixed/manual name.
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh
source lib_create_with_retry.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
GCP_ZONE="${GCP_ZONE:-us-central1-a}"
GCP_NETWORK="${GCP_NETWORK:-default}"
# c2-standard-16 had a spot stockout in every zone we tried on 2026-07-16;
# n2-standard-16 (same vCPU count, general-purpose family) had far better
# spot availability. Override in env.sh if you want to try c2 again.
# n2-standard-16's 64GB RAM was ~3x oversized -- actual actor RSS measured
# at ~1GB/actor (2026-07-19), so 16 actors only needs ~16-19GB. Using a
# custom N2 type (16 vCPU / 24GB) instead cuts the RAM-billed SKU by ~60%.
WORKER_VM_MACHINE_TYPE="${WORKER_VM_MACHINE_TYPE:-n2-custom-16-24576}"
DISK_SIZE_GB="${WORKER_VM_DISK_SIZE_GB:-50}"
SUFFIX="${1:-$(date +%Y%m%d-%H%M)}"
NAME="mali-ba-worker-${SUFFIX}"

# WORKER_IMAGE_FAMILY unset (default) -> stock Ubuntu, full ~20-40 min
# OpenSpiel build via worker_vm_setup.sh. Set it (see create_worker_image.sh)
# to boot from a pre-built golden image instead -- setup then only needs
# push_code.sh + an incremental `make pyspiel`.
WORKER_IMAGE_FAMILY="${WORKER_IMAGE_FAMILY:-}"
if [ -n "$WORKER_IMAGE_FAMILY" ]; then
  IMAGE_FLAGS=(--image-family="$WORKER_IMAGE_FAMILY" --image-project="$GCP_PROJECT_ID")
else
  IMAGE_FLAGS=(--image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud)
fi

# Assign this worker a unique actor-id block so its remote_actors.py
# --actor_id_start never collides with another worker's, no manual bookkeeping
# required. Stored as an instance label (survives even if the worker is
# stopped/restarted) rather than derived from instance count, so a deleted
# worker's block is never silently reused by a still-running one.
echo "Determining a unique actor_id_start block for this worker..."
MAX_BLOCK=0
for b in $(gcloud compute instances list --project="$GCP_PROJECT_ID" \
    --filter="labels.actor-id-block:* AND name~^mali-ba-worker-" \
    --format="value(labels.actor-id-block)" 2>/dev/null); do
  [ "$b" -gt "$MAX_BLOCK" ] 2>/dev/null && MAX_BLOCK=$b
done
ACTOR_ID_BLOCK=$((MAX_BLOCK + 50000))

echo "Creating spot worker-vm '$NAME' (trying $GCP_ZONE first, project $GCP_PROJECT_ID)..."
create_instance_with_retry "$NAME" "$GCP_ZONE" "$WORKER_VM_MACHINE_TYPE" \
  --project="$GCP_PROJECT_ID" \
  --network="$GCP_NETWORK" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  "${IMAGE_FLAGS[@]}" \
  --boot-disk-size="${DISK_SIZE_GB}GB" \
  --boot-disk-type=pd-balanced \
  --labels=actor-id-block="$ACTOR_ID_BLOCK" \
  --tags=mali-ba-worker

echo
echo "Created in zone $CREATED_ZONE with machine type $CREATED_MACHINE_TYPE."
echo "SSH in with:"
echo "  gcloud compute ssh $NAME --zone=$CREATED_ZONE --project=$GCP_PROJECT_ID"
echo
echo "This worker's unique actor_id_start: $ACTOR_ID_BLOCK (reserved via this"
echo "VM's 'actor-id-block' label -- don't reuse it on another worker)."
echo
echo "Push code and run setup with (note the zone, and the actor_id_start"
echo "passed through as the 3rd arg so ~/run_actors.sh uses this exact value):"
echo "  ./push_code.sh $NAME /media/robp/UD/Projects/mali_ba $CREATED_ZONE"
echo "  gcloud compute ssh $NAME --zone=$CREATED_ZONE --project=$GCP_PROJECT_ID \\"
echo "    --command=\"cd ~/mali_ba/open_spiel/python/games/mali_ba && ./gcp/worker_vm_setup.sh '' '' $ACTOR_ID_BLOCK\""
echo
echo "Then start the actors (survives SSH disconnect + auto-restarts on crash --"
echo "see DISTRIBUTED_TRAINING.md's tmux section if you're new to tmux):"
echo "  gcloud compute ssh $NAME --zone=$CREATED_ZONE --project=$GCP_PROJECT_ID"
echo "  tmux new -s actors"
echo "  ~/run_actors.sh"
echo "  # then detach with Ctrl-b d"
