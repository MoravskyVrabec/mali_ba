#!/bin/bash
# Creates (or replaces) the mali-ba-worker instance template, so new workers
# can be created directly from the GCP Console (Compute Engine > Instance
# Templates > mali-ba-worker-template > Create VM / "Create similar") with
# zero scripting -- no manual setup step, no local dashboard to run/maintain.
#
# Combined with the golden image (create_worker_image.sh) and the
# startup-script (worker_startup_script.sh), a VM created from this template
# needs NO manual setup at all: open its "Open in browser window" SSH button
# in the Console instance list and the login banner (MOTD) already shows
# cores, recommended actor count, and a ready-to-paste remote_actors.py
# command with a unique actor_id_start and the current trainer host.
#
# Re-run this after rebuilding the golden image (create_worker_image.sh) or
# after editing worker_startup_script.sh, to pick up the change -- instance
# templates are immutable, so this deletes and recreates it. Existing VMs
# created from the old template are unaffected.
#
# Reads config from gcp/env.sh if present.
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
GCP_NETWORK="${GCP_NETWORK:-default}"
WORKER_VM_MACHINE_TYPE="${WORKER_VM_MACHINE_TYPE:-n2-standard-16}"
DISK_SIZE_GB="${WORKER_VM_DISK_SIZE_GB:-50}"
WORKER_IMAGE_FAMILY="${WORKER_IMAGE_FAMILY:?Set WORKER_IMAGE_FAMILY -- run create_worker_image.sh first}"
TEMPLATE_NAME="mali-ba-worker-template"

if gcloud compute instance-templates describe "$TEMPLATE_NAME" --project="$GCP_PROJECT_ID" &>/dev/null; then
  echo "Deleting existing template '$TEMPLATE_NAME' (templates are immutable -- recreating)..."
  gcloud compute instance-templates delete "$TEMPLATE_NAME" --project="$GCP_PROJECT_ID" --quiet
fi

echo "Creating instance template '$TEMPLATE_NAME' (image family: $WORKER_IMAGE_FAMILY)..."
gcloud compute instance-templates create "$TEMPLATE_NAME" \
  --project="$GCP_PROJECT_ID" \
  --machine-type="$WORKER_VM_MACHINE_TYPE" \
  --network="$GCP_NETWORK" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family="$WORKER_IMAGE_FAMILY" \
  --image-project="$GCP_PROJECT_ID" \
  --boot-disk-size="${DISK_SIZE_GB}GB" \
  --boot-disk-type=pd-balanced \
  --tags=mali-ba-worker \
  --metadata-from-file=startup-script=worker_startup_script.sh

echo
echo "Done. To create a VM from this template:"
echo "  Console: Compute Engine -> Instance Templates -> $TEMPLATE_NAME -> Create VM"
echo "           (pick a zone, optionally rename it, click Create)"
echo "  CLI:     gcloud compute instances create <name> \\"
echo "             --source-instance-template=$TEMPLATE_NAME --zone=<zone> --project=$GCP_PROJECT_ID"
echo
echo "Then open its 'Open in browser window' SSH button in the Console"
echo "instance list -- the login banner shows cores, recommended actor count,"
echo "current trainer host, and a ready-to-paste remote_actors.py command."
