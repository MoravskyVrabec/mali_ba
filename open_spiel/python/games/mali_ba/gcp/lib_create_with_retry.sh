#!/bin/bash
# Shared helper: attempts `gcloud compute instances create` across a list of
# zones and, if that's still exhausted, a fallback machine type. GCP doesn't
# expose capacity ahead of time -- a ZONE_RESOURCE_POOL_EXHAUSTED error is
# only discoverable by trying, which is exactly what this automates. See
# DISTRIBUTED_TRAINING.md's "Capacity exhaustion" section for the empirical
# zone/machine-type order this defaults to (from the night this was written).
#
# Not a standalone script -- source it: `source lib_create_with_retry.sh`
#
# Usage:
#   create_instance_with_retry <name> <primary-zone> <primary-machine-type> <extra gcloud create args...>
# On success, sets CREATED_ZONE and CREATED_MACHINE_TYPE (which may differ
# from the primary ones passed in) and returns 0. Returns 1 if every
# zone/machine-type combination is exhausted, or immediately on any error
# that ISN'T a capacity issue (bad flag, quota, permissions, etc.) -- those
# are surfaced as-is rather than masked by retries.

: "${WORKER_VM_FALLBACK_ZONES:=us-central1-a us-central1-b us-central1-c us-central1-f us-west1-b us-east1-c}"
: "${WORKER_VM_FALLBACK_MACHINE_TYPE:=e2-standard-16}"

create_instance_with_retry() {
  local name="$1" primary_zone="$2" primary_type="$3"
  shift 3
  local extra_args=("$@")

  local zones_to_try="$primary_zone $WORKER_VM_FALLBACK_ZONES"
  local types_to_try="$primary_type"
  if [ "$primary_type" != "$WORKER_VM_FALLBACK_MACHINE_TYPE" ]; then
    types_to_try="$primary_type $WORKER_VM_FALLBACK_MACHINE_TYPE"
  fi

  local tried=""
  local type zone output combo
  for type in $types_to_try; do
    for zone in $zones_to_try; do
      combo="$zone|$type"
      case " $tried " in *" $combo "*) continue ;; esac
      tried="$tried $combo"

      echo "Trying $type in $zone..." >&2
      if output=$(gcloud compute instances create "$name" \
          --zone="$zone" --machine-type="$type" \
          "${extra_args[@]}" 2>&1); then
        echo "$output"
        CREATED_ZONE="$zone"
        CREATED_MACHINE_TYPE="$type"
        return 0
      fi

      if echo "$output" | grep -q "ZONE_RESOURCE_POOL_EXHAUSTED\|does not have enough resources"; then
        echo "  -> capacity exhausted in $zone for $type, trying next..." >&2
        continue
      else
        echo "$output" >&2
        echo "ERROR: create failed for a reason other than capacity -- not retrying further combinations." >&2
        return 1
      fi
    done
  done

  echo "ERROR: exhausted every zone/machine-type combination tried:$tried" >&2
  return 1
}