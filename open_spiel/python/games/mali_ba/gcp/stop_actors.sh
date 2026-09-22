#!/bin/bash
# Stops remote_actors.py on one worker, or ALL running workers if no name
# given -- without restarting it (see restart_actors.sh for kill+restart in
# one step). Useful before stopping/deleting a worker VM, or to pause a
# worker without tearing it down.
#
# Kills the whole "actors" tmux session, which cleanly takes down the
# auto-restart loop + remote_actors.py + all its spawned actor subprocesses.
# Harmless (a no-op) if no "actors" session is currently running.
#
# Reads config from gcp/env.sh if present.
#
# Usage: ./stop_actors.sh [name-or-suffix]
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"

stop_one() {
  local name="$1" zone="$2"
  echo "=== Stopping actors on $name ($zone) ==="
  gcloud compute ssh "$name" --zone="$zone" --project="$GCP_PROJECT_ID" --command='
    tmux kill-session -t actors 2>/dev/null && echo "Stopped." || echo "No actors session was running."
  ' < /dev/null 2>&1
  # < /dev/null above is required: without it, ssh reading from stdin inside
  # a `while read ... done < <(...)` loop swallows the loop's own input
  # stream, silently stopping after the first iteration.
}

if [ -n "${1:-}" ]; then
  case "$1" in
    mali-ba-worker-*) NAME="$1" ;;
    *) NAME="mali-ba-worker-${1}" ;;
  esac
  ZONE="$(gcloud compute instances list --project="$GCP_PROJECT_ID" \
    --filter="name=$NAME" --format="value(zone.basename())")"
  [ -n "$ZONE" ] || { echo "ERROR: no instance named $NAME found" >&2; exit 1; }
  stop_one "$NAME" "$ZONE"
else
  echo "No worker specified -- stopping actors on ALL running mali-ba-worker-* instances..."
  FOUND=0
  while read -r name zone; do
    [ -n "$name" ] || continue
    FOUND=1
    stop_one "$name" "$zone"
  done < <(gcloud compute instances list --project="$GCP_PROJECT_ID" \
      --filter="name~^mali-ba-worker- AND status=RUNNING" --format="value(name,zone.basename())")
  [ "$FOUND" = 1 ] || echo "No running mali-ba-worker-* instances found."
fi
