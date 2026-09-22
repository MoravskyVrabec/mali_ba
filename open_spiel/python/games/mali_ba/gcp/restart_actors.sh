#!/bin/bash
# Restarts remote_actors.py on one worker, or ALL running workers if no name
# given -- needed any time the trainer itself gets restarted (e.g. to pick up
# a new --remote_actors value), since every worker's connection needs to
# reconnect fresh, not just the trainer's.
#
# Kills the whole "actors" tmux session (which cleanly takes down the
# auto-restart loop + remote_actors.py + all its spawned actor subprocesses,
# since they're all descendants of that session -- no orphaned processes)
# and starts a fresh detached one running ~/run_actors.sh.
#
# Requires ~/run_actors.sh to already exist on the worker (written
# automatically by worker_vm_setup.sh / worker_startup_script.sh).
#
# Reads config from gcp/env.sh if present.
#
# Usage: ./restart_actors.sh [name-or-suffix]
#   Omit to restart actors on every currently RUNNING mali-ba-worker-*
#   instance. Accepts a suffix (1, 20260716-2000) or a full VM name.
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"

restart_one() {
  local name="$1" zone="$2"
  echo "=== Restarting actors on $name ($zone) ==="
  gcloud compute ssh "$name" --zone="$zone" --project="$GCP_PROJECT_ID" --command='
    if ! sudo tailscale status &>/dev/null; then
      echo "Tailscale is disconnected -- re-authenticating before restarting actors"
      echo "(a spot preemption/stop cycle can log a node out entirely, not just"
      echo " drop its connection -- especially if the auth key is ephemeral;"
      echo " see DISTRIBUTED_TRAINING.md)."
      TS_AUTHKEY="$(curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/project/attributes/tailscale-authkey)"
      if [ -n "$TS_AUTHKEY" ]; then
        sudo tailscale up --authkey="$TS_AUTHKEY"
      else
        echo "WARNING: no tailscale-authkey project metadata set -- cannot auto-fix. Run sudo tailscale up manually on this worker."
      fi
    fi
    tmux kill-session -t actors 2>/dev/null || true
    sleep 1
    tmux new-session -d -s actors "~/run_actors.sh"
    sleep 2
    echo "tmux sessions now running:"
    tmux list-sessions 2>&1
  ' < /dev/null 2>&1
  # < /dev/null above is required: without it, ssh reading from stdin inside
  # a `while read ... done < <(...)` loop swallows the loop's own input
  # stream, silently stopping after the first iteration (confirmed live --
  # this exact bug caused a 2nd worker to get skipped on 2026-07-16).
}

if [ -n "${1:-}" ]; then
  case "$1" in
    mali-ba-worker-*) NAME="$1" ;;
    *) NAME="mali-ba-worker-${1}" ;;
  esac
  ZONE="$(gcloud compute instances list --project="$GCP_PROJECT_ID" \
    --filter="name=$NAME" --format="value(zone.basename())")"
  [ -n "$ZONE" ] || { echo "ERROR: no instance named $NAME found" >&2; exit 1; }
  restart_one "$NAME" "$ZONE"
else
  echo "No worker specified -- restarting actors on ALL running mali-ba-worker-* instances..."
  FOUND=0
  while read -r name zone; do
    [ -n "$name" ] || continue
    FOUND=1
    restart_one "$name" "$zone"
  done < <(gcloud compute instances list --project="$GCP_PROJECT_ID" \
      --filter="name~^mali-ba-worker- AND status=RUNNING" --format="value(name,zone.basename())")
  [ "$FOUND" = 1 ] || echo "No running mali-ba-worker-* instances found."
fi
