#!/bin/bash
# SSHes into a worker VM and tails a log file there in real time, so you can
# watch progress without opening a full interactive SSH session yourself.
#
# Reads config from gcp/env.sh if present.
#
# Usage: ./tail_worker.sh <name-or-suffix> [log-path]
#   <name-or-suffix>: either a full VM name (mali-ba-worker-1) or just the
#     suffix (1, 20260716-2000) -- gets prefixed with mali-ba-worker- if it
#     doesn't already start with that.
#   [log-path]: defaults to ~/remote_actors.log. Use -F semantics (tail -F
#     below) so this waits/retries if the file doesn't exist yet rather than
#     erroring immediately -- handy if you run this before starting
#     remote_actors.py on that worker.
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh

: "${GCP_PROJECT_ID:?Set GCP_PROJECT_ID (see env.sh.example)}"
GCP_ZONE="${GCP_ZONE:-us-central1-b}"

RAW_NAME="${1:?Usage: $0 <name-or-suffix> [log-path]}"
case "$RAW_NAME" in
  mali-ba-worker-*) NAME="$RAW_NAME" ;;
  *) NAME="mali-ba-worker-${RAW_NAME}" ;;
esac
LOG_PATH="${2:-~/remote_actors.log}"

echo "Tailing $LOG_PATH on $NAME (Ctrl-C to stop)..."
gcloud compute ssh "$NAME" --zone="$GCP_ZONE" --project="$GCP_PROJECT_ID" \
  --command="tail -F $LOG_PATH"
