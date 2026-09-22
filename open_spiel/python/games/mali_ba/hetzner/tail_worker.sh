#!/bin/bash
# SSHes into a Hetzner worker and tails a log file there in real time, so you
# can watch progress without opening a full interactive SSH session yourself.
#
# Reads config from hetzner/env.sh if present.
#
# Usage: ./tail_worker.sh [name] [log-path]
#   [name]: server name in Hetzner (default: mali-ba-hetzner-1). IP is
#     resolved dynamically via `hcloud server ip`, so this keeps working even
#     if the server gets recreated with a new address.
#   [log-path]: defaults to ~/remote_actors.log. Uses tail -F (not -f) so
#     this waits/retries if the file doesn't exist yet rather than erroring
#     immediately -- handy if you run this before starting remote_actors.py.
set -euo pipefail
cd "$(dirname "$0")"
[ -f env.sh ] && source env.sh
export PATH="$HOME/.local/bin:$PATH"

: "${HCLOUD_TOKEN:?Set HCLOUD_TOKEN (see hetzner/env.sh)}"
SSH_KEY="${HETZNER_SSH_KEY_PATH:-$HOME/.ssh/hetzner_mali_ba}"

NAME="${1:-mali-ba-hetzner-1}"
LOG_PATH="${2:-~/remote_actors.log}"

IP="$(hcloud server ip "$NAME")"
echo "Tailing $LOG_PATH on $NAME ($IP) (Ctrl-C to stop)..."
ssh -i "$SSH_KEY" -o StrictHostKeyChecking=accept-new "root@${IP}" "tail -F $LOG_PATH"