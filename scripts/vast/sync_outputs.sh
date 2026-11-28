#!/usr/bin/env bash
# Periodically copy local training artifacts to a remote (rclone) destination.
# Never uses --delete: the remote keeps older checkpoints if the local file is
# temporarily incomplete.
#
# Usage:
#   REMOTE=gdrive:acne_project/output LOCAL=/workspace/acne_project/output \
#     INTERVAL=600 bash sync_outputs.sh
set -uo pipefail

REMOTE="${REMOTE:?Set REMOTE=rclone-remote:path}"
LOCAL="${LOCAL:-/workspace/acne_project/output}"
INTERVAL="${INTERVAL:-600}"

mkdir -p "$LOCAL"

echo "Syncing $LOCAL -> $REMOTE every ${INTERVAL}s (Ctrl-C to stop)"
while true; do
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if rclone copy "$LOCAL" "$REMOTE" \
      --transfers 4 --checkers 8 --stats 60s --stats-one-line \
      --exclude "*.tmp" --exclude "*.lock"; then
    echo "[$started] sync OK"
  else
    echo "[$started] sync FAILED (will retry in ${INTERVAL}s)"
  fi
  sleep "$INTERVAL"
done
