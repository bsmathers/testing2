#!/usr/bin/env bash
set -eo pipefail

# Periodic checkpoint backup monitor for Vast.ai volumes / persistent storage
#
# Usage:
#   bash scripts/backup_checkpoints.sh [SRC_DIR] [DEST_DIR] [INTERVAL_SEC]
#
# Examples:
#   # Run in background syncing every 5 minutes (300s):
#   nohup bash scripts/backup_checkpoints.sh ./checkpoints/taurosv1a /workspace/backup/taurosv1a 300 > backup.log 2>&1 &
#
#   # Or run directly in tmux:
#   bash scripts/backup_checkpoints.sh ./checkpoints/taurosv1a /workspace/backup/taurosv1a 300

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SRC="${1:-${REPO_DIR}/checkpoints/taurosv1a}"
DEST="${2:-/workspace/backup/taurosv1a}"
INTERVAL="${3:-300}"

echo "=========================================================="
echo " Starting Checkpoint Backup Monitor"
echo " Source:      ${SRC}"
echo " Destination: ${DEST}"
echo " Interval:    Every ${INTERVAL} seconds"
echo "=========================================================="

mkdir -p "${DEST}"

trap 'echo ""; echo "Backup monitor stopped by user."; exit 0' SIGINT SIGTERM

while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    
    if [ -d "${SRC}" ]; then
        # Count available checkpoints
        CKPT_COUNT=$(find "${SRC}" -name "policy_epoch_*.pt" 2>/dev/null | wc -l | tr -d ' ')
        LATEST_EXISTS=$([ -f "${SRC}/ckpts/latest/policy.pt" ] && echo "yes" || echo "no")
        
        echo "[${TIMESTAMP}] Syncing checkpoints (${CKPT_COUNT} epochs found, latest: ${LATEST_EXISTS})..."
        
        # rsync with --update: skips files that are already up-to-date on destination
        rsync -av --update \
            --include='*/' \
            --include='*.pt' \
            --include='*.yaml' \
            --include='*.json' \
            --exclude='*' \
            "${SRC}/" "${DEST}/" | grep -v '/$' || true

        echo "[${TIMESTAMP}] Backup sync complete. Sleeping ${INTERVAL}s."
    else
        echo "[${TIMESTAMP}] Waiting for source directory ${SRC} to be created by training..."
    fi

    sleep "${INTERVAL}"
done
