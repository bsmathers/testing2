#!/usr/bin/env bash
set -eo pipefail

# Check the comprehensive status of the training run (Learner, Collector, Buffer, Checkpoints)
# Usage:
#   bash scripts/status.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAVE_DIR="${METAMON_SAVE_DIR:-${REPO_DIR}/checkpoints}"
BUFFER_DIR="${REPO_DIR}/buffer"
LOG_DIR="${REPO_DIR}/logs"

echo "============================================================"
echo " TaurosV1A Online RL Training Status"
echo "============================================================"

# 1. Check Learner process (GPU)
echo "[1] LEARNER (GPU Training):"
LEARNER_PID=$(pgrep -f "metamon.rl.online_rl.*--mode learn" || true)
if [ -n "${LEARNER_PID}" ]; then
    echo "    Status: RUNNING (PID: ${LEARNER_PID})"
    if [ -f "${LOG_DIR}/learner.log" ]; then
        echo "    Latest log lines:"
        tail -n 3 "${LOG_DIR}/learner.log" | sed 's/^/      /'
    fi
else
    echo "    Status: NOT RUNNING"
fi

echo ""

# 2. Check Collector process (CPU)
echo "[2] COLLECTOR (CPU Battle Simulation):"
COLLECTOR_PID=$(pgrep -f "metamon.rl.online_rl.*--mode collect" || true)
if [ -n "${COLLECTOR_PID}" ]; then
    echo "    Status: RUNNING (PID: ${COLLECTOR_PID})"
    if [ -f "${LOG_DIR}/collector.log" ]; then
        echo "    Latest log lines:"
        tail -n 3 "${LOG_DIR}/collector.log" | sed 's/^/      /'
    fi
else
    echo "    Status: NOT RUNNING"
fi

echo ""

# 3. Check Buffer Replay Count
echo "[3] ONLINE FIFO BUFFER:"
if [ -d "${BUFFER_DIR}/gen1ou" ]; then
    COUNT=$(ls -1 "${BUFFER_DIR}/gen1ou" 2>/dev/null | wc -l || echo 0)
    echo "    Replays collected in buffer: ${COUNT}"
else
    echo "    Buffer directory not created yet."
fi

echo ""

# 4. Check Saved Checkpoints
echo "[4] CHECKPOINTS:"
CKPT_DIR="${SAVE_DIR}/taurosv1a/ckpts"
if [ -f "${CKPT_DIR}/latest/policy.pt" ]; then
    LATEST_TIME=$(ls -l "${CKPT_DIR}/latest/policy.pt" | awk '{print $6, $7, $8}')
    echo "    latest/policy.pt: updated at ${LATEST_TIME}"
else
    echo "    latest/policy.pt: not created yet"
fi

if [ -d "${CKPT_DIR}/policy_weights" ]; then
    SAVED_COUNT=$(ls -1 "${CKPT_DIR}/policy_weights"/policy_epoch_*.pt 2>/dev/null | wc -l || echo 0)
    echo "    Saved epoch checkpoints: ${SAVED_COUNT}"
fi

echo "============================================================"
