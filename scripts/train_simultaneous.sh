#!/usr/bin/env bash
set -eo pipefail

# Simultaneous Learner + Collector Launcher for TaurosV1A Online RL
#
# Runs:
#   1. Learner: Continuous GPU gradient updates on RTX 5090
#   2. Collector: Continuous CPU multithreaded battle generation across all CPU cores
#
# Usage:
#   bash scripts/train_simultaneous.sh [save_dir] [buffer_dir] [extra flags...]
#
# Example:
#   bash scripts/train_simultaneous.sh /root/testing2/checkpoints /root/testing2/buffer --log

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAVE_DIR="${1:-${REPO_DIR}/checkpoints}"
BUFFER_DIR="${2:-${REPO_DIR}/buffer}"

[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift

# Auto-detect number of available CPU cores
NUM_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
WORKERS="${WORKERS:-${NUM_CPUS}}"
LANES="${LANES:-128}"

mkdir -p "${SAVE_DIR}" "${BUFFER_DIR}" logs

echo "=========================================================="
echo " Starting Simultaneous TaurosV1A Online RL"
echo " Save Dir:    ${SAVE_DIR}"
echo " Buffer Dir:  ${BUFFER_DIR}"
echo " CPU Cores:   ${NUM_CPUS} detected (using ${WORKERS} worker threads)"
echo " Battle Lanes: ${LANES} parallel lanes"
echo "=========================================================="

# Clean shutdown handler for both processes
cleanup() {
    echo ""
    echo "Stopping simultaneous training..."
    if [ -n "${LEARNER_PID}" ] && kill -0 "${LEARNER_PID}" 2>/dev/null; then
        echo "Stopping Learner (PID ${LEARNER_PID})..."
        kill -TERM "${LEARNER_PID}" 2>/dev/null || true
    fi
    if [ -n "${COLLECTOR_PID}" ] && kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
        echo "Stopping Collector (PID ${COLLECTOR_PID})..."
        kill -TERM "${COLLECTOR_PID}" 2>/dev/null || true
    fi
    wait "${LEARNER_PID}" 2>/dev/null || true
    wait "${COLLECTOR_PID}" 2>/dev/null || true
    echo "All training processes stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# 1. Start Learner in background
echo "[1/2] Starting Learner on GPU (logs -> logs/learner.log)..."
bash "${SCRIPT_DIR}/train_taurosv1a.sh" learn "${SAVE_DIR}" "${BUFFER_DIR}" "$@" > logs/learner.log 2>&1 &
LEARNER_PID=$!
echo "      Learner PID: ${LEARNER_PID}"

# Give learner a few seconds to initialize
sleep 5

# 2. Start Multithreaded Collector across CPU cores
echo "[2/2] Starting Multithreaded Collector across ${WORKERS} CPU workers (logs -> logs/collector.log)..."
bash "${SCRIPT_DIR}/train_taurosv1a.sh" collect "${SAVE_DIR}" "${BUFFER_DIR}" --lanes "${LANES}" --n_workers "${WORKERS}" "$@" > logs/collector.log 2>&1 &
COLLECTOR_PID=$!
echo "      Collector PID: ${COLLECTOR_PID}"

echo "=========================================================="
echo " Both Learner and Collector are running simultaneously!"
echo " - Monitor Learner:   tail -f logs/learner.log"
echo " - Monitor Collector: tail -f logs/collector.log"
echo " - Press Ctrl+C in this terminal to stop both."
echo "=========================================================="

# Tail learner logs to stdout so the user sees live progress
tail -f logs/learner.log &
TAIL_PID=$!

# Wait for learner to exit
wait "${LEARNER_PID}"
kill "${TAIL_PID}" 2>/dev/null || true
