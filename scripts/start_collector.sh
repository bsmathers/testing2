#!/usr/bin/env bash
set -eo pipefail

# Self-contained script to start the TaurosV1A Collector in the background
# Usage:
#   bash scripts/start_collector.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAVE_DIR="${METAMON_SAVE_DIR:-${REPO_DIR}/checkpoints}"
BUFFER_DIR="${REPO_DIR}/buffer"
LOG_DIR="${REPO_DIR}/logs"
LOG_FILE="${LOG_DIR}/collector.log"

mkdir -p "${SAVE_DIR}" "${BUFFER_DIR}" "${LOG_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${HOME}/.cache/metamon}"

# Ensure node/zig are in PATH if present
[ -d "${REPO_DIR}/.node/bin" ] && export PATH="${REPO_DIR}/.node/bin:${PATH}"
[ -d "${REPO_DIR}/.zig" ] && export PATH="${REPO_DIR}/.zig:${PATH}"

# Check for native engine
if [ -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ]; then
    export METAMON_BATTLE_HOST="${REPO_DIR}/metamon/env/vectorized/battle_host_engine.js"
fi

# Detect Python
PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    PYTHON_BIN="python"
fi

# 1. Stop any existing collector process
echo "==> Checking for existing collector processes..."
OLD_PIDS=$(pgrep -f "metamon.rl.online_rl.*--mode collect" || true)
if [ -n "${OLD_PIDS}" ]; then
    echo "    Stopping stale collector PID(s): ${OLD_PIDS}"
    pkill -f "metamon.rl.online_rl.*--mode collect" || true
    sleep 2
fi

# 2. Launch the collector directly with explicit flags (no wrappers, no shifting)
TOTAL_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
if [ "${TOTAL_CPUS}" -gt 2 ]; then
    COLLECTOR_CPUS=$((TOTAL_CPUS - 2))
else
    COLLECTOR_CPUS=1
fi
echo "==> Starting TaurosV1A Collector on ${COLLECTOR_CPUS} CPU workers (128 parallel lanes, 2 cores reserved for Learner)..."
echo "    Repo:   ${REPO_DIR}"
echo "    Save:   ${SAVE_DIR}"
echo "    Buffer: ${BUFFER_DIR}"
echo "    Log:    ${LOG_FILE}"

cd "${REPO_DIR}"

nohup "${PYTHON_BIN}" -m metamon.rl.online_rl \
    --run_config "${REPO_DIR}/metamon/rl/configs/online_runs/taurosv1a.yaml" \
    --mode collect \
    --save_dir "${SAVE_DIR}" \
    --buffer_dir "${BUFFER_DIR}" \
    --lanes 128 \
    --n_workers "${COLLECTOR_CPUS}" \
    > "${LOG_FILE}" 2>&1 &

COLLECTOR_PID=$!
echo "==> Collector started with PID: ${COLLECTOR_PID}"

# 3. Health check: Wait 4 seconds to verify it initialized without crashing
echo "==> Verifying collector startup..."
sleep 4

if kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
    echo "============================================================"
    echo " SUCCESS: Collector is running healthily! (PID: ${COLLECTOR_PID})"
    echo "============================================================"
    echo "--- Recent output from ${LOG_FILE} ---"
    tail -n 15 "${LOG_FILE}"
    echo "------------------------------------------------------------"
    echo "To monitor live: tail -f ${LOG_FILE}"
else
    echo "============================================================"
    echo " ERROR: Collector process exited immediately!"
    echo " Output from ${LOG_FILE}:"
    echo "============================================================"
    cat "${LOG_FILE}"
    exit 1
fi
