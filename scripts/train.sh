#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# TaurosV1A Parallel Online RL Training (Simultaneous Collector + Learner)
#
# Runs simultaneous Collector + Learner in parallel:
#   1. Collector (CPU/GPU): Continuous battle simulation across all CPU cores,
#      generating replays directly into the online FIFO buffer (buffer/gen1ou).
#   2. Learner (GPU): Continuous gradient updates on RTX 5090, training the
#      ~35M GroupedV2 architecture (grouped_v2_medium.gin) from scratch.
#   3. Rate Monitor: Live background tracking of buffer replay production rate.
#
# Opponent Pool: TaurosV1 static roster without SmallG1, self-play starts at Epoch 300.
#
# Usage:
#   bash scripts/train.sh                # Standard parallel training (recommended in tmux)
#   bash scripts/train.sh --log          # With WandB logging enabled on Learner
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAVE_DIR="${METAMON_SAVE_DIR:-${REPO_DIR}/checkpoints}"
BUFFER_DIR="${REPO_DIR}/buffer"
LOG_DIR="${REPO_DIR}/logs"
COLLECTOR_LOG="${LOG_DIR}/collector.log"

mkdir -p "${SAVE_DIR}" "${BUFFER_DIR}/gen1ou" "${LOG_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export METAMON_SAVE_DIR="${SAVE_DIR}"

if [ -z "${METAMON_CACHE_DIR:-}" ]; then
    if [ -d "/root/cache_dir" ]; then
        export METAMON_CACHE_DIR="/root/cache_dir"
    else
        export METAMON_CACHE_DIR="${HOME}/.cache/metamon"
    fi
fi
export METAMON_ALLOW_ANY_POKE_ENV=1

# Ensure local node/zig are in PATH if present
[ -d "${REPO_DIR}/.node/bin" ] && export PATH="${REPO_DIR}/.node/bin:${PATH}"
[ -d "${REPO_DIR}/.zig" ] && export PATH="${REPO_DIR}/.zig:${PATH}"

# Check for native @pkmn/engine Zig simulator
if [ ! -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ]; then
    echo "[Info] Native Zig simulator not found. Running setup via scripts/setup_pkmn_engine.sh..."
    bash "${SCRIPT_DIR}/setup_pkmn_engine.sh"
fi

# Ensure pkmn_engine_common.js is present in @pkmn/engine package
if [ -f "${REPO_DIR}/metamon/env/vectorized/pkmn_engine_common.js" ]; then
    mkdir -p "${REPO_DIR}/metamon/env/vectorized/node_modules/@pkmn/engine/build/pkg"
    cp -f "${REPO_DIR}/metamon/env/vectorized/pkmn_engine_common.js" "${REPO_DIR}/metamon/env/vectorized/node_modules/@pkmn/engine/build/pkg/common.js" 2>/dev/null || true
fi

if [ -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ]; then
    export METAMON_BATTLE_HOST="${REPO_DIR}/metamon/env/vectorized/battle_host_engine.js"
    echo "[Info] High-speed @pkmn/engine active: ${METAMON_BATTLE_HOST}"
else
    echo "[Warn] Running with standard Showdown simulator (pkmn-showdown.node not found)"
fi

# Detect Python
PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    PYTHON_BIN="python"
fi

# Clean up any stale processes from prior runs
echo "[1/4] Terminating any stale processes..."
pkill -9 -f "metamon.rl.online_rl" 2>/dev/null || true
pkill -9 -f "battle_host" 2>/dev/null || true
pkill -9 -f "pkmn-showdown" 2>/dev/null || true
sleep 1

# Detect hardware (use all but 2 cores for collector to leave dedicated CPU for learner/OS)
TOTAL_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
if [ "${TOTAL_CPUS}" -gt 2 ]; then
    COLLECTOR_CPUS=$((TOTAL_CPUS - 2))
else
    COLLECTOR_CPUS=1
fi
LANES="${LANES:-128}"

echo "[2/4] Configuration:"
echo "      Run Name:       taurosv1a"
echo "      Architecture:   TaurosV1 (~35M params, grouped_v2_medium.gin)"
echo "      Init:           Random initialization (--from_scratch)"
echo "      Total Epochs:   1500 (1000 steps/epoch)"
echo "      Self-Play:      Starts at Epoch 300 (min_epoch: 300)"
echo "      Dataset:        pac-tauros (offline mix) + online FIFO buffer"
echo "      Collector:      ${LANES} parallel lanes across ${COLLECTOR_CPUS} CPU workers (${TOTAL_CPUS} total cores, 2 reserved for Learner)"
echo "      Save Dir:       ${SAVE_DIR}"
echo "      Buffer Dir:     ${BUFFER_DIR}"
echo "      Collector Log:  ${COLLECTOR_LOG}"
echo ""

# Pre-warm usage stats in memory to eliminate cold-start delay
echo "      Pre-warming usage stats..."
"${PYTHON_BIN}" -c "from metamon.backend.team_prediction.usage_stats import get_usage_stats; get_usage_stats('gen1ubers')" 2>/dev/null || true

# Process cleanup handler
cleanup() {
    echo ""
    echo "[Shutdown] Terminating parallel training processes..."
    [ -n "${MONITOR_PID}" ] && kill "${MONITOR_PID}" 2>/dev/null || true
    [ -n "${COLLECTOR_PID}" ] && kill -TERM "${COLLECTOR_PID}" 2>/dev/null || true
    [ -n "${LEARNER_PID}" ] && kill -TERM "${LEARNER_PID}" 2>/dev/null || true
    sleep 2
    pkill -9 -P "${COLLECTOR_PID}" 2>/dev/null || true
    pkill -9 -P "${LEARNER_PID}" 2>/dev/null || true
    [ -n "${COLLECTOR_PID}" ] && kill -9 "${COLLECTOR_PID}" 2>/dev/null || true
    [ -n "${LEARNER_PID}" ] && kill -9 "${LEARNER_PID}" 2>/dev/null || true
    pkill -9 -f "battle_host" 2>/dev/null || true
    pkill -9 -f "pkmn-showdown" 2>/dev/null || true
    pkill -9 -f "metamon.rl.online_rl" 2>/dev/null || true
    echo "[Shutdown] All training processes stopped."
    exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# Parse CLI flags (pass e.g. --log to Learner)
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--" ]; then
        EXTRA_ARGS+=("$arg")
    fi
done

cd "${REPO_DIR}"

# Step 3: Launch the Collector in background
echo "[3/4] Starting Collector in background (${LANES} lanes, ${COLLECTOR_CPUS} CPU workers)..."
"${PYTHON_BIN}" -m metamon.rl.online_rl \
    --run_config "${REPO_DIR}/metamon/rl/configs/online_runs/taurosv1a.yaml" \
    --mode collect \
    --save_dir "${SAVE_DIR}" \
    --buffer_dir "${BUFFER_DIR}" \
    --lanes "${LANES}" \
    --n_workers "${COLLECTOR_CPUS}" \
    > "${COLLECTOR_LOG}" 2>&1 &
COLLECTOR_PID=$!
echo "      Collector PID: ${COLLECTOR_PID}"

# Health check: verify collector initializes without immediately crashing
echo "      Verifying collector health and engine status..."
ENGINE_VERIFIED=0
for i in {1..20}; do
    if ! kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
        echo "============================================================"
        echo " [ERROR] Collector process died immediately on startup!"
        echo "============================================================"
        cat "${COLLECTOR_LOG}"
        exit 1
    fi
    if grep -q "HIGH-SPEED NATIVE ZIG ENGINE ACTIVE" "${COLLECTOR_LOG}" 2>/dev/null; then
        echo "      [Engine Verified] High-speed native Zig @pkmn/engine is ACTIVE!"
        ENGINE_VERIFIED=1
        break
    fi
    sleep 1
done

if [ "${ENGINE_VERIFIED}" -eq 0 ]; then
    echo "      [Info] Collector is running (PID: ${COLLECTOR_PID}). Log: ${COLLECTOR_LOG}"
fi

# Step 4: Background Buffer Rate Monitor
# Prints live buffer production statistics every 30 seconds
(
    PREV_COUNT=$(ls -1 "${BUFFER_DIR}/gen1ou" 2>/dev/null | wc -l || echo 0)
    while true; do
        sleep 30
        if ! kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
            echo ""
            echo ">>> [ALERT] Collector PID ${COLLECTOR_PID} has stopped unexpectedly! Check ${COLLECTOR_LOG} <<<"
            break
        fi
        CURR_COUNT=$(ls -1 "${BUFFER_DIR}/gen1ou" 2>/dev/null | wc -l || echo 0)
        DIFF=$((CURR_COUNT - PREV_COUNT))
        RATE=$(awk "BEGIN {printf \"%.1f\", ${DIFF} / 30.0}")
        echo ">>> [Buffer Monitor] Replays in buffer: ${CURR_COUNT} (+${DIFF} in last 30s | ${RATE} games/sec) | Collector: ACTIVE <<<"
        PREV_COUNT="${CURR_COUNT}"
    done
) &
MONITOR_PID=$!

# Step 5: Launch the Learner on GPU in foreground (and log to logs/learner.log)
LEARNER_LOG="${LOG_DIR}/learner.log"
echo "[4/4] Starting Learner on GPU (RTX 5090)..."
echo "============================================================"
echo " Parallel Online RL is active!"
echo " - Learner training progress is live below."
echo " - Learner log:   tail -f ${LEARNER_LOG}"
echo " - Collector log: tail -f ${COLLECTOR_LOG}"
echo " - Buffer monitor reports production rate every 30s."
echo " - Press Ctrl+C at any time to stop both processes cleanly."
echo "============================================================"
echo ""

"${PYTHON_BIN}" -m metamon.rl.online_rl \
    --run_config "${REPO_DIR}/metamon/rl/configs/online_runs/taurosv1a.yaml" \
    --mode learn \
    --save_dir "${SAVE_DIR}" \
    --buffer_dir "${BUFFER_DIR}" \
    "${EXTRA_ARGS[@]}" > >(tee "${LEARNER_LOG}") 2>&1 &
LEARNER_PID=$!

wait "${LEARNER_PID}"
