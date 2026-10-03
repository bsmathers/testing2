#!/usr/bin/env bash
set -eo pipefail

# ==============================================================================
# TaurosV1A Online RL Training (Single-Process Unified Mode)
#
# Exact TaurosV1 architecture (~35M params, grouped_v2_medium.gin)
# Trained from random initialization (--from_scratch) for 1500 epochs.
# Opponent pool: TaurosV1 roster without SmallG1 models.
# Self-play activates at Epoch 300 (min_epoch: 300).
# Mode: "both" (alternating collection, learning, and validation in lockstep).
#
# Usage:
#   bash scripts/train.sh               # Runs in foreground (recommended in tmux)
#   bash scripts/train.sh --log         # With WandB logging enabled
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SAVE_DIR="${REPO_DIR}/checkpoints"
BUFFER_DIR="${REPO_DIR}/buffer"
LOG_DIR="${REPO_DIR}/logs"

mkdir -p "${SAVE_DIR}" "${BUFFER_DIR}" "${LOG_DIR}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${HOME}/.cache/metamon}"

# Ensure node/zig are in PATH if installed locally
[ -d "${REPO_DIR}/.node/bin" ] && export PATH="${REPO_DIR}/.node/bin:${PATH}"
[ -d "${REPO_DIR}/.zig" ] && export PATH="${REPO_DIR}/.zig:${PATH}"

# Check for high-speed native pkmn engine (auto-build if missing)
if [ ! -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ]; then
    echo "[Info] Native Zig simulator not found. Attempting setup..."
    bash "${SCRIPT_DIR}/setup_pkmn_engine.sh" 2>/dev/null || true
fi

if [ -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ]; then
    export METAMON_BATTLE_HOST="${REPO_DIR}/metamon/env/vectorized/battle_host_engine.js"
    echo "[Info] Using high-speed @pkmn/engine: ${METAMON_BATTLE_HOST}"
else
    echo "[Warn] Running with standard Showdown simulator"
fi

# Detect Python binary
PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    PYTHON_BIN="python"
fi

# Clean up any leftover processes from previous runs
echo "[1/3] Terminating any stale processes..."
pkill -9 -f "metamon.rl.online_rl" 2>/dev/null || true
pkill -9 -f "metamon" 2>/dev/null || true
pkill -9 -f "pkmn-showdown" 2>/dev/null || true
sleep 1

# Auto-detect CPU cores for simulation workers
NUM_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)

echo "[2/3] Verified Configuration:"
echo "      Run Name:       taurosv1a"
echo "      Architecture:   TaurosV1 (~35M params, grouped_v2_medium.gin)"
echo "      Init:           Random initialization (--from_scratch)"
echo "      Total Epochs:   1500 (1000 steps/epoch)"
echo "      Self-Play:      Starts at Epoch 300 (min_epoch: 300)"
echo "      Dataset:        pac-tauros (100% self-play offline mix)"
echo "      Mode:           both (unified collection + learning)"
echo "      Lanes:          128 parallel Showdown lanes"
echo "      Workers:        ${NUM_CPUS} CPU cores"
echo "      Save Dir:       ${SAVE_DIR}"
echo "      Buffer Dir:     ${BUFFER_DIR}"
echo ""

# Filter out any bare '--' delimiter so argparse does not error
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--" ]; then
        EXTRA_ARGS+=("$arg")
    fi
done

echo "[3/3] Starting Training..."
cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -m metamon.rl.online_rl \
    --run_config "${REPO_DIR}/metamon/rl/configs/online_runs/taurosv1a.yaml" \
    --mode both \
    --save_dir "${SAVE_DIR}" \
    --buffer_dir "${BUFFER_DIR}" \
    --lanes 128 \
    --n_workers "${NUM_CPUS}" \
    "${EXTRA_ARGS[@]}"
