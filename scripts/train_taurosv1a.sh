#!/usr/bin/env bash
set -e

# Launch script for TaurosV1A Online RL Training (Single-GPU or Multi-GPU)
# Usage:
#   bash scripts/train_taurosv1a.sh [mode] [save_dir] [buffer_dir]
#
# Examples:
#   # Single process smoke test:
#   bash scripts/train_taurosv1a.sh both ./checkpoints ./buffer
#
#   # Or run collector / learner / validator separately:
#   bash scripts/train_taurosv1a.sh learn ./checkpoints ./buffer
#   bash scripts/train_taurosv1a.sh collect ./checkpoints ./buffer
#   bash scripts/train_taurosv1a.sh validate ./checkpoints ./buffer

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Ensure metamon and root directory are on PYTHONPATH
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

# Ensure local node is in PATH if installed standalone
if [ -d "${REPO_DIR}/.node/bin" ]; then
    export PATH="${REPO_DIR}/.node/bin:${PATH}"
fi

# Ensure local zig is in PATH if installed standalone
if [ -d "${REPO_DIR}/.zig" ]; then
    export PATH="${REPO_DIR}/.zig:${PATH}"
fi

cd "${REPO_DIR}"

MODE="${1:-both}"
SAVE_DIR="${2:-${REPO_DIR}/checkpoints}"
BUFFER_DIR="${3:-${REPO_DIR}/buffer}"

# Shift off the first 3 positional args if provided, leaving any extra flags (e.g. --log)
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift
[ $# -ge 1 ] && shift

mkdir -p "${SAVE_DIR}" "${BUFFER_DIR}"

export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${HOME}/.cache/metamon}"
mkdir -p "${METAMON_CACHE_DIR}"

# Check if pkmn-showdown.node exists; if not, build it automatically
if [ ! -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ]; then
    echo "Native Zig simulator (pkmn-showdown.node) not found."
    echo "Running automatic setup via scripts/setup_pkmn_engine.sh..."
    bash "${SCRIPT_DIR}/setup_pkmn_engine.sh" || true
fi

if [ -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ]; then
    export METAMON_BATTLE_HOST="${REPO_DIR}/metamon/env/vectorized/battle_host_engine.js"
    echo "Using high-speed @pkmn/engine battle host: ${METAMON_BATTLE_HOST}"
else
    echo "Warning: pkmn-showdown.node not found; falling back to standard battle_host.js"
    if [ ! -d "${REPO_DIR}/metamon/env/vectorized/node_modules/pokemon-showdown" ]; then
        echo "Installing node dependencies in metamon/env/vectorized..."
        (cd "${REPO_DIR}/metamon/env/vectorized" && npm install)
    fi
fi

# Detect python binary
PYTHON_BIN="${PYTHON:-python3}"
if ! command -v "${PYTHON_BIN}" &> /dev/null; then
    PYTHON_BIN="python"
fi

echo "=== Starting TaurosV1A Online RL ==="
echo "Repo dir:   ${REPO_DIR}"
echo "Python:     $(which ${PYTHON_BIN})"
echo "Mode:       ${MODE}"
echo "Save dir:   ${SAVE_DIR}"
echo "Buffer dir: ${BUFFER_DIR}"
echo "Cache dir:  ${METAMON_CACHE_DIR}"

# Filter out any bare '--' delimiter so argparse does not error
EXTRA_ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--" ]; then
        EXTRA_ARGS+=("$arg")
    fi
done

"${PYTHON_BIN}" -m metamon.rl.online_rl \
    --run_config "${REPO_DIR}/metamon/rl/configs/online_runs/taurosv1a.yaml" \
    --mode "${MODE}" \
    --save_dir "${SAVE_DIR}" \
    --buffer_dir "${BUFFER_DIR}" \
    "${EXTRA_ARGS[@]}"
