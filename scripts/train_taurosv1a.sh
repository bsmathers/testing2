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

# If pkmn-showdown.node exists, enable high speed battle host
if [ -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ]; then
    export METAMON_BATTLE_HOST="${REPO_DIR}/metamon/env/vectorized/battle_host_engine.js"
    echo "Using high-speed @pkmn/engine battle host: ${METAMON_BATTLE_HOST}"
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

"${PYTHON_BIN}" -m metamon.rl.online_rl \
    --run_config "${REPO_DIR}/metamon/rl/configs/online_runs/taurosv1a.yaml" \
    --mode "${MODE}" \
    --save_dir "${SAVE_DIR}" \
    --buffer_dir "${BUFFER_DIR}" \
    "$@"
