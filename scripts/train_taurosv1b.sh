#!/usr/bin/env bash
set -euo pipefail

# Safe launcher for the long TaurosV1B online phases.  Unlike scripts/train.sh,
# cleanup is PID-scoped and never pkill's unrelated metamon.rl.online_rl jobs.
#
# Usage:
#   BASE_WEIGHTS=/path/to/v1b_phase_d.pt bash scripts/train_taurosv1b.sh e --log
#   BASE_WEIGHTS=/path/to/best_phase_e.pt bash scripts/train_taurosv1b.sh f --log

PHASE="${1:-e}"
shift || true
case "${PHASE}" in
  e|E)
    CONFIG="metamon/rl/configs/online_runs/taurosv1b.yaml"
    RUN_NAME="taurosv1b_phase_e"
    ;;
  f|F)
    CONFIG="metamon/rl/configs/online_runs/taurosv1b_phase_f.yaml"
    RUN_NAME="taurosv1b_phase_f"
    ;;
  *)
    echo "phase must be 'e' or 'f'" >&2
    exit 2
    ;;
esac

: "${BASE_WEIGHTS:?Set BASE_WEIGHTS to the distilled phase-D checkpoint (E) or selected phase-E checkpoint (F)}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAVE_DIR="${METAMON_SAVE_DIR:-${REPO_DIR}/checkpoints}"
BUFFER_DIR="${BUFFER_DIR:-${REPO_DIR}/buffer_${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/${RUN_NAME}}"
LANES="${LANES:-128}"
DSET_MIN_SIZE="${DSET_MIN_SIZE:-5000}"
TOTAL_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
if [ "${TOTAL_CPUS}" -gt 2 ]; then
  N_WORKERS="${N_WORKERS:-$((TOTAL_CPUS - 2))}"
else
  N_WORKERS="${N_WORKERS:-1}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "${SAVE_DIR}" "${BUFFER_DIR}/gen1ou" "${LOG_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_ALLOW_ANY_POKE_ENV=1

COLLECTOR_PID=""
LEARNER_PID=""
cleanup() {
  set +e
  if [ -n "${COLLECTOR_PID}" ]; then
    kill -TERM "${COLLECTOR_PID}" 2>/dev/null || true
    pkill -TERM -P "${COLLECTOR_PID}" 2>/dev/null || true
  fi
  if [ -n "${LEARNER_PID}" ]; then
    kill -TERM "${LEARNER_PID}" 2>/dev/null || true
    pkill -TERM -P "${LEARNER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${REPO_DIR}"
COLLECTOR_LOG="${LOG_DIR}/collector.log"
LEARNER_LOG="${LOG_DIR}/learner.log"

"${PYTHON_BIN}" -m metamon.rl.taurosv1b_online \
  --run_config "${CONFIG}" \
  --mode collect \
  --save_dir "${SAVE_DIR}" \
  --buffer_dir "${BUFFER_DIR}" \
  --base_weights "${BASE_WEIGHTS}" \
  --lanes "${LANES}" \
  --n_workers "${N_WORKERS}" \
  "$@" >"${COLLECTOR_LOG}" 2>&1 &
COLLECTOR_PID=$!

echo "Collector PID ${COLLECTOR_PID}; prefilling FIFO to $((DSET_MIN_SIZE + 1)) battles..."
while true; do
  if ! kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
    echo "Collector exited during prefill. Last log lines:" >&2
    tail -100 "${COLLECTOR_LOG}" >&2 || true
    exit 1
  fi
  COUNT=$(find "${BUFFER_DIR}/gen1ou" -maxdepth 1 -type f \( -name '*.json' -o -name '*.json.lz4' \) | wc -l | tr -d ' ')
  if [ "${COUNT}" -gt "${DSET_MIN_SIZE}" ]; then
    break
  fi
  printf '\rFIFO prefill: %s / %s' "${COUNT}" "$((DSET_MIN_SIZE + 1))"
  sleep 5
done
printf '\nFIFO ready. Starting learner.\n'

"${PYTHON_BIN}" -m metamon.rl.taurosv1b_online \
  --run_config "${CONFIG}" \
  --mode learn \
  --save_dir "${SAVE_DIR}" \
  --buffer_dir "${BUFFER_DIR}" \
  --base_weights "${BASE_WEIGHTS}" \
  "$@" > >(tee "${LEARNER_LOG}") 2>&1 &
LEARNER_PID=$!

wait "${LEARNER_PID}"
