#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORK_DIR="${V1B_WORK_DIR:-${REPO_DIR}/taurosv1b_work}"
PRETRAIN_DIR="${WORK_DIR}/pretrain"
DAGGER1_DIR="${WORK_DIR}/dagger1"

INPUT_WEIGHTS="${INPUT_WEIGHTS:-${PRETRAIN_DIR}/phase_c_from_b1_lr1e6.pt}"
PROBE_EPOCHS="${PROBE_EPOCHS:-5}"
EVAL_BATCHES="${EVAL_BATCHES:-100}"
GAMES="${GAMES:-400}"
OUTPUT_WEIGHTS="${OUTPUT_WEIGHTS:-${PRETRAIN_DIR}/phase_d_probe${PROBE_EPOCHS}.pt}"
OUTPUT_JSON="${OUTPUT_JSON:-${OUTPUT_WEIGHTS}.probe.json}"
TOURNAMENT_JSON="${TOURNAMENT_JSON:-${PRETRAIN_DIR}/phase_d_probe${PROBE_EPOCHS}_vs_taurosv0_${GAMES}.json}"

CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
DEFAULT_WORKERS=$((CPU_COUNT / 2))
if [ "${DEFAULT_WORKERS}" -lt 1 ]; then
  DEFAULT_WORKERS=1
elif [ "${DEFAULT_WORKERS}" -gt 8 ]; then
  DEFAULT_WORKERS=8
fi
PRETRAIN_DLOADER_WORKERS="${PRETRAIN_DLOADER_WORKERS:-${DEFAULT_WORKERS}}"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${HOME}/.cache/metamon}"
export METAMON_SAVE_DIR="${METAMON_SAVE_DIR:-${WORK_DIR}/checkpoints}"
export METAMON_ALLOW_ANY_POKE_ENV=1

if [ ! -s "${INPUT_WEIGHTS}" ]; then
  echo "Missing Phase-C weights: ${INPUT_WEIGHTS}" >&2
  exit 1
fi
if [ ! -d "${DAGGER1_DIR}/gen1ou" ]; then
  echo "Missing DAgger1 replay directory: ${DAGGER1_DIR}/gen1ou" >&2
  exit 1
fi

cd "${REPO_DIR}"

echo "Running ${PROBE_EPOCHS}-epoch Phase-D probe from:"
echo "  ${INPUT_WEIGHTS}"
echo "Held-out evaluation: ${EVAL_BATCHES} batches"
echo

"${PYTHON_BIN}" -m metamon.rl.taurosv1b_d_probe \
  --input_weights "${INPUT_WEIGHTS}" \
  --output_weights "${OUTPUT_WEIGHTS}" \
  --output_json "${OUTPUT_JSON}" \
  --dagger1_dir "${DAGGER1_DIR}" \
  --epochs "${PROBE_EPOCHS}" \
  --eval_batches "${EVAL_BATCHES}" \
  --dloader_workers "${PRETRAIN_DLOADER_WORKERS}"

echo
echo "Running ${GAMES}-game actor tournament vs TaurosV0@62..."
"${PYTHON_BIN}" -m metamon.rl.taurosv1b_actor_tournament \
  --weights "${OUTPUT_WEIGHTS}" \
  --games "${GAMES}" \
  --output "${TOURNAMENT_JSON}"

echo
echo "Probe complete:"
echo "  weights:    ${OUTPUT_WEIGHTS}"
echo "  critic/KD:  ${OUTPUT_JSON}"
echo "  tournament: ${TOURNAMENT_JSON}"
