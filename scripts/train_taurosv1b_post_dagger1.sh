#!/usr/bin/env bash
set -euo pipefail

# TaurosV1B continuation pipeline after Phase A300 + completed DAgger1:
#   phase_a_300.pt + dagger1 cache -> B1 -> DAgger2 -> B2 -> C -> D -> E -> F
#
# This script deliberately skips all Phase-A work, the actor tournament, and
# DAgger1 collection. It requires the A300 checkpoint and a completed DAgger1
# cache to already exist.
#
# Every optimization stage in this repository is fixed at eta=1e-5 with no LR
# warmup. See taurosv1b_pretrain.py and the E/F online-run configs.
#
# Typical use:
#   bash scripts/train_taurosv1b_post_dagger1.sh
#
# Useful overrides:
#   V1B_WORK_DIR=/fast/nvme/v1b
#   METAMON_SAVE_DIR=/fast/nvme/v1b/checkpoints
#   METAMON_CACHE_DIR=/fast/nvme/metamon_cache
#   METAMON_WANDB_PROJECT=taurosv1b
#   WANDB_MODE=online
#   DAGGER_GAMES=75000
#   DAGGER_SHARDS=4
#   DAGGER_LANES_PER_SHARD=16
#   DAGGER_WORKERS_PER_SHARD=1
#   PRETRAIN_BATCH_SIZE=8
#   PRETRAIN_DLOADER_WORKERS=8
#   PHASE_F_ARCHIVE_DIR=/fast/nvme/v1b/phase_f_replay_archive
#   PHASE_F_ARCHIVE_MAX=2000000

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORK_DIR="${V1B_WORK_DIR:-${REPO_DIR}/taurosv1b_work}"
PRETRAIN_DIR="${WORK_DIR}/pretrain"
DAGGER1_DIR="${WORK_DIR}/dagger1"
DAGGER2_DIR="${WORK_DIR}/dagger2"
STAGE_DIR="${WORK_DIR}/stages"
DAGGER_GAMES="${DAGGER_GAMES:-75000}"
F_ARCHIVE_DIR="${PHASE_F_ARCHIVE_DIR:-${WORK_DIR}/phase_f_replay_archive}"
F_ARCHIVE_MAX="${PHASE_F_ARCHIVE_MAX:-2000000}"

CPU_COUNT=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
DEFAULT_PRETRAIN_WORKERS=$((CPU_COUNT / 2))
if [ "${DEFAULT_PRETRAIN_WORKERS}" -lt 1 ]; then
  DEFAULT_PRETRAIN_WORKERS=1
elif [ "${DEFAULT_PRETRAIN_WORKERS}" -gt 8 ]; then
  DEFAULT_PRETRAIN_WORKERS=8
fi
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-8}"
PRETRAIN_DLOADER_WORKERS="${PRETRAIN_DLOADER_WORKERS:-${DEFAULT_PRETRAIN_WORKERS}}"
PRETRAIN_ARGS=(
  --batch_size "${PRETRAIN_BATCH_SIZE}"
  --dloader_workers "${PRETRAIN_DLOADER_WORKERS}"
)

DEFAULT_DAGGER_SHARDS=$((CPU_COUNT < 4 ? CPU_COUNT : 4))
if [ "${DEFAULT_DAGGER_SHARDS}" -lt 1 ]; then
  DEFAULT_DAGGER_SHARDS=1
fi
DAGGER_SHARDS="${DAGGER_SHARDS:-${DEFAULT_DAGGER_SHARDS}}"
DAGGER_LANES_PER_SHARD="${DAGGER_LANES_PER_SHARD:-16}"
DAGGER_WORKERS_PER_SHARD="${DAGGER_WORKERS_PER_SHARD:-1}"
DAGGER_ARGS=(
  --shards "${DAGGER_SHARDS}"
  --lanes_per_shard "${DAGGER_LANES_PER_SHARD}"
  --workers_per_shard "${DAGGER_WORKERS_PER_SHARD}"
)

mkdir -p \
  "${WORK_DIR}" \
  "${PRETRAIN_DIR}" \
  "${STAGE_DIR}"

EXPERIMENT_ID_FILE="${WORK_DIR}/experiment_id.txt"
if [ -n "${V1B_EXPERIMENT_ID:-}" ]; then
  EXPERIMENT_ID="${V1B_EXPERIMENT_ID}"
elif [ -s "${EXPERIMENT_ID_FILE}" ]; then
  EXPERIMENT_ID="$(tr -d '[:space:]' < "${EXPERIMENT_ID_FILE}")"
else
  EXPERIMENT_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  printf '%s\n' "${EXPERIMENT_ID}" > "${EXPERIMENT_ID_FILE}"
fi

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export METAMON_SAVE_DIR="${METAMON_SAVE_DIR:-${WORK_DIR}/checkpoints}"
export METAMON_CACHE_DIR="${METAMON_CACHE_DIR:-${HOME}/.cache/metamon}"
export METAMON_ALLOW_ANY_POKE_ENV=1
export METAMON_WANDB_PROJECT="${METAMON_WANDB_PROJECT:-taurosv1b}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-taurosv1b-${EXPERIMENT_ID}}"
export WANDB_TAGS="${WANDB_TAGS:-taurosv1b,distilled-public}"
export V1B_EXPERIMENT_ID="${EXPERIMENT_ID}"

mkdir -p "${METAMON_SAVE_DIR}" "${METAMON_CACHE_DIR}"
ONLINE_SAVE_DIR="${METAMON_SAVE_DIR}/lr1e5_online"
mkdir -p "${ONLINE_SAVE_DIR}"

PHASE_A="${PRETRAIN_DIR}/phase_a_300.pt"
PHASE_B1="${PRETRAIN_DIR}/phase_b1_lr1e5.pt"
PHASE_B2="${PRETRAIN_DIR}/phase_b2_lr1e5.pt"
PHASE_C="${PRETRAIN_DIR}/phase_c_lr1e5.pt"
PHASE_D="${PRETRAIN_DIR}/phase_d_lr1e5.pt"
PHASE_E_FINAL="${PRETRAIN_DIR}/phase_e_lr1e5_final.pt"
PHASE_F_FINAL="${PRETRAIN_DIR}/phase_f_lr1e5_final.pt"

stage() {
  printf '\n\n============================================================\n'
  printf ' TaurosV1B: %s\n' "$1"
  printf '============================================================\n'
}

run_if_missing() {
  local output="$1"
  shift
  if [ -s "${output}" ]; then
    echo "[skip] Existing output: ${output}"
    return 0
  fi
  "$@"
  if [ ! -s "${output}" ]; then
    echo "Expected stage output was not created: ${output}" >&2
    exit 1
  fi
}


count_replays() {
  local root="$1"
  find "${root}/gen1ou" -maxdepth 1 -type f \
    \( -name '*.json' -o -name '*.json.lz4' \) 2>/dev/null | wc -l | tr -d ' '
}

# Build the fast simulator once if this checkout has not already done so.
if [ ! -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ] \
   && [ -x "${SCRIPT_DIR}/setup_pkmn_engine.sh" ]; then
  stage "native simulator setup"
  bash "${SCRIPT_DIR}/setup_pkmn_engine.sh"
fi

cd "${REPO_DIR}"
echo "A-D pretraining loader: batch=${PRETRAIN_BATCH_SIZE}, workers=${PRETRAIN_DLOADER_WORKERS}"
echo "DAgger parallelism: shards=${DAGGER_SHARDS}, lanes/shard=${DAGGER_LANES_PER_SHARD}, workers/shard=${DAGGER_WORKERS_PER_SHARD}"

if [ ! -s "${PHASE_A}" ]; then
  echo "Missing required 300-epoch Phase-A checkpoint: ${PHASE_A}" >&2
  echo "This continuation script intentionally does not retrain Phase A." >&2
  exit 1
fi

D1_COUNT="$(count_replays "${DAGGER1_DIR}")"
if [ "${D1_COUNT}" -lt "${DAGGER_GAMES}" ]; then
  echo "Completed DAgger1 cache required: found ${D1_COUNT}/${DAGGER_GAMES} in ${DAGGER1_DIR}/gen1ou" >&2
  exit 1
fi

echo "Starting downstream optimization from A300 + completed DAgger1:"
echo "  actor:   ${PHASE_A}"
echo "  DAgger1: ${D1_COUNT} replays"
echo "  LR rule: fixed eta=1e-5, no warmup"

stage "B1 — 50 epochs public + DAgger-1 distillation"
run_if_missing "${PHASE_B1}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase b1 \
    "${PRETRAIN_ARGS[@]}" \
    --input_weights "${PHASE_A}" \
    --dagger1_dir "${DAGGER1_DIR}" \
    --output_weights "${PHASE_B1}"

stage "DAgger round 2 — ${DAGGER_GAMES} student-occupancy battles (multicore)"
"${PYTHON_BIN}" -m metamon.rl.taurosv1b_collect_dagger_parallel \
  --weights "${PHASE_B1}" \
  --output_dir "${DAGGER2_DIR}" \
  --target_games "${DAGGER_GAMES}" \
  "${DAGGER_ARGS[@]}"

stage "B2 — 50 epochs public + both DAgger piles"
run_if_missing "${PHASE_B2}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase b2 \
    "${PRETRAIN_ARGS[@]}" \
    --input_weights "${PHASE_B1}" \
    --dagger1_dir "${DAGGER1_DIR}" \
    --dagger2_dir "${DAGGER2_DIR}" \
    --output_weights "${PHASE_B2}"

stage "C — 50 epochs critic-only warmup"
run_if_missing "${PHASE_C}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase c \
    "${PRETRAIN_ARGS[@]}" \
    --input_weights "${PHASE_B2}" \
    --dagger1_dir "${DAGGER1_DIR}" \
    --dagger2_dir "${DAGGER2_DIR}" \
    --output_weights "${PHASE_C}"

stage "D — 25 epochs shared-representation critic/KL bridge"
run_if_missing "${PHASE_D}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase d \
    "${PRETRAIN_ARGS[@]}" \
    --input_weights "${PHASE_C}" \
    --dagger1_dir "${DAGGER1_DIR}" \
    --dagger2_dir "${DAGGER2_DIR}" \
    --output_weights "${PHASE_D}"

stage "E — 800 epochs public-opponent online RL"
if [ ! -f "${STAGE_DIR}/phase_e_lr1e5.done" ]; then
  METAMON_SAVE_DIR="${ONLINE_SAVE_DIR}" \
  BASE_WEIGHTS="${PHASE_D}" \
  BUFFER_DIR="${WORK_DIR}/buffer_taurosv1b_phase_e_lr1e5" \
  WANDB_RUN_ID="v1b-e-lr1e5-${EXPERIMENT_ID}" \
  WANDB_RESUME=allow \
    bash "${SCRIPT_DIR}/train_taurosv1b.sh" e --log

  E_LATEST="${ONLINE_SAVE_DIR}/taurosv1b_phase_e/ckpts/latest/policy.pt"
  if [ ! -s "${E_LATEST}" ]; then
    echo "Phase E finished without latest policy: ${E_LATEST}" >&2
    exit 1
  fi
  cp -f "${E_LATEST}" "${PHASE_E_FINAL}"
  touch "${STAGE_DIR}/phase_e_lr1e5.done"
else
  echo "[skip] Phase E marked complete."
fi

stage "F — 800 epochs recency-weighted V1B self-play"
if [ ! -f "${STAGE_DIR}/phase_f_lr1e5.done" ]; then
  if [ ! -s "${PHASE_E_FINAL}" ]; then
    E_LATEST="${ONLINE_SAVE_DIR}/taurosv1b_phase_e/ckpts/latest/policy.pt"
    if [ ! -s "${E_LATEST}" ]; then
      echo "Missing Phase-E final policy." >&2
      exit 1
    fi
    cp -f "${E_LATEST}" "${PHASE_E_FINAL}"
  fi

  METAMON_SAVE_DIR="${ONLINE_SAVE_DIR}" \
  BASE_WEIGHTS="${PHASE_E_FINAL}" \
  BUFFER_DIR="${WORK_DIR}/buffer_taurosv1b_phase_f_lr1e5" \
  PHASE_F_ARCHIVE_DIR="${F_ARCHIVE_DIR}" \
  PHASE_F_ARCHIVE_MAX="${F_ARCHIVE_MAX}" \
  WANDB_RUN_ID="v1b-f-lr1e5-${EXPERIMENT_ID}" \
  WANDB_RESUME=allow \
    bash "${SCRIPT_DIR}/train_taurosv1b.sh" f --log

  F_LATEST="${ONLINE_SAVE_DIR}/taurosv1b_phase_f/ckpts/latest/policy.pt"
  if [ ! -s "${F_LATEST}" ]; then
    echo "Phase F finished without latest policy: ${F_LATEST}" >&2
    exit 1
  fi
  cp -f "${F_LATEST}" "${PHASE_F_FINAL}"
  touch "${STAGE_DIR}/phase_f_lr1e5.done"
else
  echo "[skip] Phase F marked complete."
fi

stage "complete"
echo "Final TaurosV1B policy: ${PHASE_F_FINAL}"
echo "Phase-F replay archive: ${F_ARCHIVE_DIR}/gen1ou (cap ${F_ARCHIVE_MAX})"
echo "W&B project:       ${METAMON_WANDB_PROJECT}"
echo "W&B group:         ${WANDB_RUN_GROUP}"
echo "V1B experiment ID: ${EXPERIMENT_ID}"
