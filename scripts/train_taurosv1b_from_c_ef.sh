#!/usr/bin/env bash
set -euo pipefail

# TaurosV1B continuation after the completed Phase-C critic warmup:
#   phase_c_from_b1_lr1e6.pt -> E -> F
#
# Phase D is intentionally skipped. E/F both use eta=8e-5. Their training gin
# files import metamon.rl.taurosv1b_tournament, whose checkpoint hook runs a
# 50-game tournament vs TaurosV0@62 every 5 learner epochs and logs
# tournament/v0_62_{win_rate,games,binomial_stderr} to the same W&B run.
#
# Typical use:
#   bash scripts/train_taurosv1b_from_c_ef.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORK_DIR="${V1B_WORK_DIR:-${REPO_DIR}/taurosv1b_work}"
PRETRAIN_DIR="${WORK_DIR}/pretrain"
STAGE_DIR="${WORK_DIR}/stages"
F_ARCHIVE_DIR="${PHASE_F_ARCHIVE_DIR:-${WORK_DIR}/phase_f_replay_archive}"
F_ARCHIVE_MAX="${PHASE_F_ARCHIVE_MAX:-2000000}"

mkdir -p "${WORK_DIR}" "${PRETRAIN_DIR}" "${STAGE_DIR}"

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
export WANDB_TAGS="${WANDB_TAGS:-taurosv1b,distilled-public,c-to-e}"
export V1B_EXPERIMENT_ID="${EXPERIMENT_ID}"

mkdir -p "${METAMON_SAVE_DIR}" "${METAMON_CACHE_DIR}"
ONLINE_SAVE_DIR="${METAMON_SAVE_DIR}/from_c_direct_ef_online"
mkdir -p "${ONLINE_SAVE_DIR}"

PHASE_C="${PRETRAIN_DIR}/phase_c_from_b1_lr1e6.pt"
PHASE_E_FINAL="${PRETRAIN_DIR}/phase_e_from_c_direct_final.pt"
PHASE_F_FINAL="${PRETRAIN_DIR}/phase_f_from_c_direct_final.pt"

stage() {
  printf '\n\n============================================================\n'
  printf ' TaurosV1B: %s\n' "$1"
  printf '============================================================\n'
}

# Build the fast simulator once if needed.
if [ ! -f "${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node" ] \
   && [ -x "${SCRIPT_DIR}/setup_pkmn_engine.sh" ]; then
  stage "native simulator setup"
  bash "${SCRIPT_DIR}/setup_pkmn_engine.sh"
fi

cd "${REPO_DIR}"

if [ ! -s "${PHASE_C}" ]; then
  echo "Missing required Phase-C checkpoint: ${PHASE_C}" >&2
  exit 1
fi

echo "Starting direct C -> E -> F continuation:"
echo "  Phase C: ${PHASE_C}"
echo "  Phase D: SKIPPED"
echo "  E/F LR: 8e-5"
echo "  V0 monitor: 50 games vs TaurosV0@62 every 5 learner epochs"
echo "  W&B metrics: tournament/v0_62_win_rate, tournament/v0_62_games, tournament/v0_62_binomial_stderr"

stage "E — 800 epochs public-opponent online RL, initialized directly from C"
if [ ! -f "${STAGE_DIR}/phase_e_from_c_direct.done" ]; then
  METAMON_SAVE_DIR="${ONLINE_SAVE_DIR}" \
  BASE_WEIGHTS="${PHASE_C}" \
  BUFFER_DIR="${WORK_DIR}/buffer_taurosv1b_phase_e_from_c_direct" \
  WANDB_RUN_ID="v1b-e-from-c-direct-${EXPERIMENT_ID}" \
  WANDB_RESUME=allow \
    bash "${SCRIPT_DIR}/train_taurosv1b.sh" e --log

  E_LATEST="${ONLINE_SAVE_DIR}/taurosv1b_phase_e/ckpts/latest/policy.pt"
  if [ ! -s "${E_LATEST}" ]; then
    echo "Phase E finished without latest policy: ${E_LATEST}" >&2
    exit 1
  fi
  cp -f "${E_LATEST}" "${PHASE_E_FINAL}"
  touch "${STAGE_DIR}/phase_e_from_c_direct.done"
else
  echo "[skip] Phase E marked complete."
fi

stage "F — 800 epochs recency-weighted V1B self-play"
if [ ! -f "${STAGE_DIR}/phase_f_from_c_direct.done" ]; then
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
  BUFFER_DIR="${WORK_DIR}/buffer_taurosv1b_phase_f_from_c_direct" \
  PHASE_F_ARCHIVE_DIR="${F_ARCHIVE_DIR}" \
  PHASE_F_ARCHIVE_MAX="${F_ARCHIVE_MAX}" \
  WANDB_RUN_ID="v1b-f-from-c-direct-${EXPERIMENT_ID}" \
  WANDB_RESUME=allow \
    bash "${SCRIPT_DIR}/train_taurosv1b.sh" f --log

  F_LATEST="${ONLINE_SAVE_DIR}/taurosv1b_phase_f/ckpts/latest/policy.pt"
  if [ ! -s "${F_LATEST}" ]; then
    echo "Phase F finished without latest policy: ${F_LATEST}" >&2
    exit 1
  fi
  cp -f "${F_LATEST}" "${PHASE_F_FINAL}"
  touch "${STAGE_DIR}/phase_f_from_c_direct.done"
else
  echo "[skip] Phase F marked complete."
fi

stage "complete"
echo "Final TaurosV1B policy: ${PHASE_F_FINAL}"
echo "Phase-F replay archive: ${F_ARCHIVE_DIR}/gen1ou (cap ${F_ARCHIVE_MAX})"
echo "W&B project:       ${METAMON_WANDB_PROJECT}"
echo "W&B group:         ${WANDB_RUN_GROUP}"
echo "V1B experiment ID: ${EXPERIMENT_ID}"
