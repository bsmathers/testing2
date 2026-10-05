#!/usr/bin/env bash
set -euo pipefail

# End-to-end TaurosV1B pipeline:
#   A -> DAgger1 -> B1 -> DAgger2 -> B2 -> C -> D -> E -> F
#
# The script is resumable. Completed pretraining checkpoints are skipped,
# DAgger collectors continue toward their target battle count, and the E/F
# launcher resumes the newest full optimizer state with a policy-consistent
# collector.
#
# W&B:
#   - A-D run through taurosv1b_pretrain_wandb and log per-epoch training metrics.
#     Since A-D restart from the beginning of a phase after interruption, a retry
#     intentionally creates a new W&B attempt under the same experiment group.
#   - E/F launch with --log and keep all standard AMAGO metrics.
#   - Every 5 epochs (after epoch 0), E/F additionally run 50 games against
#     TaurosV0@62 and log tournament/v0_62_win_rate to the same learner W&B run.
#   - A persistent local experiment ID gives E and F stable W&B run IDs, so a
#     crashed online learner resumes the same W&B run rather than creating a duplicate.
#   - Set METAMON_WANDB_PROJECT / METAMON_WANDB_ENTITY / WANDB_MODE as desired.
#
# Phase-F replay retention:
#   - training FIFO stays at 150k (the recipe is unchanged);
#   - a separate archive retains up to 2M completed phase-F replays;
#   - hard links are used when possible, so archived files consume new blocks only
#     after the FIFO evicts its pathname.
#
# Typical use:
#   bash scripts/train_taurosv1b_all.sh
#
# Useful overrides:
#   V1B_WORK_DIR=/fast/nvme/v1b
#   METAMON_SAVE_DIR=/fast/nvme/v1b/checkpoints
#   METAMON_CACHE_DIR=/fast/nvme/metamon_cache
#   METAMON_WANDB_PROJECT=taurosv1b
#   WANDB_MODE=online        # or offline / disabled
#   DAGGER_GAMES=75000
#   LANES=128
#   N_WORKERS=30
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

mkdir -p \
  "${WORK_DIR}" \
  "${PRETRAIN_DIR}" \
  "${DAGGER1_DIR}/gen1ou" \
  "${DAGGER2_DIR}/gen1ou" \
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

PHASE_A="${PRETRAIN_DIR}/phase_a.pt"
PHASE_B1="${PRETRAIN_DIR}/phase_b1.pt"
PHASE_B2="${PRETRAIN_DIR}/phase_b2.pt"
PHASE_C="${PRETRAIN_DIR}/phase_c.pt"
PHASE_D="${PRETRAIN_DIR}/phase_d.pt"
PHASE_E_FINAL="${PRETRAIN_DIR}/phase_e_final.pt"
PHASE_F_FINAL="${PRETRAIN_DIR}/phase_f_final.pt"

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

stage "A — 150 epochs V0 policy distillation"
run_if_missing "${PHASE_A}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase a \
    --output_weights "${PHASE_A}"

stage "DAgger round 1 — ${DAGGER_GAMES} student-occupancy battles"
D1_COUNT=$(count_replays "${DAGGER1_DIR}")
if [ "${D1_COUNT}" -lt "${DAGGER_GAMES}" ]; then
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_collect_dagger \
    --weights "${PHASE_A}" \
    --output_dir "${DAGGER1_DIR}" \
    --target_games "${DAGGER_GAMES}"
else
  echo "[skip] DAgger-1 already has ${D1_COUNT} battles."
fi

stage "B1 — 50 epochs public + DAgger-1 distillation"
run_if_missing "${PHASE_B1}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase b1 \
    --input_weights "${PHASE_A}" \
    --dagger1_dir "${DAGGER1_DIR}" \
    --output_weights "${PHASE_B1}"

stage "DAgger round 2 — ${DAGGER_GAMES} student-occupancy battles"
D2_COUNT=$(count_replays "${DAGGER2_DIR}")
if [ "${D2_COUNT}" -lt "${DAGGER_GAMES}" ]; then
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_collect_dagger \
    --weights "${PHASE_B1}" \
    --output_dir "${DAGGER2_DIR}" \
    --target_games "${DAGGER_GAMES}"
else
  echo "[skip] DAgger-2 already has ${D2_COUNT} battles."
fi

stage "B2 — 50 epochs public + both DAgger piles"
run_if_missing "${PHASE_B2}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase b2 \
    --input_weights "${PHASE_B1}" \
    --dagger1_dir "${DAGGER1_DIR}" \
    --dagger2_dir "${DAGGER2_DIR}" \
    --output_weights "${PHASE_B2}"

stage "C — 50 epochs critic-only warmup"
run_if_missing "${PHASE_C}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase c \
    --input_weights "${PHASE_B2}" \
    --dagger1_dir "${DAGGER1_DIR}" \
    --dagger2_dir "${DAGGER2_DIR}" \
    --output_weights "${PHASE_C}"

stage "D — 25 epochs shared-representation critic/KL bridge"
run_if_missing "${PHASE_D}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain_wandb \
    --phase d \
    --input_weights "${PHASE_C}" \
    --dagger1_dir "${DAGGER1_DIR}" \
    --dagger2_dir "${DAGGER2_DIR}" \
    --output_weights "${PHASE_D}"

stage "E — 800 epochs public-opponent online RL"
if [ ! -f "${STAGE_DIR}/phase_e.done" ]; then
  BASE_WEIGHTS="${PHASE_D}" \
  BUFFER_DIR="${WORK_DIR}/buffer_taurosv1b_phase_e" \
  WANDB_RUN_ID="v1b-e-${EXPERIMENT_ID}" \
  WANDB_RESUME=allow \
    bash "${SCRIPT_DIR}/train_taurosv1b.sh" e --log

  E_LATEST="${METAMON_SAVE_DIR}/taurosv1b_phase_e/ckpts/latest/policy.pt"
  if [ ! -s "${E_LATEST}" ]; then
    echo "Phase E finished without latest policy: ${E_LATEST}" >&2
    exit 1
  fi
  cp -f "${E_LATEST}" "${PHASE_E_FINAL}"
  touch "${STAGE_DIR}/phase_e.done"
else
  echo "[skip] Phase E marked complete."
fi

stage "F — 800 epochs recency-weighted V1B self-play"
if [ ! -f "${STAGE_DIR}/phase_f.done" ]; then
  if [ ! -s "${PHASE_E_FINAL}" ]; then
    E_LATEST="${METAMON_SAVE_DIR}/taurosv1b_phase_e/ckpts/latest/policy.pt"
    if [ ! -s "${E_LATEST}" ]; then
      echo "Missing Phase-E final policy." >&2
      exit 1
    fi
    cp -f "${E_LATEST}" "${PHASE_E_FINAL}"
  fi

  BASE_WEIGHTS="${PHASE_E_FINAL}" \
  BUFFER_DIR="${WORK_DIR}/buffer_taurosv1b_phase_f" \
  PHASE_F_ARCHIVE_DIR="${F_ARCHIVE_DIR}" \
  PHASE_F_ARCHIVE_MAX="${F_ARCHIVE_MAX}" \
  WANDB_RUN_ID="v1b-f-${EXPERIMENT_ID}" \
  WANDB_RESUME=allow \
    bash "${SCRIPT_DIR}/train_taurosv1b.sh" f --log

  F_LATEST="${METAMON_SAVE_DIR}/taurosv1b_phase_f/ckpts/latest/policy.pt"
  if [ ! -s "${F_LATEST}" ]; then
    echo "Phase F finished without latest policy: ${F_LATEST}" >&2
    exit 1
  fi
  cp -f "${F_LATEST}" "${PHASE_F_FINAL}"
  touch "${STAGE_DIR}/phase_f.done"
else
  echo "[skip] Phase F marked complete."
fi

stage "complete"
echo "Final TaurosV1B policy: ${PHASE_F_FINAL}"
echo "Phase-F replay archive: ${F_ARCHIVE_DIR}/gen1ou (cap ${F_ARCHIVE_MAX})"
echo "W&B project:       ${METAMON_WANDB_PROJECT}"
echo "W&B group:         ${WANDB_RUN_GROUP}"
echo "V1B experiment ID: ${EXPERIMENT_ID}"
