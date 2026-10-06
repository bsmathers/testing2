#!/usr/bin/env bash
set -euo pipefail

# Safe launcher for the long TaurosV1B online phases.
#
# Design invariant:
#   * learner owns CUDA
#   * collectors never see CUDA
#   * replay collection is sharded across CPU-only collector processes
#
# Initial run:
#   BASE_WEIGHTS=/path/to/v1b_phase_d.pt bash scripts/train_taurosv1b.sh e --log
#   BASE_WEIGHTS=/path/to/phase_e_final.pt bash scripts/train_taurosv1b.sh f --log
#
# Relaunch after interruption:
#   bash scripts/train_taurosv1b.sh e --log

PHASE="${1:-e}"
shift || true
IS_PHASE_F=0
case "${PHASE}" in
  e|E)
    CONFIG="metamon/rl/configs/online_runs/taurosv1b.yaml"
    RUN_NAME="taurosv1b_phase_e"
    ;;
  f|F)
    CONFIG="metamon/rl/configs/online_runs/taurosv1b_phase_f.yaml"
    RUN_NAME="taurosv1b_phase_f"
    IS_PHASE_F=1
    ;;
  *)
    echo "phase must be 'e' or 'f'" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SAVE_DIR="${METAMON_SAVE_DIR:-${REPO_DIR}/checkpoints}"
BUFFER_DIR="${BUFFER_DIR:-${REPO_DIR}/buffer_${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/logs/${RUN_NAME}}"
LANES="${LANES:-128}"
DSET_MIN_SIZE="${DSET_MIN_SIZE:-5000}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# CPU-only replay sharding.  The target host has 24 CPUs; reserve 4 for the
# learner/system and use 20 for collection by default.
TOTAL_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 24)
COLLECTOR_CPUS="${COLLECTOR_CPUS:-20}"
if [ "${COLLECTOR_CPUS}" -gt "${TOTAL_CPUS}" ]; then
  COLLECTOR_CPUS="${TOTAL_CPUS}"
fi
COLLECTOR_SHARDS="${COLLECTOR_SHARDS:-5}"
if [ "${COLLECTOR_SHARDS}" -lt 1 ]; then
  echo "COLLECTOR_SHARDS must be >= 1" >&2
  exit 2
fi
COLLECTOR_THREADS_PER_SHARD="${COLLECTOR_THREADS_PER_SHARD:-$(( (COLLECTOR_CPUS + COLLECTOR_SHARDS - 1) / COLLECTOR_SHARDS ))}"
COLLECTOR_LANES_PER_SHARD="${COLLECTOR_LANES_PER_SHARD:-$(( (LANES + COLLECTOR_SHARDS - 1) / COLLECTOR_SHARDS ))}"
COLLECTOR_N_WORKERS_PER_SHARD="${COLLECTOR_N_WORKERS_PER_SHARD:-1}"
COLLECTOR_SEED_BASE="${COLLECTOR_SEED_BASE:-100000}"

PHASE_F_ARCHIVE_DIR="${PHASE_F_ARCHIVE_DIR:-${REPO_DIR}/taurosv1b_phase_f_replay_archive}"
PHASE_F_ARCHIVE_MAX="${PHASE_F_ARCHIVE_MAX:-2000000}"
PHASE_F_ARCHIVE_POLL_SECONDS="${PHASE_F_ARCHIVE_POLL_SECONDS:-60}"

mkdir -p "${SAVE_DIR}" "${BUFFER_DIR}/gen1ou" "${LOG_DIR}"
if [ "${IS_PHASE_F}" -eq 1 ]; then
  mkdir -p "${PHASE_F_ARCHIVE_DIR}/gen1ou"
fi
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_ALLOW_ANY_POKE_ENV=1

COLLECTOR_PIDS=()
LEARNER_PID=""
ARCHIVER_PID=""

process_state() {
  local pid="$1"
  ps -o stat= -p "${pid}" 2>/dev/null | awk '{print $1}' || true
}

stop_collectors() {
  set +e
  local pid
  for pid in "${COLLECTOR_PIDS[@]:-}"; do
    [ -n "${pid}" ] || continue
    kill -TERM "${pid}" 2>/dev/null || true
    pkill -TERM -P "${pid}" 2>/dev/null || true
  done
  for pid in "${COLLECTOR_PIDS[@]:-}"; do
    [ -n "${pid}" ] || continue
    wait "${pid}" 2>/dev/null || true
  done
  COLLECTOR_PIDS=()
  set -e
}

cleanup() {
  local status=$?
  set +e
  stop_collectors
  if [ -n "${LEARNER_PID}" ]; then
    kill -TERM "${LEARNER_PID}" 2>/dev/null || true
    pkill -TERM -P "${LEARNER_PID}" 2>/dev/null || true
    wait "${LEARNER_PID}" 2>/dev/null || true
  fi
  if [ -n "${ARCHIVER_PID}" ]; then
    kill -TERM "${ARCHIVER_PID}" 2>/dev/null || true
    wait "${ARCHIVER_PID}" 2>/dev/null || true
  fi
  return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Only the learner owns the E/F W&B run. Other user CLI overrides are learner-only.
LEARNER_EXTRA=("$@")

RUN_DIR="${SAVE_DIR}/${RUN_NAME}/ckpts"
STATE_DIR="${RUN_DIR}/training_states"
POLICY_DIR="${RUN_DIR}/policy_weights"
LATEST_DIR="${RUN_DIR}/latest"
RESUME_EPOCH=""
if [ -d "${STATE_DIR}" ]; then
  RESUME_EPOCH=$(
    for state_path in "${STATE_DIR}/${RUN_NAME}_epoch_"*; do
      [ -d "${state_path}" ] || continue
      state_name="${state_path##*/}"
      epoch="${state_name##*_epoch_}"
      if [[ "${epoch}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "${epoch}"
      fi
    done | sort -n | tail -1
  )
fi

COLLECTOR_INIT=()
LEARNER_INIT=()
mkdir -p "${LATEST_DIR}"
if [ -n "${RESUME_EPOCH}" ]; then
  RESUME_POLICY="${POLICY_DIR}/policy_epoch_${RESUME_EPOCH}.pt"
  if [ ! -s "${RESUME_POLICY}" ]; then
    echo "Found full state epoch ${RESUME_EPOCH}, but matching raw policy is missing/empty: ${RESUME_POLICY}" >&2
    exit 1
  fi

  if [ -d "${POLICY_DIR}" ]; then
    shopt -s nullglob
    for policy_path in "${POLICY_DIR}"/policy_epoch_*.pt; do
      base="$(basename "${policy_path}")"
      policy_epoch="${base#policy_epoch_}"
      policy_epoch="${policy_epoch%.pt}"
      if [[ "${policy_epoch}" =~ ^[0-9]+$ ]] && [ "${policy_epoch}" -gt "${RESUME_EPOCH}" ]; then
        rm -f "${policy_path}"
      fi
    done
    shopt -u nullglob
  fi

  cp -f "${RESUME_POLICY}" "${LATEST_DIR}/policy.pt"
  COLLECTOR_INIT=(--base_weights "${RESUME_POLICY}")
  LEARNER_INIT=(--resume_training_state --resume_epoch "${RESUME_EPOCH}")
  echo "Resuming ${RUN_NAME} from full state epoch ${RESUME_EPOCH}."
else
  : "${BASE_WEIGHTS:?No resumable state found. Set BASE_WEIGHTS for the initial phase launch.}"
  if [ ! -s "${BASE_WEIGHTS}" ]; then
    echo "Initial BASE_WEIGHTS is missing or empty: ${BASE_WEIGHTS}" >&2
    exit 1
  fi
  cp -f "${BASE_WEIGHTS}" "${LATEST_DIR}/policy.pt"
  COLLECTOR_INIT=(--base_weights "${BASE_WEIGHTS}")
  LEARNER_INIT=(--base_weights "${BASE_WEIGHTS}")
  echo "Starting ${RUN_NAME} from ${BASE_WEIGHTS}."
fi

cd "${REPO_DIR}"
LEARNER_LOG="${LOG_DIR}/learner.log"
ARCHIVER_LOG="${LOG_DIR}/phase_f_replay_archive.log"

collector_shard_loop() {
  local shard="$1"
  local seed=$((COLLECTOR_SEED_BASE + shard))
  local shard_log="${LOG_DIR}/collector_shard_${shard}.log"

  while true; do
    # CUDA is hidden at process creation, so neither the rollout policy nor
    # sampled opponent models can ever allocate GPU memory.
    CUDA_VISIBLE_DEVICES="" \
    METAMON_CPU_COLLECTOR=1 \
    OMP_NUM_THREADS="${COLLECTOR_THREADS_PER_SHARD}" \
    MKL_NUM_THREADS="${COLLECTOR_THREADS_PER_SHARD}" \
    OPENBLAS_NUM_THREADS="${COLLECTOR_THREADS_PER_SHARD}" \
    NUMEXPR_NUM_THREADS="${COLLECTOR_THREADS_PER_SHARD}" \
      "${PYTHON_BIN}" -m metamon.rl.taurosv1b_online \
        --run_config "${CONFIG}" \
        --mode collect \
        --save_dir "${SAVE_DIR}" \
        --buffer_dir "${BUFFER_DIR}" \
        "${COLLECTOR_INIT[@]}" \
        --lanes "${COLLECTOR_LANES_PER_SHARD}" \
        --n_workers "${COLLECTOR_N_WORKERS_PER_SHARD}" \
        --seed "${seed}" \
        --epochs 1 >>"${shard_log}" 2>&1 || return $?
  done
}

start_collectors() {
  COLLECTOR_PIDS=()
  local shard
  for ((shard=0; shard<COLLECTOR_SHARDS; shard++)); do
    collector_shard_loop "${shard}" &
    COLLECTOR_PIDS+=("$!")
  done
  echo "Started ${COLLECTOR_SHARDS} CPU-only collector shards: ${COLLECTOR_PIDS[*]}"
  echo "  collector_cpus=${COLLECTOR_CPUS}/${TOTAL_CPUS}, threads/shard=${COLLECTOR_THREADS_PER_SHARD}, lanes/shard=${COLLECTOR_LANES_PER_SHARD}, workers/shard=${COLLECTOR_N_WORKERS_PER_SHARD}"
}

check_collectors() {
  local i pid state
  for i in "${!COLLECTOR_PIDS[@]}"; do
    pid="${COLLECTOR_PIDS[$i]}"
    state="$(process_state "${pid}")"
    if [ -z "${state}" ] || [[ "${state}" == Z* ]]; then
      echo "Collector shard ${i} exited unexpectedly. Last log lines:" >&2
      tail -100 "${LOG_DIR}/collector_shard_${i}.log" >&2 || true
      return 1
    fi
  done
  return 0
}

start_collectors

if [ "${IS_PHASE_F}" -eq 1 ]; then
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_replay_archive \
    --source_dir "${BUFFER_DIR}/gen1ou" \
    --archive_dir "${PHASE_F_ARCHIVE_DIR}/gen1ou" \
    --max_files "${PHASE_F_ARCHIVE_MAX}" \
    --poll_seconds "${PHASE_F_ARCHIVE_POLL_SECONDS}" \
    >"${ARCHIVER_LOG}" 2>&1 &
  ARCHIVER_PID=$!
  echo "Phase-F replay archiver PID ${ARCHIVER_PID}; cap=${PHASE_F_ARCHIVE_MAX}, dir=${PHASE_F_ARCHIVE_DIR}/gen1ou"
fi

echo "Ensuring FIFO has $((DSET_MIN_SIZE + 1)) battles..."
while true; do
  if ! check_collectors; then
    exit 1
  fi
  if [ "${IS_PHASE_F}" -eq 1 ]; then
    archiver_state="$(process_state "${ARCHIVER_PID}")"
    if [ -z "${archiver_state}" ] || [[ "${archiver_state}" == Z* ]]; then
      echo "Phase-F replay archiver exited unexpectedly. Last log lines:" >&2
      tail -100 "${ARCHIVER_LOG}" >&2 || true
      exit 1
    fi
  fi
  COUNT=$(find "${BUFFER_DIR}/gen1ou" -maxdepth 1 -type f \( -name '*.json' -o -name '*.json.lz4' \) | wc -l | tr -d ' ')
  if [ "${COUNT}" -gt "${DSET_MIN_SIZE}" ]; then
    break
  fi
  printf '\rFIFO prefill: %s / %s' "${COUNT}" "$((DSET_MIN_SIZE + 1))"
  sleep 5
done
printf '\nFIFO ready. Starting GPU learner; CPU collectors remain active.\n'

# Learner is the only long-running process allowed to see CUDA.
"${PYTHON_BIN}" -m metamon.rl.taurosv1b_online \
  --run_config "${CONFIG}" \
  --mode learn \
  --save_dir "${SAVE_DIR}" \
  --buffer_dir "${BUFFER_DIR}" \
  "${LEARNER_INIT[@]}" \
  "${LEARNER_EXTRA[@]}" > >(tee "${LEARNER_LOG}") 2>&1 &
LEARNER_PID=$!

LEARNER_STATUS=0
while true; do
  learner_state="$(process_state "${LEARNER_PID}")"
  if [ -z "${learner_state}" ] || [[ "${learner_state}" == Z* ]]; then
    break
  fi

  if ! check_collectors; then
    kill -TERM "${LEARNER_PID}" 2>/dev/null || true
    LEARNER_STATUS=1
    break
  fi

  if [ "${IS_PHASE_F}" -eq 1 ]; then
    archiver_state="$(process_state "${ARCHIVER_PID}")"
    if [ -z "${archiver_state}" ] || [[ "${archiver_state}" == Z* ]]; then
      echo "Phase-F replay archiver exited unexpectedly during training. Last log lines:" >&2
      tail -100 "${ARCHIVER_LOG}" >&2 || true
      kill -TERM "${LEARNER_PID}" 2>/dev/null || true
      LEARNER_STATUS=1
      break
    fi
  fi
  sleep 30
done

set +e
wait "${LEARNER_PID}"
child_status=$?
set -e
if [ "${child_status}" -ne 0 ] && [ "${LEARNER_STATUS}" -eq 0 ]; then
  LEARNER_STATUS="${child_status}"
fi
LEARNER_PID=""

stop_collectors

if [ "${IS_PHASE_F}" -eq 1 ]; then
  if [ -n "${ARCHIVER_PID}" ]; then
    kill -TERM "${ARCHIVER_PID}" 2>/dev/null || true
    wait "${ARCHIVER_PID}" 2>/dev/null || true
    ARCHIVER_PID=""
  fi
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_replay_archive \
    --source_dir "${BUFFER_DIR}/gen1ou" \
    --archive_dir "${PHASE_F_ARCHIVE_DIR}/gen1ou" \
    --max_files "${PHASE_F_ARCHIVE_MAX}" \
    --once || LEARNER_STATUS=1
fi

exit "${LEARNER_STATUS}"
