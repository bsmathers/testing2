#!/usr/bin/env bash
set -euo pipefail

# Safe launcher for the long TaurosV1B online phases. Unlike scripts/train.sh,
# cleanup is PID-scoped and never pkill's unrelated Metamon jobs.
#
# Initial run:
#   BASE_WEIGHTS=/path/to/v1b_phase_d.pt bash scripts/train_taurosv1b.sh e --log
#   BASE_WEIGHTS=/path/to/phase_e_final.pt bash scripts/train_taurosv1b.sh f --log
#
# Relaunch after interruption:
#   bash scripts/train_taurosv1b.sh e --log
# The launcher automatically resumes the newest full Accelerate state and aligns
# latest/policy.pt + the collector to the raw policy from that exact epoch.
#
# Phase F additionally mirrors completed FIFO replays into a separate capped
# archive. The training FIFO remains at its configured 150k size; archive files
# are hard-linked on the same filesystem (copied only across filesystems), so FIFO
# eviction does not remove the archived replay and does not change training data.

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
TOTAL_CPUS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
if [ "${TOTAL_CPUS}" -gt 2 ]; then
  N_WORKERS="${N_WORKERS:-$((TOTAL_CPUS - 2))}"
else
  N_WORKERS="${N_WORKERS:-1}"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
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

COLLECTOR_PID=""
LEARNER_PID=""
ARCHIVER_PID=""
cleanup() {
  local status=$?
  set +e
  if [ -n "${COLLECTOR_PID}" ]; then
    kill -TERM "${COLLECTOR_PID}" 2>/dev/null || true
    pkill -TERM -P "${COLLECTOR_PID}" 2>/dev/null || true
  fi
  if [ -n "${LEARNER_PID}" ]; then
    kill -TERM "${LEARNER_PID}" 2>/dev/null || true
    pkill -TERM -P "${LEARNER_PID}" 2>/dev/null || true
  fi
  if [ -n "${ARCHIVER_PID}" ]; then
    kill -TERM "${ARCHIVER_PID}" 2>/dev/null || true
  fi
  return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Only the learner owns the E/F W&B run. Passing --log to the collector would
# create a second, misleading tracker run. Other user CLI overrides are learner-only.
LEARNER_EXTRA=("$@")

RUN_DIR="${SAVE_DIR}/${RUN_NAME}/ckpts"
STATE_DIR="${RUN_DIR}/training_states"
POLICY_DIR="${RUN_DIR}/policy_weights"
LATEST_DIR="${RUN_DIR}/latest"
RESUME_EPOCH=""
if [ -d "${STATE_DIR}" ]; then
  RESUME_EPOCH=$(
    find "${STATE_DIR}" -maxdepth 1 -mindepth 1 -type d -name "${RUN_NAME}_epoch_*" -print 2>/dev/null \
      | sed -E "s#.*${RUN_NAME}_epoch_([0-9]+)$#\1#" \
      | sort -n \
      | tail -1
  )
fi

COLLECTOR_INIT=()
LEARNER_INIT=()
if [ -n "${RESUME_EPOCH}" ]; then
  RESUME_POLICY="${POLICY_DIR}/policy_epoch_${RESUME_EPOCH}.pt"
  if [ ! -f "${RESUME_POLICY}" ]; then
    echo "Found full state epoch ${RESUME_EPOCH}, but matching raw policy is missing: ${RESUME_POLICY}" >&2
    exit 1
  fi

  # Delete immutable numbered policies from the abandoned future branch. This
  # is especially important in phase F: discover:true self-play would otherwise
  # rediscover policies newer than the optimizer state we are rolling back to.
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

  mkdir -p "${LATEST_DIR}"
  # latest may be newer than the sparse full optimizer state after a crash. Roll
  # it back before the collector starts reading it.
  cp -f "${RESUME_POLICY}" "${LATEST_DIR}/policy.pt"
  COLLECTOR_INIT=(--base_weights "${RESUME_POLICY}")
  LEARNER_INIT=(--resume_training_state --resume_epoch "${RESUME_EPOCH}")
  echo "Resuming ${RUN_NAME} from full state epoch ${RESUME_EPOCH}."
else
  : "${BASE_WEIGHTS:?No resumable state found. Set BASE_WEIGHTS for the initial phase launch.}"
  COLLECTOR_INIT=(--base_weights "${BASE_WEIGHTS}")
  LEARNER_INIT=(--base_weights "${BASE_WEIGHTS}")
  echo "Starting ${RUN_NAME} from ${BASE_WEIGHTS}."
fi

cd "${REPO_DIR}"
COLLECTOR_LOG="${LOG_DIR}/collector.log"
LEARNER_LOG="${LOG_DIR}/learner.log"
ARCHIVER_LOG="${LOG_DIR}/phase_f_replay_archive.log"

"${PYTHON_BIN}" -m metamon.rl.taurosv1b_online \
  --run_config "${CONFIG}" \
  --mode collect \
  --save_dir "${SAVE_DIR}" \
  --buffer_dir "${BUFFER_DIR}" \
  "${COLLECTOR_INIT[@]}" \
  --lanes "${LANES}" \
  --n_workers "${N_WORKERS}" \
  >"${COLLECTOR_LOG}" 2>&1 &
COLLECTOR_PID=$!

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

echo "Collector PID ${COLLECTOR_PID}; ensuring FIFO has $((DSET_MIN_SIZE + 1)) battles..."
while true; do
  if ! kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
    echo "Collector exited during FIFO prefill. Last log lines:" >&2
    tail -100 "${COLLECTOR_LOG}" >&2 || true
    exit 1
  fi
  if [ "${IS_PHASE_F}" -eq 1 ] && ! kill -0 "${ARCHIVER_PID}" 2>/dev/null; then
    echo "Phase-F replay archiver exited unexpectedly. Last log lines:" >&2
    tail -100 "${ARCHIVER_LOG}" >&2 || true
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
  "${LEARNER_INIT[@]}" \
  "${LEARNER_EXTRA[@]}" > >(tee "${LEARNER_LOG}") 2>&1 &
LEARNER_PID=$!

LEARNER_STATUS=0
if [ "${IS_PHASE_F}" -eq 1 ]; then
  # Monitor archive health while F runs. A dead child that has not yet been
  # waited can remain as a zombie for which `kill -0` still succeeds, so inspect
  # process state as well and break as soon as the learner is gone/zombified.
  while true; do
    learner_state=$(ps -o stat= -p "${LEARNER_PID}" 2>/dev/null | awk '{print $1}' || true)
    if [ -z "${learner_state}" ] || [[ "${learner_state}" == Z* ]]; then
      break
    fi
    if ! kill -0 "${ARCHIVER_PID}" 2>/dev/null; then
      echo "Phase-F replay archiver exited unexpectedly during training. Last log lines:" >&2
      tail -100 "${ARCHIVER_LOG}" >&2 || true
      kill -TERM "${LEARNER_PID}" 2>/dev/null || true
      LEARNER_STATUS=1
      break
    fi
    sleep 30
  done
fi

set +e
wait "${LEARNER_PID}"
child_status=$?
set -e
if [ "${child_status}" -ne 0 ] && [ "${LEARNER_STATUS}" -eq 0 ]; then
  LEARNER_STATUS="${child_status}"
fi
LEARNER_PID=""

# Stop collection, stop the asynchronous archiver, then do one final synchronous
# pass so completed FIFO files cannot be lost in the polling interval.
if [ "${IS_PHASE_F}" -eq 1 ]; then
  if [ -n "${COLLECTOR_PID}" ]; then
    kill -TERM "${COLLECTOR_PID}" 2>/dev/null || true
    pkill -TERM -P "${COLLECTOR_PID}" 2>/dev/null || true
    wait "${COLLECTOR_PID}" 2>/dev/null || true
    COLLECTOR_PID=""
  fi
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
