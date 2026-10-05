#!/usr/bin/env bash
set -euo pipefail

# Safe launcher for the long TaurosV1B online phases.  Unlike scripts/train.sh,
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
  return "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Only the learner owns the W&B run. Passing --log to the collector would create
# a second, misleading tracker run. Other user CLI overrides are learner-only.
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

  # Delete immutable numbered policies from the abandoned future branch. This is
  # especially important in phase F: discover:true self-play would otherwise
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

  # Tournament results newer than the full optimizer state belong to the same
  # abandoned future branch and must not suppress re-evaluation after resume.
  TOURN_DIR="${SAVE_DIR}/${RUN_NAME}/tournaments_vs_taurosv0_62"
  TOURN_STATE="${TOURN_DIR}/state.json"
  if [ -f "${TOURN_STATE}" ]; then
    "${PYTHON_BIN}" - "${TOURN_DIR}" "${RESUME_EPOCH}" <<'PY'
import json, pathlib, shutil, sys
root = pathlib.Path(sys.argv[1])
resume = int(sys.argv[2])
state_path = root / "state.json"
try:
    state = json.loads(state_path.read_text())
except Exception:
    state = {}
completed = sorted(int(e) for e in state.get("completed_epochs", []) if int(e) <= resume)
# Reconstruct the best retained result at/before the resume point from each
# epoch's append-only matchup result. This avoids preserving a best model from an
# optimizer branch that no longer exists.
best_epoch = None
best_wr = None
best_src = None
for epoch in completed:
    result_file = root / f"epoch_{epoch:04d}" / "matchup_results.jsonl"
    if not result_file.exists():
        continue
    for line in result_file.read_text().splitlines():
        try:
            r = json.loads(line)
            total = int(r.get("total_battles", 0))
            if total <= 0:
                continue
            wr = int(r.get("policy_a_wins", 0)) / total
        except Exception:
            continue
        if best_wr is None or wr > best_wr:
            ckpt = root.parent / "ckpts" / "policy_weights" / f"policy_epoch_{epoch}.pt"
            if ckpt.exists():
                best_epoch, best_wr, best_src = epoch, wr, ckpt
for p in root.glob("epoch_*"):
    try:
        epoch = int(p.name.split("_")[-1])
    except ValueError:
        continue
    if epoch > resume:
        shutil.rmtree(p, ignore_errors=True)
best_path = root / "best_policy.pt"
if best_src is not None:
    shutil.copy2(best_src, best_path)
elif best_path.exists():
    best_path.unlink()
state = {
    "completed_epochs": completed,
    "best_epoch": best_epoch,
    "best_winrate": best_wr,
}
tmp = state_path.with_suffix(".tmp")
tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
tmp.replace(state_path)
PY
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

echo "Collector PID ${COLLECTOR_PID}; ensuring FIFO has $((DSET_MIN_SIZE + 1)) battles..."
while true; do
  if ! kill -0 "${COLLECTOR_PID}" 2>/dev/null; then
    echo "Collector exited during FIFO prefill. Last log lines:" >&2
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
  "${LEARNER_INIT[@]}" \
  "${LEARNER_EXTRA[@]}" > >(tee "${LEARNER_LOG}") 2>&1 &
LEARNER_PID=$!

wait "${LEARNER_PID}"
