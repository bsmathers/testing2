#!/usr/bin/env bash
set -euo pipefail

# End-to-end TaurosV1B pipeline:
#   A   150 epochs V0 -> 35M policy distillation
#   B1  75k DAgger games + 50 distillation epochs
#   B2  75k DAgger games + 50 distillation epochs
#   C   50 critic-only epochs
#   D   25 critic/KL bridge epochs
#   E   800 public-opponent online-RL epochs
#   F   800 gated self-play epochs
#
# From phase E onward, a separate evaluator plays 50 games against TaurosV0@62
# every 5 epochs and logs win rate to a W&B evaluation run grouped with the
# learner. Five-epoch policy files are pruned after evaluation except every 25th
# epoch; the best checkpoint is copied to best_policy.pt before pruning.
#
# Required for W&B:
#   export WANDB_API_KEY=...
# Optional:
#   export METAMON_WANDB_PROJECT=online-metamon
#   export METAMON_WANDB_ENTITY=...
#   export EVAL_GPU=0     # GPU index visible to this process; use a spare GPU if available
#
# To constrain training to a specific GPU, invoke this whole script under the
# desired CUDA_VISIBLE_DEVICES setting. In that case EVAL_GPU is relative to the
# same visible-device list (normally 0 on a one-GPU run).
#
# Safe to rerun: completed A-D artifacts are reused; online phases resume from
# their newest full Accelerate state; tournament state is persistent.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SAVE_DIR="${METAMON_SAVE_DIR:-${REPO_DIR}/checkpoints}"
WORK_DIR="${V1B_WORK_DIR:-${REPO_DIR}/v1b_work}"
PRETRAIN_DIR="${WORK_DIR}/pretrain"
DAGGER1_DIR="${WORK_DIR}/dagger1"
DAGGER2_DIR="${WORK_DIR}/dagger2"
EVAL_GPU="${EVAL_GPU:-0}"
WANDB_PROJECT="${METAMON_WANDB_PROJECT:-online-metamon}"
WANDB_ENTITY="${METAMON_WANDB_ENTITY:-}"

mkdir -p "${SAVE_DIR}" "${PRETRAIN_DIR}" "${DAGGER1_DIR}" "${DAGGER2_DIR}"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"
export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_ALLOW_ANY_POKE_ENV=1

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "Warning: WANDB_API_KEY is not set; learner/evaluator W&B logging may fail." >&2
fi

phase_a="${PRETRAIN_DIR}/phase_a.pt"
phase_b1="${PRETRAIN_DIR}/phase_b1.pt"
phase_b2="${PRETRAIN_DIR}/phase_b2.pt"
phase_c="${PRETRAIN_DIR}/phase_c.pt"
phase_d="${PRETRAIN_DIR}/phase_d.pt"

run_if_missing() {
  local output="$1"
  shift
  if [ -s "${output}" ]; then
    echo "[resume] Reusing ${output}"
  else
    echo "[run] $*"
    "$@"
    if [ ! -s "${output}" ]; then
      echo "Expected output was not produced: ${output}" >&2
      exit 1
    fi
  fi
}

count_games() {
  local dir="$1"
  if [ ! -d "${dir}/gen1ou" ]; then
    echo 0
    return 0
  fi
  find "${dir}/gen1ou" -maxdepth 1 -type f \( -name '*.json' -o -name '*.json.lz4' \) 2>/dev/null | wc -l | tr -d ' '
}

collect_to_75k() {
  local weights="$1"
  local outdir="$2"
  local count
  mkdir -p "${outdir}/gen1ou"
  count=$(count_games "${outdir}")
  if [ "${count}" -ge 75000 ]; then
    echo "[resume] ${outdir}: ${count} DAgger games already present"
    return
  fi
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_collect_dagger \
    --weights "${weights}" \
    --output_dir "${outdir}" \
    --target_games 75000
}

# ------------------------------ A ------------------------------------------
run_if_missing "${phase_a}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain \
  --phase a \
  --output_weights "${phase_a}"

# ------------------------------ B1 -----------------------------------------
collect_to_75k "${phase_a}" "${DAGGER1_DIR}"
run_if_missing "${phase_b1}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain \
  --phase b1 \
  --input_weights "${phase_a}" \
  --dagger1_dir "${DAGGER1_DIR}" \
  --output_weights "${phase_b1}"

# ------------------------------ B2 -----------------------------------------
collect_to_75k "${phase_b1}" "${DAGGER2_DIR}"
run_if_missing "${phase_b2}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain \
  --phase b2 \
  --input_weights "${phase_b1}" \
  --dagger1_dir "${DAGGER1_DIR}" \
  --dagger2_dir "${DAGGER2_DIR}" \
  --output_weights "${phase_b2}"

# ------------------------------ C ------------------------------------------
run_if_missing "${phase_c}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain \
  --phase c \
  --input_weights "${phase_b2}" \
  --dagger1_dir "${DAGGER1_DIR}" \
  --dagger2_dir "${DAGGER2_DIR}" \
  --output_weights "${phase_c}"

# ------------------------------ D ------------------------------------------
run_if_missing "${phase_d}" \
  "${PYTHON_BIN}" -m metamon.rl.taurosv1b_pretrain \
  --phase d \
  --input_weights "${phase_c}" \
  --dagger1_dir "${DAGGER1_DIR}" \
  --dagger2_dir "${DAGGER2_DIR}" \
  --output_weights "${phase_d}"

run_online_phase() {
  local phase="$1"
  local base_weights="$2"
  local run_name
  if [ "${phase}" = "e" ]; then
    run_name="taurosv1b_phase_e"
  else
    run_name="taurosv1b_phase_f"
  fi

  echo "[online ${phase^^}] starting ${run_name}"
  BASE_WEIGHTS="${base_weights}" \
    bash scripts/train_taurosv1b.sh "${phase}" --log &
  local train_pid=$!

  local watcher_args=(
    -m metamon.rl.taurosv1b_tournament_watch
    --save_dir "${SAVE_DIR}"
    --run_name "${run_name}"
    --phase "${phase}"
    --gpu "${EVAL_GPU}"
    --games 50
    --eval_every 5
    --retain_every 25
    --learner_pid "${train_pid}"
    --wandb_project "${WANDB_PROJECT}"
  )
  if [ -n "${WANDB_ENTITY}" ]; then
    watcher_args+=(--wandb_entity "${WANDB_ENTITY}")
  fi
  "${PYTHON_BIN}" "${watcher_args[@]}" &
  local watcher_pid=$!

  set +e
  wait "${train_pid}"
  local train_status=$?
  wait "${watcher_pid}"
  local watcher_status=$?
  set -e

  if [ "${train_status}" -ne 0 ]; then
    echo "Training phase ${phase} failed with status ${train_status}." >&2
    exit "${train_status}"
  fi
  if [ "${watcher_status}" -ne 0 ]; then
    echo "Tournament watcher phase ${phase} failed with status ${watcher_status}." >&2
    exit "${watcher_status}"
  fi
}

# ------------------------------ E ------------------------------------------
run_online_phase e "${phase_d}"

E_TOURN_DIR="${SAVE_DIR}/taurosv1b_phase_e/tournaments_vs_taurosv0_62"
E_STATE="${E_TOURN_DIR}/state.json"
E_BEST="${E_TOURN_DIR}/best_policy.pt"
if [ ! -s "${E_STATE}" ] || [ ! -s "${E_BEST}" ]; then
  echo "Phase E completed without a valid tournament best checkpoint." >&2
  exit 1
fi

V0_GATE="${V0_GATE:-0.50}"
"${PYTHON_BIN}" - "${E_STATE}" "${V0_GATE}" <<'PY'
import json, sys
state = json.load(open(sys.argv[1]))
gate = float(sys.argv[2])
wr = state.get("best_winrate")
ep = state.get("best_epoch")
if wr is None:
    raise SystemExit("No phase-E tournament win rate was recorded")
print(f"Phase-E best vs TaurosV0@62: epoch {ep}, winrate={wr:.3f}; gate={gate:.3f}")
if float(wr) < gate:
    raise SystemExit(
        "Phase E did not clear the TaurosV0 gate. Refusing to start phase F; "
        "inspect the run instead of self-playing a sub-V0 policy."
    )
PY

# ------------------------------ F ------------------------------------------
run_online_phase f "${E_BEST}"

echo "TaurosV1B pipeline complete."
echo "Phase E best: ${E_BEST}"
echo "Phase F tournament state: ${SAVE_DIR}/taurosv1b_phase_f/tournaments_vs_taurosv0_62/state.json"
