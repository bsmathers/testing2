#!/usr/bin/env bash
set -euo pipefail

# Restart TaurosV1B from an explicit Accelerate full-state checkpoint.
#
# Usage:
#   bash scripts/restart_taurosv1b_from_checkpoint.sh \
#     /path/to/training_states/taurosv1b_phase_e_epoch_50
#
# Optional override:
#   BUFFER_DIR=/path/to/existing/replay/buffer bash ...

if [ "$#" -ne 1 ]; then
  echo "usage: $0 /absolute/path/to/training_states/<run_name>_epoch_<N>" >&2
  exit 2
fi

CHECKPOINT_DIR="$(realpath "$1")"
if [ ! -d "$CHECKPOINT_DIR" ]; then
  echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
  exit 1
fi

STATE_DIR="$(dirname "$CHECKPOINT_DIR")"
if [ "$(basename "$STATE_DIR")" != "training_states" ]; then
  echo "Checkpoint must live directly under training_states/: $CHECKPOINT_DIR" >&2
  exit 1
fi

CKPT_BASENAME="$(basename "$CHECKPOINT_DIR")"
if [[ ! "$CKPT_BASENAME" =~ ^(taurosv1b_phase_[ef])_epoch_([0-9]+)$ ]]; then
  echo "Unrecognized checkpoint name: $CKPT_BASENAME" >&2
  exit 1
fi

RUN_NAME="${BASH_REMATCH[1]}"
RESUME_EPOCH="${BASH_REMATCH[2]}"
if [ "$RUN_NAME" = "taurosv1b_phase_e" ]; then
  PHASE=e
elif [ "$RUN_NAME" = "taurosv1b_phase_f" ]; then
  PHASE=f
else
  echo "Unsupported run: $RUN_NAME" >&2
  exit 1
fi

RUN_CKPT_DIR="$(dirname "$STATE_DIR")"
RUN_DIR="$(dirname "$RUN_CKPT_DIR")"
SAVE_DIR="$(dirname "$RUN_DIR")"

if [ "$(basename "$RUN_DIR")" != "$RUN_NAME" ]; then
  echo "Checkpoint path/run-name mismatch: $RUN_DIR vs $RUN_NAME" >&2
  exit 1
fi

POLICY="$RUN_CKPT_DIR/policy_weights/policy_epoch_$RESUME_EPOCH.pt"
if [ ! -s "$POLICY" ]; then
  echo "Matching raw policy missing or empty: $POLICY" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Current direct C->E->F layout:
#   WORK_DIR/ckpts/from_c_direct_ef_kl50_online/<run>/...
SAVE_PARENT="$(dirname "$SAVE_DIR")"
if [ "$(basename "$SAVE_DIR")" = "from_c_direct_ef_kl50_online" ] &&
   [ "$(basename "$SAVE_PARENT")" = "ckpts" ]; then
  WORK_DIR="$(dirname "$SAVE_PARENT")"
else
  WORK_DIR=""
fi

if [ -n "${BUFFER_DIR:-}" ]; then
  RESUME_BUFFER_DIR="$(realpath "$BUFFER_DIR")"
elif [ -n "$WORK_DIR" ]; then
  if [ "$PHASE" = e ]; then
    RESUME_BUFFER_DIR="$WORK_DIR/buffer_taurosv1b_phase_e_from_c_direct_kl50"
  else
    RESUME_BUFFER_DIR="$WORK_DIR/buffer_taurosv1b_phase_f_from_c_direct_kl50"
  fi
else
  echo "Cannot infer replay buffer from checkpoint path." >&2
  echo "Set BUFFER_DIR to the existing replay buffer root." >&2
  exit 1
fi

if [ ! -d "$RESUME_BUFFER_DIR/gen1ou" ]; then
  echo "Replay buffer not found: $RESUME_BUFFER_DIR/gen1ou" >&2
  exit 1
fi

LATEST_STATE_EPOCH="$(
  for p in "$STATE_DIR/$RUN_NAME"_epoch_*; do
    [ -d "$p" ] || continue
    n="${p##*_epoch_}"
    [[ "$n" =~ ^[0-9]+$ ]] && printf '%s\n' "$n"
  done | sort -n | tail -1
)"

if [ "$LATEST_STATE_EPOCH" != "$RESUME_EPOCH" ]; then
  echo "Requested epoch $RESUME_EPOCH, but latest full state is $LATEST_STATE_EPOCH." >&2
  echo "Refusing to resume a different checkpoint implicitly." >&2
  exit 1
fi

mkdir -p "$RUN_CKPT_DIR/latest"
cp -f "$POLICY" "$RUN_CKPT_DIR/latest/policy.pt"

echo "TaurosV1B checkpoint restart"
echo "  phase:         $PHASE"
echo "  run:           $RUN_NAME"
echo "  full state:    $CHECKPOINT_DIR"
echo "  raw policy:    $POLICY"
echo "  save dir:      $SAVE_DIR"
echo "  replay buffer: $RESUME_BUFFER_DIR"
echo "  next epoch:    $((RESUME_EPOCH + 1))"

cd "$REPO_DIR"

METAMON_SAVE_DIR="$SAVE_DIR" \
BUFFER_DIR="$RESUME_BUFFER_DIR" \
COLLECTOR_CPUS="${COLLECTOR_CPUS:-10}" \
COLLECTOR_SHARDS="${COLLECTOR_SHARDS:-5}" \
WANDB_RESUME="${WANDB_RESUME:-allow}" \
  bash "$SCRIPT_DIR/train_taurosv1b.sh" "$PHASE" --log
