#!/usr/bin/env bash
set -Eeuo pipefail

# Memory-conservative Apple-Silicon launcher. This deliberately reuses the
# audited V1a bootstrap/data path while selecting parameter-compatible portable
# attention instead of CUDA-only FlashAttention.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

export REPO_DIR
export DEVICE_BACKEND=mps
export METAMON_PORTABLE_ATTENTION=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_ENABLE_MPS_FALLBACK=0

export PERSIST_ROOT="${PERSIST_ROOT:-${REPO_DIR}/.runs/smallg1onlinev1a-mps}"
export RUN_NAME="${RUN_NAME:-smallg1onlinev1a-mps}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-smallg1onlinev1a-mps-v1}"
export WANDB_NAME="${WANDB_NAME:-smallg1onlinev1a-mps}"
export WANDB_TAGS="${WANDB_TAGS:-smallg1onlinev1a,mps,portable-attention,128x3000,14x1,gen1ou,epoch475,807-teams,zig}"

# One collector model at a time. 128 lanes x 3000 steps preserves the CUDA
# run's 384,000 lane-transition budget without concurrent model replicas.
export LANES="${LANES:-128}"
export COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-4}"
export COLLECTOR_PROCESSES="${COLLECTOR_PROCESSES:-1}"
export TRAIN_COLLECTOR_CONCURRENCY=1
export TRAIN_TIMESTEPS_PER_EPOCH="${TRAIN_TIMESTEPS_PER_EPOCH:-3000}"

# First try the desired 14x1 learner in isolation. If this produces an MPS OOM,
# rerun explicitly with BATCH_SIZE_PER_GPU=7 GRAD_ACCUM=2.
export BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-14}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export DLOADER_WORKERS="${DLOADER_WORKERS:-0}"
export MIXED_PRECISION="${MIXED_PRECISION:-no}"

export EPOCHS="${EPOCHS:-401}"
export STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000}"
export PREFILL_FILES="${PREFILL_FILES:-6000}"
export INSTALL_DEPS="${INSTALL_DEPS:-0}"
export CLEAN_OLD_RUN_WEIGHTS=0

exec "${SCRIPT_DIR}/train_smallg1onlinev1a_vast.sh"
