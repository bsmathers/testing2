#!/usr/bin/env bash
set -Eeuo pipefail

# Launch SmallG1OnlineV1a on a single Vast.ai GPU instance.
#
# Expected layout:
#   REPO_DIR/                         testing2 checkout
#   REPO_DIR/policy/                  3 epoch-475 shards + checksum
#   REPO_DIR/teams_replay/good_teams  807 .txt/.gen1ou_team files
#
# Target Vast host: RTX 5090 (32 GiB), CUDA 12.8, PyTorch 2.7.1+cu128,
# 16+ CPU cores, and 64+ GiB system RAM.
# Persistent outputs default to /workspace/smallg1onlinev1a.

die() { echo "ERROR: $*" >&2; exit 1; }
log() { printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PERSIST_ROOT="${PERSIST_ROOT:-/workspace/smallg1onlinev1a}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_NAME="${RUN_NAME:-smallg1onlinev1a}"
WANDB_PROJECT="${METAMON_WANDB_PROJECT:-smallg1onlinev1a}"
WANDB_RUN_ID="${WANDB_RUN_ID:-smallg1onlinev1a-vast-v1}"
WANDB_NAME="${WANDB_NAME:-smallg1onlinev1a}"
COLLECTOR_PROCESSES="${COLLECTOR_PROCESSES:-1}"
LANES="${LANES:-96}"
COLLECTOR_WORKERS="${COLLECTOR_WORKERS:-8}"
DLOADER_WORKERS="${DLOADER_WORKERS:-8}"
BATCH_SIZE_PER_GPU="${BATCH_SIZE_PER_GPU:-7}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MIXED_PRECISION="${MIXED_PRECISION:-no}"
PREFILL_FILES="${PREFILL_FILES:-5001}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

CACHE_DIR="${PERSIST_ROOT}/cache"
SAVE_DIR="${PERSIST_ROOT}/checkpoints"
BUFFER_DIR="${PERSIST_ROOT}/online_buffer"
CONFIG_DIR="${PERSIST_ROOT}/config"
BOOTSTRAP_DIR="${PERSIST_ROOT}/bootstrap"
LOG_DIR="${PERSIST_ROOT}/logs"
RAW_WEIGHTS="${BOOTSTRAP_DIR}/policy_epoch_475.pt"
ONLINE_WEIGHTS="${BOOTSTRAP_DIR}/policy_epoch_475_online.pt"
RUN_CONFIG="${CONFIG_DIR}/smallg1onlinev1a.yaml"
POOL_CONFIG="${CONFIG_DIR}/hl_gen1ou_smallg1onlinev1a.yaml"
EXPERT_DIR="${CACHE_DIR}/teams/smallg1onlinev1a_expert/gen1ou"

[[ -f "${REPO_DIR}/pyproject.toml" ]] || die "REPO_DIR is not testing2: ${REPO_DIR}"
mkdir -p "${CACHE_DIR}" "${SAVE_DIR}" "${BUFFER_DIR}/gen1ou" \
  "${CONFIG_DIR}" "${BOOTSTRAP_DIR}" "${LOG_DIR}"

export METAMON_CACHE_DIR="${CACHE_DIR}"
export METAMON_SAVE_DIR="${SAVE_DIR}"
export METAMON_BATTLE_HOST="${REPO_DIR}/metamon/env/vectorized/battle_host_engine.js"
export METAMON_PKMN_ENGINE="${REPO_DIR}/metamon/env/vectorized/pkmn-showdown.node"
export TAUROSV1B_RUN_NAME="epoch475_frozen"
export METAMON_WANDB_PROJECT="${WANDB_PROJECT}"
export WANDB_RUN_ID WANDB_NAME
export WANDB_RESUME="${WANDB_RESUME:-allow}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-smallg1onlinev1a}"
export WANDB_TAGS="${WANDB_TAGS:-smallg1onlinev1a,gen1ou,epoch475,807-teams,vast,zig}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MAX_JOBS="${MAX_JOBS:-8}"
export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-120}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export HF_XET_HIGH_PERFORMANCE=1

cd "${REPO_DIR}"
[[ "${COLLECTOR_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || die "COLLECTOR_PROCESSES must be a positive integer"
[[ "${LANES}" =~ ^[1-9][0-9]*$ ]] || die "LANES must be a positive integer"
[[ "${COLLECTOR_WORKERS}" =~ ^[1-9][0-9]*$ ]] || die "COLLECTOR_WORKERS must be a positive integer"
[[ "${DLOADER_WORKERS}" =~ ^[0-9]+$ ]] || die "DLOADER_WORKERS must be a non-negative integer"
[[ "${BATCH_SIZE_PER_GPU}" =~ ^[1-9][0-9]*$ ]] || die "BATCH_SIZE_PER_GPU must be a positive integer"
[[ "${GRAD_ACCUM}" =~ ^[1-9][0-9]*$ ]] || die "GRAD_ACCUM must be a positive integer"
[[ "${MIXED_PRECISION}" =~ ^(no|fp16|bf16)$ ]] || die "MIXED_PRECISION must be no, fp16, or bf16"
(( LANES % COLLECTOR_WORKERS == 0 )) \
  || die "LANES (${LANES}) must be divisible by COLLECTOR_WORKERS (${COLLECTOR_WORKERS})"

if [[ "${INSTALL_DEPS}" == "1" && "$(id -u)" -eq 0 ]] && command -v apt-get >/dev/null; then
  log "Installing required system build tools"
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    git curl ca-certificates build-essential xz-utils
fi
for required_command in git curl gcc g++ make; do
  command -v "${required_command}" >/dev/null \
    || die "Missing required command: ${required_command}"
done
command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable; start a Vast GPU instance"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
df -h "${PERSIST_ROOT}"
df -i "${PERSIST_ROOT}"

check_torch_cuda() {
  "${PYTHON_BIN}" - <<'PY'
import re
import shutil
import subprocess
import sys

import torch

if sys.version_info[:2] not in {(3, 11), (3, 12)}:
    raise SystemExit(
        f"Use Python 3.11 or 3.12 for the pinned Torch/FlashAttention stack; got {sys.version.split()[0]}"
    )

def version_tuple(value):
    match = re.match(r"(\d+)\.(\d+)", value or "")
    if not match:
        raise SystemExit(f"Could not parse version: {value!r}")
    return tuple(map(int, match.groups()))

print(f"Python: {sys.version.split()[0]}")
print(f"Torch: {torch.__version__}; compiled CUDA: {torch.version.cuda}")
if version_tuple(torch.__version__.split("+")[0]) < (2, 7):
    raise SystemExit("torch>=2.7 is required for RTX 5090/Blackwell support")
if torch.version.cuda is None:
    raise SystemExit("This is a CPU-only Torch build; select a CUDA PyTorch template")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available to PyTorch; check the Vast GPU/template configuration")
print(f"GPU: {torch.cuda.get_device_name(0)}; VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB")
if torch.cuda.get_device_properties(0).total_memory < 30 * 2**30:
    raise SystemExit(
        "The default shared learner/collector settings require at least 30 GiB VRAM. "
        "For a smaller GPU, explicitly reduce LANES and batch settings."
    )

try:
    import flash_attn  # noqa: F401
except ImportError:
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise SystemExit(
            "flash-attn is absent and nvcc is unavailable. Use a CUDA development template "
            "(not a runtime-only image)."
        )
    output = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release\s+(\d+\.\d+)", output)
    if not match:
        raise SystemExit(f"Could not parse nvcc version:\n{output}")
    compiler_cuda = match.group(1)
    if version_tuple(compiler_cuda) != version_tuple(torch.version.cuda):
        raise SystemExit(
            f"CUDA mismatch: nvcc={compiler_cuda}, torch={torch.version.cuda}. "
            "Choose a Vast template whose Torch and CUDA toolkit match."
        )
    print(f"nvcc: CUDA {compiler_cuda} (matches Torch)")
PY
}

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  log "Installing dependencies and pinning the RTX 5090 Torch/CUDA stack"
  "${PYTHON_BIN}" -m pip install -U pip setuptools wheel packaging ninja
  "${PYTHON_BIN}" -m pip install torch==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
  check_torch_cuda
  "${PYTHON_BIN}" -m pip install 'numpy<2'
  "${PYTHON_BIN}" -m pip install -e . --no-deps
  "${PYTHON_BIN}" -m pip install \
    'gymnasium>=0.26,<=0.29.1' gin-config wandb einops tqdm lz4 termcolor rich \
    huggingface_hub datasets ratarmountcore accelerate orjson requests tabulate psutil \
    'websockets==12.0' \
    'poke-env @ git+https://github.com/UT-Austin-RPL/poke-env.git' \
    'amago @ git+https://github.com/UT-Austin-RPL/amago@v3.4.0'
  "${PYTHON_BIN}" -m pip install --force-reinstall --no-deps --no-build-isolation \
    'git+https://github.com/Dao-AILab/flash-attention.git@94e22c906678e5483fa0e9e24d8e787bc2c0ed4c'
fi

check_torch_cuda
"${PYTHON_BIN}" - <<'PY'
import amago
import flash_attn
import gin
import gymnasium
import lz4
import wandb
import metamon
print("Required Python imports succeeded")
PY

"${PYTHON_BIN}" - <<'PY'
import torch
from flash_attn import flash_attn_qkvpacked_func

# Exercise an sm_120 forward and backward kernel before any large downloads.
qkv = torch.randn(2, 64, 3, 4, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
out = flash_attn_qkvpacked_func(qkv, causal=True, window_size=(32, 0))
out.float().square().mean().backward()
torch.cuda.synchronize()
print("FlashAttention RTX 5090 forward/backward smoke test succeeded")
PY

if [[ "${WANDB_MODE}" == "online" && -z "${WANDB_API_KEY:-}" ]]; then
  die "WANDB_API_KEY is required when WANDB_MODE=online"
fi
if [[ "${WANDB_MODE}" == "online" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import os
import wandb

if not wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True):
    raise SystemExit("Weights & Biases login failed")
print("Weights & Biases login succeeded")
PY
fi

log "Building and verifying the native Zig battle engine"
bash scripts/setup_pkmn_engine.sh
[[ -s "${METAMON_PKMN_ENGINE}" ]] || die "Zig addon was not created: ${METAMON_PKMN_ENGINE}"
ZIG_PROBE="$(cd metamon/env/vectorized && echo '{"cmd":"close"}' | node battle_host_engine.js 2>&1 >/dev/null || true)"
grep -q 'HIGH-SPEED NATIVE ZIG ENGINE ACTIVE' <<<"${ZIG_PROBE}" \
  || die "battle_host_engine.js did not activate the native Zig engine"

log "Assembling and validating epoch 475"
if [[ ! -s "${RAW_WEIGHTS}" ]]; then
  mapfile -t PARTS < <(find "${REPO_DIR}/policy" -maxdepth 1 -type f \
    -name 'policy_epoch_475.pt.part-*' | sort)
  [[ "${#PARTS[@]}" -eq 3 ]] || die "Expected exactly 3 epoch-475 shards; found ${#PARTS[@]}"
  tmp_weights="${RAW_WEIGHTS}.assembling"
  : >"${tmp_weights}"
  for part in "${PARTS[@]}"; do
    size="$(stat -c '%s' "${part}")"
    (( size <= 104857600 )) || die "Shard exceeds 100 MiB: ${part}"
    cat "${part}" >>"${tmp_weights}"
  done
  expected="$(awk 'NR==1 {print tolower($1)}' policy/policy_epoch_475.pt.sha256)"
  actual="$(sha256sum "${tmp_weights}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || die "Epoch-475 SHA-256 mismatch: ${actual} != ${expected}"
  mv "${tmp_weights}" "${RAW_WEIGHTS}"
fi

"${PYTHON_BIN}" -m metamon.rl.taurosv1b_kl export-online \
  --input "${RAW_WEIGHTS}" --output "${ONLINE_WEIGHTS}"

"${PYTHON_BIN}" - "${ONLINE_WEIGHTS}" <<'PY'
import sys
import torch

path = sys.argv[1]
state = torch.load(path, map_location="cpu")
bad = [k for k in state if k.startswith("_kl_anchor_") or k == "_kl_forward_step"]
assert not bad, f"KL-only keys remain: {bad[:3]}"
assert len(state) == 598, f"Expected 598 state keys, found {len(state)}"
n = sum(v.numel() for v in state.values())
assert n == 34_766_593, f"Expected 34,766,593 parameters, found {n:,}"
print(f"Validated epoch-475 online policy: {len(state)} keys, {n:,} elements")
PY

log "Creating the 807-team expert set without deduplication"
GOOD_TEAMS_DIR="${GOOD_TEAMS_DIR:-${REPO_DIR}/teams_replay/good_teams}"
export GOOD_TEAMS_DIR EXPERT_DIR
"${PYTHON_BIN}" - <<'PY'
import os
import re
import shutil
from pathlib import Path

source = Path(os.environ["GOOD_TEAMS_DIR"])
target = Path(os.environ["EXPERT_DIR"])
teams = sorted(
    p for p in source.rglob("*")
    if p.is_file() and p.name.endswith((".txt", ".gen1ou_team"))
)
if len(teams) != 807:
    raise SystemExit(f"Expected exactly 807 team files, found {len(teams)} in {source}")
if target.exists():
    shutil.rmtree(target)
target.mkdir(parents=True)
names = []
for i, src in enumerate(teams):
    stem = src.name.removesuffix(".gen1ou_team").removesuffix(".txt")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    name = f"{i:04d}__{safe}.gen1ou_team"
    shutil.copyfile(src, target / name)
    names.append(name)
with (target / "index.csv").open("w", encoding="utf-8") as f:
    f.write("filename\n")
    f.writelines(f"{name}\n" for name in names)
print(f"Wrote {len(names)} entries to {target}; duplicate content was preserved")
PY

log "Writing the SmallG1OnlineV1a opponent pool and run configuration"
cat >"${POOL_CONFIG}" <<'YAML'
# SmallG1OnlineV1.5 roster, with SmallG1OnlineV0 replaced by epoch 475.
defaults:
  team_set: hl_05_26
  battle_backend: metamon
  checkpoints: [null]
  temperatures: [0.5, 1.0, 1.25, 1.5, 1.75, 2.0]
  num_agents: 1
agents:
  TaurosV0:
    model_name: TaurosV0
    num_agents: 2
    checkpoints: "range(50, 63, 2)"
  Kakuna:
    model_name: Kakuna
    num_agents: 2
    checkpoints: "range(24, 35, 2)"
  Epoch475:
    model_name: TaurosV1B
    num_agents: 2
    checkpoints: [475]
  Alakazam:
    model_name: Alakazam
    checkpoints: "range(2, 9, 2)"
  SyntheticRLV2:
    model_name: SyntheticRLV2
    checkpoints: "range(36, 49, 2)"
  V2ADataAblation:
    model_name: V2AGroupedV2DataAblation
    checkpoints: "range(74, 91, 2)"
  Superkazam:
    model_name: Superkazam
    checkpoints: "range(38, 51, 2)"
  Kadabra3:
    model_name: Kadabra3
    checkpoints: "range(10, 21, 2)"
YAML

cat >"${RUN_CONFIG}" <<YAML
run_name: ${RUN_NAME}
base_model: TaurosV1A
battle_format: gen1ou
dataset_config: online_selfplay.yaml
train_gin_config: grouped_v2_large_isfilter.gin
train_pool: ${POOL_CONFIG}
train_team_set: smallg1onlinev1a_expert
val_pool: metamon/rl/configs/opponent_pools/taurosv1a_val.yaml
val_team_set: smallg1onlinev1a_expert
dset_max_size: 300000
dset_min_size: 5000
initial_online_weight: 0.0
online_weight: 0.4
online_anneal_start_epoch: 0
online_anneal_end_epoch: 20
learning_rate: 8.0e-5
lr_warmup_epochs: 20
seq_floor_warmup_epochs: 20
batch_size_per_gpu: ${BATCH_SIZE_PER_GPU}
grad_accum: ${GRAD_ACCUM}
epochs: 401
steps_per_epoch: 1000
ckpt_interval: 10
full_state_ckpt_interval: 10
mixed_precision: "${MIXED_PRECISION}"
lanes: ${LANES}
n_workers: ${COLLECTOR_WORKERS}
train_timesteps_per_epoch: 500
temp_low: 1.0
temp_high: 2.0
val_timesteps: 1000
val_interval: 10
dloader_workers: ${DLOADER_WORKERS}
YAML

log "Downloading offline data, usage statistics, hl_05_26 teams, and opponent weights"
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path

from metamon.data.download import (
    download_parsed_replays,
    download_self_play_data,
    download_teams,
    download_usage_stats,
)

download_usage_stats(gen=1, version="v5", force_download=False)
download_parsed_replays("gen1ou", version="v6", force_download=False)
for subset in ("pac-base", "pac-exploratory", "pac-tauros"):
    download_self_play_data(subset, "gen1ou", version="main", force_download=False, extract=False)
download_teams("gen1ou", "hl_05_26", version="v5", force_download=False)

from metamon.backend.team_prediction.team import TeamSet
import os
team_dir = Path(os.environ["EXPERT_DIR"])
for path in sorted(team_dir.glob("*.gen1ou_team")):
    team = TeamSet.from_showdown_file(str(path), "gen1ou")
    if len(team.pokemon) != 6:
        raise ValueError(f"{path.name}: parsed {len(team.pokemon)} Pokemon, expected 6")

import metamon.rl.taurosv1b_online  # registers local TaurosV1B
from metamon.rl.pretrained import get_pretrained_model
public = {
    "TaurosV0": range(50, 63, 2),
    "Kakuna": range(24, 35, 2),
    "Alakazam": range(2, 9, 2),
    "SyntheticRLV2": range(36, 49, 2),
    "V2AGroupedV2DataAblation": range(74, 91, 2),
    "Superkazam": range(38, 51, 2),
    "Kadabra3": range(10, 21, 2),
}
for model_name, epochs in public.items():
    model = get_pretrained_model(model_name)
    for epoch in epochs:
        path = Path(model.get_path_to_checkpoint(epoch))
        if not path.exists():
            raise FileNotFoundError(f"Failed to fetch {model_name}@{epoch}: {path}")
print("All datasets, 807 expert teams, and public opponents are ready")
PY

FROZEN="${SAVE_DIR}/epoch475_frozen/ckpts/policy_weights/policy_epoch_475.pt"
LATEST="${SAVE_DIR}/${RUN_NAME}/ckpts/latest/policy.pt"
mkdir -p "$(dirname "${FROZEN}")" "$(dirname "${LATEST}")"
cp -f "${ONLINE_WEIGHTS}" "${FROZEN}"
[[ -s "${LATEST}" ]] || cp "${ONLINE_WEIGHTS}" "${LATEST}"

CKPT_ROOT="${SAVE_DIR}/${RUN_NAME}/ckpts"
STATE_ROOT="${CKPT_ROOT}/training_states"
POLICY_ROOT="${CKPT_ROOT}/policy_weights"
RESUME_EPOCH=""
if [[ -d "${STATE_ROOT}" ]]; then
  RESUME_EPOCH="$({ find "${STATE_ROOT}" -mindepth 1 -maxdepth 1 -type d \
    -name "${RUN_NAME}_epoch_*" -printf '%f\n' 2>/dev/null || true; } \
    | sed -n 's/.*_epoch_\([0-9][0-9]*\)$/\1/p' | sort -n | tail -1)"
fi

LEARNER_RESUME_ARGS=()
if [[ -n "${RESUME_EPOCH}" ]]; then
  RESUME_POLICY="${POLICY_ROOT}/policy_epoch_${RESUME_EPOCH}.pt"
  [[ -s "${RESUME_POLICY}" ]] || die "Full state epoch ${RESUME_EPOCH} has no matching policy"
  # Keep collection and learning on the same exact full-state epoch.
  cp -f "${RESUME_POLICY}" "${LATEST}"
  LEARNER_RESUME_ARGS=(--resume_training_state --resume_epoch "${RESUME_EPOCH}")
  log "Resuming full learner state from completed epoch ${RESUME_EPOCH}"
else
  if find "${POLICY_ROOT}" -maxdepth 1 -type f -name 'policy_epoch_*.pt' -print -quit 2>/dev/null | grep -q .; then
    die "Policy checkpoints exist but no full training state exists; refusing an optimizer-reset resume"
  fi
  cp -f "${ONLINE_WEIGHTS}" "${LATEST}"
  LEARNER_RESUME_ARGS=(--base_weights "${ONLINE_WEIGHTS}")
  log "Starting V1a epoch 0 from cleaned epoch-475 weights"
fi

COLLECTOR_PIDS=()
cleanup_collectors() {
  local pid
  if (( ${#COLLECTOR_PIDS[@]} == 0 )); then
    return
  fi
  for pid in "${COLLECTOR_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${COLLECTOR_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  COLLECTOR_PIDS=()
}
trap cleanup_collectors EXIT
trap 'exit 130' INT TERM

start_collectors() {
  local seed_base="$1" phase="$2" i seed
  COLLECTOR_PIDS=()
  for ((i=0; i<COLLECTOR_PROCESSES; i++)); do
    seed=$((seed_base + i))
    "${PYTHON_BIN}" -m metamon.rl.taurosv1b_online \
      --run_config "${RUN_CONFIG}" --mode collect \
      --save_dir "${SAVE_DIR}" --buffer_dir "${BUFFER_DIR}" \
      --base_weights "${LATEST}" --epochs 1000000 --seed "${seed}" \
      > >(tee -a "${LOG_DIR}/collector-${phase}-${i}.log") 2>&1 &
    COLLECTOR_PIDS+=("$!")
  done
}

fifo_count() {
  find "${BUFFER_DIR}/gen1ou" -type f \( -name '*.json' -o -name '*.json.lz4' \) | wc -l
}

current="$(fifo_count)"
if (( current < PREFILL_FILES )); then
  log "Prefilling online FIFO: ${current}/${PREFILL_FILES}"
  start_collectors 0 prefill
  while (( current < PREFILL_FILES )); do
    sleep 30
    for pid in "${COLLECTOR_PIDS[@]}"; do
      kill -0 "${pid}" 2>/dev/null || die "A prefill collector exited; inspect ${LOG_DIR}/collector-prefill-*.log"
    done
    current="$(fifo_count)"
    echo "FIFO prefill: ${current}/${PREFILL_FILES}"
  done
  cleanup_collectors
else
  log "FIFO already contains ${current} files; skipping prefill"
fi

start_collectors 1000 train

log "Launching SmallG1OnlineV1a learner with W&B run ${WANDB_PROJECT}/${WANDB_RUN_ID}"
set +e
"${PYTHON_BIN}" -m metamon.rl.taurosv1b_online \
  --run_config "${RUN_CONFIG}" --mode learn \
  --save_dir "${SAVE_DIR}" --buffer_dir "${BUFFER_DIR}" --log \
  "${LEARNER_RESUME_ARGS[@]}" \
  2>&1 | tee -a "${LOG_DIR}/learner.log"
learner_status="${PIPESTATUS[0]}"
set -e
cleanup_collectors
trap - EXIT INT TERM

(( learner_status == 0 )) || die "Learner exited with status ${learner_status}; inspect ${LOG_DIR}/learner.log"
log "SmallG1OnlineV1a training completed"
