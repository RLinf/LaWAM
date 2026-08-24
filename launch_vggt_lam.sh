#!/bin/bash
# LaWAM Stage 1 on VGGT features, LIBERO, 8x GPU DDP.
#
# The paths below were the author's. Override via env before running:
#   REPO_DIR  repo root      (default: this script's own directory)
#   PY        interpreter    (default: python)
#   RUN_DIR   ckpt/log root  (default: $REPO_DIR/lam_runs)
#   NPROC     GPUs           (default: 8)
# Also edit latent_action_model/config/vggt_vae.yaml -- vision_model_id and
# data_root_dir in there still point at the author's local disk.
cd "${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}" || exit 1
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1          # repo lives on a full, no-rename volume
export TMPDIR="${TMPDIR:-/tmp}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/dev/shm/triton_cache_lam}"
export LAM_ENABLE_MANUAL_WANDB=0          # wandb-core cannot start here
export WANDB_MODE=disabled
export LAM_CONFIG_PATH="$PWD/latent_action_model/config/vggt_vae.yaml"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export TF_CPP_MIN_LOG_LEVEL=3
export TORCH_NCCL_BLOCKING_WAIT=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_TIMEOUT=720000

PY="${PY:-python}"
CKPT="${RUN_DIR:-$PWD/lam_runs}/vggt_vae_libero/checkpoints/last.ckpt"
RESUME=""
[ -f "$CKPT" ] && RESUME="--ckpt_path $CKPT"

exec $PY -m torch.distributed.run --nproc_per_node "${NPROC:-8}" \
    -m latent_action_model.main fit \
    --config latent_action_model/config/vggt_vae.yaml \
    $RESUME
