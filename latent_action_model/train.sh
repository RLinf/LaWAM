REPO_ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH}:${REPO_ROOT_DIR}"
cd "${REPO_ROOT_DIR}" || exit 1

HF_CACHE_BASE="$(dirname "$(dirname "${REPO_ROOT_DIR}")")/.hf_cache"
export HF_HOME="${HF_CACHE_BASE}"
export HF_DATASETS_CACHE="${HF_CACHE_BASE}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_BASE}/hub"

# NOTE:
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export TORCH_NUM_THREADS=1
export TORCH_INTRAOP_THREADS=1
export TORCH_INTEROP_THREADS=1
export KMP_BLOCKTIME=0
# export LAM_WORKER_OMP_THREADS=2
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_DISABLE_STATS=true
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_TIMEOUT=720000




# CONFIG_FILE="config/lam-vjepa_large.yaml"
DEFAULT_CONFIG_FILE="${REPO_ROOT_DIR}/latent_action_model/config/dino_base_ae.yaml"
TIMESTAMP="$(date +%m%d_%H%M%S)"
LOG_DIR="${REPO_ROOT_DIR}/latent_action_model/logs/train_logs/${TIMESTAMP}"

LOG_FILE="${LOG_DIR}/train_logs.log"

CKPT_PATH="${CKPT_PATH:-}"

export WANDB_API_KEY="${WANDB_API_KEY:?set this yourself, or use WANDB_MODE=disabled}"
export WANDB_DIR="${REPO_ROOT_DIR}/latent_action_model"

# export WANDB_MODE="offline"

HAS_USER_CONFIG=false
for arg in "$@"; do
    if [[ "$arg" == "--config" ]] || [[ "$arg" == --config=* ]]; then
        HAS_USER_CONFIG=true
        break
    fi
done

USER_CONFIG_FILE=""
if [[ "$HAS_USER_CONFIG" == true ]]; then
    prev_is_config=false
    for arg in "$@"; do
        if [[ "$prev_is_config" == true ]]; then
            USER_CONFIG_FILE="$arg"
            prev_is_config=false
            continue
        fi
        if [[ "$arg" == "--config" ]]; then
            prev_is_config=true
            continue
        fi
        if [[ "$arg" == --config=* ]]; then
            USER_CONFIG_FILE="${arg#--config=}"
        fi
    done
fi

if [[ "$HAS_USER_CONFIG" == true ]]; then
    CONFIG_CLI=""
    CONFIG_SHOWN="${USER_CONFIG_FILE:-command-line --config provided}"
else
    CONFIG_CLI="--config ${DEFAULT_CONFIG_FILE}"
    CONFIG_SHOWN="${DEFAULT_CONFIG_FILE}"
fi

echo "🚀 Starting LAM training..."
echo "📋 Config file: ${CONFIG_SHOWN}"
echo "📝 Log file: ${LOG_FILE}"
echo "🗂️ W&B directory: ${WANDB_DIR}/wandb"
if [[ -n "${CKPT_PATH}" ]]; then
    echo "🔁 Resuming from checkpoint: ${CKPT_PATH}"
fi

mkdir -p "${LOG_DIR}"
mkdir -p "${WANDB_DIR}/wandb"

if command -v realpath &> /dev/null; then
    export LAM_TRAIN_LOG_FILE="$(realpath -m "${LOG_FILE}")"
else
    export LAM_TRAIN_LOG_FILE="${REPO_ROOT_DIR}/${LOG_FILE}"
fi

SOURCE_CONFIG_FILE="${DEFAULT_CONFIG_FILE}"
if [[ "${HAS_USER_CONFIG}" == true && -n "${USER_CONFIG_FILE}" ]]; then
    SOURCE_CONFIG_FILE="${USER_CONFIG_FILE}"
fi

if [[ "${SOURCE_CONFIG_FILE}" = /* ]]; then
    export LAM_CONFIG_PATH="${SOURCE_CONFIG_FILE}"
else
    if command -v realpath &> /dev/null; then
        export LAM_CONFIG_PATH="$(realpath -m "${SOURCE_CONFIG_FILE}")"
    else
        export LAM_CONFIG_PATH="${REPO_ROOT_DIR}/${SOURCE_CONFIG_FILE}"
    fi
fi

if [[ -f "${LAM_CONFIG_PATH}" ]]; then
    cp "${LAM_CONFIG_PATH}" "${LOG_DIR}/$(basename "${LAM_CONFIG_PATH}")"
else
    echo "⚠️ Config file not found: ${LAM_CONFIG_PATH}" >&2
fi

if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
fi
echo "🖥️ Detected GPU count: ${NUM_GPUS}"

torchrun --nproc_per_node ${NUM_GPUS} \
         -m latent_action_model.main fit \
         ${CONFIG_CLI} \
         ${CKPT_PATH:+--ckpt_path ${CKPT_PATH}} \
         "$@" \
         2>&1 | tee -a ${LOG_FILE}

if [ -f "${LOG_FILE}" ]; then
    echo "✅ Training log saved to: ${LOG_FILE}"
    echo "📊 Log file size: $(du -h ${LOG_FILE} | cut -f1)"
else
    echo "⚠️ Warning: log file not found: ${LOG_FILE}"
fi
