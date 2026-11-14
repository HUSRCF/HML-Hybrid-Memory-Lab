# Source this file from bash: source scripts/activate.sh
SAS_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SAS_STORAGE_ROOT="${SAS_STORAGE_ROOT:-/media/990Pro/Code/SAS}"
if [[ ! -f /home/husrcf/anaconda3/etc/profile.d/conda.sh ]]; then
    echo "Missing existing Conda installation." >&2
    return 1
fi
if [[ -n "${VIRTUAL_ENV:-}" ]] && declare -F deactivate >/dev/null; then
    deactivate
fi
source /home/husrcf/anaconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME="$SAS_STORAGE_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$SAS_STORAGE_ROOT/.cache/datasets"
export TRITON_CACHE_DIR="$SAS_STORAGE_ROOT/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$SAS_STORAGE_ROOT/.cache/torchinductor"
if [[ "${UV_CACHE_DIR:-}" == "$SAS_STORAGE_ROOT/.cache/uv" ]]; then
    unset UV_CACHE_DIR
fi
export TOKENIZERS_PARALLELISM=false
