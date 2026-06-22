#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-flowbp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$HOME/miniconda3"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "conda was not found. Install Miniconda/Anaconda or source conda.sh first." >&2
    exit 1
fi

# `conda activate` is a shell function; ensure it is available in non-interactive shells.
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel packaging ninja

# Install the torch stack before source-built extensions such as flash-attn.
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# flash-attn imports torch during build, so build isolation often fails.
python -m pip install flash_attn==2.7.4.post1 --no-build-isolation

python -m pip install -r requirements.txt

# requirements.txt is the pinned paper environment. Avoid letting the shorter
# dependency list in pyproject.toml re-resolve or overwrite that environment.
python -m pip install -e . --no-deps

echo "FlowBP setup complete."
echo "Activate the environment with: conda activate $ENV_NAME"
