#!/usr/bin/env bash

set -euo pipefail

readonly MINICONDA_SCRIPT="Miniconda3-latest-Linux-x86_64.sh"
readonly MINICONDA_PREFIX="${HOME}/miniconda3"

if ! command -v conda >/dev/null 2>&1; then
  echo "Downloading Miniconda..."
  curl -fsSLo "${MINICONDA_SCRIPT}" \
    "https://repo.anaconda.com/miniconda/${MINICONDA_SCRIPT}"
  bash "${MINICONDA_SCRIPT}" -b -p "${MINICONDA_PREFIX}"
  rm -f "${MINICONDA_SCRIPT}"
fi

source "${MINICONDA_PREFIX}/etc/profile.d/conda.sh"

conda config --set auto_activate_base false

ENV_NAME="evo2_embeddings"
if ! conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
  conda create -y -n "${ENV_NAME}" python=3.12
fi

conda activate "${ENV_NAME}"

conda install -y -c nvidia cuda-nvcc cuda-cudart-dev
conda install -y -c conda-forge transformer-engine-torch=2.3.0

pip install --upgrade pip
pip install psutil
pip install flash-attn==2.8.0.post2 --no-build-isolation
pip install evo2
pip install -e .
