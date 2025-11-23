#!/usr/bin/env bash
# One-click environment setup for X2DFD (creates 'X2DFD' conda env)
# Usage: bash install.sh

set -euo pipefail

ENV_NAME="X2DFD"

echo "==> Checking for conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found. Please install Miniconda/Anaconda first:"
  echo "  https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

# Enable 'conda activate' in non-interactive shells
eval "$(conda shell.bash hook)"

echo "==> Creating conda env '${ENV_NAME}' (Python 3.10)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[INFO] Environment ${ENV_NAME} already exists. Skipping creation."
else
  conda create -y -n "${ENV_NAME}" python=3.10
fi

echo "==> Activating env"
conda activate "${ENV_NAME}"

echo "==> Upgrading pip tooling"
python -m pip install -U pip setuptools wheel

# Install PyTorch (GPU if available)
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> Detected NVIDIA GPU. Installing PyTorch with CUDA 12.1 via conda."
  conda install -y -n "${ENV_NAME}" -c pytorch -c nvidia \
    pytorch torchvision torchaudio pytorch-cuda=12.1 || {
      echo "[WARN] conda CUDA install failed; falling back to pip (cu121)."
      python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
    }
else
  echo "==> No GPU detected. Installing CPU-only PyTorch via pip."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
fi

echo "==> Installing Python dependencies via pip (pinned)"
# Core LLM stack
python -m pip install \
  transformers==4.37.2 \
  accelerate==0.28.0 \
  peft==0.10.0 \
  sentencepiece==0.1.99 \
  tokenizers==0.15.1 \
  safetensors==0.4.5

# Optional but widely used
python -m pip install bitsandbytes==0.45.5 || echo "[WARN] bitsandbytes install failed; proceeding without it."

# Vision + utils
python -m pip install \
  timm==0.6.13 \
  opencv-python==4.9.0.80 \
  numpy==1.26.4 \
  Pillow==10.4.0 \
  PyYAML==6.0.2 \
  tqdm==4.67.1 \
  einops==0.7.0

# LLaVA from network (prefer PyPI, fallback to GitHub)
echo "==> Installing LLaVA (prefer PyPI, fallback to GitHub)"
if ! python -m pip install llava==1.2.2.post1; then
  echo "[WARN] PyPI install failed; trying GitHub source..."
  if ! python -m pip install "git+https://github.com/haotian-liu/LLaVA.git#egg=llava"; then
    echo "[ERROR] Failed to install LLaVA from both PyPI and GitHub." >&2
    exit 2
  fi
fi

# Training tooling (optional)
python -m pip install deepspeed==0.12.6 || echo "[WARN] deepspeed install failed; training with deepspeed will be unavailable."

echo "==> Done. Activate the environment with:"
echo "    conda activate ${ENV_NAME}"
echo "==> To run quick tests:"
echo "    bash test.sh"
