#!/bin/bash
# ============================================================
# OpenVLA Setup Script for SciServer
# 4x 80GB H100 GPUs with CUDA 12.4
# ============================================================

set -e

# ============================================================
# Configuration - HARDCODED PATHS (no variables that might be empty)
# ============================================================
SCRATCH="/home/idies/workspace/Temporary/dpark1/scratch"
CONDA_ENVS="${SCRATCH}/conda/conda_envs"
CONDA_PKGS="${SCRATCH}/conda/conda_pkgs"
CACHE_DIR="${SCRATCH}/.cache"
ENV_NAME="openvla"

echo "============================================================"
echo "OpenVLA Setup for SciServer"
echo "============================================================"
echo "SCRATCH: ${SCRATCH}"
echo "CONDA_ENVS: ${CONDA_ENVS}"
echo "CACHE_DIR: ${CACHE_DIR}"
echo ""

# Verify SCRATCH exists
if [ ! -d "${SCRATCH}" ]; then
    echo "ERROR: SCRATCH directory does not exist: ${SCRATCH}"
    echo "Please create it first: mkdir -p ${SCRATCH}"
    exit 1
fi

# Create directories
mkdir -p "${CONDA_ENVS}"
mkdir -p "${CONDA_PKGS}"
mkdir -p "${CACHE_DIR}/huggingface"
mkdir -p "${CACHE_DIR}/torch"
mkdir -p "${SCRATCH}/libero_data"
mkdir -p "${SCRATCH}/openvla_finetune"

# ============================================================
# Step 1: Configure Conda
# ============================================================
echo "Step 1: Configuring Conda..."

conda config --add pkgs_dirs "${CONDA_PKGS}"
conda config --add envs_dirs "${CONDA_ENVS}"

# ============================================================
# Step 2: Create Environment
# ============================================================
echo "Step 2: Creating conda environment..."

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment ${ENV_NAME} already exists."
else
    echo "Creating new environment: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ============================================================
# Step 3: Install PyTorch (CUDA 12.4)
# ============================================================
echo "Step 3: Installing PyTorch for CUDA 12.4..."

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# ============================================================
# Step 4: Install OpenVLA Dependencies
# ============================================================
echo "Step 4: Installing OpenVLA dependencies..."

pip install transformers>=4.40.0
pip install accelerate>=0.30.0
pip install peft>=0.10.0
pip install bitsandbytes
pip install timm
pip install sentencepiece
pip install protobuf
pip install einops
pip install scipy
pip install pillow
pip install matplotlib
pip install tqdm
pip install h5py
pip install tensorboard
pip install jupyter jupyterlab

# Flash Attention (optional - skip if CUDA_HOME not set)
echo "Checking for Flash Attention prerequisites..."
if [ -n "${CUDA_HOME}" ] && command -v nvcc &> /dev/null; then
    echo "Installing Flash Attention 2..."
    pip install flash-attn --no-build-isolation || echo "Flash Attention installation failed - continuing without it"
else
    echo "CUDA_HOME not set or nvcc not found - skipping Flash Attention"
    echo "Training will work but may be slower on H100s"
fi

# ============================================================
# Step 5: Install LIBERO
# ============================================================
echo "Step 5: Installing LIBERO..."

LIBERO_DIR="${SCRATCH}/LIBERO"

if [ ! -d "${LIBERO_DIR}" ]; then
    echo "Cloning LIBERO repository..."
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
fi

cd "${LIBERO_DIR}"
pip install -e .

# Install robosuite (LIBERO dependency)
pip install robosuite==1.4.1

# ============================================================
# Step 6: Configure LIBERO paths
# ============================================================
echo "Step 6: Configuring LIBERO..."

mkdir -p ~/.libero
cat > ~/.libero/config.yaml << EOF
bddl_files: ${LIBERO_DIR}/libero/libero/bddl_files
datasets: ${SCRATCH}/libero_data
EOF

echo "LIBERO config:"
cat ~/.libero/config.yaml

# ============================================================
# Step 7: Set Environment Variables (in user's bashrc)
# ============================================================
echo "Step 7: Setting environment variables..."

# Add to bashrc for persistence
cat >> ~/.bashrc << EOF

# OpenVLA Environment Variables (added by setup script)
export SCRATCH="${SCRATCH}"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"
export TRANSFORMERS_CACHE="${CACHE_DIR}/huggingface/transformers"
export HF_DATASETS_CACHE="${CACHE_DIR}/huggingface/datasets"
export TF_CPP_MIN_LOG_LEVEL=3
EOF

# Source them now
export SCRATCH="${SCRATCH}"
export HF_HOME="${CACHE_DIR}/huggingface"
export TORCH_HOME="${CACHE_DIR}/torch"
export XDG_CACHE_HOME="${CACHE_DIR}"
export TF_CPP_MIN_LOG_LEVEL=3

echo "Environment variables added to ~/.bashrc"

# ============================================================
# Step 8: Download LIBERO Assets
# ============================================================
echo "Step 8: Downloading LIBERO assets..."

python << PYEOF
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    from libero.libero import benchmark
    print("LIBERO benchmark module loaded successfully!")

    # This will trigger asset download
    bench = benchmark.get_benchmark("libero_spatial")()
    print(f"LIBERO Spatial: {bench.n_tasks} tasks")
except Exception as e:
    print(f"LIBERO asset download may need manual intervention: {e}")
PYEOF

# ============================================================
# Step 9: Download LIBERO Demonstration Data
# ============================================================
echo "Step 9: Downloading LIBERO demonstration data..."

cd "${LIBERO_DIR}"
python benchmark_scripts/download_libero_datasets.py \
    --download-dir "${SCRATCH}/libero_data" \
    --datasets libero_spatial || echo "Demo download failed - you can download manually later"

# ============================================================
# Step 10: Verify Installation
# ============================================================
echo ""
echo "============================================================"
echo "Verification"
echo "============================================================"

python << PYEOF
import sys
print(f"Python: {sys.executable}")

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")

import transformers
print(f"Transformers: {transformers.__version__}")

import accelerate
print(f"Accelerate: {accelerate.__version__}")

try:
    import peft
    print(f"PEFT: {peft.__version__}")
except:
    print("PEFT: Not installed")

try:
    import flash_attn
    print(f"Flash Attention: {flash_attn.__version__}")
except:
    print("Flash Attention: Not installed (optional)")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    from libero.libero import benchmark
    print("LIBERO: Installed")
except Exception as e:
    print(f"LIBERO: Error - {e}")

# Check for demo files
from pathlib import Path
hdf5_files = list(Path("${SCRATCH}/libero_data").rglob("*.hdf5"))
print(f"LIBERO demo files: {len(hdf5_files)} found")
PYEOF

echo ""
echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "To activate environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run tutorials:"
echo "  cd ${SCRATCH}/OpenOpenVLA/tutorials/notebooks"
echo "  jupyter lab"
echo ""
echo "To run fine-tuning:"
echo "  cd ${SCRATCH}/OpenOpenVLA/tutorials/scripts"
echo "  accelerate launch --config_file accelerate_config_sciserver.yaml finetune_openvla_distributed.py"
