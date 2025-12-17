#!/bin/bash
#SBATCH -A <your_account>
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH -o finetune_%j.out
#SBATCH -e finetune_%j.err

# ============================================================
# OpenVLA Fine-tuning on NERSC Perlmutter
# ============================================================

# Exit on error
set -e

# Load modules
module load pytorch/2.0.1

# Set environment
export PSCRATCH="/pscratch/sd/d/dpark1"  # CHANGE THIS
export HF_HOME="${PSCRATCH}/.cache/huggingface"
export TORCH_HOME="${PSCRATCH}/.cache/torch"

# Suppress TensorFlow warnings
export TF_CPP_MIN_LOG_LEVEL=3

# NCCL settings for better multi-GPU performance
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn

# Navigate to script directory
cd ${PSCRATCH}/openvla/tutorials/scripts

# ============================================================
# Option 1: Single GPU (for testing)
# ============================================================
# python finetune_openvla_distributed.py --single-gpu --max-samples 1000

# ============================================================
# Option 2: Multi-GPU with accelerate (RECOMMENDED)
# ============================================================
accelerate launch \
    --config_file accelerate_config.yaml \
    --num_processes 4 \
    finetune_openvla_distributed.py \
    --batch-size 4 \
    --gradient-accumulation-steps 8 \
    --num-epochs 10 \
    --learning-rate 2e-5 \
    --save-steps 500

# ============================================================
# Option 3: DeepSpeed ZeRO-2 (for even better memory efficiency)
# ============================================================
# accelerate launch \
#     --config_file deepspeed_config.yaml \
#     finetune_openvla_distributed.py \
#     --batch-size 8 \
#     --gradient-accumulation-steps 4

echo "Training complete!"
