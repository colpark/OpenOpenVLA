# OpenVLA Understanding Tutorial

A step-by-step tutorial series for understanding OpenVLA (Vision-Language-Action models) from the ground up.

## Overview

This tutorial series covers:
1. Core concepts of Vision-Language-Action models
2. OpenVLA architecture and components
3. Running inference on sample data
4. Integrating with LIBERO simulation for evaluation

## Target Setup

- **Remote server** with 4× NVIDIA GPUs (40GB each, e.g., A100-40GB)
- **Python 3.10+**
- **CUDA 12.x** compatible environment

## Notebooks

| # | Notebook | Description | Time |
|---|----------|-------------|------|
| 01 | [Environment Setup](notebooks/01_environment_setup.ipynb) | Install dependencies, verify GPU setup | 15 min |
| 02 | [Architecture Overview](notebooks/02_architecture_overview.ipynb) | High-level OpenVLA structure | 30 min |
| 03 | [Vision Backbone Deep Dive](notebooks/03_vision_backbone_deep_dive.ipynb) | DINOv2 + SigLIP vision encoders | 45 min |
| 04 | [Action Tokenization](notebooks/04_action_tokenization.ipynb) | How continuous actions become tokens | 30 min |
| 05 | [Data Pipeline](notebooks/05_data_pipeline.ipynb) | RLDS format and data loading | 30 min |
| 06 | [Basic Inference](notebooks/06_basic_inference.ipynb) | Run OpenVLA on sample images | 30 min |
| 07 | [LIBERO Setup](notebooks/07_libero_setup.ipynb) | Configure simulation environment | 30 min |
| 08 | [Integrated Evaluation](notebooks/08_integrated_evaluation.ipynb) | Full OpenVLA + LIBERO evaluation | 60 min |

**Total estimated time: ~4-5 hours**

## Quick Start

```bash
# 1. Clone repository (if not already done)
git clone https://github.com/openvla/openvla.git
cd openvla

# 2. Create conda environment
conda create -n openvla python=3.10 -y
conda activate openvla

# 3. Install PyTorch with CUDA
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Install dependencies
pip install transformers==4.40.1 accelerate timm==0.9.10
pip install flash-attn==2.5.5 --no-build-isolation

# 5. Install LIBERO (for simulation)
pip install mujoco==2.3.7 robosuite==1.4.1 libero

# 6. Install OpenVLA
pip install -e .

# 7. Start Jupyter
jupyter notebook tutorials/notebooks/
```

## Learning Path

### Conceptual Understanding (Notebooks 1-5)
Start here if you want to understand **how OpenVLA works**:
- Architecture and design decisions
- Vision processing pipeline
- Action tokenization strategy
- Data handling

### Practical Usage (Notebooks 6-8)
Jump here if you want to **run OpenVLA immediately**:
- Load model and run inference
- Set up simulation environment
- Evaluate on LIBERO tasks

## Key Concepts

### OpenVLA Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     OpenVLA-7B                           │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [Image 224×224]    [Instruction Text]                  │
│        │                   │                             │
│        ▼                   │                             │
│  ┌──────────────┐          │                            │
│  │ Vision       │          │                            │
│  │ (DINOv2 +    │          │                            │
│  │  SigLIP)     │          │                            │
│  └──────┬───────┘          │                            │
│         │                  │                            │
│         ▼                  ▼                            │
│  ┌──────────────┐   ┌──────────────┐                   │
│  │  Projector   │   │  Tokenizer   │                   │
│  └──────┬───────┘   └──────┬───────┘                   │
│         │                  │                            │
│         └────────┬─────────┘                            │
│                  ▼                                       │
│         ┌──────────────┐                                │
│         │   Llama-2    │                                │
│         │     7B       │                                │
│         └──────┬───────┘                                │
│                │                                         │
│                ▼                                         │
│    [7 Action Tokens] → Un-discretize → [Robot Action]   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Action Space

OpenVLA outputs 7-DoF actions:
- **Position**: x, y, z (end-effector displacement)
- **Rotation**: roll, pitch, yaw (orientation change)
- **Gripper**: open/close

Actions are **discretized to 256 bins** and generated as tokens.

## GPU Memory Requirements

| Configuration | Memory | Notes |
|---------------|--------|-------|
| BF16 (default) | ~14 GB | Recommended for inference |
| 8-bit quantized | ~7 GB | Slight accuracy loss |
| 4-bit quantized | ~4 GB | Larger accuracy loss |

With your 4×40GB GPUs:
- Run **4 parallel model instances** for faster evaluation
- Or use **model parallelism** for larger batch sizes

## Troubleshooting

### CUDA Out of Memory
```python
# Use 8-bit quantization
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    load_in_8bit=True,
    trust_remote_code=True,
)
```

### MuJoCo Rendering on Headless Server
```bash
export MUJOCO_GL=osmesa  # CPU rendering
# or
export MUJOCO_GL=egl     # GPU rendering (faster)
```

### Flash Attention Build Fails
```bash
pip install flash-attn --no-build-isolation
```

## Resources

- **OpenVLA Paper**: [arXiv](https://arxiv.org/abs/2406.09246)
- **Model Weights**: [HuggingFace](https://huggingface.co/openvla/openvla-7b)
- **LIBERO Benchmark**: [Project Page](https://libero-project.github.io/)
- **Original Repository**: [GitHub](https://github.com/openvla/openvla)

## Citation

```bibtex
@article{kim2024openvla,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and others},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```
