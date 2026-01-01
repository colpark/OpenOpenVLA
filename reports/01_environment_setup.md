# Chapter 1: Environment Setup and Challenges

**Summary**: Setting up the OpenVLA environment required resolving critical dependency conflicts and configuring HPC-specific settings.

---

## 1.1 Target Environment

- **Platform**: SciServer HPC (Johns Hopkins University)
- **Hardware**: 4× NVIDIA A100 40GB GPUs
- **Software Stack**: Python 3.10+, PyTorch 2.2.0, CUDA 12.1

---

## 1.2 Critical Dependency Conflicts

The OpenVLA model has very specific version requirements that caused significant debugging effort:

| Package | Required Version | Issue with Wrong Version |
|---------|-----------------|--------------------------|
| `transformers` | **4.40.1** (exact) | Model architecture incompatible with newer versions |
| `tokenizers` | **0.19.1** (exact) | Must match transformers version |
| `timm` | **0.9.x** (0.9.10-0.9.16) | `timm.models.layers` module removed in 1.0+ |
| `protobuf` | **< 5.0** | TensorFlow 2.15 incompatible with protobuf 5.x |

### The `timm` Version Problem

The most persistent issue was with the `timm` package:

```
ImportError: No module named 'timm.models.layers'
```

**Root Cause**: OpenVLA's vision backbone (DINOv2 + SigLIP) relies on internal `timm` modules that were restructured in version 1.0.0.

**Solution**: Pin `timm==0.9.16` before importing any OpenVLA code:
```bash
pip install timm==0.9.16
```

### TensorFlow/Protobuf Conflicts

When loading Bridge V2 data via TensorFlow Datasets:

```
TypeError: Descriptors cannot be created directly
```

**Root Cause**: TensorFlow 2.9.x with protobuf 5.x has breaking API changes.

**Solution**: Use compatible stack:
```bash
pip install tensorflow==2.15.1 protobuf>=3.20,<5
```

---

## 1.3 Cache Directory Configuration

HPC environments require explicit cache configuration to avoid home directory quotas:

```python
import os

# CRITICAL: Set BEFORE any imports
SCRATCH = os.environ.get('SCRATCH', '/home/idies/workspace/Temporary/user/scratch')
CACHE_DIR = f"{SCRATCH}/.cache"

os.environ['XDG_CACHE_HOME'] = CACHE_DIR
os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"
os.environ['TRANSFORMERS_CACHE'] = f"{CACHE_DIR}/huggingface/transformers"
os.environ['TORCH_HOME'] = f"{CACHE_DIR}/torch"
```

**Why This Matters**:
- HuggingFace models are 14GB+ and will fill home directory
- TensorFlow datasets cache can be 100GB+
- PyTorch pretrained weights add several GB more

---

## 1.4 MuJoCo/LIBERO Rendering

For headless GPU servers without displays:

```python
os.environ['MUJOCO_GL'] = 'osmesa'  # or 'egl' for GPU rendering
```

Required system packages:
```bash
sudo apt-get install -y xvfb libgl1-mesa-glx libosmesa6-dev
```

**Options**:
- `osmesa`: CPU-based software rendering (slower but always works)
- `egl`: GPU-accelerated headless rendering (faster, requires proper drivers)

---

## 1.5 Complete Requirements File

```txt
# Core (EXACT versions required)
transformers==4.40.1
tokenizers==0.19.1
timm==0.9.16

# PyTorch
torch>=2.0.0
torchvision

# Fine-tuning
peft>=0.7.0
accelerate>=0.25.0

# Data (for Bridge V2)
tensorflow==2.15.1
tensorflow-datasets>=4.8.0
protobuf>=3.20,<5

# LIBERO
mujoco==2.3.7
robosuite==1.4.1
libero

# Utilities
numpy>=1.23,<2
pillow
h5py
tqdm
matplotlib
```

---

## 1.6 Lessons Learned

1. **Version pinning is critical** - Create a requirements.txt with exact versions
2. **Set environment variables first** - Before any Python imports
3. **Test imports incrementally** - Verify each component independently
4. **Check CUDA compatibility** - Ensure PyTorch CUDA version matches system
5. **Read error messages carefully** - `timm.models.layers` error always means wrong timm version

---

## Related Files

- `tutorials/notebooks/01_environment_setup.ipynb` - Interactive setup guide
- `tutorials/LIBERO_FINETUNING_GUIDE.md` - Comprehensive installation instructions

---

[← Back to Index](00_README.md) | [Next: Initial LIBERO Experiments →](02_initial_libero_experiments.md)
