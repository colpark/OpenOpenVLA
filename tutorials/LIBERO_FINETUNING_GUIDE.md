# OpenVLA Fine-tuning on LIBERO: Complete Guide

This guide provides step-by-step instructions for fine-tuning OpenVLA on the LIBERO benchmark using SciServer's 40GB GPU nodes.

## Overview

**Goal**: Fine-tune OpenVLA to predict robot actions from images + language instructions on LIBERO tasks.

**Pipeline**:
```
1. Download Data → 2. Prepare/Validate → 3. Fine-tune → 4. Evaluate
```

**Time Estimates**:
- Data download: ~10-30 minutes (depending on suite)
- Data preparation: ~5 minutes
- Fine-tuning: ~2-4 hours per epoch (libero_spatial, 10 tasks)
- Evaluation: ~10-20 minutes

---

## Prerequisites

### Environment Setup

```bash
# Set scratch directory (SciServer)
export SCRATCH="/home/idies/workspace/Temporary/$USER/scratch"

# Create working directories
mkdir -p $SCRATCH/.cache
mkdir -p $SCRATCH/libero_data
mkdir -p $SCRATCH/openvla_checkpoints
```

### Install Dependencies

```bash
# Core dependencies (CRITICAL: exact versions required)
pip install transformers==4.40.1 tokenizers==0.19.1
pip install timm==0.9.10
pip install torch>=2.0.0 torchvision

# Fine-tuning dependencies
pip install peft>=0.7.0          # LoRA
pip install accelerate>=0.25.0   # Training utilities
pip install bitsandbytes>=0.41.0 # 8-bit optimization (optional)

# Data handling
pip install h5py numpy pillow tqdm
pip install huggingface_hub

# Verify installation
python -c "import transformers; print(f'transformers: {transformers.__version__}')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

**Version Requirements Table**:
| Package | Required Version | Why |
|---------|-----------------|-----|
| transformers | 4.40.1 | OpenVLA compatibility |
| tokenizers | 0.19.1 | Matches transformers |
| timm | 0.9.x | Vision backbone compatibility |
| peft | >=0.7.0 | LoRA support |

---

## Step 1: Download LIBERO Data

### Option A: Using Download Script (Recommended)

```bash
cd tutorials/scripts

# Download libero_spatial (smallest, good for testing)
python download_libero_demos.py --suite libero_spatial

# Download specific suite
python download_libero_demos.py --suite libero_object

# Download all suites (large!)
python download_libero_demos.py --suite all

# Explore existing data
python download_libero_demos.py --explore

# Force re-download
python download_libero_demos.py --suite libero_spatial --force
```

### Option B: Manual Download

1. Visit: https://github.com/Lifelong-Robot-Learning/LIBERO
2. Download HDF5 files for your chosen suite
3. Extract to: `$SCRATCH/libero_data/<suite_name>/`

### LIBERO Suites

| Suite | Tasks | Demos/Task | Total Demos | Focus |
|-------|-------|------------|-------------|-------|
| libero_spatial | 10 | 50 | 500 | Spatial arrangements |
| libero_object | 10 | 50 | 500 | Object variations |
| libero_goal | 10 | 50 | 500 | Goal variations |
| libero_90 | 90 | 50 | 4,500 | Full benchmark |

### Verify Download

```bash
python download_libero_demos.py --verify --suite libero_spatial

# Expected output:
# Found 10 task files
# Total demos: 500
# Verification PASSED
```

---

## Step 2: Prepare and Validate Data

### Run Data Preparation Script

```bash
cd tutorials/scripts

# Explore data structure
python prepare_libero_data.py --explore --suite libero_spatial

# Compute action statistics (important for understanding data)
python prepare_libero_data.py --stats --suite libero_spatial

# Validate data for training
python prepare_libero_data.py --validate --suite libero_spatial

# Create training index (faster data loading)
python prepare_libero_data.py --index --suite libero_spatial
```

### Understanding the Data

**HDF5 Structure**:
```
task_file.hdf5
├── data/
│   ├── demo_0/
│   │   ├── obs/
│   │   │   ├── agentview_rgb     # Images: (T, 256, 256, 3)
│   │   │   └── robot0_eef_pos    # End-effector position
│   │   └── actions               # 7-DoF actions: (T, 7)
│   ├── demo_1/
│   ...
└── attrs/
    └── language_instruction      # Task description
```

**Action Format** (7-DoF):
```
actions[0:3]  = delta position (x, y, z)
actions[3:6]  = delta rotation (roll, pitch, yaw)
actions[6]    = gripper command
```

**CRITICAL: LIBERO Action Transform** (must match OpenVLA exactly):
```python
# Position/rotation (dims 0-5): clip to [-1, 1]
action[:6] = np.clip(action[:6], -1.0, 1.0)

# Gripper (dim 6): clip to [0, 1] then INVERT
# Raw LIBERO: -1 = open, +1 = close
# After transform: +1 = open, 0 = close (OpenVLA convention)
gripper = np.clip(action[6], 0.0, 1.0)
action[6] = 1.0 - gripper
```

**Expected Action Statistics** (after transform):
```
Action dim 0-5: values in [-1, 1]
Action dim 6 (grip): values in [0, 1] (1=open, 0=close)
```

---

## Step 3: Fine-tune OpenVLA

### Quick Start (Single GPU)

```bash
cd tutorials/scripts

# Basic fine-tuning with LoRA
python finetune_openvla_libero.py \
    --suite libero_spatial \
    --epochs 3 \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --lora-r 32

# Monitor training
# Checkpoints saved to: $SCRATCH/openvla_checkpoints/libero_spatial_YYYYMMDD_HHMMSS/
```

### Full Configuration Options

```bash
python finetune_openvla_libero.py \
    --suite libero_spatial \
    --data-dir $SCRATCH/libero_data \
    --output-dir $SCRATCH/openvla_checkpoints \
    --model-id openvla/openvla-7b \
    --epochs 5 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --learning-rate 2e-5 \
    --warmup-steps 100 \
    --lora-r 32 \
    --lora-alpha 32 \
    --lora-dropout 0.1 \
    --gradient-checkpointing \
    --bf16 \
    --max-steps -1 \
    --save-steps 500 \
    --eval-steps 100
```

### Memory Configuration

**40GB GPU (SciServer)**:
```bash
# Recommended settings for 40GB
python finetune_openvla_libero.py \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --gradient-checkpointing \
    --bf16

# Memory breakdown:
# - Base model: ~14GB (bf16)
# - LoRA adapters: ~0.5GB
# - Optimizer states: ~3GB
# - Activations: ~5-8GB
# - Total: ~23GB (fits in 40GB)
```

**Limited Memory (24GB)**:
```bash
python finetune_openvla_libero.py \
    --batch-size 2 \
    --gradient-accumulation 8 \
    --gradient-checkpointing \
    --bf16 \
    --lora-r 16  # Smaller LoRA
```

### Training Monitoring

**Expected Loss Progression**:
```
Step 0:     loss ≈ 8-10 (random predictions)
Step 100:   loss ≈ 4-6  (learning action distribution)
Step 500:   loss ≈ 2-4  (learning task-specific actions)
Step 1000:  loss ≈ 1-2  (fine-tuning)
Final:      loss ≈ 0.5-1.5 (converged)
```

**Checkpoints**:
```
$SCRATCH/openvla_checkpoints/libero_spatial_YYYYMMDD_HHMMSS/
├── checkpoint-500/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── trainer_state.json
├── checkpoint-1000/
├── final_model/
└── training_log.txt
```

---

## Step 4: Evaluate Fine-tuned Model

### Using Evaluation Notebook

```bash
# Open Jupyter notebook
jupyter notebook tutorials/notebooks/13_evaluate_finetuned_libero.ipynb
```

### Using Python Script

```python
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import numpy as np

# Load base model
model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True
)

# Load LoRA weights
checkpoint_path = "$SCRATCH/openvla_checkpoints/libero_spatial_.../final_model"
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()

# Run inference
def predict_action(image, instruction):
    """Predict 7-DoF action from image and instruction."""
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    inputs = processor(prompt, image).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=7, do_sample=False)

    # Decode action tokens
    action_tokens = outputs[0, -7:]
    vocab_size = 32000
    discretized = vocab_size - action_tokens.cpu().numpy()
    bins = np.linspace(-1, 1, 256)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    indices = np.clip(discretized - 1, 0, len(bin_centers) - 1)
    action = bin_centers[indices]

    return action
```

### Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| L1 Error | Mean absolute error per action dim | < 0.1 |
| Sign Accuracy | Correct direction of movement | > 80% |
| Gripper Accuracy | Correct open/close prediction | > 90% |

---

## Action Tokenization Deep Dive

### How OpenVLA Encodes Actions

```python
# OpenVLA action tokenization
vocab_size = 32000
n_bins = 256
action_token_start = vocab_size - n_bins  # 31744

# Encoding: continuous → discrete token
def encode_action(action):
    """Convert continuous action to token ID."""
    action = np.clip(action, -1.0, 1.0)
    bins = np.linspace(-1, 1, n_bins)
    discretized = np.digitize(action, bins)  # 1-256
    token_id = vocab_size - discretized      # 31744-31999
    return token_id

# Decoding: token → continuous action
def decode_action(token_id):
    """Convert token ID back to continuous action."""
    bins = np.linspace(-1, 1, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # 255 centers
    discretized = vocab_size - token_id       # 1-256
    index = np.clip(discretized - 1, 0, 254)  # 0-254
    return bin_centers[index]
```

### Token Range
```
Action tokens: [31744, 31999]
Token 31744 → bin 256 → action ≈ +1.0
Token 31999 → bin 1   → action ≈ -1.0
```

---

## Troubleshooting

### Common Issues

**Issue: "No module named 'timm.models.layers'"**
```bash
pip uninstall timm && pip install timm==0.9.10
```

**Issue: CUDA out of memory**
```bash
# Reduce batch size
--batch-size 2 --gradient-accumulation 8

# Enable gradient checkpointing
--gradient-checkpointing

# Use smaller LoRA rank
--lora-r 16
```

**Issue: Loss not decreasing**
- Check learning rate (try 1e-5 to 5e-5)
- Verify data loading (images rotated correctly?)
- Check action normalization (should be in [-1, 1])

**Issue: NaN loss**
```bash
# Disable bf16 if unstable
--no-bf16

# Reduce learning rate
--learning-rate 1e-5
```

**Issue: Slow training**
```bash
# Enable dataloader workers
--num-workers 4

# Use bf16 for faster computation
--bf16

# Increase batch size if memory allows
--batch-size 8
```

### Validation Checks

```python
# Check 1: Action statistics
python prepare_libero_data.py --stats --suite libero_spatial
# All action dims should have reasonable means near 0

# Check 2: Data loading
python -c "
import h5py
with h5py.File('path/to/task.hdf5', 'r') as f:
    demo = f['data/demo_0']
    print('Actions shape:', demo['actions'].shape)
    print('Image shape:', demo['obs/agentview_rgb'].shape)
"

# Check 3: Model loading
python -c "
from transformers import AutoModelForVision2Seq
model = AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b', trust_remote_code=True)
print('Model loaded successfully')
"
```

---

## File Reference

```
tutorials/
├── scripts/
│   ├── download_libero_demos.py   # Step 1: Data download
│   ├── prepare_libero_data.py     # Step 2: Data preparation
│   ├── finetune_openvla_libero.py # Step 3: Fine-tuning
│   └── download_bridge_episodes.py # Bridge V2 evaluation data
├── notebooks/
│   ├── 07_libero_setup.ipynb      # LIBERO environment setup
│   ├── 09_finetuning_openvla.ipynb # Fine-tuning concepts
│   ├── 11_evaluation_with_bridge.ipynb # Bridge evaluation
│   └── 13_evaluate_finetuned_libero.ipynb # Step 4: Evaluation
└── LIBERO_FINETUNING_GUIDE.md     # This guide
```

---

## Next Steps After Fine-tuning

1. **Evaluate on held-out tasks**: Test generalization to unseen LIBERO tasks
2. **Simulate rollouts**: Use LIBERO simulation for actual task execution
3. **Compare suites**: Fine-tune on different LIBERO suites
4. **Hyperparameter tuning**: Experiment with learning rates, LoRA ranks
5. **Full fine-tuning**: If resources allow, try full model fine-tuning

---

## References

- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [LIBERO Benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO)
- [Hugging Face PEFT](https://huggingface.co/docs/peft)
- [OpenVLA Model Hub](https://huggingface.co/openvla/openvla-7b)
