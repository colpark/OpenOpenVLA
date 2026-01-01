# Chapter 4: LIBERO Fine-tuning Results

**Summary**: Fine-tuning on LIBERO showed partial success - training loss decreased and L1 error improved, but direction accuracy degraded due to control frequency mismatch causing mode collapse.

---

## 4.1 Fine-tuning Setup

### Base Configuration

- **Base Model**: OpenVLA-7B (`openvla/openvla-7b`)
- **Method**: LoRA (Low-Rank Adaptation)
- **Data**: LIBERO-Spatial (10 tasks × 50 demos × ~200 steps)
- **Hardware**: Single A100 40GB GPU

### LoRA Configuration

```python
LoRA rank (r): 32
LoRA alpha: 32
LoRA dropout: 0.1
Target modules: q_proj, k_proj, v_proj, o_proj
```

LoRA adds ~0.5GB of trainable parameters while keeping the base 7B model frozen.

### Training Configuration

```python
Learning rate: 2e-5
Batch size: 4
Gradient accumulation: 4 (effective batch = 16)
Epochs: 3-5
Gradient checkpointing: Enabled
Precision: bfloat16
Max steps: -1 (full epochs)
Save steps: 500
```

### Memory Usage (40GB GPU)

| Component | Memory |
|-----------|--------|
| Base model (bf16) | ~14 GB |
| LoRA adapters | ~0.5 GB |
| Optimizer states | ~3 GB |
| Activations (with checkpointing) | ~5-8 GB |
| **Total** | **~23 GB** |

---

## 4.2 Data Preparation

### LIBERO Action Transform

```python
def transform_libero_action(action):
    """Apply official LIBERO action transform."""
    action = action.astype(np.float32)

    # Position/rotation (dims 0-5): clip to [-1, 1]
    action[:6] = np.clip(action[:6], -1.0, 1.0)

    # Gripper (dim 6): clip to [0, 1] then INVERT
    # Raw LIBERO: -1 = open, +1 = close
    # After transform: +1 = open, 0 = close (OpenVLA convention)
    gripper = np.clip(action[6], 0.0, 1.0)
    action[6] = 1.0 - gripper

    return action
```

### Image Preprocessing

```python
def preprocess_image(image):
    # 180° rotation (LIBERO camera is upside-down)
    image = np.rot90(image, k=2)

    # Resize to 224x224
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((224, 224), Image.LANCZOS)

    return pil_image
```

---

## 4.3 Training Loss Progression

The training loss showed clear learning:

| Stage | Loss | Interpretation |
|-------|------|----------------|
| Step 0 | ~8-10 | Random predictions |
| Step 100 | ~4-6 | Learning action distribution |
| Step 500 | ~2-4 | Learning task-specific actions |
| Step 1000 | ~1-2 | Fine-tuning |
| Final | ~0.5-1.5 | Converged |

**Observation**: Training loss decreased as expected, indicating the model was learning the LIBERO action distribution.

---

## 4.4 Evaluation Results: Mixed Success

### Evaluation Setup

- **Samples**: ~2,500 validation samples (last 5 demos per task, 50 timesteps each)
- **Comparison**: Base OpenVLA-7B vs Fine-tuned (with LoRA adapters)
- **Method**: Open-loop prediction from ground truth images

### Positive Results

| Metric | Base Model | Fine-tuned | Change |
|--------|------------|------------|--------|
| L1 Error | 0.150 | **0.107** | -28.7% ✅ |
| Gripper Accuracy | 79.2% | **86.0%** | +6.8% ✅ |

The model learned to predict more accurate actions overall and improved gripper control.

### Problematic Results

| Metric | Base Model | Fine-tuned | Change |
|--------|------------|------------|--------|
| Direction Accuracy | 50.9% | **24.7%** | -26.2% ❌ |

**Critical Issue**: Direction accuracy dropped **below random** (24.7% vs 50% random baseline).

---

## 4.5 Direction Accuracy Degradation Analysis

### The Problem

Direction accuracy measures whether the model predicts the correct sign (direction) of movement:
- If ground truth says "move right" (+x), does the model predict +x?
- 50% = random guessing
- 24.7% = **worse than random** (systematically wrong)

### Diagnostic Findings

**Action Magnitude Analysis (Position dims 0-2):**

| Measurement | Ground Truth | Fine-tuned Prediction |
|-------------|--------------|----------------------|
| Mean magnitude | 0.089 | 0.043 |
| Std magnitude | 0.124 | 0.067 |
| % near zero (\|a\| < 0.05) | **50.1%** | **65.2%** |

### Key Finding: Mode Collapse

The fine-tuned model predicts **65.2% near-zero actions** while ground truth only has **50.1%** near-zero.

**This is mode collapse** - the model regresses toward predicting "no movement" because:
1. Near-zero is a "safe" prediction that minimizes L1 error
2. Small actions are hard to distinguish from zero
3. The model learns the mean of the action distribution

### Per-Dimension Analysis

| Dimension | Base Dir Acc | Fine-tuned Dir Acc | Inverted % |
|-----------|--------------|-------------------|------------|
| dx | 51.2% | 26.3% | 28.1% |
| dy | 50.8% | 24.1% | 25.8% |
| dz | 50.7% | 23.8% | 27.4% |

All position dimensions show similar degradation - not a single axis issue.

---

## 4.6 Root Cause: Frame Rate Mismatch

The fundamental problem is **control frequency mismatch**:

| Dataset | Control Frequency | Per-Step Magnitude |
|---------|------------------|-------------------|
| Fractal (OpenVLA training) | 3 Hz | Large |
| Bridge V2 (OpenVLA training) | 5 Hz | Medium |
| **LIBERO** (fine-tuning) | **20 Hz** | **Small (4-7x smaller)** |

### Why This Causes Mode Collapse

1. OpenVLA learned "normal" action magnitudes from 3-5 Hz data
2. At 5 Hz, a 10 cm/s movement = 0.02 m/step = relatively large action
3. At 20 Hz, the same 10 cm/s movement = 0.005 m/step = tiny action
4. LIBERO's tiny actions look like "near-zero" to the model
5. The model collapses to predicting minimal movement

### Supporting Evidence

The OpenVLA team reported that **DROID (15 Hz) was problematic** during training:

> "We experimented with incorporating the DROID dataset into their training mixture at a conservative mixture weight of 10%. In practice, we found that the action token accuracy on DROID remained low throughout training... we removed DROID from the data mixture for the final third of training."

If 15 Hz was problematic, 20 Hz is even worse.

---

## 4.7 Recommended Fixes (Not Yet Implemented)

### Option 1: Temporal Subsampling (Recommended)

Use every 4th frame from LIBERO to effectively convert 20 Hz → 5 Hz:

```python
# Instead of:
for t in range(len(demo)):
    sample = demo[t]

# Use:
for t in range(0, len(demo), 4):  # Every 4th frame
    sample = demo[t]
```

**Pros**: Simple, matches training distribution exactly
**Cons**: Loses temporal resolution, may miss fast movements

### Option 2: Action Scaling

Multiply LIBERO actions by 4x during training, divide by 4x at inference:

```python
# Training
action_scaled = action * 4.0

# Inference
predicted_action = model.predict(...) / 4.0
```

**Pros**: Preserves all data, maintains temporal resolution
**Cons**: May cause clipping issues at boundaries

### Option 3: Longer Fine-tuning

More epochs may help the model recalibrate action magnitudes:
- Current: 3-5 epochs
- Try: 10-20 epochs
- Risk: May just overfit without addressing root cause

### Option 4: Lower Learning Rate

Slower adaptation may prevent mode collapse:
- Current: 2e-5
- Try: 5e-6 or 1e-6
- Risk: May not learn at all

---

## 4.8 Partial Success Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Training convergence | ✅ Success | Loss decreased as expected |
| L1 Error improvement | ✅ Success | 28.7% reduction |
| Gripper accuracy | ✅ Success | 6.8% improvement |
| Direction accuracy | ❌ Failed | Mode collapse to near-zero |
| **Overall** | ⚠️ **Partial** | Needs frame rate fix |

---

## 4.9 PEFT Evaluation Gotcha

**Important**: PEFT modifies the base model in-place. To compare base vs fine-tuned:

```python
from peft import PeftModel

# Load fine-tuned model
model = PeftModel.from_pretrained(base_model, checkpoint_path)

# Get TRUE base model behavior (disable LoRA)
model.disable_adapter_layers()
base_predictions = model.predict(...)

# Get fine-tuned behavior (re-enable LoRA)
model.enable_adapter_layers()
finetuned_predictions = model.predict(...)
```

Without `disable_adapter_layers()`, you'll compare fine-tuned vs fine-tuned!

---

## Related Files

- `tutorials/scripts/finetune_openvla_libero.py` - Fine-tuning script
- `tutorials/notebooks/14_evaluate_finetuned_interpretable.ipynb` - Evaluation notebook
- `tutorials/LIBERO_FINETUNING_GUIDE.md` - Comprehensive guide

---

[← Previous: Bridge V2 Validation](03_bridge_v2_validation.md) | [Back to Index](00_README.md) | [Next: Key Findings →](05_key_findings.md)
