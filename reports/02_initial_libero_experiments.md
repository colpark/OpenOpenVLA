# Chapter 2: Initial LIBERO Experiments (Pre-trained Model)

**Summary**: Zero-shot evaluation of OpenVLA on LIBERO failed due to distribution mismatch, action statistics differences, and control frequency gaps.

---

## 2.1 Experiment Goal

Evaluate pre-trained OpenVLA-7B on LIBERO benchmark tasks **without fine-tuning** (zero-shot transfer).

**Hypothesis**: OpenVLA's generalist training on Open X-Embodiment might enable reasonable zero-shot performance on LIBERO.

---

## 2.2 Setup

- **Model**: `openvla/openvla-7b` from HuggingFace
- **Benchmark**: LIBERO-Spatial (10 tasks, 50 demos each)
- **Evaluation**: Closed-loop rollouts in MuJoCo simulation
- **Metrics**: Task success rate, action quality

---

## 2.3 Results: Zero-Shot Performance was Poor

| Metric | Expected (Paper) | Observed |
|--------|------------------|----------|
| Success Rate | 70-80% (after fine-tuning) | ~0-10% |
| Behavior | Task-directed | Random-looking |
| Gripper Control | Purposeful | Erratic |

The robot would move randomly, often not even approaching the target objects.

---

## 2.4 Analysis: Why Zero-Shot Failed

### Reason 1: OpenVLA was NOT Trained on LIBERO

The OpenVLA paper's reported 70-80% success rate on LIBERO requires **fine-tuning on LIBERO demonstration data**. The base model was trained on Open X-Embodiment datasets (Bridge V2, Fractal, etc.) which have different:

- **Robot embodiments**: WidowX (Bridge) vs Franka (LIBERO)
- **Scene configurations**: Kitchen scenes vs tabletop manipulation
- **Task distributions**: Different object sets and goals
- **Action statistics**: Different normalization ranges

### Reason 2: Action Statistics Mismatch

OpenVLA uses dataset-specific normalization statistics stored in `model.config.norm_stats`:

```python
Available keys:
- bridge_orig
- fractal20220817_data
- nyu_franka_play_dataset_converted_externally_to_rlds
# ... and others
```

None of these match LIBERO's Franka robot exactly. Using `bridge_orig` or `nyu_franka_play` gave slightly different action scales, but neither was correct.

### Reason 3: Control Frequency Mismatch (Critical Discovery)

| Dataset | Control Frequency | Per-Step Action Size |
|---------|------------------|---------------------|
| **Fractal (Google RT-1)** | 3 Hz | Large |
| **Bridge V2** | 5 Hz | Medium |
| **DROID** | 15 Hz | Small |
| **LIBERO** | **20 Hz** | **Very Small** |

OpenVLA was trained primarily on 3-5 Hz data. LIBERO runs at 20 Hz, meaning:
- Actions are **4-7x smaller** per step
- The same physical motion requires 4-7x more timesteps
- Model interprets small LIBERO actions as "near-zero movement"

This explains why the model seemed to predict "do nothing" most of the time.

### Reason 4: Image Preprocessing Uncertainty

LIBERO images require specific preprocessing:

1. **180° rotation** (images are upside-down from camera mount)
2. **JPEG encode/decode** (matches OpenVLA training augmentation)
3. **Resize to 224×224** with Lanczos interpolation

```python
def preprocess_libero_image(obs, key='agentview_image'):
    image = obs[key]

    # Rotate 180 degrees (LIBERO convention)
    image = np.rot90(image, k=2)

    # Convert to PIL
    pil_image = Image.fromarray(image.astype(np.uint8))

    # JPEG encode/decode (matches training)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    pil_image = Image.open(buffer)

    # Resize
    pil_image = pil_image.resize((224, 224), Image.LANCZOS)

    return pil_image
```

Without documentation, we initially had uncertainty about the exact preprocessing pipeline.

---

## 2.5 Attempted Workarounds

We tried several zero-shot workarounds:

### 1. Action Scaling (5-10x amplification)

```python
action[:6] = action[:6] * 5.0  # Amplify position/rotation
```

**Result**: Helped marginally but still poor performance. Robot moved more but not purposefully.

### 2. Different `unnorm_key` Values

Tried:
- `bridge_orig`
- `nyu_franka_play_dataset_converted_externally_to_rlds`
- `fractal20220817_data`

**Result**: Marginal impact on action scales, no significant improvement.

### 3. Gripper Inversion

```python
# OpenVLA: 1 = close, LIBERO: -1 = close
action[-1] = -action[-1]
```

**Result**: Necessary for correct gripper behavior but not sufficient for task success.

---

## 2.6 Conclusion

**Zero-shot transfer to LIBERO is not viable.** Fine-tuning on LIBERO demonstration data is required.

The key insights:
1. OpenVLA's generalist training doesn't extend to unseen robot embodiments
2. Control frequency mismatch is a fundamental issue
3. Action statistics must match the target domain

---

## Related Files

- `tutorials/notebooks/07_libero_setup.ipynb` - LIBERO environment configuration
- `tutorials/notebooks/08_integrated_evaluation.ipynb` - Zero-shot evaluation attempts

---

[← Previous: Environment Setup](01_environment_setup.md) | [Back to Index](00_README.md) | [Next: Bridge V2 Validation →](03_bridge_v2_validation.md)
