# Chapter 7: LeRobot to OpenVLA Adaptation

**Summary**: Implementation of the data conversion pipeline and fine-tuning infrastructure to adapt OpenVLA for the in-house Trossen AI Mobile robot data collected via LeRobot.

---

## 7.1 Adaptation Overview

### Challenge Summary

| Aspect | LeRobot (Source) | OpenVLA (Target) | Gap |
|--------|------------------|------------------|-----|
| **Action Space** | 16-dim joint positions | 7-dim EE deltas | Dimension mismatch |
| **Control Frequency** | 30 Hz | 3-5 Hz | 6-10x frequency gap |
| **Robot Type** | Bimanual mobile | Single arm fixed | Coordination loss |
| **Language** | None | Required | Missing annotations |
| **Image Format** | MP4 (AV1 codec) | RGB arrays | Format conversion |

### Solution Architecture

```
LeRobot v2.1 Data
        │
        ▼
┌───────────────────────┐
│ convert_lerobot_to_   │
│ openvla.py            │
│                       │
│ • Temporal subsampling│
│ • Action extraction   │
│ • Delta computation   │
│ • Language injection  │
│ • HDF5 export         │
└───────────────────────┘
        │
        ▼
  OpenVLA HDF5 Format
        │
        ▼
┌───────────────────────┐
│ finetune_lerobot.py   │
│                       │
│ • LoRA fine-tuning    │
│ • Metric logging      │
│ • Validation          │
│ • Checkpointing       │
└───────────────────────┘
        │
        ▼
  Fine-tuned Model
```

---

## 7.2 Data Conversion Pipeline

### 7.2.1 Temporal Subsampling

**Problem**: 30 Hz control frequency is 6-10x faster than OpenVLA's training data.

**Solution**: Use every 6th frame to match Bridge V2's 5 Hz.

```python
# Subsampling implementation
chunk_size = 6  # 30 Hz → 5 Hz
frame_indices = list(range(0, n_frames, chunk_size))

for t in frame_indices:
    sample = episode[t]
```

**Effective Frequencies**:

| Chunk Size | Original Hz | Effective Hz | Match Target |
|------------|-------------|--------------|--------------|
| 3 | 30 | 10 | High-speed tasks |
| 6 | 30 | 5 | Bridge V2 (recommended) |
| 10 | 30 | 3 | Fractal/RT-1 |

### 7.2.2 Action Space Conversion

**Source**: 16-dimensional joint action vector

```
Index 0-1:   [linear_vel, angular_vel]     # Mobile base (discarded)
Index 2-8:   [left_joint_0, ..., left_joint_6]   # Left arm (7 DOF)
Index 9-15:  [right_joint_0, ..., right_joint_6] # Right arm (7 DOF)
```

**Target**: Two modes supported

| Mode | Action Dims | Output Format |
|------|-------------|---------------|
| **Single arm** | 7 | `[pos(3), rot(3), gripper(1)]` |
| **Bimanual** (recommended) | 14 | `[left_pos(3), left_rot(3), left_grip(1), right_pos(3), right_rot(3), right_grip(1)]` |

**Conversion Strategy**:

1. **Arm Selection**: Choose `left`, `right`, or `both` (bimanual)
2. **Delta Computation**: Calculate frame-to-frame changes
3. **Normalization**: Scale to [-1, 1] using 99th percentile statistics
4. **Joint-to-EE Mapping**: Approximate mapping without FK

```python
class ActionConverter:
    def convert(self, actions, prev_action=None):
        # Extract arm joints
        arm_action = actions[self.joint_indices]  # 7 dims

        if prev_action is not None:
            # Compute delta
            prev_arm_action = prev_action[self.joint_indices]
            action_delta = arm_action - prev_arm_action

            # Normalize using precomputed scale
            action_normalized = action_delta / self.delta_scale

        # Clip to valid range
        action_clipped = np.clip(action_normalized, -1.0, 1.0)

        # Map to OpenVLA format
        openvla_action = np.zeros(7, dtype=np.float32)
        openvla_action[0:3] = action_clipped[0:3]  # Position proxy
        openvla_action[3:6] = action_clipped[3:6]  # Rotation proxy
        openvla_action[6] = action_clipped[6]      # Gripper

        return openvla_action
```

### 7.2.3 Joint-to-EE Approximation

**Without forward kinematics (FK)**, we use an approximate mapping:

| Joint Index | OpenVLA Dim | Rationale |
|-------------|-------------|-----------|
| 0, 1, 2 | dx, dy, dz | Shoulder/upper arm joints primarily affect position |
| 3, 4, 5 | rx, ry, rz | Elbow/wrist joints primarily affect orientation |
| 6 | gripper | Gripper joint directly maps |

**Limitations**:
- Approximation loses precise EE control
- Works best for tasks with simple kinematics
- Future: Implement proper FK with robot URDF

### 7.2.4 Language Annotation

**Task Descriptions** (added automatically):

```python
TASK_DESCRIPTIONS = {
    'ball2': 'pick up the ball and place it in the target location',
    'ball2_groot': 'pick up the ball and place it in the target location',
    'plug_stacking_v1': 'pick up the plug and stack it on the base',
    'plug_stacking_v2': 'pick up the plug and stack it on the base',
    'plug_stacking_v3': 'pick up the plug and stack it on the base',
    'plug_stacking_v4': 'pick up the plug and stack it on the base',
    'plug_stacking_v5': 'pick up the plug and stack it on the base',
}
```

**Prompt Template**:
```
In: What action should the robot take to {language}?
Out: [action tokens]
```

---

## 7.3 Video Processing

### 7.3.1 AV1 Codec Support

LeRobot uses AV1 codec for efficient video compression. Two backends are supported:

```python
# Primary: PyAV (better AV1 support)
import av
container = av.open(video_path)
for frame in container.decode(video=0):
    img = frame.to_ndarray(format='rgb24')

# Fallback: OpenCV
import cv2
cap = cv2.VideoCapture(str(video_path))
ret, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

### 7.3.2 Frame Selection

Only frames at subsampled indices are loaded:

```python
frame_indices = set(range(0, n_frames, chunk_size))
frames = load_video_frames(video_path, frame_indices)
```

This reduces memory usage and processing time by ~6x.

---

## 7.4 Output Format

### 7.4.1 HDF5 Structure

```
{task_name}_{split}.hdf5
├── images:        (N, H, W, 3) uint8
├── actions:       (N, 7) float32
├── episode_idx:   (N,) int32
├── frame_idx:     (N,) int32
└── [attributes]
    ├── language_instruction: str
    ├── task_name: str
    ├── arm: str
    ├── camera: str
    ├── chunk_size: int
    ├── original_fps: int
    └── effective_fps: float
```

### 7.4.2 Metadata JSON

```json
{
  "task_name": "ball2",
  "language": "pick up the ball and place it in the target location",
  "arm": "right",
  "camera": "high",
  "chunk_size": 6,
  "original_fps": 30,
  "effective_fps": 5.0,
  "train_samples": 285,
  "val_samples": 73,
  "action_delta_stats": {
    "delta_mean": [...],
    "delta_std": [...],
    "delta_scale": [...]
  }
}
```

---

## 7.5 Fine-tuning Infrastructure

### 7.5.1 Dataset Class

```python
class LeRobotDataset(Dataset):
    def __init__(self, hdf5_path, processor, action_tokenizer):
        with h5py.File(hdf5_path, 'r') as f:
            self.n_samples = f['images'].shape[0]
            self.language = f.attrs['language_instruction']
            self.actions = f['actions'][:]  # Pre-load actions

    def __getitem__(self, idx):
        # Lazy load images
        with h5py.File(self.hdf5_path, 'r') as f:
            image = f['images'][idx]

        # Process for model
        prompt = f"In: What action should the robot take to {self.language}?\nOut:"
        inputs = self.processor(prompt, image, return_tensors="pt")

        # Tokenize action
        action_tokens = self.action_tokenizer.encode(self.actions[idx])

        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'pixel_values': inputs['pixel_values'],
            'action_tokens': action_tokens,
        }
```

### 7.5.2 LoRA Configuration

```python
lora_config = LoraConfig(
    r=32,                    # LoRA rank
    lora_alpha=64,           # Scaling factor
    target_modules=[         # Attention layers
        "q_proj", "v_proj",
        "k_proj", "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

**Parameter Efficiency**:
- Trainable: ~19M parameters
- Total: ~7.5B parameters
- Ratio: ~0.25%

### 7.5.3 Training Configuration

| Parameter | Default | Notes |
|-----------|---------|-------|
| Learning Rate | 2e-5 | With linear warmup |
| Batch Size | 4 | Limited by GPU memory |
| Epochs | 5 | Task-dependent |
| Warmup Steps | 100 | 10% of total steps |
| Weight Decay | 0.01 | AdamW regularization |
| Gradient Clipping | 1.0 | Prevent instability |

---

## 7.6 Metrics and Evaluation

### 7.6.1 Action Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **L1 Error** | Mean absolute error (all dims) | Lower is better |
| **Position L1** | L1 for dims 0-2 | < 0.2 |
| **Rotation L1** | L1 for dims 3-5 | < 0.3 |
| **Gripper L1** | L1 for dim 6 | < 0.15 |

### 7.6.2 Accuracy Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Direction Accuracy** | Sign match for non-zero GT | > 70% |
| **Gripper Accuracy** | Binary classification (threshold 0.5) | > 85% |

### 7.6.3 Logging

Training logs are saved as CSV:

```csv
epoch,step,loss,l1_error,position_l1,direction_acc,gripper_acc,lr
1,50,0.234,0.312,0.285,0.623,0.756,2.0e-5
1,100,0.198,0.276,0.241,0.678,0.812,1.9e-5
...
```

---

## 7.7 Usage Guide

### 7.7.1 Convert LeRobot Data

```bash
# Bimanual mode (14-dim) - RECOMMENDED for coordinated tasks
python convert_lerobot_to_openvla.py --task ball2 --arm both

# Single arm mode (7-dim)
python convert_lerobot_to_openvla.py --task ball2 --arm right

# Without task-specific language (uses generic "complete the task")
python convert_lerobot_to_openvla.py --task ball2 --arm both --no-language

# Full options
python convert_lerobot_to_openvla.py \
    --data-dir /path/to/lerobot_dataset/lerobot/recorded_data \
    --output-dir /path/to/openvla_converted \
    --task plug_stacking_v3 \
    --arm both \
    --camera high \
    --chunk-size 6 \
    --val-episodes 5 \
    --no-language
```

### 7.7.2 Fine-tune OpenVLA

```bash
# Basic fine-tuning
python finetune_lerobot.py --task ball2 --epochs 5

# Full options
python finetune_lerobot.py \
    --data-dir /path/to/openvla_converted \
    --task ball2 \
    --output-dir results \
    --epochs 10 \
    --batch-size 4 \
    --lr 2e-5 \
    --lora-rank 32
```

### 7.7.3 Expected Output Structure

```
results/
└── ball2_20250107_143022/
    ├── training_log.csv
    ├── results.json
    ├── best/
    │   ├── adapter_config.json
    │   ├── adapter_model.safetensors
    │   └── training_state.pt
    └── checkpoint_epoch5/
        └── ...
```

---

## 7.8 Known Limitations

### 7.8.1 Action Space Approximation

**Issue**: Joint-to-EE mapping is approximate without proper FK.

**Impact**:
- Position commands may not match true EE motion
- Works for demonstration but may need refinement

**Mitigation**:
- Obtain Trossen robot URDF
- Implement proper forward kinematics
- Compute true EE deltas

### 7.8.2 Single Arm Extraction

**Issue**: Bimanual coordination is lost when extracting single arm.

**Impact**:
- Tasks requiring both arms will not transfer
- Mobile base actions are discarded

**Future Work**:
- Explore bimanual VLA architectures
- Multi-arm action tokenization

### 7.8.3 No Real Robot Validation

**Issue**: Conversion validated on data format only, not on real robot.

**Impact**:
- Unknown sim-to-real gap
- Action scaling may need adjustment

**Required**:
- Deploy on Trossen robot
- Tune action normalization
- Measure task success rate

---

## 7.9 Comparison: LIBERO vs LeRobot Adaptation

| Aspect | LIBERO | LeRobot |
|--------|--------|---------|
| **Source Format** | HDF5 | Parquet + MP4 |
| **Action Space** | 7D EE delta | 16D joint positions |
| **Conversion Complexity** | Low | Medium |
| **FK Required** | No | Recommended |
| **Language** | Provided | Added manually |
| **Validation** | Simulation | Real robot (pending) |
| **Chunk Size** | 4 (20→5 Hz) | 6 (30→5 Hz) |

---

## 7.10 Next Steps

| Priority | Task | Complexity | Notes |
|----------|------|------------|-------|
| **High** | Test conversion on ball2 | Low | Verify pipeline works |
| **High** | Run fine-tuning experiment | Medium | GPU required |
| **Medium** | Implement FK conversion | Medium | Need URDF |
| **Medium** | Deploy on real robot | High | Hardware access |
| **Low** | Multi-task training | Medium | Combine all plug tasks |
| **Low** | Camera view ablation | Low | Compare high vs wrist |

---

## 7.11 File Reference

```
tutorials/scripts/
├── convert_lerobot_to_openvla.py    # Data conversion pipeline
└── finetune_lerobot.py              # Fine-tuning script

inhouse/
├── lerobot_dataset/                  # Original LeRobot data
│   └── lerobot/recorded_data/
│       ├── ball2/
│       ├── ball2_groot/
│       └── plug_stacking_data/
│           ├── plug_stacking_v1/
│           └── ...
└── openvla_converted/                # Converted data (after running)
    ├── ball2_train.hdf5
    ├── ball2_val.hdf5
    └── ball2_metadata.json
```

---

*Report generated: January 2025*

---

[← Previous: LeRobot Data Analysis](06_lerobot_inhouse_data.md) | [Back to Index](00_README.md)
