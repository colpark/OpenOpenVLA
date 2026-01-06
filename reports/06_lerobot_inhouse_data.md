# Chapter 6: LeRobot In-House Data Analysis

**Summary**: Analysis of in-house LeRobot datasets collected on Trossen AI Mobile robot, and strategies for connecting this data to OpenVLA fine-tuning.

---

## 6.1 Data Overview

### Robot Platform

| Property | Value |
|----------|-------|
| **Robot Type** | Trossen AI Mobile |
| **Configuration** | Bimanual (dual 7-DOF arms) + mobile base |
| **Control Frequency** | 30 Hz |
| **Camera Setup** | 3 views (high, left wrist, right wrist) |
| **Image Resolution** | 640 × 480 RGB |
| **Video Codec** | AV1 |

### Dataset Summary

| Dataset | Episodes | Frames | Duration (approx) |
|---------|----------|--------|-------------------|
| **ball2** | 25 | 11,090 | ~6.2 min |
| **ball2_groot** | 25 | 11,090 | ~6.2 min |
| **plug_stacking_v1** | 3 | 7,840 | ~4.4 min |
| **plug_stacking_v2** | 6 | 15,718 | ~8.7 min |
| **plug_stacking_v3** | 17 | 44,274 | ~24.6 min |
| **plug_stacking_v4** | 10 | 25,654 | ~14.3 min |
| **plug_stacking_v5** | 20 | 52,063 | ~28.9 min |
| **Total** | **106** | **167,729** | **~93 min** |

**Total Dataset Size**: 3.9 GB (excluding zip)

---

## 6.2 Data Structure

### LeRobot v2.1 Format

```
dataset_name/
├── meta/
│   ├── info.json          # Dataset metadata
│   ├── stats.json         # Action/state statistics
│   └── modality.json      # Data modality descriptions
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.images.cam_left_wrist/
        └── observation.images.cam_right_wrist/
```

### Parquet File Contents

Each parquet file contains one episode with columns:

| Column | Shape | Description |
|--------|-------|-------------|
| `action` | (16,) | Mobile base + bimanual arm commands |
| `observation.state` | (19,) | Odometry + velocity + joint positions |
| `timestamp` | scalar | Frame timestamp (seconds) |
| `frame_index` | scalar | Index within episode |
| `episode_index` | scalar | Episode number |
| `index` | scalar | Global frame index |
| `task_index` | scalar | Task identifier (currently single task) |

### Action Space (16-dimensional)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `linear_vel` | Mobile base linear velocity |
| 1 | `angular_vel` | Mobile base angular velocity |
| 2-8 | `left_joint_0-6` | Left arm joint positions (7 DOF) |
| 9-15 | `right_joint_0-6` | Right arm joint positions (7 DOF) |

### Observation State (19-dimensional)

| Index | Name | Description |
|-------|------|-------------|
| 0-2 | `odom_x, odom_y, odom_theta` | Base odometry |
| 3-4 | `linear_vel, angular_vel` | Base velocities |
| 5-11 | `left_joint_0-6` | Left arm positions |
| 12-18 | `right_joint_0-6` | Right arm positions |

### Camera Views

| Camera | Resolution | FPS | Description |
|--------|------------|-----|-------------|
| `cam_high` | 640×480 | 30 | Third-person overhead view |
| `cam_left_wrist` | 640×480 | 30 | Left gripper eye-in-hand |
| `cam_right_wrist` | 640×480 | 30 | Right gripper eye-in-hand |

---

## 6.3 Action Statistics (ball2 dataset)

### Mean and Standard Deviation

| Action | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| linear_vel | -0.28 | 0.25 | -0.78 | 0.00 |
| angular_vel | 1.70 | 0.50 | 0.63 | 2.59 |
| left_joint_0 | 0.90 | 0.42 | 0.43 | 1.91 |
| left_joint_1 | 0.19 | 0.19 | -0.76 | 0.63 |
| left_joint_2 | 0.28 | 0.17 | -0.05 | 0.85 |
| left_joint_3 | 0.12 | 0.09 | -0.03 | 0.50 |
| left_joint_4 | 0.02 | 0.01 | 0.00 | 0.04 |
| left_joint_5 | 0.44 | 0.23 | 0.00 | 0.74 |
| left_joint_6 | 1.59 | 0.57 | 0.67 | 2.65 |
| right_joint_0 | 1.08 | 0.43 | 0.43 | 2.07 |
| right_joint_1 | -0.11 | 0.28 | -0.96 | 0.63 |
| right_joint_2 | -0.15 | 0.13 | -0.53 | 0.21 |
| right_joint_3 | -0.12 | 0.09 | -0.39 | 0.08 |
| right_joint_4 | 0.02 | 0.01 | 0.00 | 0.03 |
| right_joint_5 | ~0 | ~0 | -0.04 | 0.03 |
| right_joint_6 | ~0 | ~0 | -0.08 | 0.08 |

---

## 6.4 Challenges for OpenVLA Integration

### 1. Action Dimension Mismatch

| Model | Action Dim | Format |
|-------|-----------|--------|
| **OpenVLA** | 7 | [dx, dy, dz, rx, ry, rz, gripper] |
| **LeRobot (ours)** | 16 | [base_lin, base_ang, 7×left, 7×right] |

**Problem**: OpenVLA expects 7-dimensional end-effector actions, not joint-space actions.

### 2. Control Frequency Gap

| Dataset | Hz | Notes |
|---------|-----|-------|
| OpenVLA training (Fractal) | 3 | Primary training data |
| OpenVLA training (Bridge V2) | 5 | Primary training data |
| LIBERO | 20 | Caused mode collapse |
| **Our data** | **30** | Even larger gap |

**Problem**: 30 Hz is 6-10x faster than OpenVLA's training data.

### 3. Missing Language Annotations

The `modality.json` shows:
```json
"language": {}
```

**Problem**: OpenVLA requires natural language task instructions for training.

### 4. Bimanual vs Single-Arm

OpenVLA was trained on single-arm manipulators. Our robot has:
- Mobile base (2 DOF)
- Left arm (7 DOF)
- Right arm (7 DOF)

**Problem**: No bimanual coordination in OpenVLA's training distribution.

---

## 6.5 Integration Strategies

### Strategy 1: Single-Arm Extraction (Recommended Start)

Extract single-arm demonstrations from bimanual data:

```python
def extract_single_arm(action, arm='right'):
    """Extract single arm action from bimanual data."""
    if arm == 'left':
        # left_joint_0-6 → indices 2-8
        joint_positions = action[2:9]
    else:
        # right_joint_0-6 → indices 9-15
        joint_positions = action[9:16]

    # Need forward kinematics to convert to end-effector delta
    # This requires robot URDF
    ee_delta = forward_kinematics_delta(joint_positions)

    return ee_delta  # (7,): [dx, dy, dz, rx, ry, rz, gripper]
```

**Pros**: Matches OpenVLA's expected input format
**Cons**: Loses bimanual coordination, requires FK computation

### Strategy 2: Action Space Adaptation

Train with modified action tokenization:

```python
# Option A: Train separate models for left/right
# Option B: Expand OpenVLA's action vocabulary (requires architecture change)
# Option C: Use action chunking with separate tokens per arm
```

### Strategy 3: Frame Rate Matching

Apply temporal subsampling (same as LIBERO fix):

```python
# 30 Hz → 5 Hz (match Bridge V2)
chunk_size = 6  # Use every 6th frame

for t in range(0, len(episode), chunk_size):
    sample = episode[t]
```

**Effective frequency**: 30 / 6 = 5 Hz

### Strategy 4: Add Language Annotations

Create task descriptions for each dataset:

```python
TASK_DESCRIPTIONS = {
    'ball2': 'pick up the ball and place it in the target location',
    'plug_stacking_v1': 'stack the plug on the base',
    'plug_stacking_v2': 'stack the plug on the base',
    # ... etc
}
```

---

## 6.6 Recommended Pipeline

### Phase 1: Data Preparation

1. **Add language annotations** to each task
2. **Extract single arm** (start with right arm + high camera)
3. **Apply temporal subsampling** (6x for 30→5 Hz)
4. **Convert to OpenVLA format** (similar to LIBERO preprocessing)

### Phase 2: Action Space Conversion

Convert joint-space actions to end-effector deltas:

```python
def joint_to_ee_delta(joint_positions, robot_urdf):
    """
    Convert joint positions to end-effector delta actions.

    Requires:
    1. Robot URDF for forward kinematics
    2. Previous frame's EE position
    3. Velocity limits for normalization
    """
    # Load robot model
    robot = load_urdf(robot_urdf)

    # Compute EE pose
    ee_pose = robot.forward_kinematics(joint_positions)

    # Compute delta from previous frame
    ee_delta = ee_pose - previous_ee_pose

    # Normalize to [-1, 1] range
    ee_delta_normalized = normalize_action(ee_delta)

    # Add gripper (need to identify gripper joint)
    gripper = extract_gripper_state(joint_positions)

    return np.concatenate([ee_delta_normalized, [gripper]])
```

### Phase 3: Fine-tuning

Use the chunked fine-tuning script with modifications:

```bash
python finetune_openvla_lerobot.py \
    --data-dir inhouse/lerobot_dataset \
    --task ball2 \
    --chunk-size 6 \
    --camera cam_high \
    --arm right
```

---

## 6.7 Data Conversion Script Outline

```python
# lerobot_to_openvla.py (to be implemented)

class LeRobotToOpenVLA:
    def __init__(self, data_dir, robot_urdf, chunk_size=6, arm='right'):
        self.data_dir = data_dir
        self.robot = load_urdf(robot_urdf)
        self.chunk_size = chunk_size
        self.arm = arm

    def convert_episode(self, episode_path):
        # Load parquet
        df = pd.read_parquet(episode_path)

        # Load corresponding video frames
        video_path = self._get_video_path(episode_path, 'cam_high')
        frames = load_video_frames(video_path)

        samples = []
        for t in range(0, len(df), self.chunk_size):
            row = df.iloc[t]

            # Extract single arm action
            joint_action = self._extract_arm_action(row['action'])

            # Convert to EE delta
            ee_delta = self._joint_to_ee_delta(joint_action)

            # Get image
            image = frames[t]

            samples.append({
                'image': image,
                'action': ee_delta,
                'language': self.task_description,
            })

        return samples
```

---

## 6.8 Next Steps

| Priority | Task | Complexity |
|----------|------|------------|
| **High** | Add language annotations | Low |
| **High** | Implement temporal subsampling | Low |
| **Medium** | Obtain Trossen robot URDF | Medium |
| **Medium** | Implement FK-based action conversion | Medium |
| **Low** | Explore bimanual-specific architectures | High |

### Quick Start (Minimal Viable Pipeline)

1. Start with `ball2` dataset (simplest task)
2. Use `cam_high` camera only
3. Apply 6x temporal subsampling (30→5 Hz)
4. Initially: use raw joint actions (accept dimension mismatch)
5. Later: implement proper EE delta conversion

---

## 6.9 File Reference

```
inhouse/
├── lerobot_datasets.zip          # Compressed backup (3.9 GB)
└── lerobot_dataset/
    └── lerobot/recorded_data/
        ├── ball2/                 # Ball manipulation task
        │   ├── meta/
        │   │   ├── info.json
        │   │   ├── stats.json
        │   │   └── modality.json
        │   ├── data/chunk-000/
        │   └── videos/chunk-000/
        ├── ball2_groot/           # Same task, different format
        └── plug_stacking_data/
            ├── plug_stacking_v1/  # 3 episodes
            ├── plug_stacking_v2/  # 6 episodes
            ├── plug_stacking_v3/  # 17 episodes
            ├── plug_stacking_v4/  # 10 episodes
            └── plug_stacking_v5/  # 20 episodes
```

---

## 6.10 Key Differences: LeRobot vs LIBERO

| Aspect | LIBERO | LeRobot (ours) |
|--------|--------|----------------|
| Robot | Franka Panda (single arm) | Trossen AI Mobile (bimanual) |
| Action space | 7D end-effector delta | 16D joint positions |
| Control freq | 20 Hz | 30 Hz |
| Language | Per-task instructions | None (need to add) |
| Simulation | MuJoCo | Real robot |
| Camera | Single view | 3 views |

---

*Report generated: January 2025*

---

[← Previous: Key Findings](05_key_findings.md) | [Back to Index](00_README.md)
