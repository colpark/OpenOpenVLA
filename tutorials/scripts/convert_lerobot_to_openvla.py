#!/usr/bin/env python
"""
LeRobot to OpenVLA Data Conversion

Converts LeRobot v2.1 format (Trossen AI Mobile) to OpenVLA-compatible format.

Adaptation Strategy:
1. Temporal Subsampling: 30 Hz → 5 Hz (every 6th frame)
2. Action Space Mapping:
   - Single arm: 16-dim → 7-dim (one arm extraction)
   - Both arms: 16-dim → 14-dim (bimanual, preserves coordination)
3. Action Representation: Joint deltas normalized to [-1, 1]
4. Language Annotations: Optional (can use generic or task-specific)

Usage:
    # Single arm mode (7-dim actions)
    python convert_lerobot_to_openvla.py --task ball2 --arm right

    # Bimanual mode (14-dim actions) - RECOMMENDED for coordinated tasks
    python convert_lerobot_to_openvla.py --task plug_stacking_v3 --arm both

    # Without language annotation
    python convert_lerobot_to_openvla.py --task ball2 --arm both --no-language
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm

# Video loading
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Video loading disabled.")

try:
    import av
    HAS_AV = True
except ImportError:
    HAS_AV = False


# =============================================================================
# Configuration
# =============================================================================

# Task language annotations
TASK_DESCRIPTIONS = {
    'ball2': 'pick up the ball and place it in the target location',
    'ball2_groot': 'pick up the ball and place it in the target location',
    'plug_stacking_v1': 'pick up the plug and stack it on the base',
    'plug_stacking_v2': 'pick up the plug and stack it on the base',
    'plug_stacking_v3': 'pick up the plug and stack it on the base',
    'plug_stacking_v4': 'pick up the plug and stack it on the base',
    'plug_stacking_v5': 'pick up the plug and stack it on the base',
}

# Action mapping for Trossen robot
# Action indices: [linear_vel, angular_vel, left_0-6, right_0-6]
ACTION_INDICES = {
    'base': [0, 1],
    'left': [2, 3, 4, 5, 6, 7, 8],
    'right': [9, 10, 11, 12, 13, 14, 15],
    'both': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],  # Both arms (14-dim)
}

# Gripper joint index within arm (last joint)
GRIPPER_JOINT_IDX = 6  # joint_6 is the gripper for single arm
GRIPPER_JOINT_IDX_BIMANUAL = [6, 13]  # Both grippers in bimanual mode

# Default language for when --no-language is used
DEFAULT_LANGUAGE = "complete the task"

# Camera mapping
CAMERA_KEYS = {
    'high': 'observation.images.cam_high',
    'left_wrist': 'observation.images.cam_left_wrist',
    'right_wrist': 'observation.images.cam_right_wrist',
}


# =============================================================================
# Video Loading
# =============================================================================

def load_video_frames_cv2(video_path, frame_indices=None):
    """Load video frames using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_indices is None or frame_idx in frame_indices:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        frame_idx += 1

    cap.release()
    return frames


def load_video_frames_av(video_path, frame_indices=None):
    """Load video frames using PyAV (better for AV1 codec)."""
    container = av.open(str(video_path))
    frames = []

    frame_idx = 0
    for frame in container.decode(video=0):
        if frame_indices is None or frame_idx in frame_indices:
            # Convert to numpy array
            img = frame.to_ndarray(format='rgb24')
            frames.append(img)
        frame_idx += 1

    container.close()
    return frames


def load_video_frames(video_path, frame_indices=None):
    """Load video frames with available backend."""
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Try PyAV first (better for AV1)
    if HAS_AV:
        try:
            return load_video_frames_av(video_path, frame_indices)
        except Exception as e:
            print(f"PyAV failed: {e}, trying OpenCV...")

    # Fallback to OpenCV
    if HAS_CV2:
        return load_video_frames_cv2(video_path, frame_indices)

    raise RuntimeError("No video backend available. Install opencv-python or av.")


# =============================================================================
# Action Conversion
# =============================================================================

class ActionConverter:
    """
    Converts LeRobot 16-dim joint actions to OpenVLA format.

    Modes:
    - Single arm (left/right): 7-dim output [pos(3), rot(3), gripper(1)]
    - Bimanual (both): 14-dim output [left_pos(3), left_rot(3), left_grip(1),
                                       right_pos(3), right_rot(3), right_grip(1)]

    Strategy:
    1. Extract arm joints (7 or 14 dims)
    2. Compute joint deltas (velocity-like)
    3. Normalize to [-1, 1] range

    Note: Without FK, we approximate EE motion from joint deltas.
    Joint indices 0-2 roughly correspond to position, 3-5 to orientation.
    """

    def __init__(self, arm='both', use_deltas=True, stats=None):
        self.arm = arm
        self.use_deltas = use_deltas
        self.joint_indices = ACTION_INDICES[arm]
        self.is_bimanual = (arm == 'both')
        self.action_dim = 14 if self.is_bimanual else 7

        # Normalization statistics
        if stats is not None:
            self.mean = np.array(stats['mean'])[self.joint_indices]
            self.std = np.array(stats['std'])[self.joint_indices]
        else:
            self.mean = None
            self.std = None

        # Delta normalization (computed from data)
        self.delta_scale = None

    def compute_delta_stats(self, all_actions):
        """Compute normalization statistics from action deltas."""
        # Extract arm actions
        arm_actions = all_actions[:, self.joint_indices]

        # Compute deltas
        deltas = np.diff(arm_actions, axis=0)

        # Use percentile-based scaling for robustness
        self.delta_scale = np.percentile(np.abs(deltas), 99, axis=0)
        self.delta_scale = np.maximum(self.delta_scale, 1e-6)  # Avoid division by zero

        return {
            'delta_mean': deltas.mean(axis=0).tolist(),
            'delta_std': deltas.std(axis=0).tolist(),
            'delta_scale': self.delta_scale.tolist(),
            'action_dim': self.action_dim,
            'is_bimanual': self.is_bimanual,
        }

    def convert(self, actions, prev_action=None):
        """
        Convert LeRobot action to OpenVLA format.

        Args:
            actions: Current frame action (16-dim)
            prev_action: Previous frame action for delta computation

        Returns:
            7-dim (single arm) or 14-dim (bimanual) action in [-1, 1] range
        """
        # Extract arm joints
        arm_action = actions[self.joint_indices]

        if self.use_deltas and prev_action is not None:
            prev_arm_action = prev_action[self.joint_indices]
            action_delta = arm_action - prev_arm_action

            # Normalize using delta scale
            if self.delta_scale is not None:
                action_normalized = action_delta / self.delta_scale
            else:
                # Fallback: simple scaling
                action_normalized = action_delta * 10.0  # Amplify small deltas
        else:
            # Use absolute positions (normalized)
            if self.mean is not None and self.std is not None:
                action_normalized = (arm_action - self.mean) / (self.std + 1e-6)
            else:
                action_normalized = arm_action

        # Clip to [-1, 1]
        action_clipped = np.clip(action_normalized, -1.0, 1.0)

        if self.is_bimanual:
            # Bimanual mode: 14-dim output
            # Input: [left_0-6, right_0-6] (14 dims)
            # Output: [left_pos(3), left_rot(3), left_grip(1),
            #          right_pos(3), right_rot(3), right_grip(1)]
            openvla_action = np.zeros(14, dtype=np.float32)

            # Left arm (indices 0-6 in extracted action)
            openvla_action[0:3] = action_clipped[0:3]   # Left position proxy
            openvla_action[3:6] = action_clipped[3:6]   # Left rotation proxy
            openvla_action[6] = action_clipped[6]       # Left gripper

            # Right arm (indices 7-13 in extracted action)
            openvla_action[7:10] = action_clipped[7:10]  # Right position proxy
            openvla_action[10:13] = action_clipped[10:13]  # Right rotation proxy
            openvla_action[13] = action_clipped[13]      # Right gripper
        else:
            # Single arm mode: 7-dim output
            # joints 0-2 → position-like (dx, dy, dz)
            # joints 3-5 → rotation-like (rx, ry, rz)
            # joint 6 → gripper
            openvla_action = np.zeros(7, dtype=np.float32)
            openvla_action[0:3] = action_clipped[0:3]  # Position proxy
            openvla_action[3:6] = action_clipped[3:6]  # Rotation proxy
            openvla_action[6] = action_clipped[6]      # Gripper

        return openvla_action


# =============================================================================
# Dataset Conversion
# =============================================================================

class LeRobotConverter:
    """Main converter class for LeRobot → OpenVLA."""

    def __init__(self, data_dir, output_dir, task_name,
                 arm='both', camera='high', chunk_size=6,
                 val_episodes=5, use_language=True):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.arm = arm
        self.camera = camera
        self.chunk_size = chunk_size
        self.val_episodes = val_episodes
        self.use_language = use_language

        # Find task directory
        self.task_dir = self._find_task_dir()

        # Load metadata
        self.info = self._load_info()
        self.stats = self._load_stats()

        # Language annotation (optional)
        if use_language:
            self.language = TASK_DESCRIPTIONS.get(
                task_name,
                f"complete the {task_name.replace('_', ' ')} task"
            )
        else:
            self.language = DEFAULT_LANGUAGE  # Generic "complete the task"

        # Action converter
        self.action_converter = ActionConverter(
            arm=arm,
            use_deltas=True,
            stats=self.stats.get('action') if self.stats else None
        )

        # Determine action dimension
        self.action_dim = self.action_converter.action_dim
        self.is_bimanual = self.action_converter.is_bimanual

        print(f"Task: {task_name}")
        print(f"Language: '{self.language}'" + (" (generic)" if not use_language else ""))
        print(f"Arm mode: {arm} ({'bimanual 14-dim' if self.is_bimanual else 'single 7-dim'})")
        print(f"Action dimension: {self.action_dim}")
        print(f"Camera: {camera}")
        print(f"Chunk size: {chunk_size} (30 Hz → {30/chunk_size:.1f} Hz)")

    def _find_task_dir(self):
        """Find the task directory."""
        # Direct match
        direct = self.data_dir / self.task_name
        if direct.exists():
            return direct

        # Try under plug_stacking_data
        nested = self.data_dir / "plug_stacking_data" / self.task_name
        if nested.exists():
            return nested

        raise FileNotFoundError(f"Task directory not found: {self.task_name}")

    def _load_info(self):
        """Load dataset info."""
        info_path = self.task_dir / "meta" / "info.json"
        with open(info_path) as f:
            return json.load(f)

    def _load_stats(self):
        """Load dataset statistics."""
        stats_path = self.task_dir / "meta" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                return json.load(f)
        return None

    def _get_video_path(self, episode_idx, camera):
        """Get video path for an episode."""
        camera_key = CAMERA_KEYS.get(camera, camera)
        video_name = f"episode_{episode_idx:06d}.mp4"
        return self.task_dir / "videos" / "chunk-000" / camera_key / video_name

    def _get_parquet_path(self, episode_idx):
        """Get parquet path for an episode."""
        return self.task_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"

    def convert_all(self):
        """Convert all episodes."""
        n_episodes = self.info['total_episodes']
        print(f"\nConverting {n_episodes} episodes...")

        # First pass: compute action delta statistics
        print("Computing action statistics...")
        all_actions = []
        for ep_idx in range(n_episodes):
            parquet_path = self._get_parquet_path(ep_idx)
            if parquet_path.exists():
                df = pd.read_parquet(parquet_path)
                actions = np.stack(df['action'].values)
                all_actions.append(actions)

        if all_actions:
            all_actions = np.concatenate(all_actions, axis=0)
            delta_stats = self.action_converter.compute_delta_stats(all_actions)
            print(f"  Delta scale: {delta_stats['delta_scale'][:3]}...")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Split into train/val
        train_episodes = list(range(n_episodes - self.val_episodes))
        val_episodes = list(range(n_episodes - self.val_episodes, n_episodes))

        # Convert episodes
        train_samples = self._convert_episodes(train_episodes, "train")
        val_samples = self._convert_episodes(val_episodes, "val")

        # Save to HDF5
        self._save_hdf5(train_samples, "train")
        self._save_hdf5(val_samples, "val")

        # Save metadata
        self._save_metadata(train_samples, val_samples, delta_stats)

        print(f"\nConversion complete!")
        print(f"  Train samples: {len(train_samples)}")
        print(f"  Val samples: {len(val_samples)}")
        print(f"  Output: {self.output_dir}")

    def _convert_episodes(self, episode_indices, split):
        """Convert a list of episodes."""
        samples = []

        for ep_idx in tqdm(episode_indices, desc=f"Converting {split}"):
            try:
                ep_samples = self._convert_episode(ep_idx)
                samples.extend(ep_samples)
            except Exception as e:
                print(f"Warning: Failed to convert episode {ep_idx}: {e}")

        return samples

    def _convert_episode(self, episode_idx):
        """Convert a single episode."""
        # Load parquet data
        parquet_path = self._get_parquet_path(episode_idx)
        df = pd.read_parquet(parquet_path)

        # Load video frames
        video_path = self._get_video_path(episode_idx, self.camera)

        # Determine which frames to load (with chunking)
        n_frames = len(df)
        frame_indices = list(range(0, n_frames, self.chunk_size))

        # Load frames
        frames = load_video_frames(video_path, set(frame_indices))

        # Convert samples
        samples = []
        actions = np.stack(df['action'].values)

        prev_action = None
        frame_map_idx = 0

        for t in frame_indices:
            if frame_map_idx >= len(frames):
                break

            # Get image
            image = frames[frame_map_idx]
            frame_map_idx += 1

            # Convert action
            current_action = actions[t]
            openvla_action = self.action_converter.convert(
                current_action,
                prev_action
            )
            prev_action = current_action

            # Skip first frame (no valid delta)
            if t == 0:
                continue

            samples.append({
                'image': image,
                'action': openvla_action,
                'language': self.language,
                'episode_idx': episode_idx,
                'frame_idx': t,
            })

        return samples

    def _save_hdf5(self, samples, split):
        """Save samples to HDF5 file."""
        if not samples:
            return

        output_path = self.output_dir / f"{self.task_name}_{split}.hdf5"

        with h5py.File(output_path, 'w') as f:
            # Create datasets
            n_samples = len(samples)

            # Images
            img_shape = samples[0]['image'].shape
            images = f.create_dataset(
                'images',
                shape=(n_samples,) + img_shape,
                dtype=np.uint8,
                chunks=(1,) + img_shape,
                compression='gzip'
            )

            # Actions (dynamic dimension based on arm mode)
            actions = f.create_dataset(
                'actions',
                shape=(n_samples, self.action_dim),
                dtype=np.float32
            )

            # Metadata
            episode_indices = f.create_dataset('episode_idx', shape=(n_samples,), dtype=np.int32)
            frame_indices = f.create_dataset('frame_idx', shape=(n_samples,), dtype=np.int32)

            # Fill data
            for i, sample in enumerate(tqdm(samples, desc=f"Saving {split}")):
                images[i] = sample['image']
                actions[i] = sample['action']
                episode_indices[i] = sample['episode_idx']
                frame_indices[i] = sample['frame_idx']

            # Store language as attribute
            f.attrs['language_instruction'] = self.language
            f.attrs['task_name'] = self.task_name
            f.attrs['arm'] = self.arm
            f.attrs['is_bimanual'] = self.is_bimanual
            f.attrs['action_dim'] = self.action_dim
            f.attrs['camera'] = self.camera
            f.attrs['chunk_size'] = self.chunk_size
            f.attrs['original_fps'] = 30
            f.attrs['effective_fps'] = 30 / self.chunk_size

        print(f"Saved: {output_path}")

    def _save_metadata(self, train_samples, val_samples, delta_stats):
        """Save conversion metadata."""
        # Action mapping description based on mode
        if self.is_bimanual:
            action_mapping = {
                'dims_0_2': 'left_position_proxy (left joint 0-2)',
                'dims_3_5': 'left_rotation_proxy (left joint 3-5)',
                'dim_6': 'left_gripper (left joint 6)',
                'dims_7_9': 'right_position_proxy (right joint 0-2)',
                'dims_10_12': 'right_rotation_proxy (right joint 3-5)',
                'dim_13': 'right_gripper (right joint 6)',
            }
        else:
            action_mapping = {
                'dims_0_2': 'position_proxy (joint 0-2)',
                'dims_3_5': 'rotation_proxy (joint 3-5)',
                'dim_6': 'gripper (joint 6)',
            }

        metadata = {
            'task_name': self.task_name,
            'language': self.language,
            'use_language': self.use_language,
            'arm': self.arm,
            'is_bimanual': self.is_bimanual,
            'action_dim': self.action_dim,
            'camera': self.camera,
            'chunk_size': self.chunk_size,
            'original_fps': 30,
            'effective_fps': 30 / self.chunk_size,
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'action_delta_stats': delta_stats,
            'conversion_timestamp': datetime.now().isoformat(),
            'action_mapping': action_mapping,
        }

        metadata_path = self.output_dir / f"{self.task_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved: {metadata_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Convert LeRobot data to OpenVLA format')
    parser.add_argument('--data-dir', type=str,
                        default='/Users/davidpark/Documents/Claude/openvla/inhouse/lerobot_dataset/lerobot/recorded_data',
                        help='LeRobot data directory')
    parser.add_argument('--output-dir', type=str,
                        default='/Users/davidpark/Documents/Claude/openvla/inhouse/openvla_converted',
                        help='Output directory')
    parser.add_argument('--task', type=str, default='ball2',
                        help='Task name (ball2, plug_stacking_v1, etc.)')
    parser.add_argument('--arm', type=str, default='both', choices=['left', 'right', 'both'],
                        help='Which arm(s) to use: left, right, or both (bimanual)')
    parser.add_argument('--camera', type=str, default='high',
                        choices=['high', 'left_wrist', 'right_wrist'],
                        help='Camera view to use')
    parser.add_argument('--chunk-size', type=int, default=6,
                        help='Temporal subsampling factor (6 for 30→5 Hz)')
    parser.add_argument('--val-episodes', type=int, default=5,
                        help='Number of validation episodes')
    parser.add_argument('--no-language', action='store_true',
                        help='Use generic language instead of task-specific descriptions')
    args = parser.parse_args()

    print("="*60)
    print(" LeRobot → OpenVLA Conversion")
    print("="*60)

    converter = LeRobotConverter(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        task_name=args.task,
        arm=args.arm,
        camera=args.camera,
        chunk_size=args.chunk_size,
        val_episodes=args.val_episodes,
        use_language=not args.no_language,
    )

    converter.convert_all()


if __name__ == "__main__":
    main()
