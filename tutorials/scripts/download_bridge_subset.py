#!/usr/bin/env python
"""
Download a small subset of Bridge V2 data for pipeline validation.

Bridge V2 is part of OpenVLA's original training data.
This script downloads and caches a small subset for testing.

Usage:
    pip install tensorflow-datasets gcsfs
    python download_bridge_subset.py --num-samples 50
"""

import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Configuration
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
BRIDGE_CACHE_FILE = f"{CACHE_DIR}/bridge_v2_samples.pkl"


def check_dependencies():
    """Check if required packages are installed."""
    missing = []

    try:
        import tensorflow_datasets
    except ImportError:
        missing.append("tensorflow-datasets")

    try:
        import gcsfs
    except ImportError:
        missing.append("gcsfs")

    try:
        import tensorflow
    except ImportError:
        missing.append("tensorflow")

    if missing:
        print("Missing dependencies. Please install:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def download_bridge_subset(num_samples=50, num_episodes=20):
    """Download Bridge V2 samples from Google Cloud Storage."""
    import tensorflow_datasets as tfds

    print(f"Downloading Bridge V2 data ({num_samples} samples from {num_episodes} episodes)...")
    print("This may take a few minutes on first run...")

    try:
        # Load Bridge V2 dataset
        # Bridge is part of the Open X-Embodiment collection
        builder = tfds.builder_from_directory(
            builder_dir="gs://gresearch/robotics/bridge/0.1.0"
        )

        print(f"Dataset info: {builder.info.description[:200]}...")

        dataset = builder.as_dataset(split="train")

        samples = []
        episodes_processed = 0
        seen_instructions = set()  # Track unique instructions for diversity

        # Process more episodes to ensure diversity (at least 2x what we need)
        max_episodes = max(num_episodes * 3, num_samples)

        for episode in tqdm(dataset, desc="Processing episodes", total=max_episodes):
            if episodes_processed >= max_episodes or len(samples) >= num_samples:
                break

            # Debug: print episode-level keys on first episode
            if episodes_processed == 0:
                print(f"\n[DEBUG] Episode keys: {list(episode.keys())}")

            steps = list(episode['steps'])
            if len(steps) < 10:
                continue

            # Get episode-level language instruction FIRST
            episode_instruction = None
            if 'language_instruction' in episode:
                lang_inst = episode['language_instruction']
                if hasattr(lang_inst, 'numpy'):
                    episode_instruction = lang_inst.numpy()
                else:
                    episode_instruction = lang_inst
                if isinstance(episode_instruction, bytes):
                    episode_instruction = episode_instruction.decode('utf-8')
            # Also check in the first step (some datasets store it there)
            if episode_instruction is None and len(steps) > 0:
                first_step = steps[0]
                if 'language_instruction' in first_step:
                    lang_inst = first_step['language_instruction']
                    if hasattr(lang_inst, 'numpy'):
                        episode_instruction = lang_inst.numpy()
                    else:
                        episode_instruction = lang_inst
                    if isinstance(episode_instruction, bytes):
                        episode_instruction = episode_instruction.decode('utf-8')

            if episodes_processed == 0:
                print(f"[DEBUG] Episode instruction: {episode_instruction}")

            # Sample only 1-2 frames per episode to ensure instruction diversity
            n_steps = len(steps)
            # Get frames from middle (most action) - limit per episode for diversity
            indices = [n_steps // 3, 2 * n_steps // 3]  # Only 2 samples per episode

            for idx in indices:
                if len(samples) >= num_samples:
                    break

                step = steps[idx]

                # Debug: print structure on first sample
                if len(samples) == 0:
                    print(f"[DEBUG] Step keys: {step.keys()}")
                    print(f"[DEBUG] Observation keys: {step['observation'].keys()}")
                    if 'action' in step:
                        action_val = step['action']
                        if isinstance(action_val, dict):
                            print(f"[DEBUG] Action is dict with keys: {action_val.keys()}")
                        else:
                            print(f"[DEBUG] Action type: {type(action_val)}")

                # Extract image - handle nested observation structure
                obs = step['observation']
                if 'image' in obs:
                    image_data = obs['image']
                elif 'image_0' in obs:
                    image_data = obs['image_0']
                else:
                    # Find first image key
                    img_keys = [k for k in obs.keys() if 'image' in k.lower()]
                    if img_keys:
                        image_data = obs[img_keys[0]]
                    else:
                        print(f"[WARN] No image found in observation, skipping")
                        continue

                # Convert to numpy
                if hasattr(image_data, 'numpy'):
                    image = image_data.numpy()
                else:
                    image = np.array(image_data)

                # Extract action - handle dict or tensor format
                action_data = step['action']
                if isinstance(action_data, dict):
                    # Bridge V2 uses dict format with separate components
                    # Common keys: 'world_vector', 'rotation_delta', 'gripper_closedness_action'
                    action_parts = []
                    if 'world_vector' in action_data:
                        wv = action_data['world_vector']
                        if hasattr(wv, 'numpy'):
                            wv = wv.numpy()
                        action_parts.extend(wv.flatten()[:3])  # x, y, z
                    if 'rotation_delta' in action_data:
                        rd = action_data['rotation_delta']
                        if hasattr(rd, 'numpy'):
                            rd = rd.numpy()
                        action_parts.extend(rd.flatten()[:3])  # roll, pitch, yaw
                    if 'gripper_closedness_action' in action_data:
                        gc = action_data['gripper_closedness_action']
                        if hasattr(gc, 'numpy'):
                            gc = gc.numpy()
                        action_parts.append(float(gc.flatten()[0]))  # gripper
                    elif 'open_gripper' in action_data:
                        og = action_data['open_gripper']
                        if hasattr(og, 'numpy'):
                            og = og.numpy()
                        action_parts.append(float(og.flatten()[0]))

                    action = np.array(action_parts, dtype=np.float32)
                else:
                    # Tensor format
                    if hasattr(action_data, 'numpy'):
                        action = action_data.numpy()
                    else:
                        action = np.array(action_data)

                # Get language instruction - check observation first (Bridge V2 stores it there)
                instruction = None

                # Bridge V2: instruction is in observation['natural_language_instruction']
                if 'natural_language_instruction' in obs:
                    lang_inst = obs['natural_language_instruction']
                    if hasattr(lang_inst, 'numpy'):
                        instruction = lang_inst.numpy()
                    else:
                        instruction = lang_inst
                    if isinstance(instruction, bytes):
                        instruction = instruction.decode('utf-8')

                # Fallback: episode-level instruction
                if not instruction and episode_instruction:
                    instruction = episode_instruction

                # Fallback: step-level instruction
                if not instruction:
                    lang_inst = step.get('language_instruction', None)
                    if lang_inst is not None:
                        if hasattr(lang_inst, 'numpy'):
                            instruction = lang_inst.numpy()
                        else:
                            instruction = lang_inst
                        if isinstance(instruction, bytes):
                            instruction = instruction.decode('utf-8')

                # Skip samples without real instructions
                if not instruction or instruction == "":
                    if len(samples) == 0:
                        print(f"[WARN] No instruction found, using fallback")
                    instruction = "manipulate the object"

                # Ensure action is 7-dim
                if len(action) < 7:
                    action = np.pad(action, (0, 7 - len(action)))
                else:
                    action = action[:7]

                # Filter: prefer samples with non-trivial actions
                action_magnitude = np.abs(action[:6]).max()  # Ignore gripper for this check
                if action_magnitude < 0.01 and len(samples) < num_samples // 2:
                    # For first half, skip near-zero actions to get diversity
                    if len(samples) == 0:
                        print(f"[DEBUG] Skipping near-zero action: max={action_magnitude:.6f}")
                    continue

                samples.append({
                    'image': image,  # Keep as numpy for serialization
                    'action': action.astype(np.float32),
                    'instruction': instruction,
                    'episode_idx': episodes_processed,
                    'step_idx': idx,
                })

                # Track instruction diversity
                seen_instructions.add(instruction)

                if len(samples) == 1:
                    print(f"[DEBUG] First sample action: {action[:4]}...")
                    print(f"[DEBUG] First sample instruction: {instruction[:50]}...")

            episodes_processed += 1

            if len(samples) >= num_samples:
                break

        print(f"\nDownloaded {len(samples)} samples from {episodes_processed} episodes")
        print(f"Unique instructions: {len(seen_instructions)}")
        if len(seen_instructions) <= 5:
            for instr in seen_instructions:
                print(f"  - {instr[:60]}...")
        return samples

    except Exception as e:
        print(f"Error downloading Bridge V2: {e}")
        import traceback
        traceback.print_exc()
        return None


def save_samples(samples, filepath):
    """Save samples to pickle file."""
    # Convert images to uint8 for storage
    for s in samples:
        if isinstance(s['image'], np.ndarray):
            s['image'] = s['image'].astype(np.uint8)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(samples, f)

    file_size = os.path.getsize(filepath) / (1024 * 1024)
    print(f"Saved {len(samples)} samples to {filepath} ({file_size:.1f} MB)")


def load_samples(filepath):
    """Load samples from pickle file."""
    with open(filepath, 'rb') as f:
        samples = pickle.load(f)

    # Convert images back to PIL
    for s in samples:
        if isinstance(s['image'], np.ndarray):
            s['image'] = Image.fromarray(s['image'])

    return samples


def verify_samples(samples):
    """Verify the downloaded samples are valid."""
    print("\nVerifying samples...")

    issues = []

    for i, s in enumerate(samples[:5]):
        # Check image
        img = s['image']
        if isinstance(img, np.ndarray):
            img_shape = img.shape
        else:
            img_shape = np.array(img).shape

        if len(img_shape) != 3 or img_shape[2] != 3:
            issues.append(f"Sample {i}: Invalid image shape {img_shape}")

        # Check action
        action = s['action']
        if len(action) != 7:
            issues.append(f"Sample {i}: Invalid action length {len(action)}")

        if np.any(np.abs(action) > 2):
            issues.append(f"Sample {i}: Action values out of range: {action}")

        # Check instruction
        instruction = s['instruction']
        if not isinstance(instruction, str) or len(instruction) < 3:
            issues.append(f"Sample {i}: Invalid instruction: {instruction}")

    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("All samples valid!")

        # Show some examples
        print("\nSample examples:")
        for i, s in enumerate(samples[:3]):
            img = s['image']
            if isinstance(img, np.ndarray):
                img_shape = img.shape
            else:
                img_shape = np.array(img).shape
            print(f"  {i+1}. Image: {img_shape}")
            print(f"      Action: {s['action'][:4]}...")
            print(f"      Instruction: {s['instruction'][:50]}...")

        return True


def main():
    parser = argparse.ArgumentParser(description="Download Bridge V2 subset")
    parser.add_argument("--num-samples", type=int, default=50,
                        help="Number of samples to download")
    parser.add_argument("--num-episodes", type=int, default=20,
                        help="Number of episodes to process")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cache exists")
    parser.add_argument("--output", type=str, default=BRIDGE_CACHE_FILE,
                        help="Output file path")
    args = parser.parse_args()

    print("=" * 60)
    print(" Bridge V2 Data Downloader")
    print(" (OpenVLA Original Training Data)")
    print("=" * 60)

    # Check for existing cache
    if os.path.exists(args.output) and not args.force:
        print(f"\nCache file exists: {args.output}")
        samples = load_samples(args.output)
        print(f"Loaded {len(samples)} cached samples")
        verify_samples(samples)
        print("\nTo re-download, use --force flag")
        return

    # Check dependencies
    if not check_dependencies():
        return

    # Download samples
    samples = download_bridge_subset(args.num_samples, args.num_episodes)

    if samples is None or len(samples) == 0:
        print("\nFailed to download samples.")
        print("Possible causes:")
        print("  1. Network connectivity to Google Cloud Storage")
        print("  2. Missing gcsfs package (pip install gcsfs)")
        print("  3. Dataset location changed")
        return

    # Verify samples
    if not verify_samples(samples):
        print("\nWarning: Some samples may be invalid")

    # Save to cache
    save_samples(samples, args.output)

    print("\n" + "=" * 60)
    print(" Download Complete!")
    print("=" * 60)
    print(f"\nCached at: {args.output}")
    print(f"Samples: {len(samples)}")
    print("\nYou can now run the debug pipeline:")
    print("  python tutorials/scripts/debug_pipeline.py")


if __name__ == "__main__":
    main()
