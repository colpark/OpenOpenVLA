#!/usr/bin/env python
"""
Download Bridge V2 Episodes for OpenVLA Evaluation

Downloads episodes from Bridge V2 dataset and caches them locally.
Run this BEFORE the evaluation notebook to avoid download delays.

Usage:
    python download_bridge_episodes.py [--num-episodes 20] [--max-steps 50]

Requirements:
    pip install tensorflow>=2.15.0 tensorflow-datasets gcsfs tqdm

NOTE: If you have TensorFlow 2.9.x with newer protobuf, you'll hit version
conflicts. This script will attempt to fix them automatically.
"""

import os
import sys
import argparse
import pickle
import subprocess

# =============================================================================
# Configuration
# =============================================================================
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')


def install_compatible_stack():
    """Install a compatible set of packages for TFDS/Bridge V2 access.

    The key is to use TensorFlow 2.15.x which has fewer dependency conflicts.
    """
    print("Installing compatible TensorFlow stack...")
    print("  (This may take a few minutes)")

    # Install all at once to let pip resolve dependencies together
    packages = [
        "numpy>=1.23,<2",
        "protobuf>=3.20,<5",
        "tensorflow==2.15.1",
        "tensorflow-datasets>=4.9.4",
        "gcsfs",
        "tqdm",
    ]

    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--upgrade"
    ] + packages)

    print("[OK] Packages installed. Please restart the script.")
    sys.exit(0)


def check_tensorflow_stack():
    """Check if TensorFlow stack is properly installed and compatible."""
    print("Checking TensorFlow stack...")

    issues = []

    # Check NumPy
    try:
        import numpy as np
        np_major = int(np.__version__.split('.')[0])
        if np_major >= 2:
            issues.append(f"NumPy {np.__version__} (need < 2.0)")
        else:
            print(f"  NumPy: {np.__version__} (OK)")
    except ImportError:
        issues.append("NumPy not installed")

    # Check TensorFlow
    try:
        import tensorflow as tf
        tf_version = tf.__version__
        major, minor = map(int, tf_version.split('.')[:2])
        if major != 2 or minor < 13 or minor > 16:
            issues.append(f"TensorFlow {tf_version} (recommend 2.15.x)")
        else:
            print(f"  TensorFlow: {tf_version} (OK)")
    except ImportError:
        issues.append("TensorFlow not installed")
    except Exception as e:
        # Protobuf or other import error
        issues.append(f"TensorFlow import error: {type(e).__name__}")

    # Check protobuf
    try:
        import google.protobuf
        pb_version = google.protobuf.__version__
        pb_major = int(pb_version.split('.')[0])
        if pb_major >= 5:
            issues.append(f"Protobuf {pb_version} (need < 5.0 for TF 2.15)")
        else:
            print(f"  Protobuf: {pb_version} (OK)")
    except ImportError:
        issues.append("Protobuf not installed")

    # Check tensorflow-datasets
    try:
        import tensorflow_datasets
        print(f"  TFDS: {tensorflow_datasets.__version__} (OK)")
    except ImportError:
        issues.append("tensorflow-datasets not installed")
    except Exception:
        issues.append("tensorflow-datasets import error")

    if issues:
        print(f"\n[ISSUES DETECTED]")
        for issue in issues:
            print(f"  - {issue}")
        print("\nInstalling compatible package versions...")
        install_compatible_stack()

    return True


def install_if_missing(package, min_version=None):
    """Install package if not available."""
    try:
        mod = __import__(package)
        if min_version and hasattr(mod, '__version__'):
            # Simple version check
            current = mod.__version__
            print(f"  {package}: {current}")
        return True
    except ImportError:
        spec = f"{package}>={min_version}" if min_version else package
        print(f"Installing {spec}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", spec])
        return True


def download_bridge_episodes(num_episodes=20, max_steps_per_episode=50):
    """Download diverse episodes from Bridge V2.

    Args:
        num_episodes: Number of episodes to download
        max_steps_per_episode: Maximum steps per episode

    Returns:
        List of episode dictionaries with frames, actions, instructions
    """
    import numpy as np
    from tqdm import tqdm
    import tensorflow_datasets as tfds

    print(f"\nDownloading {num_episodes} episodes from Bridge V2...")
    print("Source: gs://gresearch/robotics/bridge/0.1.0")
    print("-" * 60)

    builder = tfds.builder_from_directory(
        builder_dir="gs://gresearch/robotics/bridge/0.1.0"
    )
    dataset = builder.as_dataset(split="train")

    episodes = []
    seen_instructions = set()
    skipped_short = 0
    skipped_no_instruction = 0
    skipped_duplicate = 0

    # Process more episodes than needed to ensure diversity
    max_to_process = num_episodes * 10

    for episode_data in tqdm(dataset, desc="Processing", total=max_to_process):
        if len(episodes) >= num_episodes:
            break

        steps = list(episode_data['steps'])

        # Skip short episodes
        if len(steps) < 15:
            skipped_short += 1
            continue

        # Get instruction
        instruction = None
        first_step = steps[0]
        obs = first_step['observation']

        if 'natural_language_instruction' in obs:
            inst = obs['natural_language_instruction']
            if hasattr(inst, 'numpy'):
                inst = inst.numpy()
            if isinstance(inst, bytes):
                inst = inst.decode('utf-8')
            instruction = inst

        if not instruction or instruction == "":
            skipped_no_instruction += 1
            continue

        # Prefer diverse instructions (skip duplicates until half target)
        inst_key = instruction.lower().strip()[:30]
        if inst_key in seen_instructions and len(episodes) < num_episodes // 2:
            skipped_duplicate += 1
            continue
        seen_instructions.add(inst_key)

        # Extract all frames and actions
        episode = {
            'instruction': instruction,
            'frames': [],
            'actions': [],
            'num_steps': min(len(steps), max_steps_per_episode)
        }

        for step in steps[:max_steps_per_episode]:
            obs = step['observation']

            # Extract image
            if 'image' in obs:
                img_data = obs['image']
            elif 'image_0' in obs:
                img_data = obs['image_0']
            else:
                img_keys = [k for k in obs.keys() if 'image' in k.lower()]
                if img_keys:
                    img_data = obs[img_keys[0]]
                else:
                    continue

            if hasattr(img_data, 'numpy'):
                img = img_data.numpy()
            else:
                img = np.array(img_data)

            # Extract action
            action_data = step['action']
            if isinstance(action_data, dict):
                action_parts = []
                if 'world_vector' in action_data:
                    wv = action_data['world_vector']
                    if hasattr(wv, 'numpy'):
                        wv = wv.numpy()
                    action_parts.extend(wv.flatten()[:3])
                if 'rotation_delta' in action_data:
                    rd = action_data['rotation_delta']
                    if hasattr(rd, 'numpy'):
                        rd = rd.numpy()
                    action_parts.extend(rd.flatten()[:3])
                if 'gripper_closedness_action' in action_data:
                    gc = action_data['gripper_closedness_action']
                    if hasattr(gc, 'numpy'):
                        gc = gc.numpy()
                    action_parts.append(float(gc.flatten()[0]))
                elif 'open_gripper' in action_data:
                    og = action_data['open_gripper']
                    if hasattr(og, 'numpy'):
                        og = og.numpy()
                    action_parts.append(float(og.flatten()[0]))
                action = np.array(action_parts, dtype=np.float32)
            else:
                if hasattr(action_data, 'numpy'):
                    action = action_data.numpy()
                else:
                    action = np.array(action_data)

            # Pad/truncate to 7 dims
            if len(action) < 7:
                action = np.pad(action, (0, 7 - len(action)))
            else:
                action = action[:7]

            episode['frames'].append(img.astype(np.uint8))
            episode['actions'].append(action.astype(np.float32))

        if len(episode['frames']) >= 15:
            episodes.append(episode)
            print(f"  [{len(episodes):2d}/{num_episodes}] '{instruction[:50]}...' ({len(episode['frames'])} steps)")

    print("-" * 60)
    print(f"Downloaded: {len(episodes)} episodes")
    print(f"Skipped: {skipped_short} short, {skipped_no_instruction} no instruction, {skipped_duplicate} duplicate")
    print(f"Unique instructions: {len(seen_instructions)}")

    return episodes


def main():
    parser = argparse.ArgumentParser(
        description='Download Bridge V2 episodes for OpenVLA evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_bridge_episodes.py                    # Download 20 episodes
    python download_bridge_episodes.py --num-episodes 50  # Download 50 episodes
    python download_bridge_episodes.py --force            # Re-download even if cached
        """
    )
    parser.add_argument('--num-episodes', type=int, default=20,
                        help='Number of episodes to download (default: 20)')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Max steps per episode (default: 50)')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if cache exists')
    parser.add_argument('--output', type=str, default=None,
                        help='Custom output path (default: CACHE_DIR/bridge_v2_episodes_extended.pkl)')
    args = parser.parse_args()

    # Determine output path
    if args.output:
        cache_file = args.output
    else:
        cache_file = f"{CACHE_DIR}/bridge_v2_episodes_extended.pkl"

    print("=" * 60)
    print(" Bridge V2 Episode Downloader")
    print("=" * 60)
    print(f"\nCache directory: {CACHE_DIR}")
    print(f"Output file: {cache_file}")
    print(f"Episodes requested: {args.num_episodes}")
    print(f"Max steps per episode: {args.max_steps}")

    # Check for existing cache
    if os.path.exists(cache_file) and not args.force:
        print(f"\n[CACHE EXISTS] Loading from {cache_file}")
        with open(cache_file, 'rb') as f:
            episodes = pickle.load(f)
        print(f"Loaded {len(episodes)} cached episodes")

        # Show sample
        print("\nSample instructions:")
        for i, ep in enumerate(episodes[:5]):
            print(f"  {i+1}. {ep['instruction'][:60]}... ({len(ep['frames'])} steps)")

        if len(episodes) < args.num_episodes:
            print(f"\n[WARNING] Cache has {len(episodes)} episodes, requested {args.num_episodes}")
            print("Use --force to re-download")

        return episodes

    # Check and fix all dependencies together
    print("\nChecking dependencies...")
    check_tensorflow_stack()

    # Download
    episodes = download_bridge_episodes(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps
    )

    # Save to cache
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(episodes, f)

    # Calculate cache size
    cache_size = os.path.getsize(cache_file) / (1024 * 1024)

    print("\n" + "=" * 60)
    print(" DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nSaved {len(episodes)} episodes to:")
    print(f"  {cache_file}")
    print(f"  Size: {cache_size:.1f} MB")

    # Summary statistics
    total_frames = sum(len(ep['frames']) for ep in episodes)
    avg_steps = total_frames / len(episodes) if episodes else 0

    print(f"\nStatistics:")
    print(f"  Total frames: {total_frames}")
    print(f"  Average steps/episode: {avg_steps:.1f}")
    print(f"  Unique instructions: {len(set(ep['instruction'] for ep in episodes))}")

    print("\n[OK] Ready for evaluation notebook")

    return episodes


if __name__ == "__main__":
    main()
