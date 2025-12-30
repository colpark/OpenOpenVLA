#!/usr/bin/env python
"""
Prepare LIBERO Data for OpenVLA Fine-tuning

This script:
1. Explores LIBERO HDF5 data structure
2. Validates data integrity
3. Creates a processed dataset cache optimized for training
4. Computes action normalization statistics

Usage:
    python prepare_libero_data.py --suite libero_spatial
    python prepare_libero_data.py --suite libero_spatial --visualize
    python prepare_libero_data.py --stats-only

Requirements:
    pip install h5py numpy pillow tqdm matplotlib
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import h5py

# =============================================================================
# Configuration
# =============================================================================
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
LIBERO_DATA_DIR = f"{BASE_DIR}/libero_data"
PROCESSED_DIR = f"{BASE_DIR}/libero_processed"


def find_libero_files(data_dir, suite_name=None):
    """Find all LIBERO HDF5 files, optionally filtered by suite."""
    data_path = Path(data_dir)
    all_files = list(data_path.rglob("*.hdf5"))

    if suite_name:
        # Filter for specific suite
        filtered = [f for f in all_files if suite_name in str(f)]
        if filtered:
            return filtered
        # Try alternate naming patterns
        filtered = [f for f in all_files if suite_name.replace("_", "-") in str(f)]
        if filtered:
            return filtered

    return all_files


def explore_hdf5_structure(filepath, verbose=True):
    """Explore HDF5 file structure in detail."""
    info = {
        'filepath': str(filepath),
        'demos': [],
        'attributes': {},
        'observation_keys': [],
        'action_dim': None,
        'language': None,
    }

    with h5py.File(filepath, 'r') as f:
        # Get attributes
        for key, value in f.attrs.items():
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            info['attributes'][key] = value

            # Extract language instruction
            if 'language' in key.lower() or 'instruction' in key.lower():
                info['language'] = value

        # Explore data structure
        if 'data' in f:
            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])

            for demo_key in demo_keys:
                demo = f['data'][demo_key]
                demo_info = {
                    'key': demo_key,
                    'n_steps': None,
                    'obs_keys': [],
                }

                # Get actions
                if 'actions' in demo:
                    actions = demo['actions'][:]
                    demo_info['n_steps'] = len(actions)
                    demo_info['action_shape'] = actions.shape
                    info['action_dim'] = actions.shape[-1]

                # Get observations
                if 'obs' in demo:
                    for obs_key in demo['obs'].keys():
                        obs = demo['obs'][obs_key]
                        demo_info['obs_keys'].append({
                            'key': obs_key,
                            'shape': obs.shape,
                            'dtype': str(obs.dtype),
                        })

                    if not info['observation_keys']:
                        info['observation_keys'] = list(demo['obs'].keys())

                info['demos'].append(demo_info)

    if verbose:
        print(f"\nFile: {Path(filepath).name}")
        print(f"  Language: {info['language']}")
        print(f"  Demos: {len(info['demos'])}")
        if info['demos']:
            print(f"  Steps per demo: {info['demos'][0]['n_steps']}")
            print(f"  Action dim: {info['action_dim']}")
            print(f"  Observation keys: {info['observation_keys']}")

    return info


def compute_action_statistics(data_dir, suite_name=None, sample_frac=1.0):
    """Compute action normalization statistics from demonstrations."""
    print("\nComputing action statistics...")

    files = find_libero_files(data_dir, suite_name)
    if not files:
        print(f"No files found in {data_dir}")
        return None

    all_actions = []

    for filepath in tqdm(files, desc="Processing files"):
        with h5py.File(filepath, 'r') as f:
            if 'data' not in f:
                continue

            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]

            # Sample if requested
            if sample_frac < 1.0:
                n_sample = max(1, int(len(demo_keys) * sample_frac))
                demo_keys = np.random.choice(demo_keys, n_sample, replace=False)

            for demo_key in demo_keys:
                if 'actions' in f['data'][demo_key]:
                    actions = f['data'][demo_key]['actions'][:]
                    all_actions.append(actions)

    if not all_actions:
        print("No actions found")
        return None

    all_actions = np.concatenate(all_actions, axis=0)

    stats = {
        'mean': all_actions.mean(axis=0).tolist(),
        'std': all_actions.std(axis=0).tolist(),
        'min': all_actions.min(axis=0).tolist(),
        'max': all_actions.max(axis=0).tolist(),
        'q01': np.percentile(all_actions, 1, axis=0).tolist(),
        'q99': np.percentile(all_actions, 99, axis=0).tolist(),
        'n_samples': len(all_actions),
        'action_dim': all_actions.shape[-1],
    }

    print(f"\nAction Statistics ({stats['n_samples']} samples):")
    print(f"  Dimension: {stats['action_dim']}")
    print(f"  Mean: {[f'{x:.4f}' for x in stats['mean']]}")
    print(f"  Std:  {[f'{x:.4f}' for x in stats['std']]}")
    print(f"  Q01:  {[f'{x:.4f}' for x in stats['q01']]}")
    print(f"  Q99:  {[f'{x:.4f}' for x in stats['q99']]}")

    return stats


def create_training_index(data_dir, suite_name, output_dir):
    """Create an index of all training samples for efficient loading."""
    print(f"\nCreating training index for {suite_name}...")

    files = find_libero_files(data_dir, suite_name)
    if not files:
        print(f"No files found")
        return None

    index = {
        'suite_name': suite_name,
        'samples': [],
        'files': [],
        'tasks': {},
        'total_steps': 0,
    }

    for file_idx, filepath in enumerate(tqdm(files, desc="Indexing")):
        index['files'].append(str(filepath))

        with h5py.File(filepath, 'r') as f:
            # Get language instruction
            language = f.attrs.get('language_instruction', f.attrs.get('problem_info', 'unknown task'))
            if isinstance(language, bytes):
                language = language.decode('utf-8')

            # Track tasks
            task_name = Path(filepath).stem
            if task_name not in index['tasks']:
                index['tasks'][task_name] = {
                    'language': language,
                    'n_demos': 0,
                    'n_steps': 0,
                }

            if 'data' not in f:
                continue

            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])

            for demo_key in demo_keys:
                demo = f['data'][demo_key]

                if 'actions' not in demo:
                    continue

                n_steps = len(demo['actions'])
                index['tasks'][task_name]['n_demos'] += 1
                index['tasks'][task_name]['n_steps'] += n_steps

                # Create sample entries for each timestep
                for t in range(n_steps):
                    index['samples'].append({
                        'file_idx': file_idx,
                        'demo_key': demo_key,
                        'timestep': t,
                        'language': language,
                        'task_name': task_name,
                    })

                index['total_steps'] += n_steps

    # Save index
    os.makedirs(output_dir, exist_ok=True)
    index_path = Path(output_dir) / f"{suite_name}_index.pkl"

    with open(index_path, 'wb') as f:
        pickle.dump(index, f)

    print(f"\nIndex Summary:")
    print(f"  Files: {len(index['files'])}")
    print(f"  Tasks: {len(index['tasks'])}")
    print(f"  Total samples: {len(index['samples'])}")
    print(f"  Total steps: {index['total_steps']}")
    print(f"  Saved to: {index_path}")

    return index


def visualize_samples(data_dir, suite_name, n_samples=8):
    """Visualize sample frames from demonstrations."""
    import matplotlib.pyplot as plt

    print(f"\nVisualizing samples from {suite_name}...")

    files = find_libero_files(data_dir, suite_name)
    if not files:
        print("No files found")
        return

    # Collect sample images
    samples = []

    for filepath in files[:n_samples]:
        with h5py.File(filepath, 'r') as f:
            language = f.attrs.get('language_instruction', 'unknown')
            if isinstance(language, bytes):
                language = language.decode('utf-8')

            if 'data' not in f:
                continue

            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            if not demo_keys:
                continue

            demo = f['data'][demo_keys[0]]

            # Find the right image key
            if 'obs' in demo:
                img_key = None
                for key in ['agentview_rgb', 'agentview_image', 'rgb', 'image']:
                    if key in demo['obs']:
                        img_key = key
                        break

                if img_key:
                    # Get middle frame
                    n_frames = len(demo['obs'][img_key])
                    mid_idx = n_frames // 2
                    image = demo['obs'][img_key][mid_idx]

                    # Rotate 180 degrees (LIBERO convention)
                    image = np.rot90(image, k=2)

                    samples.append({
                        'image': image,
                        'language': language[:50] + '...' if len(language) > 50 else language,
                    })

    if not samples:
        print("No samples found")
        return

    # Plot
    n_cols = min(4, len(samples))
    n_rows = (len(samples) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

    for i, (ax, sample) in enumerate(zip(axes, samples)):
        ax.imshow(sample['image'])
        ax.set_title(sample['language'], fontsize=8)
        ax.axis('off')

    for ax in axes[len(samples):]:
        ax.axis('off')

    plt.suptitle(f"LIBERO {suite_name} - Sample Frames", fontsize=12)
    plt.tight_layout()

    # Save figure
    fig_path = Path(PROCESSED_DIR) / f"{suite_name}_samples.png"
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved visualization to: {fig_path}")

    plt.show()


def validate_data_for_training(data_dir, suite_name):
    """Validate data is ready for OpenVLA fine-tuning."""
    print(f"\nValidating {suite_name} for OpenVLA fine-tuning...")

    files = find_libero_files(data_dir, suite_name)
    if not files:
        print("  FAIL: No files found")
        return False

    issues = []
    warnings = []

    total_demos = 0
    total_steps = 0
    action_dims = set()
    image_shapes = set()

    for filepath in tqdm(files, desc="Validating"):
        with h5py.File(filepath, 'r') as f:
            # Check language instruction
            language = f.attrs.get('language_instruction', f.attrs.get('problem_info', None))
            if not language:
                issues.append(f"{filepath.name}: Missing language instruction")

            if 'data' not in f:
                issues.append(f"{filepath.name}: Missing 'data' group")
                continue

            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            if not demo_keys:
                issues.append(f"{filepath.name}: No demonstrations found")
                continue

            total_demos += len(demo_keys)

            for demo_key in demo_keys:
                demo = f['data'][demo_key]

                # Check actions
                if 'actions' not in demo:
                    issues.append(f"{filepath.name}/{demo_key}: Missing actions")
                    continue

                actions = demo['actions'][:]
                action_dims.add(actions.shape[-1])
                total_steps += len(actions)

                # Check action range
                if np.abs(actions).max() > 10:
                    warnings.append(f"{filepath.name}/{demo_key}: Large action values (max={np.abs(actions).max():.2f})")

                # Check observations
                if 'obs' not in demo:
                    issues.append(f"{filepath.name}/{demo_key}: Missing observations")
                    continue

                # Check for image
                img_key = None
                for key in ['agentview_rgb', 'agentview_image', 'rgb', 'image']:
                    if key in demo['obs']:
                        img_key = key
                        break

                if not img_key:
                    issues.append(f"{filepath.name}/{demo_key}: No image observation found")
                else:
                    img_shape = demo['obs'][img_key].shape
                    image_shapes.add(img_shape[1:])  # Exclude time dimension

    # Summary
    print(f"\n{'=' * 60}")
    print(f" Validation Results for {suite_name}")
    print(f"{'=' * 60}")
    print(f"  Files: {len(files)}")
    print(f"  Demos: {total_demos}")
    print(f"  Total steps: {total_steps}")
    print(f"  Action dims: {action_dims}")
    print(f"  Image shapes: {image_shapes}")

    if issues:
        print(f"\n  ISSUES ({len(issues)}):")
        for issue in issues[:10]:
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")

    if warnings:
        print(f"\n  WARNINGS ({len(warnings)}):")
        for warning in warnings[:5]:
            print(f"    - {warning}")
        if len(warnings) > 5:
            print(f"    ... and {len(warnings) - 5} more")

    # Final verdict
    if not issues:
        print(f"\n  VALIDATION PASSED")
        return True
    else:
        print(f"\n  VALIDATION FAILED - {len(issues)} issues found")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Prepare LIBERO data for OpenVLA fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        help='Suite to process (default: libero_spatial)')
    parser.add_argument('--data-dir', type=str, default=LIBERO_DATA_DIR,
                        help=f'LIBERO data directory (default: {LIBERO_DATA_DIR})')
    parser.add_argument('--output-dir', type=str, default=PROCESSED_DIR,
                        help=f'Output directory (default: {PROCESSED_DIR})')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only compute action statistics')
    parser.add_argument('--validate', action='store_true',
                        help='Validate data for training')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize sample frames')
    parser.add_argument('--explore', action='store_true',
                        help='Explore HDF5 structure')
    parser.add_argument('--create-index', action='store_true',
                        help='Create training sample index')
    parser.add_argument('--all', action='store_true',
                        help='Run all preparation steps')
    args = parser.parse_args()

    print("=" * 60)
    print(" LIBERO Data Preparation for OpenVLA Fine-tuning")
    print("=" * 60)
    print(f"\nData directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Suite: {args.suite}")

    # Find files
    files = find_libero_files(args.data_dir, args.suite)
    if not files:
        print(f"\nNo LIBERO files found in {args.data_dir}")
        print("Please download data first:")
        print("  python download_libero_demos.py --suite " + args.suite)
        return

    print(f"\nFound {len(files)} HDF5 files")

    # Explore structure
    if args.explore or args.all:
        print("\n" + "=" * 60)
        print(" Exploring HDF5 Structure")
        print("=" * 60)
        explore_hdf5_structure(files[0], verbose=True)

    # Validate
    if args.validate or args.all:
        validate_data_for_training(args.data_dir, args.suite)

    # Compute statistics
    if args.stats_only or args.all:
        stats = compute_action_statistics(args.data_dir, args.suite)
        if stats:
            os.makedirs(args.output_dir, exist_ok=True)
            stats_path = Path(args.output_dir) / f"{args.suite}_action_stats.json"
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"\nAction statistics saved to: {stats_path}")

    # Create index
    if args.create_index or args.all:
        create_training_index(args.data_dir, args.suite, args.output_dir)

    # Visualize
    if args.visualize or args.all:
        visualize_samples(args.data_dir, args.suite)

    print("\n" + "=" * 60)
    print(" Preparation Complete")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Run fine-tuning:")
    print(f"     python finetune_openvla_libero.py --suite {args.suite}")


if __name__ == "__main__":
    main()
