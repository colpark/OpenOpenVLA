#!/usr/bin/env python
"""
Download LIBERO Demonstration Data for OpenVLA Fine-tuning

Downloads expert demonstration episodes from LIBERO benchmark.
Run this BEFORE fine-tuning to cache the data locally.

Usage:
    python download_libero_demos.py                          # Download libero_spatial (small)
    python download_libero_demos.py --suite libero_object    # Specific suite
    python download_libero_demos.py --suite all              # All suites
    python download_libero_demos.py --explore                # Explore existing data

Requirements:
    pip install huggingface_hub h5py tqdm
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
LIBERO_DATA_DIR = f"{BASE_DIR}/libero_data"

# LIBERO suite information
LIBERO_SUITES = {
    "libero_spatial": {
        "n_tasks": 10,
        "demos_per_task": 50,
        "description": "Same objects, different spatial arrangements",
    },
    "libero_object": {
        "n_tasks": 10,
        "demos_per_task": 50,
        "description": "Same positions, different objects",
    },
    "libero_goal": {
        "n_tasks": 10,
        "demos_per_task": 50,
        "description": "Same setup, different target goals",
    },
    "libero_90": {
        "n_tasks": 90,
        "demos_per_task": 50,
        "description": "Full 90-task benchmark",
    },
}


def install_if_missing(package):
    """Install package if not available."""
    try:
        __import__(package.replace("-", "_"))
        return True
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        return True


def check_dependencies():
    """Check and install required dependencies."""
    print("Checking dependencies...")

    deps = ["huggingface_hub", "h5py", "tqdm", "numpy"]
    for dep in deps:
        install_if_missing(dep)

    print("  All dependencies satisfied")
    return True


def download_from_huggingface(suite_name, data_dir):
    """Download LIBERO data from HuggingFace."""
    from huggingface_hub import hf_hub_download, list_repo_files
    import shutil

    print(f"\nDownloading {suite_name} from HuggingFace...")
    print("  Repository: libero-project/libero")

    # Create target directory
    suite_dir = Path(data_dir) / suite_name
    suite_dir.mkdir(parents=True, exist_ok=True)

    try:
        # List files in the repository for this suite
        files = list_repo_files(
            repo_id="libero-project/libero",
            repo_type="dataset",
        )

        # Filter files for this suite
        suite_files = [f for f in files if suite_name in f and f.endswith('.hdf5')]

        if not suite_files:
            print(f"  No HDF5 files found for {suite_name}")
            print("  Trying alternative download method...")
            return download_alternative(suite_name, data_dir)

        print(f"  Found {len(suite_files)} files")

        from tqdm import tqdm
        for filepath in tqdm(suite_files, desc="Downloading"):
            local_path = hf_hub_download(
                repo_id="libero-project/libero",
                filename=filepath,
                repo_type="dataset",
                cache_dir=CACHE_DIR,
            )

            # Copy to our data directory
            target_path = suite_dir / Path(filepath).name
            if not target_path.exists():
                shutil.copy2(local_path, target_path)

        return True

    except Exception as e:
        print(f"  HuggingFace download failed: {e}")
        print("  Trying alternative method...")
        return download_alternative(suite_name, data_dir)


def download_alternative(suite_name, data_dir):
    """Alternative download method using LIBERO's built-in tools."""
    print(f"\nTrying LIBERO built-in download for {suite_name}...")

    try:
        # Try using LIBERO's download utilities
        from libero.libero.utils.download import download_datasets

        download_datasets(
            datasets=[suite_name],
            save_dir=data_dir,
        )
        return True

    except ImportError:
        print("  LIBERO download utilities not available")
    except Exception as e:
        print(f"  LIBERO download failed: {e}")

    # Try direct URL download as last resort
    return download_from_url(suite_name, data_dir)


def download_from_url(suite_name, data_dir):
    """Download from direct URL (fallback method)."""
    import urllib.request
    import zipfile

    # Known URLs (may be outdated)
    urls = {
        "libero_spatial": "https://utexas.box.com/shared/static/7xyk7vpzlmhzyeujvxyrsgvd9m0fozxh.zip",
        "libero_object": "https://utexas.box.com/shared/static/x7xjszmk1l5q2ld5emqvycokfmmcbrgb.zip",
        "libero_goal": "https://utexas.box.com/shared/static/b3x5qzlvbf7bcfh8rfvgdx7b9x5xf8w0.zip",
    }

    if suite_name not in urls:
        print(f"  No direct URL available for {suite_name}")
        print("  Please download manually from: https://github.com/Lifelong-Robot-Learning/LIBERO")
        return False

    url = urls[suite_name]
    zip_path = Path(data_dir) / f"{suite_name}.zip"

    print(f"  Downloading from: {url}")

    try:
        urllib.request.urlretrieve(url, zip_path)

        print(f"  Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        # Clean up zip file
        zip_path.unlink()
        return True

    except Exception as e:
        print(f"  URL download failed: {e}")
        return False


def explore_data(data_dir):
    """Explore downloaded LIBERO data structure."""
    import h5py
    import numpy as np

    print("\n" + "=" * 60)
    print(" LIBERO Data Exploration")
    print("=" * 60)

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"\nData directory not found: {data_dir}")
        print("Please download data first.")
        return None

    # Find all HDF5 files
    hdf5_files = list(data_path.rglob("*.hdf5"))

    if not hdf5_files:
        print(f"\nNo HDF5 files found in {data_dir}")
        return None

    print(f"\nFound {len(hdf5_files)} HDF5 files")

    # Group by suite
    suites = {}
    for f in hdf5_files:
        for suite_name in LIBERO_SUITES.keys():
            if suite_name in str(f):
                if suite_name not in suites:
                    suites[suite_name] = []
                suites[suite_name].append(f)
                break

    print(f"\nSuites found:")
    for suite_name, files in suites.items():
        print(f"  {suite_name}: {len(files)} task files")

    # Explore first file in detail
    sample_file = hdf5_files[0]
    print(f"\n" + "-" * 60)
    print(f"Sample file: {sample_file.name}")
    print("-" * 60)

    with h5py.File(sample_file, 'r') as f:
        # Print structure
        print("\nHDF5 Structure:")

        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}{name}/")

        f.visititems(print_structure)

        # Print attributes
        print("\nFile Attributes:")
        for key, value in f.attrs.items():
            val_str = str(value)
            if len(val_str) > 80:
                val_str = val_str[:80] + "..."
            print(f"  {key}: {val_str}")

        # Count demos
        if 'data' in f:
            demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
            print(f"\nDemonstrations: {len(demo_keys)}")

            if demo_keys:
                demo = f['data'][demo_keys[0]]

                # Get trajectory length
                if 'actions' in demo:
                    n_steps = len(demo['actions'])
                    actions = demo['actions'][:]
                    print(f"  Steps per demo: {n_steps}")
                    print(f"  Action shape: {actions.shape}")
                    print(f"  Action range: [{actions.min():.3f}, {actions.max():.3f}]")

                # Get observation info
                if 'obs' in demo:
                    print(f"  Observation keys:")
                    for key in demo['obs'].keys():
                        obs = demo['obs'][key]
                        print(f"    {key}: shape={obs.shape}, dtype={obs.dtype}")

    # Return summary
    return {
        'data_dir': data_dir,
        'n_files': len(hdf5_files),
        'suites': {k: len(v) for k, v in suites.items()},
        'sample_file': str(sample_file),
    }


def verify_download(suite_name, data_dir):
    """Verify downloaded data is valid."""
    import h5py

    print(f"\nVerifying {suite_name}...")

    suite_path = Path(data_dir)
    hdf5_files = list(suite_path.rglob(f"*{suite_name}*/*.hdf5"))

    if not hdf5_files:
        # Try without suite subdirectory
        hdf5_files = list(suite_path.rglob("*.hdf5"))

    if not hdf5_files:
        print(f"  No HDF5 files found for {suite_name}")
        return False

    print(f"  Found {len(hdf5_files)} task files")

    total_demos = 0
    total_steps = 0

    for filepath in hdf5_files:
        try:
            with h5py.File(filepath, 'r') as f:
                if 'data' not in f:
                    continue

                demo_keys = [k for k in f['data'].keys() if k.startswith('demo_')]
                total_demos += len(demo_keys)

                for dk in demo_keys:
                    if 'actions' in f['data'][dk]:
                        total_steps += len(f['data'][dk]['actions'])
        except Exception as e:
            print(f"  Error reading {filepath}: {e}")
            return False

    expected_info = LIBERO_SUITES.get(suite_name, {})
    expected_demos = expected_info.get('n_tasks', 10) * expected_info.get('demos_per_task', 50)

    print(f"  Total demos: {total_demos}")
    print(f"  Total steps: {total_steps}")

    if total_demos >= expected_demos * 0.9:  # Allow 10% tolerance
        print(f"  Verification PASSED")
        return True
    else:
        print(f"  Expected ~{expected_demos} demos, got {total_demos}")
        print(f"  Verification FAILED (but data may still be usable)")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download LIBERO demonstration data for OpenVLA fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python download_libero_demos.py                          # Download libero_spatial
    python download_libero_demos.py --suite libero_object    # Download libero_object
    python download_libero_demos.py --suite all              # Download all suites
    python download_libero_demos.py --explore                # Explore existing data
        """
    )
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        choices=['libero_spatial', 'libero_object', 'libero_goal', 'libero_90', 'all'],
                        help='Suite to download (default: libero_spatial)')
    parser.add_argument('--output', type=str, default=None,
                        help=f'Output directory (default: {LIBERO_DATA_DIR})')
    parser.add_argument('--explore', action='store_true',
                        help='Explore existing data instead of downloading')
    parser.add_argument('--verify', action='store_true',
                        help='Verify downloaded data')
    parser.add_argument('--force', action='store_true',
                        help='Force re-download even if data exists')
    args = parser.parse_args()

    # Determine output directory
    data_dir = args.output if args.output else LIBERO_DATA_DIR

    print("=" * 60)
    print(" LIBERO Demonstration Data Downloader")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")

    # Explore mode
    if args.explore:
        explore_data(data_dir)
        return

    # Verify mode
    if args.verify:
        if args.suite == 'all':
            for suite_name in LIBERO_SUITES.keys():
                verify_download(suite_name, data_dir)
        else:
            verify_download(args.suite, data_dir)
        return

    # Check dependencies
    check_dependencies()

    # Download
    suites_to_download = list(LIBERO_SUITES.keys()) if args.suite == 'all' else [args.suite]

    print(f"\nSuites to download: {suites_to_download}")

    for suite_name in suites_to_download:
        info = LIBERO_SUITES[suite_name]
        total_demos = info['n_tasks'] * info['demos_per_task']

        print(f"\n{'=' * 60}")
        print(f" {suite_name}")
        print(f"{'=' * 60}")
        print(f"  Description: {info['description']}")
        print(f"  Tasks: {info['n_tasks']}")
        print(f"  Demos per task: {info['demos_per_task']}")
        print(f"  Total demos: {total_demos}")

        # Check if already downloaded
        suite_dir = Path(data_dir) / suite_name
        existing_files = list(suite_dir.glob("*.hdf5")) if suite_dir.exists() else []

        if existing_files and not args.force:
            print(f"\n  Found {len(existing_files)} existing files")
            print(f"  Use --force to re-download")
            verify_download(suite_name, data_dir)
            continue

        # Download
        success = download_from_huggingface(suite_name, data_dir)

        if success:
            verify_download(suite_name, data_dir)
        else:
            print(f"\n  Download failed for {suite_name}")
            print(f"  Manual download instructions:")
            print(f"  1. Visit https://github.com/Lifelong-Robot-Learning/LIBERO")
            print(f"  2. Download {suite_name} demos")
            print(f"  3. Extract to {data_dir}/{suite_name}/")

    # Final exploration
    print("\n" + "=" * 60)
    print(" Download Summary")
    print("=" * 60)
    explore_data(data_dir)

    print("\n[OK] Ready for fine-tuning")
    print(f"\nNext step: Run the fine-tuning script")


if __name__ == "__main__":
    main()
