#!/usr/bin/env python
"""
Offline Evaluation for OpenVLA

Evaluates action prediction accuracy on demonstration data (no simulation).
This is useful for:
1. Validating the base model on its original training data (Bridge V2)
2. Quickly checking if fine-tuning improved action prediction
3. Debugging tokenization/detokenization

Much faster than simulation rollouts since it's just forward passes.

Usage:
    # Evaluate base model on Bridge V2 (original training data)
    python evaluate_offline.py --checkpoint openvla/openvla-7b --dataset bridge

    # Evaluate fine-tuned model on LIBERO
    python evaluate_offline.py --checkpoint $SCRATCH/openvla_finetune/final --dataset libero
"""

import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"
os.environ['TORCH_HOME'] = f"{CACHE_DIR}/torch"

from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel


class ActionTokenizer:
    """Matches OpenVLA's action tokenization exactly."""

    def __init__(self, vocab_size=32000, n_bins=256, min_action=-1.0, max_action=1.0):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action

        # Bin edges for discretization (257 edges for 256 bins)
        self.bins = np.linspace(min_action, max_action, n_bins + 1)
        # Bin centers for decoding (256 centers)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

    def decode(self, token_ids, action_dim=7):
        """Decode action tokens to continuous actions."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()

        # OpenVLA convention: token_id = vocab_size - discretized_action
        discretized = self.vocab_size - token_ids
        # Clip to valid range [1, 256] then convert to [0, 255] index
        discretized = np.clip(discretized - 1, 0, len(self.bin_centers) - 1)
        actions = self.bin_centers[discretized]
        return actions[:action_dim]


def load_model(checkpoint_path, device="cuda:0"):
    """Load OpenVLA model (base or fine-tuned)."""
    print(f"Loading model from {checkpoint_path}...")

    # Check if it's a LoRA checkpoint
    adapter_config = Path(checkpoint_path) / "adapter_config.json"
    is_lora = adapter_config.exists()

    if is_lora:
        print("Detected LoRA checkpoint - loading base model + adapters...")
        # Load base model
        base_model_id = "openvla/openvla-7b"
        model = AutoModelForVision2Seq.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
            low_cpu_mem_usage=True,
        )
        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
        )
    else:
        # Load full model (base or merged)
        model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
        )

    model = model.to(device).eval()
    print(f"Model loaded on {device}")

    return model, processor


def load_bridge_data(data_dir, max_samples=100):
    """
    Load Bridge V2 data samples.
    Bridge V2 is part of OpenVLA's training data.

    Expected format: RLDS/TFRecord or converted format
    For simplicity, we'll download a small sample via HuggingFace datasets.
    """
    print("Loading Bridge V2 data...")

    try:
        from datasets import load_dataset

        # Load a small subset of Bridge V2 from HuggingFace
        # This is the original training data format
        dataset = load_dataset(
            "jxu124/OpenX-Embodiment",
            "bridge",
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        samples = []
        for i, example in enumerate(tqdm(dataset, total=max_samples, desc="Loading Bridge samples")):
            if i >= max_samples:
                break

            # Extract image and action
            # Bridge V2 format varies, handle common cases
            if 'observation' in example and 'image' in example['observation']:
                image = example['observation']['image']
            elif 'image' in example:
                image = example['image']
            else:
                continue

            if 'action' in example:
                action = np.array(example['action'])
            else:
                continue

            # Bridge V2 instruction (if available)
            instruction = example.get('instruction', example.get('language_instruction',
                "Pick up the object and place it in the target location."))

            samples.append({
                'image': image,
                'action': action,
                'instruction': instruction,
            })

        print(f"Loaded {len(samples)} Bridge V2 samples")
        return samples

    except Exception as e:
        print(f"Could not load Bridge V2 from HuggingFace: {e}")
        print("Falling back to synthetic test data...")
        return create_synthetic_test_data(max_samples)


def load_libero_data(data_dir, max_samples=100):
    """Load LIBERO demonstration data."""
    import h5py

    data_path = Path(data_dir)
    hdf5_files = list(data_path.rglob("*.hdf5"))

    if not hdf5_files:
        print(f"No HDF5 files found in {data_dir}")
        return []

    print(f"Found {len(hdf5_files)} HDF5 files")

    samples = []
    for filepath in tqdm(hdf5_files, desc="Loading LIBERO samples"):
        if len(samples) >= max_samples:
            break

        try:
            with h5py.File(filepath, 'r') as f:
                # Get task instruction from filename or attributes
                task_name = filepath.stem.replace("_demo", "").replace("_", " ")
                instruction = f"Task: {task_name}"

                for demo_key in f['data'].keys():
                    if len(samples) >= max_samples:
                        break

                    demo = f['data'][demo_key]
                    images = demo['obs']['agentview_rgb'][:]
                    actions = demo['actions'][:]

                    # Sample a few frames from each demo
                    n_frames = len(images)
                    indices = np.linspace(0, n_frames - 1, min(5, n_frames), dtype=int)

                    for idx in indices:
                        if len(samples) >= max_samples:
                            break

                        image = images[idx]
                        # LIBERO images need rotation
                        image = np.rot90(image, k=2)

                        action = actions[idx]

                        samples.append({
                            'image': Image.fromarray(image),
                            'action': action,
                            'instruction': instruction,
                        })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue

    print(f"Loaded {len(samples)} LIBERO samples")
    return samples


def create_synthetic_test_data(n_samples=50):
    """Create synthetic test data for basic sanity checking."""
    print("Creating synthetic test data...")

    samples = []
    instructions = [
        "Pick up the red block and place it on the table.",
        "Move the cup to the left side.",
        "Push the object forward.",
        "Grasp the item and lift it up.",
    ]

    for i in range(n_samples):
        # Create a simple colored image
        color = np.random.randint(50, 200, size=3, dtype=np.uint8)
        image = np.full((256, 256, 3), color, dtype=np.uint8)
        # Add some variation
        image[100:150, 100:150] = np.random.randint(0, 255, size=(50, 50, 3), dtype=np.uint8)

        # Random action in valid range
        action = np.random.uniform(-1, 1, size=7)

        samples.append({
            'image': Image.fromarray(image),
            'action': action,
            'instruction': instructions[i % len(instructions)],
        })

    return samples


@torch.no_grad()
def predict_action(model, processor, image, instruction, tokenizer, device="cuda:0"):
    """Run inference to predict action."""
    # Format prompt
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    # Process inputs
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    inputs = processor(prompt, image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate action tokens
    output = model.generate(
        **inputs,
        max_new_tokens=8,  # 7 action dims + potential padding
        do_sample=False,
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # Extract generated tokens (skip input tokens)
    input_len = inputs['input_ids'].shape[1]
    action_tokens = output[0, input_len:input_len + 7]

    # Decode to continuous actions
    predicted_action = tokenizer.decode(action_tokens)

    return predicted_action


def compute_metrics(predictions, ground_truths):
    """Compute evaluation metrics."""
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    # Ensure same shape
    min_len = min(len(predictions), len(ground_truths))
    predictions = predictions[:min_len]
    ground_truths = ground_truths[:min_len]

    # L1 error (Mean Absolute Error)
    l1_error = np.mean(np.abs(predictions - ground_truths))

    # L2 error (Mean Squared Error)
    mse = np.mean((predictions - ground_truths) ** 2)
    rmse = np.sqrt(mse)

    # Per-dimension errors
    per_dim_l1 = np.mean(np.abs(predictions - ground_truths), axis=0)

    # Action magnitude comparison
    pred_magnitude = np.mean(np.linalg.norm(predictions, axis=1))
    gt_magnitude = np.mean(np.linalg.norm(ground_truths, axis=1))

    # Direction accuracy (cosine similarity)
    cos_sim = []
    for p, g in zip(predictions, ground_truths):
        p_norm = np.linalg.norm(p)
        g_norm = np.linalg.norm(g)
        if p_norm > 1e-6 and g_norm > 1e-6:
            cos_sim.append(np.dot(p, g) / (p_norm * g_norm))
    avg_cos_sim = np.mean(cos_sim) if cos_sim else 0.0

    return {
        'l1_error': float(l1_error),
        'rmse': float(rmse),
        'mse': float(mse),
        'per_dim_l1': per_dim_l1.tolist(),
        'pred_magnitude': float(pred_magnitude),
        'gt_magnitude': float(gt_magnitude),
        'cosine_similarity': float(avg_cos_sim),
        'n_samples': len(predictions),
    }


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation for OpenVLA")
    parser.add_argument("--checkpoint", type=str, default="openvla/openvla-7b",
                        help="Model checkpoint path (base model or fine-tuned)")
    parser.add_argument("--dataset", type=str, default="bridge",
                        choices=["bridge", "libero", "synthetic"],
                        help="Dataset to evaluate on")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (for LIBERO)")
    parser.add_argument("--max-samples", type=int, default=100,
                        help="Maximum samples to evaluate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for results (JSON)")
    args = parser.parse_args()

    # Set data directory
    if args.data_dir is None:
        args.data_dir = f"{BASE_DIR}/libero_data"

    # Load model
    model, processor = load_model(args.checkpoint, args.device)
    tokenizer = ActionTokenizer(vocab_size=32000)

    # Load data
    if args.dataset == "bridge":
        samples = load_bridge_data(args.data_dir, args.max_samples)
    elif args.dataset == "libero":
        samples = load_libero_data(args.data_dir, args.max_samples)
    else:
        samples = create_synthetic_test_data(args.max_samples)

    if not samples:
        print("No samples loaded!")
        return

    # Run evaluation
    print(f"\nEvaluating on {len(samples)} samples...")
    predictions = []
    ground_truths = []

    for sample in tqdm(samples, desc="Evaluating"):
        try:
            pred = predict_action(
                model, processor,
                sample['image'],
                sample['instruction'],
                tokenizer,
                args.device
            )
            gt = sample['action'][:7]  # Ensure 7 dims

            predictions.append(pred)
            ground_truths.append(gt)
        except Exception as e:
            print(f"Error on sample: {e}")
            continue

    # Compute metrics
    metrics = compute_metrics(predictions, ground_truths)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples evaluated: {metrics['n_samples']}")
    print("-" * 60)
    print(f"L1 Error (MAE):     {metrics['l1_error']:.4f}")
    print(f"RMSE:               {metrics['rmse']:.4f}")
    print(f"Cosine Similarity:  {metrics['cosine_similarity']:.4f}")
    print(f"Pred Magnitude:     {metrics['pred_magnitude']:.4f}")
    print(f"GT Magnitude:       {metrics['gt_magnitude']:.4f}")
    print("-" * 60)
    print("Per-dimension L1 error:")
    dim_names = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']
    for i, (name, err) in enumerate(zip(dim_names, metrics['per_dim_l1'])):
        print(f"  {name:8s}: {err:.4f}")
    print("=" * 60)

    # Interpretation
    print("\nINTERPRETATION:")
    if metrics['l1_error'] < 0.1:
        print("✅ Excellent - predictions closely match ground truth")
    elif metrics['l1_error'] < 0.2:
        print("✅ Good - reasonable prediction accuracy")
    elif metrics['l1_error'] < 0.4:
        print("⚠️  Moderate - some prediction errors, may work in simulation")
    else:
        print("❌ Poor - high prediction error, check model/tokenization")

    if metrics['cosine_similarity'] > 0.8:
        print("✅ Good direction alignment")
    elif metrics['cosine_similarity'] > 0.5:
        print("⚠️  Moderate direction alignment")
    else:
        print("❌ Poor direction alignment - actions may be inverted or wrong")

    # Save results
    if args.output:
        results = {
            'checkpoint': args.checkpoint,
            'dataset': args.dataset,
            'metrics': metrics,
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
