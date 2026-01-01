#!/usr/bin/env python
"""
Evaluate Fine-tuned OpenVLA with Action Chunking

Compare base model vs fine-tuned model on LIBERO validation data.

Usage:
    python evaluate_finetuned_chunked.py --checkpoint results/run_name/best
    python evaluate_finetuned_chunked.py --checkpoint results/run_name/final --max-samples 500
"""

import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import h5py
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
LIBERO_DATA_DIR = f"{BASE_DIR}/libero_data"

os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"
os.environ['TORCH_HOME'] = f"{CACHE_DIR}/torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Action Tokenizer
# =============================================================================
class ActionTokenizer:
    def __init__(self, vocab_size=32000, n_bins=256):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.bins = np.linspace(-1, 1, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        discretized = self.vocab_size - token_ids
        indices = np.clip(discretized - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[indices]


# =============================================================================
# Data Loading
# =============================================================================
def transform_action(action):
    """Transform LIBERO action to OpenVLA format."""
    action = action.astype(np.float32)
    action[:6] = np.clip(action[:6], -1.0, 1.0)
    gripper = np.clip(action[6], 0.0, 1.0)
    action[6] = 1.0 - gripper
    return action


def load_validation_samples(data_dir, suite_name, chunk_size=4, val_demos=5, max_samples=500):
    """Load validation samples with chunking."""
    data_dir = Path(data_dir)
    samples = []

    hdf5_files = list(data_dir.rglob("*.hdf5"))
    print(f"Found {len(hdf5_files)} HDF5 files")

    for filepath in tqdm(hdf5_files, desc="Loading validation data"):
        try:
            with h5py.File(filepath, 'r') as f:
                language = "complete the task"
                for key in ['language_instruction', 'problem_info', 'language']:
                    if key in f.attrs:
                        lang = f.attrs[key]
                        if isinstance(lang, bytes):
                            lang = lang.decode('utf-8')
                        language = lang
                        break

                if 'data' not in f:
                    continue

                demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
                val_demo_keys = demo_keys[-val_demos:]

                for demo_key in val_demo_keys:
                    demo = f['data'][demo_key]

                    if 'actions' not in demo or 'obs' not in demo:
                        continue

                    img_key = None
                    for key in ['agentview_rgb', 'agentview_image', 'rgb', 'image']:
                        if key in demo['obs']:
                            img_key = key
                            break
                    if img_key is None:
                        continue

                    n_steps = len(demo['actions'])

                    for t in range(0, n_steps, chunk_size):
                        image = demo['obs'][img_key][t]
                        image = np.rot90(image, k=2)

                        action = demo['actions'][t]
                        if len(action) < 7:
                            action = np.pad(action, (0, 7 - len(action)))
                        else:
                            action = action[:7]

                        samples.append({
                            'image': image,
                            'action': transform_action(action),
                            'language': language,
                        })

                        if len(samples) >= max_samples:
                            return samples

        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    return samples


# =============================================================================
# Evaluation
# =============================================================================
def predict_action(model, processor, action_tokenizer, image, instruction, device):
    """Predict action from image and instruction."""
    pil_image = Image.fromarray(image.astype(np.uint8))
    pil_image = pil_image.resize((224, 224), Image.LANCZOS)

    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    inputs = processor(prompt, pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=7,
            do_sample=False,
            pad_token_id=model.config.pad_token_id,
        )

    pred_tokens = outputs[0, -7:].cpu().numpy()
    pred_action = action_tokenizer.decode(pred_tokens)

    return pred_action


def compute_metrics(predictions, ground_truths):
    """Compute comprehensive evaluation metrics."""
    l1_error = np.abs(predictions - ground_truths).mean()
    position_l1 = np.abs(predictions[:, :3] - ground_truths[:, :3]).mean()
    rotation_l1 = np.abs(predictions[:, 3:6] - ground_truths[:, 3:6]).mean()

    # Direction accuracy
    threshold = 0.02
    dir_correct = 0
    dir_total = 0
    for dim in range(3):
        significant = np.abs(ground_truths[:, dim]) > threshold
        if significant.sum() > 0:
            same_sign = np.sign(ground_truths[:, dim][significant]) == np.sign(predictions[:, dim][significant])
            dir_correct += same_sign.sum()
            dir_total += significant.sum()
    direction_accuracy = dir_correct / dir_total if dir_total > 0 else 0.5

    # Gripper accuracy
    gripper_threshold = 0.5
    gt_gripper = (ground_truths[:, 6] > gripper_threshold).astype(int)
    pred_gripper = (predictions[:, 6] > gripper_threshold).astype(int)
    gripper_accuracy = (gt_gripper == pred_gripper).mean()

    return {
        'l1_error': float(l1_error),
        'position_l1': float(position_l1),
        'rotation_l1': float(rotation_l1),
        'direction_accuracy': float(direction_accuracy),
        'gripper_accuracy': float(gripper_accuracy),
    }


def evaluate_model(model, samples, processor, action_tokenizer, device, model_name="Model"):
    """Evaluate model on validation samples."""
    model.eval()

    predictions = []
    ground_truths = []

    for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
        try:
            pred = predict_action(model, processor, action_tokenizer,
                                sample['image'], sample['language'], device)
            predictions.append(pred)
            ground_truths.append(sample['action'])
        except Exception as e:
            continue

    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    metrics = compute_metrics(predictions, ground_truths)
    metrics['num_samples'] = len(predictions)

    return metrics, predictions, ground_truths


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned OpenVLA')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint (e.g., results/run_name/best)')
    parser.add_argument('--data-dir', type=str, default=LIBERO_DATA_DIR,
                        help='LIBERO data directory')
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        help='LIBERO suite')
    parser.add_argument('--chunk-size', type=int, default=4,
                        help='Action chunking size')
    parser.add_argument('--max-samples', type=int, default=300,
                        help='Maximum validation samples')
    parser.add_argument('--val-demos', type=int, default=5,
                        help='Validation demos per task')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading base model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import PeftModel

    base_model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
    )

    print(f"Loading LoRA adapters from {checkpoint_path}...")
    model = PeftModel.from_pretrained(base_model, str(checkpoint_path), is_trainable=False)
    model = model.to(device)
    model.eval()

    vocab_size = len(processor.tokenizer)
    action_tokenizer = ActionTokenizer(vocab_size=vocab_size)

    # Load validation data
    print(f"\nLoading validation data (chunk_size={args.chunk_size})...")
    samples = load_validation_samples(
        args.data_dir,
        args.suite,
        chunk_size=args.chunk_size,
        val_demos=args.val_demos,
        max_samples=args.max_samples,
    )
    print(f"Loaded {len(samples)} samples")

    # Evaluate base model
    print("\n" + "="*60)
    print(" Evaluating BASE model (LoRA disabled)")
    print("="*60)
    model.disable_adapter_layers()
    base_metrics, base_preds, base_gts = evaluate_model(
        model, samples, processor, action_tokenizer, device, "Base Model"
    )

    print(f"\nBase Model Results:")
    print(f"  L1 Error: {base_metrics['l1_error']:.4f}")
    print(f"  Position L1: {base_metrics['position_l1']:.4f}")
    print(f"  Direction Accuracy: {base_metrics['direction_accuracy']:.4f}")
    print(f"  Gripper Accuracy: {base_metrics['gripper_accuracy']:.4f}")

    # Evaluate fine-tuned model
    print("\n" + "="*60)
    print(" Evaluating FINE-TUNED model (LoRA enabled)")
    print("="*60)
    model.enable_adapter_layers()
    ft_metrics, ft_preds, ft_gts = evaluate_model(
        model, samples, processor, action_tokenizer, device, "Fine-tuned Model"
    )

    print(f"\nFine-tuned Model Results:")
    print(f"  L1 Error: {ft_metrics['l1_error']:.4f}")
    print(f"  Position L1: {ft_metrics['position_l1']:.4f}")
    print(f"  Direction Accuracy: {ft_metrics['direction_accuracy']:.4f}")
    print(f"  Gripper Accuracy: {ft_metrics['gripper_accuracy']:.4f}")

    # Comparison
    print("\n" + "="*60)
    print(" COMPARISON")
    print("="*60)
    print(f"\nChunk size: {args.chunk_size} (20 Hz → {20/args.chunk_size:.1f} Hz)")
    print("\n" + "-"*60)
    print(f"{'Metric':<25} {'Base':>12} {'Fine-tuned':>12} {'Change':>12}")
    print("-"*60)

    metrics = ['l1_error', 'position_l1', 'direction_accuracy', 'gripper_accuracy']
    labels = ['L1 Error', 'Position L1', 'Direction Accuracy', 'Gripper Accuracy']

    for metric, label in zip(metrics, labels):
        base_val = base_metrics[metric]
        ft_val = ft_metrics[metric]

        if 'accuracy' in metric:
            change = ft_val - base_val
            status = '✅' if change > 0 else ('⚠️' if change < -0.05 else '→')
            print(f"{label:<25} {base_val:>11.1%} {ft_val:>11.1%} {change:>+11.1%} {status}")
        else:
            change_pct = (ft_val - base_val) / base_val * 100
            status = '✅' if change_pct < 0 else ('⚠️' if change_pct > 10 else '→')
            print(f"{label:<25} {base_val:>12.4f} {ft_val:>12.4f} {change_pct:>+10.1f}% {status}")

    print("-"*60)

    # Save results
    results = {
        'checkpoint': str(checkpoint_path),
        'chunk_size': args.chunk_size,
        'effective_hz': 20 / args.chunk_size,
        'num_samples': len(samples),
        'base_model': base_metrics,
        'finetuned_model': ft_metrics,
        'improvement': {
            'l1_error_reduction_pct': (base_metrics['l1_error'] - ft_metrics['l1_error']) / base_metrics['l1_error'] * 100,
            'direction_accuracy_change': ft_metrics['direction_accuracy'] - base_metrics['direction_accuracy'],
            'gripper_accuracy_change': ft_metrics['gripper_accuracy'] - base_metrics['gripper_accuracy'],
        }
    }

    output_path = args.output or (checkpoint_path.parent / "evaluation_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Summary
    print("\n" + "="*60)
    print(" SUMMARY")
    print("="*60)
    if ft_metrics['direction_accuracy'] > base_metrics['direction_accuracy']:
        print("✅ Direction accuracy IMPROVED - action chunking is working!")
    elif ft_metrics['direction_accuracy'] > 0.5:
        print("⚠️ Direction accuracy above random but not improved from base")
    else:
        print("❌ Direction accuracy below random - further tuning needed")

    if ft_metrics['l1_error'] < base_metrics['l1_error']:
        print("✅ L1 error reduced")
    else:
        print("⚠️ L1 error increased")


if __name__ == "__main__":
    main()
