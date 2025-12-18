#!/usr/bin/env python
"""
Evaluate BASE OpenVLA model on Bridge V2 data.

If OpenVLA was trained on Bridge V2, the base model should produce
diverse, reasonable outputs without any fine-tuning.

This tests whether our inference pipeline is correct.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import pickle

# Configuration
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForVision2Seq, AutoProcessor


class ActionTokenizer:
    """OpenVLA action tokenizer."""
    def __init__(self, vocab_size=32000, n_bins=256):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.bins = np.linspace(-1, 1, n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

    def decode(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        discretized = self.vocab_size - token_ids
        discretized = np.clip(discretized - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[discretized]


def load_bridge_samples():
    """Load cached Bridge V2 samples."""
    cache_file = f"{CACHE_DIR}/bridge_v2_samples.pkl"
    if not os.path.exists(cache_file):
        print(f"Cache not found: {cache_file}")
        print("Run: python tutorials/scripts/download_bridge_subset.py --num-samples 50")
        return None

    with open(cache_file, 'rb') as f:
        samples = pickle.load(f)

    # Convert to PIL if needed
    for s in samples:
        if isinstance(s['image'], np.ndarray):
            s['image'] = Image.fromarray(s['image'])

    return samples


def main():
    print("=" * 60)
    print(" Evaluate BASE OpenVLA on Bridge V2")
    print(" (No fine-tuning - testing pretrained model)")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading BASE OpenVLA model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = model.to(device).eval()
    print(f"Model loaded: {model.config.architectures}")

    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
    )

    tokenizer = ActionTokenizer()

    # Load samples
    samples = load_bridge_samples()
    if samples is None:
        return

    print(f"\nLoaded {len(samples)} Bridge V2 samples")

    # Group by unique instructions
    instruction_groups = {}
    for s in samples:
        inst = s['instruction']
        if inst not in instruction_groups:
            instruction_groups[inst] = []
        instruction_groups[inst].append(s)

    print(f"Unique instructions: {len(instruction_groups)}")

    # Evaluate on samples with different instructions
    print("\n" + "=" * 60)
    print(" Running Inference on Bridge V2 Samples")
    print("=" * 60)

    results = []
    all_tokens = []

    # Take first sample from each unique instruction (up to 10)
    test_samples = []
    for inst, group in list(instruction_groups.items())[:10]:
        test_samples.append(group[0])

    for i, sample in enumerate(test_samples):
        instruction = sample['instruction']
        image = sample['image']
        expected_action = sample['action']

        # Format prompt - OpenVLA's exact format
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        # Process
        inputs = processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # CRITICAL: Add special empty token (29871) if not present
        # This is required to match training-time inputs!
        input_ids = inputs['input_ids']
        if input_ids[0, -1] != 29871:
            empty_token = torch.tensor([[29871]], device=device)
            input_ids = torch.cat([input_ids, empty_token], dim=1)
            inputs['input_ids'] = input_ids
            # Also extend attention mask if present
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = torch.cat([
                    inputs['attention_mask'],
                    torch.ones((1, 1), device=device, dtype=inputs['attention_mask'].dtype)
                ], dim=1)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Extract action tokens (last 7 tokens)
        action_tokens = outputs[0, -7:]
        decoded_action = tokenizer.decode(action_tokens)

        # Compute L1 error
        l1_error = np.mean(np.abs(decoded_action - expected_action[:7]))

        all_tokens.append(action_tokens.tolist())
        results.append({
            'instruction': instruction[:50],
            'expected': expected_action[:4],
            'decoded': decoded_action[:4],
            'tokens': action_tokens.tolist(),
            'l1_error': l1_error,
        })

        print(f"\nSample {i+1}: {instruction[:50]}...")
        print(f"  Expected:  {expected_action[:4]}")
        print(f"  Tokens:    {action_tokens.tolist()}")
        print(f"  Decoded:   {decoded_action[:4]}")
        print(f"  L1 Error:  {l1_error:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)

    avg_l1 = np.mean([r['l1_error'] for r in results])
    print(f"Average L1 Error: {avg_l1:.4f}")

    # Check if all outputs are the same
    unique_outputs = set(tuple(t) for t in all_tokens)
    print(f"Unique output patterns: {len(unique_outputs)} / {len(all_tokens)}")

    if len(unique_outputs) == 1:
        print("\n⚠️  WARNING: Model produces SAME output for ALL inputs!")
        print("   This suggests a problem with inference pipeline.")
        print("\n   Possible issues:")
        print("   1. Wrong prompt format")
        print("   2. Image not being processed correctly")
        print("   3. Model not loaded correctly")
        print("   4. Transformers version incompatibility")

        # Debug: print what the constant output is
        const_tokens = all_tokens[0]
        const_decoded = tokenizer.decode(np.array(const_tokens))
        print(f"\n   Constant output tokens: {const_tokens}")
        print(f"   Decoded values: {const_decoded}")
        print(f"   This maps to action: ~{const_decoded[0]:.4f} for all dimensions")

    elif len(unique_outputs) < len(all_tokens) // 2:
        print("\n⚠️  WARNING: Low output diversity")
        print(f"   Only {len(unique_outputs)} unique outputs for {len(all_tokens)} samples")
    else:
        print("\n✓ Model produces diverse outputs for different inputs")

    # Check if outputs are in valid range
    all_in_range = all(
        all(31744 <= t <= 32000 for t in tokens)
        for tokens in all_tokens
    )
    if all_in_range:
        print("✓ All outputs in valid action token range")
    else:
        print("⚠️  Some outputs outside valid action token range")

    print("\n" + "=" * 60)
    if avg_l1 < 0.3 and len(unique_outputs) > 1:
        print(" RESULT: Base model works reasonably on Bridge V2")
    else:
        print(" RESULT: Base model has issues - check inference pipeline")
    print("=" * 60)


if __name__ == "__main__":
    main()
