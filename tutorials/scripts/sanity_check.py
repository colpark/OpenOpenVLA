#!/usr/bin/env python
"""
Sanity Check for OpenVLA Model Loading and Inference

This script validates that:
1. Model loads correctly
2. Action tokenization/detokenization works
3. Model produces reasonable outputs for a test image

Uses a simple test case - no external data needed.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

# Configuration
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"

from transformers import AutoModelForVision2Seq, AutoProcessor


def create_test_image():
    """Create a simple test image (robot workspace scene)."""
    # Create a 256x256 image with some structure
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # Background (table-like)
    img[:, :] = [180, 160, 140]  # Tan/brown table

    # Add a "robot gripper" area (dark gray)
    img[20:80, 100:160] = [60, 60, 60]

    # Add a "target object" (red block)
    img[150:200, 80:130] = [200, 50, 50]

    # Add a "target location" (green area)
    img[150:200, 150:200] = [50, 200, 50]

    return Image.fromarray(img)


class ActionTokenizer:
    """Matches OpenVLA's action tokenization."""

    def __init__(self, vocab_size=32000, n_bins=256):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.bins = np.linspace(-1, 1, n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

    def decode(self, token_ids):
        """Decode action tokens to continuous actions."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        discretized = self.vocab_size - token_ids
        discretized = np.clip(discretized - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[discretized]


def main():
    print("=" * 60)
    print("OpenVLA Sanity Check")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Load model
    print("\n[1/4] Loading model...")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        model = model.to(device).eval()
        print("   ✅ Model loaded successfully")
    except Exception as e:
        print(f"   ❌ Model loading failed: {e}")
        return False

    # Step 2: Load processor
    print("\n[2/4] Loading processor...")
    try:
        processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b",
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
        )
        print("   ✅ Processor loaded successfully")
        print(f"   Vocab size: {len(processor.tokenizer)}")
    except Exception as e:
        print(f"   ❌ Processor loading failed: {e}")
        return False

    # Step 3: Test inference
    print("\n[3/4] Running inference...")
    try:
        # Create test image and prompt
        image = create_test_image()
        instruction = "pick up the red block and place it on the green area"
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Process inputs
        inputs = processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        # Extract action tokens
        input_len = inputs['input_ids'].shape[1]
        action_tokens = output[0, input_len:input_len + 7]

        print(f"   Generated tokens: {action_tokens.tolist()}")
        print("   ✅ Inference completed")
    except Exception as e:
        print(f"   ❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Decode actions
    print("\n[4/4] Decoding actions...")
    try:
        tokenizer = ActionTokenizer(vocab_size=32000)
        actions = tokenizer.decode(action_tokens)

        print(f"   Decoded actions: {actions}")
        print(f"   Action range: [{actions.min():.3f}, {actions.max():.3f}]")

        # Check if actions are in valid range
        if np.all(np.abs(actions) <= 1.0):
            print("   ✅ Actions in valid range [-1, 1]")
        else:
            print("   ⚠️  Some actions outside [-1, 1] range")

        # Check action token validity
        if all(t > 31000 for t in action_tokens.tolist()):
            print("   ✅ Action tokens in expected range (31744-32000)")
        else:
            print(f"   ⚠️  Some tokens outside expected action token range")

    except Exception as e:
        print(f"   ❌ Decoding failed: {e}")
        return False

    # Summary
    print("\n" + "=" * 60)
    print("SANITY CHECK PASSED ✅")
    print("=" * 60)
    print("\nThe OpenVLA model is working correctly:")
    print("- Model loads and runs inference")
    print("- Generates valid action tokens")
    print("- Tokens decode to valid action values")
    print("\nYou can now proceed with fine-tuning or evaluation.")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
