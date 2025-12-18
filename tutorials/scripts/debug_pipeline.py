#!/usr/bin/env python
"""
Comprehensive OpenVLA Pipeline Debugger

Tests each component of the training/evaluation pipeline independently:
1. Model loading
2. Action tokenization (encode/decode roundtrip)
3. Data loading
4. Forward pass
5. Loss computation on action tokens
6. Single training step (does loss decrease?)
7. Inference after training (do predictions change?)

Uses controlled synthetic data to isolate issues.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import json

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

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def print_pass(msg):
    print(f"{GREEN}✓ PASS{RESET}: {msg}")

def print_fail(msg):
    print(f"{RED}✗ FAIL{RESET}: {msg}")

def print_warn(msg):
    print(f"{YELLOW}⚠ WARN{RESET}: {msg}")

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


class ActionTokenizer:
    """OpenVLA-compatible action tokenizer."""

    def __init__(self, vocab_size=32000, n_bins=256, min_action=-1.0, max_action=1.0):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action
        self.bins = np.linspace(min_action, max_action, n_bins + 1)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

    def encode(self, action):
        """Encode continuous action to token IDs."""
        action = np.clip(action, self.min_action, self.max_action)
        discretized = np.digitize(action, self.bins)
        token_ids = self.vocab_size - discretized
        return token_ids

    def decode(self, token_ids):
        """Decode token IDs to continuous actions."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        discretized = self.vocab_size - token_ids
        discretized = np.clip(discretized - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[discretized]


def create_test_samples(n_samples=5):
    """Create controlled test samples with known actions."""
    samples = []

    # Create samples with distinct, known actions
    test_actions = [
        [0.5, 0.3, -0.2, 0.0, 0.1, -0.1, 1.0],   # Move right, forward, down, gripper open
        [-0.5, -0.3, 0.2, 0.0, -0.1, 0.1, -1.0], # Move left, back, up, gripper close
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],     # Stay still
        [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0],     # Max positive
        [-1.0, -1.0, -1.0, -0.5, -0.5, -0.5, -1.0], # Max negative
    ]

    instructions = [
        "pick up the red block and move it right",
        "pick up the blue block and move it left",
        "stay still and observe",
        "move to the maximum position",
        "move to the minimum position",
    ]

    for i in range(n_samples):
        # Create a distinctive image for each sample
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Different background color for each sample
        img[:, :] = [50 + i*40, 100, 150 - i*20]
        # Add a colored block
        block_color = [200, 50 + i*30, 50]
        img[100:150, 100+i*20:150+i*20] = block_color

        samples.append({
            'image': Image.fromarray(img),
            'action': np.array(test_actions[i % len(test_actions)]),
            'instruction': instructions[i % len(instructions)],
        })

    return samples


def test_1_model_loading():
    """Test 1: Model loads correctly."""
    print_section("TEST 1: Model Loading")

    try:
        print("Loading OpenVLA model...")
        model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )
        print_pass(f"Model loaded, parameters: {sum(p.numel() for p in model.parameters()):,}")

        processor = AutoProcessor.from_pretrained(
            "openvla/openvla-7b",
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
        )
        print_pass(f"Processor loaded, vocab size: {len(processor.tokenizer)}")

        return model, processor
    except Exception as e:
        print_fail(f"Model loading failed: {e}")
        return None, None


def test_2_action_tokenization():
    """Test 2: Action tokenization roundtrip."""
    print_section("TEST 2: Action Tokenization")

    tokenizer = ActionTokenizer(vocab_size=32000)

    test_actions = [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        np.array([0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.0]),
    ]

    all_passed = True
    for action in test_actions:
        encoded = tokenizer.encode(action)
        decoded = tokenizer.decode(encoded)
        error = np.abs(action - decoded).max()

        # Check token range
        in_range = all(31744 <= t <= 32000 for t in encoded)

        if error < 0.01 and in_range:
            print_pass(f"Action {action[:3]}... -> tokens {encoded[:3]}... -> decoded error: {error:.4f}")
        else:
            print_fail(f"Action {action[:3]}... -> tokens {encoded[:3]}... -> error: {error:.4f}, in_range: {in_range}")
            all_passed = False

    return all_passed, tokenizer


def test_3_data_preparation(processor, tokenizer):
    """Test 3: Data preparation (image + prompt + action tokens)."""
    print_section("TEST 3: Data Preparation")

    samples = create_test_samples(3)

    all_passed = True
    for i, sample in enumerate(samples):
        # Format prompt
        prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"

        # Process inputs
        inputs = processor(prompt, sample['image'], return_tensors="pt")

        # Tokenize action
        action_tokens = torch.tensor(tokenizer.encode(sample['action']), dtype=torch.long)

        # Append action tokens
        input_ids = torch.cat([inputs['input_ids'].squeeze(), action_tokens])
        attention_mask = torch.cat([
            inputs['attention_mask'].squeeze(),
            torch.ones(len(action_tokens), dtype=torch.long)
        ])

        # Create labels
        prompt_len = len(inputs['input_ids'].squeeze())
        labels = torch.full_like(input_ids, -100)
        labels[prompt_len:] = action_tokens

        print(f"\nSample {i+1}:")
        print(f"  Instruction: {sample['instruction'][:40]}...")
        print(f"  Original action: {sample['action'][:3]}...")
        print(f"  Input length: {len(input_ids)} (prompt: {prompt_len}, action: {len(action_tokens)})")
        print(f"  Action tokens: {action_tokens.tolist()}")
        print(f"  Labels non-(-100): {(labels != -100).sum().item()} tokens")

        # Verify labels are set correctly
        if (labels != -100).sum().item() == 7:
            print_pass("Labels correctly set for 7 action tokens")
        else:
            print_fail(f"Labels incorrect: expected 7 non-(-100), got {(labels != -100).sum().item()}")
            all_passed = False

    return all_passed, samples


def test_4_forward_pass(model, processor, tokenizer, device="cuda:0"):
    """Test 4: Forward pass produces valid output."""
    print_section("TEST 4: Forward Pass")

    model = model.to(device).eval()
    samples = create_test_samples(1)
    sample = samples[0]

    prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"
    inputs = processor(prompt, sample['image'], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        generated = outputs[0, input_len:]

        print(f"Generated {len(generated)} tokens: {generated.tolist()}")

        # Check if tokens are in action range
        action_tokens = generated[:7]
        in_range = all(31744 <= t <= 32000 for t in action_tokens.tolist())

        if in_range:
            decoded = tokenizer.decode(action_tokens)
            print_pass(f"Generated valid action tokens, decoded: {decoded}")
        else:
            print_warn(f"Generated tokens not in action range (31744-32000)")
            print(f"  This is expected for base model on out-of-domain data")

        return True
    except Exception as e:
        print_fail(f"Forward pass failed: {e}")
        return False


def test_5_loss_computation(model, processor, tokenizer, device="cuda:0"):
    """Test 5: Loss is computed on action tokens."""
    print_section("TEST 5: Loss Computation")

    model = model.to(device).train()
    samples = create_test_samples(1)
    sample = samples[0]

    # Prepare inputs with labels
    prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"
    inputs = processor(prompt, sample['image'], return_tensors="pt")

    action_tokens = torch.tensor(tokenizer.encode(sample['action']), dtype=torch.long)

    # Build full input
    input_ids = torch.cat([inputs['input_ids'].squeeze(), action_tokens]).unsqueeze(0)
    attention_mask = torch.cat([
        inputs['attention_mask'].squeeze(),
        torch.ones(len(action_tokens), dtype=torch.long)
    ]).unsqueeze(0)
    pixel_values = inputs['pixel_values']

    # Create labels
    prompt_len = inputs['input_ids'].shape[1]
    labels = torch.full_like(input_ids, -100)
    labels[0, prompt_len:] = action_tokens

    # Move to device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    pixel_values = pixel_values.to(device).to(torch.bfloat16)
    labels = labels.to(device)

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss
        print(f"Loss value: {loss.item():.4f}")

        if loss.item() > 0 and not torch.isnan(loss) and not torch.isinf(loss):
            print_pass(f"Loss computed correctly: {loss.item():.4f}")
            return True, loss.item()
        else:
            print_fail(f"Invalid loss: {loss.item()}")
            return False, None
    except Exception as e:
        print_fail(f"Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_6_training_step(model, processor, tokenizer, device="cuda:0"):
    """Test 6: Single training step reduces loss."""
    print_section("TEST 6: Training Step (Does Loss Decrease?)")

    from peft import LoraConfig, get_peft_model, TaskType

    model = model.to(device)

    # Add LoRA
    print("Adding LoRA adapters...")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Create batch
    samples = create_test_samples(4)

    losses = []
    for step in range(10):
        total_loss = 0
        for sample in samples:
            prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"
            inputs = processor(prompt, sample['image'], return_tensors="pt")

            action_tokens = torch.tensor(tokenizer.encode(sample['action']), dtype=torch.long)

            input_ids = torch.cat([inputs['input_ids'].squeeze(), action_tokens]).unsqueeze(0).to(device)
            attention_mask = torch.cat([
                inputs['attention_mask'].squeeze(),
                torch.ones(len(action_tokens), dtype=torch.long)
            ]).unsqueeze(0).to(device)
            pixel_values = inputs['pixel_values'].to(device).to(torch.bfloat16)

            prompt_len = inputs['input_ids'].shape[1]
            labels = torch.full_like(input_ids, -100)
            labels[0, prompt_len:] = action_tokens.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )

            loss = outputs.loss / len(samples)
            loss.backward()
            total_loss += loss.item()

        optimizer.step()
        optimizer.zero_grad()
        losses.append(total_loss)
        print(f"  Step {step+1}: Loss = {total_loss:.4f}")

    # Check if loss decreased
    if losses[-1] < losses[0]:
        reduction = (losses[0] - losses[-1]) / losses[0] * 100
        print_pass(f"Loss decreased from {losses[0]:.4f} to {losses[-1]:.4f} ({reduction:.1f}% reduction)")
        return True, model
    else:
        print_fail(f"Loss did NOT decrease: {losses[0]:.4f} -> {losses[-1]:.4f}")
        return False, model


def test_7_inference_after_training(model, processor, tokenizer, device="cuda:0"):
    """Test 7: Trained model produces different outputs."""
    print_section("TEST 7: Inference After Training")

    model = model.to(device).eval()
    samples = create_test_samples(3)

    print("Generating actions for test samples after training:")

    all_same = True
    prev_tokens = None

    for i, sample in enumerate(samples):
        prompt = f"In: What action should the robot take to {sample['instruction']}?\nOut:"
        inputs = processor(prompt, sample['image'], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        input_len = inputs['input_ids'].shape[1]
        action_tokens = outputs[0, input_len:input_len + 7]
        decoded = tokenizer.decode(action_tokens)

        print(f"\nSample {i+1}: {sample['instruction'][:30]}...")
        print(f"  Expected: {sample['action'][:4]}...")
        print(f"  Tokens:   {action_tokens.tolist()}")
        print(f"  Decoded:  {decoded[:4]}...")

        if prev_tokens is not None and action_tokens.tolist() != prev_tokens:
            all_same = False
        prev_tokens = action_tokens.tolist()

    if not all_same:
        print_pass("Model produces different outputs for different inputs")
    else:
        print_fail("Model produces SAME output for all inputs (mode collapse)")

    return not all_same


def main():
    print("\n" + "="*60)
    print(" OpenVLA PIPELINE DEBUGGER")
    print("="*60)
    print("This will test each component of the training pipeline.")
    print("Uses controlled synthetic data to isolate issues.\n")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results = {}

    # Test 1: Model Loading
    model, processor = test_1_model_loading()
    results['model_loading'] = model is not None
    if not results['model_loading']:
        print("\n❌ Cannot continue without model. Exiting.")
        return

    # Test 2: Action Tokenization
    passed, tokenizer = test_2_action_tokenization()
    results['tokenization'] = passed

    # Test 3: Data Preparation
    passed, samples = test_3_data_preparation(processor, tokenizer)
    results['data_prep'] = passed

    # Test 4: Forward Pass
    passed = test_4_forward_pass(model, processor, tokenizer, device)
    results['forward_pass'] = passed

    # Test 5: Loss Computation
    passed, initial_loss = test_5_loss_computation(model, processor, tokenizer, device)
    results['loss_computation'] = passed

    # Test 6: Training Step
    passed, trained_model = test_6_training_step(model, processor, tokenizer, device)
    results['training_step'] = passed

    # Test 7: Inference After Training
    passed = test_7_inference_after_training(trained_model, processor, tokenizer, device)
    results['post_training'] = passed

    # Summary
    print_section("SUMMARY")
    all_passed = True
    for test_name, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print(f"{GREEN}ALL TESTS PASSED{RESET}")
        print("The pipeline components are working correctly.")
        print("If fine-tuning still doesn't work, check:")
        print("  1. Training hyperparameters (learning rate, epochs)")
        print("  2. Data loading in actual training script")
        print("  3. LIBERO HDF5 file format vs what script expects")
    else:
        print(f"{RED}SOME TESTS FAILED{RESET}")
        print("Check the failed tests above for specific issues.")
    print("="*60)

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
