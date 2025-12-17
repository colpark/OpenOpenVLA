#!/usr/bin/env python
"""
Evaluate Fine-tuned OpenVLA on LIBERO

This script evaluates a fine-tuned OpenVLA model on LIBERO tasks
using simulation rollouts to measure task success rate.

Usage:
    # Evaluate on libero_spatial (same suite as training)
    python evaluate_finetuned.py --checkpoint /path/to/checkpoint

    # Evaluate on held-out tasks (libero_10)
    python evaluate_finetuned.py --checkpoint /path/to/checkpoint --suite libero_10

    # Quick test (1 trial per task)
    python evaluate_finetuned.py --checkpoint /path/to/checkpoint --trials 1
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import LIBERO early (before transformers trust_remote_code can interfere)
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv

# ============================================================
# Configuration
# ============================================================
if os.environ.get('PSCRATCH'):
    BASE_DIR = os.environ['PSCRATCH']
elif os.environ.get('SCRATCH'):
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"

# ============================================================
# Model Loading
# ============================================================
def load_finetuned_model(checkpoint_path, device="cuda:0"):
    """
    Load a fine-tuned OpenVLA model with LoRA weights.

    Args:
        checkpoint_path: Path to the fine-tuned checkpoint
        device: Device to load model on

    Returns:
        model, processor
    """
    from transformers import AutoModelForVision2Seq, AutoProcessor

    checkpoint_path = Path(checkpoint_path)
    print(f"Loading fine-tuned model from: {checkpoint_path}")

    # Debug: Show checkpoint contents
    if checkpoint_path.exists():
        print(f"\nCheckpoint contents:")
        for f in sorted(checkpoint_path.iterdir()):
            size = f.stat().st_size / 1024 / 1024  # MB
            print(f"  {f.name} ({size:.1f} MB)")
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Detect checkpoint type
    has_adapter_config = (checkpoint_path / "adapter_config.json").exists()
    has_adapter_model = (checkpoint_path / "adapter_model.safetensors").exists() or \
                        (checkpoint_path / "adapter_model.bin").exists()
    has_model_safetensors = (checkpoint_path / "model.safetensors").exists()
    has_pytorch_model = (checkpoint_path / "pytorch_model.bin").exists()

    print(f"\nCheckpoint type detection:")
    print(f"  adapter_config.json: {has_adapter_config}")
    print(f"  adapter_model.*: {has_adapter_model}")
    print(f"  model.safetensors: {has_model_safetensors}")
    print(f"  pytorch_model.bin: {has_pytorch_model}")

    # Determine loading strategy
    is_lora = has_adapter_config and has_adapter_model
    is_full_model = has_model_safetensors or has_pytorch_model

    if is_lora:
        from peft import PeftModel

        print("\nDetected LoRA checkpoint - loading base model + adapters...")

        # Load base model
        print("Loading base model...")
        base_model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
            attn_implementation="eager",
        )

        # Load LoRA weights
        print("Loading LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))

        # Merge for faster inference
        print("Merging LoRA weights...")
        model = model.merge_and_unload()

    elif is_full_model:
        print("\nDetected full model checkpoint - loading directly...")
        model = AutoModelForVision2Seq.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="eager",
        )
    else:
        # Fallback: try loading as LoRA anyway (might be HF Trainer checkpoint)
        print("\nUnknown checkpoint format - attempting LoRA load...")
        print("If this fails, check that training completed and saved properly.")

        from peft import PeftModel

        base_model = AutoModelForVision2Seq.from_pretrained(
            "openvla/openvla-7b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            cache_dir=f"{CACHE_DIR}/huggingface",
            attn_implementation="eager",
        )

        try:
            model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
            model = model.merge_and_unload()
        except Exception as e:
            print(f"\nError loading checkpoint: {e}")
            print("\nFalling back to base model (NOT fine-tuned!)...")
            print("WARNING: This will give zero-shot performance, not fine-tuned!")
            model = base_model

    model.to(device)
    model.eval()

    # Load processor
    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
    )

    print(f"\nModel loaded on {device}")
    return model, processor


# ============================================================
# Policy Wrapper
# ============================================================
class OpenVLAPolicy:
    """
    Policy wrapper for OpenVLA inference in LIBERO.
    """

    # Action tokenization constants (must match OpenVLA's ActionTokenizer)
    N_ACTION_BINS = 256
    MIN_ACTION = -1.0
    MAX_ACTION = 1.0

    def __init__(self, model, processor, device="cuda:0", unnorm_key=None, use_raw_actions=True):
        self.model = model
        self.processor = processor
        self.device = device
        self.unnorm_key = unnorm_key
        self.use_raw_actions = use_raw_actions  # Skip unnormalization for fine-tuned models

        # Get vocab size for action token decoding
        self.vocab_size = len(processor.tokenizer)

        # Create bin centers matching OpenVLA's ActionTokenizer for decoding
        bins = np.linspace(self.MIN_ACTION, self.MAX_ACTION, self.N_ACTION_BINS)
        self.bin_centers = (bins[:-1] + bins[1:]) / 2.0  # 255 bin centers

        print(f"Vocab size: {self.vocab_size}")
        print(f"Action bins: {self.N_ACTION_BINS}, bin_centers: {len(self.bin_centers)}")

    def decode_actions_raw(self, action_token_ids, action_dim=7):
        """
        Decode action tokens to continuous values using OpenVLA's convention.

        IMPORTANT: Must match OpenVLA's ActionTokenizer.decode_token_ids_to_actions()
        OpenVLA encoding: token_id = vocab_size - discretized_action
        OpenVLA decoding: discretized_action = vocab_size - token_id
                          action = bin_centers[clip(discretized_action - 1, 0, 254)]

        Args:
            action_token_ids: Token IDs from model generation
            action_dim: Number of action dimensions (default 7 for LIBERO)

        Returns:
            numpy array of continuous actions
        """
        # Reverse the encoding: discretized_action = vocab_size - token_id
        discretized_actions = self.vocab_size - action_token_ids.cpu().numpy()

        # Adjust indices and clip to valid range (matching OpenVLA's decode logic)
        # digitize returns [1, 256], we need indices [0, 254] for bin_centers
        discretized_actions = np.clip(discretized_actions - 1, 0, len(self.bin_centers) - 1)

        # Get continuous actions from bin centers
        actions = self.bin_centers[discretized_actions]

        return actions[:action_dim]

    def predict(self, obs, instruction):
        """
        Predict action from observation and instruction.

        Args:
            obs: Dictionary with image observation
            instruction: Natural language task instruction

        Returns:
            action: 7-DoF action array
        """
        # Get image from observation - LIBERO uses 'agentview_image' key
        if isinstance(obs, dict):
            # Try different possible keys for the image
            image = None
            for key in ['agentview_image', 'agentview_rgb', 'image', 'pixels']:
                if key in obs and obs[key] is not None:
                    image = obs[key]
                    break

            if image is None:
                # Debug: print available keys
                print(f"Warning: No image found in obs. Available keys: {list(obs.keys())}")
                # Return zero action if no image
                return np.zeros(7)
        else:
            image = obs

        # Ensure image is a proper array
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            print(f"Warning: Invalid image type={type(image)}, ndim={getattr(image, 'ndim', 'N/A')}")
            return np.zeros(7)

        # Rotate image (LIBERO convention)
        image = np.rot90(image, k=2)

        # Convert to PIL
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Format prompt
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        # Process inputs
        inputs = self.processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Convert pixel_values to bfloat16
        if 'pixel_values' in inputs:
            inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        # Predict action
        with torch.no_grad():
            if self.use_raw_actions:
                # Generate action tokens directly and decode without unnormalization
                # This is better for fine-tuned models on new datasets
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=7,  # 7 action dimensions
                    do_sample=False,
                )
                # Extract action tokens (last 7 tokens)
                action_token_ids = generated_ids[0, -7:]
                action = self.decode_actions_raw(action_token_ids)
            else:
                # Use OpenVLA's predict_action with unnormalization
                action = self.model.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_key,
                    do_sample=False,
                )
                action = np.array(action)

        # Gripper inversion note:
        # - For fine-tuned models (use_raw_actions=True): No inversion needed
        #   because we trained on raw LIBERO actions which are already in LIBERO's convention
        # - For base model (use_raw_actions=False): May need inversion
        #   because base model was pre-trained on Bridge/other datasets with different convention
        if not self.use_raw_actions and len(action) >= 7:
            action[6] = -action[6]

        return action


# ============================================================
# Evaluation Functions
# ============================================================
def create_libero_env(task, benchmark_instance, image_size=256):
    """Create LIBERO environment for a task."""
    task_id = benchmark_instance.get_task_names().index(task.name)
    bddl_file = benchmark_instance.get_task_bddl_file_path(task_id)

    env_args = {
        "bddl_file_name": bddl_file,
        "camera_heights": image_size,
        "camera_widths": image_size,
    }

    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env


def evaluate_task(policy, task, benchmark_instance, n_trials=10, max_steps=300):
    """
    Evaluate policy on a single task.

    Args:
        policy: OpenVLAPolicy instance
        task: LIBERO task object
        benchmark_instance: LIBERO benchmark instance
        n_trials: Number of evaluation trials
        max_steps: Maximum steps per episode

    Returns:
        dict with success_rate, successes, trials
    """
    env = create_libero_env(task, benchmark_instance)
    instruction = task.language

    successes = 0

    for trial in range(n_trials):
        obs = env.reset()
        done = False

        for step in range(max_steps):
            # Get action from policy
            action = policy.predict(obs, instruction)

            # Clip action to valid range
            action = np.clip(action, -1, 1)

            # Step environment
            obs, reward, done, info = env.step(action)

            if done:
                break

        # Check success
        if info.get('success', False) or reward > 0:
            successes += 1

    env.close()

    return {
        'task': task.name,
        'instruction': instruction,
        'successes': successes,
        'trials': n_trials,
        'success_rate': successes / n_trials,
    }


def evaluate_suite(policy, suite_name, n_trials=10, max_tasks=None):
    """
    Evaluate policy on an entire LIBERO suite.

    Args:
        policy: OpenVLAPolicy instance
        suite_name: Name of LIBERO suite (e.g., 'libero_spatial', 'libero_10')
        n_trials: Number of trials per task
        max_tasks: Maximum number of tasks to evaluate (None for all)

    Returns:
        dict with per-task results and overall success rate
    """
    print(f"\nEvaluating on {suite_name}")
    print("=" * 60)

    # Get benchmark
    BenchmarkClass = benchmark.get_benchmark(suite_name)
    bench = BenchmarkClass()

    tasks = bench.get_task_names()
    if max_tasks:
        tasks = tasks[:max_tasks]

    print(f"Tasks: {len(tasks)}")
    print(f"Trials per task: {n_trials}")

    results = []

    for task_name in tqdm(tasks, desc=f"Evaluating {suite_name}"):
        # Get task object
        task_id = bench.get_task_names().index(task_name)
        task = bench.get_task(task_id)

        # Evaluate
        result = evaluate_task(policy, task, bench, n_trials=n_trials)
        results.append(result)

        print(f"  {task_name}: {result['success_rate']*100:.1f}% ({result['successes']}/{result['trials']})")

    # Calculate overall success rate
    total_successes = sum(r['successes'] for r in results)
    total_trials = sum(r['trials'] for r in results)
    overall_success_rate = total_successes / total_trials

    return {
        'suite': suite_name,
        'tasks': results,
        'overall_success_rate': overall_success_rate,
        'total_successes': total_successes,
        'total_trials': total_trials,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned OpenVLA on LIBERO")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--suite", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal",
                                 "libero_10", "libero_90"],
                        help="LIBERO suite to evaluate on")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of trials per task")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="Maximum tasks to evaluate (None for all)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to run evaluation on")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")
    parser.add_argument("--unnorm-key", type=str, default="bridge_orig",
                        help="Unnormalization key for action decoding (default: bridge_orig)")
    parser.add_argument("--raw-actions", action="store_true", default=True,
                        help="Use raw action decoding without unnormalization (recommended for fine-tuned models)")
    parser.add_argument("--no-raw-actions", action="store_false", dest="raw_actions",
                        help="Use OpenVLA's unnormalization (for base model evaluation)")

    args = parser.parse_args()

    print(f"\nAction decoding mode: {'RAW (no unnorm)' if args.raw_actions else f'UNNORM ({args.unnorm_key})'}")

    # Load model
    model, processor = load_finetuned_model(args.checkpoint, args.device)

    # Create policy
    policy = OpenVLAPolicy(model, processor, args.device, args.unnorm_key, use_raw_actions=args.raw_actions)

    # Evaluate
    results = evaluate_suite(
        policy,
        args.suite,
        n_trials=args.trials,
        max_tasks=args.max_tasks,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Suite: {results['suite']}")
    print(f"Overall Success Rate: {results['overall_success_rate']*100:.1f}%")
    print(f"Total: {results['total_successes']}/{results['total_trials']}")
    print()

    print("Per-task results:")
    for task_result in results['tasks']:
        print(f"  {task_result['task']}: {task_result['success_rate']*100:.1f}%")

    # Save results
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{BASE_DIR}/openvla_finetune/eval_{args.suite}_{timestamp}.json"

    results['checkpoint'] = args.checkpoint
    results['timestamp'] = datetime.now().isoformat()

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Expected results reference
    print("\n" + "=" * 60)
    print("EXPECTED SUCCESS RATES (for reference)")
    print("=" * 60)
    print("Zero-shot (no fine-tuning): 0-10%")
    print("After fine-tuning: 70-80%")
    print("Paper reported (LIBERO-Spatial): 84.7%")


if __name__ == "__main__":
    main()
