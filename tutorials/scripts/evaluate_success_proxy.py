#!/usr/bin/env python
"""
OpenVLA Success Rate Proxy Evaluation on Bridge V2

Evaluates OpenVLA using trajectory-based success proxy metrics to estimate
task success rates without physical robot execution.

Paper reports: 70.6% +/- 3.2% success rate on real robot (closed-loop)
Our proxy should be in 50-80% range for valid pipeline.

Usage:
    python evaluate_success_proxy.py [--num-episodes 20] [--threshold moderate]
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import pickle
from tqdm import tqdm
from collections import defaultdict

# =============================================================================
# Version Check (CRITICAL)
# =============================================================================
def check_versions():
    import transformers
    import tokenizers

    if transformers.__version__ != "4.40.1":
        print("=" * 60)
        print(" CRITICAL: Wrong transformers version!")
        print("=" * 60)
        print(f"  Installed: {transformers.__version__}")
        print(f"  Required:  4.40.1")
        print()
        print("  Fix: pip install transformers==4.40.1 tokenizers==0.19.1")
        print("=" * 60)
        return False
    return True

if not check_versions():
    print("\nContinuing anyway, but results may be invalid...")

# =============================================================================
# Configuration
# =============================================================================
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


# =============================================================================
# Success Proxy Evaluator
# =============================================================================
class SuccessProxyEvaluator:
    """Evaluate trajectory quality and estimate success probability."""

    THRESHOLDS = {
        'strict': {
            'l1_error_max': 0.20,
            'sign_accuracy_min': 0.65,
            'position_corr_min': 0.30,
            'gripper_accuracy_min': 0.70,
        },
        'moderate': {
            'l1_error_max': 0.30,
            'sign_accuracy_min': 0.55,
            'position_corr_min': 0.20,
            'gripper_accuracy_min': 0.60,
        },
        'lenient': {
            'l1_error_max': 0.40,
            'sign_accuracy_min': 0.50,
            'position_corr_min': 0.10,
            'gripper_accuracy_min': 0.50,
        }
    }

    def __init__(self, threshold_level='moderate'):
        self.thresholds = self.THRESHOLDS[threshold_level]
        self.threshold_level = threshold_level

    def compute_episode_metrics(self, pred_actions, gt_actions):
        """Compute detailed metrics for an episode."""
        T = len(pred_actions)

        # L1 Error
        l1_errors = np.abs(pred_actions - gt_actions)
        l1_mean = l1_errors.mean()
        l1_per_dim = l1_errors.mean(axis=0)

        # Sign Accuracy
        significant_mask = np.abs(gt_actions) > 0.05
        sign_match = (np.sign(pred_actions) == np.sign(gt_actions))
        if significant_mask.sum() > 0:
            sign_accuracy = sign_match[significant_mask].mean()
        else:
            sign_accuracy = sign_match.mean()

        # Correlation per dimension
        correlations = []
        for dim in range(7):
            gt_dim = gt_actions[:, dim]
            pred_dim = pred_actions[:, dim]
            if np.std(gt_dim) > 0.01 and np.std(pred_dim) > 0.01:
                corr = np.corrcoef(pred_dim, gt_dim)[0, 1]
                correlations.append(corr if not np.isnan(corr) else 0)
            else:
                correlations.append(0)

        correlations = np.array(correlations)
        position_corr = correlations[:3].mean()

        # Gripper Accuracy
        gripper_sign_match = (np.sign(pred_actions[:, 6]) == np.sign(gt_actions[:, 6]))
        gripper_accuracy = gripper_sign_match.mean()

        # Final position error
        pred_traj = np.cumsum(pred_actions[:, :3], axis=0)
        gt_traj = np.cumsum(gt_actions[:, :3], axis=0)
        final_position_error = np.linalg.norm(pred_traj[-1] - gt_traj[-1])

        return {
            'l1_mean': l1_mean,
            'sign_accuracy': sign_accuracy,
            'correlations': correlations,
            'position_corr': position_corr,
            'gripper_accuracy': gripper_accuracy,
            'final_position_error': final_position_error,
            'num_steps': T,
        }

    def evaluate_success(self, metrics):
        """Determine if episode would likely succeed."""
        th = self.thresholds

        checks = {
            'l1_error': metrics['l1_mean'] <= th['l1_error_max'],
            'sign_accuracy': metrics['sign_accuracy'] >= th['sign_accuracy_min'],
            'position_corr': metrics['position_corr'] >= th['position_corr_min'],
            'gripper_accuracy': metrics['gripper_accuracy'] >= th['gripper_accuracy_min'],
        }

        passed = sum(checks.values())

        confidence_factors = [
            1 - min(metrics['l1_mean'] / th['l1_error_max'], 1.5) / 1.5,
            metrics['sign_accuracy'],
            max(0, metrics['position_corr']),
            metrics['gripper_accuracy'],
        ]
        confidence = np.mean(confidence_factors)

        success = (passed >= 3) or (confidence > 0.6)

        return success, confidence

    def compute_success_score(self, metrics):
        """Compute continuous success score (0-100%)."""
        th = self.thresholds

        l1_score = max(0, 1 - metrics['l1_mean'] / th['l1_error_max'])
        sign_score = metrics['sign_accuracy']
        corr_score = max(0, (metrics['position_corr'] - th['position_corr_min']) /
                        (1 - th['position_corr_min']) + th['position_corr_min'])
        gripper_score = metrics['gripper_accuracy']

        weights = [0.25, 0.25, 0.20, 0.30]
        scores = [l1_score, sign_score, corr_score, gripper_score]

        return sum(w * s for w, s in zip(weights, scores)) * 100


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


def normalize_action(action, stats):
    """Normalize action to [-1, 1]."""
    q01 = np.array(stats['q01'])
    q99 = np.array(stats['q99'])
    action = np.clip(action, q01, q99)
    return 2 * (action - q01) / (q99 - q01 + 1e-8) - 1


def download_bridge_episodes(num_episodes=20, max_steps=50):
    """Download episodes from Bridge V2."""
    import tensorflow_datasets as tfds

    print(f"Downloading {num_episodes} episodes from Bridge V2...")

    builder = tfds.builder_from_directory(
        builder_dir="gs://gresearch/robotics/bridge/0.1.0"
    )
    dataset = builder.as_dataset(split="train")

    episodes = []
    seen_instructions = set()

    for episode_data in tqdm(dataset, desc="Processing", total=num_episodes * 5):
        if len(episodes) >= num_episodes:
            break

        steps = list(episode_data['steps'])
        if len(steps) < 15:
            continue

        # Get instruction
        obs = steps[0]['observation']
        if 'natural_language_instruction' not in obs:
            continue

        inst = obs['natural_language_instruction']
        if hasattr(inst, 'numpy'):
            inst = inst.numpy()
        if isinstance(inst, bytes):
            inst = inst.decode('utf-8')

        if not inst:
            continue

        inst_key = inst.lower().strip()[:30]
        if inst_key in seen_instructions and len(episodes) < num_episodes // 2:
            continue
        seen_instructions.add(inst_key)

        episode = {'instruction': inst, 'frames': [], 'actions': []}

        for step in steps[:max_steps]:
            obs = step['observation']

            # Extract image
            img_data = obs.get('image', obs.get('image_0'))
            if img_data is None:
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
                action = np.array(action_parts, dtype=np.float32)
            else:
                action = np.array(action_data.numpy() if hasattr(action_data, 'numpy') else action_data)

            if len(action) < 7:
                action = np.pad(action, (0, 7 - len(action)))
            else:
                action = action[:7]

            episode['frames'].append(img.astype(np.uint8))
            episode['actions'].append(action.astype(np.float32))

        if len(episode['frames']) >= 15:
            episodes.append(episode)
            print(f"  [{len(episodes):2d}] '{inst[:50]}...' ({len(episode['frames'])} steps)")

    return episodes


def run_evaluation(episode, model, processor, action_tokenizer, device, bridge_stats, evaluator):
    """Run inference and compute metrics for one episode."""
    instruction = episode['instruction']
    frames = episode['frames'][::2]  # Subsample
    gt_actions_raw = np.array(episode['actions'][::2])

    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    predicted_actions = []

    for frame in frames:
        image = Image.fromarray(frame)

        inputs = processor(prompt, image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        if inputs['input_ids'][0, -1] != 29871:
            empty_token = torch.tensor([[29871]], device=device)
            inputs['input_ids'] = torch.cat([inputs['input_ids'], empty_token], dim=1)
            if 'attention_mask' in inputs:
                inputs['attention_mask'] = torch.cat([
                    inputs['attention_mask'],
                    torch.ones((1, 1), device=device, dtype=inputs['attention_mask'].dtype)
                ], dim=1)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        action_tokens = outputs[0, -7:]
        action = action_tokenizer.decode(action_tokens)
        predicted_actions.append(action)

    predicted_actions = np.array(predicted_actions)
    gt_actions_norm = np.array([normalize_action(a, bridge_stats) for a in gt_actions_raw])

    metrics = evaluator.compute_episode_metrics(predicted_actions, gt_actions_norm)
    success, confidence = evaluator.evaluate_success(metrics)
    success_score = evaluator.compute_success_score(metrics)

    return {
        'instruction': instruction,
        'metrics': metrics,
        'success': success,
        'confidence': confidence,
        'success_score': success_score,
    }


def main():
    parser = argparse.ArgumentParser(description='OpenVLA Success Proxy Evaluation')
    parser.add_argument('--num-episodes', type=int, default=20, help='Number of episodes')
    parser.add_argument('--threshold', choices=['strict', 'moderate', 'lenient'],
                        default='moderate', help='Threshold level')
    args = parser.parse_args()

    print("=" * 70)
    print(" OpenVLA Success Rate Proxy Evaluation")
    print("=" * 70)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print(f"Threshold level: {args.threshold}")

    # Load or download episodes
    cache_file = f"{CACHE_DIR}/bridge_v2_episodes_extended.pkl"
    if os.path.exists(cache_file):
        print(f"\nLoading cached episodes...")
        with open(cache_file, 'rb') as f:
            episodes = pickle.load(f)
    else:
        episodes = download_bridge_episodes(args.num_episodes)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(episodes, f)

    episodes = episodes[:args.num_episodes]
    print(f"Evaluating {len(episodes)} episodes")

    # Load model
    print("\nLoading OpenVLA model...")
    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model = model.to(device).eval()

    processor = AutoProcessor.from_pretrained(
        "openvla/openvla-7b",
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
    )

    action_tokenizer = ActionTokenizer()
    evaluator = SuccessProxyEvaluator(args.threshold)

    # Get normalization stats
    bridge_keys = [k for k in model.config.norm_stats.keys() if 'bridge' in k.lower()]
    bridge_key = bridge_keys[0] if bridge_keys else list(model.config.norm_stats.keys())[0]
    bridge_stats = model.config.norm_stats[bridge_key]['action']

    # Run evaluation
    print("\n" + "=" * 70)
    print(" Running Evaluation")
    print("=" * 70)

    results = []
    for i, episode in enumerate(episodes):
        print(f"\n[{i+1:2d}/{len(episodes)}] {episode['instruction'][:50]}...")

        result = run_evaluation(episode, model, processor, action_tokenizer,
                               device, bridge_stats, evaluator)
        results.append(result)

        status = "PASS" if result['success'] else "FAIL"
        print(f"        {status} (score: {result['success_score']:.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print(" RESULTS SUMMARY")
    print("=" * 70)

    success_rate = sum(1 for r in results if r['success']) / len(results) * 100
    avg_score = np.mean([r['success_score'] for r in results])
    std_score = np.std([r['success_score'] for r in results])

    print(f"\nBinary Success Rate: {success_rate:.1f}%")
    print(f"Average Success Score: {avg_score:.1f}% +/- {std_score:.1f}%")
    print(f"\nPaper reports: 70.6% +/- 3.2% (real robot, closed-loop)")

    # Validation
    print("\n" + "=" * 70)
    if 50 <= success_rate <= 80:
        print(" VERDICT: PIPELINE VALIDATED")
        print(" Success rate is within expected range (50-80%)")
    elif success_rate < 50:
        print(" VERDICT: NEEDS INVESTIGATION")
        print(" Success rate below expected - check versions and pipeline")
    else:
        print(" VERDICT: UNEXPECTEDLY HIGH")
        print(" Success rate above expected - may indicate evaluation issues")
    print("=" * 70)

    # Save results
    results_path = f"{CACHE_DIR}/success_proxy_results.pkl"
    with open(results_path, 'wb') as f:
        pickle.dump({
            'results': results,
            'success_rate': success_rate,
            'avg_score': avg_score,
            'threshold': args.threshold,
        }, f)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
