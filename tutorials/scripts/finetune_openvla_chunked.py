#!/usr/bin/env python
"""
OpenVLA Fine-tuning on LIBERO with Action Chunking (Temporal Subsampling)

Addresses the control frequency mismatch by subsampling LIBERO's 20 Hz to ~5 Hz
(matching Bridge V2's control frequency used in OpenVLA training).

Key Features:
- Action chunking: Uses every Nth frame (default N=4: 20 Hz → 5 Hz)
- Comprehensive metric logging (L1 error, direction accuracy, gripper accuracy)
- Validation during training with configurable frequency
- Results saved to structured folder with figures

Usage:
    python finetune_openvla_chunked.py --suite libero_spatial --chunk-size 4
    python finetune_openvla_chunked.py --resume results/run_xxx/checkpoints/checkpoint-1000

Frame Rate Reference:
    - Fractal (OpenVLA training): 3 Hz
    - Bridge V2 (OpenVLA training): 5 Hz
    - LIBERO (raw): 20 Hz
    - LIBERO (chunked, N=4): 5 Hz ✓ matches Bridge V2
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from functools import partial
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
from tqdm import tqdm

# =============================================================================
# Version Check
# =============================================================================
def check_versions():
    """Check for version compatibility issues."""
    import transformers
    import tokenizers

    issues = []

    if transformers.__version__ != "4.40.1":
        issues.append(f"transformers=={transformers.__version__} (need 4.40.1)")

    if tokenizers.__version__ != "0.19.1":
        issues.append(f"tokenizers=={tokenizers.__version__} (need 0.19.1)")

    try:
        import timm
        if not timm.__version__.startswith("0.9."):
            issues.append(f"timm=={timm.__version__} (need 0.9.x)")
    except ImportError:
        issues.append("timm not installed (need 0.9.x)")

    if issues:
        print("=" * 60)
        print(" VERSION INCOMPATIBILITY DETECTED")
        print("=" * 60)
        for issue in issues:
            print(f"  - {issue}")
        print("\nTo fix:")
        print("  pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.16")
        print("=" * 60)
        return False
    return True


# =============================================================================
# Configuration
# =============================================================================
if 'SCRATCH' in os.environ:
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"

CACHE_DIR = f"{BASE_DIR}/.cache"
LIBERO_DATA_DIR = f"{BASE_DIR}/libero_data"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")

os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"
os.environ['TORCH_HOME'] = f"{CACHE_DIR}/torch"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Results Manager
# =============================================================================
class ResultsManager:
    """Manages logging, checkpoints, and figures for a training run."""

    def __init__(self, run_dir):
        self.run_dir = Path(run_dir)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.figures_dir = self.run_dir / "figures"

        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(exist_ok=True)
        self.figures_dir.mkdir(exist_ok=True)

        # Initialize log files
        self.training_log_path = self.run_dir / "training_log.csv"
        self.validation_log_path = self.run_dir / "validation_log.csv"

        # Write headers
        self._init_training_log()
        self._init_validation_log()

        # In-memory storage for plotting
        self.training_history = defaultdict(list)
        self.validation_history = defaultdict(list)

    def _init_training_log(self):
        with open(self.training_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'epoch', 'loss', 'lr', 'timestamp'])

    def _init_validation_log(self):
        with open(self.validation_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'step', 'epoch', 'val_loss', 'l1_error',
                'direction_accuracy', 'gripper_accuracy',
                'position_l1', 'rotation_l1', 'timestamp'
            ])

    def log_training(self, step, epoch, loss, lr):
        """Log training metrics."""
        timestamp = datetime.now().isoformat()

        with open(self.training_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([step, epoch, f'{loss:.6f}', f'{lr:.2e}', timestamp])

        self.training_history['step'].append(step)
        self.training_history['loss'].append(loss)
        self.training_history['lr'].append(lr)

    def log_validation(self, step, epoch, metrics):
        """Log validation metrics."""
        timestamp = datetime.now().isoformat()

        with open(self.validation_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step, epoch,
                f'{metrics["val_loss"]:.6f}',
                f'{metrics["l1_error"]:.6f}',
                f'{metrics["direction_accuracy"]:.4f}',
                f'{metrics["gripper_accuracy"]:.4f}',
                f'{metrics["position_l1"]:.6f}',
                f'{metrics["rotation_l1"]:.6f}',
                timestamp
            ])

        self.validation_history['step'].append(step)
        for k, v in metrics.items():
            self.validation_history[k].append(v)

    def save_config(self, config):
        """Save training configuration."""
        with open(self.run_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def save_summary(self, summary):
        """Save final training summary."""
        with open(self.run_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def generate_figures(self):
        """Generate training/validation curves."""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Training loss curve
            if self.training_history['step']:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(self.training_history['step'], self.training_history['loss'],
                       label='Training Loss', alpha=0.7)
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.set_title('Training Loss Curve')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.savefig(self.figures_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

            # Validation metrics
            if self.validation_history['step']:
                # Loss comparison
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # Validation Loss
                ax = axes[0, 0]
                ax.plot(self.validation_history['step'], self.validation_history['val_loss'],
                       'b-o', label='Val Loss', markersize=4)
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.set_title('Validation Loss')
                ax.grid(True, alpha=0.3)

                # L1 Error
                ax = axes[0, 1]
                ax.plot(self.validation_history['step'], self.validation_history['l1_error'],
                       'g-o', label='L1 Error', markersize=4)
                ax.set_xlabel('Step')
                ax.set_ylabel('L1 Error')
                ax.set_title('Action L1 Error')
                ax.grid(True, alpha=0.3)

                # Direction Accuracy
                ax = axes[1, 0]
                ax.plot(self.validation_history['step'], self.validation_history['direction_accuracy'],
                       'r-o', label='Direction Acc', markersize=4)
                ax.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')
                ax.set_xlabel('Step')
                ax.set_ylabel('Accuracy')
                ax.set_title('Direction Accuracy')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Gripper Accuracy
                ax = axes[1, 1]
                ax.plot(self.validation_history['step'], self.validation_history['gripper_accuracy'],
                       'purple', marker='o', label='Gripper Acc', markersize=4)
                ax.set_xlabel('Step')
                ax.set_ylabel('Accuracy')
                ax.set_title('Gripper Accuracy')
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                fig.savefig(self.figures_dir / 'validation_metrics.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Combined training + validation loss
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(self.training_history['step'], self.training_history['loss'],
                       alpha=0.5, label='Training Loss')
                ax.plot(self.validation_history['step'], self.validation_history['val_loss'],
                       'b-o', label='Validation Loss', markersize=6)
                ax.set_xlabel('Step')
                ax.set_ylabel('Loss')
                ax.set_title('Training vs Validation Loss')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig.savefig(self.figures_dir / 'loss_comparison.png', dpi=150, bbox_inches='tight')
                plt.close(fig)

            print(f"Figures saved to {self.figures_dir}")

        except ImportError:
            print("matplotlib not available, skipping figure generation")
        except Exception as e:
            print(f"Error generating figures: {e}")


# =============================================================================
# Action Tokenizer
# =============================================================================
class ActionTokenizer:
    """OpenVLA-compatible action tokenizer."""

    def __init__(self, vocab_size=32000, n_bins=256, min_action=-1.0, max_action=1.0):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.min_action = min_action
        self.max_action = max_action

        self.bins = np.linspace(min_action, max_action, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

        self.action_token_start = vocab_size - n_bins
        self.action_token_end = vocab_size - 1

    def encode(self, action):
        """Encode continuous action to token IDs."""
        action = np.clip(action, self.min_action, self.max_action)
        discretized = np.digitize(action, self.bins)
        token_ids = self.vocab_size - discretized
        return token_ids

    def decode(self, token_ids):
        """Decode token IDs back to continuous actions."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        discretized = self.vocab_size - token_ids
        indices = np.clip(discretized - 1, 0, len(self.bin_centers) - 1)
        return self.bin_centers[indices]


# =============================================================================
# LIBERO Dataset with Action Chunking
# =============================================================================
class LIBEROChunkedDataset(Dataset):
    """
    LIBERO Dataset with temporal subsampling (action chunking).

    Action Chunking Strategy:
    - Uses every Nth frame to match target control frequency
    - N=4: 20 Hz → 5 Hz (matches Bridge V2)
    - N=7: 20 Hz → ~3 Hz (matches Fractal)

    This addresses the mode collapse issue caused by control frequency mismatch.
    """

    def __init__(self, data_dir, suite_name, processor, action_tokenizer,
                 chunk_size=4, max_samples=None, image_size=224,
                 split='train', val_demos_per_task=5):
        """
        Args:
            chunk_size: Take every Nth frame (default 4 for 20→5 Hz)
            split: 'train' or 'val'
            val_demos_per_task: Number of demos per task for validation
        """
        self.data_dir = Path(data_dir)
        self.suite_name = suite_name
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.split = split
        self.val_demos_per_task = val_demos_per_task

        self.hdf5_files = self._find_hdf5_files()
        if not self.hdf5_files:
            raise FileNotFoundError(f"No HDF5 files found in {data_dir}")

        self.samples = self._build_index(max_samples)

        effective_hz = 20 / chunk_size
        print(f"Dataset [{split}]: {len(self.samples)} samples from {len(self.hdf5_files)} files")
        print(f"  Chunk size: {chunk_size} (20 Hz → {effective_hz:.1f} Hz)")

    def transform_action(self, action):
        """Transform LIBERO action to match OpenVLA preprocessing."""
        action = action.astype(np.float32)
        action[:6] = np.clip(action[:6], -1.0, 1.0)
        gripper = np.clip(action[6], 0.0, 1.0)
        action[6] = 1.0 - gripper
        return action

    def _find_hdf5_files(self):
        """Find all HDF5 files for this suite."""
        patterns = [
            f"**/*{self.suite_name}*/*.hdf5",
            f"**/{self.suite_name}/*.hdf5",
            "**/*.hdf5",
        ]

        for pattern in patterns:
            files = list(self.data_dir.rglob(pattern.replace("**/*", "*")))
            if not files:
                files = list(self.data_dir.glob(pattern))
            if files:
                return sorted(files)
        return []

    def _build_index(self, max_samples):
        """Build index with action chunking (temporal subsampling)."""
        samples = []

        for filepath in tqdm(self.hdf5_files, desc=f"Indexing {self.split} data"):
            try:
                with h5py.File(filepath, 'r') as f:
                    language = self._get_language(f)

                    if 'data' not in f:
                        continue

                    demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])

                    # Split demos for train/val
                    if self.split == 'val':
                        demo_keys = demo_keys[-self.val_demos_per_task:]
                    else:
                        demo_keys = demo_keys[:-self.val_demos_per_task]

                    for demo_key in demo_keys:
                        demo = f['data'][demo_key]

                        if 'actions' not in demo or 'obs' not in demo:
                            continue

                        n_steps = len(demo['actions'])

                        # ACTION CHUNKING: Use every chunk_size-th frame
                        for t in range(0, n_steps, self.chunk_size):
                            samples.append({
                                'filepath': str(filepath),
                                'demo_key': demo_key,
                                'timestep': t,
                                'language': language,
                            })

                            if max_samples and len(samples) >= max_samples:
                                return samples

            except Exception as e:
                print(f"Warning: Error reading {filepath}: {e}")

        return samples

    def _get_language(self, f):
        """Extract language instruction from HDF5 file."""
        for key in ['language_instruction', 'problem_info', 'language']:
            if key in f.attrs:
                lang = f.attrs[key]
                if isinstance(lang, bytes):
                    lang = lang.decode('utf-8')
                return lang
        return "complete the task"

    def _get_image_key(self, obs):
        """Find the image observation key."""
        for key in ['agentview_rgb', 'agentview_image', 'rgb', 'image']:
            if key in obs:
                return key
        for key in obs.keys():
            if 'image' in key.lower() or 'rgb' in key.lower():
                return key
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with h5py.File(sample['filepath'], 'r') as f:
            demo = f['data'][sample['demo_key']]
            t = sample['timestep']

            img_key = self._get_image_key(demo['obs'])
            if img_key is None:
                raise ValueError(f"No image key found in {sample['filepath']}")

            image = demo['obs'][img_key][t]
            image = np.rot90(image, k=2)  # 180° rotation for LIBERO

            action = demo['actions'][t]
            if len(action) < 7:
                action = np.pad(action, (0, 7 - len(action)))
            else:
                action = action[:7]

        pil_image = Image.fromarray(image.astype(np.uint8))
        if pil_image.size != (self.image_size, self.image_size):
            pil_image = pil_image.resize((self.image_size, self.image_size), Image.LANCZOS)

        instruction = sample['language']
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        inputs = self.processor(prompt, pil_image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        action_transformed = self.transform_action(action)
        action_tokens = self.action_tokenizer.encode(action_transformed)
        action_tokens = torch.tensor(action_tokens, dtype=torch.long)

        inputs['input_ids'] = torch.cat([inputs['input_ids'], action_tokens])
        inputs['attention_mask'] = torch.cat([
            inputs['attention_mask'],
            torch.ones(len(action_tokens), dtype=inputs['attention_mask'].dtype)
        ])

        prompt_len = len(inputs['input_ids']) - len(action_tokens)
        labels = torch.full_like(inputs['input_ids'], -100)
        labels[prompt_len:] = action_tokens

        inputs['labels'] = labels

        # Store ground truth action for metric computation
        inputs['gt_action'] = torch.tensor(action_transformed, dtype=torch.float32)

        return inputs


def collate_fn(batch, pad_token_id=0):
    """Custom collate function with RIGHT padding."""
    max_len = max(item['input_ids'].size(0) for item in batch)

    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'pixel_values': [],
        'gt_action': [],
    }

    for item in batch:
        seq_len = item['input_ids'].size(0)
        pad_len = max_len - seq_len

        padded_batch['input_ids'].append(
            torch.cat([item['input_ids'], torch.full((pad_len,), pad_token_id, dtype=torch.long)])
        )
        padded_batch['attention_mask'].append(
            torch.cat([item['attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        )
        padded_batch['labels'].append(
            torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
        )
        padded_batch['pixel_values'].append(item['pixel_values'])
        padded_batch['gt_action'].append(item['gt_action'])

    return {
        'input_ids': torch.stack(padded_batch['input_ids']),
        'attention_mask': torch.stack(padded_batch['attention_mask']),
        'labels': torch.stack(padded_batch['labels']),
        'pixel_values': torch.stack(padded_batch['pixel_values']),
        'gt_action': torch.stack(padded_batch['gt_action']),
    }


# =============================================================================
# Training Functions
# =============================================================================
def setup_lora(model, config):
    """Add LoRA adapters to the model."""
    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        target_modules=config['lora_target_modules'],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model


def compute_metrics(pred_actions, gt_actions):
    """
    Compute comprehensive evaluation metrics.

    Args:
        pred_actions: (N, 7) predicted actions
        gt_actions: (N, 7) ground truth actions

    Returns:
        dict with metrics: l1_error, direction_accuracy, gripper_accuracy, etc.
    """
    if isinstance(pred_actions, torch.Tensor):
        pred_actions = pred_actions.cpu().numpy()
    if isinstance(gt_actions, torch.Tensor):
        gt_actions = gt_actions.cpu().numpy()

    # Overall L1 error
    l1_error = np.abs(pred_actions - gt_actions).mean()

    # Position L1 (dims 0-2)
    position_l1 = np.abs(pred_actions[:, :3] - gt_actions[:, :3]).mean()

    # Rotation L1 (dims 3-5)
    rotation_l1 = np.abs(pred_actions[:, 3:6] - gt_actions[:, 3:6]).mean()

    # Direction accuracy (sign agreement for position dims)
    # Only count where ground truth has meaningful movement
    threshold = 0.02
    direction_correct = 0
    direction_total = 0

    for dim in range(3):  # Position dimensions
        gt_dim = gt_actions[:, dim]
        pred_dim = pred_actions[:, dim]

        # Only evaluate where GT has clear direction
        significant = np.abs(gt_dim) > threshold
        if significant.sum() > 0:
            same_sign = (np.sign(gt_dim[significant]) == np.sign(pred_dim[significant]))
            direction_correct += same_sign.sum()
            direction_total += significant.sum()

    direction_accuracy = direction_correct / direction_total if direction_total > 0 else 0.5

    # Gripper accuracy (dim 6)
    gripper_threshold = 0.5
    gt_gripper_binary = (gt_actions[:, 6] > gripper_threshold).astype(int)
    pred_gripper_binary = (pred_actions[:, 6] > gripper_threshold).astype(int)
    gripper_accuracy = (gt_gripper_binary == pred_gripper_binary).mean()

    return {
        'l1_error': float(l1_error),
        'position_l1': float(position_l1),
        'rotation_l1': float(rotation_l1),
        'direction_accuracy': float(direction_accuracy),
        'gripper_accuracy': float(gripper_accuracy),
    }


def train_step(model, batch, device, gradient_accumulation_steps=1):
    """Single training step."""
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    pixel_values = batch['pixel_values'].to(device).to(torch.bfloat16)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        labels=labels,
    )

    loss = outputs.loss / gradient_accumulation_steps
    loss.backward()

    return outputs.loss.item()


def validate(model, dataloader, device, action_tokenizer, max_samples=200):
    """
    Run validation with comprehensive metrics.

    Args:
        max_samples: Maximum samples for metric computation (for speed)
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    all_pred_actions = []
    all_gt_actions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            pixel_values = batch['pixel_values'].to(device).to(torch.bfloat16)
            gt_actions = batch['gt_action']

            # Compute loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            num_batches += 1

            # Generate predictions for metric computation
            if len(all_pred_actions) * dataloader.batch_size < max_samples:
                for i in range(len(batch['input_ids'])):
                    if len(all_pred_actions) >= max_samples:
                        break

                    try:
                        # Find prompt length (where labels start)
                        label_mask = labels[i] != -100
                        prompt_len = (~label_mask).sum().item()

                        # Use input up to prompt
                        prompt_ids = input_ids[i:i+1, :prompt_len]
                        prompt_mask = attention_mask[i:i+1, :prompt_len]

                        gen_outputs = model.generate(
                            input_ids=prompt_ids,
                            attention_mask=prompt_mask,
                            pixel_values=pixel_values[i:i+1],
                            max_new_tokens=7,
                            do_sample=False,
                            pad_token_id=model.config.pad_token_id,
                        )

                        pred_tokens = gen_outputs[0, -7:].cpu().numpy()
                        pred_action = action_tokenizer.decode(pred_tokens)

                        all_pred_actions.append(pred_action)
                        all_gt_actions.append(gt_actions[i].numpy())

                    except Exception as e:
                        continue

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    # Compute metrics
    if all_pred_actions:
        pred_array = np.array(all_pred_actions)
        gt_array = np.array(all_gt_actions)
        metrics = compute_metrics(pred_array, gt_array)
    else:
        metrics = {
            'l1_error': float('inf'),
            'position_l1': float('inf'),
            'rotation_l1': float('inf'),
            'direction_accuracy': 0.5,
            'gripper_accuracy': 0.5,
        }

    metrics['val_loss'] = avg_loss

    model.train()
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, step, results_manager, config):
    """Save training checkpoint."""
    checkpoint_dir = results_manager.checkpoints_dir / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(checkpoint_dir)

    torch.save({
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
    }, checkpoint_dir / "training_state.pt")

    with open(checkpoint_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Checkpoint saved: {checkpoint_dir}")
    return checkpoint_dir


# =============================================================================
# Main Training Loop
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Fine-tune OpenVLA on LIBERO with action chunking')
    parser.add_argument('--suite', type=str, default='libero_spatial',
                        help='LIBERO suite (default: libero_spatial)')
    parser.add_argument('--chunk-size', type=int, default=4,
                        help='Temporal subsampling factor (default: 4 for 20→5 Hz)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Per-GPU batch size')
    parser.add_argument('--grad-accum', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--lora-r', type=int, default=16,
                        help='LoRA rank')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max training samples (for debugging)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint directory')
    parser.add_argument('--data-dir', type=str, default=LIBERO_DATA_DIR,
                        help='LIBERO data directory')
    parser.add_argument('--results-dir', type=str, default=RESULTS_DIR,
                        help='Results directory')
    parser.add_argument('--val-steps', type=int, default=100,
                        help='Validation frequency (steps)')
    parser.add_argument('--save-steps', type=int, default=500,
                        help='Checkpoint frequency (steps)')
    parser.add_argument('--log-steps', type=int, default=10,
                        help='Training log frequency (steps)')
    parser.add_argument('--val-demos', type=int, default=5,
                        help='Number of demos per task for validation')
    args = parser.parse_args()

    if not check_versions():
        print("\nContinuing anyway, but results may be incorrect...")

    print("=" * 60)
    print(" OpenVLA Fine-tuning with Action Chunking")
    print("=" * 60)
    print(f"\nAction chunking: {args.chunk_size}x subsampling")
    print(f"  20 Hz → {20/args.chunk_size:.1f} Hz (target: ~5 Hz like Bridge V2)")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda:0":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Configuration
    config = {
        'suite': args.suite,
        'chunk_size': args.chunk_size,
        'effective_hz': 20 / args.chunk_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'gradient_accumulation_steps': args.grad_accum,
        'effective_batch_size': args.batch_size * args.grad_accum,
        'learning_rate': args.lr,
        'lora_r': args.lora_r,
        'lora_alpha': min(args.lora_r, 16),
        'lora_dropout': 0.0,
        'lora_target_modules': ["q_proj", "v_proj", "k_proj", "o_proj"],
        'warmup_ratio': 0.03,
        'weight_decay': 0.01,
        'max_grad_norm': 1.0,
        'image_size': 224,
        'val_steps': args.val_steps,
        'save_steps': args.save_steps,
        'val_demos_per_task': args.val_demos,
    }

    print(f"\nConfiguration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Setup results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.suite}_chunk{args.chunk_size}_{timestamp}"
    run_dir = Path(args.results_dir) / run_name

    results_manager = ResultsManager(run_dir)
    results_manager.save_config(config)

    print(f"\nResults directory: {run_dir}")

    # Load model and processor
    print("\nLoading model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    model = AutoModelForVision2Seq.from_pretrained(
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

    model.gradient_checkpointing_enable()

    print("\nAdding LoRA adapters...")
    model = setup_lora(model, config)
    model = model.to(device)

    vocab_size = len(processor.tokenizer)
    action_tokenizer = ActionTokenizer(vocab_size=vocab_size)

    # Load datasets with chunking
    print(f"\nLoading datasets from {args.data_dir}...")

    train_dataset = LIBEROChunkedDataset(
        data_dir=args.data_dir,
        suite_name=args.suite,
        processor=processor,
        action_tokenizer=action_tokenizer,
        chunk_size=args.chunk_size,
        max_samples=args.max_samples,
        split='train',
        val_demos_per_task=args.val_demos,
    )

    val_dataset = LIBEROChunkedDataset(
        data_dir=args.data_dir,
        suite_name=args.suite,
        processor=processor,
        action_tokenizer=action_tokenizer,
        chunk_size=args.chunk_size,
        split='val',
        val_demos_per_task=args.val_demos,
    )

    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    collate_with_pad = partial(collate_fn, pad_token_id=pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_pad,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_with_pad,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * config['warmup_ratio'])

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"\nTraining steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Validation every {args.val_steps} steps")

    # Initial validation
    print("\n" + "=" * 60)
    print(" Initial Validation (Step 0)")
    print("=" * 60)

    val_metrics = validate(model, val_loader, device, action_tokenizer)
    results_manager.log_validation(0, 0, val_metrics)

    print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
    print(f"  L1 Error: {val_metrics['l1_error']:.4f}")
    print(f"  Direction Accuracy: {val_metrics['direction_accuracy']:.4f}")
    print(f"  Gripper Accuracy: {val_metrics['gripper_accuracy']:.4f}")

    # Training loop
    print("\n" + "=" * 60)
    print(" Starting Training")
    print("=" * 60)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        model.train()
        epoch_loss = 0
        epoch_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(pbar):
            loss = train_step(model, batch, device, args.grad_accum)
            epoch_loss += loss
            epoch_batches += 1

            if (batch_idx + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Log training
                if global_step % args.log_steps == 0:
                    avg_loss = epoch_loss / epoch_batches
                    lr = scheduler.get_last_lr()[0]
                    results_manager.log_training(global_step, epoch + 1, avg_loss, lr)
                    pbar.set_postfix({'loss': f'{loss:.4f}', 'avg': f'{avg_loss:.4f}', 'lr': f'{lr:.2e}'})

                # Validation
                if global_step % args.val_steps == 0:
                    print(f"\n  Validation at step {global_step}...")
                    val_metrics = validate(model, val_loader, device, action_tokenizer)
                    results_manager.log_validation(global_step, epoch + 1, val_metrics)

                    print(f"    Val Loss: {val_metrics['val_loss']:.4f}")
                    print(f"    L1 Error: {val_metrics['l1_error']:.4f}")
                    print(f"    Direction Acc: {val_metrics['direction_accuracy']:.4f}")
                    print(f"    Gripper Acc: {val_metrics['gripper_accuracy']:.4f}")

                    # Save best model
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        best_dir = results_manager.run_dir / "best"
                        model.save_pretrained(best_dir)
                        print(f"    New best model saved!")

                    model.train()

                # Save checkpoint
                if global_step % args.save_steps == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step,
                                  results_manager, config)

        # End of epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {epoch_loss / epoch_batches:.4f}")

    # Final validation and save
    print("\n" + "=" * 60)
    print(" Final Validation")
    print("=" * 60)

    val_metrics = validate(model, val_loader, device, action_tokenizer)
    results_manager.log_validation(global_step, args.epochs, val_metrics)

    print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
    print(f"  L1 Error: {val_metrics['l1_error']:.4f}")
    print(f"  Direction Accuracy: {val_metrics['direction_accuracy']:.4f}")
    print(f"  Gripper Accuracy: {val_metrics['gripper_accuracy']:.4f}")

    # Save final model
    final_dir = results_manager.run_dir / "final"
    model.save_pretrained(final_dir)

    # Generate figures
    print("\nGenerating figures...")
    results_manager.generate_figures()

    # Save summary
    summary = {
        'total_steps': global_step,
        'best_val_loss': best_val_loss,
        'final_metrics': val_metrics,
        'config': config,
        'completed_at': datetime.now().isoformat(),
    }
    results_manager.save_summary(summary)

    print("\n" + "=" * 60)
    print(" Training Complete")
    print("=" * 60)
    print(f"\nResults saved to: {results_manager.run_dir}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"\nTo evaluate:")
    print(f"  python evaluate_finetuned_chunked.py --checkpoint {final_dir}")


if __name__ == "__main__":
    main()
