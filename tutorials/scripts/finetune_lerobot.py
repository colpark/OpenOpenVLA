#!/usr/bin/env python
"""
Fine-tune OpenVLA on Converted LeRobot Data

Fine-tunes OpenVLA using LoRA on converted LeRobot HDF5 data.
Includes comprehensive logging, validation, and checkpoint saving.

Usage:
    python finetune_lerobot.py --task ball2 --epochs 5
    python finetune_lerobot.py --task plug_stacking_v3 --lr 2e-5 --epochs 10
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import h5py
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Default data directory (local development)
DEFAULT_DATA_DIR = "/Users/davidpark/Documents/Claude/openvla/inhouse/openvla_converted"
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache")

# Check for HPC environment
if 'SCRATCH' in os.environ:
    DEFAULT_DATA_DIR = f"{os.environ['SCRATCH']}/lerobot_converted"
    DEFAULT_CACHE_DIR = f"{os.environ['SCRATCH']}/.cache"

os.environ.setdefault('HF_HOME', f"{DEFAULT_CACHE_DIR}/huggingface")
os.environ.setdefault('TORCH_HOME', f"{DEFAULT_CACHE_DIR}/torch")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Action Tokenizer
# =============================================================================

class ActionTokenizer:
    """Tokenize continuous actions to discrete bins.

    Supports both single-arm (7-dim) and bimanual (14-dim) actions.
    """

    def __init__(self, vocab_size=32000, n_bins=256, action_dim=7):
        self.vocab_size = vocab_size
        self.n_bins = n_bins
        self.action_dim = action_dim
        self.bins = np.linspace(-1, 1, n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2

    def encode(self, actions):
        """Encode continuous actions to token IDs."""
        actions = np.clip(actions, -1, 1)
        discretized = np.digitize(actions, self.bins) - 1
        discretized = np.clip(discretized, 0, self.n_bins - 2)
        token_ids = self.vocab_size - discretized - 1
        return token_ids

    def decode(self, token_ids):
        """Decode token IDs to continuous actions."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.cpu().numpy()
        discretized = self.vocab_size - token_ids - 1
        indices = np.clip(discretized, 0, len(self.bin_centers) - 1)
        return self.bin_centers[indices]


# =============================================================================
# Dataset
# =============================================================================

class LeRobotDataset(Dataset):
    """PyTorch Dataset for converted LeRobot data.

    Supports both single-arm (7-dim) and bimanual (14-dim) actions.
    """

    def __init__(self, hdf5_path, processor, action_tokenizer, image_size=224):
        self.hdf5_path = Path(hdf5_path)
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.image_size = image_size

        # Load data
        with h5py.File(self.hdf5_path, 'r') as f:
            self.n_samples = f['images'].shape[0]
            self.language = f.attrs.get('language_instruction', 'complete the task')
            if isinstance(self.language, bytes):
                self.language = self.language.decode('utf-8')

            # Get action dimension from data
            self.action_dim = f.attrs.get('action_dim', f['actions'].shape[1])
            self.is_bimanual = f.attrs.get('is_bimanual', self.action_dim == 14)

            # Pre-load actions (small)
            self.actions = f['actions'][:]

        # Update tokenizer with correct action dimension
        self.action_tokenizer.action_dim = self.action_dim

        print(f"Loaded {self.n_samples} samples from {self.hdf5_path.name}")
        print(f"Language: '{self.language}'")
        print(f"Action dim: {self.action_dim} ({'bimanual' if self.is_bimanual else 'single-arm'})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Load image (lazy loading for memory efficiency)
        with h5py.File(self.hdf5_path, 'r') as f:
            image = f['images'][idx]

        # Resize and process image
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Create prompt
        prompt = f"In: What action should the robot take to {self.language.lower()}?\nOut:"

        # Process with model processor
        inputs = self.processor(prompt, pil_image, return_tensors="pt")

        # Tokenize action
        action = self.actions[idx]
        action_tokens = self.action_tokenizer.encode(action)

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'action_tokens': torch.tensor(action_tokens, dtype=torch.long),
            'action': torch.tensor(action, dtype=torch.float32),
        }


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""
    # Find max sequence length
    max_len = max(item['input_ids'].shape[0] for item in batch)

    input_ids = []
    attention_masks = []
    pixel_values = []
    action_tokens = []
    actions = []

    for item in batch:
        # Pad input_ids
        seq_len = item['input_ids'].shape[0]
        padding = max_len - seq_len

        input_ids.append(torch.cat([
            item['input_ids'],
            torch.zeros(padding, dtype=torch.long)
        ]))
        attention_masks.append(torch.cat([
            item['attention_mask'],
            torch.zeros(padding, dtype=torch.long)
        ]))
        pixel_values.append(item['pixel_values'])
        action_tokens.append(item['action_tokens'])
        actions.append(item['action'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_masks),
        'pixel_values': torch.stack(pixel_values),
        'action_tokens': torch.stack(action_tokens),
        'actions': torch.stack(actions),
    }


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(predictions, ground_truths, is_bimanual=False):
    """Compute comprehensive evaluation metrics.

    Supports both single-arm (7-dim) and bimanual (14-dim) actions.
    """
    predictions = np.array(predictions)
    ground_truths = np.array(ground_truths)

    action_dim = predictions.shape[1]
    is_bimanual = action_dim == 14

    # L1 errors
    l1_error = np.abs(predictions - ground_truths).mean()

    if is_bimanual:
        # Bimanual: [left_pos(3), left_rot(3), left_grip(1), right_pos(3), right_rot(3), right_grip(1)]
        left_pos_l1 = np.abs(predictions[:, 0:3] - ground_truths[:, 0:3]).mean()
        left_rot_l1 = np.abs(predictions[:, 3:6] - ground_truths[:, 3:6]).mean()
        right_pos_l1 = np.abs(predictions[:, 7:10] - ground_truths[:, 7:10]).mean()
        right_rot_l1 = np.abs(predictions[:, 10:13] - ground_truths[:, 10:13]).mean()

        position_l1 = (left_pos_l1 + right_pos_l1) / 2
        rotation_l1 = (left_rot_l1 + right_rot_l1) / 2

        # Gripper indices for bimanual
        gripper_indices = [6, 13]
    else:
        # Single arm: [pos(3), rot(3), grip(1)]
        position_l1 = np.abs(predictions[:, :3] - ground_truths[:, :3]).mean()
        rotation_l1 = np.abs(predictions[:, 3:6] - ground_truths[:, 3:6]).mean()
        gripper_indices = [6]

    gripper_l1 = np.abs(predictions[:, gripper_indices] - ground_truths[:, gripper_indices]).mean()

    # Direction accuracy (for non-zero ground truth)
    threshold = 0.02
    dir_correct = 0
    dir_total = 0

    # Check position and rotation dims (exclude gripper)
    motion_dims = list(range(6)) if not is_bimanual else list(range(6)) + list(range(7, 13))

    for dim in motion_dims:
        significant = np.abs(ground_truths[:, dim]) > threshold
        if significant.sum() > 0:
            same_sign = np.sign(ground_truths[:, dim][significant]) == np.sign(predictions[:, dim][significant])
            dir_correct += same_sign.sum()
            dir_total += significant.sum()

    direction_accuracy = dir_correct / dir_total if dir_total > 0 else 0.5

    # Gripper accuracy (binary)
    gripper_threshold = 0.5
    gripper_correct = 0
    gripper_total = 0

    for g_idx in gripper_indices:
        gt_gripper = (ground_truths[:, g_idx] > gripper_threshold).astype(int)
        pred_gripper = (predictions[:, g_idx] > gripper_threshold).astype(int)
        gripper_correct += (gt_gripper == pred_gripper).sum()
        gripper_total += len(gt_gripper)

    gripper_accuracy = gripper_correct / gripper_total if gripper_total > 0 else 0.5

    return {
        'l1_error': float(l1_error),
        'position_l1': float(position_l1),
        'rotation_l1': float(rotation_l1),
        'gripper_l1': float(gripper_l1),
        'direction_accuracy': float(direction_accuracy),
        'gripper_accuracy': float(gripper_accuracy),
        'is_bimanual': is_bimanual,
        'action_dim': action_dim,
    }


# =============================================================================
# Results Manager
# =============================================================================

class ResultsManager:
    """Manage training results and checkpoints."""

    def __init__(self, output_dir, task_name):
        self.output_dir = Path(output_dir)
        self.run_name = f"{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_dir = self.output_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.log_file = self.run_dir / "training_log.csv"
        self.metrics_history = []

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("epoch,step,loss,l1_error,position_l1,direction_acc,gripper_acc,lr\n")

    def log_step(self, epoch, step, loss, metrics, lr):
        """Log training step."""
        with open(self.log_file, 'a') as f:
            f.write(f"{epoch},{step},{loss:.6f},"
                    f"{metrics['l1_error']:.6f},{metrics['position_l1']:.6f},"
                    f"{metrics['direction_accuracy']:.4f},{metrics['gripper_accuracy']:.4f},"
                    f"{lr:.2e}\n")

        self.metrics_history.append({
            'epoch': epoch,
            'step': step,
            'loss': loss,
            **metrics,
            'lr': lr,
        })

    def save_checkpoint(self, model, optimizer, epoch, step, metrics, is_best=False):
        """Save model checkpoint."""
        checkpoint_dir = self.run_dir / ("best" if is_best else f"checkpoint_epoch{epoch}")
        model.save_pretrained(checkpoint_dir)

        # Save optimizer state
        torch.save({
            'epoch': epoch,
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }, checkpoint_dir / "training_state.pt")

        print(f"  Saved checkpoint: {checkpoint_dir.name}")

    def save_final_results(self, config, final_metrics):
        """Save final training results."""
        results = {
            'config': config,
            'final_metrics': final_metrics,
            'metrics_history': self.metrics_history,
            'run_name': self.run_name,
        }

        with open(self.run_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {self.run_dir}")


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, action_tokenizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    ground_truths = []
    action_dim = action_tokenizer.action_dim

    progress = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch in progress:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device).to(torch.bfloat16)
        action_tokens = batch['action_tokens'].to(device)

        # Create labels (action tokens) - use dynamic action_dim
        labels = torch.full_like(input_ids, -100)
        labels[:, -action_dim:] = action_tokens

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )

        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Get predictions for metrics - use dynamic action_dim
        with torch.no_grad():
            pred_tokens = outputs.logits[:, -action_dim:, :].argmax(dim=-1)
            pred_actions = action_tokenizer.decode(pred_tokens.cpu().numpy())
            predictions.extend(pred_actions)
            ground_truths.extend(batch['actions'].numpy())

        progress.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(predictions, ground_truths)

    return avg_loss, metrics


def evaluate(model, dataloader, action_tokenizer, device):
    """Evaluate model on validation set."""
    model.eval()
    predictions = []
    ground_truths = []
    total_loss = 0
    action_dim = action_tokenizer.action_dim

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device).to(torch.bfloat16)
            action_tokens = batch['action_tokens'].to(device)

            # Create labels - use dynamic action_dim
            labels = torch.full_like(input_ids, -100)
            labels[:, -action_dim:] = action_tokens

            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )

            total_loss += outputs.loss.item()

            # Get predictions - use dynamic action_dim
            pred_tokens = outputs.logits[:, -action_dim:, :].argmax(dim=-1)
            pred_actions = action_tokenizer.decode(pred_tokens.cpu().numpy())
            predictions.extend(pred_actions)
            ground_truths.extend(batch['actions'].numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(predictions, ground_truths)

    return avg_loss, metrics


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Fine-tune OpenVLA on LeRobot data')
    parser.add_argument('--data-dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory with converted HDF5 files')
    parser.add_argument('--task', type=str, default='ball2',
                        help='Task name')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--lora-rank', type=int, default=32,
                        help='LoRA rank')
    parser.add_argument('--validate-every', type=int, default=100,
                        help='Validate every N steps')
    parser.add_argument('--model-id', type=str, default='openvla/openvla-7b',
                        help='Base model ID')
    args = parser.parse_args()

    print("="*60)
    print(" OpenVLA Fine-tuning on LeRobot Data")
    print("="*60)

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    print("\nLoading model...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=f"{DEFAULT_CACHE_DIR}/huggingface",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        cache_dir=f"{DEFAULT_CACHE_DIR}/huggingface",
    )

    # Configure LoRA
    print(f"\nConfiguring LoRA (rank={args.lora_rank})...")
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Action tokenizer
    vocab_size = len(processor.tokenizer)
    action_tokenizer = ActionTokenizer(vocab_size=vocab_size)

    # Load datasets
    print("\nLoading datasets...")
    data_dir = Path(args.data_dir)
    train_path = data_dir / f"{args.task}_train.hdf5"
    val_path = data_dir / f"{args.task}_val.hdf5"

    if not train_path.exists():
        print(f"ERROR: Training data not found: {train_path}")
        print(f"Run convert_lerobot_to_openvla.py --task {args.task} first")
        sys.exit(1)

    train_dataset = LeRobotDataset(train_path, processor, action_tokenizer)
    val_dataset = LeRobotDataset(val_path, processor, action_tokenizer) if val_path.exists() else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    ) if val_dataset else None

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = min(100, total_steps // 10)

    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # Results manager
    results_manager = ResultsManager(args.output_dir, args.task)

    # Training configuration
    config = {
        'task': args.task,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'lora_rank': args.lora_rank,
        'model_id': args.model_id,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset) if val_dataset else 0,
    }

    print(f"\nTraining configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Training loop
    print("\n" + "="*60)
    print(" Starting Training")
    print("="*60)

    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        train_loss, train_metrics = train_epoch(
            model, train_loader, optimizer, scheduler,
            action_tokenizer, device, epoch
        )

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"  L1 Error: {train_metrics['l1_error']:.4f}")
        print(f"  Position L1: {train_metrics['position_l1']:.4f}")
        print(f"  Direction Acc: {train_metrics['direction_accuracy']:.4f}")
        print(f"  Gripper Acc: {train_metrics['gripper_accuracy']:.4f}")

        # Validation
        if val_loader:
            val_loss, val_metrics = evaluate(model, val_loader, action_tokenizer, device)

            print(f"\nVal Loss: {val_loss:.4f}")
            print(f"  L1 Error: {val_metrics['l1_error']:.4f}")
            print(f"  Direction Acc: {val_metrics['direction_accuracy']:.4f}")
            print(f"  Gripper Acc: {val_metrics['gripper_accuracy']:.4f}")

            # Log metrics
            results_manager.log_step(
                epoch, global_step, val_loss, val_metrics,
                optimizer.param_groups[0]['lr']
            )

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                results_manager.save_checkpoint(
                    model, optimizer, epoch, global_step, val_metrics, is_best=True
                )
        else:
            results_manager.log_step(
                epoch, global_step, train_loss, train_metrics,
                optimizer.param_groups[0]['lr']
            )

        global_step += len(train_loader)

    # Save final checkpoint
    print("\n" + "="*60)
    print(" Training Complete")
    print("="*60)

    final_metrics = val_metrics if val_loader else train_metrics
    results_manager.save_checkpoint(
        model, optimizer, args.epochs, global_step, final_metrics, is_best=False
    )
    results_manager.save_final_results(config, final_metrics)

    print(f"\nFinal Results:")
    print(f"  L1 Error: {final_metrics['l1_error']:.4f}")
    print(f"  Direction Accuracy: {final_metrics['direction_accuracy']:.4f}")
    print(f"  Gripper Accuracy: {final_metrics['gripper_accuracy']:.4f}")


if __name__ == "__main__":
    main()
