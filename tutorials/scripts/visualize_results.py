#!/usr/bin/env python
"""
Visualize OpenVLA Fine-tuning Results

Generate training curves and analysis figures from CSV logs.

Usage:
    python visualize_results.py --run results/libero_spatial_chunk4_xxx
    python visualize_results.py --run results/libero_spatial_chunk4_xxx --compare results/another_run
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_run(run_dir):
    """Load all data from a training run."""
    run_dir = Path(run_dir)

    data = {
        'name': run_dir.name,
        'config': None,
        'training_log': None,
        'validation_log': None,
        'summary': None,
    }

    # Load config
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            data['config'] = json.load(f)

    # Load training log
    training_log_path = run_dir / "training_log.csv"
    if training_log_path.exists():
        data['training_log'] = pd.read_csv(training_log_path)

    # Load validation log
    validation_log_path = run_dir / "validation_log.csv"
    if validation_log_path.exists():
        data['validation_log'] = pd.read_csv(validation_log_path)

    # Load summary
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            data['summary'] = json.load(f)

    return data


def plot_training_curves(run_data, output_dir):
    """Generate training curve plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_log = run_data['training_log']
    validation_log = run_data['validation_log']

    if training_log is None or validation_log is None:
        print("Missing log data, skipping plots.")
        return

    # Figure 1: Training and Validation Loss
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(training_log['step'], training_log['loss'],
            alpha=0.5, label='Training Loss', color='blue')
    ax.plot(validation_log['step'], validation_log['val_loss'],
            'b-o', label='Validation Loss', markersize=6, linewidth=2)

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(f'Training Progress: {run_data["name"]}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(output_dir / 'loss_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'loss_curves.png'}")

    # Figure 2: All Validation Metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Validation Loss
    ax = axes[0, 0]
    ax.plot(validation_log['step'], validation_log['val_loss'], 'b-o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss')
    ax.grid(True, alpha=0.3)

    # L1 Error
    ax = axes[0, 1]
    ax.plot(validation_log['step'], validation_log['l1_error'], 'g-o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('L1 Error')
    ax.set_title('Action L1 Error')
    ax.grid(True, alpha=0.3)

    # Direction Accuracy
    ax = axes[1, 0]
    ax.plot(validation_log['step'], validation_log['direction_accuracy'], 'r-o', markersize=4)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random (50%)', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Direction Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    # Gripper Accuracy
    ax = axes[1, 1]
    ax.plot(validation_log['step'], validation_log['gripper_accuracy'], 'purple',
            marker='o', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('Accuracy')
    ax.set_title('Gripper Accuracy')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.suptitle(f'Validation Metrics: {run_data["name"]}', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'validation_metrics.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'validation_metrics.png'}")

    # Figure 3: Learning Rate Schedule
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(training_log['step'], training_log['lr'])
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    fig.savefig(output_dir / 'learning_rate.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'learning_rate.png'}")


def plot_comparison(runs_data, output_dir):
    """Compare multiple training runs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = plt.cm.tab10.colors

    for i, run_data in enumerate(runs_data):
        color = colors[i % len(colors)]
        label = run_data['name'][:30]  # Truncate long names

        val_log = run_data['validation_log']
        if val_log is None:
            continue

        # Validation Loss
        axes[0, 0].plot(val_log['step'], val_log['val_loss'],
                       '-o', markersize=3, color=color, label=label)

        # L1 Error
        axes[0, 1].plot(val_log['step'], val_log['l1_error'],
                       '-o', markersize=3, color=color, label=label)

        # Direction Accuracy
        axes[1, 0].plot(val_log['step'], val_log['direction_accuracy'],
                       '-o', markersize=3, color=color, label=label)

        # Gripper Accuracy
        axes[1, 1].plot(val_log['step'], val_log['gripper_accuracy'],
                       '-o', markersize=3, color=color, label=label)

    axes[0, 0].set_title('Validation Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].set_title('L1 Error')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('L1 Error')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].set_title('Direction Accuracy')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].set_title('Gripper Accuracy')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylim(0, 1)

    plt.suptitle('Run Comparison', fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / 'comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_dir / 'comparison.png'}")


def print_summary(run_data):
    """Print a summary of the training run."""
    print("\n" + "="*60)
    print(f" Run: {run_data['name']}")
    print("="*60)

    if run_data['config']:
        config = run_data['config']
        print(f"\nConfiguration:")
        print(f"  Chunk size: {config.get('chunk_size', 'N/A')}")
        print(f"  Effective Hz: {config.get('effective_hz', 'N/A')}")
        print(f"  Epochs: {config.get('epochs', 'N/A')}")
        print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
        print(f"  LoRA rank: {config.get('lora_r', 'N/A')}")

    if run_data['validation_log'] is not None:
        val_log = run_data['validation_log']
        print(f"\nBest Metrics:")
        print(f"  Best val_loss: {val_log['val_loss'].min():.4f}")
        print(f"  Best L1 error: {val_log['l1_error'].min():.4f}")
        print(f"  Best direction accuracy: {val_log['direction_accuracy'].max():.4f}")
        print(f"  Best gripper accuracy: {val_log['gripper_accuracy'].max():.4f}")

        print(f"\nFinal Metrics (last validation):")
        print(f"  val_loss: {val_log['val_loss'].iloc[-1]:.4f}")
        print(f"  L1 error: {val_log['l1_error'].iloc[-1]:.4f}")
        print(f"  Direction accuracy: {val_log['direction_accuracy'].iloc[-1]:.4f}")
        print(f"  Gripper accuracy: {val_log['gripper_accuracy'].iloc[-1]:.4f}")

    if run_data['summary']:
        summary = run_data['summary']
        print(f"\nTraining Summary:")
        print(f"  Total steps: {summary.get('total_steps', 'N/A')}")
        print(f"  Completed at: {summary.get('completed_at', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='Visualize OpenVLA fine-tuning results')
    parser.add_argument('--run', type=str, required=True,
                        help='Path to run directory')
    parser.add_argument('--compare', type=str, nargs='+', default=[],
                        help='Additional runs to compare')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: run_dir/figures)')
    args = parser.parse_args()

    # Load primary run
    run_data = load_run(args.run)
    print_summary(run_data)

    # Generate plots
    output_dir = args.output or (Path(args.run) / "figures")
    plot_training_curves(run_data, output_dir)

    # Compare runs if specified
    if args.compare:
        all_runs = [run_data]
        for run_path in args.compare:
            compare_data = load_run(run_path)
            print_summary(compare_data)
            all_runs.append(compare_data)

        plot_comparison(all_runs, output_dir)

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
