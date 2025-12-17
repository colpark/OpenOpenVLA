#!/usr/bin/env python
"""
OpenVLA Fine-tuning Script for LIBERO (Optimized Distributed Training)

Key optimizations:
- Uses accelerate for proper distributed setup (model loaded ONCE, then distributed)
- DeepSpeed ZeRO Stage 2 for memory efficiency
- Gradient checkpointing for larger batch sizes
- Proper LoRA configuration

Usage:
    # Single GPU (RECOMMENDED for first run)
    CUDA_VISIBLE_DEVICES=0 python finetune_openvla_distributed.py

    # Multi-GPU with accelerate (for faster training)
    accelerate launch --config_file accelerate_config.yaml finetune_openvla_distributed.py

IMPORTANT: OpenVLA's custom vision backbone doesn't work with PyTorch DataParallel.
           For multi-GPU training, you MUST use accelerate launch or torchrun.
"""

import os
import sys

# CRITICAL: Force single GPU if not using accelerate/torchrun
# Must be done BEFORE importing torch to take effect
if 'WORLD_SIZE' not in os.environ and 'CUDA_VISIBLE_DEVICES' not in os.environ:
    # Not running with accelerate/torchrun and no explicit GPU selection
    # Force single GPU to avoid DataParallel issues with OpenVLA
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("INFO: Forcing single GPU mode (CUDA_VISIBLE_DEVICES=0)")
    print("      For multi-GPU, use: accelerate launch --multi_gpu ...")

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import h5py
from tqdm import tqdm
import json
import logging

# Suppress TensorFlow warnings (from LIBERO dependencies)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ============================================================
# Configuration (auto-detect NERSC vs SciServer)
# ============================================================
# NERSC Perlmutter: PSCRATCH environment variable
# SciServer: SCRATCH environment variable or default path
if os.environ.get('PSCRATCH'):
    BASE_DIR = os.environ['PSCRATCH']
elif os.environ.get('SCRATCH'):
    BASE_DIR = os.environ['SCRATCH']
else:
    BASE_DIR = "/home/idies/workspace/Temporary/dpark1/scratch"  # SciServer default

MODEL_ID = "openvla/openvla-7b"
DATA_DIR = f"{BASE_DIR}/libero_data"
OUTPUT_DIR = f"{BASE_DIR}/openvla_finetune"
CACHE_DIR = f"{BASE_DIR}/.cache"

# Set cache directories BEFORE importing HuggingFace
os.environ['HF_HOME'] = f"{CACHE_DIR}/huggingface"
os.environ['TORCH_HOME'] = f"{CACHE_DIR}/torch"
os.environ['XDG_CACHE_HOME'] = CACHE_DIR

# Now import HuggingFace libraries
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import TrainingArguments, Trainer, TrainerCallback
from accelerate import Accelerator
from accelerate.utils import set_seed

try:
    from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. Install with: pip install peft")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# Dataset
# ============================================================
class LIBERODataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for LIBERO demonstrations.
    Optimized for distributed training with proper caching.
    """

    # Action tokenization constants (must match OpenVLA's ActionTokenizer exactly!)
    # OpenVLA uses 256 bins per dimension
    N_ACTION_BINS = 256
    MIN_ACTION = -1.0
    MAX_ACTION = 1.0

    def __init__(self, data_dir, processor, max_samples=None):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_samples = max_samples

        # Get vocab size for action token mapping
        self.vocab_size = len(processor.tokenizer)

        # Create uniform bins matching OpenVLA's ActionTokenizer
        # bins = np.linspace(-1, 1, 256) gives 256 bin edges
        self.bins = np.linspace(self.MIN_ACTION, self.MAX_ACTION, self.N_ACTION_BINS)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0  # 255 bin centers

        logger.info(f"Vocab size: {self.vocab_size}")
        logger.info(f"Action bins: {self.N_ACTION_BINS}, range [{self.MIN_ACTION}, {self.MAX_ACTION}]")

        # Index samples (only on rank 0, then broadcast)
        self.samples = self._index_samples()

        logger.info(f"Dataset initialized with {len(self.samples)} samples")

    def _index_samples(self):
        samples = []

        # Find all HDF5 files
        hdf5_files = list(self.data_dir.rglob("*.hdf5"))
        if not hdf5_files:
            raise ValueError(f"No HDF5 files found in {self.data_dir}")

        logger.info(f"Found {len(hdf5_files)} HDF5 files")

        for filepath in tqdm(hdf5_files, desc="Indexing demos"):
            try:
                with h5py.File(filepath, 'r') as f:
                    # Get language instruction
                    language = f.attrs.get('language_instruction', 'perform task')
                    if isinstance(language, bytes):
                        language = language.decode('utf-8')

                    # Index each demo and timestep
                    if 'data' not in f:
                        continue

                    for demo_key in f['data'].keys():
                        if not demo_key.startswith('demo_'):
                            continue

                        demo = f['data'][demo_key]
                        if 'actions' not in demo:
                            continue

                        n_steps = len(demo['actions'])

                        for t in range(n_steps):
                            samples.append({
                                'filepath': str(filepath),
                                'demo_key': demo_key,
                                'timestep': t,
                                'language': language,
                            })

                            # Early exit if max_samples reached
                            if self.max_samples and len(samples) >= self.max_samples:
                                return samples

            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")
                continue

        return samples

    def __len__(self):
        return len(self.samples)

    def tokenize_action(self, action):
        """
        Convert continuous action to discrete action tokens.

        IMPORTANT: Must match OpenVLA's ActionTokenizer exactly!
        OpenVLA uses: token_id = vocab_size - digitized_action
        where digitized_action = np.digitize(action, bins) returns indices [1, 256]

        Args:
            action: numpy array of shape (7,) with values in [-1, 1]

        Returns:
            tensor of action token IDs
        """
        # Clip action to valid range
        action = np.clip(action, self.MIN_ACTION, self.MAX_ACTION)

        # Discretize using np.digitize (same as OpenVLA's ActionTokenizer)
        # Returns indices from 1 to N_ACTION_BINS (inclusive)
        discretized_action = np.digitize(action, self.bins)

        # Convert to token IDs using OpenVLA's convention:
        # token_id = vocab_size - discretized_action
        # This maps: action=-1 → token vocab_size-1, action=+1 → token vocab_size-256
        action_tokens = self.vocab_size - discretized_action

        return torch.tensor(action_tokens, dtype=torch.long)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        with h5py.File(sample['filepath'], 'r') as f:
            demo = f['data'][sample['demo_key']]
            t = sample['timestep']

            # Load image
            image = demo['obs']['agentview_rgb'][t]
            image = np.rot90(image, k=2)  # LIBERO convention

            # Load action
            action = demo['actions'][t].astype(np.float32)

        # Format for OpenVLA
        prompt = f"In: What action should the robot take to {sample['language'].lower()}?\nOut:"
        pil_image = Image.fromarray(image.astype(np.uint8))

        # Process with OpenVLA processor
        inputs = self.processor(prompt, pil_image, return_tensors="pt")

        # Squeeze batch dimension added by processor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Tokenize the action into discrete tokens
        action_tokens = self.tokenize_action(action)

        # Append action tokens to input_ids
        inputs['input_ids'] = torch.cat([inputs['input_ids'], action_tokens])

        # Extend attention mask for action tokens
        inputs['attention_mask'] = torch.cat([
            inputs['attention_mask'],
            torch.ones(len(action_tokens), dtype=inputs['attention_mask'].dtype)
        ])

        # Create labels: -100 for prompt tokens (no loss), action tokens for prediction
        # The model should learn to predict the action tokens given the image and prompt
        prompt_len = len(inputs['input_ids']) - len(action_tokens)
        labels = torch.full_like(inputs['input_ids'], -100)  # -100 = ignore in loss
        labels[prompt_len:] = action_tokens  # Only compute loss on action tokens

        inputs['labels'] = labels

        return inputs


def collate_fn(batch):
    """Custom collate function for variable-length sequences."""

    # Find max length
    max_len = max(item['input_ids'].shape[0] for item in batch)

    padded_batch = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
        'pixel_values': [],
    }

    for item in batch:
        seq_len = item['input_ids'].shape[0]
        pad_len = max_len - seq_len

        # Pad input_ids and attention_mask
        padded_batch['input_ids'].append(
            torch.nn.functional.pad(item['input_ids'], (0, pad_len), value=0)
        )
        padded_batch['attention_mask'].append(
            torch.nn.functional.pad(item['attention_mask'], (0, pad_len), value=0)
        )
        padded_batch['labels'].append(
            torch.nn.functional.pad(item['labels'], (0, pad_len), value=-100)
        )
        padded_batch['pixel_values'].append(item['pixel_values'])

    # Stack - only model-compatible keys
    # Convert pixel_values to bfloat16 to match model dtype
    result = {
        'input_ids': torch.stack(padded_batch['input_ids']),
        'attention_mask': torch.stack(padded_batch['attention_mask']),
        'labels': torch.stack(padded_batch['labels']),
        'pixel_values': torch.stack(padded_batch['pixel_values']).to(torch.bfloat16),
    }

    return result


# ============================================================
# Training
# ============================================================
class ProgressCallback(TrainerCallback):
    """Callback to log training progress."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Step {state.global_step}: {logs}")


def setup_model_and_processor(args):
    """
    Load model and processor.
    This should be called AFTER accelerator.main_process_first() context.
    """
    logger.info(f"Loading model: {MODEL_ID}")

    # Load processor first (lightweight)
    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
    )

    # Load model with proper dtype
    # Use attn_implementation="eager" to avoid SDPA compatibility issues with custom models
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=f"{CACHE_DIR}/huggingface",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # Avoid _supports_sdpa error with newer transformers
    )

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    # Note: torch.compile is incompatible with PEFT + HF Trainer
    # Skip torch.compile when using LoRA to avoid '_orig_mod' error

    # Add LoRA if available
    if PEFT_AVAILABLE and args.use_lora:
        logger.info("Adding LoRA adapters...")

        # Match official OpenVLA fine-tuning config:
        # - target_modules="all-linear" targets ALL linear layers (not just attention)
        # - lora_alpha = min(rank, 16) as per official config
        # - lora_dropout = 0.0 as per official config
        # - init_lora_weights = "gaussian" as per official config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=min(args.lora_r, 16),  # Official uses min(rank, 16)
            lora_dropout=0.0,  # Official uses 0.0
            target_modules="all-linear",  # Official targets ALL linear layers
            init_lora_weights="gaussian",  # Official uses gaussian init
        )

        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        logger.warning("Training full model (high memory usage!)")

    return model, processor


def main():
    parser = argparse.ArgumentParser(description="Fine-tune OpenVLA on LIBERO")

    # Model arguments
    parser.add_argument("--use-lora", action="store_true", default=True,
                        help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--lora-r", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                        help="Learning rate (official OpenVLA uses 5e-4)")
    parser.add_argument("--num-epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--warmup-ratio", type=float, default=0.03,
                        help="Warmup ratio")

    # Data arguments
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max samples for debugging")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers (increase for faster data loading)")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--save-steps", type=int, default=500,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging-steps", type=int, default=50,
                        help="Log every N steps")

    # Debug arguments
    parser.add_argument("--single-gpu", action="store_true",
                        help="Force single GPU mode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    # Performance arguments
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--tf32", action="store_true", default=True,
                        help="Enable TF32 for faster matmul on Ampere/Hopper GPUs")

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Enable TF32 for faster matmul on H100/A100
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for faster matmul")

    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
        project_dir=args.output_dir,
    )

    # Only main process should print
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("OpenVLA Fine-tuning on LIBERO")
        logger.info("=" * 60)
        logger.info(f"Num GPUs: {accelerator.num_processes}")
        logger.info(f"Mixed precision: {accelerator.mixed_precision}")
        logger.info(f"Batch size per GPU: {args.batch_size}")
        logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        logger.info(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")

    # Load model (only download on main process)
    with accelerator.main_process_first():
        model, processor = setup_model_and_processor(args)

    # Synchronize after model loading
    accelerator.wait_for_everyone()

    # Load dataset (index on main process first)
    with accelerator.main_process_first():
        dataset = LIBERODataset(
            DATA_DIR,
            processor,
            max_samples=args.max_samples
        )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="tensorboard",
        ddp_find_unused_parameters=False,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collate_fn,
        callbacks=[ProgressCallback()],
    )

    # Train
    if accelerator.is_main_process:
        logger.info("Starting training...")

    trainer.train()

    # Save final model
    if accelerator.is_main_process:
        final_path = f"{args.output_dir}/final"
        model.save_pretrained(final_path)
        processor.save_pretrained(final_path)
        logger.info(f"Model saved to {final_path}")

        # Save training config
        config = vars(args)
        with open(f"{args.output_dir}/training_config.json", 'w') as f:
            json.dump(config, f, indent=2)

    accelerator.wait_for_everyone()
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
