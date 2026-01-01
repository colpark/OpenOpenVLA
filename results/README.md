# OpenVLA Fine-tuning Results

This folder stores training runs for OpenVLA fine-tuning experiments.

## Directory Structure

Each training run creates a folder with the following structure:

```
run_name/
├── config.json           # Training configuration
├── training_log.csv      # Per-step training metrics
├── validation_log.csv    # Validation metrics (every N steps)
├── summary.json          # Final training summary
├── checkpoints/          # Model checkpoints
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
├── best/                 # Best model (lowest val loss)
├── final/                # Final model after training
└── figures/              # Training curves (auto-generated)
    ├── training_loss.png
    ├── validation_metrics.png
    └── loss_comparison.png
```

## Log Files

### training_log.csv
| Column | Description |
|--------|-------------|
| step | Global training step |
| epoch | Current epoch |
| loss | Training loss (averaged) |
| lr | Current learning rate |
| timestamp | ISO timestamp |

### validation_log.csv
| Column | Description |
|--------|-------------|
| step | Global step at validation |
| epoch | Current epoch |
| val_loss | Validation loss |
| l1_error | Mean L1 error across all action dims |
| direction_accuracy | Position direction accuracy (sign agreement) |
| gripper_accuracy | Gripper open/close accuracy |
| position_l1 | L1 error for position dims (0-2) |
| rotation_l1 | L1 error for rotation dims (3-5) |
| timestamp | ISO timestamp |

## Usage

### Run Training

```bash
python tutorials/scripts/finetune_openvla_chunked.py \
    --suite libero_spatial \
    --chunk-size 4 \
    --epochs 10 \
    --val-steps 100
```

### Visualize Results

```bash
# Single run
python tutorials/scripts/visualize_results.py --run results/run_name

# Compare multiple runs
python tutorials/scripts/visualize_results.py \
    --run results/run1 \
    --compare results/run2 results/run3
```

### Evaluate Model

Use the notebook:
```
tutorials/notebooks/15_evaluate_chunked_finetuned.ipynb
```

Or run evaluation script:
```bash
python tutorials/scripts/evaluate_finetuned_chunked.py \
    --checkpoint results/run_name/best
```

## Key Metrics

| Metric | Target | Notes |
|--------|--------|-------|
| Direction Accuracy | >50% | Must exceed random baseline |
| L1 Error | Lower is better | Measures action prediction quality |
| Gripper Accuracy | >80% | Binary open/close correctness |

## Action Chunking Reference

| Chunk Size | Effective Hz | Matches |
|------------|--------------|---------|
| 4 | 5 Hz | Bridge V2 (recommended) |
| 5 | 4 Hz | Between Bridge V2 and Fractal |
| 7 | ~3 Hz | Fractal |

## Naming Convention

Run folders are named:
```
{suite}_{chunk_size}_{timestamp}
```

Example: `libero_spatial_chunk4_20250101_143022`
