# OpenVLA Experimentation Report

**Project**: Fine-tuning and Evaluation of OpenVLA on LIBERO Benchmark
**Date Range**: December 2024 - January 2025
**Environment**: SciServer 4×40GB GPU Nodes

---

## Report Chapters

| Chapter | Title | Description |
|---------|-------|-------------|
| [01](01_environment_setup.md) | Environment Setup and Challenges | Dependency conflicts, version pinning, HPC configuration |
| [02](02_initial_libero_experiments.md) | Initial LIBERO Experiments | Zero-shot evaluation, why it failed |
| [03](03_bridge_v2_validation.md) | Bridge V2 Re-implementation | Pipeline validation on training distribution |
| [04](04_libero_finetuning.md) | LIBERO Fine-tuning Results | LoRA fine-tuning, partial success, mode collapse |
| [05](05_key_findings.md) | Key Findings and Lessons Learned | Technical discoveries, methodology insights |

---

## Quick Summary

### What Worked
- Environment setup (after resolving dependencies)
- Bridge V2 pipeline validation (67.5% proxy success rate)
- LoRA fine-tuning infrastructure
- Training loss convergence
- L1 error improvement (28.7% reduction)
- Gripper accuracy improvement (+6.8%)

### What Needs Further Work
- Frame rate mismatch resolution (20 Hz LIBERO vs 3-5 Hz training data)
- Direction accuracy recovery (mode collapse to near-zero predictions)
- Real robot evaluation
- Multi-task generalization testing

---

## Key Technical Finding

**Control Frequency Mismatch** is the root cause of fine-tuning issues:

| Dataset | Control Frequency | Status |
|---------|------------------|--------|
| Fractal (Google RT-1) | 3 Hz | Primary training data |
| Bridge V2 | 5 Hz | Primary training data |
| DROID | 15 Hz | Removed from training (issues) |
| **LIBERO** | **20 Hz** | Fine-tuning target (4-7x mismatch) |

**Solution**: Temporal subsampling (use every 4th frame) or action scaling (multiply by 4x).

---

*See individual chapters for detailed analysis.*
