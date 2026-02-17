---
title: NVIDIA RTX 5070 PyTorch Compatibility Guide
---

# NVIDIA RTX 5070 PyTorch Compatibility Guide

Complete reference for PyTorch compatibility with NVIDIA RTX 50-series GPUs (RTX 5070, 5080, 5090).

## Quick Answer

**Use PyTorch 2.9.1 with CUDA 12.8 for full RTX 5070 support.**

```bash
pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## The Problem

RTX 50-series GPUs (launched January 2026) have compute capability **sm_120**, which requires PyTorch binaries compiled with explicit kernel support.

### Symptoms

```
UserWarning: NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not 
compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 
sm_75 sm_80 sm_86 sm_90.
```

Or runtime errors:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

## Why This Happens

When PyTorch is compiled, it must include GPU kernels for specific compute capabilities. Even if you have the right CUDA version, the PyTorch binary needs to be compiled with sm_120 support.

**Key Insight:** CUDA toolkit version ≠ PyTorch kernel support

- CUDA 12.4+ has sm_120 support ✓
- But PyTorch must be **compiled with sm_120 kernels** ✓✓

## PyTorch Version Compatibility

### ✅ Working Versions

| PyTorch Version | CUDA Version | RTX 5070 Support | Notes |
|----------------|--------------|------------------|-------|
| **2.9.1+cu128** | **12.8** | ✅ **Full support** | **Recommended** |
| 2.9.0+cu128 | 12.8 | ✅ Full support | Older 2.9 release |

### ❌ Non-Working Versions

| PyTorch Version | CUDA Version | RTX 5070 Support | Issues |
|----------------|--------------|------------------|--------|
| 2.6.0+cu124 | 12.4 | ❌ No sm_120 | Warning + runtime errors |
| 2.7.0.dev (nightly) | 12.4 | ❌ No sm_120 | Even nightly lacks support |
| 2.5.x and older | Any | ❌ No sm_120 | Pre-RTX 5070 release |

## Installation Instructions

### Full Installation (Clean Setup)

```bash
# Remove old PyTorch versions
pip uninstall -y torch torchvision torchaudio

# Install PyTorch 2.9.1 with CUDA 12.8
pip install torch==2.9.1+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

### Verification

After installation, verify GPU support:

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')

# Test GPU functionality
x = torch.randn(1000, 1000, device='cuda')
y = x @ x
print('✓ GPU working!')
"
```

Expected output:
```
PyTorch: 2.9.1+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070
Compute: sm_120
✓ GPU working!
```

### In requirements.txt

```
# For RTX 50-series GPU support
--index-url https://download.pytorch.org/whl/cu128
torch==2.9.1+cu128
torchvision
torchaudio
```

### In Docker

```dockerfile
# Use CUDA 12.8 base image
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Install PyTorch with RTX 5070 support
RUN pip install torch==2.9.1+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

## Virtual Environment Setup

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install PyTorch
pip install torch==2.9.1+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

## Troubleshooting

### Installation Issues

**Problem:** `ERROR: Could not find a version that satisfies the requirement torch==2.9.1+cu128`

**Solution:** Make sure you include the `--index-url` flag pointing to PyTorch's CUDA 12.8 repository:
```bash
pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### CUDA Not Available

**Problem:** `torch.cuda.is_available()` returns `False`

**Solutions:**
1. Check NVIDIA drivers: `nvidia-smi`
2. Verify CUDA toolkit: `nvcc --version`
3. Ensure you installed CUDA-enabled PyTorch (not CPU-only)

### Version Conflicts

**Problem:** Package dependency conflicts during installation

**Solution:**
```bash
# Uninstall all PyTorch packages first
pip uninstall -y torch torchvision torchaudio

# Clear pip cache
pip cache purge

# Install fresh
pip install torch==2.9.1+cu128 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128
```

### Hugging Face `tokenizers` — TOKENIZERS_PARALLELISM warning

**Symptom:**
```
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
```
**Cause:** `tokenizers` starts a threadpool; if your process is forked after threads exist, the child may deadlock. The library disables parallelism in the child and emits this warning.

**Fixes:**
- Quick/safe: set the env var before running training:
  ```bash
  export TOKENIZERS_PARALLELISM=false
  ./run_trainer.sh
  ```
- In Python (before importing transformers/tokenizers):
  ```py
  import os
  os.environ['TOKENIZERS_PARALLELISM'] = 'false'
  from transformers import AutoTokenizer
  ```
- Avoid importing/initializing tokenizers before forking, or use multiprocessing start method `spawn`:
  ```py
  import multiprocessing as mp
  mp.set_start_method('spawn', force=True)
  ```

Setting `TOKENIZERS_PARALLELISM=false` is recommended for training scripts where forking (DataLoader, child processes) occurs. It suppresses the warning and prevents unsafe parallelism in forked children.

## System Requirements

### Minimum

- **GPU:** NVIDIA RTX 5070, 5080, or 5090
- **Drivers:** NVIDIA driver 550+ (for CUDA 12.8 support)
- **RAM:** 16GB+ system RAM
- **Storage:** ~5GB for PyTorch installation

### Recommended

- **Drivers:** Latest NVIDIA drivers
- **CUDA Toolkit:** 12.8 (optional, for development)

## Performance Notes

### With Correct PyTorch (2.9.1+cu128)
- ✅ Full native performance
- ✅ All CUDA operations supported
- ✅ No warnings or errors
- ✅ Optimal memory usage

### With Incompatible PyTorch (2.6.0, 2.7.0)
- ⚠️ Runtime errors on most operations
- ⚠️ "No kernel image" errors
- ❌ Cannot run training/inference

## Timeline

| Date | Event |
|------|-------|
| January 2026 | RTX 5070 launched with sm_120 |
| December 2025 | PyTorch 2.6 released (pre-RTX 5070) |
| Early 2026 | PyTorch 2.9 released with sm_120 support |

## Additional Resources

- **PyTorch Docs:** https://pytorch.org/get-started/locally/
- **CUDA Compatibility:** https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/
- **GPU Compute Capabilities:** https://developer.nvidia.com/cuda-gpus

## Quick Reference Card

```
┌──────────────────────────────────────────────────────┐
│ RTX 5070 PyTorch Setup - Quick Reference            │
├──────────────────────────────────────────────────────┤
│ GPU:          RTX 5070 / 5080 / 5090                 │
│ Compute Cap:  sm_120                                 │
│ PyTorch:      2.9.1+cu128 ✓                          │
│ CUDA:         12.8 ✓                                 │
│ Install:      pip install torch==2.9.1+cu128 \\       │
│               --index-url https://download.pytor...  │
└──────────────────────────────────────────────────────┘
```

## License

This document is released into the public domain (CC0). Share freely.

---

**Last Updated:** February 2026  
**Maintainer:** Stephen Smetana
