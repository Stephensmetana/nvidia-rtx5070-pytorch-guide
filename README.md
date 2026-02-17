# NVIDIA RTX 5070 PyTorch Compatibility Guide

[![License: CC0-1.0](https://img.shields.io/badge/License-CC0_1.0-lightgrey.svg)](http://creativecommons.org/publicdomain/zero/1.0/)
[![Last Updated](https://img.shields.io/badge/Updated-February_2026-blue.svg)]()

**Quick reference for running PyTorch on NVIDIA RTX 50-series GPUs (RTX 5070, 5080, 5090).**

## 🚀 Quick Start

```bash
pip install torch==2.9.1+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 📖 Documentation

👉 **[Read the full guide](NVIDIA_RTX_5070_PYTORCH_GUIDE.md)**

The complete guide covers:
- ✅ Why RTX 5070 requires PyTorch 2.9.1+cu128
- ✅ Installation instructions (pip, Docker, venv)
- ✅ Troubleshooting common errors
- ✅ System requirements and performance notes
- ✅ Verification scripts

## 🔍 Quick Verification

After installing PyTorch, verify your setup:

```bash
python verify_gpu.py
```

Or use the inline check:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## 🧩 The Problem

RTX 50-series GPUs have compute capability **sm_120**. PyTorch binaries need explicit sm_120 kernel support, which is only available in PyTorch 2.9.0+ with CUDA 12.8.

**Symptoms without proper version:**
- `UserWarning: NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not compatible...`
- `RuntimeError: CUDA error: no kernel image is available for execution on the device`

## 💡 Solution

Use **PyTorch 2.9.1 with CUDA 12.8** — the first stable release with full sm_120 support.

## 📝 Contributing

Found an issue or have an update? Please open an issue or submit a pull request!

Useful contributions:
- Updates for newer PyTorch versions
- Additional troubleshooting scenarios
- Driver compatibility notes
- Performance benchmarks

## 📜 License

This guide is released into the public domain under [CC0 1.0 Universal](LICENSE).

---

**Maintainer:** Stephen Smetana  
**Last Updated:** February 17, 2026
