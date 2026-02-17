#!/usr/bin/env python3
"""
NVIDIA RTX 5070 PyTorch Verification Script

Checks if PyTorch is properly configured for RTX 50-series GPUs.
Verifies compute capability, CUDA version, and basic GPU functionality.
"""

import sys

def check_pytorch():
    """Verify PyTorch installation and GPU support."""
    
    print("=" * 60)
    print("RTX 5070 PyTorch Verification")
    print("=" * 60)
    print()
    
    # Check PyTorch import
    try:
        import torch
    except ImportError:
        print("❌ ERROR: PyTorch is not installed!")
        print("\nInstall with:")
        print("pip install torch==2.9.1+cu128 torchvision torchaudio \\")
        print("    --index-url https://download.pytorch.org/whl/cu128")
        return False
    
    # Print versions
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA version: {torch.version.cuda}")
    print()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA is not available!")
        print("\nPossible causes:")
        print("  1. NVIDIA drivers not installed (need 550+)")
        print("  2. Installed CPU-only PyTorch")
        print("  3. GPU not detected by system")
        print("\nCheck with: nvidia-smi")
        return False
    
    print(f"✓ CUDA is available")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    # Check GPU details
    for i in range(torch.cuda.device_count()):
        device_name = torch.cuda.get_device_name(i)
        capability = torch.cuda.get_device_capability(i)
        compute_cap = f"sm_{capability[0]}{capability[1]}"
        
        print(f"GPU {i}: {device_name}")
        print(f"  Compute capability: {compute_cap}")
        
        # Check for RTX 5070/5080/5090
        is_rtx_50_series = "RTX 50" in device_name or "RTX 5070" in device_name or "RTX 5080" in device_name or "RTX 5090" in device_name
        
        if is_rtx_50_series:
            if capability[0] == 12 and capability[1] == 0:  # sm_120
                print(f"  Status: ✅ Correct compute capability for RTX 50-series")
            else:
                print(f"  Status: ⚠️  Unexpected compute capability for RTX 50-series")
                print(f"           Expected sm_120, got {compute_cap}")
        
        # Check PyTorch version for RTX 5070
        if compute_cap == "sm_120":
            pytorch_version = torch.__version__
            if "2.9" in pytorch_version and "cu128" in pytorch_version:
                print(f"  PyTorch: ✅ Correct version for sm_120 ({pytorch_version})")
            else:
                print(f"  PyTorch: ❌ Incompatible version ({pytorch_version})")
                print(f"           Need PyTorch 2.9.1+cu128 for sm_120 support")
                print("\nInstall correct version:")
                print("pip install torch==2.9.1+cu128 torchvision torchaudio \\")
                print("    --index-url https://download.pytorch.org/whl/cu128")
                return False
        
        print()
    
    # Test GPU operations
    print("Testing GPU operations...")
    try:
        # Simple matrix multiplication
        device = torch.device("cuda:0")
        x = torch.randn(1000, 1000, device=device)
        y = x @ x
        
        # Ensure operation completed
        torch.cuda.synchronize()
        
        print("✓ Matrix multiplication test passed")
        print()
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")
        print("\nThis usually means:")
        print("  - PyTorch doesn't have kernels for your GPU compute capability")
        print("  - Need to install PyTorch 2.9.1+cu128 for RTX 5070 support")
        return False
    
    # Success
    print("=" * 60)
    print("✅ All checks passed! Your RTX GPU is properly configured.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = check_pytorch()
    sys.exit(0 if success else 1)
