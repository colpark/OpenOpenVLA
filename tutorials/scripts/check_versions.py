#!/usr/bin/env python
"""
Check and diagnose version compatibility for OpenVLA.

OpenVLA was trained and tested with specific versions:
- transformers==4.40.1
- tokenizers==0.19.1
- timm==0.9.10

Using different versions (especially newer transformers) can cause
inference-time regressions including constant output issues.
"""

import sys

def check_versions():
    print("=" * 60)
    print(" OpenVLA Version Compatibility Check")
    print("=" * 60)

    issues = []
    warnings = []

    # Check transformers
    try:
        import transformers
        installed = transformers.__version__
        expected = "4.40.1"
        if installed != expected:
            issues.append(f"transformers: {installed} (expected {expected})")
            print(f"\n❌ transformers: {installed}")
            print(f"   Expected: {expected}")
            print(f"   CRITICAL: This is likely causing inference issues!")
        else:
            print(f"\n✅ transformers: {installed}")
    except ImportError:
        issues.append("transformers not installed")
        print("\n❌ transformers: NOT INSTALLED")

    # Check tokenizers
    try:
        import tokenizers
        installed = tokenizers.__version__
        expected = "0.19.1"
        if installed != expected:
            if installed > "0.19.1":
                warnings.append(f"tokenizers: {installed} (expected {expected})")
            else:
                issues.append(f"tokenizers: {installed} (expected {expected})")
            print(f"\n⚠️  tokenizers: {installed}")
            print(f"   Expected: {expected}")
        else:
            print(f"\n✅ tokenizers: {installed}")
    except ImportError:
        issues.append("tokenizers not installed")
        print("\n❌ tokenizers: NOT INSTALLED")

    # Check timm
    try:
        import timm
        installed = timm.__version__
        valid_versions = {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}
        if installed not in valid_versions:
            issues.append(f"timm: {installed} (expected one of {valid_versions})")
            print(f"\n❌ timm: {installed}")
            print(f"   Expected: one of {valid_versions}")
        else:
            print(f"\n✅ timm: {installed}")
    except ImportError:
        issues.append("timm not installed")
        print("\n❌ timm: NOT INSTALLED")

    # Check torch
    try:
        import torch
        installed = torch.__version__
        print(f"\n📦 torch: {installed}")
        if torch.cuda.is_available():
            print(f"   CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("   CUDA: Not available")
    except ImportError:
        print("\n❌ torch: NOT INSTALLED")

    # Summary
    print("\n" + "=" * 60)

    if issues:
        print(" ❌ ISSUES FOUND - Inference may not work correctly!")
        print("=" * 60)
        print("\nTo fix, run the following in your environment:")
        print()
        print("  pip install transformers==4.40.1 tokenizers==0.19.1 timm==0.9.10")
        print()
        print("Or create a fresh environment:")
        print()
        print("  conda create -n openvla python=3.10")
        print("  conda activate openvla")
        print("  pip install -e .")
        print()
        print("This will install the correct versions from pyproject.toml.")
        return False
    elif warnings:
        print(" ⚠️  WARNINGS - Some versions differ but may work")
        print("=" * 60)
        return True
    else:
        print(" ✅ ALL VERSIONS COMPATIBLE")
        print("=" * 60)
        return True


def show_fix_instructions():
    """Show detailed fix instructions."""
    print("""
================================================================================
 HOW TO FIX VERSION ISSUES
================================================================================

Option 1: Quick Fix (pip install)
---------------------------------
pip uninstall transformers tokenizers -y
pip install transformers==4.40.1 tokenizers==0.19.1

Option 2: Fresh Conda Environment (Recommended)
-----------------------------------------------
conda create -n openvla python=3.10 -y
conda activate openvla
cd /path/to/openvla
pip install -e .

Option 3: SciServer (where you can't use conda)
-----------------------------------------------
# Create requirements file
echo "transformers==4.40.1" > /tmp/version_fix.txt
echo "tokenizers==0.19.1" >> /tmp/version_fix.txt

# Install specific versions
pip install -r /tmp/version_fix.txt

================================================================================
 WHY VERSION MATTERS
================================================================================

Between transformers 4.40.1 and 4.50+, significant changes were made to:
1. GenerationMixin.generate() - how token generation works
2. Cache handling (DynamicCache vs tuple format)
3. LlamaForCausalLM internals

These changes cause OpenVLA to produce constant "zero action" outputs
instead of task-appropriate actions.

The OpenVLA code explicitly checks for version 4.40.1 and logs a warning:
  "there might be inference-time regressions due to dependency changes"

================================================================================
""")


if __name__ == "__main__":
    success = check_versions()

    if not success:
        print()
        show_fix_instructions()
        sys.exit(1)
    else:
        sys.exit(0)
