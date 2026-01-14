"""
Script to copy necessary DLLs into the Faiss Python package for wheel building.
"""
import os
import shutil
from pathlib import Path

# Paths
BUILD_DIR = Path(__file__).parent.parent.parent  # faiss/build
PYTHON_DIR = Path(__file__).parent  # faiss/build/faiss/python
FAISS_PKG_DIR = PYTHON_DIR / "faiss"

# Where to find DLLs
MKL_BIN = Path(r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin")
COMPILER_BIN = Path(r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin")
BUILD_RELEASE = BUILD_DIR / "faiss" / "Release"

# Required MKL DLLs
MKL_DLLS = [
    "mkl_core.2.dll",
    "mkl_intel_thread.2.dll",
    "mkl_rt.2.dll",
    "mkl_def.2.dll",
    "mkl_avx2.2.dll",
    "mkl_vml_def.2.dll",
    "mkl_vml_avx2.2.dll",
    "mkl_vml_cmpt.2.dll",
]

# OpenMP DLL
OPENMP_DLLS = [
    "libiomp5md.dll",
]

def copy_dll(src_path, dll_name, dest_dir):
    """Copy a DLL if it exists."""
    src = src_path / dll_name
    if src.exists():
        dest = dest_dir / dll_name
        if not dest.exists():
            print(f"Copying {dll_name} from {src_path}")
            shutil.copy2(src, dest)
        return True
    return False

def main():
    print("Packaging DLLs for Faiss wheel...")
    
    # Ensure faiss package directory exists
    if not FAISS_PKG_DIR.exists():
        print(f"Error: Faiss package directory not found: {FAISS_PKG_DIR}")
        return 1
    
    copied = []
    missing = []
    
    # Copy MKL DLLs
    print("\nCopying MKL DLLs...")
    for dll in MKL_DLLS:
        if copy_dll(MKL_BIN, dll, FAISS_PKG_DIR):
            copied.append(dll)
        else:
            print(f"  Warning: {dll} not found in {MKL_BIN}")
            missing.append(dll)
    
    # Copy OpenMP DLL
    print("\nCopying OpenMP DLL...")
    for dll in OPENMP_DLLS:
        if copy_dll(COMPILER_BIN, dll, FAISS_PKG_DIR):
            copied.append(dll)
        else:
            print(f"  Warning: {dll} not found in {COMPILER_BIN}")
            missing.append(dll)
    
    # Copy CUDA DLLs from uv environment (torch includes them)
    print("\nLooking for CUDA DLLs in torch package...")
    try:
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        cuda_dlls = [
            "cudart64_12.dll",
            "cublas64_12.dll",
            "cublasLt64_12.dll",
            "cusparse64_12.dll",
        ]
        for dll in cuda_dlls:
            if copy_dll(torch_lib, dll, FAISS_PKG_DIR):
                copied.append(dll)
    except ImportError:
        print("  Warning: torch not available, CUDA DLLs not copied")
    
    print(f"\n{'='*60}")
    print(f"Copied {len(copied)} DLLs to {FAISS_PKG_DIR}")
    if missing:
        print(f"Missing {len(missing)} DLLs: {', '.join(missing[:5])}")
    print(f"{'='*60}\n")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
