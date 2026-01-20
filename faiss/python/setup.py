# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

import os
import platform
import shutil
import glob

from setuptools import setup
from setuptools.dist import Distribution


# Tell setuptools this is a platform-specific binary wheel
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


# make the faiss python package dir
shutil.rmtree("faiss", ignore_errors=True)
os.mkdir("faiss")
shutil.copytree("contrib", "faiss/contrib")
shutil.copyfile("__init__.py", "faiss/__init__.py")
shutil.copyfile("loader.py", "faiss/loader.py")
shutil.copyfile("class_wrappers.py", "faiss/class_wrappers.py")
shutil.copyfile("gpu_wrappers.py", "faiss/gpu_wrappers.py")
shutil.copyfile("extra_wrappers.py", "faiss/extra_wrappers.py")
shutil.copyfile("array_conversions.py", "faiss/array_conversions.py")

if platform.system() != "AIX":
    ext = ".pyd" if platform.system() == "Windows" else ".so"
else:
    ext = ".a"

# Check for Release/ prefix (MSBuild) or no prefix (Ninja/Make)
# Try Release/ first, fall back to current directory
def find_lib(name):
    """Find library with or without Release/ prefix"""
    release_path = f"Release/{name}"
    if os.path.exists(release_path):
        return release_path
    if os.path.exists(name):
        return name
    return None

swigfaiss_generic_lib = find_lib(f"_swigfaiss{ext}")
swigfaiss_avx2_lib = find_lib(f"_swigfaiss_avx2{ext}")
swigfaiss_avx512_lib = find_lib(f"_swigfaiss_avx512{ext}")
swigfaiss_avx512_spr_lib = find_lib(f"_swigfaiss_avx512_spr{ext}")
callbacks_lib = find_lib(f"libfaiss_python_callbacks{ext}")
swigfaiss_sve_lib = find_lib(f"_swigfaiss_sve{ext}")
faiss_example_external_module_lib = f"_faiss_example_external_module{ext}"

found_swigfaiss_generic = swigfaiss_generic_lib is not None
found_swigfaiss_avx2 = swigfaiss_avx2_lib is not None
found_swigfaiss_avx512 = swigfaiss_avx512_lib is not None
found_swigfaiss_avx512_spr = swigfaiss_avx512_spr_lib is not None
found_callbacks = callbacks_lib is not None
found_swigfaiss_sve = swigfaiss_sve_lib is not None
found_faiss_example_external_module_lib = os.path.exists(faiss_example_external_module_lib)

if platform.system() != "AIX":
    assert (
        found_swigfaiss_generic
        or found_swigfaiss_avx2
        or found_swigfaiss_avx512
        or found_swigfaiss_avx512_spr
        or found_swigfaiss_sve
        or found_faiss_example_external_module_lib
    ), (
        f"Could not find _swigfaiss{ext} or _swigfaiss_avx2{ext} or "
        f"_swigfaiss_avx512{ext} or _swigfaiss_avx512_spr{ext} or "
        f"_swigfaiss_sve{ext} or _faiss_example_external_module{ext}. "
        f"Faiss may not be compiled yet."
    )

if found_swigfaiss_generic:
    print(f"Copying {swigfaiss_generic_lib}")
    shutil.copyfile("swigfaiss.py", "faiss/swigfaiss.py")
    shutil.copyfile(swigfaiss_generic_lib, f"faiss/_swigfaiss{ext}")

if found_swigfaiss_avx2:
    print(f"Copying {swigfaiss_avx2_lib}")
    shutil.copyfile("swigfaiss_avx2.py", "faiss/swigfaiss_avx2.py")
    shutil.copyfile(swigfaiss_avx2_lib, f"faiss/_swigfaiss_avx2{ext}")

if found_swigfaiss_avx512:
    print(f"Copying {swigfaiss_avx512_lib}")
    shutil.copyfile("swigfaiss_avx512.py", "faiss/swigfaiss_avx512.py")
    shutil.copyfile(swigfaiss_avx512_lib, f"faiss/_swigfaiss_avx512{ext}")

if found_swigfaiss_avx512_spr:
    print(f"Copying {swigfaiss_avx512_spr_lib}")
    shutil.copyfile("swigfaiss_avx512_spr.py", "faiss/swigfaiss_avx512_spr.py")
    shutil.copyfile(swigfaiss_avx512_spr_lib, f"faiss/_swigfaiss_avx512_spr{ext}")

if found_callbacks:
    print(f"Copying {callbacks_lib}")
    shutil.copyfile(callbacks_lib, f"faiss/libfaiss_python_callbacks{ext}")

if found_swigfaiss_sve:
    print(f"Copying {swigfaiss_sve_lib}")
    shutil.copyfile("swigfaiss_sve.py", "faiss/swigfaiss_sve.py")
    shutil.copyfile(swigfaiss_sve_lib, f"faiss/_swigfaiss_sve{ext}")

if found_faiss_example_external_module_lib:
    print(f"Copying {faiss_example_external_module_lib}")
    shutil.copyfile("faiss_example_external_module.py", "faiss/faiss_example_external_module.py")
    shutil.copyfile(faiss_example_external_module_lib, f"faiss/_faiss_example_external_module{ext}")

# Windows: Bundle MKL DLLs (but not libiomp5md.dll - use PyTorch's OpenMP)
if platform.system() == "Windows":
    mkl_base = r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin"
    
    mkl_dlls = [
        "mkl_core.2.dll",
        "mkl_intel_thread.2.dll",
        "mkl_rt.2.dll",
        "mkl_def.2.dll",
        "mkl_avx2.2.dll",
        "mkl_vml_def.2.dll",
        "mkl_vml_avx2.2.dll",
        "mkl_vml_cmpt.2.dll",
    ]
    
    for dll in mkl_dlls:
        src = os.path.join(mkl_base, dll)
        if os.path.exists(src):
            print(f"Bundling {dll}")
            shutil.copyfile(src, f"faiss/{dll}")
        else:
            print(f"Warning: {dll} not found at {src}")

long_description = """
Faiss is a library for efficient similarity search and clustering of dense
vectors. It contains algorithms that search in sets of vectors of any size,
up to ones that possibly do not fit in RAM. It also contains supporting
code for evaluation and parameter tuning. Faiss is written in C++ with
complete wrappers for Python/numpy. Some of the most useful algorithms
are implemented on the GPU. It is developed by Facebook AI Research.
"""
setup(
    name="faiss",
    version="1.13.2+wingpu",
    description="A library for efficient similarity search and clustering of dense vectors",
    long_description=long_description,
    long_description_content_type="text/plain",
    url="https://github.com/facebookresearch/faiss",
    author="Matthijs Douze, Jeff Johnson, Herve Jegou, Lucas Hosseini",
    author_email="faiss@meta.com",
    license="MIT",
    keywords="search nearest neighbors",
    install_requires=["numpy>=1.21,<2.0", "packaging"],
    packages=["faiss", "faiss.contrib", "faiss.contrib.torch"],
    package_data={
        "faiss": ["*.so", "*.pyd", "*.a", "*.dll"],
    },
    zip_safe=False,
    distclass=BinaryDistribution,
)
