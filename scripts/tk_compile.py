"""
ThunderKittens CUDA compilation utilities.

This module provides functions for compiling ThunderKittens CUDA kernels
into Python modules, both locally and on Modal.
"""

import os
import subprocess
import sys
import tempfile
import glob
import site
import sysconfig
from typing import Optional


def find_thunderkittens_path(repo_top_path: Optional[str] = None) -> str:
    """
    Find the ThunderKittens installation path.
    
    Args:
        repo_top_path: Optional path to the repository root for local searches
        
    Returns:
        Path to ThunderKittens directory
        
    Raises:
        RuntimeError: If ThunderKittens is not found
    """
    # Try environment variables first
    tk_path = os.environ.get("THUNDERKITTENS_PATH") or os.environ.get("THUNDERKITTENS_ROOT")
    
    if not tk_path:
        # Try common locations
        candidates = []
        
        # Add repo-relative path if provided
        if repo_top_path:
            candidates.append(os.path.join(repo_top_path, "ThunderKittens"))
        
        # Add standard locations
        candidates.extend([
            "/root/ThunderKittens",
            os.path.expanduser("~/ThunderKittens")
        ])
        
        for path in candidates:
            if os.path.exists(os.path.join(path, "include", "kittens.cuh")):
                tk_path = path
                break
    
    if not tk_path or not os.path.exists(tk_path):
        raise RuntimeError(
            "ThunderKittens not found. Set THUNDERKITTENS_PATH or THUNDERKITTENS_ROOT "
            "environment variable, or ensure ThunderKittens is in a standard location."
        )
    
    return tk_path


def compile_thunderkittens_cuda(
    cuda_src_path: str,
    module_name: str = "tk_kernels",
    build_dir: Optional[str] = None,
    verbose: bool = False,
    repo_top_path: Optional[str] = None
) -> str:
    """
    Compile a ThunderKittens .cu file into a Python module (local compilation).
    
    Args:
        cuda_src_path: Path to the .cu file
        module_name: Name of the compiled module (default: tk_kernels)
        build_dir: Build directory for compiled artifacts (default: temp directory)
        verbose: Whether to print compilation output
        repo_top_path: Optional path to repository root for finding ThunderKittens
        
    Returns:
        Path to the directory containing the compiled module
    """
    # Find ThunderKittens
    tk_path = find_thunderkittens_path(repo_top_path)
    print(f"[INFO] Using ThunderKittens at: {tk_path}")
    
    # Read the CUDA source
    with open(cuda_src_path, 'r') as f:
        cuda_source = f.read()
    
    # Create build directory
    if build_dir is None:
        build_dir = tempfile.mkdtemp(prefix="tk_build_")
    os.makedirs(build_dir, exist_ok=True)
    
    # Write the CUDA source to the build directory
    cu_file = os.path.join(build_dir, f"{module_name}.cu")
    with open(cu_file, 'w') as f:
        f.write(cuda_source)
    
    # Create setup.py for compilation
    # Note: torch.utils.cpp_extension automatically includes pybind11 headers
    # We don't need to import pybind11 - CUDAExtension handles it
    setup_py = f'''
import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

TK_PATH = "{tk_path}"

setup(
    name="{module_name}",
    ext_modules=[
        CUDAExtension(
            name="{module_name}",
            sources=["{cu_file}"],
            include_dirs=[
                TK_PATH,
                os.path.join(TK_PATH, "include"),
            ],
            extra_compile_args={{
                "cxx": ["-std=c++20", "-O3", "-fPIC"],
                "nvcc": [
                    "-std=c++20", "-O3",
                    "-arch=sm_90a",
                    "-DNDEBUG",
                    "-DKITTENS_HOPPER",
                    "-DKITTENS_BLACKWELL",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "-Xcompiler", "-fPIC",
                    "-diag-suppress=20012",
                ],
            }},
            extra_link_args=["-lcuda"],
            language="c++",
        )
    ],
    cmdclass={{"build_ext": BuildExtension}},
)
'''
    
    setup_file = os.path.join(build_dir, "setup.py")
    with open(setup_file, 'w') as f:
        f.write(setup_py)
    
    # Compile the extension
    print(f"[INFO] Compiling {cuda_src_path} as module '{module_name}'...")
    
    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "9.0"
    
    try:
        result = subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            cwd=build_dir,
            capture_output=not verbose,
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Compilation failed:")
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
            raise RuntimeError(f"Failed to compile {cuda_src_path}")
        
        if verbose and result.stdout:
            print(result.stdout)
            
    except Exception as e:
        raise RuntimeError(f"Failed to compile {cuda_src_path}: {e}")
    
    print(f"[INFO] Successfully compiled {module_name} to {build_dir}")
    return build_dir


def compile_cuda_on_modal(
    cuda_src: str,
    module_name: str,
    gpu_arch: list,
    repo_top_path: Optional[str] = None
) -> str:
    """
    Compile CUDA source on Modal using nvcc directly (matching the Makefile approach).
    
    Args:
        cuda_src: CUDA source code as a string
        module_name: Name of the compiled module
        gpu_arch: List of GPU architectures (e.g., ["Hopper"])
        repo_top_path: Optional path to repository root for finding ThunderKittens
        
    Returns:
        Path to the directory containing the compiled module
    """
    from src.utils import set_gpu_arch
    
    set_gpu_arch(gpu_arch)
    
    # Find ThunderKittens
    tk_path = find_thunderkittens_path(repo_top_path)
    print(f"[Modal] Using ThunderKittens at: {tk_path}")
    
    # Create build directory
    build_dir = tempfile.mkdtemp(prefix="tk_modal_build_")
    os.makedirs(build_dir, exist_ok=True)
    
    # Write the CUDA source
    cu_file = os.path.join(build_dir, f"{module_name}.cu")
    with open(cu_file, 'w') as f:
        f.write(cuda_src)
    
    # Get pybind11 includes - try command line first, then find in site-packages
    pybind11_includes = ""
    try:
        pybind11_result = subprocess.run(
            [sys.executable, "-m", "pybind11", "--includes"],
            capture_output=True,
            text=True,
            check=True
        )
        pybind11_includes = pybind11_result.stdout.strip()
    except:
        # Fallback: find pybind11 in site-packages
        for site_pkg in site.getsitepackages():
            pybind11_paths = glob.glob(os.path.join(site_pkg, "pybind11", "include"))
            if pybind11_paths:
                pybind11_includes = f"-I{pybind11_paths[0]}"
                break
        
        # If still not found, try common locations
        if not pybind11_includes:
            common_paths = [
                "/usr/local/include/pybind11",
                "/usr/include/pybind11",
                os.path.expanduser("~/.local/include/pybind11"),
            ]
            for path in common_paths:
                if os.path.exists(path):
                    pybind11_includes = f"-I{path}"
                    break
        
        if not pybind11_includes:
            print("[Modal WARNING] pybind11 includes not found, compilation may fail")
    
    # Get Python config - try python3-config first, then python-config
    python_ldflags = ""
    try:
        python_config_result = subprocess.run(
            ["python3-config", "--ldflags"],
            capture_output=True,
            text=True,
            check=True
        )
        python_ldflags = python_config_result.stdout.strip()
    except:
        try:
            python_config_result = subprocess.run(
                ["python-config", "--ldflags"],
                capture_output=True,
                text=True,
                check=True
            )
            python_ldflags = python_config_result.stdout.strip()
        except:
            # Fallback - try to construct from sysconfig
            python_ldflags = f"-L{sysconfig.get_config_var('LIBDIR')} -lpython{sys.version_info.major}.{sys.version_info.minor}"
    
    # Get Python extension suffix
    try:
        ext_suffix_result = subprocess.run(
            ["python3-config", "--extension-suffix"],
            capture_output=True,
            text=True,
            check=True
        )
        ext_suffix = ext_suffix_result.stdout.strip()
    except:
        try:
            ext_suffix_result = subprocess.run(
                ["python-config", "--extension-suffix"],
                capture_output=True,
                text=True,
                check=True
            )
            ext_suffix = ext_suffix_result.stdout.strip()
        except:
            # Fallback
            ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    
    # Build nvcc command matching the Makefile
    output_so = os.path.join(build_dir, f"{module_name}{ext_suffix}")
    
    # Parse pybind11 includes (they come as "-I/path1 -I/path2")
    pybind11_include_list = pybind11_includes.split() if pybind11_includes else []
    
    # Parse python ldflags (they come as "-L/path -lpython3.10 ...")
    python_ldflags_list = python_ldflags.split() if python_ldflags else []
    
    nvcc_flags = [
        "-DNDEBUG",
        "-Xcompiler", "-fPIE",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
        "-Xcompiler", "-Wno-psabi",
        "-Xcompiler", "-fno-strict-aliasing",
        "--use_fast_math",
        "-forward-unknown-to-host-compiler",
        "-O3",
        "-Xnvlink=--verbose",
        "-Xptxas=--verbose",
        "-Xptxas=--warn-on-spills",
        "-std=c++20",
        "-x", "cu",
        "-lrt", "-lpthread", "-ldl", "-lcuda", "-lcudadevrt", "-lcudart_static", "-lcublas",
        f"-I{tk_path}/include",
    ]
    
    # Add prototype include if it exists
    if os.path.exists(os.path.join(tk_path, "prototype")):
        nvcc_flags.append(f"-I{tk_path}/prototype")
    
    nvcc_flags.extend(pybind11_include_list)
    nvcc_flags.extend(python_ldflags_list)
    nvcc_flags.extend([
        "-shared",
        "-fPIC",
        f"-lpython{sys.version_info.major}.{sys.version_info.minor}",
        "-DKITTENS_HOPPER",
        "-DKITTENS_BLACKWELL",
        "-arch=sm_90a",
        cu_file,
        "-o", output_so
    ])
    
    # Filter out empty strings
    nvcc_flags = [f for f in nvcc_flags if f]
    
    print(f"[Modal] Compiling {module_name} with nvcc...")
    print(f"[Modal] Build directory: {build_dir}")
    print(f"[Modal] CUDA file: {cu_file}")
    print(f"[Modal] Output: {output_so}")
    
    # Run nvcc
    result = subprocess.run(
        ["nvcc"] + nvcc_flags,
        cwd=build_dir,
        capture_output=True,
        text=True
    )
    
    # Always print output for debugging
    if result.stdout:
        print(f"[Modal] Compilation stdout:\n{result.stdout}")
    if result.stderr:
        print(f"[Modal] Compilation stderr:\n{result.stderr}")
    
    if result.returncode != 0:
        print(f"[Modal ERROR] Compilation failed with return code {result.returncode}")
        print(f"[Modal ERROR] Full stdout:\n{result.stdout}")
        print(f"[Modal ERROR] Full stderr:\n{result.stderr}")
        raise RuntimeError(f"Failed to compile CUDA module: {result.stderr[:500] if result.stderr else 'Unknown error'}")
    
    # Verify the .so file was created
    if not os.path.exists(output_so):
        raise RuntimeError(f"Compilation succeeded but .so file not found: {output_so}")
    
    print(f"[Modal] Successfully compiled {module_name}")
    print(f"[Modal] Generated .so file: {output_so}")
    return build_dir


def prepare_kernel_src_with_cuda(
    kernel_py_src: str,
    cuda_module_path: str,
    module_name: str = "tk_kernels"
) -> str:
    """
    Prepare the Python kernel source to use the pre-compiled CUDA module.
    Adds the module path to sys.path so import works.
    
    Args:
        kernel_py_src: Original Python kernel source code
        cuda_module_path: Path to the directory containing the compiled module
        module_name: Name of the compiled module (default: tk_kernels)
        
    Returns:
        Modified Python source code with import hook
    """
    import_hook = f'''
import sys
import os
# Add compiled CUDA module to path
_tk_module_path = "{cuda_module_path}"
if _tk_module_path not in sys.path:
    sys.path.insert(0, _tk_module_path)
'''
    return import_hook + "\n" + kernel_py_src

