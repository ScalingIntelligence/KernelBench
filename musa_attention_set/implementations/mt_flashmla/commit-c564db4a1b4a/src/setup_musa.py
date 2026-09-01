import os
from pathlib import Path
from datetime import datetime
import subprocess

from setuptools import setup, find_packages

from torch_musa.utils.musa_extension import MUSAExtension, BuildExtension

DISABLE_FP16 = os.getenv("FLASH_MLA_DISABLE_FP16", "FALSE") == "TRUE"


def get_sources():
    sources = [
        "csrc/flash_api.cpp",
        "csrc/flash_fwd_mla_bf16_mp31.mu",
        "csrc/flash_fwd_mla_metadata.mu",
    ]

    #if not DISABLE_FP16:
    #    sources.append("csrc/flash_fwd_mla_fp16_mp31.mu")

    return sources

def get_features_args():
    features_args = []
    if DISABLE_FP16:
        features_args.append("-DFLASH_MLA_DISABLE_FP16")
    return features_args

subprocess.run(["git", "submodule", "update", "--init", "csrc/mutlass"])


cc_flag = []
cc_flag.append("--offload-arch=mp_31")

this_dir = os.path.dirname(os.path.abspath(__file__))


cxx_args = ["force_mcc", "-O3", "-std=c++17", "-DNDEBUG", "-DUSE_MUSA", "-Wno-deprecated-declarations"]

ext_modules = []
ext_modules.append(
    MUSAExtension(
      name="flash_mla_musa",
      sources=get_sources(),
      extra_compile_args={
          "cxx": cxx_args + get_features_args(),
          "mcc": [
                    "-O2",
                    "-Od3",
                    "-std=c++17",
                    "-Wno-deprecated-declarations",
                    "-DNDEBUG",
                    "-DUSE_MUSA",
                  ]
                  + cc_flag + get_features_args(),
      },
      include_dirs=[
          Path(this_dir) / "csrc",
          Path(this_dir) / "csrc" / "mutlass" / "include",
          Path(this_dir) / ".." / "mudnn" / "include",
          Path(this_dir) / ".." / "muThrust",
      ],
    )
)

try:
    cmd = ['git', 'rev-parse', '--short', 'HEAD']
    rev = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
except Exception as _:
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
    rev = '+' + date_time_str


setup(
    name="flash_mla",
    version="1.0.0" + rev,
    packages=find_packages(include=['flash_mla']),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
)
