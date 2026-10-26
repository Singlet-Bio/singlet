from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

import sys

ext_modules = [
    Pybind11Extension(
        "singlet._pz",
        ["src/bindings/python/_pz.cpp"],
        include_dirs=["include"],
        libraries=["zstd"],
        cxx_std=17,
        # The codec headers contain non-ASCII characters in comments; MSVC reads source as the
        # ANSI code page unless told otherwise.
        extra_compile_args=["/utf-8"] if sys.platform == "win32" else [],
    ),
    # The store matrix codec (include/singlet/pz2/, header-only; the SIMD paths are chosen at run time, so no -march flag)
    # and its client helpers (src/bindings/common/singlet/pz2_client.h).
    Pybind11Extension(
        "singlet._pz2",
        ["src/bindings/python/_pz2.cpp"],
        include_dirs=["include", "src/bindings/common"],
        libraries=["zstd"],
        cxx_std=17,
        # the encoder and the multi-threaded decode use std::thread: glibc < 2.34 (manylinux_2_28) needs -pthread
        extra_compile_args=["/utf-8"] if sys.platform == "win32" else ["-pthread"],
        extra_link_args=[] if sys.platform == "win32" else ["-pthread"],
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
