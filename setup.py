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
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
