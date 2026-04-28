"""Build script for singlepress C++ extension.

Builds:
  _pz_codec     — .1pz format read/write (VOCSC + byte-split + zstd)
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    """Custom build extension for pybind11."""

    def build_extensions(self):
        import pybind11

        for ext in self.extensions:
            ext.include_dirs.append(pybind11.get_include())
            ext.include_dirs.append(pybind11.get_include(user=True))

        # Compiler flags
        ct = self.compiler.compiler_type
        for ext in self.extensions:
            opts = list(ext.extra_compile_args or [])
            if ct == "unix":
                if "-std=c++17" not in opts:
                    opts.insert(0, "-std=c++17")
                if "-O3" not in opts:
                    opts.append("-O3")
                opts.append("-fvisibility=hidden")
            elif ct == "msvc":
                opts.append("/std:c++17")
                opts.append("/O2")
            ext.extra_compile_args = opts

        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "singlepress._pz_codec",
        sources=["singlepress/pz_codec.cpp", "singlepress/lz4.c", "singlepress/lz4hc.c"],
        libraries=["zstd"],
        include_dirs=["singlepress"],
        extra_compile_args=["-std=c++17", "-O3", "-fopenmp", "-march=native"],
        extra_link_args=["-fopenmp"],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
