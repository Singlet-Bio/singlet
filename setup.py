"""Build configuration for singlet with C++ extension module."""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11


class BuildExt(build_ext):
    """Custom build extension for C++17 support."""

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = ["-std=c++17", "-O2", "-fvisibility=hidden"]
        if ct == "unix":
            opts.append("-fPIC")
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "singlet._singlepress",
        sources=["src/_singlepress.cpp"],
        include_dirs=[
            pybind11.get_include(),
            pybind11.get_include(user=True),
            "include",
        ],
        language="c++",
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
)
