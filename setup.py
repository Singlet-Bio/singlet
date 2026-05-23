from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "singlet._pz",
        ["src/bindings/python/_pz.cpp"],
        include_dirs=["include"],
        libraries=["zstd"],
        cxx_std=17,
    ),
]

setup(ext_modules=ext_modules, cmdclass={"build_ext": build_ext})
