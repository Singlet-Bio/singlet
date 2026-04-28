# Installation

## Python Client

```bash
pip install singlet
```

Or install from GitHub:
```bash
pip install "singlet @ git+https://github.com/Singlet-Bio/singlet#subdirectory=python"
```

## R Package

```r
# install.packages("remotes")
remotes::install_github("Singlet-Bio/singlet", subdir = "singlepress/R")
```

## GPU Library (requires CUDA 12+)

```bash
git clone https://github.com/Singlet-Bio/singlet
cd singlet/gpu
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

## Pipeline Binary (C++17)

```bash
cd singlet/pipeline
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSTAR_DIR=../../star/source
make -j$(nproc) singlify
```
