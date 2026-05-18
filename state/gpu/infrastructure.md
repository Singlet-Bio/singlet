# singlet-gpu — Infrastructure

> **HOME QUOTA**: `/mnt/home/debruinz/` is at the 100 GB hard limit. Writing benchmark outputs, build artifacts, envs, or caches to `~/` is a hard error. Source code (text, headers, CMake, tests) lives at `~/Singlet-AI/singlet-gpu/`. Everything else goes to `/mnt/projects/debruinz_project/singlet-gpu/`.

## Output paths

| What | Write here |
|---|---|
| Benchmark results, timing logs | `/mnt/projects/debruinz_project/singlet-gpu/benchmarks/` |
| Compiled test binaries, build artifacts | `/mnt/projects/debruinz_project/singlet-gpu/build/` |
| Python/R envs for benchmarking | `/mnt/projects/debruinz_project/singlet-gpu/envs/` |
| Experiment data, correctness fixtures | `/mnt/projects/debruinz_project/singlet-gpu/experiments/` |
| Source code (headers, CMake, tests, docs) | `~/Singlet-AI/singlet-gpu/` |
| Transient job scratch | `/dev/shm/` (cleaned on job end) |

Never use `/tmp` for persistent data.

## Cluster (Clipper)

- **Login node**: c001 / current — no GPU.
- **GPU partition** (`sinfo -p gpu`):
  - g001–g004, g008: V100S 32 GB (sm_70).
  - g005, g050: mixed-state V100S.
  - g051, g052: H100 NVL (sm_90).
- **CPU partition**: standard.
- **CUDA**: 12.x at `/usr/local/cuda/`. cuBLAS, cuSPARSE, cuSOLVER, cuRAND, cuFFT, cuDNN.
- **GCC**: gcc-toolset-13 at `/opt/rh/gcc-toolset-13/root/bin/g++` (required for C++20 + CUDA 12).
- **Python**: system `python3` is **3.9.23** (past EOL). Our `pyproject.toml` requires `>=3.10`. Load a newer Python via lmod before any `pip install`:
  ```bash
  source /etc/profile.d/lmod.sh
  module load python/3.11.14   # or 3.10.19 / 3.12.12 / 3.13.8 / 3.14.0
  ```
  `module avail python` lists all five. Default is the EOL'd 3.9.23; never use it for the wheel/editable install.

  **Important**: after `module load`, the `python` and `python3` aliases update — but the `pip` binary (without version suffix) still resolves to `/usr/bin/pip` which is hardcoded to Python 3.9. **Always invoke pip as `python -m pip install ...`** after the module load, not bare `pip install`.

  Second gotcha: the Spack-built Python modules (`python/3.11.14`, etc.) **do not include pip** out of the box. Bootstrap before first use:
  ```bash
  module load python/3.11.14
  python -m ensurepip --user --upgrade
  python -m pip install ...   # now works
  ```

## Data

- singlet quant outputs (read-only): `/mnt/projects/debruinz_project/singlet_pipeline/quant/scrna/GSE*/GSE*/GSM*/`
- factornet (GPL-2.0): `/mnt/home/debruinz/factornet/include/factornet/`
- Eigen 3.4.0: `/opt/gvsu/clipper/2024.05/spack/apps/linux-rhel9-cascadelake/gcc-11.4.1/eigen-3.4.0-fb3i5nzixuz47jnbcqemhiuwv4kmftft/include/eigen3`

## Environment variables

### Build / runtime (non-secret, safe to inline)

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export CXX=/opt/rh/gcc-toolset-13/root/bin/g++
export CC=/opt/rh/gcc-toolset-13/root/bin/gcc
```

### Supabase / publishing (secrets — never inline)

Phase G requires `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, and `SUPABASE_ANON_KEY`.

**Never write these values into any file under `singlet-gpu/`.** They live in `~/.config/singlet/supabase.env` (chmod 600, outside the repo). Source the helper before any publish step:

```bash
source ~/Singlet-AI/singlet-gpu/scripts/load_secrets.sh
```

The helper reads `~/.config/singlet/supabase.env` and exports the three vars. SLURM jobs that need to publish should add the source line above to the script preamble. To make this automatic for interactive shells, add to `~/.bashrc`:

```bash
[ -f ~/.config/singlet/supabase.env ] && source ~/.config/singlet/supabase.env
```

If `~/.config/singlet/supabase.env` is missing or unreadable, `load_secrets.sh` prints the create-with-perms recipe to stderr and exits non-zero.

#### Rotating keys

If a key is leaked (committed to a repo, pasted in chat, etc.):
1. In the Supabase dashboard, regenerate the service-role key.
2. Overwrite `~/.config/singlet/supabase.env` with the new value.
3. No source-tree changes required — the repo only references variable names.

## Build template

`-O3 --use_fast_math -std=c++20 -DFACTORNET_HAS_GPU=1`. CMake INTERFACE target `singlet-gpu::singlet-gpu` sets the flag for everyone who links it.

## Canonical sbatch template (the reason Cycle 88's first build failed)

Every cycle's verify/bench script must:
1. `export PATH=/usr/local/cuda/bin:$PATH` (otherwise `nvcc` is not found).
2. Pass `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`.
3. Pass `-DCMAKE_CUDA_ARCHITECTURES="70;80;90"` (covers V100S + H100).
4. Pass `-DCMAKE_CXX_COMPILER=/opt/rh/gcc-toolset-13/root/bin/g++`.
5. Pass `-DEIGEN_INCLUDE_DIR=...` (factornet transitively needs it).
6. Use a build dir under `/mnt/projects/debruinz_project/singlet-gpu/build/cycle{N}_*/`.
7. `rm -rf "$BUILD_DIR"` first if doing a cold rebuild — cmake cache contamination otherwise.

Reference: `state/cycle88_verify.sh` (working) vs `state/cycle88_build.sh` (broken — missing PATH export). Always copy the working template.

## SOTA reference environments

| Tool | Language | Where |
|---|---|---|
| rapids-singlecell | Python/cupy | benchmark venv on g051 (RAPIDS install verified) |
| Scanpy | Python | benchmark venv |
| Seurat / scran | R | per-node R install (g001 confirmed; g008 missing scran — see blockers) |
| cuml / cuGraph | Python/C++ | RAPIDS bundle |
| FAISS-GPU | C++/Python | benchmark venv |
| RAFT / cuVS / CAGRA | C++ | requires `pip install cuvs-cu12` — see blockers |
| scvi-tools | Python/PyTorch | benchmark venv (correctness reference only; we do not ship it) |
| factornet (CPU) | C++ | header-only at the path above |
| fgsea / AUCell / harmonypy | R / Python | per-env |

## Python wrapper test environment (CYCLE-189)

Verified install set used in SLURM job 372552 (g003, V100S-PCIE-32GB, CUDA 12.8):

| Package | Version | Notes |
|---|---|---|
| Python | 3.11.14 | via `module load python/3.11.14` |
| cupy-cuda12x | 14.0.1 | cupy 14 — triggers dtype-strictness + cupy.sparse removal |
| numpy | 2.4.4 | |
| anndata | 0.12.11 | |
| scanpy | 1.11.5 | |
| scipy | 1.17.1 | |
| scikit-learn | 1.8.0 | |
| pandas | 2.3.3 | |
| pytest | 9.0.3 | |

**pyproject.toml pins (CYCLE-189)**: `cupy-cuda12x>=13.0,<15`, `numpy>=1.24,<2.6`,
`anndata>=0.10,<0.13`, `scanpy>=1.10,<1.12` (optional), `scipy>=1.11,<1.18` (dev).

**cupy 14 compat fixes applied (CYCLE-189)**:
- `cupy.sparse` → `cupyx.scipy.sparse` try/except fallback in 14 files.
- `cp.asarray(dict)` → `cp.asarray(_CaiView(dict))` shim in `io/loader.py` and `io.py`
  (cupy 14 requires `__cuda_array_interface__` as an object attribute, not a bare dict).

## Canonical test samples

| Sample | Cells | Notes |
|---|---|---|
| GSM4037629 (scRNA) | 11,560 | smoke + small-scale correctness; full artifact suite |
| 5-sample concat (scRNA) | ~100k | medium-scale benchmark |
| All available | 1M+ | large-scale streaming benchmark |
| Tiny synthetic (fixed seed) | 500 × 200 | unit-test smoke |
