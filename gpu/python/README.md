# singlet-gpu (Python)

GPU-native single-cell analysis library. Reads singlify `.1pz` outputs zero-copy into a CuSPARSE-compatible CSC, runs the entire foundational EDA workflow on device, and scales out-of-core to billion-cell datasets.

This package follows scanpy conventions for ergonomics — most functions are drop-in compatible with the corresponding `scanpy.pp.*` / `scanpy.tl.*` calls.

## Install

Requires CUDA 12.x and a GPU with `sm_70` (V100), `sm_80` (A100), or `sm_90` (H100).

```bash
pip install singlet-gpu
```

Local editable install:

```bash
cd singlet-gpu/python
pip install -e .
```

## Quick start

```python
import singlet_gpu as sg

# Zero-copy load
adata = sg.io.read_anndata("/path/to/.1pz_dir/")

# Standard pipeline (scanpy-compatible API)
sg.preprocess.calculate_qc_metrics(adata, qc_vars=("MT", "RIBO"))
sg.preprocess.filter_cells(adata, min_genes=200)
sg.preprocess.normalize_total(adata, target_sum=1e4)
sg.preprocess.log1p(adata)
sg.preprocess.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
sg.preprocess.scale(adata, max_value=10.0)

sg.reduce.svd.pca(adata, n_comps=50, backend="auto")
sg.pp.neighbors(adata, n_neighbors=15)

sg.qc.run_doublet_score(adata, embedding_key="X_pca")
sg.tools.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
```

## Documentation

- API reference + speedup numbers: <https://singlet.bio/docs>
- Notebooks: <https://singlet.bio/notebooks>
- Live benchmarks: <https://singlet.bio/benchmarks>

## License

GPL-2.0-or-later (inherited from factornet).
