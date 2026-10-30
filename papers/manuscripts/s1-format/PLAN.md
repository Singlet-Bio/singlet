# Experiment plan: the .s1 manuscript benchmark

This is the work order for the agent that runs the benchmark on a cluster. Nothing here has been downloaded or run.
Every number in the manuscript must come out of this pipeline; the manuscript only reads
`generated/numbers.tex` and the `generated/*.tsv` tables, and both are written only by `python -m s1bench report`.

Read this file, then `README.md` (layout and commands), then `datasets.yaml`.

## 0. Ground rules

1. **No number is typed by hand.** If a value is needed in the text, add a macro to `s1bench/report.py`; the
   manuscript prints a red `[??]` for any macro the pipeline has not produced. Do not edit `generated/` by hand.
2. **Lossless or it does not count.** Every format write is followed by a read and an exact comparison with the
   canonical matrix (`indptr`, `indices`, `data`, gene names, barcodes). A format that fails the round trip on a
   dataset is reported as failed, not dropped.
3. **Same work for every format.** All formats store the same canonical matrix (genes x cells, integer counts, gene
   ids and barcodes). Stored size always includes names. Timings always end with the same object (a SciPy CSC or a
   `torch.sparse_csr_tensor` of the same shape and dtype).
4. **Same hardware budget.** The headline uses 4 CPU cores per reader (`taskset`), whatever the reader does with them
   (threads, worker processes). Sweeps over 1-16 threads go to the supplement.
5. **Public data only, our code only.** Datasets are the public matrices in `datasets.yaml`. The manuscript names the
   `singlet` codebase and the `.s1` format; it does not refer to any hosted service or internal corpus.
6. **Record provenance.** Every result row carries the dataset id, matrix name, format id, the package versions
   (`s1bench env` writes `results/env.json`), the git commit of this repository and the host name.

## 1. Questions and the experiment that answers each

| # | Question (manuscript claim) | Stage | Figure / table |
|---|---|---|---|
| Q1 | How large is each format, in bits per non-zero, across protocols and sizes? | `write` | Fig 1a, Table 2 |
| Q2 | How fast does each format decode to an in-memory CSC on 1 and 4 threads? | `decode` | Fig 1c, Table 4 |
| Q3 | How long from a cold start to the first minibatch on the device, and what dominates? | `minibatch` | Fig 1b, Table 3 |
| Q4 | What sustained training throughput does each loader deliver at 0.1, 1 and 10 Gbit/s from the shared filesystem and from node-local disk? | `minibatch` | Fig 1d |
| Q5 | Is compression worth it on the wire, i.e. is R_eff(B) of .s1 above that of uncompressed CSC at every bandwidth? | `report` (from Q1, Q2) and `minibatch` (measured) | Fig 1c iso-lines, Table 4 |
| Q6 | Where do the bits go, step by step, from raw arrays to .s1? | `ladder` | Fig 3a |
| Q7 | How close is .s1 to the best achievable code length under count models (Poisson, NB, per-gene x depth, cluster-conditioned)? | `floor` | Fig 3b, Table 5 |
| Q8 | How much of the code length is sampling noise that no decoder can remove? | `floor` | Fig 3c |
| Q9 | Does any general-purpose compressor find redundancy left in .s1 output? | `second` | Fig 3d |
| Q10 | What does it cost to get a .s1 file into AnnData, SingleCellExperiment, Seurat, PyTorch and C++? | `eco` | Fig 4 code, Table 6 |
| Q11 | Do the results hold on count data outside single-cell (bulk RNA-seq, text)? | `write`, `floor` on `extensions` | Supplementary Table S2 |

## 2. Datasets

`datasets.yaml` lists 29 datasets (31 count matrices: the multiome peak matrix and the CITE-seq antibody matrix are
second matrices of their datasets) across 11 protocol classes and four size tiers (10 S, 9 M, 7 L, 3 XL), plus two
out-of-domain extensions.

- 22 entries are marked `verify: true`; 17 have `url: null`. For each, open the portal, find the file named in
  `source.file`, put its direct URL in `url`, and after the first fetch commit its `sha256` and `bytes`. Where the
  portal offers several versions (filtered vs raw beads, cell-bin vs bin50), keep the one named here and write the
  choice in `notes`.
- Use the filtered (called-cell) matrix every time. Empty droplets are out of scope.
- If a dataset cannot be obtained (portal login, removed file), replace it with one of the same protocol class and
  similar size from the reserve list below and record the swap in `datasets.yaml` and in the commit message.

Reserve list (same classes): 10x PBMC 1k v3 (droplet_3p, S); 10x Human Brain 3' nuclei (droplet_3p_nuclei); Seq-Well
PBMC (Gierahn et al. 2017, droplet_other); sci-RNA-seq3 human fetal atlas, one organ (Cao et al. 2020,
combinatorial); Smart-seq2 Tabula Sapiens subset (plate_reads); Visium mouse brain sagittal anterior (spatial_seq);
Vizgen MERSCOPE FFPE human immuno-oncology, one slide (spatial_imaging); CosMx NSCLC lung 5-1 (spatial_imaging).

**Canonical form** (`canon` stage). One `.npz` per matrix with: genes x cells CSC, `indices` uint32 strictly ascending in
every column, `data` uint32 > 0 (explicit zeros removed), `indptr` int64, gene ids (Ensembl where the source has them,
else symbols) and barcodes as UTF-8. Non-integer values abort the dataset with a clear message (some GEO tables are
normalised). Feature rows with zero total count are kept (formats must store what the source gave). Datasets with
several samples are concatenated in source order. Record the descriptors in `results/stats.tsv`: cells, genes, nnz,
density, mean and median UMIs per cell, fraction of non-zeros equal to 1, maximum value, and the fitted NB overdispersion
summary (median gene-wise 1/theta from the `floor` stage).

## 3. Formats

All writers are in `bench/s1bench/formats/` (Python) and `bench/R/formats.R` (R). Versions are pinned in
`bench/environment.yml`; `s1bench env` records what actually ran.

| id | what | writer | reader for decode timing | minibatch loader |
|---|---|---|---|---|
| `csc_i32` | CSC as raw int32 arrays (.npy), no compression | numpy | numpy | - |
| `mtx_gz` | Matrix Market + gzip -6 | scipy.io.mmwrite | scipy.io.mmread | - |
| `tenx_h5` | Cell Ranger HDF5 layout, gzip-4 + shuffle | h5py | h5py | - |
| `h5ad_none` | AnnData HDF5, no compression | anndata | anndata.read_h5ad | backed mode, scDataset |
| `h5ad_gzip` | AnnData HDF5, gzip-4 | anndata | anndata.read_h5ad | backed mode, scDataset |
| `h5ad_lzf` | AnnData HDF5, lzf | anndata | anndata.read_h5ad | - |
| `zarr` | AnnData Zarr, default Blosc compressor | anndata | anndata.read_zarr | anndata lazy Zarr |
| `loom` | loom (dense chunks, gzip) | loompy | loompy | - (skipped above 2e9 matrix entries) |
| `soma` | TileDB-SOMA experiment, default filters | tiledbsoma.io | tiledbsoma | tiledbsoma-ml |
| `parquet_coo` | COO triplets in Parquet, zstd | pyarrow | pyarrow | - |
| `csc_zstd19` | CSC int32 arrays, zstd level 19 | zstandard | zstandard | - |
| `rds_gz`, `rds_xz` | R `dgCMatrix` via `saveRDS` | R | R readRDS | - |
| `bpcells` | BPCells bit-packed matrix directory | R (BPCells) | R (BPCells) | R, time to dgCMatrix |
| `s1` | `.s1` sample object, one pz2 member per matrix, names inline | `s1bench.formats.s1` | `singlet._pz2` | `s1bench.s1_loader` |

Notes.
- `.s1` is written with names inline, not with shared vocabulary packs, so the comparison is like for like. The
  size with a shared gene-vocabulary pack is reported as a separate line (`s1_pack`), since packs are a format feature.
- The package has no public writer yet (gap G2 below). `s1bench.formats.s1` builds the object with
  `singlet.s1.container.S2Writer` and `singlet.s1.codec.NativeCodec`, the same calls the package uses internally.
  When G2 lands, switch to the public call and re-run `write` for `s1` only; sizes must not change.

## 4. Stages

Run order and dependencies: `fetch -> canon -> write -> {decode, ladder, floor, second} -> minibatch -> eco -> report`.
Every stage is idempotent, writes one TSV per (dataset, matrix) under `results/<stage>/`, and skips work whose
output exists unless `--force`. `slurm/array.sbatch` runs one dataset per array task.

### 4.1 `fetch`
Download every URL of a dataset into `data/raw/<id>/`, verify `sha256` if recorded, print it if not. No network access
in any later stage.

### 4.2 `canon`
Parse the source format into the canonical `.npz` (section 2) and append the descriptors to `results/stats/`.

### 4.3 `write` (Q1)
For every format: write from the canonical matrix, time the write (wall, 4 threads where the writer has threads),
measure bytes on disk (sum of all files for directory formats), read back and compare exactly. Output columns:
`dataset matrix format bytes bits_per_nnz write_s roundtrip_ok`.

### 4.4 `decode` (Q2, Q5)
For every format, time from a file on node-local disk to an in-memory SciPy CSC (or dgCMatrix for R formats), and,
cold at the headline thread count, from the shared NFS filesystem (`storages` in config.yaml). On Clipper the timing host
is an ordinary GPU node (H100, spinning RAID `/tmp`), deliberately not one with fast local storage.
- Cold: evict the files from the page cache first (`os.posix_fadvise(..., POSIX_FADV_DONTNEED)` on every file;
  verify with `vmtouch` if present). Warm: second read. Report both; the headline is cold.
- Threads: 1, 2, 4, 8, 16 for readers that take threads; everything is pinned with `taskset` to that many cores.
- `.s1` extras: `isa` in {scalar, avx2, avx512} (whatever the CPU supports) and `mode` in {native, canonical}
  (native keeps the frequency-ordered rows and gives the model a fixed gene permutation; canonical restores the source
  row order).
- 5 repetitions after 1 warm-up; report median and interquartile range. Output Mnnz/s = nnz / seconds.
- Also measure decode in C++ (`cpp/decode_bench`), which excludes Python overhead, for `.s1` only.

### 4.5 `ladder` (Q6)
For each matrix compute the payload bits per non-zero of: (L0) CSC as int32 indices + int32 values; (L1) absolute row
ids and values as LEB128 varints; (L2) row gaps + LEB128 in separate streams, zstd-3; (L3) the same, zstd-19; (L4)
the `.s1` member (whole member and chunk payload only); (F) the floor from the `floor` stage
(best cross-validated count model). The figure shows the median and range over the 31 matrices.

### 4.6 `floor` (Q7, Q8)
Ideal code lengths (sum of -log2 p over the whole matrix, zeros included, divided by nnz) under count models. All
parameters are fitted on one half of the cells and evaluated on the other half, then the halves swap (2-fold
cross-validation), so no model is charged or credited for memorising the data. Model description cost is not charged,
which makes every model's number a generous lower bound for a decoder that would ship that model.
- `poisson`: x_gc ~ Poisson(s_c mu_g), s_c = cell total / mean total.
- `nb`: x_gc ~ NB(s_c mu_g, theta_g), theta_g by moments with the sctransform-style smoothing over mean expression.
- `gene_depth`: per gene and per depth bin, a presence probability (32 quantile bins of cell total) and a value
  distribution (4 coarser depth bins; exact for values 1..63, Elias-gamma escape at 64 and above), Krichevsky-Trofimov
  smoothed. This is the nonparametric version of what `.s1` models with shared tables.
- `cluster`: as `gene_depth`, additionally conditioned on the cell's cluster. Leiden on the training half (at most
  200,000 training cells; the rest are assigned to the nearest centroid). Test cells are assigned to the nearest
  training centroid, and the cluster label is charged at its empirical entropy, so the model cannot gain from a label
  it was not paid for. Sparse (gene, cluster, depth) contexts back off to the `gene_depth` estimate with a pseudo-count
  of beta = 8 (hierarchical KT), so a cluster model is never much worse than its parent.
- `sampling`: the sampling term. For 2,000 random test cells, take pi_c as the cell's own normalised counts shrunk
  toward its cluster profile with weight alpha (fitted on the training half by binomial thinning: hold out a random half
  of each cell's molecules and maximise their likelihood), and compute H(Multinomial(N_c, pi_c)) analytically: entropy of
  the multinomial = log N! terms plus per-gene E[log x!] under Binomial(N_c, pi_gc), summed exactly for means below 30 and
  by the delta method above. This is the part of the code length that any lossless decoder must spend.
Zero-count terms for `poisson` and `nb` are summed per (gene, depth bin) using the bin mean depth; for matrices with
at most 5e7 entries the exact per-cell sum is also computed and the largest relative difference is reported
(`nbBinningErrPctMax`).
The floor reported in Fig 3a and the text is, per matrix, the best (smallest) of the four models. Decomposition for
Fig 3c, per matrix and then median per protocol class: sampling; the cell's own state = best model - sampling;
`.s1` above the floor = `.s1` payload - best model. Separately, information shared with other cells =
`gene_depth` - `cluster` on the sampled cells (`sharedBpn*`). Unit tests for the models are in `tests/test_floor.py`.

### 4.7 `second` (Q9)
Compress each `.s1` file again with zstd -22 --ultra, xz -9e, brotli -q 11, bzip2 -9, and, for S datasets only, a
context-mixing compressor if one is installed (`cmix` or `paq8px`; record which). Report output / input.

### 4.8 `minibatch` (Q3, Q4, Q5 measured)
Loader panel: the five `loader_panel: true` datasets (S to XL, droplet, combinatorial and plate).

*Workload.* A seeded sequence of minibatches of 512 cells. Each minibatch is 8 blocks of 64 contiguous cells drawn
without replacement (the block-shuffling scheme of scDataset, which every loader can serve); 3 seeds. The fully random
(512 independent cells) and sequential workloads are run on `pbmc10k_v3` and `hlca_core` for the supplement. Every
loader returns `torch.sparse_csr_tensor` (float32 values, int64 indices, genes as columns) on the CPU; if a GPU is
present also time `.to("cuda")` and report it separately.

*Storage tiers.* (a) node-local disk, cold page cache; (a') the shared NFS filesystem, read in place; (b) an S3-compatible object store (MinIO on another node of the same
cluster, or on the same node through a loopback) reached through Toxiproxy with a bandwidth toxic of 0.1, 1 and
10 Gbit/s and 1 ms added latency; (c) the same at 1 Gbit/s with 30 ms latency (a WAN-like case). Readers that can read
ranges from S3 do so (`s1`, `zarr`, `soma`, `h5ad` through `fsspec`/`s3fs`); the others download the object first and
then read it locally, which is how they are used in practice. Record the path each loader took.

*Loaders.* `anndata_inmem` (read the whole h5ad, then index), `anndata_backed` (h5ad backed mode), `scdataset` (block
shuffling over backed h5ad), `anndata_zarr_lazy`, `soma_ml` (tiledbsoma-ml), `bpcells` (R; time to a dgCMatrix
minibatch, converted to a tensor via reticulate is not attempted: report it as R-side time), `s1_stream`
(`s1bench/s1_loader.py`, range reads of the chunks covering each block, decode on 4 threads in native order, with a
canonical-order variant). Each loader gets 4 cores; DataLoader `num_workers` is tuned per loader in {0, 2, 4} and the
best is reported (record it).

*Measurements.* (i) time to first minibatch from a cold process, split into open (metadata and index), first read
(transfer and decode, which overlap for third-party loaders), and conversion to a tensor; for `s1_stream` transfer and
decode are also reported separately. (ii) Sustained cells per second over 200 minibatches after 10 warm-up batches.
(iii) Bytes fetched (count at the proxy or through fsspec callbacks). (iv) Peak RSS.

*Model check.* Compare measured sustained delivery with R_eff(B) = (b/B + 1/D)^-1 from the `write` and `decode`
results. Report the ratio; a large gap means latency or request overhead, which is itself a result.

### 4.9 `eco` (Q10)
On `pbmc10k_v3` and `hlca_core`: time and peak memory of `singlet.load("x.s1")` to AnnData, `singlet::load("x.s1",
as = "sce")`, `singlet::load("x.s1", as = "seurat")`, the C++ `pz2::Matrix::open` + `decode`, and the PyTorch loader,
each against the common alternative (`anndata.read_h5ad`, `zellkonverter::readH5AD`, `Seurat::Read10X_h5`). Verify the
objects are equal (counts, gene ids, barcodes). The exact calls shown in Fig 4 must be the ones timed.

### 4.10 `report`
Aggregate the TSVs into `generated/*.tsv` (one per figure panel and table, columns documented in `report.py`) and
`generated/numbers.tex`. Then `make` in this directory builds `main.pdf`. Commit `generated/`, `results/` (TSVs and
`env.json` only, no data) and `datasets.yaml` with the recorded checksums.

## 5. Hardware and software

- One cluster node type for all timings: record CPU model, cores, ISA flags (AVX2 / AVX-512), RAM, local disk model,
  OS, kernel. If AVX-512 is not available, report AVX2 as the headline and say so.
- No other jobs on the node during timings (`--exclusive`). Fix the CPU frequency governor to `performance` if allowed;
  otherwise record it.
- Software: `bench/environment.yml` (conda) for Python and R; build the `singlet` package from this commit with
  `pip install ./` (builds `singlet._pz2`) and `R CMD INSTALL r/`.
- Network tier needs `minio`, `mc` and `toxiproxy-server` / `toxiproxy-cli` binaries (single static binaries; paths in
  `config.yaml`). No root is needed.

## 6. Gaps in the codebase to close first

- **G1, streaming .s1 loader.** `singlet.torch.DataLoader` currently loads a whole matrix before batching. The benchmark
  ships its own streaming loader (`bench/s1bench/s1_loader.py`: one prefix read, then one range read per block,
  decode on N threads straight into the tensor's CSR arrays). Port it into `python/singlet/torch/` so Fig 4 can show
  the package call, then make the benchmark import the package loader and confirm the timings do not change.
- **G2, public writer.** There is no `singlet.write_s1(adata, path)` or R `singlet::write_s1(x, path)`. Add both
  (AnnData / SCE / Seurat / dgCMatrix / scipy sparse in, one sample object out, names inline or from a pack). Until
  then Fig 4 shows these calls in red as planned, and the benchmark uses the internal writer.
- **G3, extra coders for Fig 3d (optional).** The `second` stage runs `cmix` or `paq8px` only if one is on `PATH`.
  If neither is available, the context-mixing row is left out of Fig 3d.
- **G4, ideal-cost hook.** To split ".s1 above the floor" into model coarseness and coder overhead, add a
  `pz2tool cost` subcommand that reports the ideal code length of a member under its own stored tables. Optional.

## 7. Compute estimate

Order of magnitude, to size the allocation. Downloads: about 0.3-0.5 TB (Tahoe plate, MOCA, Replogle, Siletti, ABC
MERFISH dominate). Format outputs: about 3x the canonical data. The XL datasets need 256-512 GB RAM for the
`canon`, `loom`, `h5ad_*` and R stages; everything else fits in 64 GB. Wall time is dominated by `write` for R formats
and loom, and by `minibatch` (5 datasets x 7 loaders x 5 tiers x 3 seeds). Expect 2-4 node-days.

## 8. Done when

- `results/` has a row for every (matrix, format, stage) in scope, with failures recorded as failures.
- `python -m s1bench report` exits 0 and `generated/numbers.tex` defines every macro the manuscript uses (the report
  stage lists missing ones).
- `make` builds `main.pdf` with no red `[??key]` markers (`report` exits 0).
- `datasets.yaml` has `url`, `sha256` and `bytes` for every dataset used.
- The draft text in `sections/results.tex` is checked against the generated numbers. Any claim the data contradict is
  rewritten or removed, and the change is listed in the PR.
