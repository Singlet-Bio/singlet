# Benchmark notes

How the benchmark behind *Fewer Bits, Faster Cells* keeps its comparisons fair, and where each setting lives. The
manuscript's methods (main text and Supplementary Methods) are the full description; this file is a guide for anyone
reading or rerunning the pipeline. All settings are in `bench/config.yaml`, and every number in the paper is written by
`python -m s1bench report`.

## Every format does the same work

- **Same matrix, same names.** Every format stores the same canonical matrix: genes by cells, integer counts, gene
  identifiers and cell barcodes. File sizes always include the names.
- **Lossless or it doesn't count.** Every write is followed by a read and an exact comparison with the canonical
  matrix. A format that fails the round trip on a dataset is reported as failed, not dropped.
- **Only the library's own read is timed.** Each format is read by its own library into its native object (a SciPy
  matrix, an AnnData object, an R matrix). Converting to a common form and checking it happen afterwards, untimed.
- **Same gene order.** `.s1` stores genes from most to least often detected. Its headline rows are timed in the source
  gene order, like every other format; reading in the stored order is a separate row.
- **Same cores.** Readers are pinned to the same cores with `taskset`, four for the headline numbers, whatever they do
  with them (threads or worker processes). Thread sweeps are in the supplement.
- **Like-for-like baselines.** In R, `singlet::load(as = "seurat")` is compared with
  `CreateSeuratObject(Read10X_h5(...))`, since `Read10X_h5` alone returns a bare matrix.

## Reading cold and warm

- Before timing, each process writes and reads a 60 × 40 matrix in the same format, untimed, so library loading is not
  counted.
- "Cold" means the file was evicted from the operating system's memory before each read. The disk controller's own
  cache cannot be cleared from a job; the methods say how this was checked. Across 1,238 configurations, the first cold
  read took a median 1.00 times the median of three (90th percentile 1.08).
- Raw times and repeat counts are kept with every result row. The pipeline records which decoder kernel actually ran,
  and keeps only rows where the requested kernel ran. AVX-512 and ARM processors were not measured.

## Ratios and medians

- Every ratio in the paper is a median of per-matrix ratios, not a ratio of medians.
- "Smaller than the best alternative" is paired by matrix: on each matrix, `.s1` against the smallest other file.

## The information floor

The floor is a reference code length, not a bound: the best of four count models, fitted with cross-validation and not
charged for their own parameters. The share of bits due to molecular sampling is an estimate for UMI matrices only.
Stronger models could lower the floor; the discussion covers this.

## Minibatch loaders

- **One workload for all.** 512-cell minibatches, timed until the cells are decoded into PyTorch sparse tensors.
  Competing loaders are set to the same workload: scDataset fetches one batch per read (`fetch_factor: 1`), and its
  default setting (`fetch_factor` 16) is a separate row. TileDB-SOMA ML uses its own prefetching reader with as many I/O
  and compute threads as cores.
- **Sharded Zarr v3 control.** CSR arrays in chunks of 49,152 elements (one `.s1` block), 64 chunks per shard, Blosc
  zstd with bit shuffling. It separates what a generic codec achieves at the same granularity from what `.s1`'s codec
  adds.
- **DataLoader workers.** Loaders were also run with PyTorch DataLoader workers. `S1Cells` gains nothing from them,
  because its decoder already uses every core.

## Network tests

- **Emulated object store.** Bandwidth is limited by one token bucket shared by all connections, and latency is added
  separately. The rate each tier actually delivered is recorded: 0.10, 1.00 and 9.13 Gbit/s for the 0.1, 1 and
  10 Gbit/s tiers.
- **Latency and requests in flight.** Round-trip times of 0–200 ms at 1 Gbit/s, and 1–64 requests in flight with 0–8
  minibatches requested ahead. The `singlet` defaults (32 requests in flight, four minibatches ahead) were chosen from
  this sweep.
- **A real object store.** Cloudflare R2, served by range through a Worker, read over the public internet from a
  cluster node: about 70 ms per request. Costs per epoch use list prices for transfer and requests.

## Block size and random access

Bytes, requests and over-read per random minibatch are computed from each format's own index. `.s1`'s default block of
49,152 non-zeros was kept after a sweep from 4,096 to 393,216 on three matrices: on the Human Lung Cell Atlas core,
the smallest blocks remove over-read for an 8% larger file, and the largest double it for a 0.5% smaller one.

## Writing

Write time and peak memory were measured for every format on three matrices (PBMC 10k, Tabula Muris FACS and the HLCA
core). Writes are not part of the read timings.

## Data and records

- Every downloaded file's SHA-256 and size are in `datasets.yaml`; `fetch` verifies each one.
- Each dataset's run keeps its own host and package record in `results/env/`.
- Settings that differ for the largest matrices (`decode.big_nnz`, `decode.big_formats`) are in `config.yaml` and are
  applied by the pipeline, not by hand.
