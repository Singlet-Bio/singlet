# Your own counts as `.s1`

`write_s1` stores any integer count matrix, with its gene ids and cell barcodes, as a `.s1` sample object
([format-s1.md](format-s1.md)): one pz2 matrix member named `counts`, coded by the same compiled codec the atlas uses. You
can then:

- read it back exactly (`read_s1`, or `load`);
- serve it from any static host that answers HTTP Range requests;
- stream minibatches of cells from it into PyTorch with `singlet.torch.S1Cells`. Only the chunks that hold those cells
  are read, from a local disk or over HTTP.

Python and R write the same object for the same matrix (identical bytes when both are built against the same zstd
library; see below).

## Python

```python
import singlet

singlet.write_s1(adata, "pbmc.s1")                         # AnnData: X (or layer="counts"), var_names, obs_names
singlet.write_s1(x, "pbmc.s1", genes=gene_ids, barcodes=barcodes)   # scipy sparse or dense, cells x genes
adata = singlet.load("pbmc.s1")                            # = singlet.read_s1("pbmc.s1"): AnnData, counts in X
```

```text
singlet.write_s1(x, path, *, genes=None, barcodes=None, layer=None, sample_id="sample", member="counts",
                 gene_pack=None, threads=4) -> pathlib.Path
singlet.read_s1(path, *, member="counts", genes=None, dtype=numpy.float32) -> anndata.AnnData
```

- **Input.** `x` is AnnData, a scipy sparse matrix, or a dense array, oriented **cells x genes** as in AnnData. Values
  must be non-negative integers below 2^32. Integer-valued floats are accepted; anything else raises `ValueError`.
  Explicit zeros are dropped.
- **Writing.** The file is written under a temporary name and renamed into place. `threads` speeds up encoding and does
  not change the bytes.
- **Reading.** `read_s1` returns `X` as CSR `float32`, which holds counts up to 2^24 exactly. Pass `dtype=numpy.uint32`
  or `float64` for deeper data.
- **Gene packs.** `gene_pack="<16 hex>"` (or 8 bytes) stores the gene ids as a reference to a vocabulary pack instead of
  inline. The reader then needs the pack, or `read_s1(path, genes=...)`.
- **Loading.** `singlet.load(path)` (and `singlet.s1.client.load`) reads any `.s1` file that has no atlas members and
  one count matrix: the member `counts`, or else its only matrix member. Atlas sample objects keep their own reader.

## R

```r
library(singlet)
write_s1(m, "pbmc.s1")                 # dgCMatrix or matrix (genes x cells, with dimnames), SingleCellExperiment, Seurat
m   <- read_s1("pbmc.s1")              # dgCMatrix
sce <- load("pbmc.s1")                 # SingleCellExperiment (as = "seurat" for Seurat)
```

```text
write_s1(x, path, sample_id = "sample", assay = "counts", member = "counts", gene_pack = NULL, threads = 4L)
read_s1(path, as = c("matrix", "sce", "seurat"), member = "counts", genes = NULL, threads = 4L)
```

- **Orientation.** R matrices are **genes x cells**, the Bioconductor orientation.
- **Inputs.** A `SingleCellExperiment` gives `assay(x, assay)`, and a Seurat object gives layer `assay` of its default
  assay.
- **Encoder.** The R package calls the header-only C++ encoder (`inst/include/singlet/pz2/`) directly, with the options
  the Python binding uses. `r/tests/testthat/test-s1-write.R` and `tests/python/test_s1_write.py` both check one shared
  fixture object: container, header, canonical digest and decoded counts and names.
- **Bytes across builds.** The names and probability tables inside a member are zstd level 19, and zstd releases can
  compress the same input to different bytes. Two builds that link the same zstd (Python and R from one conda
  environment, say) write identical files. Builds against different zstd releases write files that differ in those
  sections and decode to the same matrix.

## Streaming into PyTorch

```python
from torch.utils.data import DataLoader
from singlet.torch import S1Cells, S1Stream

ds = S1Cells("https://example.org/pbmc.s1", batch_size=512, block_size=64, threads=4)   # or a local path
for epoch in range(10):
    ds.set_epoch(epoch)
    for batch in DataLoader(ds, batch_size=None, num_workers=2):
        x = batch["X"]          # torch.sparse_csr_tensor, cells x genes, float32
        idx = batch["index"]    # the cells' column numbers, to look up labels
```

```text
S1Cells(source, *, batch_size=512, block_size=64, shuffle=True, seed=0, drop_last=False, member="counts",
        threads=4, order="native", io_threads="auto", prefetch=4)
S1Stream(source, *, member="counts", threads=4, order="native", size=None, prefix_hint=262144, io_threads="auto")
    .batch(blocks) -> torch.sparse_csr_tensor        blocks: [(c0, c1), ...] cell ranges, concatenated in order
    .cells(c0, c1) -> scipy.sparse.csr_matrix
    .gene_names, .canonical_gene_names, .gene_order, .barcodes, .n_cells, .n_genes, .bytes_fetched, .requests
block_batches(n_cells, batch_size=512, block_size=64, seed=0, n_batches=None, drop_last=False)
```

**Batching.** A batch is `batch_size // block_size` blocks of `block_size` contiguous cells, the block shuffling of
scDataset:

- Each epoch shuffles the blocks with `(seed, epoch)` and covers every cell once.
- With several DataLoader workers, each worker opens the file itself and takes every n-th batch. This holds for forked
  workers too: a worker does not use the connections of the process it was forked from.
- `block_batches(..., n_batches=k)` returns a fixed number of batches of whole blocks. This is the workload of the `.s1`
  benchmark.

**Reading.** `S1Stream` opens the object with one prefix read (header, member table, the matrix's chunk index and
tables). It then makes one range request per run of touching blocks, decodes on `threads` threads, and writes straight
into the tensor's CSR arrays.

- Range requests are sent together on `io_threads` threads. `"auto"` means 32 for an HTTP URL, where every request
  waits a network round trip, and 1 for a local file. `io_threads=1` sends them one at a time.
- A link is full only with about bandwidth x round-trip time in flight: 12.5 MB at 1 Gbit/s and 100 ms, several
  minibatches' requests. Raise `io_threads` and `prefetch` together on a slower store.
- These threads mostly wait on the network; decoding uses `threads`.
- `S1Cells` sends the requests of the next `prefetch` batches while one decodes (default 4, `prefetch=0` for none), so
  fetching overlaps with decoding and with the work on the current batch. DataLoader workers each do this on their own.

**Gene order.** `order="native"` keeps the stored gene order: genes sorted by how many cells express them, which the
codec uses.

- `gene_names` lists genes in the column order of the batches.
- `gene_order` maps stored rows to source rows, so a model can permute its input layer once instead of permuting every
  batch.
- `order="canonical"` returns the source order at a small decode cost.

An atlas sample object can be streamed with `member="exon_counts.1pz"`; its gene ids live in a vocabulary pack.
`singlet.torch.AtlasCells` streams whole atlas cohorts with labels and study-held-out splits ([ml-ready.md](ml-ready.md)).
