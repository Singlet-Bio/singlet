# singlet (R)

An R client for the [Singlet](https://singlet.bio) single-cell RNA-seq atlas —
search and download processed public studies as a `SingleCellExperiment` or
`Seurat` object — plus an R package for reading `.1pz` sparse-matrix files and
working with singlet pipeline outputs directly.

All decompression and format parsing happens in a **header-only C++ reader**
(namespace `singlet::pz`) that is byte-identical to the Python package. R is a
thin Rcpp binding.

## Install

```r
install.packages(c("remotes", "BiocManager"))
BiocManager::install(c("SingleCellExperiment", "SummarizedExperiment", "S4Vectors"))
remotes::install_github("Singlet-Bio/singlet", subdir = "r")
```

Building from source needs a C++17 compiler and `libzstd` (>= 1.4) at link
time: `apt install libzstd-dev` (Debian/Ubuntu), `yum install libzstd-devel`
(RHEL/Fedora) or `brew install zstd` (macOS).

`singlet_load()` returns a `SingleCellExperiment` by default, which is why the
Bioconductor packages above are part of the install. For Seurat output
(`as = "seurat"`) also run `install.packages("Seurat")`.

## Quickstart: browse and load the atlas

```r
library(singlet)

# Load one study (downloaded from data.singlet.bio and cached) -> SingleCellExperiment
sce <- singlet_load("GSE138867")
table(sce$gsm_id)

# One sample: resolves to its study and keeps only that sample's cells
sce <- singlet_load("GSM4120733")

# Several studies, combined on their shared gene axis
sce <- singlet_load(c("GSE138867", "GSE146974"))

# ...or as a Seurat object
obj <- singlet_load("GSE138867", as = "seurat")

# Search the atlas with natural language -> GSE accessions
hits <- singlet_find("T cells from pediatric AML")
sce <- singlet_load(hits[1:3])

# Search and load in one step
sce <- singlet_find_load("human pancreas islet cells", limit = 3)

# A local .singlet bundle works the same way as an accession
sce <- singlet_load("/path/to/GSE138867.singlet")
```

`load()`, `find()`, `find_load()` and `download()` are the same functions
under shorter names. They mask `base::load()` and `utils::find()` once the
package is attached, which is why the examples use the `singlet_` names.

Samples with no usable cells (for example an empty count matrix left by a
failed pipeline run) are skipped with a warning rather than failing the whole
study. The skipped samples, and why, are in
`S4Vectors::metadata(sce)$skipped_samples`.

## Beyond gene counts: other modalities

A `.singlet` bundle carries more than the gene-level counts `singlet_load()`
assembles: splice junctions and PSI, mitochondrial heteroplasmy, donor
assignments, non-host reads, per-cell QC tables and the raw exon/intron
matrices. Download the bundle and ask what a sample has before reading it:

```r
path <- singlet_download("GSE128639")   # CITE-seq bone marrow
singlet_modalities(path)                # what the first sample carries

gsm <- "GSM3681518"
if (singlet_has(path, "mt_heteroplasmy", gsm)) {
  het <- singlet_read(path, gsm, "mt_heteroplasmy")   # variants x cells
}
if (singlet_has(path, "cell_qc", gsm)) {
  qc <- singlet_read(path, gsm, "cell_qc")            # data.frame
}

# Raw counts for one sample with spliced/unspliced assays
if (singlet_has(path, "exon_counts", gsm)) {
  raw <- singlet_raw_counts(path, gsm)
}
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SINGLET_DATA_BASE` | `https://data.singlet.bio` | Host serving bundles at `<base>/data/<GSE>/<GSE>.singlet` |
| `SINGLET_API_BASE` | `https://singlet.bio/api` | Search and GSM-to-GSE lookups |
| `SINGLET_CACHE_DIR` | `tools::R_user_dir("singlet", "cache")` | Where downloaded bundles are kept |
| `SINGLET_API_KEY` | unset | Raises the natural-language search limit (see `set_api_key()`) |

## Reading pipeline output directly

For `.1pz` files and pipeline output directories rather than atlas
accessions. Every `samples/<GSM>/` folder inside a bundle is such a directory:

```r
library(singlet)

path <- singlet_download("GSE138867")
exdir <- tempfile()
utils::unzip(path, exdir = exdir)
sample_dir <- file.path(exdir, "samples", "GSM4120733")

# Low-level: one file -> dgCMatrix
mat <- read_1pz(file.path(sample_dir, "exon_counts.1pz"))
dim(mat)
attr(mat, "user_kv")[["gsm_id"]]     # "GSM4120733"

# Mid-level: a whole sample directory -> named list of matrices
dd <- read_singlet_dir(sample_dir)
print(dd)
```

`as_sce()` and `as_seurat()` convert a directory written by running the
pipeline yourself (they need its per-gene `spliced.1pz` or
`gene_counts.1pz`, which atlas bundles do not ship; use `singlet_load()` for
those).

## Embedded Metadata

Every `.1pz` file carries the GEO context from the pipeline:

- `gsm_id`, `gse_id`, `srr_ids`
- `organism`, `protocol`
- `singlet_version`, `pipeline_date`
- `read_count`

## License

MIT
