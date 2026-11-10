# singlet (R)

An R client for the [Singlet](https://singlet.bio) single-cell RNA-seq atlas: search the atlas for studies or cells and load
them as a `SingleCellExperiment` or `Seurat` object. Each public GEO study is published as one `.s1` object per sample plus a
manifest; the package fetches only the parts it needs by HTTP range requests and caches them.

Count matrices are decoded by a header-only C++ reader shared with the Singlet Python package
([`pip install singlet`](https://pypi.org/project/singlet/)); R is a thin Rcpp binding. Zstandard is bundled, so installing
needs only a C++17 compiler.

## Install

```r
install.packages(c("remotes", "BiocManager"))
BiocManager::install(c("SingleCellExperiment", "SummarizedExperiment", "S4Vectors"))
remotes::install_github("Singlet-Bio/singlet", subdir = "r")
```

Once the package is on CRAN, `install.packages("singlet")` replaces the `remotes` line.

`singlet_load()` returns a `SingleCellExperiment` by default, which is why the Bioconductor packages above are part of the
install. For Seurat output (`as = "seurat"`) also run `install.packages("Seurat")`.

## Quickstart: load studies

```r
library(singlet)

# One study -> SingleCellExperiment
sce <- singlet_load("GSE138867")
table(sce$gsm_id)

# One sample: resolves to its study and keeps only that sample's cells
sce <- singlet_load("GSM4120733")

# Several studies, combined on their shared gene axis
sce <- singlet_load(c("GSE138867", "GSE146974"))

# Per-cell labels are in colData (?singlet_labels); GEO sample fields are geo_*
table(sce$cell_type, sce$sex_chrom)
heart <- singlet_cells("GSE286650", cell_type == "cardiac muscle cell" & called_ed)

# ...or as a Seurat object
obj <- singlet_load("GSE138867", as = "seurat")

# Search studies with natural language -> GSE accessions
hits <- singlet_find("T cells from pediatric AML")
sce <- singlet_load(hits[1:3])
```

`load()`, `find()`, `find_load()` and `download()` are the same functions under shorter names. They mask `base::load()` and
`utils::find()` once the package is attached, which is why the examples use the `singlet_` names.

A study that is not in the atlas (an unknown accession, or one Research Alpha withholds) stops with a message pointing to
<https://singlet.bio/known-issues>.

## Search and fetch cells across the atlas

```r
# Count and locate cells by label and sample metadata (ontology terms match their descendants)
hits <- singlet_search_cells(list(cell_type = "T cell", sex = "female", tissue = "blood"))
hits                       # cells, samples and studies; hits$studies, hits$samples, hits$facets
hits$unmapped              # how much of the atlas each sample-level filter cannot see

# Fetch the matching cells (only their matrix chunks are downloaded)
sce <- singlet_fetch_cells(list(cell_type = "B cell", organism = "Homo sapiens", tissue = "blood"), max_cells = 2000)

# Cells listed elsewhere (MCP get_cells_page / find_cells, Python atlas): (GSM, 0-based columns)
sce <- singlet_gather(list(list(gsm = "GSM4736413", columns = c(115L, 175L, 260L))))

singlet_cell_fields()      # every searchable field and its coverage
```

Cell types below lineage level are previews (per-study accuracy 0.84 compartment, 0.75 lineage, 0.43 cell type, 0.32
subtype). Genetic ancestry covers under 2% of cells, and disease is recorded for a minority of samples, so combined filters
can match few cells.

## Other modalities

`singlet_study_modalities("GSE144434")` (CITE-seq of human PBMC) lists what a study's samples carry beyond gene counts (spliced and unspliced layers,
splice junctions and PSI, V(D)J usage), and `singlet_load(..., modalities = )` fetches them. Research Alpha withholds
per-cell mitochondrial DNA, doublet scores and per-donor germline data from every published object.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `SINGLET_DATA_BASE` | `https://data.singlet.bio` | Host serving the `.s1` objects at `<base>/s1/...` |
| `SINGLET_API_BASE` | `https://singlet.bio/api` | Search, cell search and GSM-to-GSE lookups |
| `SINGLET_CACHE_DIR` | `tools::R_user_dir("singlet", "cache")` | Where fetched objects are kept |
| `SINGLET_API_KEY` | unset | Raises the natural-language search limit (see `set_api_key()`) |

## Reading pipeline output directly

For `.1pz` files and pipeline output directories you produced yourself:

```r
mat <- read_1pz("samples/GSM4120733/exon_counts.1pz")   # one file -> dgCMatrix
attr(mat, "user_kv")[["gsm_id"]]
dd <- read_singlet_dir("samples/GSM4120733")            # a sample directory -> named list of matrices
```

`as_sce()` and `as_seurat()` convert such a directory (they need its per-gene `spliced.1pz` or `gene_counts.1pz`).

## License

MIT. The bundled Zstandard library is BSD-licensed (see `inst/COPYRIGHTS`). Atlas data is CC0.
