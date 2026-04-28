# Changelog

All notable changes to `singlify` (Python) are documented here. The
format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the package uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **`singlify.io.aggregate_features_to_gene(feature_mat, feature_rownames)`**
  — collapses a per-feature (per-exon or per-intron) sparse matrix to
  a per-gene matrix by summing all features whose Ensembl-style
  rowname starts with the same gene-ID prefix
  (``ENSG..._GENE_chr:start-end``). Verified bit-exactly on real
  pipeline outputs: ``aggregate(exon_counts) == spliced`` per gene.
- **AnnData adapter self-heals on Tier 2 drops**:
  :func:`singlify.interop.anndata.read_anndata` now detects a missing
  ``spliced.1pz`` / ``unspliced.1pz`` and reconstructs the
  corresponding layer on the fly via
  :func:`aggregate_features_to_gene`. The derivation is recorded in
  ``adata.uns["singlify_derived_layers"]`` for audit. Tested with a
  synthetic "dropped" fixture.
- ``tests/test_pz_io.py::test_aggregate_exon_counts_equals_spliced``
  — enshrines the per-gene aggregation invariant as a regression
  canary.

### Storage policy

The pipeline's NFS copy step now drops three verified-redundant
files: ``gene_counts.1pz``, ``ambiguous.1pz``, ``splice_psi.1pz``.
The Tier 2 drop (``spliced.1pz`` + ``unspliced.1pz``, ~880 GB at
catalog scale) is now **safe** thanks to the on-read aggregator,
pending the user's go-ahead.

## [0.2.0] — 2026-04-13

The pipeline-reader surface lands. The package is now two-in-one:
the pre-existing live-BAM pileup engine plus a new pipeline output
reader that decodes `.1pz` files + per-cell TSV sidecars into
scipy / AnnData / scanpy objects.

### Added

- **`singlify.io`** submodule — high-level pipeline output readers:
  - `read_matrix(path)` → `(scipy.sparse.csc_matrix, PZMetadata)`
  - `read_dir(path, include, exclude)` → `PipelineDirectory` dataclass
  - `read_metadata_from_file(path)` — GEO context only
  - `open_pz(path, verify_crc=True)` — pure-Python PZFile context manager
  - `PZHeader`, `PZFooter`, `PZMetadata`, `PZFile`, `PZError` dataclasses
  - `VTCode`, `Flags`, `FeatureFlags` enums for file introspection
- **`singlify.interop.anndata.read_anndata(dir)`** — one-call directory →
  fully-populated `AnnData` with:
  - `adata.X` = primary per-gene matrix
  - `adata.layers["unspliced"]`, `["ambiguous"]`, `["gene_counts_em"]`
  - `adata.obsm["exon_counts"]`, `["intron_counts"]`, `["sj_counts"]`,
    `["splice_psi"]`, `["mt_heteroplasmy"]`, `["vdj_gene_usage"]`
  - `adata.obs` auto-populated from `cell_qc_metrics.tsv`,
    `cell_cycle_scores.tsv`, `doublet_scores.tsv`, `read_stats.tsv`,
    `ambient_contamination.tsv`
  - `adata.uns["singlify"]` = full GEO context dict
- **`singlify.interop.scanpy`** convenience helpers — `read()`,
  `ensure_scvelo_layers()`, `quick_qc()`, `normalize_log()`.
- **`singlify._pz_io`** pybind11 binding — ~130-LOC wrapper around the
  header-only C++ reader at
  `../include/singlet-pileup/pz_reader.h`.
- **Command-line interface** — `singlify convert`, `singlify info`,
  `singlify verify`. Wired via `[project.scripts]` so `pip install
  singlify` puts `singlify` on PATH.
- **Sphinx documentation** — `quickstart.rst`, `io.rst`, `interop.rst`,
  `format.rst`, `architecture.rst`, `api.rst` with autodoc +
  autosummary + napoleon + myst.
- **pytest suite** — 11 tests in `tests/test_pz_io.py` covering import
  smoke, shape/dtype invariants, scipy CSC drop-in, metadata fields,
  velocity-trio reconstruction, and three error paths. Test fixtures
  resolved via env var or NFS fallback (CI-friendly).
- **GitHub Actions workflows** under `.github/workflows/`:
  `python-ci.yml` (pytest matrix), `python-docs.yml` (Sphinx build
  with `-W`), `python-build-check.yml` (`build` + `twine check`).
- **`zstandard`** as a required dependency (was implicit via pybind11
  extension before).

### Changed

- Version bumped from 0.1.0 → 0.2.0.
- Top-level `singlify/__init__.py` now uses a lazy `__getattr__` for
  `_core` and `io`. You can `import singlify.io` without having the
  pybind11 `_core` extension compiled — useful for read-only
  environments and for documentation builds.
- `pyproject.toml`: added `zstandard` to required deps, added
  `scanpy` / `dev` extras, added `[project.scripts]` entry, added
  `[tool.scikit-build] sdist.exclude` to keep the source
  distribution under 50 KB.
- `CMakeLists.txt`: `PILEUP_DIR` switched from the stale
  `../../singlet-pileup/include` to the active `../include` tree;
  second `pybind11_add_module(_pz_io)` added.

### Fixed

- Corrected the `VTCode` enum in `singlify.io._format` to match the
  writer's actual mapping (`1=uint8`, `2=uint16`, `3=uint32`). The
  prose `1PZ_FORMAT_SPEC.md` documented a different, never-shipped
  layout; the writer source is authoritative.
- Metadata block is a **TLV** (tag-length-value) stream, not a JSON
  blob — the Python `_metadata.py` now parses it correctly against
  the writer's `push_strings_tlv` / `push_kv_tlv` layout.

## [0.1.0] — earlier

Initial release with the live BAM pileup engine (`singlify.pileup`,
`PileupResult`, `PileupStats`) via `_core.cpp`. No pipeline-output
reader.
