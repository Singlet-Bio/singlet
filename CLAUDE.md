# singlet-bio: notes for Claude sessions

- **Cell data architecture (`docs/CELL-ARCHITECTURE.md`) is binding** for any change that stores, indexes or serves
  per-sample or per-cell data (the canonical copy is in zdebruine/singlet-website; change that one first, then this copy).
  - Raw counts (exon, intron, junctions, heteroplasmy, VDJ) stay in immutable per-sample `.s1` objects in R2
    (`docs/format-s1.md`), read by byte range through `data.singlet.bio/s1`; never copied into a database.
  - D1 holds sample-level metadata and per-sample summaries of cell properties (`sample_term`, `sample_property`,
    `cell_feature`, `sample_cell_summary`), never one row per cell.
  - Per-cell labels and feature scores live in R2 columnar objects (cell-index shards, per-sample label and feature
    sidecars) joined by `(sample ordinal, col)`, where `col` is the column in the sample's store `cells` table.
  - Queries go tier 1 (D1, milliseconds) -> tier 2 (bounded Worker reads, otherwise a plan of keys and byte ranges) ->
    client gather. Custom scripts run client-side or on Clipper/Anvil, never in a Worker; results worth keeping become
    registered `cell_feature`s, computed in batch and versioned.
  - Ontology ids are shared between sample and cell level: UBERON, CL, MONDO (+ PATO:0000461 normal), PATO sex,
    HANCESTRO (children of HANCESTRO:0004), HsapDv/MmusDv.
  - Genotypes are never stored in public objects or served; inferred ancestry and chromosomal sex labels may be.
  - A change that does not fit the architecture updates `docs/CELL-ARCHITECTURE.md` first.
