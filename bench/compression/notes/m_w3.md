# m_w3: verify-or-store recipes for the legacy bundle members (python/singlet/v2/recipes*.py)

Owner: W3. Written 2026-10-22. Status: DONE for the brief. Fourteen recipe ids, 51 tests green (29 synthetic + 22 golden on the local corpora, 138 s),
and two end-to-end runs through W2's real migration. Section 9 lists what is not done. This file is the hand-over for W2, the orchestrator and whoever resumes.

## 0. Summary

* Every member the brief asked for is regenerated **byte for byte** on every local sample for which its inputs exist (section 3); the members that stay stored, and why, are in
  section 4. Pass counts: `splice_psi.1pz` 58/58 (53 real + 5 stubs), `cell_calls.tsv` 64/65, all per-cell tables 11/11 samples (77 member-sample pairs), STARsolo members 4/4 samples
  (human ARC, mouse v2, mouse v3 x2), `ambient_profile.tsv`, `gene_expression.tsv`, `splice_events.tsv` 11/11, `feature_vocab.json` 5/5 studies.
* End to end through `python -m singlet.v2.migrate` with `singlet.v2.recipes` loaded: GSE145370/GSM4317429 (gen A) 13.08 MB -> 2.54 MB (5.09x; psi, cell_calls, the mt stub, feature_vocab.json derived);
  gen-B STARsolo sample GSE171161/GSM5220137 74.04 MB -> 2.58 MB (28.7x; 16 derived members, stage 2 converged in one round).
* **Psi (section 5):** the criterion is the decoded canonical matrix, and the container bytes are reproduced as a bonus whenever a `write_1pz` build with the right zstd exists. The two
  evidence points: 53/53 non-stub members reproduce byte for byte with one of two writers (zstd 1.5.5 for the 19 files written with `nearbyint`, 1.5.7 for the 34 written with the truncating cast),
  and none reproduces with the wrong one.
* New facts: the FPKM normaliser of `gene_expression.tsv` is `pileup_stats.json:mapped_reads` (closes the open item of w4_mtx); `ambient_contamination.tsv` rho is `clip(S / N_cell, 0.001, 0.5)` with S from the two
  integers of `ambient_contamination.json`; `doublet_scores.is_doublet` is a model threshold, not a function of the printed score; STARsolo's gene order is the first-appearance order of the GTF, which is
  genomic for human and not for mouse (mouse needs a 172 kB per-build permutation).

## 1. Interface as implemented (the agreed signatures are unchanged)

```python
RECIPES: dict[str, Recipe]                                     # Recipe: .id .doc .regenerate .regenerate_matrix
plan(member_name, legacy_bytes, ctx) -> Plan | None            # Plan = namedtuple(recipe_id, params, needs); unpacks as a 3-tuple
regenerate(recipe_id, params, ctx) -> bytes
regenerate_matrix(recipe_id, params, ctx) -> Matrix            # derived .1pz members only (psi-v1-*, stub-v1)
cells_columns(member_name, legacy_bytes, ctx) -> {name: (dtype, ndarray)}    # dtype names 'u8 u16 u32 i32 f32 f64 bool str'
```

* `plan()` **verifies itself**: a returned plan has been run against `legacy_bytes` (derived `.1pz`: against the decoded matrix and, if a writer exists, the bytes). W2 verifies again (stage 1, stage 2).
  A planner that raises is skipped and its error text kept in `recipes.LAST_ERRORS[member]`; `plan()` never raises. The invariant "a plan implies exact regeneration of exactly those bytes" is tested under
  mutation of the member (`test_plan_implies_exact_regeneration_under_mutation`).
* `NotByteExact` (psi plan whose `params["writer"]` is None) and a writer that is missing on this machine (`LookupError`) both derive from `LookupError`, which is what `rcp_adapter` already treats as "no bytes here".
* **Context.** W2's `context.py` subclasses my `recipes_ctx.Context`; `recipes_ctx.DirContext` (directory / zip sources, auxiliary members through `provide_*`) is the implementation the tests use. Used:
  `matrix`, `member_bytes`, `cells()` (dict of numpy arrays), `vocab_names(kind)`, `genes()` (list of `(gene_id, gene_name)`), `whitelist(name)`, `whitelist_names()`, `features_order()`, `vector(name)`,
  `provide_matrix`, `provide_vector` (inside `plan()` only), `sample_id`, `reference_build`, optional `pack_for` / `packs`. W2's flavours are accepted as they are: `whitelist()` may be a list of str or a numpy `S<L>` array (the array is
  faster; converting W2's 3.7 M-entry list costs 2-3 s per plan); `features_order()` may be None (the recipe then loads its own per-build resource, section 6).
* `needs` strings: `cells:<column>` (the candidate columns W2 keeps), `matrix:<member or aux name>`, `member:<legacy member>` (another member whose bytes are read, restorable first),
  `pack:exon|intron|genes|whitelist|features_order`, `vector:<name>`.
* Auxiliary members: `star_Solo.out/Gene/raw/matrix.mtx#csc` (the decoded CSC of a STARsolo text matrix; naming convention of m_w2.md) and the vector `ambient_k`. Both are registered inside `plan()`.
  Member order is not mine: `filtered/matrix.mtx` is planned before `raw/matrix.mtx`; the filter plan registers the raw CSC itself when it is absent and the raw plan reuses it. Documented failure mode: if the filtered
  plan is dropped later (stage 2), its aux matrix goes with it and the raw plan fails stage 2 as well (both stored). It did not occur on the corpus.
* `cells_columns` names are global within a sample and a name returned by two members carries equal arrays (`cc_pos` from `cell_calls.tsv` and from tables ordered like it; `dbl_*` only from `doublet_scores.tsv`).

## 2. Recipes

| recipe id | legacy member(s) | reads from the context | what the container keeps (besides `called`, `barcode`, `n_umi_exon`) |
|---|---|---|---|
| `psi-v1-rint` `psi-v1-trunc` `psi-v1-stub` | `splice_psi.1pz` | `sj_counts.1pz` matrix (names) | params `{level, chunk_cols, writer, [user_kv]}` = 49 B; stub: nothing |
| `stub-v1` | any `.1pz` that is the constant 151-byte 0 x 0 file (`mt_heteroplasmy.1pz`, stub `sj/vdj/...`) | nothing | `{}` |
| `cellcalls-v1` | `cell_calls.tsv` | cells table | params 73-80 B; `cc_listed`, `cc_pvalue` (emptydrops), `cc_pos` (top_n_fallback) |
| `tsvcols-v1` | `cell_qc_metrics.tsv` `doublet_scores.tsv` `cell_cycle_scores.tsv` `ambient_contamination.tsv` `auto_barcodes.tsv` `donor/donor_assignments.tsv` `mt/mt_summary.tsv` `spatial_coordinates.tsv` `visium_qc.tsv` `nonhost/nonhost_per_cell.tsv` | cells table, exon/intron matrices (qc derivations), `ambient_contamination.json` (rho) | params 33-974 B; typed columns (section 6) |
| `whitelist-sorted-v1` | `star_Solo.out/Gene/raw/barcodes.tsv` | whitelist pack | `{wl}` 25 B |
| `features-v1` | `star_Solo.out/Gene/{raw,filtered}/features.tsv` | gene table; human: exon vocabulary (positions); mouse: permutation resource | `{order}` 16 B; human `{order, contigs}` 397 B |
| `mtx-text-v1` | `star_Solo.out/Gene/raw/matrix.mtx` (and any stored `matrix.mtx`) | aux CSC `<member>#csc` | `{m}` 11 B; the aux pz2 matrix (W1) |
| `mtx-filter-v1` | `star_Solo.out/Gene/filtered/matrix.mtx`, `filtered/barcodes.tsv` | raw aux CSC, raw `barcodes.tsv`, called set | `{what}` 17-19 B |
| `ambient-v1` | `ambient_profile.tsv` | exon vocabulary, vector `ambient_k` | vector u16/u32 per exon row (9 kB for 386 k rows) |
| `geneexpr-v1` | `gene_expression.tsv` | gene table, exon vocabulary, exon + intron matrices | `{total_mapped}` 24 B |
| `splice-events-v1` | `splice_events.tsv` | `sj_counts.1pz` row names | `{}` |
| `vocab-json-v1` | study-level `feature_vocab.json` | gene table of the build | `{build, schema, pack}` |

Definitions (numpy only; pure functions of `(params, context)`; the C++ they port is cited):

* **psi-v1-*** (`recipes_psi.py`). `splice_psi.h`: junction `chrom:start-end` (first `:`, first `-` after it, `std::stoul`, uint32); donor group `chrom:start-D`, acceptor group `chrom:end-A`; groups of >= 2 junctions
  are events; events sorted **bytewise by id**; a junction belongs to the *first* event of that order (so `chr1:100007033-A` beats `chr1:99993741-D`; the in-source comment "donor first" is wrong, w4_bugs item 8); per-cell event sum
  in float32 (exact below 2^24; the plan refuses larger sums); PSI = float32(count) / float32(sum); entries with sum > 0 in sj order. The member stores `nearbyint(PSI)` (`rint`), `static_cast<uint32_t>(PSI)` (`trunc`, the repository
  writer) or the constant stub. Where the data cannot tell the rules apart (every PSI in {0, 1}: L_a, h_split, m_mars) the planner reports `rint`. `numpy == line-by-line Python port of the C++` bit for bit (float32 bit patterns) in the unit test.
* **cellcalls-v1** (`recipes_cells.py`). Header, `log10_umi` (`%.4f` of log10 of the exon column sum, 0.0000 for an empty column) and the method string are constants/functions of the cells table. Rows: `star_solo` = the called cells
  sorted bytewise, sentinel p `0.0000e+00`; `emptydrops` = every tested barcode in column order, TRUE/FALSE, the BH FDR printed `%.4e`; `top_n_fallback` = the top-N exon-UMI barcodes in the order of an *unstable* `std::sort` over all columns,
  and the N-th/(N+1)-th may tie (bn_dropseq, h_celseq2, h_mars, m_mars), so which tied barcodes are called is data (the `called` bitmap) and their row order is stored (`cc_pos`).
* **tsvcols-v1** (`recipes_tables.py`, `recipes_tsv.py`, `recipes_qc.py`). One generic recipe for every per-cell table. `fit_column` finds a typed numpy column + printf format that reproduces the text exactly (int u8/u16/u32/i32 with an
  optional NA token, float f32/f64 with `%.Nf` / `%.Ne` / `%.Ng`, bool, categorical u8, string); params hold header, row rule (`all`, `called`, or an explicit row bitmap; order `sorted`, `colorder`, or `cc`) and per column
  `{source, format}`. Columns that are exact functions of data the container holds are *derived*, not stored: `barcode`, `n_umi_exon` (doublet `total_umis`), constants (`saturation` = NA, all-zero cycle scores on mouse samples,
  `n_genes_used` = 0), and for `cell_qc_metrics.tsv`: `n_umi`, `n_genes`, `pct_mt`, `pct_ribo`, `intronic_pct` (`cell_qc_metrics.h`: gene = row-name prefix; mt = chrom `chrM`/`MT`/`chrMT` or symbol `[Mm][Tt]-`; ribo = `[Rr][Pp][LlSs]`;
  pct = float32(x * (100 / total)); n_genes = genes with a positive exon + intron sum) and `duplication_rate` (`read_stats.h`: float32(max(total - unique, 0)) / float32(total), the clamp is why
  `1 - umi/reads` fails in a few cells); `ambient_contamination.tsv` `rho`. Shared columns: `cell_qc_metrics.doublet_score` is `%.4f` of the `doublet_scores.tsv` score (NaN = "NA" for uncalled cells),
  `cell_cycle_phase` is the cycle phase. `auto_barcodes.tsv` is the barcode list itself (0 B).
* **whitelist / features / mtx** (`recipes_star.py`, `recipes_mtx.py`). raw `barcodes.tsv` = the sorted protocol whitelist (3 contents atlas-wide). `features.tsv` = `gene_id, gene_name, Gene Expression` in STAR's gene order: **human** = sort by
  (contig in order of first appearance, gene start, gene end), reproduced 38,606/38,606, params carry the 39 contig names; **mouse** = a per-build permutation (consistent with STAR numbering genes by first appearance in the GTF, which is not in the repo, so unverified; the mouse order is not position sorted:
  chromosome run order 1,2,X,3,4,...,10,8,14,9,..., and within a chromosome a merge of several position-sorted runs; I found no key). `matrix.mtx` = `%%MatrixMarket matrix coordinate integer general\n%\n<m> <n> <nnz>\n`
  then `"%d %d %d\n"` sorted by (column, row), 1-based; header lines other than the default go into params. Filtered = the raw columns whose barcode is called, renumbered 1..k, kept barcodes in raw order.
* **ambient-v1** (`recipes_extra.py`). `ambient_profile.tsv`: `value_i = (k_i + 0.5) * (1 / sum_j(k_j + 0.5))` printed `%.6g`, names = exon vocabulary. `k` (UMIs of the ambient pool per exon row) is not a function of the `.1pz` (the pool uses barcodes
  the `.1pz` dropped); it is recovered from the text by searching the `T = sum k` that fits the smallest printed value, then checked by re-rendering (the unit test also covers a vector without zeros).
* **geneexpr-v1.** `gene_expression.tsv`: effective_length = sum of exon (end - start) per gene, count = exon + intron UMIs of the gene over all cells, TPM = `rpk * (1e6 / sum rpk)` with the sequential sum, FPKM =
  `count * 1e9 / (len * total_mapped)`, all `%.6f`; `total_mapped` = `pileup_stats.json:mapped_reads` (3,566,396 for GSE171161, 1,364,832 for GSE309702; solved from the FPKM column when the JSON is not in the context).
* **splice-events-v1.** `splice_events.tsv` from the sj row names (same `build_events` as psi; junction order inside an event = ascending row index). **vocab-json-v1.** `json.dumps({schema_version, reference_build, n_genes, genes}, indent=2)`, no trailing newline.

## 3. Pass rates, params size, time

Measured with `python tests/python/test_v2_recipes_corpus.py --full --tsv out.tsv` (275 member rows over 58 samples: GSE145370 x14, `corpus_w4_corpus` x40 incl. the stub sample, `corpus_w4_mtx` x4; the per-cell tables on 11 samples
that have them: the 4 STARsolo samples plus m_dropseq, h_microwell, m_mars, m_celseq2, h_visium, m_visium, m_v3_a). Times are from a laptop shared with other jobs (median / max), `plan` is about the same as `regenerate` plus the verification.

| target member | rows tried | recipes chosen | exact | not planned | params B (med / max) | regenerate s (med / max) |
|---|---|---|---|---|---|---|
| `splice_psi.1pz` | 58 | rint x22, trunc x31, stub x5 | 58 (matrix); 53/53 non-stub also byte-exact with a writer | 0 | 49 / 49 | 5.1 / 22.0 |
| `cell_calls.tsv` | 58 (+7 original-corpus samples in an earlier run of the same code) | star_solo 45+7, emptydrops 4, top_n_fallback 8 | 57 (+7) | 1 (x_stub) | 73 / 80 | 0.0 / 0.5 |
| `mt_heteroplasmy.1pz` | 18 | stub-v1 x15 | 15 | 3 (real matrices) | 2 / 2 | 0.1 |
| `vdj_gene_usage.1pz` | 14 | - | - | 14 (real matrices) | - | - |
| `cell_qc_metrics.tsv` | 11 | tsvcols-v1 | 11 | 0 | 680 / 886 | 0.9 / 4.9 |
| `doublet_scores.tsv` | 11 | tsvcols-v1 | 11 | 0 | 328 / 330 | 0.0 / 0.2 |
| `cell_cycle_scores.tsv` | 11 | tsvcols-v1 | 11 | 0 | 180 / 321 | 0.2 / 2.2 |
| `ambient_contamination.tsv` | 11 | tsvcols-v1 | 11 | 0 | 147 / 149 | 0.0 / 0.1 |
| `auto_barcodes.tsv` | 11 | tsvcols-v1 | 11 | 0 | 33 / 33 | 0.2 / 1.0 |
| `donor/donor_assignments.tsv` | 9 | tsvcols-v1 | 9 | 0 | 613 / 656 | 0.1 / 0.9 |
| `mt/mt_summary.tsv` | 9 | tsvcols-v1 | 9 | 0 | 967 / 974 | 0.1 / 1.3 |
| `spatial_coordinates.tsv`, `visium_qc.tsv` | 2 + 2 | tsvcols-v1 | 4 | 0 | 269 / 597 | 0.1 |
| `star_Solo.out/Gene/raw/barcodes.tsv` | 4 | whitelist-sorted-v1 | 4 | 0 | 25 | 0.9 / 1.9 |
| `star_Solo.out/Gene/filtered/barcodes.tsv` | 4 | mtx-filter-v1 | 4 | 0 | 19 | 2.3 / 4.0 |
| `star_Solo.out/Gene/{raw,filtered}/features.tsv` | 8 | features-v1 (mouse perm x6, human pos x2) | 8 | 0 | 16 (mouse) / 397 (human) | 1.3 / 5.0 |
| `star_Solo.out/Gene/raw/matrix.mtx` | 4 | mtx-text-v1 | 4 | 0 | 11 | 1.6 / 16.7 |
| `star_Solo.out/Gene/filtered/matrix.mtx` | 4 | mtx-filter-v1 | 4 | 0 | 17 | 2.4 / 14.4 |
| `ambient_profile.tsv` | 4 | ambient-v1 | 4 | 0 | 2 (+ vector) | 1.0 / 1.2 |
| `gene_expression.tsv` | 11 | geneexpr-v1 | 11 | 0 | 25 | 6.3 / 7.3 |
| `splice_events.tsv` | 11 | splice-events-v1 | 11 | 0 | 2 | 1.1 / 3.1 |
| `feature_vocab.json` (study) | 5 (GSE145370 + the 4 gen-B studies) | vocab-json-v1 | 5 | 0 | ~100 | < 1 |

Largest case: GSE263963/GSM8208009, `raw/matrix.mtx` 126,023,384 B, 8.1 M entries: parse 15.8 s, render 16.7 s, peak RSS 640 MB (including the input and output buffers). The two largest costs of a plan are the psi writers (up to 17.7 s plan for
10 MB members, two writers tried) and parsing a 62 MB barcode list (8-10 s, mostly W2's list-of-str flavour of `whitelist()`).

What the recipes save (measured, `deflate` = zlib level 6 of the legacy text, "v2" = params + stored columns + vector; the aux pz2 matrix of mtx-text is W1's and not included here):

| sample | derived members | their deflated text | v2 (params + cells columns + vector) |
|---|---|---|---|
| GSE171161/GSM5220137 (mouse v2) | 17 | 12.56 MB (of which the two matrix.mtx 3.78 MB) | 30.0 kB without the matrices (8.79 MB before: 293x) |
| GSE309702/GSM9277757 (mouse v3) | 17 | 20.72 MB (matrices 2.03 MB) | 100 kB (18.7 MB: 186x) |
| GSE311511/GSM9326201 (human ARC) | 17 | 16.67 MB (matrices 5.74 MB) | 661 kB (10.9 MB: 17x; mt_summary 553 kB of real numbers) |
| m_mars / m_celseq2 (MARS-seq2 / CEL-seq2, top_n) | 10 | 1.14 / 1.19 MB | 43 / 52 kB (26x / 23x) |
| h_visium / m_visium | 12 | 1.28 / 1.67 MB | 120 / 170 kB (11x / 10x) |

In the S3 end-to-end object (2,576,299 B) what remains is the count matrices (exon 859 kB, intron 229 kB, sj 290 kB), the raw STARsolo matrix as an aux pz2 matrix (927 kB: the next target, conditional coding against exon is
a v2.1 codec item, w4_mtx section 3), `star_SJ.out.tab` (125 kB, stored), the cells table (31.6 kB), the ambient vector (9.3 kB, against 4.16 MB deflated `ambient_profile.tsv`) and ~10 kB of small JSON/CSV members.

## 4. Members that stay stored, and why

| member | samples | reason |
|---|---|---|
| `cell_calls.tsv` | x_stub (33,091 rows) | its barcodes are not columns of the (empty, 0 x 0) matrices: nothing to anchor a `called` bitmap on |
| `mt_heteroplasmy.1pz` real matrices | GSE171161, GSE263963, GSE311511 (42-561 kB) | independent information (allele counts are not a member); the matrix route (pz2) handles them |
| `vdj_gene_usage.1pz` | all non-stub | independent (E4: 16% of entries equal the exon sum of the same symbol) |
| `star_SJ.out.tab` | all | not derivable. A column-wise typed copy does not beat zstd-19 of the text (123,230 vs 124,689 B on GSE171161), start and end (49 kB each) would need delta coding, not tried |
| `cell_cycle_scores.tsv` scores on human | GSE311511, h_visium, h_microwell | `s_score`, `g2m_score` stored as 2 f32 columns (the `cell_cycle.h` scoring is not ported); constant all-zero on the mouse samples (free) |
| `doublet_score` / `is_doublet`, `rho` p-values | all | model outputs (kNN / GMM / Monte Carlo): stored as typed columns |
| raw `barcodes.tsv` of Visium | m_visium | no whitelist pack for the 4,992 spot list (not fetched, not verified) |
| small JSON/CSV (`summary`, `pileup_stats`, `provenance`, `ambient_contamination.json`, `metrics_summary.csv`, `saturation_curve.tsv`, `read_stats.tsv`, STAR logs) | all | 1-2 kB each; E4 derived 15 of 34 numeric `summary.json` scalars but the three JSONs are 0.012% of a bundle: not worth a recipe |
| loose-directory artefact | `donor__donor_assignments.tsv`, `mt__mt_summary.tsv` in W2's `_open_dir` | the loose-directory loader maps `star_Solo.out__` to `/` but not `donor__`, `mt__`, `nonhost__`; in real bundles the names are `donor/...`, so my profiles match there (only the loose-dir run of GSE171161 stored them: 83 kB) |

## 5. Derived `.1pz` members: bytes or matrix? (evidence and recommendation)

`python tests/python/test_v2_recipes_corpus.py --psi` prints one line per sample. 58 psi members (GSE145370 x14, corpus x40, gen-B x4):

| group | samples | stored values | zstd build needed for the bytes | bytes reproduced |
|---|---|---|---|---|
| GSE145370 (pipeline 0.3.0, git 8e1803b, 2026-06-05) | 14 | `nearbyint(PSI)` | 1.5.5 / 1.5.6 (helper `ccd0ac25`) | 14/14; 0/14 with the 1.5.7 writer |
| corpus, same build | 5 (h_celseq2, h_quartz, h_v2_a, h_v2_b, h_v3_a) | `nearbyint` | 1.5.5 | 5/5; 0/5 with 1.5.7 |
| corpus, later build | 34 (27 + 4 gen-B + 3 where rint == trunc: L_a, h_split, m_mars) | truncation | 1.5.7 (`singlet._pz` of this laptop, `8ebf0e03`) | 34/34; 0/34 with 1.5.5 |
| failed output | 5 (L_b, h_v2_c, h_v3_b, h_v3_c, x_stub) | constant 151-byte stub | none | 5/5 |

No file is reproduced by both writers. The rounding rule and the zstd build change together (two pipeline binaries), which is why a per-sample recipe, not a global rule, is needed. Matrix level: 58/58.

**Recommendation.** (1) The bit-equivalence criterion for every `.1pz` member is equality of the decoded canonical matrix, as the orchestrator proposed. For derived members `regenerate_matrix` is the contract and it is exact on all 58;
it needs no zstd and no compiled code. (2) The container bytes are a bonus: `params["writer"]` is the *calibration id* of the writer that reproduced them (first 8 hex of the SHA-256 of the file a writer produces for a fixed synthetic
matrix; a property of the zstd build, portable between machines), `regenerate` returns the bytes when a writer with that id exists and raises `NotByteExact` / `LookupError` otherwise, and W2 records `bytes_exact` (it does already).
A v2 -> v1 shim therefore emits the original bytes where it can and a decoding-identical file elsewhere. (3) To keep byte exactness available on the migration host (Clipper): `SINGLET_PZW_EXES` takes helpers built with
`recipes_pzw.build_exe_system(out)` against libzstd 1.5.5 and 1.5.7 (the helper source is embedded; it includes `pz_writer.h`); the laptop auto-builds the 1.5.5 one with Rtools. Nothing depends on this for correctness.
(4) The member costs 0 bytes in v2 either way (params 49 B in the manifest); storing the true float PSI is `singlet.psi()`'s job (D8) and `psi_matrix`/`compute_psi` are the shared definition.

## 6. Cells-table columns W2 has to carry (names are global within a sample; dtype width varies by sample: u8/u16/u32 as the data allow)

| column | dtype | contributed by | meaning / used by |
|---|---|---|---|
| `cc_listed` | bool | `cell_calls.tsv` | row membership of an emptydrops table (tested barcodes); `cellcalls-v1`, tables whose rows are the same set |
| `cc_pvalue` | f32 | `cell_calls.tsv` | BH FDR column of emptydrops (4 significant digits, `%.4e`) |
| `cc_pos` | u32 | `cell_calls.tsv`, `doublet_scores.tsv`, `ambient_contamination.tsv` | 1-based row position in `cell_calls.tsv` for top_n_fallback (0 = not listed); sparse |
| `qc_n_reads` | u16/u32 | `cell_qc_metrics.tsv` | reads per cell (the one column of the QC table that is not derivable) |
| `qc_n_umi qc_pct_mt qc_pct_ribo qc_intronic_pct qc_duplication_rate qc_doublet_score` | u16, f32 | `cell_qc_metrics.tsv` | only when the derivation fails or the alias does not hold (Visium) |
| `dbl_doublet_score` | f32 (NaN = not scored) | `doublet_scores.tsv` | also feeds `cell_qc_metrics.doublet_score` |
| `dbl_is_doublet` | bool | `doublet_scores.tsv` | GMM class (not a function of the score) |
| `cyc_phase` | u8 categorical (labels in params) | `cell_cycle_scores.tsv` | also `cell_qc_metrics.cell_cycle_phase`; free when constant |
| `cyc_s_score cyc_g2m_score` | f32 | `cell_cycle_scores.tsv` | only when not constant |
| `amb_rho` | f32 | `ambient_contamination.tsv` | only when `ambient_contamination.json` is absent or the formula fails |
| `don_donor_id don_prob_max don_prob_doublet don_best_singlet don_best_doublet don_cell_status` | u8 / f32 | `donor/donor_assignments.tsv` | constants cost nothing; categorical labels in params |
| `mts_donor_id mts_mt_reads mts_mt_coverage_mean mts_n_loci_with_event mts_n_substitutions mts_n_indels mts_n_frameshifts mts_n_premature_stops mts_pct_heteroplasmic` | u8..u32, f32 | `mt/mt_summary.tsv` | real data (the largest: 553 kB on GSE311511 against 1.25 MB deflated) |
| `vq_total_umis vq_genes_detected vq_mt_pct`, `sp_*`, `nh_*` | | Visium tables | stored columns |
| `<stem>_rows` | bool | any table whose rows are neither all columns nor the called cells | row bitmap |

Vector (not per cell): `ambient_k` (u16 or u32 per exon row; `provide_vector`).

Resources outside the sample object:

* **features_order for GRCm39-2024-A**: `python/singlet/v2/recipes_data/features_order_GRCm39-2024-A.u32.zst` (171,912 B: u32 LE permutation, zstd-19; `features.tsv` row i is vocabulary gene `perm[i]`), byte-identical `features.tsv`
  on the three mouse samples with STARsolo members. It is read at *regeneration* time too, so every reader that wants the legacy `features.tsv` must be able to reach it: ship it as a pack (`pack:features_order` in `needs`;
  `ctx.features_order()` is the hook) or keep it in the wheel (current). `recipes_ctx.save_features_order / load_features_order` and `recipes_star.features_order_from_legacy` build and read it. Human needs nothing.
  No other build has STARsolo members in the corpus (the rat, macaque, zebrafish and barnyard samples are the 34-member layout without `star_Solo.out`).
* **writers** for byte-exact psi (section 5): `recipes_pzw.py`; `SINGLET_PZW_EXES` (os.pathsep list of helper executables), `SINGLET_PZW_AUTOBUILD=0` disables the Rtools auto-build, `SINGLET_V2_RECIPES_CACHE` is the build directory
  (not `SINGLET_V2_CACHE`, which W2's pack store uses).

## 7. Tests

* `tests/python/test_v2_recipes.py` (29 tests, about 30 s, synthetic only): mtx text round trips and rejects; tsv fitting; a line-by-line Python port of `compute_psi` against the numpy version (bit-identical float32 incl. names that cross a power of ten,
  unparsable names, duplicate names); event rules; every recipe on small synthetic members; the aux-matrix order independence; psi without any writer (matrix level); the mutation invariant.
* `tests/python/test_v2_recipes_corpus.py` (22 golden tests by default, about 2.5 min; `SINGLET_V2_FULL=1` runs every sample, about an hour): gen A bundle through `ZipSource`, `corpus_w4_corpus`, `corpus_w4_mtx`, `corpus_m_w3`. They skip when a corpus is absent.
  Script mode: `--full --tsv out.tsv` (the table above), `--psi` (writer table), `--fetch [dir ...]` (refetch `corpus_m_w3`).
* Integration: `PYTHONPATH=python python -m singlet.v2.migrate --bundle bench/compression/corpus/bundles/GSE145370.singlet --out DIR --samples GSM4317429`; the gen-B run needs a loose directory that merges `corpus_w4_mtx/<id>` and `corpus_m_w3/<id>`
  (what I did; W2's `test_corpus_mtx_gen_b_members` uses `--recipes none`).

## 8. Data used

* `corpus/bundles/GSE145370.singlet` (14 samples), `corpus_w4_corpus/` (40 samples), `corpus_w4_mtx/` (4 samples), the 7 original corpus samples (cell_calls only).
* `corpus_m_w3/` (new, git-ignored like the other corpora, 151 MB, 11 directories): members the other corpora lack, fetched by HTTP range request from the public bundles on data.singlet.bio (about 190 MB transferred over ~25 bundle opens, each a tail read plus member ranges; no other service was
  contacted): for the 4 STARsolo samples `auto_barcodes`, `cell_cycle_scores`, `donor/`, `mt/`, `splice_events`, `sj_counts`, `splice_psi`, `pileup_stats`, ...; for m_dropseq, h_microwell, m_mars, m_celseq2, h_visium, m_visium, m_v3_a the per-cell tables,
  `gene_expression`, `splice_events`, `pileup_stats`, `ambient_contamination.json`. The member list is `FETCH_MEMBERS` in the golden test.
* Packs: W2's `python/singlet/v2/packs/` through the real migration; in the tests the vocabulary, gene table and whitelists come from the sample's own legacy members and the package whitelist text files (same values by construction; the real runs
  confirm it).

## 9. Not done / limits / hand-over notes

* No recipe for the small JSON members (section 4). No port of `cell_cycle.h` scoring, the doublet model or EmptyDrops.
* STARsolo members were available for 2 builds only (GRCh38, GRCm39) and 4 samples (10x v2, v3 x2, ARC); other builds, a Visium whitelist pack, and non-10x chemistries with `star_Solo.out` are untested (verify-or-store keeps them safe).
* Barnyard row names (species tag inside the name) were not exercised by the QC / gene-expression derivations; a failure there only means the columns are stored.
* Byte-exact psi was verified with two writers (zstd 1.5.5 helper, 1.5.7 `singlet._pz`); 1.5.6 is assumed identical to 1.5.5 (Track A saw identical files), not re-tested. On Linux the helper must be built (`build_exe_system`).
* `plan()` time is dominated by verification (a psi plan on a 10 MB member is up to ~18 s with two writers, a 126 MB mtx 35-55 s); it could be halved by caching the verified output on the context instead of regenerating in W2's stage 1.
* For W2: (a) the loose-directory loader should also map `donor__`, `mt__`, `nonhost__` (section 4); (b) `ctx.whitelist()` as a numpy `S<L>` array saves 2-3 s per STARsolo sample; (c) `features_order` as a pack, or tell me the id and I read it from the pack store;
  (d) the filtered / raw aux-matrix coupling (section 1); (e) packaging: `pyproject.toml` package-data lists `v2/packs/*.pack` but not `v2/recipes_data/*.zst`, so a wheel would lack the mouse features permutation (my path list does not include `pyproject.toml`); (f) `needs` of psi plans list `matrix:sj_counts.1pz` and of tsvcols plans the `cells:*` columns, which is what `_assemble` uses to keep candidate columns.
* How to resume: `python -m pytest tests/python/test_v2_recipes.py tests/python/test_v2_recipes_corpus.py`; add a recipe by writing `plan_x / regenerate_x` in a `recipes_*.py`, registering it in `recipes.py` (`_reg`, `_PLANNERS`) and adding a unit test with the mutation invariant;
  a changed behaviour is a new versioned id, never an edit of an old one.
