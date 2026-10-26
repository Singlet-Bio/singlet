<!-- Copied from zdebruine/singlet-website PR #54 (branch claude/ontology-filters-46958f), docs/ONTOLOGY.md: handoff section and status -->
# Ontology mapping and filters

Status 2026-10-23. Sample metadata extracted from GEO text (`sample_attr`, schema/030) is mapped to ontology terms, and
search, the API, the MCP server and the Browse rail filter by those terms with descendants included. Applied to
production D1 on 2026-10-23 (034, load, refresh; see [Applying](#applying)). Where this fits in the cell architecture:
`docs/CELL-ARCHITECTURE.md`.

| Filter (`<name>_term`) | From `sample_attr.attr` | Ontology |
|---|---|---|
| `tissue` | tissue | UBERON |
| `cell_type` | cell_type_sorted | CL |
| `development_stage` | age, life_stage (with the organism) | HsapDv (human), MmusDv (mouse) |
| `disease` | disease | MONDO; PATO:0000461 "normal" for healthy/control values |
| `ancestry` | ancestry | HANCESTRO |
| `sex` | sex | PATO (female, male, mixed sex) |

Per-cell labels made elsewhere (cell type, chromosomal sex, inferred genetic ancestry) use the same ontologies (CL, PATO
sex terms, HANCESTRO), so the same ids filter samples and cells. Genotypes are never stored or served.

## Tables (schema/034_ontology.sql)

| Table | Rows (2026-10-23 build) | What |
|---|---|---|
| `ontology_term` | 2,622 | terms used by the catalog and all their ancestors: id, ontology, label, release, depth |
| `ontology_closure` | 40,974 | every (term, ancestor) pair, the term itself at distance 0; is_a everywhere, part_of too for UBERON, HsapDv, MmusDv |
| `ontology_label` | 12,630 | names a filter value may use: the id, the label, exact synonyms (normalised by `normLabel`) |
| `ontology_value_map` | 2,237 | one row per distinct `sample_attr` value and its term, with method, source, confidence and evidence |
| `sample_term` | 62,660 | per-sample terms, filled by `scripts/ontology/refresh.sql` |
| `ontology_term_count` | 2,614 | downloadable approved samples and studies under each term, descendants included |

`sample_attr.ontology_id` is set by the refresh where a value maps to exactly one term (78,573 rows touched; 62,689 get an id).

## Mapping (scripts/ontology/)

`build.ts` reads the OBO releases (`fetch-obo.sh`) and the distinct values (`export-values.sh`), and writes
`ontology-load.sql`, `value-map.jsonl`, `coverage.json` and `candidates.jsonl`. The committed outputs are in
`scripts/ontology/generated/` (CL 2026-06-08, UBERON 2026-10-01, MONDO 2026-10-06, PATO 2026-10-01, HANCESTRO 2025-10-14,
HsapDv and MmusDv current).

In order, for each value (`map-core.ts`):

1. **id** — the value is an ontology id of the right ontology (confidence 0.99).
2. **curated** — `aliases.ts`: abbreviations and spellings the ontologies do not list (PBMC → blood for tissue and
   peripheral blood mononuclear cell for cell type, NSCLC, COVID, US census ancestry categories, healthy → PATO normal).
   Every alias names its term's label; the build fails if the id is unknown, obsolete or has another label.
   Abbreviations whose reading depends on the study (AD, CD, MS) are not curated.
3. **label / exact_synonym / synonym** — the normalised value equals a primary label (0.95), an exact synonym (0.9) or a
   related/narrow synonym (0.75). A string two terms share at the best tier is ambiguous and stays unmapped. Related
   synonyms of strings under 6 characters are not used ("CD" is a related synonym of Cowden disease).
4. **variant** — the same after reading off qualifiers that do not change the entity (human, primary, sorted, tissue,
   biopsy, plural endings; for cell types sort markers and "blood" before a type: "CD3+ T cells" → T cell, "blood
   monocytes" → monocyte). Labels and exact synonyms only (0.85 / 0.8); organoids 0.7 with "(organoid)" in the evidence.
5. **stage** — ages are converted to days after birth (mouse E-days and human pcw/gestational ages to days before birth)
   and placed in the narrowest HsapDv/MmusDv stage whose `start_*`/`end_*` interval contains the whole range. Human ages
   in days or weeks are skipped unless the text says prenatal or postnatal; ranges without a unit ("18-30") are skipped.
   `life_stage` values map to fixed stages (adult → adult stage, aged → late adult stage, pediatric, fetal, embryonic).
6. **model** — the long tail (values on at least 2 samples that steps 1–5 left unmapped, 2,662 values) was read by Claude
   (Sonnet) with the study titles that use each value and lexical candidates, under the rules in `model-pass.md`:
   precision first, null when unsure, never a tissue from a disease word, ambiguous abbreviations only when every study
   agrees. 691 of 2,662 were mapped; one decision was rejected in review ("PCD" in a perianal Crohn's study read as primary
   ciliary dyskinesia). Decisions are `model-decisions.jsonl` (reviewed, committed); the build keeps one only if the id
   exists, is not obsolete, is in the attribute's ontology, is not a root and the label matches. Source `model`,
   confidence 0.5–0.8, the reason as evidence.

Ancestry terms must descend from HANCESTRO:0004 "ancestry category", the tree the per-cell inferred ancestry uses
(HANCESTRO:0005 European, 0008 Asian, 0010 African, 0014 Latin or Admixed American, ...), so self-reported "Asian" maps to
HANCESTRO:0008 rather than the geography-based HANCESTRO:0847; the build drops any ancestry mapping outside that tree.

Terms that are true of everything (cell, anatomical entity, organ, tissue, disease) are never targets. Values with no
defensible term stay unmapped.

## Coverage

Before: `sample_attr.ontology_id` was NULL on every row, and the only structured filters were the coarse groups
(`tissue_group`, `disease_group`) and free-text cell type.

After, share of `sample_attr` rows mapped (all samples with a fact; deterministic only → with the model pass):

| Attribute | Distinct values mapped | Sample rows mapped | Deterministic only |
|---|---|---|---|
| tissue | 908 / 2,119 | 26,315 / 32,652 (80.6%) | 71.4% |
| cell_type_sorted | 612 / 2,076 | 11,045 / 18,546 (59.6%) | 53.1% |
| disease | 302 / 639 | 4,398 / 6,000 (73.3%) | 59.4% |
| ancestry | 28 / 38 | 1,224 / 1,282 (95.5%) | 95.5% |
| sex | 3 / 3 | 7,718 / 7,718 (100%) | 100% |
| age → stage | 330 / 397 | 6,838 / 7,207 (94.9%) | 94.9% |
| life_stage → stage | 5 / 6 | 5,151 / 5,168 (99.7%) | 99.7% |

On the headline basis (approved samples of downloadable studies, 53,438 samples), samples that state the attribute and
now carry a term: tissue 24,461 / 30,533 (80.1%), cell type 10,481 / 17,414 (60.2%), developmental stage 6,649 / 6,770
(98.2%), disease 4,164 / 5,690 (73.2%), ancestry 1,220 / 1,278 (95.5%), sex 7,321 / 7,321 (100%).

Largest unmapped values, left unmapped on purpose: tissue "bronchoalveolar lavage" (no UBERON term for the fluid),
"esophageal adenocarcinoma", "tumor", "organoid", "cell line"; cell type "NeuN" (sign not given), "Adult", cell lines,
gate names; disease "CD", "AD" (two readings), "Cases", "infected"; ancestry "Not Hispanic or Latino" (ethnicity only),
lists of several donors. Values on a single sample (about 3,000) were not sent to the model pass.

Example filters on a local copy (study counts / sample counts, has_bundle on for studies):

| Filter | Studies | Samples |
|---|---|---|
| `tissue_term=lung` (UBERON:0002048, with lobes and other parts) | 272 | 2,150 |
| `tissue_term=lung&term_expand=none` | 260 | 2,074 |
| `cell_type_term=T cell` (with CD4/CD8 alpha-beta, gamma-delta, MAIT, Treg) | 236 | 2,281 |
| `cell_type_term=T cell&term_expand=none` | 122 | 1,191 |
| `disease_term=COVID-19` | 19 | 369 |
| `disease_term=PATO:0000461` (normal) | 225 | 1,205 |
| `development_stage_term=HsapDv:0000258` (human adult stage) | 373 | 4,368 |
| `ancestry_term=European ancestry` | 41 | 782 |
| `sex_term=female` | 489 | 3,595 |
| `tissue_term=lung&sex_term=female` | 21 | 218 |

Each filter is three indexed lookups (label → closure → `sample_term`); the counts above ran in 10–330 ms on a laptop with a warm cache.

## Filters

`/api/search`, `/api/nl-search`, `/api/facets`, `/api/manifest` and the export take `<name>_term` (repeatable, or
`|`-separated): ids in any common form (`UBERON:0002048`, `UBERON_0002048`, the PURL) or a label or exact synonym,
case-insensitive. Within one filter values are ORed (`<name>_term_mode=all` requires all); filters are ANDed with each
other and with everything else. A term matches samples mapped to it or any descendant; `term_expand=none` matches the
named terms only. At study level a study matches if any of its samples does. A value no label resolves matches nothing,
and nl-search offers dropping it like any other filter. Existing free-text filters (`cell_type`, `q`) are unchanged.

`GET /api/ontology/terms?filter=<name>[&q=][&values=a|b][&limit=]` lists the terms most samples are mapped to, searches
labels and synonyms (ancestors with no direct samples included, so "T cell" is offered when samples say "CD4 T cell"), and
resolves values to labels. Counts are downloadable approved samples and studies under each term.

The `search_datasets` MCP tool takes the same six filters as arrays (`tissue_term`, `cell_type_term`, ...). The Browse
rail has an "Ontology terms" section with one picker per filter; chips show the term labels.

## Applying

Applied 2026-10-23: 58,464 rows by the load, 143,877 by the refresh (10.4 s, 2.28M rows read). To apply again (in order;
each is idempotent):

```
npx wrangler d1 execute singlet-catalog --remote --file=schema/034_ontology.sql
npx wrangler d1 execute singlet-catalog --remote --file=scripts/ontology/generated/ontology-load.sql
npx wrangler d1 execute singlet-catalog --remote --file=scripts/ontology/refresh.sql
```

Expected rows written: about 58,460 by the load (terms, closure, labels, value map) and about 143,900 by the refresh
(62,660 `sample_term`, 78,573 `sample_attr` updates, 2,614 counts), plus index writes. The slowest refresh statement
(tissue counts) took 7 s on a laptop copy. Apply `034_ontology.sql` before merging: a search with a `<name>_term` filter
fails while the tables are missing (previews share the production database), searches without one are unaffected. With
the tables created but not yet loaded, term filters match nothing and the term endpoint returns empty lists.

Refreshing: run `refresh.sql` after new `sample_attr` rows are ingested (values already in the map are picked up). To
map new values, rerun `fetch-obo.sh`, `export-values.sh` and `npm run ontology:build -- --obo ontology-obo --values
values.json --model scripts/ontology/model-decisions.jsonl --out ontology-out`, review `candidates.jsonl` for a model pass,
then load and refresh.

## Not done

- The cohort bit index (`/api/cohort/*`, docs/COHORT-QUERY.md) still reads `sample_attr.value_norm`; once the refresh has
  set `sample_attr.ontology_id`, the index builder can encode terms and expand descendants the same way.
- No automatic refresh: `refresh.sql` is not in the catalog crank or the enrichment workflow yet.
- Values used by a single sample and abbreviations that need study context are unmapped; a per-study map would need a
  `gse_id` scope in `ontology_value_map`.

## Handoff (2026-10-26, research beta wind-down)

- **Production D1** (read-only check 2026-10-26): ontology_term 2,622, ontology_closure 40,974, ontology_value_map 2,237,
  sample_term 62,660, ontology_term_count 2,614, sample_attr rows with ontology_id 62,689, sample_property 22,555,
  property_value_map 994; cell_feature and sample_cell_summary exist and are empty. These match what was applied on 2026-10-23.
- **Migration numbering clash:** `main` now has `schema/034_mcp_oauth.sql`. This branch's `034_ontology.sql` and
  `035_cell_tier.sql` must be renumbered (e.g. 036/037) before merge, together with the references in `docs/`, `CLAUDE.md`,
  `src/test/ontology.test.ts` and the comments in `scripts/ontology/`. The tables are `CREATE ... IF NOT EXISTS` and already
  live, so renaming the files changes nothing in D1.
- **Not merged.** CI and the preview were green on 2026-10-23; rerun CI after the renumbering and a rebase on main.
- **Open:** load the cell index's `--d1-export` (or the feature job's ingest rows) into `cell_feature` / `sample_cell_summary`;
  natural language to cell query (waits for the cell-index Worker, website #55); schedule `refresh.sql` and `properties.sql`
  after ingests; the cohort bit index does not read terms yet; the cell-type labels need the work described in the
  "Singlet Cell Label Evidence" review (Singlet-Bio/singlet#32 and #44 are the follow-up).
