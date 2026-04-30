# singlet-gpu — Website & Publishing Contract

The website at **singlet.bio** is the public face of singlet-gpu. This file defines exactly what data flows where, in which phase, and what the orchestrator can rely on.

## Owned surfaces

| URL | Owner | Content |
|---|---|---|
| `singlet.bio/benchmarks` | singlet-gpu | Pareto frontier table, per-feature speedup charts |
| `singlet.bio/benchmarks/{feature}` | singlet-gpu | Per-feature deep dive, real-data correctness, citations |
| `singlet.bio/docs` | singlet-gpu | API reference (auto-generated from `docs/api/*.md`) |
| `singlet.bio/docs/install` | singlet-gpu | Build + pip + R install instructions |
| `singlet.bio/docs/quickstart` | singlet-gpu | Load `.1pz` → standard pipeline → results |
| `singlet.bio/notebooks` | singlet-gpu | Reproducibility notebooks rendered from `docs/notebooks/` |
| `singlet.bio/blog/singlet-gpu-{slug}` | gpu-doc-scribe | One blog post per major frontier promotion |
| `singlet.bio/changelog` | singlet-gpu | Auto-rendered from `CHANGELOG.md` |

## Supabase tables

URL: `https://vbswbitfyallghbgxkuw.supabase.co`. Writes require `SUPABASE_SERVICE_KEY`.

**Secrets policy**: never write the key value into any file under `singlet-gpu/`. The repo references the variable name only. Values live in `~/.config/singlet/supabase.env` (chmod 600). Source before publishing:

```bash
source ~/Singlet-AI/singlet-gpu/scripts/load_secrets.sh
```

See `state/infrastructure.md` § Supabase for the full pattern + key rotation steps.

| Table | Owner | Schema highlights |
|---|---|---|
| `gpu_frontier` | singlet-gpu | feature, scale, wall_ms, mem_mb, accuracy_metric, sota_lib, sota_wall_ms, ratio, commit |
| `gpu_correctness` | singlet-gpu | feature, scale, metric_name, value, tolerance, reference, sample_id |
| `gpu_releases` | singlet-gpu | version, date, features_added, features_deprecated, breaking |
| `samples` | singlify | (read-only for us) |
| `e2e_results` | singlify | (read-only for us) |
| `pipeline_batches` | singlify | (read-only for us) |

## Phase G — Publish flow

Run after every cycle that updates the frontier. Order matters; each step is gated on the previous succeeding.

### G.1 — Frontier sync (always)

```bash
python3 singlet-gpu/scripts/frontier_sync.py
```
Pushes updated rows from `state/pareto-frontier.md` and `state/correctness-registry.md` to Supabase `gpu_frontier` + `gpu_correctness`.

If `SUPABASE_SERVICE_KEY` is unset or the script fails: log to `state/blockers.md` as `INFRA-WEBSITE-FRONTIER-SYNC` and continue. Never block a cycle on publishing. The script's offline-cache path writes `state/frontier_sync_cache.json` so the next run with credentials can replay the upload.

### G.2 — Docs page (mandatory if Phase H ran)

`docs/api/{feature}.md` must exist and contain:
- One-line summary
- Public C++ signature (from `singlet_gpu.hpp`)
- Python signature (from wrapper)
- R signature (from wrapper)
- Config struct documentation
- Complexity (time + memory + streaming)
- Correctness tolerance + reference tool
- Citation
- Minimal example (5–15 lines)

Build the static site:
```bash
cd singlet-gpu/docs && mdbook build
```
(Future: copy output to website repo. Currently writes locally.)

### G.3 — Notebook (required for `documented` transition)

`docs/notebooks/{feature_slug}.ipynb` must follow this structure:

1. **Overview** — feature description, which SOTA tools it replaces.
2. **Setup** — install singlet-gpu, load real `.1pz`, show version + commit.
3. **Run singlet-gpu** — show API call, display outputs.
4. **Run reference tool(s)** — Scanpy / rapids-sc / Seurat on same data.
5. **Formal equivalence** — correlation plots + metrics table (r, RMSE, Jaccard).
6. **Performance benchmark** — 3-scale timing comparison with bar charts.
7. **Biological validation** — real-data application showing meaningful results.
8. **Conclusion** — summary table: equivalence metric, speedup, memory.

**Real data only.** Planted-signal tests are useful for unit tests but never sufficient for the docs notebook. GSM4037629 minimum.

A frontier feature without a passing notebook is NOT `documented`, regardless of benchmark numbers.

### G.4 — Blog post (only on major promotions)

Trigger: feature transitions `documented → released` AND the kernel introduces a new SOTA-beating result OR a "first GPU implementation in the field." Not every frontier promotion warrants a blog.

```bash
python3 /mnt/home/debruinz/Singlet-AI/singletai-website/scripts/etl/publish_blog.py \
  --slug "singlet-gpu-{feature_slug}" \
  --title "singlet-gpu: {Feature Title}" \
  --summary "{one-paragraph summary}" \
  --tags "gpu,benchmark,{feature_area}" \
  --content-file /path/to/content.md \
  --author "Singlet Team"
```

If the website repo (`singletai-website`) is not present locally or `publish_blog.py` is missing: log to `state/blockers.md` as `INFRA-WEBSITE-BLOG` and continue.

### G.5 — Changelog (every cycle that promotes)

Append to `CHANGELOG.md`:
```markdown
## [unreleased]
### Added
- {feature}: {one-line description} ({speedup vs SOTA}). docs: docs/api/{slug}.md
```

At release tag time, the `[unreleased]` section becomes `[MAJOR.MINOR.PATCH] - YYYY-MM-DD`.

## What gets backfilled (one-time, before any new feature work)

The following frontier features were promoted before this contract existed and need backfill rows in Supabase + docs pages:

- io/pz_device_loader (#0)
- preprocess/lognorm (#2 total-count + log1p)
- preprocess/lognorm-deconvolution (#2 scran sub-variant)
- preprocess/hvg-seurat-v3 (#3)
- preprocess/hvg-pearson-residuals (#3)
- reduce/svd-deflation (#4 winner)
- reduce/nmf (#5)
- qc/metrics (#6)
- preprocess/scale (#7)
- graph/knn-exact (#8 partial)
- de/wilcoxon (#11)
- de/ttest (#11)

This is the **website backfill cycle** — tag it `[INFRA-WEBSITE-BACKFILL]` in cycle-log. Run once after `frontier_sync.py` is verified working.

## Emergency: nothing publishes

If `SUPABASE_SERVICE_KEY` is unset, `frontier_sync.py` is broken, or the website repo isn't local:
1. Log each as a `INFRA-WEBSITE-*` entry in `state/blockers.md`.
2. Continue all subsequent cycles. Local state files (`pareto-frontier.md`, `cycle-log.md`, `docs/api/`) are the durable record.
3. When unblocked, the next cycle picks up Phase G for everything that ran in the meantime.
