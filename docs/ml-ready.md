# Training on the atlas without downloading it

`singlet.torch.AtlasCells` is a PyTorch `IterableDataset` that streams the cells of any atlas filter from
`data.singlet.bio` by HTTP range reads. Nothing is written to disk: the cell index picks the samples, each sample's `.s1`
object is read with byte ranges, and the cells arrive as shuffled mini-batches on one fixed gene axis, with their Cell
Ontology, sex and ancestry labels.

```python
from torch.utils.data import DataLoader
from singlet.torch import AtlasCells

train = AtlasCells({"organism": "Homo sapiens", "tissue": "UBERON:0002048"}, split="train", batch_size=512)
test = AtlasCells({"organism": "Homo sapiens", "tissue": "UBERON:0002048"}, split="test", genes=train.gene_axis.ids)
for batch in DataLoader(train, batch_size=None, num_workers=2):
    x, y = batch["X"], batch["cell_type_l2"]      # dense float32 log1p(CP10k); CL ids at level 2 (lineage)
```

## What it does

| Part | Behaviour |
|---|---|
| Filter | Any cell-index filter (`singlet.s1.atlas.find_cells` keywords). Sample-level: `organism`, `tissue` (UBERON), `disease` (MONDO, healthy = PATO:0000461), `development_stage` (HsapDv/MmusDv), `donor_status`, `perturbation`, `sample_kind`, `studies`, `samples`. Cell-level: `cell_type` (CL id or name, descendants included), `sex` (PATO or female/male), `ancestry` (HANCESTRO), `min_umi`/`max_umi`, `min_genes`/`max_genes`, `min_cell_type_score`. Or a `cohort`: a `/api/manifest` TSV, a DataFrame with `gse`/`gsm`, or accessions. |
| Planning | One cell-index count request (`POST /s1/cidx/query`, grouped by study and sample, paged when large) lists the samples with matching cells. Human, every tissue: 3,201 studies, 27,561 samples, 122.3M cells, planned in about 12 s. |
| Reading | Each sample's cells table and the matrix chunks holding the wanted cells, by range (about 2 requests per sample). Cell-level filters are applied to the cells table with the index's semantics; the streamed count per sample equals the index count (tested live). |
| Splits | `split="train" / "val" / "test"`: a study's split is SHA-256 of `(split_seed, GSE)` cut at 0.8 / 0.1 / 0.1. Every cell of a study is in the same split, on every machine, worker and epoch: no study leaks across splits. |
| Shuffling | A new sample order each epoch (`set_epoch`); every batch draws from `mix_samples` samples held open at once, each in a random cell order, so a batch mixes about ten studies (measured). `max_cells_per_sample` bounds the weight of very large samples. |
| Scale-out | Samples are divided between DataLoader worker processes and `torch.distributed` ranks; `prefetch` reader threads per process. Each batch reports its process's requests and bytes (`batch["stream"]`). |
| Labels | `cell_type_l1` .. `cell_type_l3` (CL ids along the labeller's path: compartment, lineage, cell type), `cell_type_score`, `sex`, `ancestry`, plus `gse`, `gsm`, `col` (the cell's column in its sample object) for provenance. |

The labels are the Research Alpha labeller's calls. Its per-study accuracy against CELLxGENE author labels is 0.84
(compartment), 0.75 (lineage), 0.43 (cell type) and 0.32 (subtype); levels 3 and deeper are previews
([cell_annotation.md](cell_annotation.md)). Genotypes are never stored or served; ancestry is a continental HANCESTRO group
and sex is chromosomal sex.

## Measured

All numbers come from `bench/stream/` and are collected in [`ml-ready/preliminary.json`](ml-ready/preliminary.json)
(`python bench/stream/summarize.py --out docs/ml-ready/preliminary.json`).

### Streaming throughput

Human, every tissue, random samples from the plan; a laptop (8 cores, Windows) on a residential link, one process, `mix_samples=16`, batch 512. The link itself (4 parallel downloads from Cloudflare's speed test) measured 8.94 MB/s before and 3.74 MB/s after the runs, so the stream ran at the link's rate; the per-cell figures are what carries over to a faster network: about 1.6-1.9 KB read per cell and 2 range requests per sample, with each batch drawing on about ten studies once 8 or more samples are read at a time. The training run below read 2.49M cells at 688 cells/s end to end, including the CPU training step.

| Reader threads | Samples (studies) | Cells | Wall s | Cells/s | MB/s | Bytes/cell | Requests/sample | Studies per batch |
|---|---|---|---|---|---|---|---|---|
| 4 | 147 (142) | 629,143 | 322 | 1,956 | 3.65 | 1,868 | 2.03 | 3.8 |
| 8 | 147 (134) | 650,328 | 368 | 1,767 | 3.14 | 1,779 | 1.93 | 9.5 |
| 16 | 144 (133) | 798,243 | 489 | 1,634 | 2.54 | 1,558 | 1.94 | 10.5 |

Skipped samples (listed by the index, absent from the released manifests): 3, 3, 6 of 150.

### Baseline: cell type on held-out studies

A two-layer MLP (1.46M parameters) on log1p(CP10k) of the 4,096 most variable genes (chosen from 40 training samples only),
one softmax head per CL level, trained for one pass over **2.49M human cells from 574 training studies**, streamed in 60 min
on the laptop (6.8 GB read by range; no file kept). Scored on **250,499 cells from 90 test studies** that training never saw
(the per-study figures cover the 87 with at least 20 labelled cells at the level).

| CL level | Accuracy | Per-study mean | Per-study median | Classes seen in training | Most common test label |
|---|---|---|---|---|---|
| 1 compartment | 0.957 | 0.918 | 0.994 | 8 | 0.534 |
| 2 lineage | 0.794 | 0.717 | 0.886 | 46 | 0.383 |
| 3 cell type | 0.617 | 0.515 | 0.582 | 163 | 0.250 |

The score is agreement with the atlas labels on unseen studies, not accuracy against author annotations. It shows that a
model trained on streamed atlas cells, with studies held out, learns the labelling across studies; the share of the most
common test label is what a constant answer would reach. One pass, one seed, no tuning.

### Reproduce

```bash
python bench/stream/throughput.py --samples 150 --workers 0 --prefetch 4 8 16 --out bench/stream/results/throughput_human_laptop.json
python bench/stream/train_celltype.py --train-cells 2500000 --cap-train 5000 --test-cells 250000 --cap-test 2000 --workers 0 --prefetch 8 --out bench/stream/results/train_celltype_human_laptop.json
python bench/stream/summarize.py --out docs/ml-ready/preliminary.json
```

## Notes

- **Windows receive buffer.** A Windows socket keeps its 64 KiB receive buffer, which held one connection to about 2 MB/s
  at a 30 ms round trip. The client now asks for 4 MiB on Windows (16-30 MB/s per connection measured); elsewhere the system's
  autotuning is left alone. `SINGLET_RCVBUF` overrides.
- **CPU training.** `EmbeddingBag` with per-sample weights (a sparse first layer over all 38,606 genes) took about 1.7 s per
  backward step on the laptop CPU, so the baseline reads a gene panel instead. On a GPU the full gene axis is practical.
- **Index and manifests.** A few samples the cell index lists are not in the released manifests (withheld studies, excluded
  samples); the loader skips and counts them (12 of 827 in the training run).
- **Machine-readable metadata.** The atlas and every study also publish MLCommons Croissant metadata on singlet.bio: the atlas at `https://singlet.bio/croissant.json`, a study at `https://singlet.bio/api/croissant/<GSE>`, a sample at `https://singlet.bio/api/croissant/<GSE>/<GSM>` (live once the website change is deployed).
