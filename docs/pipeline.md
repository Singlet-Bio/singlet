# Running the pipeline

End-to-end: a URL or SRA accession in, the canonical
[per-sample output layout](CANONICAL_OUTPUT_FORMAT.md) out.

The same entry point is available from Python and the command line.

## Install

```bash
pip install singlet
```

The Python wrapper invokes the `singlet` C++ binary. Build it once with
`cmake --build singlet/build` (or `pip install singlet[pipeline]` once
the wheel ships a pre-built binary) and either place it on `$PATH`,
set `$SINGLET_BINARY`, or pass `binary=` to `singlet.pipeline.run`.

You will also need a reference bundle. Either set `$SINGLET_REF_BASE`
to a directory containing `GRCh38-2024-A/` (and `GRCm39-2024-A/` for
mouse) or pass `ref_base=` explicitly.

## From the command line

```bash
# SRA Run accession
singlet-process SRR11537951 --output-dir ./out --organism human --threads 8

# Direct URL to an .sra / .1fq / .fastq file
singlet-process \
    https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR11537951/SRR11537951 \
    --output-dir ./out

# Paired FASTQ URLs
singlet-process --reads R1.fastq.gz R2.fastq.gz --output-dir ./out

# Pass-through to the underlying binary after `--`
singlet-process SRR11537951 -o ./out -- --min-mapq 30 --cell-calling
```

Run `singlet-process --help` for the full flag set.

## From Python

```python
from singlet.pipeline import run

result = run(
    "SRR11537951",
    output_dir="./out",
    organism="human",
    threads=8,
    nonhost=True,        # populate nonhost.json + nonhost_species.1pz
)

assert result.success
print(result.output_dir, result.elapsed_s)
```

Accepted `source` forms:

| Form                              | Example                                                                       |
| --------------------------------- | ----------------------------------------------------------------------------- |
| SRA Run accession                 | `"SRR11537951"`                                                               |
| SRA / ENA URL                     | `"https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR.../SRR..."`                |
| `.1fq` URL or local path          | `"/data/SRR11537951.1fq"`                                                     |
| Paired FASTQ URLs or local paths  | `["R1.fastq.gz", "R2.fastq.gz"]`                                              |

## Key hyper-parameters

| Argument                  | Default | Purpose                                                  |
| ------------------------- | ------- | -------------------------------------------------------- |
| `organism`                | `"human"` | Resolves reference bundle (GRCh38, GRCm39, …)          |
| `threads`                 | auto    | Worker threads.                                          |
| `enable_snps`             | `True`  | Emit `snp.1pz`.                                          |
| `enable_pipeline_extras`  | `False` | Donor demux + mt heteroplasmy (`--pipeline`).            |
| `cascade`                 | `"off"` | Secondary aligner cascade (`on` / `auto`).               |
| `te_classify`             | `"off"` | TE / repeat classification.                              |
| `nonhost`                 | `False` | Viral + microbial screening.                             |
| `raw_matrix`              | `False` | Also emit unfiltered (all-barcodes) matrix.              |
| `metadata_json`           | `None`  | Sample metadata to embed in outputs.                     |
| `extra_args`              | `()`    | Pass-through list of binary flags.                       |

## Reading the outputs

Every run produces a directory matching
[`CANONICAL_OUTPUT_FORMAT.md`](CANONICAL_OUTPUT_FORMAT.md). Load it
with the `SingletSample` reader:

```python
from singlet.io import SingletSample
from singlet.views import gene_counts, usa, psi

sample = SingletSample(result.output_dir)

# Top-level metadata
print(sample.summary["n_cells"], sample.summary["mapping_rate"])
print(sample.cell_meta.head())

# Sparse views (CSC, genes × cells)
genes = gene_counts(sample)           # spliced + unspliced + ambiguous projected to genes
trio = usa(sample)                    # UsaTriplet(spliced, unspliced, ambiguous)
psi_mat = psi(sample)                 # per-junction × cell PSI

# Raw row blocks of counts.1pz
exon = sample.counts.exon_body()
intron = sample.counts.intron_body()
junctions = sample.counts.junctions()

# Variant tracks (two-layer CSC: AD + DP)
snp_ad = sample.snp.ad()
snp_dp = sample.snp.dp()
mt_vaf = sample.mt.vaf()              # AD / DP

# Non-host (Kraken2 + Bracken) — present only when --nonhost was set
if sample.nonhost is not None:
    print(sample.nonhost.summary())
    species_per_cell = sample.nonhost.per_cell()
```

All reader methods are lazy: nothing is decoded until you request a
specific block, and matrices come back as `scipy.sparse.csc_matrix`
ready for downstream analysis.

## What gets written

For a full reference of every file the pipeline produces, see
[Canonical output format](CANONICAL_OUTPUT_FORMAT.md). The minimal
always-present set is:

```text
out/
├── counts.1pz              # 3 row blocks: exon_body | intron_body | junctions
├── snp.1pz                 # two-layer CSC (AD, DP) on population SNP panel
├── mt.1pz                  # two-layer CSC (AD, DP) on chrM positions
├── cell_meta.parquet       # all cell-level scalars
├── summary.json            # all sample-level scalars + provenance
├── saturation_curve.tsv
└── star_Log.final.out
```

Optional siblings appear when their feature layer is enabled:
`nonhost.json` + `nonhost_species.1pz`, `guides.1pz`,
`antibodies.1pz`, `vdj_gene_usage.1pz`, `donor_*`,
`ambient_profile.npy`, `splice_events.tsv`.

## Errors

The runner raises `singlet.pipeline.PipelineError` on any failure
(binary missing, reference missing, non-zero exit). The exception
message includes the last 2 KB of the binary's stderr.

```python
from singlet.pipeline import run, PipelineError

try:
    run("SRR_does_not_exist", "./out")
except PipelineError as exc:
    print("pipeline failed:", exc)
```

## See also

- [Canonical output format](CANONICAL_OUTPUT_FORMAT.md)
- [`.1pz` format](format-1pz.md)
- [`.1fq` format](format-1fq.md)
- [Quickstart notebook](notebooks/pipeline_quickstart.ipynb)
