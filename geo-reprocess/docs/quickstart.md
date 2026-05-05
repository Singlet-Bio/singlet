# Quick Start

This guide walks through the complete workflow: catalog discovery → filtering → index building → processing → monitoring.

## 1. Build a catalog

Query NCBI GEO for single-cell RNA-seq series and enrich with metadata:

```python
import scgeo

# Full catalog build (discovery → metadata → SOFT → SRA)
catalog = scgeo.build_catalog(output="catalog.parquet")

# Check stats
stats = scgeo.get_catalog_stats(catalog)
print(f"Found {stats['total_series']} series, {stats['total_samples']} samples")
```

## 2. Filter the catalog

```python
# Human 10x datasets with 10-500 samples per series
filtered = scgeo.filter_catalog(
    catalog,
    organisms=["Homo sapiens"],
    min_samples=10,
    max_samples=500,
)
```

## 3. Build reference indices

```python
# Build splici index for human (downloads genome + GTF automatically)
scgeo.build_index("human")

# Check index path
print(scgeo.get_index_path("human"))
```

## 4. Process a single sample

```python
result = scgeo.process_sample(
    gsm_id="GSM3308545",
    gse_id="GSE115978",
    organism="human",
)
print(f"Cells: {result.n_cells}, Genes: {result.n_genes}")
print(f"Mapping rate: {result.mapping_rate:.1%}")
```

## 5. Submit a SLURM batch job

```python
job_id = scgeo.submit_batch(
    filtered,
    job_name="singlet_human",
    partition="general",
    cpus=36,
    memory="96G",
    time="4:00:00",
)
print(f"Submitted job {job_id}")
```

## 6. Monitor progress

```python
scgeo.monitor_job(job_id)
```

## CLI usage

All operations are also available via the `sc-geo` command:

```bash
# Build catalog
sc-geo catalog build --output catalog.parquet

# Filter
sc-geo catalog filter --input catalog.parquet --organisms "Homo sapiens" --min-samples 10

# Process single sample
sc-geo process --gsm GSM3308545 --gse GSE115978 --organism human
```
