# VAL2: 500+ Sample Validation Design

## Goal
Validate singlify across 500+ SRA samples spanning all supported protocols, species, and modalities.

## Sample Selection Criteria
- **Protocol diversity**: ≥30 samples per major protocol family (10x v2/v3, Drop-seq, sci-RNA-seq, Smart-seq2, Bulk RNA, ATAC, CITE-seq, Visium)
- **Species diversity**: ≥20 samples per top-5 species (human, mouse, zebrafish, rat, drosophila)
- **Modality coverage**: ≥10 samples for each supported modality
- **Size range**: Mix of small (<10M reads), medium (10-100M), large (100M+)
- **Edge cases**: Known-difficult samples from VAL1 triage

## Architecture
```
val2_samples.csv (500+ rows)
    ↓
scripts/val2_download.sh (SLURM array, %100 concurrent)
    ↓
scripts/val2_process.sh (SLURM array, %50 concurrent)
    ↓
scripts/val2_analyze.py (aggregate results)
    ↓
state/val2_results.csv + state/val2_flagged.csv
```

## val2_samples.csv Format
```
srr,species,protocol_expected,modality,reads_expected,source
SRR...,human,10x-3p-v3,scRNA,50000000,geo_catalog
```

## Download Script Template
- Input: val2_samples.csv
- Per task: fasterq-dump from NCBI → singlify encode → .1fq
- Retry: 3 attempts with exponential backoff
- Memory: 128G (for large samples)
- Timeout: 6h per sample
- Output: val2/<SRR>/<SRR>.1fq + metadata.json

## Process Script Template
- Input: val2/<SRR>/<SRR>.1fq
- Per task: singlify process (auto-detect everything)
- Memory: 384G
- Timeout: 2h per sample
- Output: val2/<SRR>/output/ (standard pipeline outputs)

## Success Criteria
- ≥90% of human/mouse scRNA samples: mapping rate ≥50%
- ≥80% of all scRNA samples: cells called ≥100
- ≥75% of non-human/non-mouse: mapping rate ≥30%
- 0 pipeline crashes (exit code ≠ 0 that isn't a known data issue)
- Protocol auto-detection accuracy ≥95%
- Species auto-detection accuracy ≥99% (for supported species)

## Risk Mitigation
- Run in waves: first 100, then 200, then 200
- Each wave: download → process → analyze → fix bugs → next wave
- Bug fixes get regression tests before next wave
- Known bad samples (from VAL1) excluded from success rate calculation

## Timeline (estimated)
- Wave 1 (100 samples): select + download + process + analyze
- Wave 2 (200 samples): expand protocols + species
- Wave 3 (200 samples): edge cases + stress testing
