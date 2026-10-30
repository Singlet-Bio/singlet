# bench/mtdna_lineage

Benchmark behind `docs/mtdna-lineage.md`: how Singlet should store and use chrM data for clonality / lineage.

Data: MAESTER (Miller et al. 2022, GSE182685): the same cDNA sequenced as a standard 10x 3' v3 library and as a
MAESTER mtDNA-enriched library. Full depth, every read SRA holds (no read cap).

| name | run | library | truth |
|---|---|---|---|
| `k562_3p` | SRR15598773 | 10x 3' v3, K562 + BT142 cell lines | RNA cell-line labels (`TenX_CellLineMix_Cells.rds`, authors) |
| `k562_maester` | SRR15598774 | MAESTER of the same cells | K562 subclonal variants by the authors' rule |
| `bpdcn_3p` | SRR15598776 | 10x 3' v3, BPDCN712 bone marrow (clonal hematopoiesis) | 23 MAESTER clones (`4.4_positive_cells.txt`, authors) |
| `bpdcn_maester` | SRR15598777 | MAESTER of the same cells | |

Steps (Clipper, Slurm; `$W/jobs/mtl` holds copies of these files):

```
fetch.sbatch <SRR> <name> <sra|ena>     # 3' runs from SRA (R1 is a technical read ENA drops), MAESTER from ENA
setup_env.sbatch                         # envs/mtl: mgatk 0.9.0, maegatk, MQuad, vireoSNP, R; truth tables; gnomAD chrM
run_singlet.sbatch <name>                # singlet-modfix --pipeline --tagged-bam on a 3' library
align_maester.sbatch <name>              # STARsolo on MAESTER reads (same GRCh38-2024-A index), chrM BAM
pile.sbatch <name> <bam> <cells> [mgatk|maegatk]   # mtpile (read + molecule level, strand) and the reference tool
analyze.sbatch lines|bpdcn|size          # analyze.py -> results/*.json
haplogrep.sbatch                         # Haplogrep 3 on pseudobulk homoplasmic calls (private outputs)
survey.sbatch                            # atlas_survey.py: chrM reads per cell in 400 random human atlas studies
python atlas_depth.py ...                # full SRA depth of the surveyed samples (ENA)
```

`mtpile.cpp`: per-cell chrM allele counts by strand from a CB/UB-tagged BAM, at read level (Singlet's counting rules:
MAPQ >= 20, base quality >= 10) and at molecule level (UMI consensus, >= 75% of reads; `umi3` = molecules with >= 3
reads, as maegatk `-mr 3`). `mtlib.py`: loaders, site AD/DP matrices, mgatk/Signac variant statistics, vireoSNP clone
fits. `synth_test.py`: smoke test of `analyze.py` on simulated piles.
