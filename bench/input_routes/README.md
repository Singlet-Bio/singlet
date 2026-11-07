# Input routes: FASTQ, BAM and protocol specs for the benchmark corpus

`singlet download` used to stage reads only from SRA runs. In the 100-row benchmark corpus, 52 rows were marked as planned and 17 had no
route. This branch (`feat/input-routes`, off `perf/gpu-candidate-pr2`) adds two input paths and three protocol specs. With them, 77 rows have a
route and 23 do not ([corpus-singlet-routes.tsv](corpus-singlet-routes.tsv)). Every test below ran on a development library: a 10x sample, a
library outside the corpus, or a sibling run of a corpus row. No held-out corpus row was run (`sprint/intel/corpus-engine-touch.md` in
singlet-agents).

## What changed

**FASTQ and BAM inputs** ([spot_source.h](../../include/singlet/fq/spot_source.h)). A `SpotSource` gives the SRA encoder the same spots a VDB
run would, so detection, orientation, whitelist checks, R1 trimming and the writer are unchanged. Inputs:

- `--r1/--r2[/--i1/--i2] F[,F...]`, plain or gzipped, one file per lane;
- a FASTQ folder, a 10x tar or a URL to either. Illumina names (`_S1_L001_R1_001`) and Cell Ranger 1.x GemCode names
  (`read-RA/I1/I2_si-*_lane-*`) are recognized. A tar with several libraries needs `--fastq-sample`;
- `--fastq-spec` for any other layout;
- `--bam` for a Cell Ranger BAM (local or URL). The barcode read is rebuilt from CR/CY and UR/UY through samtools, and the cDNA read comes back
  in its sequenced orientation.

URL fetches resume across dropped connections. Staged files are removed unless you pass `--keep-staged`. Pass `--source-label` to name the
source in the .1fq metadata.

**Protocol specs** ([protocol.h](../../include/singlet/fq/protocol.h)). The new specs use the read layouts measured on development runs:
per-position entropy plus distinct k-mers, where barcodes repeat and UMIs do not.

| Spec | Layout of the barcode read | Measured on |
|---|---|---|
| `celseq2-u6b8` (id 26) | UMI 6 at 0, barcode 8 at 6; R1 of 30-32 nt trimmed to 14 | SRR8606533 (Tian), SRR9621729 (Mereu) |
| `scrbseq` (id 27) | barcode 6 at 0, UMI 10 at 6 | SRR9621767 |
| `icell8` (id 28) | barcode 11 at 0, UMI 14 at 11 | SRR9621748 |

The new specs are **forced-only**: you reach them with `--protocol`, and auto-detection skips them. Adding one cannot change the call on any
library that already processes. In auto mode the engine still refuses non-10x protocols, so Drop-seq and Seq-Well rows pass
`--protocol dropseq|seqwell`. The encoder already trims an over-long R1, such as Mereu's 76-nt Drop-seq read, to the spec length.

The .1fq aligner parameters now record the barcode start (`soloCBstart`). Before this, every simple spec had its barcode at offset 0. The first
CEL-Seq2 run showed the gap as 34,011 one-UMI "cells".

## Results

The FASTQ path matches the SRA path byte for byte. SRR24098804 (10x 3' v3, 32.2M spots, outside the corpus) was staged three ways: from the
.sra file, from `fasterq-dump --split-files --include-technical` FASTQ, and from the two biological reads gzipped. All three .1fq files share
sha256 `1b12ec5174291c58…`.

The BAM path recovers the reads exactly. The pbmc1k_v3 Cell Ranger 3 BAM and the FASTQ folder of the same library each give 66,601,887 read
pairs. The sorted pair multisets have the same hash, with no pair on either side only.

Each route then ran the CPU pipeline on the staged reads (`jobs/routes.sbatch`; 4M spots for the SRA libraries):

| Route | Library | Cells | Median UMI | Reads in cells |
|---|---|---|---|---|
| BAM, Cell Ranger 3 | pbmc1k_v3 | 1,225 | 6,960 | 0.91 |
| GemCode tar (URL) | pbmc3k (10x reports ~2,700 cells) | 2,709 | 2,239 | 0.93 |
| `--protocol dropseq`, R1 76 | Mereu N703, SRR9621743 | 431 | 371 | 0.58 |
| `celseq2-u6b8` | Tian, SRR8606533 | 198 | 1,405 | 0.89 |
| `celseq2-u6b8` | Mereu, SRR9621729 | 159 | 225 | 0.81 |
| `scrbseq` | Mereu, SRR9621767 | 621 | 271 | 0.74 |
| `icell8` | Mereu, SRR9621748 | 813 | 1,476 | 0.93 |
| auto (10x 3' v2, 94% whitelist) | Mereu nuclei, SRR9621777 | 1,475 | 76 | 0.53 |
| `--protocol seqwell` | Gierahn Species Mixing 2, SRR5250841 | 396 | 393 | 0.66 |

The Cell Ranger 1.x BAM (Kang 2018 batch 2 stim, SRR5398239, 10x v1; not a corpus row) is still open. ENA served the 22.8 GB file at
about 0.6 MB/s, and the first attempt failed with a dropped TLS stream, which led to the resumable fetch. A run on the first 3 GB of the BAM
(`ROUTE=bam_kang_head`, Clipper job 803380) was running when this was written. Its summary lands in
`$W/jobs/input-routes/routes/bam_kang_head-803380/summary.txt`. The Kang rows (15-18) are marked yes on the strength of the CR3 BAM test,
because the tags used (CR/CY/UR/UY) are the same in Cell Ranger 1.x.

CPU and GPU outputs match on pbmc1k_v3 and neuron1k_v3: all 65 files, with only the accepted run-dependent diffs (out_equiv 65/5/4).
The gate ran on df7d680e (input paths) and again on 2cb9d941, which carries every engine change; the head adds only a test fix. ctest on the
head passes all 159 tests, with test_gpu_pileup_labels skipped because the node has no GPU.

## Open rows

- Mereu ddSEQ (52-nt R1) puts a phase block of variable length before the first barcode and uses other linkers than the `ddseq` and `surecell`
  specs, so it needs an anchored extractor.
- sci-RNA-seq v1 (Cao 2017) spreads each cell over one SRA run per PCR well, which needs a per-run barcode prefix.
- Mereu inDrop has a single deposited run, the corpus row, so no development run exists to check the layout on.
- Visium, GEM-X v4 and Flex need barcode or probe lists that ship only with 10x software.
- Plate-based full-length protocols have no barcode read.
- The Seq-Well corpus row has a 32-nt R1, but the available sibling has 20 nt. The 32-nt read takes the same trimming path as the 76-nt
  Drop-seq read, which was tested.

## Reproduce

Job scripts are in `jobs/`. Results live on Clipper under `$W/jobs/input-routes/` (`identity/`, `routes/`, `results/gate-*`, `ctest/`), and
the summaries are copied to `results/`.
