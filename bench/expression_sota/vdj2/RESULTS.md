# vdj2: results

Pre-registration: `PREREG.md` (commit 4ca0993b; Addenda 1 and 2 committed after it, before any held-out run).

**Frozen code: `08279d8dc75bc7261439f3eb4e7b6fd86ceee001`.**
- Defaults: `ambient_rescue="clone"`, `v_open_frac=0.3`, `orphan_max_umis=0`.
- Human references: GENCODE/RefSeq plus 7 AIRR-C IG V genes (CC0; see `python/singlet/vdj/data/README.md`).

**Held-out run (submitted once, 2026-10-31):**
- Singlet: array 795124.
- TRUST4: array 795125.
- Scoring: job 795126, writing `scores/heldout.md`.

## Diagnosis (development libraries only)

`diag2.py` classifies every reference chain of a reference-paired cell that Singlet's top contig does not reproduce
(`results/diag/diag2_*_base.json`, baseline = current code, reference by Cell Ranger convention).

| library (CR) | paired cells | Singlet pairing | lost chains: cross-filtered (exact junction assembled) | UMI tie / not top | other junction | no contig of the chain |
|---|---|---|---|---|---|---|
| S149B (3.0.1, donor of burned S149A) | 1,291 | 0.855 | **110** (95% leave the chain empty) | 25 | 34 | 25 |
| S133A (3.0.1) | 3,483 | 0.858 | **153** | 143 | 170 | 44 |
| S135B (3.0.1) | 1,219 | 0.912 | 35 | 16 | 31 | 24 |
| S150B (3.0.1, donor of burned S150A) | 544 | 0.886 | 3 | 13 | 38 | 8 |
| pbmc2_t / pbmc10k_t | 2,479 / 4,737 | 0.931 / 0.939 | 3 / 2 | 40 / 122 | 75 / 59 | 52 / 102 |

* **Main cause of the tumour-library chain losses: the cross ("ambient") filter.** A junction with < 10% of the UMIs it has in
  its strongest barcode (when that has >= 10) is removed as ambient. In ESCC tumour libraries with expanded clones, a real
  member of a clone that expresses little TCR mRNA has the clone's junction with 1 UMI (median; ~2,500 reads) against a
  sibling's 17-23, so its only contig of that chain is removed. Cell Ranger 3.0.1 has no such filter and calls these cells.
* Not the cause: the adaptive molecule read threshold (`diag_reads.py`: of 25 S149B chains with no contig, 24 have UMIs above
  the threshold; the "no contig" class is small everywhere), barcode correction, productivity.
* UMI ties: 40-85% of "reference junction kept but not top" are UMI ties; Cell Ranger's tie order does not follow reads
  (Singlet already breaks ties by reads), so nothing to fix there.
* BCR V / J (pbmc10k_b, pbmc2_b): V disagreements go to genes Cell Ranger does not call: IGHV1-8 / IGHV4-30-4 absent from
  GENCODE/RefSeq (-> IGHV1-2, IGHV4-30-2/4-55), germline-stop paralogs kept by the reference's open-V rescue (IGHV4-55 takes
  IGHV4-61/4-59/4-39 reads, 56 cells in pbmc2_b; IGLV1-41 takes IGLV1-51, 34), and near-identical pairs (IGHV3-30/3-33). J:
  IGLJ2 -> IGLJ3 (12) and IGHJ4 -> other IGHJ (25) in pbmc10k_b; the read vote past F/W118 sees mostly the shared J 3' end.
  V/J naming does not change pairing (junction nt).

## Changes (python/singlet/vdj)

* `filter_contigs` (refactor): the contig filters can be re-applied to a finished `contigs.csv`
  (`refilter.py`; with unchanged settings it reproduces the assembler's `kept` exactly on all 9 dev libraries).
* `ambient_rescue` = `cell` | `pair` | `clone`; `clone`: a cross-filtered junction is kept when the droplet's own kept
  partner chain and it are the top pair of another barcode where both passed the filter.
* `v_open_frac`: a V gene with a germline stop yields to an open V with >= this share of its k-mer votes.

## Development results so far (six earlier dev libraries; S149B/S150B/S135B wait for TRUST4)

`scores/dev_*.md` on Clipper (`$WORK/expression-sota/vdj2/scores`). Pooled pairing (13,493 cells), Singlet vs TRUST4 0.9164:

| variant | pooled | TCR | BCR | S133A | pbmc2_t ARI | wrong lineage (pooled) | dev criterion |
|---|---|---|---|---|---|---|---|
| base | 0.9177 | 0.9121 | 0.9485 | 0.8576 | 0.9003 | 0.0025 | - |
| cell | 0.9268 | 0.9220 | 0.9528 | 0.8880 | 0.8891 | 0.0026 | REJECT (pbmc2_t ARI -0.011) |
| pair | 0.9270 | 0.9223 | 0.9528 | 0.8889 | 0.8891 | 0.0029 (pbmc2_b +0.0033) | REJECT |
| clone | 0.9254 | 0.9213 | 0.9485 | 0.8857 | 0.8891 | 0.0025 | literal REJECT (ARI); Addendum 1 |

`clone` rescues 115 contigs on S149B (82 of them exactly the reference contig), 141 on S133A, 5 on pbmc2_t, 2 on pbmc2_b.
The pbmc2_t ARI drop comes from 4 cells of 2,982, two of which become identical to Cell Ranger's call (`ari_check.py`);
ARI's own bootstrap SD is 0.008.

## Final development decisions (all 9 dev libraries; jobs 795043, 795077)

| variant | pooled pairing (16,547 cells) | vs TRUST4 0.9104 | BCR V agreement pbmc2_b / pbmc10k_b | wrong lineage | literal dev rule | decision |
|---|---|---|---|---|---|---|
| clone | 0.9238 [0.920, 0.928] | +0.0134 [+0.009, +0.018] | 0.802 / 0.849 | 47/19,052 = 0.0025 (CR 0.0002) | REJECT (pbmc2_t ARI -0.011) | adopted (Addendum 1) |
| clone + orphan <= 1 UMI | = | = | = | 24/18,982 | REJECT (F1 S158A, S135B) | rejected |
| clone + orphan <= 2 UMI | = | = | = | 14/18,885 | REJECT (F1 S158A -0.06, S135B -0.07) | rejected |
| AIRR-C refs | = | = | 0.782 / 0.867 | = | REJECT (pooled pairing +0.000) | - |
| AIRR-C + v_open_frac 0.3 | = | = | 0.802 / 0.868 | = | REJECT (pooled pairing +0.000) | adopted (V agreement) |
| AIRR-C + v_open_frac 0.5 / 0.7 | = | = | 0.798 / 0.868; 0.789 / 0.868 | = | REJECT (pooled pairing +0.000) | - |

The orphan rule is in the code (off). Lone low-UMI single-chain contigs are wrong-lineage on the PBMC libraries, but often
Cell Ranger's own contigs on the ESCC tumour libraries, so contig F1 drops beyond the tolerance there.

## Held-out results (single run of the frozen code `08279d8d`; jobs 795124 / 795125 / 795126)

The full table is in `results/heldout/heldout.md` and `heldout.json`. The metric is pairing accuracy over reference-paired
cells, restricted to GEX cells where GEX exists, with 2,000-bootstrap 95% CIs.

| group | cells | Singlet [95% CI] | TRUST4 | difference [95% CI] | pre-registered rule (>= -0.01) |
|---|---|---|---|---|---|
| TCR | 21,996 | 0.9193 [0.916, 0.923] | 0.9139 | +0.0054 [+0.001, +0.009] | **PASS** |
| BCR | 7,267 | 0.8698 [0.862, 0.877] | 0.9207 | -0.0509 [-0.057, -0.045] | **FAIL (clear: the CI is entirely below -0.01)** |
| all | 29,263 | 0.9070 [0.904, 0.910] | 0.9156 | -0.0086 [-0.012, -0.005] | PASS on the point estimate (the CI is not entirely below -0.01) |

**Wrong lineage, pooled over the 7 libraries with GEX:**
- Singlet 293/35,529 = 0.0082; Cell Ranger 70/32,587 = 0.0021.
- The difference is +0.0061. The rule is <= 0.005, so **FAIL**.
- 207 of Singlet's 293 come from ngpbmc3_b, where Cell Ranger also has 57/2,110.
- Post hoc, not pre-registered: without ngpbmc3_b, Singlet has 86/33,212 = 0.0026 against Cell Ranger's 13/30,477 = 0.0004.

**Pairing difference per library (Singlet minus TRUST4):**

| library | difference |
|---|---|
| S134A | +0.018 |
| S134B | +0.013 |
| S159A | -0.009 |
| S159B | +0.022 |
| pbmc3_t | +0.015 |
| ngpbmc3_t | +0.007 |
| pbmc1k_t | -0.002 |
| pbmc3_b | -0.040 |
| ngpbmc3_b | -0.061 |
| pbmc1k_b | -0.012 |
| melanoma_b | -0.051 |

**Secondary metrics (no pass rule):**
- **Contig F1:** Singlet leads TRUST4 on all 11 libraries.
- **Clonotype ARI:** Singlet leads on 10 of 11. It does not lead on pbmc3_b (0.174 vs 0.222).
- **BCR V agreement:** about equal to TRUST4's.
- **BCR J agreement:** below TRUST4 on all 4 BCR libraries (0.949-0.975 vs 0.978-0.994).
- **BCR chain recovery:** about equal (for example 0.953 vs 0.952 on ngpbmc3_b). So the BCR pairing loss comes from which
  contig or junction becomes the cell's pair, not from failing to assemble chains.

The BCR loss is not diagnosed. The held-out set is now used, so any fix needs new held-out libraries. The dev set had only 2
BCR libraries (2,077 cells); on pbmc10k_b, Singlet already trailed TRUST4 by 0.004.

**Literal dev-rule verdicts, reported alongside the decisions:**
- Clone rescue: REJECT on pbmc2_t ARI; adopted under Addendum 1.
- AIRR-C + `v_open_frac`: REJECT, because pooled pairing does not improve; adopted on V agreement.
- On the held-out set, no claimed ARI lead relies on the Addendum 1 tolerance. The one ARI loss, pbmc3_b, is reported as
  a loss.

### Speed (16 threads, held-out set)

| tool | per library | peak memory | total for 11 libraries |
|---|---|---|---|
| Singlet | 0:37-8:34 wall clock | 2.4-24.9 GB | about 58 min |
| TRUST4 | 4.8 min-1.43 h job time | – | about 8.9 h |

Singlet is about 9 times faster.

## Not done

- The held-out BCR pairing loss and the ngpbmc3_b wrong lineage are not diagnosed. Candidates: the junction choice between
  near-equal heavy or light contigs, J calls, and clone rescue on BCR. This needs new development BCR libraries.
- No held-out ablation with the old defaults, to separate the frozen changes from the baseline.
- No C++ port.
