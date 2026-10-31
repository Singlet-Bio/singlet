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

## Held-out results

(pending job 795126)
