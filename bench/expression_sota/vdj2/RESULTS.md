# vdj2: status (interim; Clipper bridge dropped before the held-out run)

Pre-registration: `PREREG.md` (commit 4ca0993b, Addendum 1 after it). **The fresh held-out set has not been run.** Nothing below
uses it except the denominators in PREREG.md.

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

## Changes (python/singlet/vdj, all off by default so far)

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

## Not done

* Dev scoring of S149B / S150B / S135B (TRUST4 job 794913 was running), the full re-assembly with
  `ambient_rescue=clone, v_open_frac=0.5` (job 794973, queued), freezing the defaults, and the single held-out run
  (`submit_heldout.sh`, SHA-pinned).
