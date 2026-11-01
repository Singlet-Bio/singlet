# BCR pairing: status and leftovers (2026-11-01)

## Status
BCR is not at SOTA. Nothing is frozen. The spent 11-library held-out set is untouched. The new held-out set (pbmc22_b, cd19_b, b1k_b) has not been run.

Held-out result (single run, frozen rules): TCR passes; BCR pairing and wrong lineage fail vs TRUST4 / Cell Ranger. Speed and contig-F1 lead kept.

Dev (bcrdev: melv11_b, nsclc22_b, chx10k_b, pbmc2_b, pbmc10k_b; scoring partly pending):
- Baseline pairing vs TRUST4, pooled over 4 libraries: -0.021.
- IG rescue (`ig_top_min_umis`): pooled gap closes to about +0.009, but hurts nsclc22_b ARI (not covered by Addendum 1).
- `ig_lone_spread=5`: wrong lineage 91 -> 44.
- First-Cys reference for CR 4.x (double-Cys CDR3 anchor): melv11_b +0.019 pairing, +0.010 F1.
- Mechanisms: cross-filter removal of light chains in plasma-cell clones; CR 4.x double-Cys anchor; lone-chain wrong lineage.

## Leftovers for Thursday
BCR:
1. Collect dev scoring, decide rules on dev (literal dev-rule verdicts next to the ARI addendum).
2. Pre-register the first-Cys mapping for CR 4.x; freeze; record the SHA.
3. Run the new held-out set once for Singlet and TRUST4. Do not touch the spent set.
4. Ask why the first-Cys fix would need the full contig sequence in the member (it is a reference change).
5. Reword #143's description: bench, results, docs only; the fix is delivered by #147. Then send S5 the merged BCR fix SHA.

Jobs running or finished on Clipper (output under `$WORK/expression-sota/vdj2`, `runs/` and results): 796448, 796451 (dev scoring), 796537 (`ig_top` variants), 796566 (pair-based rescue score), 796567 (`ari_check`), 796490 (chx10k_b TRUST4).

HTO (after BCR): doublet-miss fix on Howitt's dev split; pre-register I3 from GSE256089 after a vireo per-tag purity check (S15's lane A1 is in its pilot); S5 gate-2 HTO rows (7 failing, `hto_rg.sbatch`), say plainly if match_or_beat is unreachable.
