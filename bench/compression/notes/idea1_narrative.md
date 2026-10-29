# Idea 1 + Idea 5: survival-function gap coding with an out-of-sample feature x depth model

Status: complete 2026-10-22 (agent instance 5300). Tests hypothesis H1 of `notes/brainstorm_fable.md`:
*Idea 1 reaches within 0.2 bits/nnz of the per-feature x depth cross-entropy at >= 20 Mnnz/s decode.*

## 0. Verdict and numbers in one place

**H1 is split into two clauses and they come apart.**

- **Size clause: supported, with a large margin.** The coder's payload (everything except the name/kv streams) is within
  **-0.21 ... +0.05 bits/nnz of the cross-entropy of the shipped (8-bit quantised) model on all 14 corpus files**
  (median +0.002; Table 4). Against the *unquantised* cloglog fit the quantised models cost 0.00-0.04 bits/nnz
  (Table 2). The quantisation needed to fit 310k features x 16 buckets in 5 MiB is therefore worth ~0.01 bits/nnz,
  far below the 0.3 bits/nnz at which the brainstorm said the idea would be falsified; the rank-1 form that
  needs only 0.55-0.78 MB (8 bit per feature + 16 numbers) costs ~0.01-0.1 more.
- **Speed clause: not met on this machine.** Best single-thread payload decode of the stage-4 decoder:
  **13.9 Mnnz/s (rank-1 model, median 10.7) and 9.1 Mnnz/s (dense per-bucket table, median 6.8), expressed at a
  3.0 GHz core clock** (Table 3; wall-clock numbers on this machine are meaningless, see section 7). 20 Mnnz/s needs
  ~150 core cycles per non-zero; we measure 215-390 (rank-1) and 330-860 (dense). Extrapolating the cycle counts to
  the 4.8 GHz turbo of the i7-1185G7 gives 16-22 Mnnz/s for the rank-1 model on deep files, i.e. the target is within
  reach of an idle laptop but I could not demonstrate it. Stage 4 made the reference decoder 2-3x faster
  (4 lanes, 3 independent rANS chains, 16-way SIMD tree), after which the decoder is instruction-bound
  (~220 instructions per nnz), not memory-bound.

What actually limits the bits is not the coder but the model: see section 5 (training-set composition,
cross-study drift) and section 6 (a per-file choice among single-sample models recovers 0.3-1.1 bits/nnz at a
cost of ~3 bits per file and beats today's VOCSC+zstd payload on several files).

All 129 harness round trips (`decode(encode(M)) == M` on indptr/indices/data/names/kv) passed.
