# Idea 4: low-rank manifold prior (generative prior, done honestly)

Status: in progress (2026-10-21/22). Hypothesis H4: a shipped gene-loading model with a per-cell factor vector beats
the static per-gene x depth model by >= 1 bit/nnz on called cells, out of sample, net of side information.

Files: `kernels/idea4_core.h` (integer exp, rANS64, adaptive models), `kernels/idea4.h` (codec),
`harness/idea4_bench.cpp`, `tools/idea4_lib.py` (loading, gene grouping), `tools/idea4_nmf.py` +
`tools/idea4_em.cpp` (multinomial topic model EM, C++ via ctypes, training aid only), `tools/idea4_exp*.py`
(cross-entropy studies), `corpus_idea4/` (9 training samples copied from `corpus_train/`: 5 human PBMC/10x from
GSE165080 x2, GSE285335, GSE244515, GSE268210; 4 mouse brain from GSE249382 x2, GSE245201, GSE254360).
All test studies (GSE142444, GSE244263, GSE152522, GSE145370, GSE285936) are disjoint from the training studies,
so every "out of sample" number below is leave-all-test-studies-out, which is stricter than leave-one-out.

## 0. Setup

- Rows are exon intervals (310,797 human / 386,028 mouse) grouped into genes by the ENSG prefix (38,606 / 78,298
  genes, contiguous row blocks). The model lives at gene level; the interval split is coded separately with a
  static per-gene interval table `q` (training-data interval proportions + length prior).
- Model: multinomial topic model (PLSA = KL-NMF with simplex-normalised columns). `pi_c = Phi @ theta_c`, Phi (G x k)
  column-stochastic, theta_c on the simplex. Trained by EM with a Dirichlet-like smoothing `beta * pi0` on Phi
  (beta = 500 UMIs), on <= 3000 called cells per training sample (14,468 human training cells, 10.0 M gene-level nnz).
- The static "per-gene x depth" baseline in the same framework: `pi_c = pi0` (train-set gene proportions), rate
  `lambda_g = N_c * pi0_g`, i.e. Multinomial(N_c, pi0). This is the model the manifold prior must beat.
- Cross-entropy is computed as the Poisson/multinomial log-likelihood of the gene-level counts plus the static split
  cost, and reported per interval-level nnz of the called cells so it is comparable with every other track.

## 1. Cross-entropy, gene level, out of sample (human, k = 16, unquantised theta, 30 EM iterations per test cell)

Training: 5 human samples from other studies. Test: called cells only. Bits per interval-level nnz.

| sample | cells | nnz/cell | static per-gene x depth | NMF k=16 | gain gross | 16 B/cell side info | gain net |
|---|---|---|---|---|---|---|---|
| pbmc_v2_typical | 5428 | 443 | 7.508 | 7.092 | 0.415 | 0.289 | +0.13 |
| tcell_5p | 12363 | 763 | 7.283 | 6.965 | 0.318 | 0.168 | +0.15 |
| pbmc_v3_lowgenes | 11124 | 145 | 17.943 | 17.380 | 0.564 | 0.885 | -0.32 |
| tumor_3p | 3820 | 37 | 14.426 | 11.731 | 2.695 | 3.459 | -0.76 |

Decomposition of the static cost (pbmc_v2_typical): 4.51 bits/nnz Poisson "constant" part (N log2 e + sum log2 x!),
2.42 bits/nnz interval split given gene totals, 0.58 bits/nnz the gene-level rate term. The split is a large,
model-independent chunk shared by both coders; Idea 3 owns it.

Early reading: the gross gain of a k=16 manifold prior on typical PBMC is 0.3-0.4 bits/nnz, far from the
>= 1 bit/nnz of H4, and the 16 B/cell side info eats most of it. Only the degenerate tumour sample (37 nnz/cell,
ESCC tumour cell types absent from a PBMC-trained pi0) gains > 1 bit/nnz gross, and there the side info costs more
than the gain. A 1 byte/cell codebook of theta vectors is needed for sparse cells. Numbers for k = 8 / 32, in-sample
upper bounds, calibrated (non-Poisson) presence/count tables, mouse brain, and cross-domain follow below as they land.

## 2. Stage log

- exp1 (`tools/idea4_exp1.py`): interval-level static Poisson, pi from 5 human training samples, called cells:
  pbmc_v2_typical 7.54, tcell_5p 7.28, tumor_3p 14.2, pbmc_v3_lowgenes 17.96 bits/nnz (pseudo-count alpha has
  < 0.1 bits/nnz effect). Interval-level static == gene-level static + split (7.508 above) to within 0.03 bits.
- exp2 (`tools/idea4_exp2.py`): table in section 1. Training took 330 s single-threaded in numpy; replaced by the
  C++ EM sweep (`tools/idea4_em.cpp`) for the k sweep.
- `kernels/idea4_core.h` unit test (`harness/idea4_unit.cpp`): integer exp(-D) table max abs error 9.3e-10 (one
  Q30 ulp), monotone; rANS64 roundtrip of 1 M symbols with random 1-28-bit scales OK.
