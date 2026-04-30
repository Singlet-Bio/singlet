# Reproducibility Notebooks

Every released feature has a notebook here. Each notebook follows the structure in [`../../state/website-contract.md`](../../state/website-contract.md) § G.3:

1. Overview
2. Setup
3. Run singlet-gpu
4. Run reference tool
5. Formal equivalence (correlation plots + metrics table)
6. Performance benchmark (3 scales, error bars from 3+ runs)
7. Biological validation
8. Conclusion

**Real data only**. Planted-signal tests are useful for unit tests but never sufficient for these notebooks. GSM4037629 is the minimum sample.

## Index

(populated as features reach `documented` state)

- _none yet — backfill cycle pending_

## Why notebooks matter

The notebooks are the public proof that singlet-gpu produces results equivalent to Scanpy / rapids-singlecell / Seurat / scran on real data. A user must be able to see, with their own eyes, that `r ≥ 0.9999` for deterministic operations and `r ≥ 0.999` for stochastic operations.

Notebooks render at https://singlet.bio/notebooks. Each notebook also has a "Run on Colab" link and a "View on GitHub" link.
