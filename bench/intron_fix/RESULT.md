# Intron rule default and the mouse mt-gene drop (2026-10-29)

Real data only: pbmc1k v3/v2/CITE/5', nuclei2k, hgmm1k, neuron1k, scored against STARsolo Velocyto and alevin-fry USA
(`sweep_eval.py`, bench/public-counts metrics). Binary: `fix/intron-rule-default` (non-PGO), Clipper jobs 792265 (sweep2),
792322 (default), 792358 (scoring). Raw results: `results/sweep2.json`, `results/default_389.json`.

## Default: SINGLET_INTRON_RULES = 389 = 133 + 256
133 = kUnique (1) + kNoSplicedIntron (4) + kUmiGeneCR (128): a multimapped read gives no intronic evidence, a spliced read
gives none, a UMI seen in several genes of a cell keeps the gene with most reads. It matched or beat every other
combination of sweep2 (pass counts of 9 / 9 / 9 / 11 / 9 metrics, pbmc v3 / nuclei / hgmm / CITE / v2):

| dataset | public (rules 0) | rules 133 | reference agreement (STARsolo Velocyto vs alevin-fry) |
|---|---|---|---|
| pbmc v3 unspliced gene r | 0.949 | 0.968 | 0.964 |
| pbmc CITE | 0.949 | 0.967 | 0.964 |
| hgmm | 0.971 | 0.979 | 0.974 |
| pbmc v2 | 0.832 | 0.9185 | 0.9189 |
| nuclei | 0.931 | 0.958 | 0.964 |
| barnyard U cross-species | 0.474% | 0.302% | STARsolo 0.271% |

The 0.96 target is met wherever the two references themselves agree at 0.96 or more; on pbmc v2 the references agree at 0.919
and Singlet is at 0.9185. The remaining failing check everywhere is `umi_ratio_distance.unspliced`, a dedup-policy
difference (STARsolo does not correct intronic UMIs); nuclei2k still fails unspliced gene r by 0.006.

## Mouse mt genes: the loss is a cross-worker NUMT multimapper, not STAR
neuron1k: about 1.08M reads in MT:7927-8607 are NH=2 ties between the chr1:24.65 Mb NUMT (annotated Atp6-ps, Gm28437, ...)
and chrM; STAR picks the NUMT as primary, chrM as the secondary. `run_parallel` splits the two records over different
workers, the cross-worker merge sees two different genes and drops the read. Rule 256 (kMtPrefer) counts such reads for the
chrM gene (a read tied between chrM and a nuclear copy is mitochondrial: hundreds of copies per cell vs one).

| neuron1k molecules | mt-Atp6 | mt-Co3 | mt-Co2 | mt-Co1 | mt-Cytb |
|---|---|---|---|---|---|
| Singlet public | 16 | 4 | 2 | 14,148 | 178,490 |
| STARsolo Gene | 3,978 | 1,438 | 378 | 13,965 | 182,671 |
| rule 256 | 325,228 | 294,602 | 202,898 | 157,770 | 178,244 |

The corrected levels are of the order of mt-Cytb and mt-Nd4, as expected for polycistronic mt transcripts. 3.09M reads
(neuron1k) and 4.31M reads (nuclei2k) are rescued; human datasets change by 18 to 527 reads. Not independently scored against
long-read truth (none exists for mouse brain here).
