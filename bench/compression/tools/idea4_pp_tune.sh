#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export IDEA4_MODEL_STATIC=results/idea4_pp_model_F142444_k0.bin IDEA4_ADAPT=1 IDEA4_KC=50000
for kgkq in "50 3" "500 3" "200 1" "200 10" "1000 3" "100 2"; do
  set -- $kgkq
  export IDEA4_KG=$1 IDEA4_KQ=$2 IDEA4_TAG=kg$1_kq$2
  bin/idea4_pp_bench.exe --calls --no-file-check --reps 1 --codec idea4_static_$IDEA4_TAG --out results/idea4_pp_tune.tsv corpus/pbmc_v2_typical/exon_counts.1pz | grep -E "payload|FAIL" | cut -c1-260
done
