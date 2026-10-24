#!/usr/bin/env bash
# Final Track B measurement run (sequential, one process at a time): codec frontier on all 14 files, then the knob grid.
cd "$(dirname "$0")/.."
B=./bin/track_b_bench6.exe
$B --reps 5 --codec b_rans16,b_rans16_scalar,b_leb_raw,b_leb_z3,b_leb_z19 --out results/track_b_codecs2.tsv corpus/*/exon_counts.1pz corpus/*/intron_counts.1pz > results/track_b_codecs2.log 2>&1
$B --reps 5 --codec b_rans16_c256,b_rans16_c4096,b_rans16_pb11,b_rans16_k0_pb10,b_rans16_k1_pb10,b_rans16_k2_pb10,b_rans16_k3_pb10,b_rans16_k4_pb10,b_rans16_k4_pb9,b_rans16_k2_pb9,b_rans16_k0_pb12 --out results/track_b_variants2.tsv corpus/pbmc_v2_typical/exon_counts.1pz corpus/tumor_3p/exon_counts.1pz corpus/brain_sn_v3_deep/intron_counts.1pz > results/track_b_variants2.log 2>&1
echo finished > results/.run_all.done
