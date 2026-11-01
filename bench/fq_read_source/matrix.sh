#!/bin/bash
# Submits the fq-read-source evidence matrix (login node: sbatch only). Usage: bash matrix.sh <PGO binary name>
#   per sample (none = runs downloaded --quality none, ilmn4 = default binned qualities):
#     camp_none         singlet-campaign (prod/campaign), the reference
#     fifo_none_nmiss   this branch, --read-source fifo, R11 reader fix off (SINGLET_TEST_N_MISS_BARCODES): = reference?
#     chunk_none_nmiss  this branch, --read-source chunk, R11 off: = reference?, and core-s vs fifo (same binary)
#     chunk_none        R11 on: the barcode-restore change alone
#     chunk_ilmn4       binned qualities: the quality change alone
#   multi-run sample (GSM6470208): the reference is camp_pooled, singlet-campaign on the orchestrator's decode + re-encode
#     pool (production today); this branch's runs take the two runs directly (--1fq run1,run2), no pool
set -eu
BIN=$1; J=/mnt/projects/debruinz_project/debruinz/singlet/jobs/fq-read-source; cd $J
sub() { sbatch --parsable --export=ALL,"$1" run.sbatch; }
ids=()
for SET in GSM4006843 GSM7901264 GSM6470208 GSM7474576; do
  if [ $SET = GSM6470208 ]; then   # two runs: the campaign binary reads one .1fq, so its reference is the pool
    ids+=( $(sub "SET=$SET,IN=pooled,BIN=singlet-campaign,MODE=-,TAG=camp_pooled") )
  else
    ids+=( $(sub "SET=$SET,IN=none,BIN=singlet-campaign,MODE=-,TAG=camp_none") )
  fi
  ids+=( $(sub "SET=$SET,IN=none,BIN=$BIN,MODE=fifo,TAG=fifo_none_nmiss,SINGLET_TEST_N_MISS_BARCODES=1") )
  ids+=( $(sub "SET=$SET,IN=none,BIN=$BIN,MODE=chunk,TAG=chunk_none_nmiss,SINGLET_TEST_N_MISS_BARCODES=1") )
  ids+=( $(sub "SET=$SET,IN=none,BIN=$BIN,MODE=chunk,TAG=chunk_none") )
  ids+=( $(sub "SET=$SET,IN=ilmn4,BIN=$BIN,MODE=chunk,TAG=chunk_ilmn4") )
done
dep=$(IFS=:; echo "${ids[*]}")
P=""
for SET in GSM4006843 GSM7901264 GSM6470208 GSM7474576; do
  REF=camp_none; [ $SET = GSM6470208 ] && REF=camp_pooled
  P="$P $SET:$REF:fifo_none_nmiss $SET:fifo_none_nmiss:chunk_none_nmiss $SET:chunk_none_nmiss:chunk_none $SET:chunk_none:chunk_ilmn4"
done
c=$(sbatch --parsable --dependency=afterany:$dep --export=ALL,PAIRS="$P" compare.sbatch)
echo "runs ${ids[*]}"; echo "compare $c"
