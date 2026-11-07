#!/bin/bash
# S15 store (runs as a Slurm job): clear the incomplete datasets of the 11-06 harvest (partial index rows, leftovers) and resubmit them together with every `sra|url|bam` row not queued yet.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; S=$W/benchmark-store; mkdir -p $S/_stale
for id in kang2018_b1_b kang2018_b1_c kang2018_ctrl kang2018_stim norman2019_gem1 pbmc3k_v1 pbmc6k_v1 hek293t_v1 jurkat_v1 frozen_pbmc_bc_v1 pbmc4k_v2 pbmc10k_v3; do
  [ -d $S/$id ] && mv $S/$id $S/_stale/$id.$(date +%s)
  grep -v -P "^$id\t" $S/index.tsv > $S/i.tmp && mv $S/i.tmp $S/index.tsv
  grep -v -P "^$id\t" $D/storeids.tsv > $D/s.tmp && mv $D/s.tmp $D/storeids.tsv
  rm -rf $W/pilot/$id/fq $W/pilot/$id/bamdl $W/pilot/$id/tar $W/pilot/$id/tar.dl $W/pilot/$id/starsolo_bam $W/pilot/$id/starsolo $W/pilot/$id/sra
done
cd $D; THROTTLE=5 bash submit_store.sh
