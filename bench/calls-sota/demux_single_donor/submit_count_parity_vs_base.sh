#!/bin/bash
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/v140; D=$W/jobs/donorfix; R=$W/results/validation-engine/donorfix
b=$(sbatch -A debruinz --parsable --export=ALL,BR=fix/donor-single-split-base,NAME=donorfix-base,LAYOUT=campaign,SKIP_TAG_CHECK=1,TESTS=0 $J/build.sbatch); echo build $b
E="SINGLET_HLA=defer,SINGLET_ENGINE_GPU_MIN_CC=70,KEEPOUT=1,T=18"
S="pbmc2m pbmc10m pbmc1k_v3 pbmc1k_v2 pbmc1k_cite_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2"
g=$(sbatch -A debruinz --parsable -p short-cpu --gres=none -c 18 --mem=200G --time=03:59:00 --job-name=donorfix-gate-base --dependency=afterok:$b \
  --export=ALL,$E,REFRESH_REF=1,MODES=cpu,PAIRS=cpu:cpu,SAMPLES="$S",BINDIR=$W/builds/6c559c88/bin,REFD=$R/ref-base,BRANCH=release/v1.4.0 $D/gate_cpu.sbatch); echo gate $g
c=$(sbatch -A debruinz --parsable --dependency=afterany:$g --job-name=donorfix-refcmp2 -o $W/logs/donorfix-refcmp2-%j.out \
  --export=ALL,XALLOW="pileup_stats.json:/version pileup_stats.json:/provenance/engine_version provenance.json:/singlet_version summary.json:/singlet_version summary.json:/git_sha private/hla/pending/manifest.json:/seconds_prefilter",RA=$R/ref-base,RB=$R/ref,OUT=$R/refcmp-vs-base.txt $J/refcmp.sbatch); echo refcmp $c
