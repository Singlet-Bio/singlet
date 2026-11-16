#!/bin/bash
# Count-parity check of a candidate engine build against the v1.3.9 CPU references (validation-engine/v139/ref3), on CPU nodes only:
# build the branch (both binaries, with ctest), run the 8 gate samples on the CPU engine as the gate does (SINGLET_HLA=defer, T=18),
# then compare every published file with the v1.3.9 reference (refcmp.sbatch: byte comparison and field check, same allowances as
# the v1.4.0 gate). Chain: build -> gate (afterok) -> refcmp (afterany). Run once; do not poll.
#   BR=fix/donor-single-split NAME=donorfix bash submit_count_parity.sh
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/v140; D=$W/jobs/donorfix; R=$W/results/validation-engine/donorfix
: "${BR:?}" "${NAME:?}" "${C8:?first 8 characters of the commit to build}"; mkdir -p $R
sed -e 's#BIN=$BINDIR/singlet-gpu; ENGARG="--engine gpu"#BIN=$BINDIR/singlet-cpu; ENGARG="--engine gpu"#' \
    -e 's#^RES0=.*#RES0=$W/results/validation-engine/donorfix/gate-$SLURM_JOB_ID; mkdir -p $RES0 $REFD#' $J/gate.sbatch > $D/gate_cpu.sbatch
grep -c 'singlet-cpu' $D/gate_cpu.sbatch
b=$(sbatch -A debruinz --parsable --export=ALL,BR=$BR,NAME=$NAME,LAYOUT=campaign,SKIP_TAG_CHECK=1 $J/build.sbatch)
echo "build $b"
E="SINGLET_HLA=defer,SINGLET_ENGINE_GPU_MIN_CC=70,KEEPOUT=1,T=18"
S="pbmc2m pbmc10m pbmc1k_v3 pbmc1k_v2 pbmc1k_cite_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2"
g=$(sbatch -A debruinz --parsable -p short-cpu --gres=none -c 18 --mem=200G --time=03:59:00 --job-name=donorfix-gate --dependency=afterok:$b \
   --export=ALL,$E,REFRESH_REF=1,MODES=cpu,PAIRS=cpu:cpu,SAMPLES="$S",BINDIR=$W/builds/$C8/bin,REFD=$R/ref,BRANCH=release/v1.4.0 $D/gate_cpu.sbatch)
echo "gate $g"
c=$(sbatch -A debruinz --parsable --dependency=afterany:$g --job-name=donorfix-refcmp -o $W/logs/donorfix-refcmp-%j.out \
   --export=ALL,XALLOW="pileup_stats.json:/version pileup_stats.json:/provenance/engine_version provenance.json:/singlet_version summary.json:/singlet_version summary.json:/git_sha private/hla/pending/manifest.json:/seconds_prefilter",RA=$W/results/validation-engine/v139/ref3,RB=$R/ref,OUT=$R/refcmp.txt $J/refcmp.sbatch)
echo "refcmp $c"
