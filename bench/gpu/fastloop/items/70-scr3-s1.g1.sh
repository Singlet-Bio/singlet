#!/bin/bash
# PILOT-MINUTES: 57
# fastloop item (bench/gpu/fastloop, design.py): 3 ideas wshrink,stepsauto,screen, tier S1: pbmc2m, one run at a time, 3 runs per cell point (the first is warm-up), whole pipeline.
# 8 cells x 2 blocks, resolution full. Effects, CIs and stages: fastloop.md in the results directory.
W=/mnt/projects/debruinz_project/debruinz/singlet; P2=$W/jobs/gpu-pr2; FL=$W/jobs/gpu-fastloop; N=99105bd8
export BINDIR=$P2/bin2/99105bd8-multi EXPECT_COMMIT=$N REFDIR=$P2/bin2/9560b8db REFCOMMIT=9560b8db ALLOW_NO_EMBED=1 IDENT=0 PINDEV=${PINDEV:-0}
export FL_IDEAS=$FL/ideas.tsv FL_SOLE=1 SAMPLE=pbmc2m
export POINTS="k1t8s1r3@stepsauto+screen#b1 k1t8s1r3@wshrink+stepsauto#b1 k1t8s1r3@screen#b1 k1t8s1r3@wshrink+screen#b1 k1t8s1r3@wshrink+stepsauto+screen#b1 k1t8s1r3@base#b1 k1t8s1r3@wshrink#b1 k1t8s1r3@stepsauto#b1 k1t8s1r3@stepsauto#b2 k1t8s1r3@wshrink#b2 k1t8s1r3@base#b2 k1t8s1r3@wshrink+stepsauto+screen#b2 k1t8s1r3@wshrink+screen#b2 k1t8s1r3@screen#b2 k1t8s1r3@wshrink+stepsauto#b2 k1t8s1r3@stepsauto+screen#b2"
bash $FL/tp_gc.sbatch; rc=$?
RES=$W/jobs/gpu-integ/results/tp-${RUNID:-$SLURM_JOB_ID}
$W/envs/pbench/bin/python3 $FL/analyze.py --res $RES --ideas wshrink,stepsauto,screen --drop-first 1 --metric held --theta 0.03 > $RES/fastloop.md 2>&1
cat $RES/fastloop.md
exit $rc
