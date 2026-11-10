#!/bin/bash
#SBATCH --job-name=gsota-probe
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
for b in mc99cpu mc98; do B=$W/pipeline-benchmark/bin/$b; echo "== $b $(sha256sum $B | cut -c1-16)"; $B --version 2>&1 | head -3; done
B=$W/pipeline-benchmark/bin/mc99cpu
$B encode --help 2>&1 | grep -iE "qual|dedup|trim" | head
ls $W/pipeline-benchmark | head -30
cd $W/src/singlet 2>/dev/null && ls | head -3
ls -d $W/bench-public/data/*/ $W/data/* 2>/dev/null | head -40
ls $W/bench-public/data/kang_b2stim/fastq; ls -la $W/bench-public/data/kang_b2stim/*.1fq
