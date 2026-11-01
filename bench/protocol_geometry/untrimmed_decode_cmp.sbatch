#!/bin/bash
#SBATCH --job-name=s14-untrimmed-cmp
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# The auto-detected .1fq (NEW, env whitelist) and the --protocol 10x-3p-v3 one (OLD, manual) of untrimmed_gate.sbatch differ in size and in 814 bytes starting at the
# header (byte 9). Are only the header metadata different, or the reads? Compare the DECODED R2 streams, the file header strings, and the sizes. Env: D (gate output dir), DEC (a binary with `decode`).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; D=${D:?}; DEC=${DEC:-$W/pipeline-benchmark/bin/singlet-w3}
export PATH=$W/envs/pbench/bin:$PATH
ls -l $D/new_env.1fq $D/old_manual.1fq
for f in new_env old_manual; do
  echo "== $f: header strings"; head -c 4096 $D/$f.1fq | strings -n 5 | head -12 | cut -c1-300
  echo "== $f: decoded R2 (cDNA) sha256 / records"; $DEC decode $D/$f.1fq --r2 /dev/stdout 2>$D/$f.dec.err | tee >(sha256sum | cut -c1-16 > $D/$f.r2.sha) | wc -c
  cat $D/$f.r2.sha; tail -2 $D/$f.dec.err | cut -c1-200
done
echo "== decoded barcode/UMI streams if the tool has them"; $DEC decode --help 2>&1 | head -12 | cut -c1-200
