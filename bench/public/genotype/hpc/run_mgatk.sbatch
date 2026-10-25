#!/bin/bash
#SBATCH --job-name=bpg-mgatk
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_mgatk.sbatch <pool>
# Competitor / reference for mtDNA: mgatk 0.9.0 (bcall) on the chrM reads of Singlet's tagged BAM, restricted to the
# same called cells, barcode tag CB, UMI tag UB, genome rCRS, all other options at their defaults.
set -u
ulimit -n $(ulimit -Hn) 2>/dev/null; echo "open-files limit soft $(ulimit -Sn) hard $(ulimit -Hn)"
POOL=$1
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/mtl/bin:$PATH; export PYTHONPATH=$W/envs/mgatk_extra
S=$W/bench-public/runs/$POOL/singlet_auto; O=$W/bench-public/runs/$POOL/mgatk; mkdir -p $O; cd $O
cp $W/bench-public/runs/$POOL/vireo/barcodes.txt barcodes.txt 2>/dev/null || true
if [ ! -s barcodes.txt ]; then
  TAGFMT=$(samtools view $S/possorted.bam chr1 | head -2000 | grep -o -m1 "CB:Z:[ACGT]*[^[:space:]]*" | head -1)
  SUF=""; case "$TAGFMT" in *-1) SUF="-1";; esac
  awk -F"\t" -v s="$SUF" 'NR>1 && $2=="TRUE"{print $1 s}' $S/cell_calls.tsv > barcodes.txt
fi
wc -l barcodes.txt
[ -e $S/possorted.bam.bai ] || samtools index -@ 8 $S/possorted.bam
samtools view -@ 8 -b $S/possorted.bam chrM > chrM.bam && samtools index chrM.bam
samtools view -c chrM.bam
/usr/bin/time -v -o mgatk.time.txt mgatk bcall -i chrM.bam -n $POOL -o out -c 8 -bt CB -b barcodes.txt -ub UB -g rCRS > mgatk.log 2>&1
grep -E "Elapsed|Maximum res" mgatk.time.txt; tail -15 mgatk.log; ls -la out out/final 2>/dev/null | head -30
