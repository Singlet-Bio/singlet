#!/bin/bash
#SBATCH --job-name=pb-inspect
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Describe the Singlet STAR BAMs kept from the GRCh38 campaign: tags, UMI length,
# read length, protocol and depth, to pick the double-count panel.
set -uo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
R=/mnt/projects/debruinz_project/cellarium/full_campaign/results
for g in GSM7090360 GSM7090362 GSM7412962 GSM7596594 GSM7596595 GSM7596602 GSM7596603 GSM7983360 GSM9359399; do
  d=$R/$g/out; b=$d/star_Aligned.sortedByCoord.out.bam
  echo "=== $g size=$(stat -c %s $b) files: $(ls $d | tr '\n' ' ' | cut -c1-400)"
  grep -m1 -oE "soloType [A-Za-z_]+|soloUMIlen [0-9]+|soloCBlen [0-9]+" $d/star_Log.out | tr '\n' ' '; echo
  grep -E "Number of input reads|Uniquely mapped reads %|Average input read length" $d/star_Log.final.out | tr -s ' \t' ' '
  samtools view $b 2>/dev/null | head -200000 | awk '{for(i=12;i<=NF;i++){split($i,a,":"); t[a[1]]++; if(a[1]=="UR"||a[1]=="UB") ul[length(a[3])]++}} END{for(k in t) printf "%s=%d ",k,t[k]; printf "| umi_len:"; for(k in ul) printf " %s(%d)",k,ul[k]; print ""}'
  ls $d/*.json 2>/dev/null | head; [ -f $d/summary.json ] && grep -m3 -E "protocol|chemistry" $d/summary.json
done
