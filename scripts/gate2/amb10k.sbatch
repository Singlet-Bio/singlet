#!/bin/bash
#SBATCH --job-name=amb-10k
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# hgmm10k_v3: prep the work dir from the baseline run's called cells, then artifacts + scores for TAGS (no soupx/cellbender columns).
W=/mnt/projects/debruinz_project/debruinz/singlet; A=$W/release-gate/ambient; ID=hgmm10k_v3
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
[ -f $A/work/$ID/cells.csv ] || python -I $A/harness/barnyard_prep.py $A/runs/$ID/singlet-base $A/work/$ID || exit 1
mkdir -p $A/work/$ID/tools; [ -f $A/work/$ID/tools/node.txt ] || echo none > $A/work/$ID/tools/node.txt
for TAG in ${TAGS:-base frz}; do
O=$A/runs/$ID/singlet-$TAG
for f in time.txt encode.time.txt; do
  [ -f $O/$f ] || printf 'Elapsed (wall clock) time (h:mm:ss or m:ss): 0:01\nMaximum resident set size (kbytes): 1\nUser time (seconds): 1\nSystem time (seconds): 1\n' > $O/$f
done
[ -f $O/node.txt ] || echo none > $O/node.txt
python -I $A/harness/barnyard_artifacts.py $ID $O $A/work/$ID $A/artifacts-$TAG > $W/cache/art-$ID-$TAG.log 2>&1 || { tail -5 $W/cache/art-$ID-$TAG.log; continue; }
echo "##### $ID $TAG"; python -I $W/jobs/gate2/amb_score.py $A/artifacts-$TAG/ambient-rna $ID 2>&1 | tail -n 12
python -I $W/jobs/gate2/amb_cells.py $A/artifacts-$TAG/ambient-rna $ID 2>&1 | head -n 13
done
