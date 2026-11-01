#!/bin/bash
#SBATCH --job-name=amb-score
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; A=$W/release-gate/ambient
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
for ID in hgmm1k_v3 hgmm1k_v2; do for TAG in ${TAGS:-base cap}; do
O=$A/runs/$ID/singlet-$TAG
cp -n $A/runs/$ID/singlet/time.txt $A/runs/$ID/singlet/encode.time.txt $A/runs/$ID/singlet/node.txt $O/ 2>/dev/null
python -I $A/harness/barnyard_artifacts.py $ID $O $A/work/$ID $A/artifacts-$TAG > $W/cache/art-$ID-$TAG.log 2>&1 || { tail -5 $W/cache/art-$ID-$TAG.log; continue; }
echo "##### $ID $TAG"; python -I $W/jobs/gate2/amb_score.py $A/artifacts-$TAG/ambient-rna $ID 2>&1 | tail -n 40
done; done
