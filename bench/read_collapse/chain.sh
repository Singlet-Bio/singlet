#!/bin/bash
# Login-node submitter (sbatch only). Usage: chain.sh [test]
# Full: map array (8 samples) -> cross -> summary, each step afterany so the summary always runs.
W=/mnt/projects/debruinz_project/debruinz/singlet; RC=$W/jobs/read-collapse; cd $RC
if [ "${1:-}" = test ]; then
  RES=$RC/results/test
  J1=$(sbatch --parsable -A debruinz -p debug-cpu --array=1 --cpus-per-task=8 --mem=64G --time=00:29:00 --export=ALL,TEST=1,T=8,RES=$RES map_extract.sbatch)
  J2=$(sbatch --parsable -A debruinz -p debug-cpu --dependency=afterany:$J1 --cpus-per-task=8 --mem=64G --time=00:29:00 --export=ALL,RES=$RES,SAMPLES_TSV=$RC/test_samples.tsv cross.sbatch)
  J3=$(sbatch --parsable -A debruinz -p debug-cpu --dependency=afterany:$J2 --time=00:10:00 --export=ALL,RES=$RES summary.sbatch)
else
  RES=$RC/results/overnight; mkdir -p $RES
  J1=$(sbatch --parsable --export=ALL,RES=$RES map_extract.sbatch)
  J2=$(sbatch --parsable --dependency=afterany:$J1 --export=ALL,RES=$RES cross.sbatch)
  J3=$(sbatch --parsable --dependency=afterany:$J2 --export=ALL,RES=$RES summary.sbatch)
fi
echo "map=$J1 cross=$J2 summary=$J3"
