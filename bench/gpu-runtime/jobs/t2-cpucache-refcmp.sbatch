#!/bin/bash
#SBATCH -A debruinz -p short-gpu --exclude=g050,g051,g052 -c 4 --mem=32G --time=01:30:00 -J t2-cpucache-refcmp -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-cpucache: the CPU-engine outputs a gate run kept under REFD (made by the modified binary) against the baseline's stored references (ref-t2):
# out_equiv.py on out/, the sorted-read hashes, CellReads.stats. usage: NEW=<REFD of the gate run>/T18 [OLD=...ref-t2/T18] sbatch t2-cpucache-refcmp.sbatch
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; H=$W/jobs/gpu-runtime
NEW=${NEW:?}; OLD=${OLD:-$H/ref-t2/T18}; export PATH=$W/envs/pbench/bin:$PATH
for s in pbmc2m pbmc10m pbmc1k_v3 pbmc1k_v2 pbmc1k_cite_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2; do
  echo "=== $s"; head -1 $OLD/$s/DONE | cut -c1-120; head -1 $NEW/$s/DONE | cut -c1-120
  python3 $H/t2-cpucache-out_equiv.py $OLD/$s/out $NEW/$s/out > /tmp/oe-$$.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' /tmp/oe-$$.txt | cut -c1-200 | head -12
  cmp -s $OLD/$s/readhash.txt $NEW/$s/readhash.txt && echo "readhash identical" || { echo "readhash DIFFERS"; diff $OLD/$s/readhash.txt $NEW/$s/readhash.txt; }
  cmp -s <(LC_ALL=C sort $OLD/$s/CellReads.stats) <(LC_ALL=C sort $NEW/$s/CellReads.stats) && echo "CellReads.stats identical (sorted rows)" || echo "CellReads.stats DIFFERS (sorted rows)"
done
