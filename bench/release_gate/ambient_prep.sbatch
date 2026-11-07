#!/bin/bash
#SBATCH --job-name=rg-ambprep
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Release gate, ambient-rna: archive the 10-26 harness (bench/public-ambient-doublets) and point it at $RG/ambient and the pinned binary.
# The competitors (SoupX, DecontX, CellBender) consume Singlet's cells and raw matrix, so they are re-run on the new outputs.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; RG=$W/release-gate; A=$RG/ambient; H=$A/harness; BARE=$W/repos/singlet-py.git
mkdir -p $A/runs $A/work $A/artifacts $A/data $H
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/bench/public-ambient-doublets:refs/rg/ambient
git -C $BARE archive refs/rg/ambient bench/public | tar -x -C $A
cp $A/bench/public/capabilities/harness/{barnyard_artifacts.py,barnyard_prep.py,barnyard_tools.R,barnyard_tools.sbatch,cellbender.sbatch,artifacts.sbatch,scrublet_run.py} $H/
cp $A/bench/public/capabilities/clipper/run_singlet_10x.sbatch $H/
cd $H
sed -i -e "s|BP=\$W/bench-public|BP=\$W/release-gate/ambient|" -e "s|H=\$W/jobs/bp/harness|H=\$W/release-gate/ambient/harness|" *.sbatch
# run_singlet_10x: pinned binary (+ engine args), whitelist/stream assertion, data from the old tree
sed -i -e '/^BIN=\$PB\/bin\/singlet-campaign/c\source $W/release-gate/jobs/rg_common.sh; rg_check_bin' \
       -e 's|--raw-matrix > \$O/run.log|--raw-matrix "${ENGINE_ARGS[@]}" > $O/run.log|' run_singlet_10x.sbatch
echo 'rg_stream_ok $O/run.log | tee $O/stream_path.txt' >> run_singlet_10x.sbatch
sed -i 's|^#SBATCH --job-name=bp-|#SBATCH --job-name=rg-|' *.sbatch
grep -n "BP=\|BIN\|rg_" *.sbatch | cut -c1-160
