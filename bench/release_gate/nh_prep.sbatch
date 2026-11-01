#!/bin/bash
#SBATCH --job-name=rg-nhprep
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Release gate, non-host: archive the 10-26 harness (bench/public-perf) under $RG/src/perf and re-point it at $RG/perf and the pinned binary.
# The Kraken2 databases and the SARS-CoV-2 reference are shared read-only. Competitors (Kraken2, minimap2 oracle) are re-run on the new
# Singlet outputs because they consume its unmapped reads and cell list.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; RG=$W/release-gate; P=$RG/perf; SRC=$RG/src/perf; BARE=$W/repos/singlet-py.git
mkdir -p $P/logs $P/nonhost/cells $SRC $RG/artifacts-perf
ln -sfn $W/bench-public/db $P/db
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/bench/public-perf:refs/rg/perf
git -C $BARE archive refs/rg/perf bench/public | tar -x -C $SRC
H=$SRC/bench/public/harness
sed -i -e "s|^BP=\$WORK/bench-public.*|BP=\$WORK/release-gate/perf|" \
       -e "s|^SRC=.*|SRC=\${BP_SRC:-\$WORK/release-gate/src/perf}|" \
       -e "s|^ART=.*|ART=\$WORK/release-gate/artifacts-perf|" \
       -e "s|^BIN=\$PB/bin/singlet-campaign|BIN=\$WORK/release-gate/bin/d9834b8a/singlet-cpu|" \
       -e "s|^BIN_SHA256=.*|BIN_SHA256=\$(sha256sum \$BIN \| cut -d' ' -f1)|" \
       -e "s|^BIN_COMMIT=.*|BIN_COMMIT=\$(cat \$WORK/release-gate/bin/d9834b8a/commit)|" $H/perf_common.sh
grep -n "^BP=\|^SRC=\|^ART=\|^BIN" $H/perf_common.sh
for f in $H/non_host/*.sbatch $H/collect_and_compute.sbatch; do
  sed -i -e "s|/bench-public/src/singlet|/release-gate/src/perf|g" -e "s|/bench-public/logs/|/release-gate/perf/logs/|g" -e "s|^#SBATCH --partition=short-cpu\$|#SBATCH --partition=short-cpu,scavenger\n#SBATCH --requeue|" $f
done
sed -i "s|\$BP/results|\$WORK/release-gate/results|g" $H/collect_and_compute.sbatch
sed -i 's|^export SINGLET_REF_BASE|export SINGLET_REF_BASE|' $H/non_host/10_singlet.sbatch
grep -n "perf_common\|results" $H/collect_and_compute.sbatch | head
# non_host/10_singlet writes `exit N` into run.log on failure; also assert the stream path
echo 'source $WORK/release-gate/jobs/rg_common.sh; rg_stream_ok $O/run.log | tee $O/stream_path.txt || true' >> $H/non_host/10_singlet.sbatch
ls $H/non_host
