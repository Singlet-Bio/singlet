#!/bin/bash
# Login node: only sbatch. Non-host chain on the pinned binary.
W=/mnt/projects/debruinz_project/debruinz/singlet; H=$W/release-gate/src/perf/bench/public/harness
prep=$(sbatch --parsable $W/release-gate/jobs/nh_prep.sbatch)
ids=()
for L in gse166766_hbec_mock gse166766_hbec_dpi2 pbmc1k_v3; do
  s=$(sbatch --parsable --dependency=afterok:$prep $H/non_host/10_singlet.sbatch $L)   # script path exists once prep ran: submit after prep below
  ids+=($s)
done
echo $prep ${ids[*]}
