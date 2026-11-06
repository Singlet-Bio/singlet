#!/bin/bash
# Submits the demux-truth2 chain. Login node: sbatch only. Run from the directory that holds the .sbatch files.
#   setup (clone, build, install Vireo, check souporcell)      stage x4 (download) -> run x4 (pipeline)
#   qc and build x11 wait for setup and the runs; engine, vireo and souporcell x11 wait for build; summary waits for all.
set -euo pipefail
sub() { sbatch --parsable "$@"; }
setup=$(sub setup.sbatch)
stage=$(sub stage.sbatch)
runs=$(sub --dependency=aftercorr:$stage run_sample.sbatch)
dep="afterok:$setup,afterany:$runs"
qc=$(sub --dependency=$dep qc.sbatch)
build=$(sub --dependency=$dep build.sbatch)
eng=$(sub --dependency=afterany:$build engine.sbatch)
vir=$(sub --dependency=afterany:$build vireo.sbatch)
soup=$(sub --dependency=afterany:$build souporcell.sbatch)
sum=$(sub --dependency=afterany:$eng:$vir:$soup:$qc summarize.sbatch)
echo "setup=$setup stage=$stage run=$runs qc=$qc build=$build engine=$eng vireo=$vir souporcell=$soup summary=$sum"
