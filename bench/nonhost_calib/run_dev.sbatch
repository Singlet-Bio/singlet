#!/bin/bash
#SBATCH --job-name=nh-dev
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Non-host calibration data, one whole node, development libraries only (touch class gate, bench, pilot or atlas; none of class none):
#   positives  ravindra2021_dpi1, ravindra2021_dpi2 (SARS-CoV-2), lcl_gse230239_HG01277, lcl_gse230239_HG00362 (EBV, lymphoblastoid lines)
#   negatives  ravindra2021_mock, pbmc1k_v3, pbmc1k_v2, pbmc5p_v1, hgmm1k_v3, neuron1k_v3, nuclei2k_v2, kang2018_ctrl
# Builds the branch (CPU), fetches the three libraries not yet on disk (first 50 M read pairs), runs the engine with the non-host screen on
# every library, writes $J/runs/<id>/nonhost/{nonhost_candidates.tsv,nonhost_summary.json,...}.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/nhcal; BARE=$W/repos/singlet-py.git; PB=$W/pipeline-benchmark; RG4=$W/release-gate4
REFB=/mnt/projects/debruinz_project/cellarium/reference; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BRANCH=${BRANCH:-feat/nonhost-infection-probability}
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR $J/runs $J/data $J/bin; trap 'rm -rf $SCR' EXIT; NC=$(nproc)
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
# ---- build ----
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/nhcal/head || exit 2
COMMIT=$(git -C $BARE rev-parse refs/nhcal/head); BIN=$J/bin/${COMMIT:0:8}/singlet-cpu; echo "commit $COMMIT"
if [ ! -x $BIN ]; then
  mkdir -p $SCR/src; git -C $BARE archive $COMMIT | tar -x -C $SCR/src; S=$SCR/src
  grep -rl --include=CMakeLists.txt -e "-march=native" $S | xargs -r sed -i 's/-march=native/-march=x86-64-v3/' || true
  grep -q "sam_hdr_destroy(bam_hdr_t" $S/src/star/bam_cat.c || sed -i '/^#include "htslib\/htslib\/sam.h"/a extern "C" void sam_hdr_destroy(bam_hdr_t *h);' $S/src/star/bam_cat.c
  cat > $SCR/build.sh <<EOS
set -euo pipefail
export PATH=$ENV/bin:\$PATH CONDA_PREFIX=$ENV CMAKE_PREFIX_PATH=$ENV PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LD_LIBRARY_PATH=$ENV/lib CMAKE_POLICY_VERSION_MINIMUM=3.5
export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"
cmake -S $S -B $SCR/b -DCMAKE_BUILD_TYPE=Release -DSINGLET_BUILD_PIPELINE=ON -DSINGLET_ENGINE_GPU=OFF -DSINGLET_GIT_COMMIT=$COMMIT > $SCR/c.log 2>&1 || { tail -40 $SCR/c.log; exit 1; }
cmake --build $SCR/b --target singlet -j $NC > $SCR/b.log 2>&1 || { grep -m30 -B3 -A8 "error" $SCR/b.log || tail -40 $SCR/b.log; exit 1; }
mkdir -p $(dirname $BIN); cp $SCR/b/src/pipeline/singlet $BIN
EOS
  apptainer exec -B $W -B $SCR $IMG bash $SCR/build.sh || { echo "[build] FAILED"; exit 3; }
  echo $COMMIT > $(dirname $BIN)/commit
fi
[ -e $J/bin/whitelists ] || ln -s $RG4/bin/whitelists $J/bin/whitelists
echo "[nh] build ok $(date -Is)"
# ---- fetch the libraries not on disk: first 25 M spots of each run, R1 = 28 nt, R2 = the long mate ----
fetch() {  # id run1 [run2]
  id=$1; shift; D=$J/data/$id; [ -s $D/R2.fastq.gz ] && return 0; mkdir -p $D; T=$D/tmp; mkdir -p $T
  for r in "$@"; do
    $W/envs/sratools/bin/fastq-dump -X 25000000 --split-files --gzip -O $T $r > $T/$r.log 2>&1 || echo "[fetch] $id $r failed"
  done
  for f in $T/*_*.fastq.gz; do l=$(zcat $f | head -2 | tail -1 | tr -d '\n' | wc -c); if [ $l -eq 28 ]; then echo $f >> $T/r1.list; elif [ $l -gt 60 ]; then echo $f >> $T/r2.list; fi; done
  sort $T/r1.list | xargs cat > $D/R1.fastq.gz; sort $T/r2.list | xargs cat > $D/R2.fastq.gz; rm -rf $T
  echo "[fetch] $id done $(zcat $D/R1.fastq.gz | head -2 | tail -1 | wc -c) $(date -Is)"
}
fetch ravindra2021_dpi1 SRR13711614 & fetch lcl_gse230239_HG01277 SRR24236331 SRR24236332 & fetch lcl_gse230239_HG00362 SRR24236333 SRR24236334 & wait
OLD=$W/modality-bench/sj_nonhost
cat > $SCR/libs.txt <<EOF
ravindra2021_dpi2 GRCh38-2024-A pair:$OLD/data/dpi2/R1.fastq.gz,$OLD/data/dpi2/R2.fastq.gz
ravindra2021_mock GRCh38-2024-A pair:$OLD/data/mock/R1.fastq.gz,$OLD/data/mock/R2.fastq.gz
ravindra2021_dpi1 GRCh38-2024-A pair:$J/data/ravindra2021_dpi1/R1.fastq.gz,$J/data/ravindra2021_dpi1/R2.fastq.gz
lcl_gse230239_HG01277 GRCh38-2024-A pair:$J/data/lcl_gse230239_HG01277/R1.fastq.gz,$J/data/lcl_gse230239_HG01277/R2.fastq.gz
lcl_gse230239_HG00362 GRCh38-2024-A pair:$J/data/lcl_gse230239_HG00362/R1.fastq.gz,$J/data/lcl_gse230239_HG00362/R2.fastq.gz
pbmc1k_v3 GRCh38-2024-A fq:$PB/data/pbmc1k_v3/gex_fastqs.txt
pbmc1k_v2 GRCh38-2024-A fq:$PB/data/pbmc1k_v2/gex_fastqs.txt
pbmc5p_v1 GRCh38-2024-A fq:$PB/data/pbmc5p_v1/gex_fastqs.txt
hgmm1k_v3 GRCh38-GRCm39-barnyard fq:$PB/data/hgmm1k_v3/gex_fastqs.txt
neuron1k_v3 GRCm39-2024-A fq:$PB/data/neuron1k_v3/gex_fastqs.txt
nuclei2k_v2 GRCm39-2024-A fq:$PB/data/nuclei2k_v2/gex_fastqs.txt
kang2018_ctrl GRCh38-2024-A pair:$W/modality-bench/genotype_pool/fastq/R1.fastq.gz,$W/modality-bench/genotype_pool/fastq/R2.fastq.gz
EOF
run_lib() {
  ID=$1; SP=$2; SRC=$3; O=$J/runs/$ID; S=$SCR/run-$ID
  [ -s $O/exit.txt ] && grep -q "^exit 0" $O/exit.txt && return
  rm -rf $O $S; mkdir -p $O/nonhost $S
  case $SRC in
    fq:*) L=${SRC#fq:}; cat $(grep "_R1_" $L) > $S/R1.fastq.gz; cat $(grep "_R2_" $L) > $S/R2.fastq.gz; R1=$S/R1.fastq.gz; R2=$S/R2.fastq.gz ;;
    pair:*) P=${SRC#pair:}; R1=${P%,*}; R2=${P#*,} ;;
  esac
  $BIN encode --reads $R1 $R2 -o $S/s.1fq > $O/encode.log 2>&1; rm -f $S/R1.fastq.gz $S/R2.fastq.gz
  export SINGLET_REF_BASE=$REFB SINGLET_STALL_TIMEOUT=10800
  (cd $S && /usr/bin/time -v -o $O/time.txt $BIN $S/s.1fq --ref-base $REFB --genome-dir $REFB/$SP/star_2.7.11b --exons $REFB/$SP/genes/genes.gtf --out-prefix $O/ --threads 8 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1); echo "exit $?" > $O/exit.txt
  grep -q "BAM: unsorted STAR stream into the pileup" $O/run.log && echo stream > $O/stream_path.txt || echo NOT-stream > $O/stream_path.txt
  rm -rf $S; echo "[run] $ID $(cat $O/exit.txt) $(date -Is)"
}
export -f run_lib; export J SCR BIN REFB
xargs -P 5 -L 1 bash -c 'run_lib "$@"' _ < $SCR/libs.txt
echo "[nh] runs done $(date -Is)"; ls $J/runs/*/nonhost/nonhost_candidates.tsv | wc -l
