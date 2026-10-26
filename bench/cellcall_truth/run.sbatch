#!/bin/bash
#SBATCH --job-name=cct
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Independent-truth cell-calling study (bench/cellcall_truth). Scripts are put into $W/jobs/cellcall_truth/. MODE:
#   fetch        Cell Ranger published filtered barcodes (10x) and the Kang GEO lane barcodes -> $S/cellranger/<ds>.barcodes.tsv.gz
#   cv2          cv2_stored-2.1 on each dataset's Singlet run (stored counts) -> $S/cv2/<ds>.tsv
#   starsolo     array 0-1 (hgmm5k_v3, hgmm1k_v2): STARsolo 2.7.11b EmptyDrops_CR, Cell Ranger 4+ emulation -> $S/starsolo/<ds>
#   cellbender   array 0-1, GPU (override partition): CellBender 0.3.2 on that STARsolo raw matrix -> $S/cellbender/<ds>
#   kang_pileup  cellsnp-lite on every Kang b2 ctrl barcode with >= 10 UMIs (reference SNP sites) -> $S/kang_b2ctrl/cellsnp
#   truth        array 0-6: genotruth.py / barnyard.py -> $S/truth/<ds>.truth.{tsv.gz,json}
#   score        score.py -> $S/score.json
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BP=$W/bench-public
J=$W/jobs/cellcall_truth; S=$W/state/cellcall_truth; G=$W/modality-bench/genotype_pool
E=$W/envs/gpool; PY="$E/venv/bin/python -P"
REF=/mnt/projects/debruinz_project/cellarium/reference
MODE=${MODE:-truth}; TASK=${SLURM_ARRAY_TASK_ID:-0}
SCR=/tmp/$USER-${SLURM_JOB_ID:-x}-$TASK; mkdir -p $SCR $S; trap 'rm -rf $SCR' EXIT
DSS=(kang_b2ctrl hgmm1k_v3 hgmm5k_v3 hgmm1k_v2 kang_b1A kang_b1B kang_b1C)
declare -A RUN=([kang_b2ctrl]=$PB/runs/kang2018_ctrl/singlet-public [hgmm1k_v3]=$PB/runs/hgmm1k_v3/singlet-public
                [hgmm5k_v3]=$BP/runs/hgmm5k_v3/singlet [hgmm1k_v2]=$BP/runs/hgmm1k_v2/singlet
                [kang_b1A]=$BP/runs/kang2018_b1_a/singlet [kang_b1B]=$BP/runs/kang2018_b1_b/singlet [kang_b1C]=$BP/runs/kang2018_b1_c/singlet)
echo "[cct] mode=$MODE task=$TASK node=$(hostname) $(date -Is)"
case $MODE in
  fetch)
    mkdir -p $S/cellranger; cd $SCR
    for x in "hgmm1k_v3 hgmm_1k_v3" "hgmm5k_v3 hgmm_5k_v3" "hgmm1k_v2 hgmm_1k_v2"; do
      set -- $x
      curl -fsSL -o f.tar.gz https://cf.10xgenomics.com/samples/cell-exp/3.0.0/$2/$2_filtered_feature_bc_matrix.tar.gz
      tar -xzf f.tar.gz; cp filtered_feature_bc_matrix/barcodes.tsv.gz $S/cellranger/$1.barcodes.tsv.gz; rm -rf filtered_feature_bc_matrix f.tar.gz
      echo "$1 $(zcat $S/cellranger/$1.barcodes.tsv.gz | wc -l) Cell Ranger 3.0.0 barcodes"
    done
    U=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2560nnn/GSM2560248/suppl/
    curl -fsSL $U | grep -o 'GSM2560248_[^"<>]*' | sort -u
    curl -fsSL -o $S/cellranger/kang_b2ctrl.barcodes.tsv.gz ${U}GSM2560248_barcodes.tsv.gz && \
      echo "kang_b2ctrl $(zcat $S/cellranger/kang_b2ctrl.barcodes.tsv.gz | wc -l) GEO lane barcodes" || echo "kang GEO barcodes not found"
    # batch 1 lanes A, B, C (GSM2560245-7): saved per GSM, the lane is matched by overlap with the authors' table in fetch_b1 below
    for g in GSM2560245 GSM2560246 GSM2560247; do
      U=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2560nnn/$g/suppl/
      for n in $(curl -fsSL $U | grep -o "${g}_[^\"<>]*barcodes[^\"<>]*" | sort -u); do
        curl -fsSL -o $S/cellranger/$n $U$n && echo "$n $(zcat $S/cellranger/$n | wc -l)"; done
    done ;;
  cv2)
    mkdir -p $S/cv2; export LD_LIBRARY_PATH=$W/envs/pbench/lib
    for ds in "${DSS[@]}"; do
      $PB/bin/cv2_stored-2.1 ${RUN[$ds]} $S/cv2/$ds.tsv 16 > $S/cv2/$ds.params.json
      echo "$ds $(awk -F'\t' 'NR>1{n+=$3}END{print n}' $S/cv2/$ds.tsv) called_v2 (cv2_stored 2.1); run log: $(grep -h 'called_v2:' ${RUN[$ds]}/run.log | head -1)"
    done ;;
  starsolo)
    SD=(hgmm5k_v3 hgmm1k_v2 kang_b1A kang_b1B kang_b1C); ID=${SD[$TASK]}; export PATH=$W/envs/pbench/bin:$PATH
    GEN=GRCh38-GRCm39-barnyard; UMISTART=17
    case $ID in hgmm5k_v3) CBLEN=16; UMILEN=12; WL=$PB/whitelists/3M-february-2018.txt ;;
                hgmm1k_v2) CBLEN=16; UMILEN=10; WL=$PB/whitelists/737K-august-2016.txt ;;
                kang_b1*)  CBLEN=14; UMISTART=15; UMILEN=10; WL=$PB/whitelists/737K-april-2014.txt; GEN=GRCh38-2024-A ;; esac
    D=$BP/data/$ID
    if [ -d $D/fastq ]; then R1=$D/fastq/R1.fastq.gz; R2=$D/fastq/R2.fastq.gz
    else R1=$(grep "_R1_" $D/gex_fastqs.txt | paste -sd,); R2=$(grep "_R2_" $D/gex_fastqs.txt | paste -sd,); fi
    O=$S/starsolo/$ID; rm -rf $O; mkdir -p $O; cd $O
    CMD=(STAR --runThreadN $SLURM_CPUS_PER_TASK --genomeDir $REF/$GEN/star_2.7.11b --readFilesIn $R2 $R1 --readFilesCommand zcat
         --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen $CBLEN --soloUMIstart $UMISTART --soloUMIlen $UMILEN --soloBarcodeReadLength 0
         --soloStrand Forward --outSAMtype None --outTmpDir $SCR/startmp --soloFeatures Gene --soloCellFilter EmptyDrops_CR
         --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR
         --clipAdapterType CellRanger4 --outFilterScoreMin 30)
    echo "${CMD[*]}" > command.txt
    /usr/bin/time -v -o time.txt "${CMD[@]}" > run.log 2>&1
    grep -E "Elapsed \(wall" time.txt; wc -l Solo.out/Gene/filtered/barcodes.tsv ;;
  cellbender)
    SD=(hgmm5k_v3 hgmm1k_v2 kang_b1A kang_b1B kang_b1C); ID=${SD[$TASK]}; RAW=$S/starsolo/$ID/Solo.out/Gene/raw; O=$S/cellbender/$ID; rm -rf $O; mkdir -p $O
    SIF=$W/images/cellbender-0.3.2.sif; export APPTAINER_CACHEDIR=$W/cache/apptainer XDG_CACHE_HOME=$W/cache MPLCONFIGDIR=$W/cache/mpl
    read EXP TOT < <($W/envs/pbench/bin/python -I - "$RAW/matrix.mtx" <<'PY'
import sys, numpy as np, scipy.io
m = scipy.io.mmread(sys.argv[1]).tocsc()
tot = np.sort(np.asarray(m.sum(axis=0)).ravel())[::-1]
cut = np.quantile(tot[:3000], 0.99) / 10.0
exp = int((tot >= cut).sum())
print(exp, max(3 * exp, 10000))
PY
)
    echo "$ID expected-cells=$EXP total-droplets-included=$TOT" | tee $O/params.txt
    mkdir -p $SCR/raw; for f in matrix.mtx barcodes.tsv features.tsv; do gzip -c $RAW/$f > $SCR/raw/$f.gz; done; cd $SCR
    CMD=(cellbender remove-background --cuda --input $SCR/raw --output $SCR/cb.h5 --expected-cells $EXP --total-droplets-included $TOT --fpr 0.01 --epochs 150)
    echo "${CMD[*]}" >> $O/params.txt
    apptainer exec --nv --bind $W --bind /tmp $SIF "${CMD[@]}" > $O/run.log 2>&1 || true
    if [ ! -s $SCR/cb_cell_barcodes.csv ]; then
      CMD+=(--learning-rate 0.00005); echo "retried: ${CMD[*]}" >> $O/params.txt
      apptainer exec --nv --bind $W --bind /tmp $SIF "${CMD[@]}" > $O/run.log 2>&1 || true
    fi
    cp $SCR/cb_cell_barcodes.csv $SCR/cb_metrics.csv $O/ 2>/dev/null || true; wc -l $O/cb_cell_barcodes.csv ;;
  kang_pileup)
    # array index into DSS (0 = kang_b2ctrl: reference-run SNP sites; 4-6 = batch-1 lanes: 1000 Genomes AF > 5% panel, sites with
    # >= 20 UMIs and minor allele fraction >= 0.1 over the universe)
    ds=${DSS[$TASK]}; O=$S/$ds; mkdir -p $O; cd $J
    PYTHONPATH=$J $PY -c "
import common as C
t, miss = C.gather_callers('$ds')
print('universe', len(t), 'missing', miss)
open('$O/barcodes.txt', 'w').writelines(b + '-1\n' for b in t['barcode'])
"
    rm -rf $O/cellsnp
    if [ $ds = kang_b2ctrl ]; then BAM=$G/singlet_k8/possorted.bam; SITES="-R $G/ref_k8/cellsnp/cellSNP.base.vcf.gz --minMAF 0 --minCOUNT 0"
    else BAM=${RUN[$ds]}/possorted.bam; SITES="-R $G/ref/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz --minMAF 0.1 --minCOUNT 20"; fi
    apptainer exec --bind $W $E/cellsnp-lite.sif cellsnp-lite -s $BAM -b $O/barcodes.txt -O $O/cellsnp $SITES \
      -p $SLURM_CPUS_PER_TASK --cellTAG CB --UMItag UB > $O/cellsnp.log 2>&1
    tail -2 $O/cellsnp.log; ls $O/cellsnp ;;
  truth)
    ds=${DSS[$TASK]}; mkdir -p $S/truth; cd $J
    if [[ $ds = kang* ]]; then
      $PY genotruth.py $ds $S/$ds/cellsnp $S/truth
    else
      $PY barnyard.py $ds $S/truth
    fi ;;
  score)
    cd $J; $PY score.py $S/truth $S/score.json ;;
  variants)
    # cellcall.py = python/singlet/v2/cellcall.py (2.1) and kcommon.py = bench/called_v2_truth/common.py, both from origin/main, put
    # next to these scripts
    ds=${DSS[$TASK]}; cd $J; export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_CACHE_DIR=$W/cache/numba-cct
    CELLCALL_DIR=$J $PY variants.py calls $ds $S/truth ;;
  vscore)
    cd $J; ls $S/truth/*.variants.tsv.gz | xargs -n1 basename | sed "s/.variants.tsv.gz//" | xargs $PY variants.py score $S/truth $S/variants.json ;;
esac
echo "[cct] done $(date -Is)"
