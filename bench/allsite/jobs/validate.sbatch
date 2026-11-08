#!/bin/bash
#SBATCH --job-name=allsite-val
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=28
#SBATCH --mem=400G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# D-101 validation on one library (touched libraries only: corpus-engine-touch.md). One library per job; chain libraries with
# --dependency=afterany so only one node is in use. Needs BINDIR (singlet-gpu of the branch), NAME (library id), INPUT (a .1fq, or
# "R1a,R1b:R2a,R2b" FASTQ lists, encoded once to $O/input.1fq), optional TRUTH (1000 Genomes sample id of a one-donor library).
#   1. out/: the CPU engine end to end on the serial pileup (--pipeline --no-parallel-pileup, mtx outputs): the serial pass is the one
#      that collects the records the all-site member is computed from, and the stream path that the campaign takes is serial too
#   2. bam/: a second engine run with --tagged-bam (the tagged BAM needs the parallel sorted-BAM pileup), whose BAM cellsnp-lite reads;
#      same reads and the same aligner, so both runs align the same reads
#   3. cellsnp-lite 1.2.3 mode 1 on that BAM at the engine's all-site sites over out/'s called cells (--minCOUNT 0 --minMAF 0)
#   4. with TRUTH: cellsnp-lite mode 2 (de novo, documented 10x settings --minMAF 0.1 --minCOUNT 20) on the same BAM and cells, then
#      score_genotypes.py against the donor's 1000 Genomes genotypes
# WLFILE: the 10x whitelist file of the chemistry (e.g. $W/pipeline-benchmark/whitelists/737K-april-2014.txt). Engine runs before the
# run-path fix (fix/encode-whitelist) ignore $SINGLET_WHITELIST_DIR and fall back to barcode auto-discovery, which gives each run its own
# barcode set; it is also the --barcodes list, because barcode discovery differs between the serial and parallel paths. --barcodes turns
# cell calling off unless asked, so --cell-calling is passed too (the stored member keeps called cells only). FORMAT defaults to mtx: the cellsnp comparison and PARCHECK read the mtx files (format all writes 1pz only for these members).
# Steps whose output exists are skipped, so a timed-out job can be resubmitted. Outputs (owner-only):
# $W/genotypes/allsite-val/$NAME/{out,bam,cellsnp_sites,cellsnp_denovo}; counts-only JSON in $W/jobs/allsite/results/$NAME/.
set -uo pipefail
: "${BINDIR:?}" "${NAME:?}" "${INPUT:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
E=$W/envs/gpool; PY=$W/envs/pbench/bin/python3; export APPTAINER_BIND=/mnt/projects:/mnt/projects PATH=$W/envs/pbench/bin:$PATH
O=$W/genotypes/allsite-val/$NAME; R=$W/jobs/allsite/results/$NAME; mkdir -p $O $R; chmod 700 $W/genotypes/allsite-val $O
BARE=$W/repos/singlet-py.git; BR=${BR:-feat/allele-outputs}; T=${THREADS:-$SLURM_CPUS_PER_TASK}
SCR=/tmp/$USER-asv-$SLURM_JOB_ID-$NAME; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/asv/$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/asv/$SLURM_JOB_ID bench/allsite | tar -x -C $SCR
echo "node $(hostname) bin $BINDIR commit $(cat $BINDIR/commit-gpu) scripts $(git -C $BARE rev-parse --short=8 refs/asv/$SLURM_JOB_ID) threads $T $(date -Is)" | tee $R/provenance.txt
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists SINGLET_REF_BASE=$REF SINGLET_STALL_TIMEOUT=3600
case "$INPUT" in
  *.1fq) FQ=$INPUT ;;
  *) FQ=$O/input.1fq
     if [ ! -s $FQ ]; then
       cat $(echo "${INPUT%%:*}" | tr , ' ') > $SCR/R1.fastq.gz; cat $(echo "${INPUT##*:}" | tr , ' ') > $SCR/R2.fastq.gz
       $BINDIR/singlet-gpu encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz ${PROTOCOL:+--protocol $PROTOCOL} -o $FQ.tmp > $O/encode.log 2>&1 && mv $FQ.tmp $FQ
       echo "encode $(du -b $FQ 2>/dev/null)"; tail -3 $O/encode.log; rm -f $SCR/R1.fastq.gz $SCR/R2.fastq.gz
     fi ;;
esac
[ -s "$FQ" ] || { echo "no input $FQ"; exit 2; }
engine() {   # $1 out dir, $2 log, rest: extra flags
  local d=$1 log=$2; shift 2; rm -rf $d; mkdir -p $d
  ( cd $SCR && /usr/bin/time -v -o $log.time $BINDIR/singlet-gpu "$FQ" --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $d/ --threads $T --pipeline --output-format ${FORMAT:-mtx} ${WLFILE:+--whitelist $WLFILE --barcodes $WLFILE --cell-calling} "$@" > $log 2>&1 )
  local rc=$?; echo "engine $(basename $d) exit $rc"; grep -h -E "Elapsed|Maximum res" $log.time; return $rc
}
# a tagged-BAM run left in out/ by the first version of this job becomes bam/
[ -f $O/out/possorted.bam ] && [ ! -e $O/bam ] && mv $O/out $O/bam
[ -f $O/out/private/allsite/sites.tsv.gz ] || engine $O/out $O/engine.log --no-parallel-pileup
cp $O/engine.log.time $R/engine.time.txt 2>/dev/null
grep -a -E "^\[allsite\]|\[genotype\]|pileup_path|parallel_pileup|BAM:" $O/engine.log | cut -c1-400 | tee $R/engine_lines.txt
du -b $O/out/private/allsite/* $O/out/private/genotype/*.vcf.gz $O/out/private/snp_*.mtx.gz 2>/dev/null | tee $R/sizes.txt
[ -f $O/out/private/allsite/sites.tsv.gz ] || { echo "no all-site member in out/"; exit 4; }
[ -f $O/bam/possorted.bam ] || engine $O/bam $O/engine_bam.log --tagged-bam $O/bam/possorted.bam --bam-sort-mode disk
# PARCHECK=1: the tagged-BAM run takes the region-parallel pileup; its all-site member must equal the serial run's entry for entry
if [ "${PARCHECK:-0}" = 1 ]; then
  grep -a -E "^\[allsite\]" $O/engine_bam.log | cut -c1-300 | tee $R/parallel_lines.txt
  for f in allele_ad.mtx.gz allele_dp.mtx.gz allele_ad_features.tsv.gz allele_ad_barcodes.tsv.gz sites.tsv.gz; do
    a=$(zcat $O/out/private/allsite/$f 2>/dev/null | md5sum | cut -c1-32); b=$(zcat $O/bam/private/allsite/$f 2>/dev/null | md5sum | cut -c1-32)
    [ -n "$a" ] && [ "$a" = "$b" ] && [ -s $O/bam/private/allsite/$f ] && echo "parallel=serial $f" || echo "parallel!=serial $f ($a $b)"
  done | tee $R/parallel_check.txt
fi
BAM=$O/bam/possorted.bam; [ -f $BAM ] || { echo "no tagged BAM"; exit 3; }; [ -f $BAM.bai ] || samtools index -@ $T $BAM
SUF=""; samtools view $BAM chr1 2>/dev/null | head -5000 | grep -q -E "CB:Z:[ACGTN]+-1" && SUF="-1"
awk -F"\t" -v s="$SUF" 'NR>1 && $2=="TRUE"{print $1 s}' $O/out/cell_calls.tsv > $O/barcodes.txt; echo "called cells $(wc -l < $O/barcodes.txt) suffix '$SUF'"
( printf '##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n'
  zcat $O/out/private/allsite/sites.tsv.gz | awk -F"\t" 'NR>1{print $1"\t"$2"\t.\t"$3"\t"$4"\t.\t.\t."}' ) > $O/allsite_sites.vcf
rm -rf $O/cellsnp_sites
/usr/bin/time -v -o $R/cellsnp_sites.time.txt apptainer exec $E/cellsnp-lite.sif cellsnp-lite -s $BAM -b $O/barcodes.txt -O $O/cellsnp_sites -R $O/allsite_sites.vcf \
  -p $T --minMAF 0 --minCOUNT 0 --cellTAG CB --UMItag UB > $O/cellsnp_sites.log 2>&1; echo "cellsnp sites exit $?"
$PY -I $SCR/bench/allsite/compare_cellsnp.py $O/out $O/cellsnp_sites $R/compare_cellsnp.json
if [ -n "${TRUTH:-}" ]; then
  if [ ! -f $O/cellsnp_denovo/cellSNP.tag.AD.mtx ]; then
    rm -rf $O/cellsnp_denovo
    /usr/bin/time -v -o $R/cellsnp_denovo.time.txt apptainer exec $E/cellsnp-lite.sif cellsnp-lite -s $BAM -b $O/barcodes.txt -O $O/cellsnp_denovo \
      -p $T --minMAF 0.1 --minCOUNT 20 --cellTAG CB --UMItag UB --chrom $(seq -s, -f chr%g 1 22) > $O/cellsnp_denovo.log 2>&1; echo "cellsnp denovo exit $?"
  fi
  $PY -I $SCR/bench/allsite/score_genotypes.py $TRUTH $W/data/1kgp_grch38 $O/out $O/cellsnp_denovo $R/genotypes.json $W/envs/pbench/bin/bcftools
fi
grep -h -E "Elapsed|Maximum res" $R/*.time.txt
chmod -R go-rwx $O
echo done $(date -Is)
