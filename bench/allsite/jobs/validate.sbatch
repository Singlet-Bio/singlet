#!/bin/bash
#SBATCH --job-name=allsite-val
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# D-101 validation on one library (touched libraries only: corpus-engine-touch.md). Needs BINDIR (singlet-gpu of the branch), NAME (library
# id), INPUT (a .1fq, or R1,R1b:R2,R2b FASTQ lists), optional TRUTH (1000 Genomes sample id of a one-donor library), SKIP_ENGINE=1 to reuse.
#   1. the CPU engine end to end (--pipeline, mtx outputs, tagged BAM), so the all-site member and the per-cell VCF come from this build
#   2. cellsnp-lite 1.2.3 mode 1 on that BAM at the engine's all-site sites over the called cells (--minCOUNT 0 --minMAF 0): count agreement
#   3. cellsnp-lite mode 2 (de novo, documented 10x settings --minMAF 0.1 --minCOUNT 20) on the same BAM and cells: the genotype comparator
#   4. with TRUTH: score_genotypes.py against the donor's 1000 Genomes genotypes
# Outputs (owner-only): $W/genotypes/allsite-val/$NAME/{out,cellsnp_sites,cellsnp_denovo} and counts-only JSON in $W/jobs/allsite/results/$NAME/.
set -uo pipefail
: "${BINDIR:?}" "${NAME:?}" "${INPUT:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
E=$W/envs/gpool; export APPTAINER_BIND=/mnt/projects:/mnt/projects PATH=$W/envs/pbench/bin:$PATH
O=$W/genotypes/allsite-val/$NAME; R=$W/jobs/allsite/results/$NAME; mkdir -p $O $R; chmod 700 $W/genotypes/allsite-val $O
BARE=$W/repos/singlet-py.git; BR=${BR:-feat/allele-outputs}
SCR=/tmp/$USER-asv-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
( flock 9; git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/asv/$SLURM_JOB_ID ) 9>$BARE/.gn-fetch.lock
git -C $BARE archive refs/asv/$SLURM_JOB_ID bench/allsite | tar -x -C $SCR
echo "node $(hostname) bin $BINDIR commit $(cat $BINDIR/commit-gpu) scripts $(git -C $BARE rev-parse --short=8 refs/asv/$SLURM_JOB_ID) $(date -Is)" | tee $R/provenance.txt
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists SINGLET_REF_BASE=$REF SINGLET_STALL_TIMEOUT=3600
if [ "${SKIP_ENGINE:-0}" != 1 ]; then
  rm -rf $O/out; mkdir -p $O/out
  case "$INPUT" in *.1fq) IN=("$INPUT") ;; *) IN=(--r1 "${INPUT%%:*}" --r2 "${INPUT##*:}") ;; esac
  ( cd $SCR && /usr/bin/time -v -o $R/engine.time.txt $BINDIR/singlet-gpu "${IN[@]}" --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $O/out/ --threads $SLURM_CPUS_PER_TASK --pipeline --output-format mtx --tagged-bam $O/out/possorted.bam --bam-sort-mode disk > $O/engine.log 2>&1 )
  echo "engine exit $?"; grep -E "Elapsed|Maximum res" $R/engine.time.txt
fi
grep -a -E "^\[allsite\]|\[genotype\]|Collector|gpu_pileup" $O/engine.log | cut -c1-400 | tee $R/engine_lines.txt
du -b $O/out/private/allsite/* $O/out/private/genotype/*.vcf.gz $O/out/private/snp_*.mtx.gz 2>/dev/null | tee $R/sizes.txt
BAM=$O/out/possorted.bam; [ -f $BAM.bai ] || samtools index -@ 16 $BAM
TAGFMT=$(samtools view $BAM chr1 | head -2000 | grep -o -m1 "CB:Z:[ACGT]*[^[:space:]]*" | head -1); SUF=""; case "$TAGFMT" in *-1) SUF="-1";; esac
awk -F"\t" -v s="$SUF" 'NR>1 && $2=="TRUE"{print $1 s}' $O/out/cell_calls.tsv > $O/barcodes.txt; echo "called cells $(wc -l < $O/barcodes.txt)"
zcat $O/out/private/allsite/sites.tsv.gz | awk -F"\t" 'NR>1{print $1"\t"$2"\t.\t"$3"\t"$4}' > $SCR/sites.vcf
( echo -e "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT"; cat $SCR/sites.vcf ) > $O/allsite_sites.vcf
/usr/bin/time -v -o $R/cellsnp_sites.time.txt apptainer exec $E/cellsnp-lite.sif cellsnp-lite -s $BAM -b $O/barcodes.txt -O $O/cellsnp_sites -R $O/allsite_sites.vcf \
  -p 16 --minMAF 0 --minCOUNT 0 --cellTAG CB --UMItag UB > $O/cellsnp_sites.log 2>&1; echo "cellsnp sites exit $?"
/usr/bin/time -v -o $R/cellsnp_denovo.time.txt apptainer exec $E/cellsnp-lite.sif cellsnp-lite -s $BAM -b $O/barcodes.txt -O $O/cellsnp_denovo \
  -p 22 --minMAF 0.1 --minCOUNT 20 --cellTAG CB --UMItag UB --chrom $(seq -s, -f chr%g 1 22) > $O/cellsnp_denovo.log 2>&1; echo "cellsnp denovo exit $?"
grep -h -E "Elapsed|Maximum res" $R/cellsnp_*.time.txt
source $W/src/singlet-agents/main/hpc/clipper/in_runtime.sh
in_runtime python -I $SCR/bench/allsite/compare_cellsnp.py $O/out $O/cellsnp_sites $R/compare_cellsnp.json
[ -n "${TRUTH:-}" ] && in_runtime python -I $SCR/bench/allsite/score_genotypes.py $TRUTH $W/data/1kgp_grch38 $O/out $O/cellsnp_denovo $R/genotypes.json $W/envs/pbench/bin/bcftools
chmod -R go-rwx $O
echo done $(date -Is)
