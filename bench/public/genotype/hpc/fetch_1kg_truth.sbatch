#!/bin/bash
#SBATCH --job-name=bpg-1kg
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:55:00
#SBATCH --array=1-23%8
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# PRIVATE truth genotypes of four 1000 Genomes individuals at the shared SNP panel, from the 1000 Genomes high-coverage
# GRCh38 phased callset (3,202 samples, 20220422 release): HG00362 and HG01277 (lcl_gse230239), NA12878 = GM12878 and
# NA18502 = GM18502 (osorio2019_lcl_mix). Output stays under $WORK/bench-public/private (mode 700): genotypes are never
# published; only concordance counts leave the machine.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
C=$SLURM_ARRAY_TASK_ID; [ "$C" = 23 ] && C=X
B=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV
PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
O=$W/bench-public/private/truth_gt; mkdir -p $O; chmod 700 $W/bench-public/private $O
F=1kGP_high_coverage_Illumina.chr$C.filtered.SNV_INDEL_SV_phased_panel.vcf.gz
# sample columns (1-based) in the VCF header: HG00362 HG01277 NA12878 NA18502
COLS=$(curl -s $B/$F | zcat 2>/dev/null | head -n 400 | grep -m1 "^#CHROM" | tr '\t' '\n' | grep -n -E "^(HG00362|HG01277|NA12878|NA18502)$" | cut -d: -f1 | tr '\n' ',' | sed 's/,$//') || true
[ -n "$COLS" ] || COLS=176,534,2342,2358
echo "chr$C sample columns $COLS"
curl -s --retry 5 $B/$F | zcat | grep -v "^##" | cut -f1,2,4,5,$COLS | awk -F'\t' -v OFS='\t' -v chr=chr$C '
  BEGIN{ cmd="zcat '"$PANEL"'"; while ((cmd | getline l) > 0) { if (l ~ /^#/) continue; split(l, a, "\t"); if (a[1]==chr) P[a[2]"\t"a[4]"\t"a[5]]=1 } close(cmd) }
  /^#CHROM/ { print "chrom","pos","ref","alt","HG00362","HG01277","NA12878","NA18502"; next }
  { k=$2"\t"$3"\t"$4; if (k in P) { print ($1 ~ /^chr/ ? $1 : "chr"$1),$2,$3,$4,substr($5,1,3),substr($6,1,3),substr($7,1,3),substr($8,1,3) } }' \
  | gzip > $O/chr$C.tsv.gz
chmod 600 $O/chr$C.tsv.gz
echo "chr$C sites: $(zcat $O/chr$C.tsv.gz | wc -l)"
