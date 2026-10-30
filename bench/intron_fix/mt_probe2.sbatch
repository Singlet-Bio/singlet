#!/bin/bash
#SBATCH --job-name=if-mtprobe2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCm39-2024-A
O=$PB/runs/neuron1k_v3/singlet-if-keepbam
B=$O/star_Aligned.sortedByCoord.out.bam
for g in Atp6-ps Gm28437 Gm28661 Gm28438 mt-Atp6 mt-Co3; do echo "## $g"; grep "gene_name \"$g\"" $REF/genes/genes.gtf | awk -F'\t' '{print $1,$3,$4,$5,$7}' | sort -u | head -8; grep -m1 "gene_name \"$g\"" $REF/genes/genes.gtf | grep -o 'gene_type "[^"]*"'; done
echo "## STARsolo Gene features with these names + total counts in cells"
S=$PB/runs/neuron1k_v3/starsolo-public-cr/Solo.out
grep -n -E "Atp6-ps|Gm28437|Gm28661|Gm28438|\tmt-Atp6|\tmt-Co3|\tmt-Co2" $S/Gene/raw/features.tsv
ls $REF/star_2.7.11b | head -30; grep -c . $REF/star_2.7.11b/geneInfo.tab; grep -E "Atp6-ps|Gm28437" $REF/star_2.7.11b/geneInfo.tab
echo "## reads at the chr1 NUMT (primary, NH=2), CIGAR + position"
$ENV/bin/samtools view -F 256 $B 1:24600000-24700000 | awk '$0 ~ /NH:i:2/' | cut -f3,4,6 | sort | uniq -c | sort -k1nr | head -15
