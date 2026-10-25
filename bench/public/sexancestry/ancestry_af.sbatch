#!/bin/bash
#SBATCH --job-name=bpub-af
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Leave-out allele frequencies for the ancestry benchmark: the same table singlet.celllabels.ancestry uses in production
# ($WORK/data/annotate/1kgp_superpop_af.parquet: chrom, pos, ref, alt, AFR..SAS on the pipeline's SNP panel), recomputed
# from 1000 Genomes phase 3 GRCh38 genotypes WITHOUT the benchmark's test individuals (every MAGE subset individual and
# the four scRNA-seq LCL donors). Genotypes are read here and reduced to population frequencies; nothing per individual
# is written. Output: $BP/work/ancestry/af_leaveout.parquet and af_leaveout.json (counts, excluded ids).
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
BPE=$WORK/envs/bpub; export PATH=$BPE/bin:$PATH
O=$BP/work/ancestry; R=$WORK/data/1kgp_grch38; mkdir -p $O $R
BASE=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000_genomes_project/release/20190312_biallelic_SNV_and_INDEL
PANEL=https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel
[ -s $R/panel.txt ] || curl -sSL --retry 5 -o $R/panel.txt $PANEL
# test individuals: MAGE subset + the scRNA-seq LCL donors (lcl_gse230239, osorio2019_lcl_mix)
{ tail -n +2 $SRC/bench/public/sexancestry/mage_subset.tsv | cut -f2; printf "HG00362\nHG01277\nNA12878\nNA18502\n"; } | sort -u > $O/excluded.txt
awk 'NR>1 {print $1"\t"$3}' $R/panel.txt | grep -vwFf $O/excluded.txt > $O/groups.txt   # sample <tab> super-population
echo "excluded $(wc -l < $O/excluded.txt), kept $(wc -l < $O/groups.txt) individuals"
curl -sSL $BASE/ | grep -oE 'ALL\.chr[0-9]+\.shapeit2[^"]*phased\.vcf\.gz' | sort -u > $O/vcf_names.txt
wc -l < $O/vcf_names.txt
$BPE/bin/python - <<EOF
import pandas as pd
af = pd.read_parquet("$WORK/data/annotate/1kgp_superpop_af.parquet", columns=["chrom", "pos"])
af.to_csv("$O/targets.nochr.tsv", sep="\t", header=False, index=False)
af.assign(chrom="chr" + af["chrom"].astype(str)).to_csv("$O/targets.chr.tsv", sep="\t", header=False, index=False)
print(len(af), "panel sites")
EOF
for f in $(cat $O/vcf_names.txt); do
  c=$(echo $f | grep -oE 'chr[0-9]+'); out=$O/af.$c.tsv
  [ -s $out ] && continue
  [ -s $R/$f ] || curl -sSL --retry 5 -C - -o $R/$f $BASE/$f
  pre=$(bcftools view -h $R/$f | grep -m1 '^##contig' | grep -q 'ID=chr' && echo chr || echo nochr)
  ( bcftools view --threads 2 -S $(cut -f1 $O/groups.txt | paste -sd, ) --force-samples -m2 -M2 -v snps -T $O/targets.$pre.tsv -Ou $R/$f \
    | bcftools +fill-tags -Ou -- -S $O/groups.txt -t AF \
    | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%AF_AFR\t%AF_AMR\t%AF_EAS\t%AF_EUR\t%AF_SAS\n' > $out.part && mv $out.part $out ) &
  while [ $(jobs -r | wc -l) -ge 7 ]; do sleep 5; done
done
wait
$BPE/bin/python - <<EOF
import glob, json, pandas as pd, numpy as np
cols = ["chrom", "pos", "ref", "alt", "AFR", "AMR", "EAS", "EUR", "SAS"]
df = pd.concat([pd.read_csv(f, sep="\t", names=cols, dtype={"chrom": str}) for f in sorted(glob.glob("$O/af.chr*.tsv"))])
df["chrom"] = df["chrom"].str.replace("^chr", "", regex=True)
for p in cols[4:]:
    df[p] = df[p].astype(np.float32)
old = pd.read_parquet("$WORK/data/annotate/1kgp_superpop_af.parquet")
df = df.merge(old[["chrom", "pos", "ref", "alt"]].astype({"chrom": str}), on=["chrom", "pos", "ref", "alt"])  # same sites, same alleles
f = df[cols[4:]].to_numpy()
df = df[((f >= 0.01) & (f <= 0.99)).any(1)]
df.to_parquet("$O/af_leaveout.parquet", index=False)
d = (df[cols[4:]].to_numpy() - old.set_index(["chrom","pos","ref","alt"]).loc[pd.MultiIndex.from_frame(df[cols[:4]])][cols[4:]].to_numpy())
json.dump({"n_sites": int(len(df)), "n_sites_production": int(len(old)), "excluded": open("$O/excluded.txt").read().split(),
           "mean_abs_af_change": float(np.abs(d).mean()), "max_abs_af_change": float(np.abs(d).max())},
          open("$O/af_leaveout.json", "w"), indent=1)
print(open("$O/af_leaveout.json").read()[:400])
EOF
