#!/bin/bash
#SBATCH -J usa-m2
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 01:30:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# sbatch --export=ALL,SHA8=<8> usa_metrics2.sbatch : usa_metrics.py on ec runs of a build (pbmc1k_v3 hgmm1k_v3 nuclei2k_v2) vs STARsolo Velocyto and alevin-fry
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; X=$W/expression-sota; mkdir -p $X/usa
for d in pbmc1k_v3 hgmm1k_v3 nuclei2k_v2; do $W/envs/pbench/bin/python3 -I $X/code/usa_metrics.py $PB $d $W/release-gate/runs/ec/$SHA8-cpu/$d $X/usa/$d.$SHA8.json --label $SHA8 > /dev/null; done
$W/envs/pbench/bin/python3 -I - <<EOF
import json
for d in ["pbmc1k_v3","hgmm1k_v3","nuclei2k_v2"]:
    a=json.load(open("$X/usa/%s.json"%d)); b=json.load(open("$X/usa/%s.$SHA8.json"%d))
    f=lambda j:{k:round(v["pearson_log1p"],3) for k,v in j["unspliced_pooled_gene_r"].items()}
    print(d,"old U r",f(a),"new",f(b)); print("  Ufrac old",{k:round(x,3) for k,x in a["unspliced_fraction"].items()},"new",{k:round(x,3) for k,x in b["unspliced_fraction"].items()})
    print("  cellr old",{k:round(x,3) for k,x in a["unspliced_cell_median_r"].items()},"new",{k:round(x,3) for k,x in b["unspliced_cell_median_r"].items()})
    print("  shares new",{t:{k:round(v,3) for k,v in s.items() if k!="umi"} for t,s in b["layer_shares"].items()})
    if "cross_species_leak" in b: print("  leak old",{t:round(x["intron_leak"],4) for t,x in a["cross_species_leak"].items()},"new",{t:round(x["intron_leak"],4) for t,x in b["cross_species_leak"].items()})
EOF
