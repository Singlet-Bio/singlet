#!/bin/bash
# Reproduce the ATAC / Multiome / Visium benchmark on Clipper (run on the login node: it only calls sbatch).
#   bash bench/atac_spatial/submit.sh            # everything, chained with Slurm dependencies
# Steps: data download -> comparator env -> barcode lists -> build -> Singlet runs -> Chromap / STARsolo comparators ->
# artifacts (make_artifacts.py) -> provenance (make_provenance.py). The capability documents are then computed from the
# artifacts by bench/public/capabilities/{atac,multiome,spatial}.py (laptop, minutes).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; RN=$PB/runs/atacsp; WL=$PB/whitelists
D=$W/data/atac-spatial; A=$W/bench-atacsp/artifacts; H=$(cd "$(dirname "$0")" && pwd)
S="sbatch --parsable -A debruinz"; L="--partition=short-cpu,scavenger --requeue"
D3=$D/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k; D10=$D/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k
F=$D/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_fastqs
M=$D/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_Section_1_fastqs

dl=$($S $H/download.sbatch)
env=$($S $H/setup_env.sbatch)
lists=$($S --dependency=afterok:$dl:$env $H/pyjob.sbatch make_lists.py $D $WL/atacsp $WL)
b=$($S $L --dependency=afterok:$lists $H/build.sbatch)
run() { $S $L --cpus-per-task=16 --mem=${MEM:-110G} --dependency=afterok:$1 --job-name=$2 --export=ALL,$3 $H/run_sample.sbatch; }
g3=$(run $b as-mo3k-gex RUN=pbmc_multiome3k_gex,KIND=gex,FQ=$D3/gex,GLOB=pbmc_granulocyte_sorted_3k,GENOME=GRCh38-2024-A)
a3=$(run $g3 as-mo3k-atac RUN=pbmc_multiome3k_atac,KIND=atac,FQ=$D3/atac,GLOB=pbmc_granulocyte_sorted_3k,GENOME=GRCh38-2024-A,PROTO=10x-atac,PAIRED_GEX=$RN/pbmc_multiome3k_gex)
g10=$(run $b as-mo10k-gex RUN=pbmc_multiome10k_gex,KIND=gex,FQ=$D10/gex,GLOB=pbmc_granulocyte_sorted_10k,GENOME=GRCh38-2024-A)
a10=$(MEM=150G run $g10 as-mo10k-atac RUN=pbmc_multiome10k_atac,KIND=atac,FQ=$D10/atac,GLOB=pbmc_granulocyte_sorted_10k,GENOME=GRCh38-2024-A,PROTO=10x-atac,PAIRED_GEX=$RN/pbmc_multiome10k_gex)
a5=$(run $b as-atac5k RUN=atac_pbmc_5k_nextgem,KIND=atac,FQ=$D/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fastqs,GLOB=atac_pbmc_5k_nextgem,GENOME=GRCh38-2024-A,PROTO=10x-atac)
vm=$(run $b as-visium-mba RUN=visium_mouse_brain_ant,KIND=visium,FQ=$M,GLOB=V1_Mouse_Brain_Sagittal_Anterior,GENOME=GRCm39-2024-A,PROTO=10x-visium)
vf=$(run $b as-visium-ffpe RUN=visium_ffpe_human_breast,KIND=visium,FQ=$F,GLOB=Visium_FFPE_Human_Breast_Cancer,GENOME=GRCh38-2024-A,PROTO=10x-visium)
c3=$($S $L --dependency=afterok:$lists --export=ALL,RUN=pbmc_multiome3k,FQ=$D3/atac,GLOB=pbmc_granulocyte_sorted_3k,WL=$WL/737K-arc-v1-atac-gex.tsv.gz,TRANSLATE=$WL/737K-arc-v1-atac-gex.tsv.gz $H/run_chromap.sbatch)
c10=$($S $L --dependency=afterok:$lists --export=ALL,RUN=pbmc_multiome10k,FQ=$D10/atac,GLOB=pbmc_granulocyte_sorted_10k,WL=$WL/737K-arc-v1-atac-gex.tsv.gz,TRANSLATE=$WL/737K-arc-v1-atac-gex.tsv.gz $H/run_chromap.sbatch)
c5=$($S $L --dependency=afterok:$lists --export=ALL,RUN=atac_pbmc_5k_nextgem,FQ=$D/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fastqs,GLOB=atac_pbmc_5k_nextgem,WL=$WL/atacsp/737K-cratac-v1.txt $H/run_chromap.sbatch)
sm=$($S $L --dependency=afterok:$lists $H/run_starsolo_visium.sbatch)
sf=$($S $L --dependency=afterok:$lists --export=ALL,RUN=visium_ffpe_human_breast_rev,FQ=$F,GENOME=GRCh38-2024-A,STRAND=Reverse $H/run_starsolo_visium.sbatch)
art() { $S $L --mem=64G --dependency=afterok:$1 $H/pyjob.sbatch make_artifacts.py "${@:2}"; }
x1=$(art $a5:$c5 atac atac_pbmc_5k_nextgem --tenx-dir $D/atac_pbmc_5k_nextgem --prefix atac_pbmc_5k_nextgem --kind cratac --singlet $RN/atac_pbmc_5k_nextgem --chromap $RN/chromap_atac_pbmc_5k_nextgem --out $A/atac)
x2=$(art $a10:$c10 atac pbmc_multiome10k --tenx-dir $D/pbmc_granulocyte_sorted_10k --prefix pbmc_granulocyte_sorted_10k --kind arc --singlet $RN/pbmc_multiome10k_atac --chromap $RN/chromap_pbmc_multiome10k --gex $RN/pbmc_multiome10k_gex --out $A/atac)
x3=$(art $a10:$c10 atac pbmc_multiome10k --tenx-dir $D/pbmc_granulocyte_sorted_10k --prefix pbmc_granulocyte_sorted_10k --kind arc --singlet $RN/pbmc_multiome10k_atac --chromap $RN/chromap_pbmc_multiome10k --gex $RN/pbmc_multiome10k_gex --out $A/multiome)
x4=$(art $a3:$c3 atac pbmc_multiome3k --tenx-dir $D/pbmc_granulocyte_sorted_3k --prefix pbmc_granulocyte_sorted_3k --kind arc --singlet $RN/pbmc_multiome3k_atac --chromap $RN/chromap_pbmc_multiome3k --gex $RN/pbmc_multiome3k_gex --out $A/dev)
x5=$(art $vm:$sm spatial visium_mouse_brain_ant --tenx-dir $D/V1_Mouse_Brain_Sagittal_Anterior --prefix V1_Mouse_Brain_Sagittal_Anterior --singlet $RN/visium_mouse_brain_ant --starsolo $RN/starsolo_visium_mouse_brain_ant --out $A/spatial)
x6=$(art $vf:$sf spatial visium_ffpe_human_breast --tenx-dir $D/Visium_FFPE_Human_Breast_Cancer --prefix Visium_FFPE_Human_Breast_Cancer --singlet $RN/visium_ffpe_human_breast --starsolo $RN/starsolo_visium_ffpe_human_breast_rev --out $A/spatial)
echo "submitted; artifacts in $A when jobs $x1 $x2 $x3 $x4 $x5 $x6 finish."
echo "then: $S --dependency=afterok:$x1:$x2:$x3:$x5:$x6 $H/pyjob.sbatch make_provenance.py $A <commit> $PB/bin/singlet-atacsp"
