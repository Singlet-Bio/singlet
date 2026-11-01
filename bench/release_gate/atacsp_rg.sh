#!/bin/bash
# Release gate: atac / multiome / spatial with the pinned binary. Login node: only sbatch. Competitor runs (chromap, STARsolo Visium)
# and the 10x published outputs are reused from pipeline-benchmark/runs/atacsp; only the Singlet runs are repeated.
#   bash atacsp_rg.sh
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; RG=$W/release-gate; RN=$RG/runs/atacsp; CN=$PB/runs/atacsp
D=$W/data/atac-spatial; A=$RG/artifacts-atacsp; J=$RG/jobs; H=$W/src/singlet-py/claude__atac-spatial-benchmark-2ba7df__scripts/bench/atac_spatial
D10=$D/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k
F=$D/Visium_FFPE_Human_Breast_Cancer/Visium_FFPE_Human_Breast_Cancer_fastqs
M=$D/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_Section_1_fastqs
S="sbatch --parsable -A debruinz --partition=short-cpu,scavenger --requeue"
run() { $S --cpus-per-task=16 --mem=${MEM:-110G} ${2:+--dependency=afterok:$2} --job-name=$3 --export=ALL,$4 $J/run_sample_rg.sbatch; }
g10=$(run x "" rg-mo10k-gex RUN=pbmc_multiome10k_gex,KIND=gex,FQ=$D10/gex,GLOB=pbmc_granulocyte_sorted_10k,GENOME=GRCh38-2024-A)
a10=$(MEM=150G run x $g10 rg-mo10k-atac RUN=pbmc_multiome10k_atac,KIND=atac,FQ=$D10/atac,GLOB=pbmc_granulocyte_sorted_10k,GENOME=GRCh38-2024-A,PROTO=10x-atac,PAIRED_GEX=$RN/pbmc_multiome10k_gex)
a5=$(run x "" rg-atac5k RUN=atac_pbmc_5k_nextgem,KIND=atac,FQ=$D/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fastqs,GLOB=atac_pbmc_5k_nextgem,GENOME=GRCh38-2024-A,PROTO=10x-atac)
vm=$(run x "" rg-visium-mba RUN=visium_mouse_brain_ant,KIND=visium,FQ=$M,GLOB=V1_Mouse_Brain_Sagittal_Anterior,GENOME=GRCm39-2024-A,PROTO=10x-visium)
vf=$(run x "" rg-visium-ffpe RUN=visium_ffpe_human_breast,KIND=visium,FQ=$F,GLOB=Visium_FFPE_Human_Breast_Cancer,GENOME=GRCh38-2024-A,PROTO=10x-visium)
art() { $S --job-name=rg-art --mem=64G --dependency=afterok:$1 $J/pyjob_rg.sbatch make_artifacts.py "${@:2}"; }
L=$W/cache/liftover/hg19ToHg38.over.chain.gz
x1=$(art $a5 atac atac_pbmc_5k_nextgem --tenx-dir $D/atac_pbmc_5k_nextgem --prefix atac_pbmc_5k_nextgem --kind cratac --singlet $RN/atac_pbmc_5k_nextgem --chromap $CN/chromap_atac_pbmc_5k_nextgem --tenx-chain $L --out $A/atac)
x2=$(art $a10 atac pbmc_multiome10k --tenx-dir $D/pbmc_granulocyte_sorted_10k --prefix pbmc_granulocyte_sorted_10k --kind arc --singlet $RN/pbmc_multiome10k_atac --chromap $CN/chromap_pbmc_multiome10k --gex $RN/pbmc_multiome10k_gex --out $A/atac)
x3=$(art $a10 atac pbmc_multiome10k --tenx-dir $D/pbmc_granulocyte_sorted_10k --prefix pbmc_granulocyte_sorted_10k --kind arc --singlet $RN/pbmc_multiome10k_atac --chromap $CN/chromap_pbmc_multiome10k --gex $RN/pbmc_multiome10k_gex --out $A/multiome)
x5=$(art $vm spatial visium_mouse_brain_ant --tenx-dir $D/V1_Mouse_Brain_Sagittal_Anterior --prefix V1_Mouse_Brain_Sagittal_Anterior --singlet $RN/visium_mouse_brain_ant --starsolo $CN/starsolo_visium_mouse_brain_ant --out $A/spatial)
x6=$(art $vf spatial visium_ffpe_human_breast --tenx-dir $D/Visium_FFPE_Human_Breast_Cancer --prefix Visium_FFPE_Human_Breast_Cancer --singlet $RN/visium_ffpe_human_breast --starsolo $CN/starsolo_visium_ffpe_human_breast_rev --out $A/spatial)
echo "runs: $g10 $a10 $a5 $vm $vf; artifacts: $x1 $x2 $x3 $x5 $x6"
