#!/bin/bash
# Check which model organism references are installed
REF_BASE="/mnt/projects/debruinz_project/cellarium/reference"
declare -A ASSEMBLIES
ASSEMBLIES[human]="GRCh38-2024-A"
ASSEMBLIES[mouse]="GRCm39-2024-A"
ASSEMBLIES[zebrafish]="GRCz11"
ASSEMBLIES[rat]="mRatBN7.2"
ASSEMBLIES[fruit_fly]="BDGP6.46"
ASSEMBLIES[nematode]="WBcel235"
ASSEMBLIES[chicken]="GRCg7b"
ASSEMBLIES[pig]="Sscrofa11.1"
ASSEMBLIES[cow]="ARS-UCD1.3"
ASSEMBLIES[dog]="ROS_Cfam_1.0"
ASSEMBLIES[macaque]="Mmul_10"
ASSEMBLIES[horse]="EquCab3.0"
ASSEMBLIES[cat]="Felis_catus_9.0"
ASSEMBLIES[rabbit]="OryCun2.0"
ASSEMBLIES[yeast]="R64-1-1"
ASSEMBLIES[frog]="UCB_Xtro_10.0"
ASSEMBLIES[sheep]="ARS-UI_Ramb_v2.0"

for species in human mouse zebrafish rat fruit_fly nematode chicken pig cow dog macaque horse cat rabbit yeast frog sheep; do
    asm="${ASSEMBLIES[$species]}"
    star_dir="$REF_BASE/$asm/star_2.7.11b"
    if [[ -f "$star_dir/SA" ]]; then
        size=$(ls -lh "$star_dir/SA" | awk '{print $5}')
        echo "✅ $species ($asm): $size"
    else
        echo "❌ $species ($asm): not installed"
    fi
done
