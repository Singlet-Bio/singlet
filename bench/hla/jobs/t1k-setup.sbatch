#!/bin/bash
#SBATCH --job-name=t1k-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# D-101 HLA typing: install T1K (MIT, github.com/mourisl/T1K, tag T1K_TAG) and its IPD-IMGT/HLA index where the engine looks for them:
#   $REF/hla/t1k/bin/{fastq-extractor,genotyper,analyzer}   $REF/hla/t1k/hlaidx/hlaidx_rna_seq.fa (+ _dna_seq.fa)
#   $REF/hla/t1k/manifest.json: T1K tag and commit, IPD-IMGT/HLA release (from hla.dat), sha256 of hla.dat.zip and of the index
# Built with the pbench conda toolchain (rpath to its zlib), so the binaries run on the hosts like singlet-gpu.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; ENV=$W/envs/pbench
T1K_TAG=${T1K_TAG:-v1.0.10}; DEST=${DEST:-$REF/hla/t1k}
SCR=/tmp/$USER-t1k-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
[ -e $DEST/manifest.json ] && { echo "refusing to overwrite $DEST"; exit 2; }
git clone -q --depth 1 --branch $T1K_TAG https://github.com/mourisl/T1K.git $SCR/T1K
COMMIT=$(git -C $SCR/T1K rev-parse HEAD)
cd $SCR/T1K
make -j8 CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ LINKFLAGS="-lpthread -lz -L$ENV/lib -Wl,-rpath,$ENV/lib" CXXFLAGS="-O3 -g -I$ENV/include" fastq-extractor genotyper analyzer
mkdir -p $SCR/out/bin $SCR/out/hlaidx
cp fastq-extractor genotyper analyzer run-t1k t1k-build.pl ParseDatFile.pl AddGeneCoord.pl LICENSE $SCR/out/bin/
cp -r $SCR/T1K/LICENSE $SCR/out/LICENSE.T1K
cd $SCR/out
curl -fsS --retry 5 -o hlaidx/hla.dat.zip https://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/hla.dat.zip
( cd hlaidx && unzip -q -o hla.dat.zip )
REL=$(grep -m1 -oE "IPD-IMGT/HLA Release [0-9.]+|Release: [0-9.]+|version: [0-9.]+" hlaidx/hla.dat || true)
[ -n "$REL" ] || REL=$(grep -m3 -iE "release|version" hlaidx/hla.dat | head -1)
perl bin/t1k-build.pl -o hlaidx -d hlaidx/hla.dat
ls -l hlaidx
SHA_DAT=$(sha256sum hlaidx/hla.dat.zip | cut -c1-64); SHA_RNA=$(sha256sum hlaidx/hlaidx_rna_seq.fa | cut -c1-64)
NALLELE=$(grep -c "^>" hlaidx/hlaidx_rna_seq.fa)
rm -f hlaidx/hla.dat
cat > manifest.json <<EOJ
{"tool": "T1K", "license": "MIT", "url": "https://github.com/mourisl/T1K", "tag": "$T1K_TAG", "commit": "$COMMIT",
 "reference": "IPD-IMGT/HLA", "reference_release": "$REL", "reference_url": "https://ftp.ebi.ac.uk/pub/databases/ipd/imgt/hla/hla.dat.zip",
 "hla_dat_zip_sha256": "$SHA_DAT", "rna_seq_fa": "hlaidx/hlaidx_rna_seq.fa", "rna_seq_fa_sha256": "$SHA_RNA", "rna_seq_alleles": $NALLELE,
 "preset": "hla (genotyper -s 0.97)", "built": "$(date -u +%Y-%m-%dT%H:%M:%SZ)", "builder": "bench/hla/jobs/t1k-setup.sbatch"}
EOJ
cat manifest.json
mkdir -p $(dirname $DEST); cp -r $SCR/out $DEST.tmp-$SLURM_JOB_ID && mv $DEST.tmp-$SLURM_JOB_ID $DEST
$DEST/bin/genotyper 2>&1 | head -3 || true
ls -l $DEST $DEST/bin $DEST/hlaidx
