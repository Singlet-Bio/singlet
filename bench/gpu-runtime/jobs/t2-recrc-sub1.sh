W=/mnt/projects/debruinz_project/debruinz/singlet
EX="ALL,CUDA_VISIBLE_DEVICES=0,SINGLET_ENGINE_GPU_MIN_CC=70,BINDIR=$W/jobs/gpu-runtime/bin/t2-recrc,BRANCH=gpu/w4-v137,ALLOW_NO_EMBED=1,KEEPOUT=1,SINGLET_HLA=defer,T=18,SAMPLES=pbmc2m pbmc10m pbmc1k_v3 pbmc1k_v2 pbmc1k_cite_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2,MODES=cpu gpu,PAIRS=cpu:gpu,REFD=$W/jobs/gpu-runtime/ref-t2"
cd $W/jobs/gpu-pr2
echo V100S-new $(sbatch --parsable -J t2-rc-gateV -p short-gpu --nice=0 --exclude=g050 --gres=gpu:tesla_v100s:1 -c 18 --mem=170G --time=03:59:00 --export="$EX" pr2-gate.sbatch)
echo V100S-old $(sbatch --parsable -J t2-rc-gateV0 -p short-gpu --nice=0 --exclude=g050 --gres=gpu:tesla_v100s:1 -c 18 --mem=170G --time=03:59:00 --export="$EX,SINGLET_ENGINE_GPU_REC_CHUNK=0" pr2-gate.sbatch)
echo RTX-new $(sbatch --parsable -J t2-rc-gateR -p short-gpu --nice=0 --exclude=g050 --gres=gpu:quadro_rtx_8000:1 -c 18 --mem=170G --time=03:59:00 --export="$EX" pr2-gate.sbatch)
cd $W/jobs/gpu-runtime
echo H100-t3 $(BIN=$W/jobs/gpu-runtime/bin/t2-recrc/singlet-gpu ITEMS="d100s_3p nuclei2k_v2" ARMS=new TAG=t3 sbatch --parsable t2-recrc-item.sbatch)
