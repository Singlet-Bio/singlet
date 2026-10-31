cd /mnt/projects/debruinz_project/debruinz/singlet/jobs/cpu-perf
cp /dev/null ids.txt
for S in pbmc1k_v3 neuron1k_v3 GSM7036403; do
  M=0; [ $S = neuron1k_v3 ] && M=1
  for T in 4 8 16; do
    sbatch --parsable --partition=scavenger --cpus-per-task=$T --mem=64G --time=03:30:00 --job-name=cpuperf-$S-t$T --export=ALL,SET=$S,T=$T,PACK=1,MOUSE=$M profile.sbatch >> ids.txt
  done
  sbatch --parsable --partition=scavenger --cpus-per-task=16 --mem=170G --time=03:30:00 --job-name=cpuperf-$S-p4 --export=ALL,SET=$S,T=4,PACK=4,MOUSE=$M profile.sbatch >> ids.txt
done
cat ids.txt | tr '\n' ' '
