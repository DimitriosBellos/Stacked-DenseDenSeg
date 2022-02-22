#!/bin/bash
source /usr2/share/gpu.sbatch
cd $1
python $2 > $3
#while true; do
#nvidia-smi > $3_gpus.txt
#sleep 1
#> $3_gpus.txt
#done
#> slurm-${SLURM_JOBID}-net1.out &
#python train2.py > slurm-${SLURM_JOBID}-net2.out &
#wait
