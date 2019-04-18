#!/bin/bash
cd $1
python $2
#> slurm-${SLURM_JOBID}-net1.out &
#python train2.py > slurm-${SLURM_JOBID}-net2.out &
#wait
