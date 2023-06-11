#!/bin/bash
#SBATCH --job-name=array_parallelized_deepsight
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=100gb
#SBATCH -t 30-00:00
#SBATCH --open-mode=append
#SBATCH --output=output/array.out
#SBATCH --error=output/array.err
#SBATCH --array=1-20

##. /etc/profile,

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

echo "end time is `date`"

export CUDA_VISIBLE_DEVICES=2
export TF_GPU_ALLOCATOR=cuda_malloc_async
export TF_FORCE_GPU_ALLOW_GROWTH=true


python odl_executer.py 5 custom $SLURM_ARRAY_TASK_ID 1

echo "Finish Run"
