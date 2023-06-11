#!/bin/bash
#SBATCH --job-name=array_parallelized_deepsight
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=28gb
#SBATCH -t 30-00:00
#SBATCH --output=output/array_%A-%a.out
#SBATCH --error=output/array_%A-%a.err
#SBATCH --array=1-36

##. /etc/profile,

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O

python outlier_executor.py $SLURM_ARRAY_TASK_ID

echo "Finish Run"
echo "end time is `date`"