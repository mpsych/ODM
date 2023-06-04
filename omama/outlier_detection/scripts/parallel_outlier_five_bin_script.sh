#!/bin/bash
#SBATCH --job-name=array_parallelized_deepsight
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
#SBATCH --mail-user=ryan.zurrin001@umb.edu # Where to send mail
#SBATCH -p haehn -q haehn_unlim
#SBATCH -w chimera12
#SBATCH -n 16 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH --mem=750gb
#SBATCH -t 15-00:00
#SBATCH --output=output/array_%A-%a.out
#SBATCH --error=output/array_%A-%a.err
#SBATCH --array=1-4

##. /etc/profile,

# check cpu number per task, should be equal to -n
echo "using $SLURM_CPUS_ON_NODE CPUs"
echo `date`

eval "$(conda shell.bash hook)"
conda activate O


python binary_executer.py $SLURM_ARRAY_TASK_ID

echo "Finish Run"
echo "end time is `date`"