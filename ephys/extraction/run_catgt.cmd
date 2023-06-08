#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=apv2@princeton.edu

module load anacondapy
source activate iblenv
python /jukebox/witten/Alex/PYTHON/rewardworld/ephys/extraction/run_catgt.py $SLURM_ARRAY_TASK_ID