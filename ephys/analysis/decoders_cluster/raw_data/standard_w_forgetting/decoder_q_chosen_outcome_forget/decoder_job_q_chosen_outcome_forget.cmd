#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=apv2@princeton.edu

module load anacondapy
source activate iblenv
python /jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/raw_data/standard_w_forgetting/decoder_q_chosen_outcome_forget/outcome_q_chosen.py $SLURM_ARRAY_TASK_ID

