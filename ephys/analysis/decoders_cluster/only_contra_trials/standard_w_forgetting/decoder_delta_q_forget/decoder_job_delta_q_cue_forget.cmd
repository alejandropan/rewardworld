#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=apv2@princeton.edu

module load anacondapy
source activate iblenv
python /jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/only_contra_trials/standard_w_forgetting/decoder_delta_q_forget/cue_delta_q.py $SLURM_ARRAY_TASK_ID

