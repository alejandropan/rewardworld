import subprocess
print(subprocess.run(['sbatch --array=0-6 /jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/residuals/merge_results.cmd'], shell=True))