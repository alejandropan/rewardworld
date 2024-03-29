import subprocess

models = ['decoder_totalq_cue_forget',
'decoder_choice_cue_forget',
'decoder_delta_q_forget',
'decoder_outcome_outcome_forget',
'decoder_q_chosen_cue_forget',
'decoder_q_chosen_outcome_forget',
'decoder_qcontra_cue_forget',
'decoder_qipsi_cue_forget',
'decoder_RPE_outcome']

root_residuals = 'python /jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/residuals/standard_w_forgetting/'
root_raw = 'python /jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/raw_data/standard_w_forgetting/'

for m in models:
    p = root_residuals + m + '/master.py'
    print(subprocess.run([p], shell=True))
    p = root_raw + m + '/master.py'
    print(subprocess.run([p], shell=True))


