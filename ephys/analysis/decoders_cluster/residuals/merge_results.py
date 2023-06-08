import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
import pandas as pd
import numpy as np
from decoders_summary import *

variable_paths = [
'/jukebox/witten/Alex/decoders_residuals_results/decoder_output_choice_cue_forget',
'/jukebox/witten/Alex/decoders_residuals_results/decoder_output_deltaq_cue_forget',
'/jukebox/witten/Alex/decoders_residuals_results/decoder_output_outcome_outcome_forget',
'/jukebox/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget',
'/jukebox/witten/Alex/decoders_residuals_results/decoder_output_qchosen_outcome_forget',
'/jukebox/witten/Alex/decoders_residuals_results/decoder_output_qchosen_outcome_post_forget',
'/jukebox/witten/Alex/decoders_residuals_results/decoder_totalq_cue_forget']

varss = [
'choice',
'deltaq',
'outcome',
'qchosen_pre',
'qchosen_pre',
'qchosen_post',
'totalq']

epochs = [
'cue',
'cue',
'outcome',
'cue',
'outcome',
'outcome',
'cue']

nsummary = load_decoders(variable_paths[int(sys.argv[1])], var = varss[int(sys.argv[1])], 
                         epoch = epochs[int(sys.argv[1])], x_type = 'residuals', null=True)
nsummary.to_csv('/jukebox/witten/Alex/decoders_residuals_results/nsummary_' + 
                varss[int(sys.argv[1])] + '_' + epochs[int(sys.argv[1])] + '.csv')
