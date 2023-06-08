import sys
import numpy as np
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as npm
from encoding_model_summary_to_df import load_all_residuals, common_trials, common_neural_data
from decoding_debugging import *
import warnings
import glob
warnings.filterwarnings('ignore')

##########################
####### Parameters #######
##########################

ROOT='/jukebox/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
id_dict = pd.read_csv('/jukebox/witten/Alex/decoders_raw_results/decoder_output_decision_time_cue_forget/id_dict.csv')
alignment_time = 'goCue_time'
output_folder = '/jukebox/witten/Alex/decoders_raw_results/decoder_output_decision_time_cue_forget'
temp_folder = '/jukebox/witten/Alex/decoder_wd'
n_trials_minimum = 100

##########################
####### Load Data ########
##########################

ses = ROOT+id_dict.loc[id_dict['id']==int(sys.argv[1]),'ses'].to_string(index=False)
area = id_dict.loc[id_dict['id']==int(sys.argv[1]),'area'].to_string(index=False)

# Load behavior
alfio = alf(ses, ephys=True)
alfio.mouse = Path(ses).parent.parent.name
alfio.date = Path(ses).parent.name
alfio.ses = Path(ses).name
alfio.path = ses

# Load neurons
encoding_res_path = ROOT_NEURAL+'/'+ \
                    id_dict.loc[id_dict['id']==int(sys.argv[1]),'ses'].to_string(index=False)+\
                    '/alf/encodingmodels/inputs/neurons/' 
neural_data = load_all_residuals(encoding_res_path, filetype='real')
neural_data = neural_data.loc[neural_data['location']==area]

# Trials used
c_neural_data = common_neural_data(neural_data, n_trials_minimum = int(0.8*len(alfio.choice)))

# Load variable to be decoded and aligment times
regressed_variable = np.copy(alfio.response_times) - np.copy(alfio.goCue_trigger_times)

#For now qchosen
# Only trials included in analysis
#weights = get_session_sample_weights(alfio.to_df(), categories = ['choice','probabilityLeft', 'outcome'])
weights = None

##########################
## Run nulls (linear) ##
##########################
'''
null_sesssions = []
for i in np.arange(200):
    n_temp =  pd.read_csv('/jukebox/witten/Alex/null_sessions/laser_only/'+str(i)+'.csv')
    n_temp = n_temp.iloc[:, np.where(n_temp.columns=='choice')[0][0]:]
    latency = np.copy(n_temp.response_times) - np.copy(n_temp.goCue_trigger_times)
    latency = latency[:len(regressed_variable)]
    null_sesssions.append(latency)
'''
run_decoder_for_session_residual(c_neural_data, area, alfio, regressed_variable, weights, alignment_time, etype = 'real', output_folder=output_folder)
#for n, null_ses in enumerate(null_sesssions):
    #run_decoder_for_session_residual(c_neural_data, area, alfio, null_ses, weights, alignment_time, etype = 'null', n=n, output_folder=output_folder)


