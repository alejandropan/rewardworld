import sys
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
warnings.filterwarnings('ignore')

##########################
####### Parameters #######
##########################
ROOT='/jukebox/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
id_dict = pd.read_csv('/jukebox/witten/Alex/decoder_output/id_dict.csv')
n_neurons_minimum = 10
alignment_time = 'goCue_time'
pre_time = 0.5
post_time  = 4
smoothing=0
bin_size=0.1
output_folder = '/jukebox/witten/Alex/decoder_output'
temp_folder = '/jukebox/witten/Alex/decoder_wd'

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
neural_data = load_all_residuals(encoding_res_path)
neural_data = neural_data.loc[neural_data['location']==area]

# Trials used
trials_included = common_trials(neural_data)
c_neural_data = common_neural_data(neural_data, trials_included)

# Load variable to be decoded and aligment times
alfio.fQRreward_cue = np.copy(np.roll(alfio.fQRreward,1))
alfio.fQLreward_cue = np.copy(np.roll(alfio.fQLreward,1))
alfio.fQRreward_cue[0] = 0
alfio.fQLreward_cue[0] = 0
regressed_variable_rl = alfio.fQRreward_cue - alfio.fQLreward_cue
regressed_variable_lr = alfio.fQLreward_cue - alfio.fQRreward_cue
regressed_variable_rl = regressed_variable_rl[trials_included.astype(int)]
regressed_variable_lr = regressed_variable_lr[trials_included.astype(int)]
regressed_variable = [regressed_variable_lr, regressed_variable_rl]

# Only trials included in analysis
#weights = get_session_sample_weights(alfio.to_df(), categories = ['choice','probabilityLeft', 'outcome'])
weights = None

# Load and run null distributions
null_sesssions = []
for i in np.arange(200):
    n_temp =  pd.read_csv('/jukebox/witten/Alex/null_sessions/laser_only/'+str(i)+'.csv')
    n_temp = n_temp.iloc[:, np.where(n_temp.columns=='choice')[0][0]:]
    qchosen = n_temp['QRreward'].to_numpy()
    qchosen[np.where(n_temp.choice==-1)] = n_temp.QLreward.to_numpy()[np.where(n_temp.choice==-1)]
    null_sesssions.append(qchosen)

##########################
## Run decoder (linear) ##
##########################

run_decoder_for_session_residual(c_neural_data, area, alfio, regressed_variable, weights, alignment_time, etype = 'real')
#for n, null_ses in enumerate(null_sesssions):
#    run_decoder_for_session_residual(c_neural_data, area, alfio, regressed_variable, weights, alignment_time, etype = 'null', n=n)