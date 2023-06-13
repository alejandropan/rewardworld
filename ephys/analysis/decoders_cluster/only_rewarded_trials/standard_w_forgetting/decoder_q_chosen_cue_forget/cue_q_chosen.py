import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as np
from encoding_model_summary_to_df import load_all_residuals, common_trials, common_neural_data
from decoding_debugging import *
import warnings
warnings.filterwarnings('ignore')

##########################
####### Parameters #######
##########################

ROOT='/jukebox/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
id_dict = pd.read_csv('/jukebox/witten/Alex/decoders_rewarded_results/decoder_output_qchosen_cue_forget/id_dict.csv')
n_neurons_minimum = 10
alignment_time = 'goCue_time'
pre_time = 0.5
post_time  = 4
smoothing=0
bin_size=0.1
output_folder = '/jukebox/witten/Alex/decoders_rewarded_results/decoder_output_qchosen_cue_forget'
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
neural_data = load_all_residuals(encoding_res_path, filetype='real')
neural_data = neural_data.loc[neural_data['location']==area]

# Trials used
c_neural_data = common_neural_data(neural_data, n_trials_minimum = int(0.8*len(alfio.choice)))

# Load variable to be decoded and aligment times
alfio.fQRreward_cue = np.copy(np.roll(alfio.fQR,1))
alfio.fQLreward_cue = np.copy(np.roll(alfio.fQL,1))
alfio.fQRreward_cue[0] = 0
alfio.fQLreward_cue[0] = 0
regressed_variable = np.copy(alfio.fQRreward_cue) 
regressed_variable[np.where(alfio.choice==-1)] = alfio.fQLreward_cue[np.where(alfio.choice==-1)]

# Only trials included in analysis
#weights = get_session_sample_weights(alfio.to_df(), categories = ['choice','probabilityLeft', 'outcome'])
weights = None

# Load and run null distributions
null_sesssions = []
for i in np.arange(200):
    n_temp =  pd.read_csv('/jukebox/witten/Alex/null_sessions/laser_only/'+str(i)+'.csv')
    n_temp = n_temp.iloc[:, np.where(n_temp.columns=='choice')[0][0]:]
    qchosen_R = np.roll(np.copy(n_temp['fQR'].to_numpy()),1)
    qchosen_L = np.roll(np.copy(n_temp['fQL'].to_numpy()),1)
    qchosen_R[0] = 0
    qchosen_L[0] = 0   
    qchosen = np.copy(qchosen_R)
    qchosen[np.where(n_temp.choice==-1)] = qchosen_L[np.where(n_temp.choice==-1)] #For now qchosen
    qchosen = qchosen[:len(regressed_variable)]
    null_sesssions.append(qchosen)

##########################
## Run decoder (linear) ##
##########################

run_decoder_for_session_residual(c_neural_data, area, alfio, regressed_variable, weights, alignment_time, etype = 'real', 
                                 output_folder=output_folder, trial_filter = np.where(alfio.outcome==1)[0])
#for n, null_ses in enumerate(null_sesssions[:100]):
#    run_decoder_for_session_residual(c_neural_data, area, alfio, null_ses, weights, alignment_time, etype = 'null', n=n+100, output_folder=output_folder)
