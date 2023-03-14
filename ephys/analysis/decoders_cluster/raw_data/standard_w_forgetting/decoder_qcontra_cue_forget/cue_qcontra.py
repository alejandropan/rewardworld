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
import glob
warnings.filterwarnings('ignore')

##########################
####### Parameters #######
##########################

ROOT='/jukebox/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
id_dict = pd.read_csv('/jukebox/witten/Alex/decoders_raw_results/decoder_output_qcontra_cue_forget/id_dict.csv')
alignment_time = 'goCue_time'
output_folder = '/jukebox/witten/Alex/decoders_raw_results/decoder_output_qcontra_cue_forget'
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
neural_data = load_all_residuals(encoding_res_path,  filetype='real')
neural_data = neural_data.loc[neural_data['location']==area]

# Trials used
c_neural_data = common_neural_data(neural_data, n_trials_minimum = int(0.8*len(alfio.choice)))

# Load variable to be decoded and aligment times
alfio.fQRreward_cue = np.copy(np.roll(alfio.fQRreward,1))
alfio.fQLreward_cue = np.copy(np.roll(alfio.fQLreward,1))
alfio.fQRreward_cue[0] = 0
alfio.fQLreward_cue[0] = 0
regressed_variable_rl = alfio.fQRreward_cue
regressed_variable_lr = alfio.fQLreward_cue
regressed_variable = [regressed_variable_lr, regressed_variable_rl]

# Only trials included in analysis
#weights = get_session_sample_weights(alfio.to_df(), categories = ['choice','probabilityLeft', 'outcome'])
weights = None

##########################
## Run decoder (linear) ##
##########################

run_decoder_for_session_residual(c_neural_data, area, alfio, regressed_variable, weights, alignment_time, etype = 'real', output_folder=output_folder)


'''
##########################
## Run nulls (linear) ##
##########################
# Run nulls (first look for missing nulls)
path_search_0 = output_folder+'/*null_' + area + '_' + alfio.mouse + '_' + alfio.date+'*0_p_summary.npy'
files_0 = glob.glob(path_search_0)
files_max_0 = len(files_0)
path_search_1 = output_folder+'/*null_' + area + '_' + alfio.mouse + '_' + alfio.date+'*1_p_summary.npy'
files_1 = glob.glob(path_search_1)
files_max_1 = len(files_1)-1
files_max = np.min(files_max_0,files_max_1)
objective = np.arange(files_max,200)

ses_len = len(regressed_variable)
null_sesssions = []

for i in objective:
    n_temp =  pd.read_csv('/jukebox/witten/Alex/null_sessions/laser_only/'+str(i)+'.csv')
    n_temp = n_temp.iloc[:, np.where(n_temp.columns=='choice')[0][0]:]
    qchosen_R = np.roll(np.copy(n_temp['fQRreward'].to_numpy()),1)
    qchosen_L = np.roll(np.copy(n_temp['fQLreward'].to_numpy()),1)
    qchosen_R[0] = 0
    qchosen_L[0] = 0   
    qchosen = np.copy(qchosen_R)
    qchosen[np.where(n_temp.choice==-1)] = qchosen_L[np.where(n_temp.choice==-1)] #For now qchosen
    qchosen = qchosen[:ses_len]
    null_sesssions.append(qchosen)


for n, null_ses in enumerate(null_sesssions):
    run_decoder_for_session_residual(c_neural_data, area, alfio, null_ses, weights, alignment_time, etype = 'null', n=n, output_folder=output_folder)
'''