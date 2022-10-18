import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as npm
from decoding_debugging import *

##########################
####### Parameters #######
##########################
ROOT='/jukebox/witten/Alex/Data/Subjects/'
id_dict = pd.read_csv('/jukebox/witten/Alex/decoder_output/id_dict.csv')
n_neurons_minimum = 10
alignment_time = 'response_times'
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

alfio = alf(ses, ephys=True)
alfio.mouse = Path(ses).parent.parent.name
alfio.date = Path(ses).parent.name
alfio.ses = Path(ses).name
alfio.path = ses

# Load variable to be decoded and aligment times
regressed_variable = np.copy(alfio.QRreward) #For now qchosen
regressed_variable[np.where(alfio.choice==-1)] = alfio.QLreward[np.where(alfio.choice==-1)] #For now qchosen
all_alignment_times = getattr(alfio, alignment_time)
#weights = get_session_sample_weights(alfio.to_df(), categories = ['choice','probabilityLeft', 'outcome'])
weights = None

##
alfio.QRreward_cue = np.roll(alfio.QRreward,1)
alfio.QLreward_cue = np.roll(alfio.QLreward,1)
alfio.QRreward_cue[0] = 0
alfio.QLreward_cue[0] = 0
regressed_variable = np.copy(alfio.QRreward_cue) #For now qchosen
regressed_variable[np.where(alfio.choice==-1)] = alfio.QLreward_cue[np.where(alfio.choice==-1)] #For now qchosen
alignment_times_all = np.copy(getattr(alfio, alignment_time))
#weights = get_session_sample_weights(alfio.to_df(), categories = ['choice','probabilityLeft', 'outcome'])
weights = None
##

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
run_decoder_for_session_residual(area, alfio, regressed_variable, weights,all_alignment_times, etype = 'real')
t_limit = len(regressed_variable)
for n, null_ses in enumerate(null_sesssions):
    run_decoder_for_session_residual(area, alfio, regressed_variable, weights,all_alignment_times, etype = 'null', n=n)