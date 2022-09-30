import os
os.chdir('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as np
from decoding_debugging import *

##########################
####### Parameters #######
##########################
SESSIONS = ['/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-16/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',  
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-19/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-27/003', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-20/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-10/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001']

ses = SESSIONS[int(sys.argv[1])]
print(ses)
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

# Get areas in recording
areas = []
for p in np.arange(4): # Max 4 probes
    try:
        areas.append(alfio.probe[p].cluster_group_locations.unique()[~pd.isna(alfio.probe[p].cluster_group_locations.unique())])
    except:
        continue
areas  = np.unique(np.concatenate(areas))
# Load and run null distributions
null_sesssions = []
null_weights = []
for i in np.arange(200):
    n_temp =  pd.read_csv('/jukebox/witten/Alex/null_sessions/laser_only/'+str(i)+'.csv')
    n_temp = n_temp.iloc[:, np.where(n_temp.columns=='choice')[0][0]:]
    qchosen = n_temp['QRreward'].to_numpy()
    qchosen[np.where(n_temp.choice==-1)] = n_temp.QLreward.to_numpy()[np.where(n_temp.choice==-1)]
    null_sesssions.append(qchosen)
    null_weights.append(get_session_sample_weights(n_temp, categories = ['choice','probabilityLeft', 'outcome']))

##########################
## Run decoder (linear) ##
##########################

for area in areas:
    run_decoder_for_session(area, alfio, regressed_variable, weights,all_alignment_times, type = 'real')
    t_limit = len(regressed_variable)
    for n, null_ses in enumerate(null_sesssions):
        run_decoder_for_session(area, alfio, regressed_variable, weights,all_alignment_times, type = 'null')