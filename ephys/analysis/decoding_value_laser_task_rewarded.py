import os
os.chdir('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
from brainbox.singlecell import calculate_peths
from sklearn.linear_model import Lasso as LR
from scipy.stats import pearsonr
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import zscore
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
for ses in SESSIONS:
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
    regressed_variable = regressed_variable[alfio.outcome==1]
    alignment_times_all = np.copy(getattr(alfio, alignment_time))
    alignment_times_all = alignment_times_all[alfio.outcome==1]
    weights = None
    #weights = get_session_sample_weights(alfio.to_df().loc[alfio.to_df()['outcome']==1],alignment_time, categories = ['choice','probabilityLeft'])

    '''
        n = len(regressed_variable) #trial
        selt = np.random.choice(n,int(n/2)) #trial
        regressed_variable = regressed_variable[selt]
        alignment_times = np.copy(getattr(alfio, alignment_time))
        alignment_times = alignment_times[selt]
        weights = get_session_sample_weights(alfio.to_df().iloc[selt,:], categories = ['choice','probabilityLeft'])
    '''

    # Get areas in recording
    areas = []
    for p in np.arange(4): # Max 4 probes
        try:
            areas.append(alfio.probe[p].cluster_group_locations.unique()[~pd.isna(alfio.probe[p].cluster_group_locations.unique())])
        except:
            continue
    areas  = np.unique(np.concatenate(areas))

    ##########################
    ## Run decoder (linear) ##
    ##########################

    for area in areas:
        run_decoder_for_session(area, alfio, regressed_variable, weights,alignment_times_all, etype = 'real')