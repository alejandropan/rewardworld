import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
from pathlib import Path
from glob import glob
import os
from scipy.stats import ttest_rel as pairedt
from brainbox.singlecell import calculate_peths
from sklearn.decomposition import PCA
from scipy.stats import zscore
from functools import reduce
from sklearn.preprocessing import scale
import copy
import logistic_regression as lr
from scipy.stats import ttest_rel as pttest
#from mpl_chord_diagram import chord_diagram

from itertools import permutations 

LIST_OF_SESSIONS_YFP = ['/Volumes/witten/Alex/Data/Subjects/dop_18/2021-04-15/008',
'/Volumes/witten/Alex/Data/Subjects/dop_18/2021-04-15/002',
'/Volumes/witten/Alex/Data/Subjects/dop_18/2021-04-14/003',
'/Volumes/witten/Alex/Data/Subjects/dop_18/2021-03-17/001']

LIST_OF_SESSIONS_CHR2 = \
['/Volumes/witten/Alex/Data/Subjects/dop_14/2021-03-30/001',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_14/2021-04-06/001',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_14/2021-04-08/001',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_14/2021-04-10/001',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_14/2021-04-12/001',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_14/2021-04-14/001',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_14/2021-04-16/001',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_13/2021-03-14/003',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_13/2021-03-16/002',
'/Volumes/witten/Alex/Data/ephys_bandit/dop_13/2021-03-19/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/004',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-19/002',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-14/002',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-05-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-30/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-26/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-19/006',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-16/003',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-14/002',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-03-26/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-03-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-03-12/002',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-03-14/002',
'/Volumes/witten/Alex/Data/Subjects/dop_20/2021-04-16/001',
'/Volumes/witten/Alex/Data/Subjects/dop_20/2021-04-21/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-02/001']

LIST_OF_SESSIONS_CHR2_GOOD_REC = \
['/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-19/002',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-05-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-30/001',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-20/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-17/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-19/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-23/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-02/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-04/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-07/002',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-09/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-11/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-15/001']


LIST_OF_SESSIONS_ILANA = \
['/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-19/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-17/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-19/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-23/001']



LIST_OF_SESSIONS_ALEX = \
['/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-19/002',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-20/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-17/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-19/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-23/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-02/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-04/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-07/002',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-09/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-11/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-15/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-05-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-30/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-26/001']

ALL_NEW_SESSIONS = [
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-16/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',  
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-19/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-27/003', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-10/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-22/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-23/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-26/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-28/001',
'/Volumes/witten/Alex/Data/Subjects/dop_52/2022-10-02/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-04/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-03/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-02/001']

LASER_ONLY = [
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-16/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',  
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-19/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-27/003', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-10/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-06/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-04/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-03/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-02/001']

DUAL_TASK = ['/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-22/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-23/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-26/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-28/001',
'/Volumes/witten/Alex/Data/Subjects/dop_52/2022-10-02/001']

def get_binned_spikes(spike_times, spike_clusters, cluster_id, epoch_time,
    pre_time=0.5,post_time=1.0, bin_size=0.025, smoothing=0.025, return_fr=True):
    binned_firing_rate = calculate_peths(
    spike_times, spike_clusters, cluster_id, epoch_time,
    pre_time=pre_time,post_time=post_time, bin_size=bin_size,
    smoothing=smoothing, return_fr=return_fr)[1]
    return binned_firing_rate

def add_transition_info(ses_d, trials_forward=10):
    trials_back=5 # Current cannot be changed
    ses_df = ses_d.copy()
    ses_df['choice_1'] = ses_df['choice']>0
    ses_df['transition_analysis'] = np.nan
    ses_df['transition_type'] = np.nan
    ses_df['transition_analysis_real'] = np.nan
    ses_df['transition_type_real'] = np.nan
    ses_df.loc[ses_df['trial_within_block']<trials_forward,'transition_analysis']=1
    ses_df['transition_analysis_real']=1
    for i in np.arange(len(ses_df['block_number'].unique())):
        if i>0:
            ses_ses_past = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']<0)]

            ses_ses_next = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']>=0)]

            ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']<0) ,'transition_type'] = \
                                    ses_ses_past['probabilityLeft'].astype(str)+ \
                                    ' to '\
                                   +ses_ses_past['probabilityLeft_next'].astype(str)

            ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']>=0) ,'transition_type'] = \
                                    ses_ses_next['probabilityLeft_past'].astype(str)+ \
                                     ' to '\
                                   +ses_ses_next['probabilityLeft'].astype(str)

            blocks = np.array([0.1,0.7])
            ses_ses_next_real = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis_real']==1) &
            (ses_df['trial_within_block_real']>=0)]
            past_block = blocks[blocks!=ses_ses_next_real['probabilityLeft'].iloc[0]]

            ses_df.loc[(ses_df['block_number_real']==i) &
            (ses_df['transition_analysis_real']==1) &
            (ses_df['trial_within_block_real']>=0) ,'transition_type_real'] = \
                                    ses_ses_next_real['probabilityLeft'].iloc[0].astype(str) + \
                                     ' to '\
                                   + str(past_block[0])

    return ses_df

def trial_within_block(behav):
    behav['trial_within_block'] = np.nan
    behav['block_number'] = np.nan
    behav['trial_within_block_real'] = np.nan
    behav['probabilityLeft_next'] = np.nan # this is for plotting trials before block change
    behav['opto_block_next'] = np.nan # this is for plotting trials before block change
    behav['probabilityLeft_past'] = np.nan # this is for plotting trials before block change
    behav['opto_block_past'] = np.nan # this is for plotting trials before block change
    behav['block_change'] = np.concatenate([np.zeros(1),
                                            1*(np.diff(behav['probabilityLeft'])!=0)])
    block_switches = np.concatenate([np.zeros(1),
                                     behav.loc[behav['block_change']==1].index]).astype(int)
    col_trial_within_block = np.where(behav.columns == 'trial_within_block')[0][0]
    col_probabilityLeft_next = np.where(behav.columns == 'probabilityLeft_next')[0][0]
    col_block_number = np.where(behav.columns == 'block_number')[0][0]
    col_opto_block_next = np.where(behav.columns == 'opto_block_next')[0][0]
    col_opto_block_past = np.where(behav.columns == 'opto_block_past')[0][0]
    col_opto_probabilityLeft_past = np.where(behav.columns == 'probabilityLeft_past')[0][0]
    col_trial_within_block_real = np.where(behav.columns == 'trial_within_block_real')[0][0]
    for i in np.arange(len(block_switches)):
        if i == 0:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]:block_switches[i+1], col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i+1]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_next] = \
                        behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_next] = \
            np.arange(block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], col_block_number] = i
            behav.iloc[block_switches[i]:block_switches[i+1], col_trial_within_block_real] = \
            np.arange(len(behav.iloc[block_switches[i]:block_switches[i+1], col_block_number]))
        elif i == len(block_switches)-1:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:, col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]-5:, col_opto_block_next] = \
                behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:, col_opto_probabilityLeft_past] = \
                behav['probabilityLeft'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:, col_opto_block_past] = \
                behav['opto_block'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:, col_trial_within_block] = \
                np.arange(-5, len(behav) - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:, col_block_number] = i
            behav.iloc[block_switches[i]:, col_trial_within_block_real] = np.arange(len(behav.iloc[block_switches[i]:, col_block_number]))
        else:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_opto_block_next] = \
                behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_probabilityLeft_past] = \
                behav['probabilityLeft'][block_switches[i-1]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_past] = \
                behav['opto_block'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_trial_within_block] = \
                np.arange(-5, block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], col_block_number] = i
            behav.iloc[block_switches[i]:block_switches[i+1], col_trial_within_block_real] = np.arange(len(behav.iloc[block_switches[i]:block_switches[i+1], col_block_number]))
    #Assign next block to negative within trial
    behav['block_number_real'] = behav['block_number'].copy()
    behav.loc[behav['trial_within_block']<0,'block_number'] = \
                behav.loc[behav['trial_within_block']<0,'block_number']+1
    behav.loc[behav['trial_within_block']>=0,'opto_block_next'] = np.nan
    behav.loc[behav['trial_within_block']>=0,'probabilityLeft_next'] = np.nan
    behav.loc[behav['trial_within_block']<0,'opto_block_past'] = np.nan
    behav.loc[behav['trial_within_block']<0,'probabilityLeft_past'] = np.nan
    return behav


def average_hz_trial(cluster_id, spike_clusters, spike_times, epoch, window=-1):
    # Epoch - e.g go_cue_times
    # Window - window from epoch in seconds to calculate rate
    spikes_from_cluster = spike_times[np.where(spike_clusters==cluster_id)[0]]
    rate_trial = []
    for i in np.arange(len(epoch)):
        if window>0:
            rate = np.sum(np.where((spikes_from_cluster >= epoch[i])
                                    & (spikes_from_cluster <= epoch[i]+window)))/abs(window)
            rate_trial.append(rate)
        if window<0:
            rate = len(np.where((spikes_from_cluster <= epoch[i])
                                            & (spikes_from_cluster >= epoch[i]+window))[0])/abs(window)
            rate_trial.append(rate)
    return np.array(rate_trial)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))




######## Classes
class probe:
    def __init__(self,path):
        self.spike_times = np.load(path+'/spikes.times.npy')
        self.spike_clusters = np.load(path+'/spikes.clusters.npy')
        self.cluster_channels = np.load(path+'/clusters.channels.npy')
        self.channel_locations = np.load(path+'/channels.locations.npy', allow_pickle = True)
        self.channel_xyz = pd.read_json(path + '/channel_locations.json', orient='index').iloc[:-1,:3].to_numpy()/1000000
        self.cluster_xyz = self.channel_xyz[self.cluster_channels,:]
        self.channel_xyz  = np.load(path+'/channels.localCoordinates.npy')
        self.channel_hem = np.load(path+'/channels.hemisphere.npy')
        self.cluster_hem = self.channel_hem[self.cluster_channels]
        self.cluster_locations = self.channel_locations[self.cluster_channels]
        metrics = pd.read_csv(path+'/clusters.metrics.csv')
        self.cluster_metrics = [None] * (metrics['cluster_id'].max()+1)
        for i in metrics.cluster_id.to_numpy():
            self.cluster_metrics[i] = metrics.loc[metrics['cluster_id']==i, 'group'].to_list()[0]
        self.cluster_selection = np.load(path+'/clusters_selection.npy')
        try:
            self.cluster_id = pd.read_csv(path+'/clusters.metrics.csv')['cluster_id']
        except:
            self.cluster_id = pd.read_csv(path+'/clusters.metrics.csv')['id']           
        try:
            try:
                groups = pd.read_csv('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
            except:
                groups = pd.read_csv('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
            #groups = pd.read_csv('/mnt/s0/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
            groups = groups.iloc[:,1:3]
            groups = groups.set_index('original')
            group_dict = groups.to_dict()['group']
            self.cluster_group_locations = \
                pd.Series(self.cluster_locations).map(group_dict)
        except:
            print("Simplified locations not available")

class alf_ephys:
    def __init__(self, n_probes):
        self.probes = [None]*n_probes
    def __setitem__(self, probe_number, data):
        self.probes[probe_number] = data
    def __getitem__(self, probe_number):
        return self.probes[probe_number]

class alf:
    def __init__(self,path, ephys=False):
        self.choice  = -1*(np.load(path+'/alf/_ibl_trials.choice.npy'))
        self.outcome  = (np.load(path+'/alf/_ibl_trials.feedbackType.npy')>0)*1
        self.left_reward  = np.load(path+'/alf/_ibl_trials.left_reward.npy')
        self.right_reward  = np.load(path+'/alf/_ibl_trials.right_reward.npy')
        self.probabilityLeft  = np.load(path+'/alf/_ibl_trials.probabilityLeft.npy')
        self.no_reward_block = False
        if Path(path+'/alf/forgetting_QRlaser.npy').is_file()==True:
            self.opto_block = np.load(path+'/alf/_ibl_trials.opto_block.npy')
        else:
            self.opto_block = np.zeros(len(self.choice))
            print('No reward type blocks')
            self.no_reward_block = True
        self.laser_reward  = (1*(np.load(path+'/alf/_ibl_trials.rewardVolume.npy')>0)) * self.opto_block
        self.water_reward = (1*(np.load(path+'/alf/_ibl_trials.rewardVolume.npy')>0)) \
                            * (1*(self.opto_block==0))
        self.goCue_trigger_times = np.load(path+'/alf/_ibl_trials.goCue_times.npy')
        self.stimOn_times = np.load(path+'/alf/_ibl_trials.stimOn_times.npy')
        self.response_times = np.load(path+'/alf/_ibl_trials.response_times.npy')
        self.start_time = np.load(path+'/alf/_ibl_trials.intervals.npy')[:,0]
        self.first_move = np.load(path+'/alf/_ibl_trials.firstMovement_times.npy')
        try:
            self.firstlaser_times = np.load(path+'/alf/_ibl_trials.first_laser_times.npy')
        except:
            self.firstlaser_times = np.array([np.nan])
        if os.path.isfile(path+'/alf/standard_QL.npy'):
            self.QL = np.load(path+'/alf/standard_QL.npy')
            self.QR = np.load(path+'/alf/standard_QR.npy')
            if self.no_reward_block==False:
                self.QLlaser = np.load(path+'/alf/standard_QLlaser.npy')
                self.QRlaser = np.load(path+'/alf/standard_QRlaser.npy')
            self.QLstay  = np.load(path+'/alf/standard_QLstay.npy')
            self.QRstay  = np.load(path+'/alf/standard_QRstay.npy')
            self.QLreward  = np.load(path+'/alf/standard_QLreward.npy')
            self.QRreward  = np.load(path+'/alf/standard_QRreward.npy')
            self.choice_prediction  = np.load(path+'/alf/standard_choice_prediction.npy')
            self.accuracy = np.mean((1*(self.choice>0))==(1*(self.choice_prediction>0.5)))
        
        if os.path.isfile(path+'/alf/forgetting_QL.npy'):
            self.fQL = np.load(path+'/alf/forgetting_QL.npy')
            self.fQR = np.load(path+'/alf/forgetting_QR.npy')
            if self.no_reward_block==False:
                self.fQLlaser = np.load(path+'/alf/forgetting_QLlaser.npy')
                self.fQRlaser = np.load(path+'/alf/forgetting_QRlaser.npy')
            self.fQLstay  = np.load(path+'/alf/forgetting_QLstay.npy')
            self.fQRstay  = np.load(path+'/alf/forgetting_QRstay.npy')
            self.fQLreward  = np.load(path+'/alf/forgetting_QLreward.npy')
            self.fQRreward  = np.load(path+'/alf/forgetting_QRreward.npy')
            self.fchoice_prediction  = np.load(path+'/alf/forgetting_choice_prediction.npy')
            self.faccuracy = np.mean((1*(self.choice>0))==(1*(self.fchoice_prediction>0.5)))
        
        if os.path.isfile(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_laser.npy'):
            if self.no_reward_block==False:
                self.DQlaser = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_laser.npy')
            self.DQwater = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_water.npy')
            self.RQLstay  = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_QLstay.npy')
            self.RQRstay  = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_QRstay.npy')
            if self.no_reward_block==False:
                self.DQ = self.DQlaser + self.DQwater + (self.QRstay - self.QLstay)
            else:
                self.DQ = self.DQwater + (self.QRstay - self.QLstay)
            self.reinforce_choice_prediction  = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_choice_prediction.npy')
            self.raccuracy = np.mean((1*(self.choice>0))==(1*(self.reinforce_choice_prediction>0.5)))

        if ephys==True: # Clunky section to be streamlined
            probe_paths = sorted(glob(path +'/alf/*[0-9]*/'))
            self.probe = alf_ephys(len(probe_paths))
            for p, p_path in enumerate(probe_paths):
                if Path(p_path+'/pykilosort').is_dir():
                    p_path=p_path+'/pykilosort'
                self.probe[p] = probe(p_path)

    def to_df(self):
        return pd.DataFrame.from_dict(self.__dict__)

    def fr_bytrial(self, cluster_id, aligning_var,
                        probe_name='probe00', window=-1):
        prb = getattr(self,probe_name)
        bytrial = average_hz_trial(cluster_id, prb.spike_clusters, prb.spike_times, aligning_var,
                         window=window)
        return bytrial

    def plot_session(self):
        example = self.to_df()
        example['choice_r'] = (example['choice']==1)*1
        example['choice_l'] = (example['choice']==-1)*1
        if self.no_reward_block==False: 
            example['value_laser'] = example['fQRlaser']-example['fQLlaser'] 
        example['value_reward'] = example['fQRreward']-example['fQLreward']
        example['value_stay'] = example['fQRstay']-example['fQLstay']
        example['probabilityRight']=0.1
        example.loc[example['probabilityLeft']==0.1, 'probabilityRight'] = 0.7
        example['reward_r'] = example['outcome']*example['choice_r']*(1*(example['opto_block']!=1))
        example['reward_l'] = example['outcome']*example['choice_l']*(1*(example['opto_block']!=1))
        if self.no_reward_block==False:
            example['laser_r'] = example['outcome']*example['choice_r']*example['opto_block']
            example['laser_l'] = example['outcome']*example['choice_l']*example['opto_block']
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=2,
                                 height_ratios=[2, 5])
        spec.update(wspace=0.025, hspace=0.05)
        ax1 = fig.add_subplot(spec[1])
        ax1.plot(example['choice_r'].rolling(10, center=False).mean(),color='k')
        ax1.plot(example['fchoice_prediction'].rolling(10,center=False).mean(),color='k', linestyle='dashed', linewidth=2)
        ax1.spines['top'].set_visible(False)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        if self.no_reward_block==False:
            ax2.plot(example['value_laser'].rolling(10, center=False).mean(),color='orange', linestyle='dashed', linewidth=2)
        ax2.plot(example['value_reward'].rolling(10, center=False).mean(),color='dodgerblue', linestyle='dashed', linewidth=2)
        ax2.spines['top'].set_visible(False)
        ax2.plot(example['value_stay'].rolling(10, center=False).mean(),color='gray', linestyle='dashed', linewidth=2)
        # Only show ticks on the left and bottom spines
        ax1.set_ylim(-0.1,1.1)
        ax1.set_ylabel('Fraction of right choices')
        ax2.set_ylabel('QR-QL')
        ax3 = fig.add_subplot(spec[0], sharex=ax1)
        ax3.plot(example['probabilityRight'],color='k',
                 linestyle='--', alpha =0.5)
        #plt.xlim(0,400)
        plt.sca(ax3)
        plt.vlines(np.where(example['choice']==1),0.925,1.0, color='k')
        plt.vlines(np.where(example['choice']==-1),-0.225,-0.15, color='k')
        plt.vlines(np.where(example['reward_l']==1),-0.225,-0.15, color='dodgerblue')
        plt.vlines(np.where(example['reward_r']==1),0.925,1.0, color='dodgerblue')
        if self.no_reward_block==False:
            plt.vlines(np.where(example['laser_r']==1),0.925,1.0, color='orange')
            plt.vlines(np.where(example['laser_l']==1),-0.225,-0.15, color='orange')
        plt.axis('off')
        plt.ylabel('Reward probability')
        return fig

    def plot_session_REINFORCE(self):
        example = self.to_df()
        example['choice_r'] = (example['choice']==1)*1
        example['choice_l'] = (example['choice']==-1)*1
        example['value_laser'] = example['DQlaser']
        example['value_reward'] = example['DQwater']
        example['value_stay'] = example['RQRstay']-example['RQLstay']
        example['probabilityRight']=0.1
        example.loc[example['probabilityLeft']==0.1, 'probabilityRight'] = 0.7
        example['reward_r'] = example['outcome']*example['choice_r']*(1*(example['opto_block']!=1))
        example['reward_l'] = example['outcome']*example['choice_l']*(1*(example['opto_block']!=1))
        example['laser_r'] = example['outcome']*example['choice_r']*example['opto_block']
        example['laser_l'] = example['outcome']*example['choice_l']*example['opto_block']
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=2,
                                 height_ratios=[2, 5])
        spec.update(wspace=0.025, hspace=0.05)
        ax1 = fig.add_subplot(spec[1])
        ax1.plot(example['choice_r'].rolling(10, center=False).mean(),color='k')
        ax1.plot(example['reinforce_choice_prediction'].rolling(10, center=False).mean(),color='k', linestyle='dashed', linewidth=2)
        ax1.spines['top'].set_visible(False)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.plot(example['value_laser'].rolling(10, center=False).mean(),color='orange', linestyle='dashed', linewidth=2)
        ax2.plot(example['value_reward'].rolling(10, center=False).mean(),color='dodgerblue', linestyle='dashed', linewidth=2)
        ax2.spines['top'].set_visible(False)
        ax2.plot(example['value_stay'].rolling(10, center=False).mean(),color='gray', linestyle='dashed', linewidth=2)
        # Only show ticks on the left and bottom spines
        ax1.set_ylim(-0.1,1.1)
        ax1.set_ylabel('Fraction of right choices')
        ax2.set_ylabel('DQ')
        ax3 = fig.add_subplot(spec[0], sharex=ax1)
        ax3.plot(example['probabilityRight'],color='k',
                 linestyle='--', alpha =0.5)
        #plt.xlim(0,400)
        plt.sca(ax3)
        plt.vlines(np.where(example['choice']==1),0.925,1.0, color='k')
        plt.vlines(np.where(example['choice']==-1),-0.225,-0.15, color='k')
        plt.vlines(np.where(example['reward_l']==1),-0.225,-0.15, color='dodgerblue')
        plt.vlines(np.where(example['reward_r']==1),0.925,1.0, color='dodgerblue')
        plt.vlines(np.where(example['laser_r']==1),0.925,1.0, color='orange')
        plt.vlines(np.where(example['laser_l']==1),-0.225,-0.15, color='orange')
        plt.axis('off')
        plt.ylabel('Reward probability')
        return fig

    def plot_correlations(self):
        QR = ses.QRreward
        QRl = ses.QRlaser
        QRs = ses.QRstay
        QL = ses.QLreward
        QLl = ses.QLlaser
        QLs = ses.QLstay
        QRQL = QR-QL
        QRlQLl = QRl - QLl
        QRsQLs = QRs - QLs
        fig, ax  = plt.subplots(4,3)
        plt.sca(ax[0,0])
        plt.scatter(QR, QRl)
        plt.xlabel('QR reward')
        plt.ylabel('QR Laser')
        plt.sca(ax[0,1])
        plt.scatter(QR, QRs)
        plt.xlabel('QR reward')
        plt.ylabel('QR stay')
        plt.sca(ax[0,2])
        plt.scatter(QRl, QRs)
        plt.xlabel('QR Laser')
        plt.ylabel('QR stay')
        plt.sca(ax[1,0])
        plt.scatter(QL, QLl)
        plt.xlabel('QL reward')
        plt.ylabel('QL Laser')
        plt.sca(ax[1,1])
        plt.scatter(QL, QLs)
        plt.xlabel('QL reward')
        plt.ylabel('QL stay')
        plt.sca(ax[1,2])
        plt.scatter(QLl, QLs)
        plt.xlabel('QL Laser')
        plt.ylabel('QL stay')
        plt.sca(ax[2,0])
        plt.scatter(QRQL, QRlQLl)
        plt.xlabel('QR-QL reward')
        plt.ylabel('QR-QL Laser')
        plt.sca(ax[2,1])
        plt.scatter(QRQL, QRsQLs)
        plt.xlabel('QR-QL reward')
        plt.ylabel('QR-QL stay')
        plt.sca(ax[2,2])
        plt.scatter(QRlQLl, QRsQLs)
        plt.xlabel('QR-QL Laser')
        plt.ylabel('QR-QL stay')
        plt.sca(ax[3,0])
        plt.scatter(np.arange(len(QR)), QR+QL)
        plt.ylabel('QR+QL reward')
        plt.xlabel('Trial')
        plt.sca(ax[3,1])
        plt.scatter(np.arange(len(QR)), QRl+QLl)
        plt.ylabel('QR+QL laser')
        plt.xlabel('Trial')
        plt.sca(ax[3,2])
        plt.scatter(np.arange(len(QR)), QR+QL+QRl+QLl+QRs+QLs)
        plt.ylabel('QR+QL')
        plt.xlabel('Trial')
        plt.tight_layout()
        sns.despine()

class ephys_behavior_dataset:
        def __init__(self,LIST_OF_SESSIONS_CHR2_GOOD_REC, start=0,end=-150): #start to end which trials to include
            self.sessions = pd.DataFrame()
            for ses in LIST_OF_SESSIONS_CHR2_GOOD_REC:
                print(ses)
                ses_df = alf(ses, ephys=False).to_df()
                ses_df = ses_df.iloc[start:end,:]
                ses_df['mouse'] = Path(ses).parent.parent.name
                ses_df['date'] = Path(ses).parent.name
                ses_df['ses'] = Path(ses).name
                ses_df['path'] = ses
                ses_df = trial_within_block(ses_df)
                ses_df = add_transition_info(ses_df)
                self.sessions = pd.concat([self.sessions, ses_df])
            self.sessions = self.sessions.reset_index()

        def plot_stay(self):
            byanimal = self.sessions
            byanimal['repeated'] = (1*(byanimal['choice']==byanimal['choice'].shift(1)))
            byanimal['prev_reward'] = byanimal['outcome'].shift(1)
            byanimal = byanimal.groupby(['mouse','opto_block',
                        'prev_reward']).mean()['repeated'].reset_index()
            p_water = pttest(byanimal.loc[(byanimal['prev_reward']==0) & (byanimal['opto_block']==0),'repeated'],
                    byanimal.loc[(byanimal['prev_reward']==1) & (byanimal['opto_block']==0),'repeated'])[1]
            p_laser = pttest(byanimal.loc[(byanimal['prev_reward']==0) & (byanimal['opto_block']==1),'repeated'],
                    byanimal.loc[(byanimal['prev_reward']==1) & (byanimal['opto_block']==1),'repeated'])[1]
            water_stars = lr.num_star(p_water)
            laser_stars = lr.num_star(p_laser)

            fig, ax = plt.subplots(1,2,sharey=True)
            plt.sca(ax[0])
            sns.barplot(x='prev_reward', y='repeated', ci=0,
                data=byanimal.loc[byanimal['opto_block']==0],
                palette=['grey','dodgerblue'],zorder=0)
            sns.pointplot(x='prev_reward', y='repeated', hue='mouse',
                data=byanimal.loc[byanimal['opto_block']==0],
                color='k',  linewidth=1)
            plt.ylim(0,1)
            plt.ylabel('Repeated Choices (%)')
            plt.xlabel('Previous Water Reward')
            ax[0].get_legend().remove()
            plt.annotate(water_stars+' p=%s' %str(round(p_water,4)), xy= [0.05, 0.9], fontsize=12)
            plt.sca(ax[1])
            sns.barplot(x='prev_reward', y='repeated', ci=0,
                data=byanimal.loc[byanimal['opto_block']==1],
                palette=['grey','orange'],zorder=0)
            sns.pointplot(x='prev_reward', y='repeated', hue='mouse',
                data=byanimal.loc[byanimal['opto_block']==1],
                color='k', linewidth=1)
            plt.annotate(laser_stars+' p=%s' %str(round(p_laser,4)), xy= [0.05, 0.9], fontsize=12)
            plt.ylabel('  ')
            plt.ylim(0,1)
            plt.xlabel('Previous Laser')
            ax[1].get_legend().remove()
            sns.despine()

        def stats_stay(self):
            byanimal = self.sessions
            byanimal['repeated'] = (1*(byanimal['choice']==byanimal['choice'].shift(1)))
            byanimal['prev_reward'] = byanimal['outcome'].shift(1)
            byanimal = byanimal.groupby(['mouse','opto_block',
                                    'prev_reward']).mean()['repeated'].reset_index()
            laser_test = pairedt(byanimal.loc[(byanimal['opto_block']==1)&
                    (byanimal['prev_reward']==0),'repeated'],
                    byanimal.loc[(byanimal['opto_block']==1)&
                    (byanimal['prev_reward']==1),'repeated'])
            water_test = pairedt(byanimal.loc[(byanimal['opto_block']==0)&
                    (byanimal['prev_reward']==0),'repeated'],
                    byanimal.loc[(byanimal['opto_block']==0)&
                    (byanimal['prev_reward']==1),'repeated'])
            return water_test, laser_test

        def plot_transition(self):
            negative_trials = self.sessions.loc[self.sessions['trial_within_block']<0].copy()
            positive_trials = self.sessions.loc[self.sessions['trial_within_block_real']>=0].copy()
            positive_trials['trial_within_block']=positive_trials['trial_within_block_real']
            positive_trials['transition_type']=positive_trials['transition_type_real']
            ses_df = pd.concat([negative_trials,positive_trials])
            sns.lineplot(data = ses_df.loc[(ses_df['transition_type']=='0.1 to 0.7') |
                 (ses_df['transition_type']=='0.7 to 0.1')].reset_index(), x='trial_within_block', y='choice_1',
                 ci=68, hue='opto_block', err_style='bars', style='transition_type', palette=['dodgerblue','orange'])
            plt.ylim(0,1)
            plt.xlim(-5,15)
            plt.vlines(0,0,1,linestyles='dashed', color='k')
            plt.ylabel('% Right Choices')
            plt.xlabel('Trials from block switch')
            sns.despine()

        def average_block_length(self):
            return self.sessions.groupby(['mouse','ses','date','block_number_real']).count().median()
        def plot_block_stats(self):
            fig,ax =plt.subplots(1,4)
            # Trials per block
            plt.sca(ax[0])
            sns.histplot(self.sessions.groupby(['mouse','ses','date','block_number_real']).count()['choice'],
                color='gray', stat='probability')
            plt.xlabel('Block Length')
            plt.ylim(0,0.2)
            plt.ylabel('Fraction of blocks')
            sns.despine()
            # Number of block per session
            plt.sca(ax[1])
            sns.swarmplot(y=self.sessions.groupby(['mouse','ses','date'])['block_number_real'].max(), color='k')
            sns.barplot(y=self.sessions.groupby(['mouse','ses','date'])['block_number_real'].max(), color='gray')
            plt.xlabel('Sessions')
            plt.ylabel('n Blocks per session')
            plt.ylim(0,40)
            sns.despine()
            # Trials per session
            plt.sca(ax[2])
            sns.swarmplot(y=self.sessions.groupby(['mouse','ses','date']).count()['choice'], color='k')
            sns.barplot(y=self.sessions.groupby(['mouse','ses','date']).count()['choice'], color='gray')
            plt.xlabel('Sessions')
            plt.ylabel('n Trials per session')
            plt.ylim(0,1000)
            sns.despine()
            plt.tight_layout()
            # Rewards per session
            plt.sca(ax[3])
            sns.swarmplot(y=self.sessions.groupby(['mouse','ses','date']).count()['outcome'], color='k')
            sns.barplot(y=self.sessions.groupby(['mouse','ses','date']).count()['outcome'], color='gray')
            plt.xlabel('Sessions')
            plt.ylabel('Fraction of trials rewarded')
            plt.ylim(0,1)
            sns.despine()
            plt.tight_layout()
            # Rewards per session
            plt.sca(ax[3])
            sns.swarmplot(y=self.sessions.groupby(['mouse','ses','date']).mean()['outcome'], color='k')
            sns.barplot(y=self.sessions.groupby(['mouse','ses','date']).mean()['outcome'], color='gray')
            plt.xlabel('Sessions')
            plt.ylabel('Fraction of trials rewarded')
            plt.ylim(0,1)
            sns.despine()
            plt.tight_layout()


        def plot_performance(self):
            # Performance
            sns.barplot(data = self.sessions.groupby(['mouse','ses','date','opto_block']).mean()['outcome'].reset_index(),
                        x = 'opto_block', y='outcome', palette=['dodgerblue','orange'])
            sns.swarmplot(data = self.sessions.groupby(['mouse','ses','date','opto_block']).mean()['outcome'].reset_index(),
                        x = 'opto_block', y='outcome', color='k')
            sns.despine()
            plt.xticks([0,1],['Water', 'Laser'])
            plt.ylim(0,0.60)
            plt.xlabel('Reward block')
            plt.ylabel('Fraction of rewarded trials')

        def plot_logistic_regression(self):
            ses_df = lr.add_laser_block_regressors(self.sessions)
            params = lr.fit_GLM_w_laser_10(ses_df)
            lr.plot_GLM_blocks(params)

class ephys_ephys_dataset:
        def __init__(self,n_sessions):
            self.sessions = [None]*n_sessions
        def __setitem__(self, ses_number, data):
            self.sessions[ses_number] = data
        def __getitem__(self, ses_number):
            return self.sessions[ses_number]
        def get_regions(self, save=False):
            try:
                groups = pd.read_csv('/Users/alex/Documents/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
                return groups.group.unique()
            except:
                all_areas = []
                simplified_regions = pd.DataFrame()
                for i in range(len(self.sessions)):
                     ses_data = sessions[i]
                     for p in range(len(ses_data.probe.probes)):
                         prb = ses_data.probe[p]
                         all_areas.append(prb.cluster_locations[prb.cluster_metrics=='good'])
                all_areas = [item for sublist in all_areas for item in sublist]
                simplified_regions['original']= np.unique(all_areas)
                simplified_regions['group'] = np.nan
                simplified_regions['n_neurons'] = np.nan
                for reg in np.unique(all_areas):
                    n_neuron_good = 0
                    for i in range(len(sessions.sessions)):
                         ses_data = sessions[i]
                         for p in range(len(ses_data.probe.probes)):
                             prb = ses_data.probe[p]
                             n_neuron_all = 1*(prb.cluster_locations==reg)
                             n_neuron_good += np.sum(n_neuron_all[prb.cluster_metrics=='good'])
                    simplified_regions.loc[simplified_regions['original']==reg,'n_neurons']=\
                             n_neuron_good
                    if save==True:
                        simplified_regions.to_csv('simplified_regions_raw.csv')
                    return simplified_regions['original'].unique()

def plot_connectivity_map(SESSIONS, criterion=['good'], n_neurons_minimum = 20):
    pooled_region_info = pd.DataFrame()
    for ses in SESSIONS:
        print(ses)
        alfio = alf(ses, ephys=True)
        region_info = pd.DataFrame()
        for hemisphere in np.array([0,1]):
            regions = pd.DataFrame()
            for probe_id in np.arange(len(alfio.probe.probes)):
                        unique_regions = alfio.probe[probe_id].cluster_group_locations[np.where(
                            np.isin(alfio.probe[probe_id].cluster_metrics,criterion) & 
                            (alfio.probe[probe_id].cluster_hem==hemisphere))[0]].value_counts()
                        unique_regions = unique_regions[unique_regions>=n_neurons_minimum]
                        regions = pd.concat([regions,unique_regions])
            regions = regions.reset_index().groupby('index').sum().reset_index()
            regions['hemisphere'] = hemisphere
            region_info=pd.concat([region_info, regions])
        region_info['mouse'] = Path(ses).parent.parent.name
        region_info['date'] = Path(ses).parent.name
        region_info['ses'] = Path(ses).name
        region_info['id'] = region_info['mouse']+region_info['date']+region_info['ses']+ region_info['hemisphere'].astype(str)
        pooled_region_info = pd.concat([pooled_region_info,region_info])

    # Plot connectivity map
    chord_data = pooled_region_info.copy()
    selected_regions = np.array(['OFC','PFC', 'NAc', 'PFC', 'MO', 'DMS', 'VP', 'DLS', 'SS', 'GPe'])
    summary = np.zeros([len(selected_regions),len(selected_regions)])
    chord_data = chord_data.loc[np.isin(chord_data['index'],selected_regions)]
    for id in chord_data.id.unique():
        s_chord_data_r = chord_data.loc[chord_data['id']==id,'index']
        idx = [np.where(selected_regions==r)[0][0] for r in s_chord_data_r]
        lidx = np.array(list(permutations(idx,2)))
        for l in lidx:        
            summary[l[0],l[1]]+=1
    chord_diagram(summary, names=selected_regions, rotate_names=True, cmap='Dark2')
    return summary




def yield_by_region(yields):
        summary = yields.groupby(['regions']).sum().reset_index()
        #summary['bias'] = yields.groupby(['regions']).mean().reset_index()['bias']
        yields['relative_count'] = np.nan
        for reg  in summary.regions.unique():
            yields.loc[yields['regions']==reg, 'relative_count'] = yields.loc[yields['regions']==reg, 'count'].to_numpy() / \
                                                                    summary.loc[summary['regions']==reg,'count'].to_numpy()
        pen_by_reg = np.zeros(len(summary))
        mouse_by_reg = np.zeros(len(summary))
        accu_weighted_by_reg = np.zeros(len(summary))
        bias_weighted_by_reg = np.zeros(len(summary))
        accu_m_by_reg = np.zeros(len(summary))
        accu_sem_by_reg = np.zeros(len(summary))
        good_y = yields.loc[yields['count']>=20]
        good_y = good_y.groupby(['regions','mouse','id']).count().reset_index()
        model_performance =  yields.loc[yields['count']>=20].groupby(['regions','mouse','id']).mean().reset_index()['model_accuracy']
        good_y['model_performance'] = model_performance
        yields['model_performance_weighted'] =  yields['model_accuracy'] * yields['relative_count']
        yields['bias_weighted'] = yields['bias'] * yields['relative_count']
        for i, reg  in enumerate(summary.regions.unique()):
            pen_by_reg[i] = good_y.loc[good_y['regions']==reg].id.unique().shape[0]
            accu_m_by_reg[i]= good_y.loc[good_y['regions']==reg].model_performance.mean()
            accu_sem_by_reg[i]= good_y.loc[good_y['regions']==reg].model_performance.sem()
            mouse_by_reg[i] = good_y.loc[good_y['regions']==reg].mouse.unique().shape[0]
            accu_weighted_by_reg[i] = yields.loc[yields['regions']==reg].model_performance_weighted.sum()
            bias_weighted_by_reg[i] = yields.loc[yields['regions']==reg].bias_weighted.sum()
        summary['n_insertions'] = pen_by_reg
        summary['accuracy'] = accu_m_by_reg
        summary['accuracy_sem'] = accu_sem_by_reg
        summary['n_mice'] = mouse_by_reg
        summary['accuracy_weighted'] = accu_weighted_by_reg
        summary['bias_weighted'] = bias_weighted_by_reg    
        return summary.sort_values('accuracy_weighted', ascending=False)


if __name__=="__main__":
    # 1. Load all data
    sessions = ephys_ephys_dataset(len(ALL_NEW_SESSIONS))
    for i, ses in enumerate(ALL_NEW_SESSIONS):
            print(ses)
            ses_data = alf(ses, ephys=True)
            ses_data.mouse = Path(ses).parent.parent.name
            ses_data.date = Path(ses).parent.name
            ses_data.ses = Path(ses).name
            ses_data.trial_within_block = \
                        trial_within_block(ses_data.to_df())['trial_within_block']
            ses_data.trial_within_block_real = \
                        trial_within_block(ses_data.to_df())['trial_within_block_real']
            ses_data.block_number = \
                        trial_within_block(ses_data.to_df())['block_number']
            ses_data.block_number_real = \
                        trial_within_block(ses_data.to_df())['block_number_real']
            ses_data.probabilityLeft_next = \
                        trial_within_block(ses_data.to_df())['probabilityLeft_next']
            ses_data.probabilityLeft_past = \
                        trial_within_block(ses_data.to_df())['probabilityLeft_past']
            ses_data.transition_type = \
                        add_transition_info(ses_data.to_df())['transition_type']
            ses_data.transition_analysis = \
                        add_transition_info(ses_data.to_df())['transition_analysis']
            sessions[i] = ses_data
    
    # Load at unique regions
    loc = [] 
    for i in np.arange(len(ALL_NEW_SESSIONS)):
        ses = sessions[i]
        for j in np.arange(4):
            try:
                loc.append(np.unique(ses.probe[j].cluster_locations.astype(str)))
            except:
                continue
    unique_regions = np.unique(np.concatenate(loc))
    unique_regions = unique_regions[np.where(unique_regions!='nan')]
    # Look for any unreferenced regions
    groups = pd.read_csv('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
    groups = groups.iloc[:,1:3]
    groups = groups.set_index('original')
    group_dict = groups.to_dict()['group']
    current_regions = groups.original.unique()
    [group_dict[r] for r in current_regions] # This will error if dictionary is not complete

    # Stats by regions
    yields = pd.DataFrame()
    for i in np.arange(len(ALL_NEW_SESSIONS)):
        ses = sessions[i]
        for j in np.arange(4):
            try:
                prob = pd.DataFrame()
                good_units = ses.probe[j].cluster_selection
                prob[['regions','count']] = pd.Series(ses.probe[j].cluster_locations[good_units]).map(group_dict).value_counts().reset_index()
                prob['mouse'] = ses.mouse
                prob['date'] = ses.date
                prob['ses'] = ses.ses
                prob['probe'] = j 
                prob['id'] =  ses.mouse + '_' + ses.date + '_' + ses.ses + '_' + str(j) 
                yields = pd.concat([yields,prob])
            except:
                continue

    yield_by_region(yields)
    laser_mice = ['dop_47','dop_48','dop_49','dop_50','dop_53']
    laser_yield  = yields.loc[np.isin(yields['mouse'], laser_mice)]
    yield_by_region(laser_yield)
    water_yield  =  yields.loc[~(np.isin(yields['mouse'], laser_mice))]
    yield_by_region(water_yield)




    # Interconnected pairs
    mat = plot_connectivity_map(LASER_ONLY)
    mat = plot_connectivity_map(DUAL_TASK)
