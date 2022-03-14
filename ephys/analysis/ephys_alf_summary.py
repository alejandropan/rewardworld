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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale
import copy
import logistic_regression as lr
from scipy.stats import ttest_rel as pttest

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
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-12/001',
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



def get_binned_spikes(spike_times, spike_clusters, cluster_id, epoch_time,
    pre_time=0.2,post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True):
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

############################
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
############################

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
        self.cluster_locations = self.channel_locations[self.cluster_channels]
        self.cluster_metrics = pd.read_csv(path+'/clusters.metrics.csv')['group']
        try:
            self.cluster_id = pd.read_csv(path+'/clusters.metrics.csv')['cluster_id']
        except:
            self.cluster_id = pd.read_csv(path+'/clusters.metrics.csv')['id']           
        try:
            groups = pd.read_csv('/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
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
        self.opto_block = np.load(path+'/alf/_ibl_trials.opto_block.npy')
        self.laser_reward  = (1*(np.load(path+'/alf/_ibl_trials.rewardVolume.npy')>0)) * self.opto_block
        self.water_reward = (1*(np.load(path+'/alf/_ibl_trials.rewardVolume.npy')>0)) \
                            * (1*(self.opto_block==0))
        self.goCue_trigger_times = np.load(path+'/alf/_ibl_trials.goCue_times.npy')
        self.stimOn_times = np.load(path+'/alf/_ibl_trials.stimOn_times.npy')
        self.response_times = np.load(path+'/alf/_ibl_trials.response_times.npy')
        self.start_time = np.load(path+'/alf/_ibl_trials.intervals.npy')[:,0]
        self.first_move = np.load(path+'/alf/_ibl_trials.firstMovement_times.npy')


        if os.path.isfile(path+'/alf/QLearning_alphalaserdecay_QL.npy'):
            self.QL = np.load(path+'/alf/QLearning_alphalaserdecay_QL.npy')
            self.QR = np.load(path+'/alf/QLearning_alphalaserdecay_QR.npy')
            self.QLlaser = np.load(path+'/alf/QLearning_alphalaserdecay_QLlaser.npy')
            self.QRlaser = np.load(path+'/alf/QLearning_alphalaserdecay_QRlaser.npy')
            self.QLstay  = np.load(path+'/alf/QLearning_alphalaserdecay_QLstay.npy')
            self.QRstay  = np.load(path+'/alf/QLearning_alphalaserdecay_QRstay.npy')
            self.QLreward  = np.load(path+'/alf/QLearning_alphalaserdecay_QLreward.npy')
            self.QRreward  = np.load(path+'/alf/QLearning_alphalaserdecay_QRreward.npy')
            self.choice_prediction  = np.load(path+'/alf/QLearning_alphalaserdecay_choice_prediction.npy')
            self.accuracy = np.mean((1*(self.choice>0))==(1*(self.choice_prediction>0.5)))

            self.DQlaser = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_laser.npy')
            self.DQwater = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_water.npy')
            self.RQLstay  = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_QLstay.npy')
            self.RQRstay  = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_QRstay.npy')
            self.DQ = self.DQlaser + self.DQwater + (self.QRstay - self.QLstay)
            self.reinforce_choice_prediction  = np.load(path+'/alf/REINFORCE_mixedstay_alphalaserdecay_choice_prediction.npy')
            self.raccuracy = np.mean((1*(self.choice>0))==(1*(self.reinforce_choice_prediction>0.5)))


        if ephys==True: # Clunky section to be streamlined
            probe_paths = glob(path +'/alf/*[0-9]*/')
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
        example['value_laser'] = example['QRlaser']-example['QLlaser']
        example['value_reward'] = example['QRreward']-example['QLreward']
        example['value_stay'] = example['QRstay']-example['QLstay']
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
        ax1.plot(example['choice_prediction'].rolling(10,center=False).mean(),color='k', linestyle='dashed', linewidth=2)
        ax1.spines['top'].set_visible(False)
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
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


if __name__=="__main__":
    # 1. Load all data
    sessions = ephys_ephys_dataset(len(LIST_OF_SESSIONS_CHR2_GOOD_REC))
    for i, ses in enumerate(LIST_OF_SESSIONS_CHR2_GOOD_REC):
            print(ses)
            ses_data = alf(ses, ephys=False)
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
    '''


    # 2. Make a PCA plot
    trial_based=False # Whether we care about trials
    transition_based=True # 5 last , 5 first of every block type
    epoch_of_interest = 'goCue_trigger_times'
    pre_time=1.0
    post_time=0
    bin_size=1


    selected_regions = [['NAcc', 'NAcsh'], ['DMS','VMS'],
        'PFC', 'FA', 'VP', 'SPT', 'MOC']


    def very_good_clusters():
        sessions[s].probe[p].sp


    for reg in selected_regions:
        counter=0
        for s in np.arange(len(sessions.sessions)):
            for p in np.arange(len(sessions[s].probe.probes)):
                loc_clusters = \
                    np.where(np.isin(sessions[s].probe[p].cluster_group_locations,reg))[0]
                if len(loc_clusters)==0:
                    continue
                counter+=1
                good_clusters = \
                    np.where(sessions[s].probe[p].cluster_metrics=='good')[0]
                cluster_selection = np.intersect1d(loc_clusters,good_clusters)
                binned_fr = get_binned_spikes(sessions[s].probe[p].spike_times,
                                sessions[s].probe[p].spike_clusters,
                                cluster_selection, getattr(sessions[s],epoch_of_interest),
                                pre_time=pre_time, post_time=post_time,
                                bin_size=bin_size)/bin_size
                #Flatten matrix to n_neuron n_neuron x n_trials
                binned_fr_flat = np.squeeze(binned_fr,axis=2).T
                if transition_based==True:
                    left_block = np.where(sessions[s].probabilityLeft==0.7)[0]
                    right_block = np.where(sessions[s].probabilityLeft==0.1)[0]
                    water_block = np.where(sessions[s].opto_block==0)[0]
                    laser_block =  np.where(sessions[s].opto_block==1)[0]

                    # Select spikes based on interesting trials

                    left_water = np.intersect1d(left_block,water_block)
                    right_water = np.intersect1d(right_block,water_block)
                    left_laser = np.intersect1d(left_block,laser_block)
                    right_laser = np.intersect1d(right_block,laser_block)

                    X = np.zeros([binned_fr_flat.shape[0],40])#Start matrix that will hold data
                    X[:]=np.nan
                    # This matrix has been triple checked test below
                    for j,i in enumerate(np.arange(-5,5)):
                        if i<0:
                            X[:,j]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                        left_water)].mean(axis=1) # last 5 t of left water
                            X[:,10+j]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                        right_water)].mean(axis=1) # last 5 t of right water
                            X[:,20+j]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                        left_laser)].mean(axis=1) # last 5 t of left laser
                            X[:,30+j]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                        right_laser)].mean(axis=1) # last 5 t of right laser
                        if i>-1:
                            X[:,5+i]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                        left_water)].mean(axis=1) # first 5 t of left water
                            X[:,15+i]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                        right_water)].mean(axis=1) # first 5 t of right water
                            X[:,25+i]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                        left_laser)].mean(axis=1) # first 5 t of left laser
                            X[:,35+i]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                        right_laser)].mean(axis=1) # first 5 t of right laser

                    if counter==1:
                        X_reg=X
                    else:
                        X_reg = np.concatenate([X_reg, X])



        #Plot PCA 3D
        Z_reg = scale(X_reg.T)
        pca = PCA(n_components=3)
        pca.fit(Z_reg)
        DX = pca.transform(Z_reg)
        fig = plt.figure(1, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        ax.scatter(DX[:4, 0], DX[:4, 1], DX[:4, 2], color='blue') # last 5 t of left water
        ax.plot(DX[:5, 0], DX[:5, 1], DX[:5, 2], color='blue') # last 5 t of left water
        ax.scatter(DX[4, 0], DX[4, 1], DX[4, 2], marker='>', color='blue', s=50) # last 5 t of left water

        ax.scatter(DX[25:29, 0], DX[25:29, 1], DX[25:29, 2], color='salmon') # first 5 t of right water
        ax.plot(DX[25:30, 0], DX[25:30, 1], DX[25:30, 2], color='salmon') # first 5 t of right water
        ax.scatter(DX[29, 0], DX[29, 1], DX[29, 2], marker='>', color='salmon', s=50) # last 5 t of left water

        ax.scatter(DX[5:9, 0], DX[5:9, 1], DX[5:9, 2], color='red') # last 5 t of right water
        ax.plot(DX[5:10, 0], DX[5:10, 1], DX[5:10, 2], color='red') # last 5 t of right water
        ax.scatter(DX[9, 0], DX[9, 1], DX[9, 2], marker='>', color='red', s=50) # last 5 t of left water

        ax.scatter(DX[20:24, 0], DX[20:24, 1], DX[20:24, 2], color='deepskyblue') # first 5 t of left water
        ax.plot(DX[20:25, 0], DX[20:25, 1], DX[20:25, 2], color='deepskyblue') # first 5 t of left water
        ax.scatter(DX[24, 0], DX[24, 1], DX[24, 2], marker='>', color='deepskyblue', s=50) # last 5 t of left water

        ax.scatter(DX[10:14, 0], DX[10:14, 1], DX[10:14, 2], color='magenta') # last 5 t of left laser
        ax.plot(DX[10:15, 0], DX[10:15, 1], DX[10:15, 2], color='magenta') # last 5 t of left laser
        ax.scatter(DX[14, 0], DX[14, 1], DX[14, 2], marker='>', color='magenta', s=50) # last 5 t of left water

        ax.scatter(DX[35:39, 0], DX[35:39, 1], DX[35:39, 2], color='navajowhite') # first 5 t of right laser
        ax.plot(DX[35:40, 0], DX[35:40, 1], DX[35:40, 2], color='navajowhite') # first 5 t of right laser
        ax.scatter(DX[39, 0], DX[39, 1], DX[39, 2], marker='>', color='navajowhite', s=50) # last 5 t of left water

        ax.scatter(DX[15:19, 0], DX[15:19, 1], DX[15:19, 2], color='darkorange') # last 5 t of right laser
        ax.plot(DX[15:20, 0], DX[15:20, 1], DX[15:20, 2], color='darkorange') # last 5 t of right laser
        ax.scatter(DX[19, 0], DX[19, 1], DX[19, 2], marker='>', color='darkorange', s=50) # last 5 t of left water

        ax.scatter(DX[30:34, 0], DX[30:34, 1], DX[30:34, 2], color='violet') # first 5 t of left laser
        ax.plot(DX[30:35, 0], DX[30:35, 1], DX[30:35, 2], color='violet') # first 5 t of left laser
        ax.scatter(DX[34, 0], DX[34, 1], DX[34, 2], marker='>', color='violet', s=50) # last 5 t of left water

        ax.text2D(0.05, 0.95,reg, transform=ax.transAxes)
        plt.show()

        del X_reg
                    """
                    X = np.empty((binned_fr_flat.shape[0],40),  dtype="S10")#Start matrix that will hold data
                    X[:]='nan'
                    for j,i in enumerate(np.arange(-5,5)):
                        if i<0:
                            X[:,j]="%dLW" % i # last 5 t of left water
                            X[:,10+j]="%dRW" % i
                            X[:,20+j]="%dLL" % i
                            X[:,30+j]="%dLW" % i
                        if i>-1:
                            X[:,5+i]="%dFLW" % i # first 5 t of left water
                            X[:,15+i]="%dFRW" % i
                            X[:,25+i]="%dFLL" % i
                            X[:,35+i]="%dFLW" % i
                    '''

    #### Plots for annual meeting 2021
    # t25=alf('/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/004')
    # fr = t25.fr_bytrial(234, t25.goCue_trigger_times, probe_name='probe00')
    # Logistic regression

    '''

    def plot_error_dist(data):
        data =  data.loc[~np.isnan(data['choice_prediction'])]
        data['correct_prediction'] = 1*((1*data['choice_prediction']>0.5) == \
                                            1*(data['choice_1']))
        fig, ax = plt.subplots(2,2)
        plt.sca(ax[0,0])
        sns.lineplot(data=data, x='index', y=data['correct_prediction'].rolling(5).mean(),
                color='k', ci=68)
        plt.title('Accuracy by trial no')
        plt.xlabel('Trial number')
        plt.ylabel('Accuracy')
        plt.sca(ax[0,1])
        sns.barplot(data=data, x='probabilityLeft', y='correct_prediction', color='k', ci=68)
        plt.title('Accuracy by side block')
        plt.xlabel('Probability Left')
        plt.ylabel('Accuracy')
        plt.sca(ax[1,0])
        negative_trials = data.loc[data['trial_within_block']<0].copy()
        positive_trials = data.loc[data['trial_within_block_real']>=0].copy()
        positive_trials['trial_within_block']=positive_trials['trial_within_block_real']
        positive_trials['transition_type']=positive_trials['transition_type_real']
        ses_df = pd.concat([negative_trials,positive_trials])
        sns.lineplot(data = ses_df.loc[(ses_df['transition_type']=='0.1 to 0.7') |
            (ses_df['transition_type']=='0.7 to 0.1')], x='trial_within_block', y='correct_prediction',
            ci=68, hue='transition_type', err_style='bars', style='opto_block', palette=['dodgerblue','orange'])
        plt.ylim(0,1)
        plt.xlim(-5,15)
        plt.vlines(0,0,1,linestyles='dashed', color='k')
        plt.title('Accuracy by transition type')
        plt.ylabel('Accuracy')
        plt.xlabel('Trials from block switch')
        plt.sca(ax[1,1])
        sns.barplot(data=data, x='opto_block', y='correct_prediction', palette=['dodgerblue', 'orange'], ci=68)
        plt.title('Accuracy by reward indentity block')
        plt.xlabel('Opto block')
        plt.ylabel('Accuracy')
        sns.despine()
        plt.tight_layout()
        # Rolling average of the prediction derivative
        mice = len(data.mouse.unique())
        ses_max = 6
        fig, ax = plt.subplots(mice, ses_max, figsize=(20,15))
        for m , mouse in enumerate(data.mouse.unique()):
            data_mouse = data.loc[data['mouse']==mouse]
            for s , ses in enumerate(data_mouse.date.unique()):
                plt.sca(ax[m,s])
                data_ses = data_mouse.loc[data_mouse['date']==ses]
                sns.lineplot(data=data_ses, x='index', y=data['correct_prediction'].rolling(5).mean(),
                            color='gray', ci=68)
                plt.xlim(0,400)
                plt.vlines(np.where(data_ses['opto_block']==1), 0, 1, linestyles='solid', alpha=0.25, color='orange')
                plt.vlines(np.where(data_ses['opto_block']==0), 0, 1, linestyles='solid', alpha=0.25, color='dodgerblue')
                plt.vlines(np.where(data_ses['block_change']==1), 0, 1, linestyles='dashed', color='k')
                plt.xlabel('trial no')
                plt.ylabel('Accuracy')
        sns.despine()
        plt.tight_layout()

        '''

