#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:26:44 2020

@author: alex
"""
import os
import numpy as np
import pandas as pd
import rew_alf.npy2pd as npy2pd
import brainbox as bb
from brainbox.plot import *
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os
from os import path
from rew_alf.alf_functions import *
import sys
from path import Path
import random
import seaborn as sns
import scipy.stats as stats
from matplotlib.ticker import FormatStrFormatter
import matplotlib
from matplotlib.lines import Line2D
from scipy.stats import mannwhitneyu
from scipy.signal import convolve, gaussian

### Developing test ###



### collect_paths


# Analysis for ChR2 Sessions need to be in the same order 

session_paths = ['/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-14/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-15/002',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-17/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-18/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_8/2020-03-19/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_9/2020-03-17/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_9/2020-03-18/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/chr2/dop_9/2020-03-19/001']


# Analysis for NphR Sessions need to be in the same order 

session_paths = ['/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_4/2020-01-16/002',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_4/2020-01-17/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_11/2020-03-14/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_11/2020-03-15/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_11/2020-03-16/001',
                 '/Volumes/witten/Alex/recordings_march_2020_dop/nphr/dop_11/2020-03-19/001']




for j,i in enumerate(session_paths):
    
    if j == 0:
        session_folder = i
        session = load_behavior(session_folder)
        clusters_depths, cluster_metrics, \
        spikeclusters, spiketimes, \
        good_clusters, allen, channel_coord = load_neural_data(session_folder)
        gen_clusters_depths = clusters_depths[np.unique(spikeclusters)] #Limit allen variable to cluster that actually fire
        gen_cluster_metrics = cluster_metrics['ks2_label'][np.unique(spikeclusters)] #Limit allen variable to cluster that actually fire
        gen_spikeclusters = spikeclusters
        gen_spiketimes = spiketimes
        gen_good_clusters = good_clusters
        #Limit allen variable to cluster that actually fire
        allen  = allen[np.unique(spikeclusters)]
        gen_allen = allen
        gen_channel_coord = channel_coord 
        assert len(np.unique(spikeclusters)) == len(allen)
        
        
    else:
        session_folder = i
        session = pd.concat([session,load_behavior(session_folder)])
        clusters_depths, cluster_metrics, \
        spikeclusters, spiketimes, \
        good_clusters, allen, channel_coord = load_neural_data(session_folder)
        
        gen_clusters_depths = np.append(gen_clusters_depths, clusters_depths[np.unique(spikeclusters)]) #Limit allen variable to cluster that actually fire)
        gen_cluster_metrics = np.append(gen_cluster_metrics, cluster_metrics['ks2_label'][np.unique(spikeclusters)]) #Limit allen variable to cluster that actually fire
        gen_spikeclusters = np.append(gen_spikeclusters, (spikeclusters + (j * 10000)))
        gen_spiketimes = np.append(gen_spiketimes, spiketimes)
        gen_good_clusters = np.append(gen_good_clusters, (good_clusters + (j * 10000)))
        allen  = allen[np.unique(spikeclusters)] #Limit allen variable to cluster that actually fire
        gen_allen = np.append(gen_allen, allen)
        gen_channel_coord = np.append(gen_channel_coord, channel_coord, axis = 0)
        assert len(np.unique(spikeclusters)) == len(allen)

    
# Pool regions
keys = np.unique(gen_allen)
pooled_region = np.zeros(len(keys))
pooled_region[:] = str('nan')

pooler = pd.DataFrame(np.transpose([keys, pooled_region]))    
pooler.iloc[:,1] = 'void'
print('Manually cluster areas')
    
# rename regions
pooled_allen = []
for i in range(len(gen_allen)):
    pooled_allen.append(pooler.loc[pooler[0] == gen_allen[i],1].tolist())

pooled_allen = np.array(pooled_allen)

session  = session.reset_index()
session = session.rename(columns={'index':'idx_per_session'})
session = session.reset_index()
session = session.rename(columns={'index':'idx'})

for j, region in enumerate(np.unique(pooled_allen)):
    region_clusters_quality = gen_cluster_metrics[np.where(pooled_allen  == region)[0]] 
    region_clusters = np.unique(gen_spikeclusters)[np.where(pooled_allen  == region)[0]]
    region_goodlusters = region_clusters[region_clusters_quality=='good']
    print(str(len(region_goodlusters)) + ' ' + region)
    
    sort_non_opto_plot_by_block_and_subtraction_per_region(gen_spikeclusters, gen_spiketimes,session, 
                                       'goCue_times', region_goodlusters, gen_clusters_depths,
                                       bin_size=0.025, extra_ref = str(region), smoothing_win = 0.05)
    sort_non_opto_plot_by_block_and_subtraction_per_region(gen_spikeclusters, gen_spiketimes,session,  
                                       'feedback_times', region_goodlusters,gen_clusters_depths,
                                       bin_size=0.025, extra_ref = str(region),
                                       thirdclass = 'feedbackType', fourthclass = -1, smoothing_win = 0.05)
    sort_non_opto_plot_by_block_and_subtraction_per_region(gen_spikeclusters, gen_spiketimes,session,  
                                       'feedback_times', region_goodlusters,gen_clusters_depths,
                                       bin_size=0.025, extra_ref = str(region),
                                       thirdclass = 'feedbackType', fourthclass = 1, smoothing_win = 0.05)
    
    ### Section in trial ###
    sort_non_opto_plot_by_block_and_subtraction_per_region_classified_by_contrast(gen_spikeclusters, gen_spiketimes,session, 
                                       'goCue_times', region_goodlusters, gen_clusters_depths,
                                       bin_size=0.025, extra_ref = str(region), smoothing_win = 0.05)
    ########################
    
    comparisons_str = ['neutral_side_selective_total',
                       'left_block_side_selective', 
                       'right_block_side_selective',
                       'l_trials_block_selective', 
                       'r_trials_block_selective']
    
    s_fr_neurons = significant_firing_rate_neurons(gen_spiketimes, gen_spikeclusters, 
                                                   'feedback_times', region_goodlusters,
                                     session, comparisons = len(comparisons_str), bin_size = 0.025)
    
    
    if j == 0: # Start dataframe in first iteration
        FR_regions = pd.DataFrame(columns=['Region', 'Count', 'Type',
                                           'Region_Total'])
        FR_regions['Region'] = np.sort(list(np.unique(pooled_allen))*len(comparisons_str)) 
    
    FR_regions.loc[(FR_regions['Region']==region), 'Type'] = comparisons_str
    FR_regions.loc[(FR_regions['Region']==region) & 
                       (FR_regions['Type']==comparisons_str[0]), 'Count'] = s_fr_neurons[0]
    FR_regions.loc[(FR_regions['Region']==region) &
                       (FR_regions['Type']==comparisons_str[1]), 'Count'] = s_fr_neurons[1]
    FR_regions.loc[(FR_regions['Region']==region) &
                       (FR_regions['Type']==comparisons_str[2]), 'Count'] = s_fr_neurons[2]
    FR_regions.loc[(FR_regions['Region']==region) &
                       (FR_regions['Type']==comparisons_str[3]), 'Count'] = s_fr_neurons[3]
    FR_regions.loc[(FR_regions['Region']==region) &
                       (FR_regions['Type']==comparisons_str[4]), 'Count'] = s_fr_neurons[4]
        
    FR_regions.loc[FR_regions['Region']==region, 'Region_Total'] = len(region_goodlusters)

    
FR_regions['Percentage'] = FR_regions['Count']/FR_regions['Region_Total']

FR_regions_plots  = FR_regions.loc[FR_regions['Region'] != 'Other']

g = sns.FacetGrid(FR_regions_plots, col='Type', height =5, aspect = 1.2)
g.map(sns.barplot, 'Percentage', 'Region', color=".3", ci=None)

# Analysis of firing rate across regions
def significant_firing_rate_neurons (gen_spiketimes, gen_spikeclusters, epoch, region_goodlusters,
                                     session, comparisons = 4, bin_size = 0.025):
    '''
    Returns number of neurons in a region with a signficant
    difference in firing rates 500ms after epoch
    '''
    
    
    binned_firing_rate = bb.singlecell.calculate_peths(
                gen_spiketimes, gen_spikeclusters, region_goodlusters, session[epoch],
                bin_size=bin_size)[1]
    
    
    cluster_session = np.floor(region_goodlusters/10000)
    
    
    # Divide by choice and opto block
    
    left_stim_trials_neutral_block = np.intersect1d(np.where(session['choice'] > 0), \
                                    np.where(session['opto_probability_left'] == -1),
                                    np.where(session['feedbackType'] == 1))
    right_stim_trials_neutral_block = np.intersect1d(np.where(session['choice'] <= 0), \
                                    np.where(session['opto_probability_left'] == -1),
                                    np.where(session['feedbackType'] == 1))
    left_stim_trials_left_block = np.intersect1d(np.where(session['choice'] > 0), \
                                    np.where(session['opto_probability_left'] == 1),
                                    np.where(session['feedbackType'] == 1))
    right_stim_trials_left_block = np.intersect1d(np.where(session['choice'] <= 0), \
                                    np.where(session['opto_probability_left'] == 1),
                                    np.where(session['feedbackType'] == 1))
    left_stim_trials_right_block = np.intersect1d(np.where(session['choice'] > 0), \
                                    np.where(session['opto_probability_left'] == 0),
                                    np.where(session['feedbackType'] == 1))
    right_stim_trials_right_block = np.intersect1d(np.where(session['choice'] <= 0), \
                                    np.where(session['opto_probability_left'] == 0),
                                    np.where(session['feedbackType'] == 1))
    
    L = binned_firing_rate[left_stim_trials_neutral_block,:,:]
    R = binned_firing_rate[right_stim_trials_neutral_block,:,:]
    L_blockL = binned_firing_rate[left_stim_trials_left_block,:,:]
    R_blockL = binned_firing_rate[right_stim_trials_left_block,:,:]
    L_blockR = binned_firing_rate[left_stim_trials_right_block,:,:]
    R_blockR = binned_firing_rate[right_stim_trials_right_block,:,:]
    
    
    
    neutral_side_selective = average_fr_2_con_multi_session(L, R, 
                                       left_stim_trials_neutral_block, 
                                       right_stim_trials_neutral_block, session, 
                                       cluster_session,
                                       comparisons = comparisons)
     
    left_block_side_selective = average_fr_2_con_multi_session(L_blockL, R_blockL, 
                                       left_stim_trials_left_block, 
                                       right_stim_trials_left_block, session,
                                       cluster_session,
                                       comparisons = comparisons)
     
    right_block_side_selective = average_fr_2_con_multi_session(L_blockR, R_blockR, 
                                       left_stim_trials_right_block, 
                                       right_stim_trials_right_block, session,
                                       cluster_session,
                                       comparisons = comparisons)
    
    l_trials_block_selective = average_fr_2_con_multi_session(L_blockL, L_blockR, 
                                       left_stim_trials_left_block, 
                                       left_stim_trials_right_block, session,
                                       cluster_session,
                                       comparisons = comparisons)
    
    r_trials_block_selective = average_fr_2_con_multi_session(R_blockL, R_blockR, 
                                       right_stim_trials_left_block, 
                                       right_stim_trials_right_block, session,
                                       cluster_session,
                                       comparisons = comparisons)
    
    
    return neutral_side_selective, left_block_side_selective, right_block_side_selective, \
        l_trials_block_selective, r_trials_block_selective

    
def average_fr_2_con_multi_session(binned_firing_rate1, binned_firing_rate2, 
                                       index_1, index_2, session, cluster_session, comparisons=4):
        '''
        "Returns total number of units with a significantly different firing rate between conditions
        in range of 500 ms after epoch""
        binned_firing_rate1: binned firing rate for condition 1
        binned_firing_rate2: binned firing rate for condition 2
        index_1: trials for condition 1
        index_2: trials for condition 2
        session: pooled behavior data
        comparisons: Number of comparison, standard is 4 (2 blocks, 2 choices)
        '''
        # mean_binned_fr_test1 = None
        # mean_binned_fr_test2 = None
        significant = 0
        counter_session = 0
        for mouse in session['mouse_name'].unique():
            for ses in session.loc[session['mouse_name'] == mouse, 'ses'].unique():
                # First number from cluster determines session
                
                cluster_in_session = np.where(cluster_session == counter_session)
                if np.size(cluster_in_session) == 0:
                    counter_session += 1
                    continue
                
                counter_session += 1
                
                m_ses = session.loc[(session['mouse_name'] == mouse) & 
                                    (session['ses'] == ses)]
                trials_in_session = m_ses['idx'].to_numpy()
                trials_in_sess_in_var1 = np.intersect1d(trials_in_session, index_1)
                tvar1 = (index_1[:, None] == trials_in_sess_in_var1).argmax(axis=0)
    
                trials_in_sess_in_var2 = np.intersect1d(trials_in_session, index_2)
                tvar2 = (index_2[:, None] == trials_in_sess_in_var2).argmax(axis=0)
                
                
                binned_firing_rate1_select = binned_firing_rate1[:, cluster_in_session[0],:]
                binned_firing_rate2_select = binned_firing_rate2[:, cluster_in_session[0],:]
                
                for i in range(len(cluster_in_session[0])):
                    
                    c1 = np.mean(binned_firing_rate1_select[tvar1, i, 7:], axis = 1)/bin_size
                    c2 = np.mean(binned_firing_rate2_select[tvar2, i, 7:],axis = 1)/bin_size
                    if(np.sum(c1) == 0)  & (np.sum(c2) == 0):
                        continue
                    else:
                        _, p = mannwhitneyu(c1, 
                                            c2)
                    if p<(0.05/comparisons): #Significance with Bonferroni
                        significant += 1
                #mean_N_binned_fr_test1 =  np.mean(binned_firing_rate1_select[tvar1, :,:], axis =0)/bin_size
                #mean_N_binned_fr_test2 =  np.mean(binned_firing_rate2_select[tvar2, :,:], axis =0)/bin_size
                
                #if mean_binned_fr_test1 is None:
                #    mean_binned_fr_test1 = mean_N_binned_fr_test1
                
                #else:
                #    mean_binned_fr_test1 = np.concatenate((mean_binned_fr_test1,
                #                                          mean_N_binned_fr_test1) , axis = 0)
                #if mean_binned_fr_test2 is None:
                #    mean_binned_fr_test2 = mean_N_binned_fr_test2
                
                #else:
                #    mean_binned_fr_test2 = np.concatenate((mean_binned_fr_test2,
                #                                          mean_N_binned_fr_test2) , axis = 0)
                    
                #z_score_binned_spikes_test1 = stats.zscore(mean_binned_fr_test1, axis =1)
                #z_score_binned_spikes_test1 = np.nan_to_num(z_score_binned_spikes_test1)
                #z_score_binned_spikes_test2 = np.nan_to_num(z_score_binned_spikes_test2)    
                #z_score_binned_spikes_test2 = stats.zscore(mean_binned_fr_test2, axis =1)
                
        return significant# z_score_binned_spikes_test1, z_score_binned_spikes_test2

    



# Analysis for individial sessions

for i in session_paths:
    
    session_folder = i
    session = load_behavior(session_folder)
    clusters_depths, cluster_metrics, \
    spikeclusters, spiketimes, \
    good_clusters, allen, channel_coord = load_neural_data(session_folder)
    
    os.chdir(i)
    if not os.path.exists('figures'):
        os.makedirs('figures')
    os.chdir(i + '/figures')
    
    # Heatmaps of session
    sort_non_opto_plot_by_block_and_subtraction(spikeclusters, spiketimes,session, 
                               'goCue_times', good_clusters,gen_clusters_depths,
                               bin_size=0.025)
    sort_non_opto_plot_by_block_and_subtraction(spikeclusters, spiketimes,session,  
                               'feedback_times', good_clusters,gen_clusters_depths,
                               bin_size=0.025)
    
    # heatmaps per region
    if 'allen' in globals(): 
        for region in np.unique(allen):
            region_goodlusters = np.intersect1d(np.where(allen  == region), good_clusters)
            sort_non_opto_plot_by_block_and_subtraction(spikeclusters, spiketimes,session,  
                        'goCue_times', region_goodlusters,gen_clusters_depths,
                                       bin_size=0.025, extra_ref = str(region))
            sort_non_opto_plot_by_block_and_subtraction(spikeclusters, spiketimes,session,  
                                    'feedback_times', region_goodlusters,gen_clusters_depths,
                                       bin_size=0.025, extra_ref = str(region))
    
    # PSTH and raster per session

    session_single_neuron_psth_summary(spikeclusters, spiketimes,session, session_folder)


##############################################################################
################################ Function ####################################
##############################################################################



# Loading data


def load_data (root_folder):
    """Generates a dataframe with all available information in project folder
    INPUT: root folder include several subjects and viruses
    OUTPUT:  macro (dataframe per animal per session)"""
    #root_folder =  '/mnt/s0/Data/Subjects_personal_project/rewblocks8040/'
    viruses = sorted([x for x in (os.listdir (root_folder)) if ".DS_Store" not in x])
    for virus in viruses:
        mice = sorted([x for x in (os.listdir (root_folder + virus +'/')) if ".DS_Store" not in x])
        for mouse in mice:
            dates =  sorted([x for x in (os.listdir (root_folder + virus + '/' + mouse)) if ".DS_Store" not in x])
            df = pd.DataFrame(index=dates, columns = col)
            for day in dates:
                #merge sessions from the same day
                path = root_folder + virus + '/' + mouse + '/' + day +'/'

def load_behavior(session_folder):
    '''
    input: session folder  (str)
    output: dataframe with behavior from one ephys sessions
    '''
    variables = npy2pd.pybpod_vars()
    session = npy2pd.session_loader(session_folder, variables)
    folder_path =  Path(session_folder)
    session['mouse_name'] =  str(folder_path.parent.parent.basename())
    session['ses'] = str(folder_path.parent.basename())
    del session['goCueTrigger_times']
    del session['intervals']
    ##Fix opto, opto variable is not the truth
    del session['opto_dummy']
    del session['opto.npy']
    ##
    session['response_times'] = session['response_times']
    session = pd.DataFrame.from_dict(session)
    session = add_trial_within_block(session)
    session['real_opto'] = np.load(session_folder + '/alf/_ibl_trials.laser_epoch_in.npy')
    session['opto.npy'] = (~np.isnan(session['real_opto']) == 1)*1
    session['after_opto'] = session['opto.npy'].shift(periods=1)
    session['after_reward'] = session['feedbackType'].shift(periods=1)
    return session

def load_neural_data(session_folder, histology = True):
    '''
    Parameters
    ----------
    session_folder : folder of ephy session (str)

    Returns
    -------
    arrays with cluster measures
    '''
    
    if Path(session_folder + '/alf/probe01/').exists() == True:
        clusters_depths = np.load(session_folder + '/alf/probe00/clusters.depths.npy')
        cluster_metrics = pd.read_csv(session_folder + '/alf/probe00/clusters.metrics.csv')
        spikeclusters = np.load(session_folder + '/alf/probe00/spikes.clusters.npy')
        spiketimes = np.load(session_folder + '/alf/probe00/spikes.times.npy')
        cluster_channels = np.load(session_folder + '/alf/probe00/clusters.channels.npy')
        clusters_depths = np.append(clusters_depths, np.load(session_folder + '/alf/probe01/clusters.depths.npy'))
        cluster_metrics = pd.concat([cluster_metrics, pd.read_csv(session_folder + '/alf/probe01/clusters.metrics.csv')])
        spikeclusters = np.append(spikeclusters, np.load(session_folder + '/alf/probe01/spikes.clusters.npy'))
        spiketimes = np.append(spiketimes, np.load(session_folder + '/alf/probe01/spikes.times.npy'))
        good_clusters = cluster_metrics.loc[cluster_metrics['ks2_label']=='good', 'cluster_id']
        cluster_channels = np.append(cluster_channels, np.load(session_folder + '/alf/probe01/clusters.channels.npy'))
        if histology == True:  
            allen = np.load(session_folder + '/alf/probe00/cluster_location.npy', allow_pickle=True) 
            allen = np.append(allen, np.load(session_folder + '/alf/probe01/cluster_location.npy', allow_pickle=True))
            allen  = allen[cluster_channels]
            channel_coord = np.load(session_folder + '/alf/probe00/channels_xyz.npy', allow_pickle=True)
            channel_coord = np.concatenate([channel_coord, np.load(session_folder + '/alf/probe01/channels_xyz.npy', allow_pickle=True)])
            return clusters_depths, cluster_metrics, spikeclusters, spiketimes, good_clusters, allen, channel_coord
        if histology == False:
            return clusters_depths, cluster_metrics, spikeclusters, spiketimes, good_clusters
        
    else:
        clusters_depths = np.load(session_folder + '/alf/probe00/clusters.depths.npy')
        cluster_metrics = pd.read_csv(session_folder + '/alf/probe00/clusters.metrics.csv')
        spikeclusters = np.load(session_folder + '/alf/probe00/spikes.clusters.npy')
        spiketimes = np.load(session_folder + '/alf/probe00/spikes.times.npy')
        good_clusters = cluster_metrics.loc[cluster_metrics['ks2_label']=='good', 'cluster_id']
        cluster_channels = np.load(session_folder + '/alf/probe00/clusters.channels.npy')
        if histology == True:  
            allen = np.load(session_folder + '/alf/probe00/cluster_location.npy', allow_pickle=True)
            allen  = allen[cluster_channels]
            channel_coord = np.load(session_folder + '/alf/probe00/channels_xyz.npy', allow_pickle=True)
            return clusters_depths, cluster_metrics, spikeclusters, spiketimes, good_clusters, allen, channel_coord
        if histology == False:
            return clusters_depths, cluster_metrics, spikeclusters, spiketimes, good_clusters



# Plot heatmaps

def heatmap_sorted_by_pool_stimulus_mistakes_only(spikeclusters, spiketimes,session, i, epoch = 'goCue_times',
                           bin_size=0.025):

    #Sorted with all data, plotted with blocks
    # ax[0,0] is not crossvalidated    
    # Calculate bins
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, good_clusters, session[epoch],
            bin_size=bin_size)[1]
    
    
    # Divide by stimulus side and opto block
    left_stim_trials_left_block = np.intersect1d(np.where(session['contrastLeft'] > 0), \
                                    np.where(session['opto_probability_left'] == 1))
    right_stim_trials_left_block = np.intersect1d(np.where(session['contrastRight'] > 0), \
                                    np.where(session['opto_probability_left'] == 1))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] > 0), \
                                    np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 0))
    
        
    # Intersect with mistakes

    left_stim_trials_left_block = np.intersect1d(np.where(session['feedbackType'] == -1), 
                                                   left_stim_trials_left_block)
    right_stim_trials_left_block = np.intersect1d(np.where(session['feedbackType'] == -1), 
                                                   right_stim_trials_left_block)
    left_stim_trials_right_block = np.intersect1d(np.where(session['feedbackType'] == -1), 
                                                   left_stim_trials_right_block)
    right_stim_trials_right_block = np.intersect1d(np.where(session['feedbackType'] == -1), 
                                                   right_stim_trials_right_block)
    # Divide by slide block
        
    sorte, order = heatmap_w_external_sorting(binned_firing_rate[np.where(session['feedbackType'] == -1)], 
                                              order = None)
    sorte_cv, order_cv = heatmap_cross_validated(binned_firing_rate[np.where(session['feedbackType'] == -1)])
    
    sorte_left_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_stim_trials_left_block,:,:],order)
    sorte_left_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_stim_trials_right_block,:,:],order)
    sorte_right_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_stim_trials_left_block,:,:],order)
    sorte_right_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_stim_trials_right_block,:,:],order)
    
    fig, ax  = plt.subplots(3,2, figsize=(10,17))
    plt.sca(ax[0,0])
    sns.heatmap(sorte)
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('Neuron Location (depth um)')
    ax[0,0].set_yticklabels(clusters_depths[order])
    ax[0,0].set_title("All incorrect")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte_cv)
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,1].set_ylabel('Neuron Location')
    ax[0,1].set_yticklabels(clusters_depths[order_cv])
    ax[0,1].set_title("All incorrect cross-validated")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('Neuron Location (depth um)')
    ax[1,0].set_yticklabels(clusters_depths[order])
    ax[1,0].set_title("Stimulus Left Side, Left Opto Block Incorrect")
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_right_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,1].set_ylabel('Neuron Location')
    ax[1,1].set_yticklabels(clusters_depths[order])
    ax[1,1].set_title("Stimulus Right Side, Left Opto Block Incorrect")
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_left_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,0].set_ylabel('Neuron Location')
    ax[2,0].set_yticklabels(clusters_depths[order])
    ax[2,0].set_title("Stimulus Left Side, Right Opto Block Incorrect")
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,1].set_ylabel('Neuron Location')
    ax[2,1].set_yticklabels(clusters_depths[order])
    ax[2,1].set_title("Stimulus Right Side, Right Opto Block Incorrect")
    plt.tight_layout()
    plt.savefig('Stimulus_heatmap_common_order_incorrect.png')
    plt.savefig('Stimulus_heatmap_common_order_incorrect.svg')

def heatmap_sorted_by_pool_stimulus_correct_only(spikeclusters, spiketimes,session, i, epoch = 'goCue_times',
                           bin_size=0.025):

    #Sorted with all data, plotted with blocks
    # ax[0,0] is not crossvalidated    
    # Calculate bins
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, good_clusters, session[epoch],
            bin_size=bin_size)[1]
    
    
    # Divide by stimulus side and opto block
    left_stim_trials_left_block = np.intersect1d(np.where(session['contrastLeft'] > 0), \
                                    np.where(session['opto_probability_left'] == 1))
    right_stim_trials_left_block = np.intersect1d(np.where(session['contrastRight'] > 0), \
                                    np.where(session['opto_probability_left'] == 1))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] > 0), \
                                    np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 0))
    
        
    # Intersect with mistakes

    left_stim_trials_left_block = np.intersect1d(np.where(session['feedbackType'] == 1), 
                                                   left_stim_trials_left_block)
    right_stim_trials_left_block = np.intersect1d(np.where(session['feedbackType'] == 1), 
                                                   right_stim_trials_left_block)
    left_stim_trials_right_block = np.intersect1d(np.where(session['feedbackType'] == 1), 
                                                   left_stim_trials_right_block)
    right_stim_trials_right_block = np.intersect1d(np.where(session['feedbackType'] == 1), 
                                                   right_stim_trials_right_block)
    # Divide by slide block
        
    sorte, order = heatmap_w_external_sorting(binned_firing_rate[np.where(session['feedbackType'] == 1)], 
                                              order = None)
    sorte_cv, order_cv = heatmap_cross_validated(binned_firing_rate[np.where(session['feedbackType'] == 1)])
    
    sorte_left_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_stim_trials_left_block,:,:],order)
    sorte_left_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_stim_trials_right_block,:,:],order)
    sorte_right_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_stim_trials_left_block,:,:],order)
    sorte_right_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_stim_trials_right_block,:,:],order)
    
    fig, ax  = plt.subplots(3,2, figsize=(10,17))
    plt.sca(ax[0,0])
    sns.heatmap(sorte)
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('Neuron Location (depth um)')
    ax[0,0].set_yticklabels(clusters_depths[order])
    ax[0,0].set_title("All correct")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte_cv)
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,1].set_ylabel('Neuron Location')
    ax[0,1].set_yticklabels(clusters_depths[order_cv])
    ax[0,1].set_title("All correct cross-validated")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('Neuron Location (depth um)')
    ax[1,0].set_yticklabels(clusters_depths[order])
    ax[1,0].set_title("Stimulus Left Side, Left Opto Block correct")
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_right_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,1].set_ylabel('Neuron Location')
    ax[1,1].set_yticklabels(clusters_depths[order])
    ax[1,1].set_title("Stimulus Right Side, Left Opto Block correct")
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_left_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,0].set_ylabel('Neuron Location')
    ax[2,0].set_yticklabels(clusters_depths[order])
    ax[2,0].set_title("Stimulus Left Side, Right Opto Block correct")
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,1].set_ylabel('Neuron Location')
    ax[2,1].set_yticklabels(clusters_depths[order])
    ax[2,1].set_title("Stimulus Right Side, Right Opto Block correct")
    plt.tight_layout()
    plt.savefig('Stimulus_heatmap_common_order_correct.png')
    plt.savefig('Stimulus_heatmap_common_order_correct.svg')


def heatmap_sorted_by_pool_choice_correct_only(spikeclusters, spiketimes,session, i, epoch = 'response_times',
                           bin_size=0.025):

    #Sorted with all data, plotted with blocks
    # ax[0,0] is not crossvalidated    
    # Calculate bins
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, good_clusters, session[epoch],
            bin_size=bin_size)[1]
    
    
    # Divide by stimulus side and opto block
    left_choice_trials_left_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 1))
    right_choice_trials_left_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 1))
    left_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 0))
    right_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 0))
    
        
    # Intersect with mistakes

    left_choice_trials_left_block = np.intersect1d(np.where(session['feedbackType'] == 1), 
                                                   left_choice_trials_left_block)
    right_choice_trials_left_block = np.intersect1d(np.where(session['feedbackType'] == 1), 
                                                   right_choice_trials_left_block)
    left_choice_trials_right_block = np.intersect1d(np.where(session['feedbackType'] == 1), 
                                                   left_choice_trials_right_block)
    right_choice_trials_right_block = np.intersect1d(np.where(session['feedbackType'] == 1), 
                                                   right_choice_trials_right_block)
    # Divide by slide block
        
    sorte, order = heatmap_w_external_sorting(binned_firing_rate[np.where(session['feedbackType'] == 1)], order = None)
    sorte_cv, order_cv = heatmap_cross_validated(binned_firing_rate[np.where(session['feedbackType'] == 1)])
    
    sorte_left_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_choice_trials_left_block,:,:],order)
    sorte_left_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_choice_trials_right_block,:,:],order)
    sorte_right_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_choice_trials_left_block,:,:],order)
    sorte_right_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_choice_trials_right_block,:,:],order)
    
    fig, ax  = plt.subplots(3,2, figsize=(10,17))
    plt.sca(ax[0,0])
    sns.heatmap(sorte)
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('Neuron Location (depth um)')
    ax[0,0].set_yticklabels(clusters_depths[order])
    ax[0,0].set_title("All correct")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte_cv)
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,1].set_ylabel('Neuron Location')
    ax[0,1].set_yticklabels(clusters_depths[order_cv])
    ax[0,1].set_title("All correct cross-validated")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('Neuron Location (depth um)')
    ax[1,0].set_yticklabels(clusters_depths[order])
    ax[1,0].set_title("Choice Left Side, Left Opto Block correct")
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_right_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,1].set_ylabel('Neuron Location')
    ax[1,1].set_yticklabels(clusters_depths[order])
    ax[1,1].set_title("Choice Right Side, Left Opto Block correct")
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_left_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,0].set_ylabel('Neuron Location')
    ax[2,0].set_yticklabels(clusters_depths[order])
    ax[2,0].set_title("Choice Left Side, Right Opto Block correct")
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,1].set_ylabel('Neuron Location')
    ax[2,1].set_yticklabels(clusters_depths[order])
    ax[2,1].set_title("Choice Right Side, Right Opto Block correct")
    plt.tight_layout()
    plt.savefig('choice_heatmap_common_order_correct.png')
    plt.savefig('choice_heatmap_common_order_correct.svg')


def heatmap_sorted_by_pool_choice_mistakes_only(spikeclusters, spiketimes,session, i, epoch = 'response_times',
                           bin_size=0.025):

    #Sorted with all data, plotted with blocks
    # ax[0,0] is not crossvalidated    
    # Calculate bins
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, good_clusters, session[epoch],
            bin_size=bin_size)[1]
    
    
    # Divide by stimulus side and opto block
    left_choice_trials_left_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 1))
    right_choice_trials_left_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 1))
    left_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 0))
    right_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 0))
    
        
    # Intersect with mistakes

    left_choice_trials_left_block = np.intersect1d(np.where(session['feedbackType'] == -1), 
                                                   left_choice_trials_left_block)
    right_choice_trials_left_block = np.intersect1d(np.where(session['feedbackType'] == -1), 
                                                   right_choice_trials_left_block)
    left_choice_trials_right_block = np.intersect1d(np.where(session['feedbackType'] == -1), 
                                                   left_choice_trials_right_block)
    right_choice_trials_right_block = np.intersect1d(np.where(session['feedbackType'] == -1), 
                                                   right_choice_trials_right_block)
    # Divide by slide block
        
    sorte, order = heatmap_w_external_sorting(binned_firing_rate[np.where(session['choice'] == -1)], order = None)
    sorte_cv, order_cv = heatmap_cross_validated(binned_firing_rate[np.where(session['choice'] == -1)])
    
    sorte_left_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_choice_trials_left_block,:,:],order)
    sorte_left_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_choice_trials_right_block,:,:],order)
    sorte_right_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_choice_trials_left_block,:,:],order)
    sorte_right_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_choice_trials_right_block,:,:],order)
    
    fig, ax  = plt.subplots(3,2, figsize=(10,17))
    plt.sca(ax[0,0])
    sns.heatmap(sorte)
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('Neuron Location (depth um)')
    ax[0,0].set_yticklabels(clusters_depths[order])
    ax[0,0].set_title("All")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte_cv)
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,1].set_ylabel('Neuron Location')
    ax[0,1].set_yticklabels(clusters_depths[order_cv])
    ax[0,1].set_title("All incorrect cross-validated")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('Neuron Location (depth um)')
    ax[1,0].set_yticklabels(clusters_depths[order])
    ax[1,0].set_title("Choice Left Side, Left Opto Block Incorrect")
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_right_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,1].set_ylabel('Neuron Location')
    ax[1,1].set_yticklabels(clusters_depths[order])
    ax[1,1].set_title("Choice Right Side, Left Opto Block Incorrect")
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_left_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,0].set_ylabel('Neuron Location')
    ax[2,0].set_yticklabels(clusters_depths[order])
    ax[2,0].set_title("Choice Left Side, Right Opto Block Incorrect")
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,1].set_ylabel('Neuron Location')
    ax[2,1].set_yticklabels(clusters_depths[order])
    ax[2,1].set_title("Choice Right Side, Right Opto Block Incorrect")
    plt.tight_layout()
    plt.savefig('choice_heatmap_common_order_incorrect.png')
    plt.savefig('choice_heatmap_common_order_incorrect.svg')



def heatmap_sorted_by_pool_choice(spikeclusters, spiketimes,session, i, epoch,
                           bin_size=0.025):

    #Sorted with all data, plotted with blocks
    # ax[0,0] is not crossvalidated    
    # Calculate bins
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, good_clusters, session[epoch],
            bin_size=bin_size)[1]
    
    
    # Divide by stimulus side and opto block
    left_choice_trials_left_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 1))
    right_choice_trials_left_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 1))
    left_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 0))
    right_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 0))
    
    # Divide by slide block
        
    sorte, order = heatmap_w_external_sorting(binned_firing_rate, order = None)
    sorte_cv, order_cv = heatmap_cross_validated(binned_firing_rate)
    
    sorte_left_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_choice_trials_left_block,:,:],order)
    sorte_left_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_choice_trials_right_block,:,:],order)
    sorte_right_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_choice_trials_left_block,:,:],order)
    sorte_right_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_choice_trials_right_block,:,:],order)
    
    fig, ax  = plt.subplots(3,2, figsize=(10,17))
    plt.sca(ax[0,0])
    sns.heatmap(sorte)
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('Neuron Location (depth um)')
    ax[0,0].set_yticklabels(clusters_depths[order])
    ax[0,0].set_title("All")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte_cv)
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,1].set_ylabel('Neuron Location')
    ax[0,1].set_yticklabels(clusters_depths[order_cv])
    ax[0,1].set_title("All cross-validated")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('Neuron Location (depth um)')
    ax[1,0].set_yticklabels(clusters_depths[order])
    ax[1,0].set_title("Choice Left Side, Left Opto Block")
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_right_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,1].set_ylabel('Neuron Location')
    ax[1,1].set_yticklabels(clusters_depths[order])
    ax[1,1].set_title("Choice Right Side, Left Opto Block")
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_left_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,0].set_ylabel('Neuron Location')
    ax[2,0].set_yticklabels(clusters_depths[order])
    ax[2,0].set_title("Choice Left Side, Right Opto Block")
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,1].set_ylabel('Neuron Location')
    ax[2,1].set_yticklabels(clusters_depths[order])
    ax[2,1].set_title("Choice Right Side, Right Opto Block")
    plt.tight_layout()
    plt.savefig('choice_heatmap_common_order.png')
    plt.savefig('choice_heatmap_common_order.svg')
    

    
def sort_non_opto_plot_by_block_and_subtraction(spikeclusters, spiketimes,session, epoch, cluster_select, cluster_depths,
                           bin_size=0.025, extra_ref = None):
    
    matplotlib.rcParams.update({'font.size': 22})
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, cluster_select, session[epoch],
            bin_size=bin_size)[1]
    
    # Divide by stimulus side and opto block
    left_stim_trials_neutral_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == -1))
    right_stim_trials_neutral_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == -1))
    left_stim_trials_left_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == 1))
    right_stim_trials_left_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == 1))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == 0))
    

    # Concatenate neutral L and R trials
    L = binned_firing_rate[left_stim_trials_neutral_block,:,:]
    R = binned_firing_rate[right_stim_trials_neutral_block,:,:]
    L_blockL = binned_firing_rate[left_stim_trials_left_block,:,:]
    R_blockL = binned_firing_rate[right_stim_trials_left_block,:,:]
    L_blockR = binned_firing_rate[left_stim_trials_right_block,:,:]
    R_blockR = binned_firing_rate[right_stim_trials_right_block,:,:]
        
        
    # Sort based by neutral block
    sorte, order = heatmap_append_L_R(L,R, order = None )
    
    # Apply sort
    
    sorte_left_block = \
        heatmap_append_L_R(L_blockL,R_blockL,order)
    sorte_right_block = \
        heatmap_append_L_R(L_blockR,R_blockR,order)
    
    # Plot
    
    fig, ax  = plt.subplots(4,3, figsize=(25,30), sharex=True)
    plt.sca(ax[0,0])
    sns.heatmap(sorte[:,:int(np.shape(sorte)[1]/2)], vmin=-3, vmax=7, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('NEUTRAL BLOCK \nLocation (depth um)', rotation='vertical',x=-0.1,y=0.5)
    ax[0,0].set_yticklabels(clusters_depths[order])
    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax[0,0].set_title("LEFT STIM TRIALS")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte[:,int(np.shape(sorte)[1]/2):], vmin=-3, vmax=7, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[0,1].set_yticklabels(clusters_depths[order])
    ax[0,1].set_title("RIGHT STIM TRIALS")
    
    plt.sca(ax[0,2])
    sns.heatmap((sorte[:,:int(np.shape(sorte)[1]/2)] - sorte[:,int(np.shape(sorte)[1]/2):]), 
                vmin=-3, vmax=7, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,2].set_xlabel('Time from event (ms)')
    ax[0,2].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[0,2].set_yticklabels(clusters_depths[order])
    ax[0,2].set_title(r"$\Delta$ LEFT TRIALS - RIGHT TRIALS")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_block[:,:int(np.shape(sorte_left_block)[1]/2)], vmin=-3, center=0,  cmap="bwr",
                vmax=7, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('LEFT BLOCK \nLocation (depth um)')
    ax[1,0].set_yticklabels(clusters_depths[order])
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_left_block[:,int(np.shape(sorte_left_block)[1]/2):], vmin=-3,  center=0, cmap="bwr",
                vmax=7, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[1,1].set_yticklabels(clusters_depths[order])
    
    plt.sca(ax[1,2])
    stimulus_delta_left_block =  sorte_left_block[:,:int(np.shape(sorte_left_block)[1]/2)] \
                                - sorte_left_block[:,int(np.shape(sorte_left_block)[1]/2):]
    sns.heatmap(stimulus_delta_left_block, vmin=-3, center=0,  cmap="bwr",
                vmax=7, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,2].set_xlabel('Time from event (ms)')
    ax[1,2].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[1,2].set_yticklabels(clusters_depths[order])
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_right_block[:,:int(np.shape(sorte_right_block)[1]/2)], center=0,  cmap="bwr",
                vmin=-3, vmax=7, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[2,0].set_ylabel('RIGHT BLOCK \nLocation (depth um)')
    ax[2,0].set_yticklabels(clusters_depths[order])
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_block[:,int(np.shape(sorte_right_block)[1]/2):], center=0,  cmap="bwr",
                vmin=-3, vmax=7, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[2,1].set_yticklabels(clusters_depths[order])
    
    plt.sca(ax[2,2])
    stimulus_delta_right_block =  sorte_right_block[:,:int(np.shape(sorte_right_block)[1]/2)] \
        - sorte_right_block[:,int(np.shape(sorte_right_block)[1]/2):]
    
    sns.heatmap(stimulus_delta_right_block, vmin=-3, center=0, cmap="bwr",
                vmax=7, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,2].set_xlabel('Time from event (ms)')
    ax[2,2].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[2,2].set_yticklabels(clusters_depths[order])
    
    plt.sca(ax[3,0])
    block_delta_right_block =  sorte_left_block[:,:int(np.shape(sorte_left_block)[1]/2)] \
                                - sorte_right_block[:,:int(np.shape(sorte_right_block)[1]/2)]
    sns.heatmap(block_delta_right_block, vmin=-3, vmax=7, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,0].set_xlabel('Time from event (ms)')
    ax[3,0].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[3,0].set_ylabel(r'$\Delta$' + ' LEFT BLOCK -  RIGHT BLOCK \nLocation (depth um)')
    ax[3,0].set_yticklabels(clusters_depths[order])
   
    plt.sca(ax[3,1])
    block_delta_left_block = sorte_left_block[:,int(np.shape(sorte_left_block)[1]/2):] \
                            - sorte_right_block[:,int(np.shape(sorte_right_block)[1]/2):]
    sns.heatmap(block_delta_left_block, vmin=-3, vmax=7, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,1].set_xlabel('Time from event (ms)')
    ax[3,1].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[3,1].set_ylabel('Location')
    ax[3,1].set_yticklabels(clusters_depths[order])

    plt.sca(ax[3,2])
    delta_delta =  stimulus_delta_left_block - stimulus_delta_right_block
    sns.heatmap(delta_delta, vmin=-3, vmax=7, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,2].set_xlabel('Time from event (ms)')
    ax[3,2].set_xticklabels(np.arange(-200,500, bin_size*1000)
        , rotation='vertical')
    ax[3,2].set_yticklabels(clusters_depths[order])
    ax[3,2].set_title(r'$\Delta$ STIM SIDE (L-R) $\Delta$ BLOCK (L-R)')
    plt.tight_layout()
    
    if extra_ref is not None:
        if '/' in extra_ref: # Changes / from e.g layer 2/3 to avoid error at saving
           l_region = list(extra_ref) 
           l_region[region.find('/')] = '_'
           extra_ref = ''.join(l_region)

           plt.savefig(epoch + '_' + extra_ref +'_heatmap_common_L_2_R_order.png')
           plt.savefig(epoch + '_' + extra_ref +'_heatmap_common_L_2_R_order.svg')
    
def heatmap_sorted_by_pool_stimulus(spikeclusters, spiketimes,session, i, epoch,
                           bin_size=0.025):

    #Sorted with all data, plotted with blocks
    # ax[0,0] is not crossvalidated    
    # Calculate bins
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, good_clusters, session[epoch],
            bin_size=bin_size)[1]
    
    
    # Divide by stimulus side and opto block
    left_stim_trials_left_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == 1))
    right_stim_trials_left_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == 1))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == 0))
    
    # Divide by slide block
        
    sorte, order = heatmap_w_external_sorting(binned_firing_rate, order = None)
    sorte_cv, order_cv = heatmap_cross_validated(binned_firing_rate)
    
    sorte_left_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_stim_trials_left_block,:,:],order)
    sorte_left_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[left_stim_trials_right_block,:,:],order)
    sorte_right_stim_left_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_stim_trials_left_block,:,:],order)
    sorte_right_stim_right_block = \
        heatmap_w_external_sorting(binned_firing_rate[right_stim_trials_right_block,:,:],order)
    
    fig, ax  = plt.subplots(3,2, figsize=(10,17))
    plt.sca(ax[0,0])
    sns.heatmap(sorte)
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('Neuron Location (depth um)')
    ax[0,0].set_yticklabels(clusters_depths[order])
    ax[0,0].set_title("All")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte_cv)
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,1].set_ylabel('Neuron Location')
    ax[0,1].set_yticklabels(clusters_depths[order_cv])
    ax[0,1].set_title("All cross-validated")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('Neuron Location (depth um)')
    ax[1,0].set_yticklabels(clusters_depths[order])
    ax[1,0].set_title("Stimulus Left Side, Left Opto Block")
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_right_stim_left_block)
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,1].set_ylabel('Neuron Location')
    ax[1,1].set_yticklabels(clusters_depths[order])
    ax[1,1].set_title("Stimulus Right Side, Left Opto Block")
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_left_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,0].set_ylabel('Neuron Location')
    ax[2,0].set_yticklabels(clusters_depths[order])
    ax[2,0].set_title("Stimulus Left Side, Right Opto Block")
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_stim_right_block)
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[2,1].set_ylabel('Neuron Location')
    ax[2,1].set_yticklabels(clusters_depths[order])
    ax[2,1].set_title("Stimulus Right Side, Right Opto Block")
    plt.tight_layout()

    plt.savefig('stimulus_heatmap_common_order.png')
    plt.savefig('stimulus_heatmap_common_order.svg')
    


def heatmap_per_session_stimulus(spikeclusters, spiketimes,session, i, epoch, bin_size=0.025):

    #Cross validated plot, get order from half of the data, plot on the other half.
    #The halfs are chosen randomly
    
    # Calculate bins
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, good_clusters, session[epoch],
            bin_size=bin_size)[1]
    
    
    # Divide by stimulus side and opto block
    left_stim_trials_left_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == 1))
    right_stim_trials_left_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == 1))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == 0))
    
    # Divide by slide block
        
    
    sorte_left_stim_left_block, id_left_stim_left_block = \
        heatmap_cross_validated(binned_firing_rate[left_stim_trials_left_block,:,:])
    sorte_left_stim_right_block, id_left_stim_right_block = \
        heatmap_cross_validated(binned_firing_rate[left_stim_trials_right_block,:,:])
    sorte_right_stim_left_block, id_right_stim_left_block = \
        heatmap_cross_validated(binned_firing_rate[right_stim_trials_left_block,:,:])
    sorte_right_stim_right_block, id_right_stim_right_block = \
        heatmap_cross_validated(binned_firing_rate[right_stim_trials_right_block,:,:])
    
    fig, ax  = plt.subplots(2,2, figsize=(10,10))
    plt.sca(ax[0,0])
    sns.heatmap(sorte_left_stim_left_block)
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('Neuron Location (depth um)')
    ax[0,0].set_yticklabels(clusters_depths[id_left_stim_left_block])
    ax[0,0].set_title("Stimulus Left Side, Left Opto Block")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte_right_stim_left_block)
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,1].set_ylabel('Neuron Location')
    ax[0,1].set_yticklabels(clusters_depths[id_right_stim_left_block])
    ax[0,1].set_title("Stimulus Right Side, Left Opto Block")
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_stim_right_block)
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('Neuron Location')
    ax[1,0].set_yticklabels(clusters_depths[id_left_stim_right_block])
    ax[1,0].set_title("Stimulus Left Side, Right Opto Block")
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_right_stim_right_block)
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,1].set_ylabel('Neuron Location')
    ax[1,1].set_yticklabels(clusters_depths[id_right_stim_right_block])
    ax[1,1].set_title("Stimulus Right Side, Right Opto Block")
    plt.tight_layout()
    
    
    plt.savefig('stimulus_heatmap.png')
    plt.savefig('stimulus_heatmap.svg')
    
def heatmap_per_session_choice(spikeclusters, spiketimes,session, i, epoch, bin_size=0.025):

    #Cross validated plot, get order from half of the data, plot on the other half.
    #The halfs are chosen randomly
    
    # Calculate bins
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, good_clusters, session[epoch],
            bin_size=bin_size)[1]
    
    
    # Divide by stimulus side and opto block
    left_choice_trials_left_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 1))
    right_choice_trials_left_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 1))
    left_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 1))
    right_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 1))
    
    # Divide by slide block
        
    
    sorte_left_stim_left_block, id_left_stim_left_block = \
        heatmap_cross_validated(binned_firing_rate[left_choice_trials_left_block,:,:])
    sorte_left_stim_right_block, id_left_stim_right_block = \
        heatmap_cross_validated(binned_firing_rate[left_choice_trials_right_block,:,:])
    sorte_right_stim_left_block, id_right_stim_left_block = \
        heatmap_cross_validated(binned_firing_rate[right_choice_trials_left_block,:,:])
    sorte_right_stim_right_block, id_right_stim_right_block = \
        heatmap_cross_validated(binned_firing_rate[right_choice_trials_right_block,:,:])
    
    fig, ax  = plt.subplots(2,2, figsize=(10,10))
    plt.sca(ax[0,0])
    sns.heatmap(sorte_left_stim_left_block)
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,0].set_ylabel('Neuron Location (depth um)')
    ax[0,0].set_yticklabels(clusters_depths[id_left_stim_left_block])
    ax[0,0].set_title("Choice Left Side, Left Opto Block")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte_right_stim_left_block)
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[0,1].set_ylabel('Neuron Location')
    ax[0,1].set_yticklabels(clusters_depths[id_right_stim_left_block])
    ax[0,1].set_title("Choice Right Side, Left Opto Block")
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_stim_right_block)
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,0].set_ylabel('Neuron Location')
    ax[1,0].set_yticklabels(clusters_depths[id_left_stim_right_block])
    ax[1,0].set_title("Choice Left Side, Right Opto Block")
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_right_stim_right_block)
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500,bin_size*2*1000)
        , rotation='vertical')
    ax[1,1].set_ylabel('Neuron Location')
    ax[1,1].set_yticklabels(clusters_depths[id_right_stim_right_block])
    ax[1,1].set_title("Choice Right Side, Right Opto Block")
    plt.tight_layout()

    plt.savefig('choice_heatmap.png')
    plt.savefig('choice_heatmap.svg')
    


def heatmap_cross_validated(binned_firing_rate, bin_size=0.025):
    '''
    Given a set of trialsxbinxspikes array, generates a cross validated heatmap
    
    Parameters
    -------
    binned_firing_rate: trialsxbinxspikes array from bb.singlecell.calculate_peths
    
    Returns
    -------
    cross validated sorted heatmap
    
    '''

    # Get training and test set
    size_training = round(0.5 * binned_firing_rate.shape[0])
    train_set_trials =  random.sample(range(0, binned_firing_rate.shape[0]), 
                               size_training)
    test_set_trials  = np.setdiff1d(np.arange(binned_firing_rate.shape[0]),
                                    train_set_trials)
    
    assert len(np.intersect1d(train_set_trials,test_set_trials))==0
    
    train_set = binned_firing_rate[train_set_trials, :,:]
    test_set = binned_firing_rate[test_set_trials, :,:]
    ## z score and order by training set
    mean_binned_fr =  np.mean(train_set/bin_size, axis =0)
    
    z_score_binned_spikes = stats.zscore(mean_binned_fr, axis =1)
    
    # Replace np.nan by 0
    z_score_binned_spikes = np.nan_to_num(z_score_binned_spikes)
    
    order = np.argmax(z_score_binned_spikes, 1)
    
    # make heatmap matrix
    mean_binned_fr_test =  np.mean(test_set/bin_size, axis =0)
    z_score_binned_spikes_test =  stats.zscore(mean_binned_fr_test, axis =1)
    z_score_binned_spikes_test = np.nan_to_num(z_score_binned_spikes_test)
        
    sorte = z_score_binned_spikes_test[order.argsort()]
    
    # map new order to cluster ids base on sum of bins in row to be extra sure
    ordered_ids = order.argsort()                     
    
    return sorte, ordered_ids


def heatmap_append_L_R(binned_firing_rate1,binned_firing_rate2, order = None , bin_size = 0.025):
    '''
    Warning: Without external sorting order, this function does not crossvalidate
    Given a set of trialsxbinxspikes array, generates a cross validated heatmap
    
    Parameters
    -------
    binned_firing_rate: trialsxbinxspikes array from bb.singlecell.calculate_peths
    
    Returns
    -------
    cross validated sorted heatmap
    
    '''

    # make heatmap matrix
    mean_binned_fr_test1 =  np.mean(binned_firing_rate1, axis =0)/bin_size
    mean_binned_fr_test2 =  np.mean(binned_firing_rate2, axis =0)/bin_size
    mean_binned_fr_test = np.concatenate((mean_binned_fr_test1,
                                          mean_binned_fr_test2),axis =1)
    z_score_binned_spikes_test = stats.zscore(mean_binned_fr_test, axis =1)
    z_score_binned_spikes_test = np.nan_to_num(z_score_binned_spikes_test)
    
    if order is not None:
        sorte = z_score_binned_spikes_test[order]
        return sorte
    else:
        order = np.argmax(z_score_binned_spikes_test, 1)
        sorte = z_score_binned_spikes_test[order.argsort()]
    
        return sorte, order.argsort()





def heatmap_append_L_R_pooled_session(binned_firing_rate1,binned_firing_rate2, 
                                      index_1, index_2, session, cluster_session, order = None , bin_size = 0.025):
    '''
    Heatmap function that allows for several sessions and animals for same region
    Warning: Without external sorting order, this function does not crossvalidate
    Given a set of trialsxbinxspikes array, generates a cross validated heatmap
    
    Parameters
    -------
    binned_firing_rate: trialsxbinxspikes array from bb.singlecell.calculate_peths
    session : dataframe with session information
    index_1 and  index_2:  index to relate binned firing rate1 and 2 to dataframe with pooled
    behavior
    
    Returns
    -------
    cross validated sorted heatmap
    
    '''

    # make heatmap matrix
    mean_binned_fr_test = None
    counter_session = 0
    for mouse in session['mouse_name'].unique():
        for ses in session.loc[session['mouse_name'] == mouse, 'ses'].unique():
            # First number from cluster determines session
            
            cluster_in_session = np.where(cluster_session == counter_session)
            if np.size(cluster_in_session) == 0:
                counter_session += 1
                continue
            
            counter_session += 1
            
            m_ses = session.loc[(session['mouse_name'] == mouse) & 
                                (session['ses'] == ses)]
            trials_in_session = m_ses['idx'].to_numpy()
            trials_in_sess_in_var1 = np.intersect1d(trials_in_session, index_1)
            tvar1 = (index_1[:, None] == trials_in_sess_in_var1).argmax(axis=0)

            trials_in_sess_in_var2 = np.intersect1d(trials_in_session, index_2)
            tvar2 = (index_2[:, None] == trials_in_sess_in_var2).argmax(axis=0)
            
            
            binned_firing_rate1_select = binned_firing_rate1[:, cluster_in_session[0],:]
            binned_firing_rate2_select = binned_firing_rate2[:, cluster_in_session[0],:]

            
            mean_binned_fr_test1 =  np.mean(binned_firing_rate1_select[tvar1, :,:], axis =0)/bin_size
            mean_binned_fr_test2 =  np.mean(binned_firing_rate2_select[tvar2, :,:], axis =0)/bin_size
            mean_binned_fr_test_temp = np.concatenate((mean_binned_fr_test1,
                                                  mean_binned_fr_test2),axis =1)
            if mean_binned_fr_test is None:
                mean_binned_fr_test = mean_binned_fr_test_temp
            
            else:
                mean_binned_fr_test = np.concatenate((mean_binned_fr_test,
                                                      mean_binned_fr_test_temp) , axis = 0)
        
    z_score_binned_spikes_test = stats.zscore(mean_binned_fr_test, axis =1)
    z_score_binned_spikes_test = np.nan_to_num(z_score_binned_spikes_test)
        
    
    if order is not None:
        sorte = z_score_binned_spikes_test[order]
        return sorte
    else:
        order = np.argmax(z_score_binned_spikes_test, 1)
        sorte = z_score_binned_spikes_test[order.argsort()]
    
        return sorte, order.argsort()





def mean_binned_firing_rate(binned_firing_rate1,
                                      index_1, session, cluster_session, 
                                      order=None , bin_size = 0.025,
                                      smoothing=0):
    '''
    UNDER deveelopment
    Heatmap function that allows for several sessions and animals for same region
    Warning: Without external sorting order, this function does not crossvalidate
    Given a set of trialsxbinxspikes array, generates a cross validated heatmap
    
    Parameters
    -------
    binned_firing_rate: trialsxbinxspikes array from bb.singlecell.calculate_peths
    session : dataframe with session information
    index_1 :  index to relate binned firing rate1 to dataframe with pooled
    behavior
    
    Returns
    -------
    cross validated sorted heatmap
    
    '''

    # Create convolution window
    if smoothing != 0:
        n_bins = np.shape(binned_firing_rate1)[2]
        window = gaussian(n_bins, std=smoothing/ bin_size)
        window /= np.sum(window)

    # make heatmap matrix
    mean_binned_fr_test = None
    counter_session = 0
    for mouse in session['mouse_name'].unique():
        for ses in session.loc[session['mouse_name'] == mouse, 'ses'].unique():
            # First number from cluster determines session
            
            cluster_in_session = np.where(cluster_session == counter_session)
            if np.size(cluster_in_session) == 0:
                counter_session += 1
                continue
            
            counter_session += 1
            
            m_ses = session.loc[(session['mouse_name'] == mouse) & 
                                (session['ses'] == ses)]
            trials_in_session = m_ses['idx'].to_numpy()
            trials_in_sess_in_var1 = np.intersect1d(trials_in_session, index_1)
            tvar1 = (index_1[:, None] == trials_in_sess_in_var1).argmax(axis=0)
            
            
            binned_firing_rate1_select = binned_firing_rate1[:, cluster_in_session[0],:]
            binned_firing_rate1_select = binned_firing_rate1_select[tvar1, :,:]
            
            if smoothing != 0:  
                for i in range(np.shape(binned_firing_rate1_select)[0]):
                    for j in range(np.shape(binned_firing_rate1_select)[1]):
                        binned_firing_rate1_select[i,j,:] = convolve(binned_firing_rate1_select[i,j,:], window,
                                                mode='same', method='auto')
            
            mean_binned_fr_test_temp = np.mean(binned_firing_rate1_select, axis=0)/bin_size
            # Trim to standard size -0.2 to +0.5
            
            if smoothing != 0:  
                to_trim = int(smoothing/bin_size)
                mean_binned_fr_test_temp = mean_binned_fr_test_temp[:, to_trim:-to_trim]
            
            if mean_binned_fr_test is None:
                mean_binned_fr_test = mean_binned_fr_test_temp
            
            else:
                mean_binned_fr_test = np.concatenate((mean_binned_fr_test,
                                                      mean_binned_fr_test_temp) , axis = 0)
            

    return mean_binned_fr_test
            

def z_score_and_order(concatenated_mean_firing_rates, window, order = None ):
   
    z_score_binned_spikes_test = stats.zscore(concatenated_mean_firing_rates, axis =1)
    z_score_binned_spikes_test = np.nan_to_num(z_score_binned_spikes_test)
    
    z_score_binned_spikes_test_select= z_score_binned_spikes_test[:,
                                                window[0]:window[1]]
    
    if order is not None:
        sorte = z_score_binned_spikes_test_select[order]
        return sorte
    else:
        order = np.argmax(z_score_binned_spikes_test_select, 1)
        sorte = z_score_binned_spikes_test_select[order.argsort()]
    
        return sorte, order.argsort()




def heatmap_w_external_sorting(binned_firing_rate, order = None, bin_size=0.025):
    '''
    Warning: Without external sorting order, this function does not crossvalidate
    Given a set of trialsxbinxspikes array, generates a cross validated heatmap
    
    Parameters
    -------
    binned_firing_rate: trialsxbinxspikes array from bb.singlecell.calculate_peths
    
    Returns
    -------
    cross validated sorted heatmap
    
    '''

    # make heatmap matrix
    mean_binned_fr_test =  np.mean(binned_firing_rate, axis =0)/bin_size
    z_score_binned_spikes_test = stats.zscore(mean_binned_fr_test, axis =1)
    z_score_binned_spikes_test = np.nan_to_num(z_score_binned_spikes_test)
    
    if order is not None:
        sorte = z_score_binned_spikes_test[order]
        return sorte
    else:
        order = np.argmax(z_score_binned_spikes_test, 1)
        sorte = z_score_binned_spikes_test[order.argsort()]
    
        return sorte, order.argsort()
    

    



# Ploting PSTH
# ******************************** Handler ************************************#
def session_single_neuron_psth_summary(spikeclusters, spiketimes,session, session_folder):
    
    if path.exists(path.exists(session_folder + '/figures/')):
        pass
    else:
        os.mkdir(session_folder + '/figures/')
    
    for i in good_clusters:
        early_vs_late_block (spikeclusters, spiketimes,session, i)
        plt.savefig(session_folder + '/figures/%d_early_vs_late_psth.pdf' %i,dpi=300)
        feedback_opto_vs_non_opto(spikeclusters, spiketimes,session, i)
        plt.savefig(session_folder + '/figures/%d_feedback_opto_vs_nonopto.pdf' %i,dpi=300)
        stim_after_opto(spikeclusters, spiketimes,session, i)
        plt.savefig(session_folder + '/figures/%d_stim_after_opto_psth.pdf' %i,dpi=300)
        stim_after_reward(spikeclusters, spiketimes,session, i)
        plt.savefig(session_folder + '/figures/%d_stim_after_reward_psth.pdf' %i,dpi=300)
        tuning_opto(spikeclusters, spiketimes,session, i)
        plt.savefig(session_folder + '/figures/%d_tuning_psth.pdf' %i,dpi=300)
        direct_opto_effect(spikeclusters, spiketimes,session, i)
        plt.savefig(session_folder + '/figures/%d_direct_opto_effect.pdf' %i,dpi=300)
        raster_plot_cue(spikeclusters, spiketimes,session, i)
        plt.savefig(session_folder + '/figures/%d_raster_cue.pdf' %i,dpi=300)
        raster_plot_feedback(spikeclusters, spiketimes,session, i)
        plt.savefig(session_folder + '/figures/%d_raster_outocome.pdf' %i,dpi=300)

def direct_opto_effect(spikeclusters, spiketimes,session, i):

    try:
        figure, ax = plt.subplots(2, 2,figsize=(25, 25))
        figure.suptitle('Direct opto effect, cluster %d'%i, fontsize=25)
        plt.sca(ax[0,0])
        ax[0,0].set_title('Right Choice, Rewarded, non-opto(leftblock) vs opto(rightblock)', fontsize=12)
        epoch1 = session.loc[(session['choice'] == -1) & (session['opto.npy'] == 0) & (session['opto_probability_left'] == 1) & (session['feedbackType'] == 1), 'response_times']
        epoch2 = session.loc[(session['choice'] == -1) & (session['opto.npy'] == 1) & (session['opto_probability_left'] == 0) & (session['feedbackType'] == 1), 'response_times']
        
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        
        #
        plt.sca(ax[0,1])
        ax[0,1].set_title('Right Choice, Unrewarded, non-opto(leftblock) vs opto(rightblock)', fontsize=12)
        epoch1 = session.loc[(session['choice'] == -1) & (session['opto.npy'] == 0) & (session['opto_probability_left'] == 1) & (session['feedbackType'] == -1), 'response_times']
        epoch2 = session.loc[(session['choice'] == -1) & (session['opto.npy'] == 1) & (session['opto_probability_left'] == 0) & (session['feedbackType'] == -1), 'response_times']
        
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        
        
        #
        plt.sca(ax[1,0])
        ax[1,0].set_title('Left Choice, Rewarded, non-opto(rightblock) vs opto(leftblock)', fontsize=12)
        epoch1 = session.loc[(session['choice'] == 1) & (session['opto.npy'] == 0) & (session['opto_probability_left'] == 0) & (session['feedbackType'] == 1), 'response_times']
        epoch2 = session.loc[(session['choice'] == 1) & (session['opto.npy'] == 1) & (session['opto_probability_left'] == 1) & (session['feedbackType'] == 1), 'response_times']
        
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        
        #
        plt.sca(ax[1,1])
        ax[1,1].set_title('Left Choice, Unrewarded, non-opto(rightblock) vs opto(leftblock)', fontsize=12)
        epoch1 = session.loc[(session['choice'] == 1) & (session['opto.npy'] == 0) & (session['opto_probability_left'] == 0) & (session['feedbackType'] == -1), 'response_times']
        epoch2 = session.loc[(session['choice'] == 1) & (session['opto.npy'] == 1) & (session['opto_probability_left'] == 1) & (session['feedbackType'] == -1), 'response_times']
        
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
    except:
                print('error in direct opto effect figure')
        
def tuning_opto(spikeclusters, spiketimes,session, i):
    try:
        figure, ax = plt.subplots(2, 4,figsize=(25, 10))
        figure.suptitle('tuning before and after, cluster %d'%i, fontsize=25)
        plt.sca(ax[0,0])
        ax[0,0].set_title('Stim: 0', fontsize=12)
        epoch1 = session.loc[((session['contrastLeft'] == 0) | (session['contrastRight'] == 0)) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[((session['contrastLeft'] == 0) | (session['contrastRight'] == 0)) & (session['after_opto'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[0,1])
        ax[0,1].set_title('Stim: -6.25%', fontsize=12)
        epoch1 = session.loc[(session['contrastLeft'] == 0.0625) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] == 0.0625) & (session['after_opto'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[0,2])
        ax[0,2].set_title('Stim: -12.5%', fontsize=12)
        epoch1 = session.loc[(session['contrastLeft'] == 0.125) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] == 0.125) & (session['after_opto'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,2],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,2],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[0,3])
        ax[0,3].set_title('Stim: -25%', fontsize=12)
        epoch1 = session.loc[(session['contrastLeft'] == 0.25) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] == 0.25) & (session['after_opto'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,3],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,3],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[1,0])
        ax[1,0].set_title('Stim: 0', fontsize=12)
        epoch1 = session.loc[((session['contrastLeft'] == 0) | (session['contrastRight'] == 0)) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[((session['contrastLeft'] == 0) | (session['contrastRight'] == 0)) & (session['after_opto'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[1,1])
        ax[1,1].set_title('Stim: 6.25%', fontsize=12)
        epoch1 = session.loc[(session['contrastRight'] == 0.0625) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] == 0.0625) & (session['after_opto'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[1,2])
        ax[1,2].set_title('Stim: 12.5%', fontsize=12)
        epoch1 = session.loc[(session['contrastRight'] == 0.125) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] == 0.125) & (session['after_opto'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,2],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,2],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[1,3])
        ax[1,3].set_title('Stim: 25%', fontsize=12)
        epoch1 = session.loc[(session['contrastRight'] == 0.25) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] == 0.25) & (session['after_opto'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,3],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,3],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
                
        blue_patch = mpatches.Patch(color='blue', label='After non-opto')
        green_patch  =  mpatches.Patch(color='green', label='After opto')
        plt.legend(handles=[green_patch, blue_patch], loc = 'lower right')
            
    except:
                print('error in tuning figure')

def stim_after_reward(spikeclusters, spiketimes,session,i):
    try:
        figure, ax = plt.subplots(1, 3,figsize=(15, 6))
        figure.suptitle('Feedback Stim after reward vs non-reward, cluster %d'%i, fontsize=25)
        plt.sca(ax[0])
        ax[0].set_title('Stim: Left, Epoch: Stim, Block: Neutral', fontsize=12)
        epoch1 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==-1) & (session['after_reward'] == -1), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==-1) & (session['after_reward'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[1])
        ax[1].set_title('Stim: Right, Epoch: Stim, Block: Neutral', fontsize=12)
        
        epoch1 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']==-1) & (session['after_reward'] == -1), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']==-1) & (session['after_reward'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        #
        plt.sca(ax[2])
        ax[2].set_title('Stim: 0, Epoch: Stim, Block: Neutral', fontsize=12)
        
        epoch1 = session.loc[((session['contrastRight'] == 0) | (session['contrastLeft'] == 0)) & (session['opto_probability_left']==-1) & (session['after_reward'] == -1), 'goCue_times']
        epoch2 = session.loc[((session['contrastRight'] == 0) | (session['contrastLeft'] == 0)) & (session['opto_probability_left']==-1) & (session['after_reward'] == 1), 'goCue_times']
        try:
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2],
                    pethline_kwargs={'color': 'blue', 'lw': 2},
                    errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
                
                peri_event_time_histogram(
                    spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                    t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                    include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2],
                    pethline_kwargs={'color': 'green', 'lw': 2},
                    errbar_kwargs={'color': 'green', 'alpha': 0.5},
                    eventline_kwargs={'color': 'black', 'alpha': 0.5},
                    raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
                print('not enough trials')
        blue_patch = mpatches.Patch(color='blue', label='After error')
        green_patch  =  mpatches.Patch(color='green', label='After reward')
        plt.legend(handles=[green_patch, blue_patch], loc = 'lower right')
            
    except:
        print('error after reward figure')

def stim_after_opto(spikeclusters, spiketimes,session, i):
    
    try:
        figure, ax = plt.subplots(3, 2,figsize=(20, 20))
        figure.suptitle('Feedback Stim after opto vs non-opto, cluster %d'%i, fontsize=25)
        plt.sca(ax[0,0])
        ax[0,0].set_title('Stim: Left, Epoch: Stim, Block: Neutral', fontsize=12)
        epoch1 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==-1) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==-1) & (session['after_opto'] == 1), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        #
        plt.sca(ax[1,0])
        ax[1,0].set_title('Stim: Left, Epoch: Stim, Block: Right', fontsize=12)
        epoch1 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']== 0) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']== 0) & (session['after_opto'] == 1), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        #  
        plt.sca(ax[2,0])
        ax[2,0].set_title('Stim: Left, Epoch: Stim, Block: Left', fontsize=12)
        epoch1 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']== 1) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']== 1) & (session['after_opto'] == 1), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        #  
        plt.sca(ax[0,1])
        ax[0,1].set_title('Stim: Right, Epoch: Stim, Block: Neutral', fontsize=12)
        epoch1 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']== -1) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']== -1) & (session['after_opto'] == 1), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        
        #  
        plt.sca(ax[1,1])
        ax[1,1].set_title('Stim: Right, Epoch: Stim, Block: Right', fontsize=12)
        epoch1 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']== 0) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']== 0) & (session['after_opto'] == 1), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        #
        plt.sca(ax[2,1])
        ax[2,1].set_title('Stim: Right, Epoch: Stim, Block: Left', fontsize=12)
        epoch1 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']== 1) & (session['after_opto'] == 0), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']== 1) & (session['after_opto'] == 1), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        blue_patch = mpatches.Patch(color='blue', label='After non-laser')
        green_patch  =  mpatches.Patch(color='green', label='after laser')
        plt.legend(handles=[green_patch, blue_patch], loc = 'lower right')
    
    
    
    except:
        print('error in feedback stim_after_opto figure')
        
    
def feedback_opto_vs_non_opto(spikeclusters, spiketimes,session, i):
    try:
        figure, ax = plt.subplots(3, 2,figsize=(20, 20))
        figure.suptitle('Feedback Opto vs non-opto, cluster %d'%i, fontsize=25)
        plt.sca(ax[0,0])
        ax[0,0].set_title('Rewarded: True, Epoch: Feedback, Block: Neutral', fontsize=12)
        epoch1 = session.loc[(session['feedbackType'] == 1) & (session['opto_probability_left']==-1) & (session['opto.npy'] == 0), 'feedback_times']
        epoch2 = session.loc[(session['feedbackType'] == 1) & (session['opto_probability_left']==-1) & (session['opto.npy'] == 1), 'feedback_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        #
        plt.sca(ax[1,0])
        ax[1,0].set_title('Rewarded: True, Epoch: Feedback, Block: Right', fontsize=12)
        epoch1 = session.loc[(session['feedbackType'] == 1) & (session['opto_probability_left']== 0) & (session['opto.npy'] == 0), 'feedback_times']
        epoch2 = session.loc[(session['feedbackType'] == 1) & (session['opto_probability_left']== 0) & (session['opto.npy'] == 1), 'feedback_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        #  
        plt.sca(ax[2,0])
        ax[2,0].set_title('Rewarded: True, Epoch: Feedback, Block: Left', fontsize=12)
        epoch1 = session.loc[(session['feedbackType'] == 1) & (session['opto_probability_left']== 1) & (session['opto.npy'] == 0), 'feedback_times']
        epoch2 = session.loc[(session['feedbackType'] == 1) & (session['opto_probability_left']== 1) & (session['opto.npy'] == 1), 'feedback_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        #  
        plt.sca(ax[0,1])
        ax[0,1].set_title('Rewarded: False, Epoch: Feedback, Block: Neutral', fontsize=12)
        epoch1 = session.loc[(session['feedbackType'] == -1) & (session['opto_probability_left']== -1) & (session['opto.npy'] == 0), 'feedback_times']
        epoch2 = session.loc[(session['feedbackType'] == -1) & (session['opto_probability_left']== -1) & (session['opto.npy'] == 1), 'feedback_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        
        #  
        plt.sca(ax[1,1])
        ax[1,1].set_title('Rewarded: False, Epoch: Feedback, Block: Right', fontsize=12)
        epoch1 = session.loc[(session['feedbackType'] == -1) & (session['opto_probability_left']== 0) & (session['opto.npy'] == 0), 'feedback_times']
        epoch2 = session.loc[(session['feedbackType'] == -1) & (session['opto_probability_left']== 0) & (session['opto.npy'] == 1), 'feedback_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        #
        plt.sca(ax[2,1])
        ax[2,1].set_title('Rewarded: False, Epoch: Feedback, Block: Left', fontsize=12)
        epoch1 = session.loc[(session['feedbackType'] == -1) & (session['opto_probability_left']== 1) & (session['opto.npy'] == 0), 'feedback_times']
        epoch2 = session.loc[(session['feedbackType'] == -1) & (session['opto_probability_left']== 1) & (session['opto.npy'] == 1), 'feedback_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        blue_patch = mpatches.Patch(color='blue', label='Laser off (1s)')
        green_patch  =  mpatches.Patch(color='green', label='Laser on (1s)')
        plt.legend(handles=[green_patch, blue_patch], loc = 'lower right')


def raster_plot_cue(spikeclusters, spiketimes,session, i):
    try:
        epoch1 = session['goCue_times']
        fig, ax = plt.subplots()
        peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=True, n_rasters=50, error_bars='sem',
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
    except:
        print('not enough trials')

def raster_plot_feedback(spikeclusters, spiketimes,session, i):
    try:
        epoch1 = session['feedback_times']
        fig, ax = plt.subplots()
        peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=True, n_rasters=50, error_bars='sem',
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
    except:
        print('not enough trials')

def early_vs_late_block (spikeclusters, spiketimes,session, i):
    '''
    i: clusterID
    '''
    try:
        figure, ax = plt.subplots(3, 4, figsize=(20, 20))
        figure.suptitle('Early vs Late in block, cluster %d'%i, fontsize=25)
        # Stim_side: Left, Epoch: GoCue, Block: Neutral
        plt.sca(ax[0,0])
        ax[0,0].set_title('Stim_side: Left, Epoch: GoCue, Block: Neutral', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==-1) & (session['trial_within_block']<15), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==-1) & (session['trial_within_block']>35), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        # Stim_side: Left, Epoch: GoCue, Block: Right
        plt.sca(ax[1,0])
        ax[1,0].set_title('Stim_side: Left, Epoch: GoCue, Block: Right', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==0) & (session['trial_within_block']<15), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==0) & (session['trial_within_block']>35), 'goCue_times']
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        # Stim_side: Left, Epoch: GoCue, Block: Left
        plt.sca(ax[2,0])
        ax[2,0].set_title('Stim_side: Left, Epoch: GoCue, Block: Left', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==1) & (session['trial_within_block']<15), 'goCue_times']
        epoch2 = session.loc[(session['contrastLeft'] > 0) & (session['opto_probability_left']==1) & (session['trial_within_block']>35), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,0],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,0],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        ##
        # Stim_side: Right, Epoch: GoCue, Block: Neutral
        plt.sca(ax[0,1])
        ax[0,1].set_title('Stim_side: Right, Epoch: GoCue, Block: Neutral', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']==-1) & (session['trial_within_block']<15), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']==-1) & (session['trial_within_block']>35), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        # Stim_side: Right, Epoch: GoCue, Block: Right
        plt.sca(ax[1,1])
        ax[1,1].set_title('Stim_side: Right, Epoch: GoCue, Block: Right', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']==0) & (session['trial_within_block']<15), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']==0) & (session['trial_within_block']>35), 'goCue_times']
        
        try:  
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        
        except:
            print('not enough trials')
        
        # Stim_side: Right, Epoch: GoCue, Block: Left
        plt.sca(ax[2,1])
        ax[2,1].set_title('Stim_side: Right, Epoch: GoCue, Block: Right', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']==1) & (session['trial_within_block']<15), 'goCue_times']
        epoch2 = session.loc[(session['contrastRight'] > 0) & (session['opto_probability_left']==1) & (session['trial_within_block']>35), 'goCue_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,1],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,1],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        
        ##
        # choice: Left, Epoch: response, Block: Neutral
        plt.sca(ax[0,2])
        ax[0,2].set_title('choice: Left, Epoch: response, Block: Neutral', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['choice'] == 1) & (session['opto_probability_left']==-1) & (session['trial_within_block']<15), 'response_times']
        epoch2 = session.loc[(session['choice'] == 1) & (session['opto_probability_left']==-1) & (session['trial_within_block']>35), 'response_times']
        
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,2],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,2],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        
        # choice: Left, Epoch: response, Block: Right
        plt.sca(ax[1,2])
        ax[1,2].set_title('choice: Left, Epoch: response, Block: Right', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['choice'] == 1) & (session['opto_probability_left']==0) & (session['trial_within_block']<15), 'response_times']
        epoch2 = session.loc[(session['choice'] == 1) & (session['opto_probability_left']==0) & (session['trial_within_block']>35), 'response_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,2],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,2],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        # choice: Left, Epoch: response, Block: Left
        plt.sca(ax[2,2])
        ax[2,2].set_title('choice: Left, Epoch: response, Block: Left', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['choice'] == 1) & (session['opto_probability_left']==1) & (session['trial_within_block']<15), 'response_times']
        epoch2 = session.loc[(session['choice'] == 1) & (session['opto_probability_left']==1) & (session['trial_within_block']>35), 'response_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,2],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,2],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        ##
        # choice: Right, Epoch: response, Block: Neutral
        plt.sca(ax[0,3])
        ax[0,3].set_title('choice: Right, Epoch: response, Block: Neutral', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['choice'] == -1) & (session['opto_probability_left']==-1) & (session['trial_within_block']<15), 'response_times']
        epoch2 = session.loc[(session['choice'] == -1) & (session['opto_probability_left']==-1) & (session['trial_within_block']>35), 'response_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,3],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[0,3],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        # choice: right, Epoch: response, Block: Right
        plt.sca(ax[1,3])
        ax[1,3].set_title('choice: Right, Epoch: response, Block: Right', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['choice'] == -1) & (session['opto_probability_left']==0) & (session['trial_within_block']<15), 'response_times']
        epoch2 = session.loc[(session['choice'] == -1) & (session['opto_probability_left']==0) & (session['trial_within_block']>35), 'response_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,3],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[1,3],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        except:
            print('not enough trials')
        # choice: right, Epoch: response, Block: Left
        plt.sca(ax[2,3])
        ax[2,3].set_title('choice: Right, Epoch: response, Block: Left', fontsize=12)
        
        # Divide trials
        epoch1 = session.loc[(session['choice'] == -1) & (session['opto_probability_left']==1) & (session['trial_within_block']<15), 'response_times']
        epoch2 = session.loc[(session['choice'] == -1) & (session['opto_probability_left']==1) & (session['trial_within_block']>35), 'response_times']
        
        try:
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch1, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,3],
                pethline_kwargs={'color': 'blue', 'lw': 2},
                errbar_kwargs={'color': 'blue', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
            
            peri_event_time_histogram(
                spiketimes, spikeclusters, epoch2, i,  # Everything you need for a basic plot
                t_before=0.2, t_after=0.5, bin_size=0.01, smoothing=0.025, as_rate=True,
                include_raster=False, n_rasters=False, error_bars='sem', ax=ax[2,3],
                pethline_kwargs={'color': 'green', 'lw': 2},
                errbar_kwargs={'color': 'green', 'alpha': 0.5},
                eventline_kwargs={'color': 'black', 'alpha': 0.5},
                raster_kwargs={'color': 'black', 'lw': 0.5})
        
        except:
            print('not enough trials')
        blue_patch = mpatches.Patch(color='blue', label='Early')
        green_patch  =  mpatches.Patch(color='green', label='Late')
        plt.legend(handles=[green_patch, blue_patch], loc = 'lower right')
        
    except:
        print('error in early vs late fig')
        




    
def sort_non_opto_plot_by_block_and_subtraction_per_region(spikeclusters, spiketimes,session, 
                           epoch, cluster_select, clusters_depths,
                           bin_size=0.025, extra_ref = None, thirdclass = None,
                           fourthclass = None, smoothing_win = 0):
    
    
    #Selecta bigger window note pre_time and postdime.  Smooths and the cuts edges to standard -200 to +500 window
    
    
    matplotlib.rcParams.update({'font.size': 22})
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, cluster_select, session[epoch],
            bin_size=bin_size,  smoothing = 0, pre_time= round(0.2 + smoothing_win, 3), 
            post_time=round(0.5 + smoothing_win, 3))[1]
    
    # Cluster are added 10000 in every session to identify them when pooling, this
    # retrieves session identity
    cluster_session = np.floor(cluster_select/10000)

    
    # Divide by stimulus side and opto block
    left_stim_trials_neutral_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                        np.where(session['opto_probability_left'] == -1))
    right_stim_trials_neutral_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                        np.where(session['opto_probability_left'] == -1))
    left_stim_trials_left_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                        np.where(session['opto_probability_left'] == 1))
    right_stim_trials_left_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                        np.where(session['opto_probability_left'] == 1))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                        np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                        np.where(session['opto_probability_left'] == 0))
    if thirdclass != None:
        left_stim_trials_neutral_block = np.intersect1d(left_stim_trials_neutral_block, \
                                            np.where(session[thirdclass] == fourthclass))
        right_stim_trials_neutral_block = np.intersect1d(right_stim_trials_neutral_block, \
                                            np.where(session[thirdclass] == fourthclass))
        left_stim_trials_left_block = np.intersect1d(left_stim_trials_left_block, \
                                            np.where(session[thirdclass] == fourthclass))
        right_stim_trials_left_block = np.intersect1d(right_stim_trials_left_block, \
                                            np.where(session[thirdclass] == fourthclass))
        left_stim_trials_right_block = np.intersect1d(left_stim_trials_right_block, \
                                            np.where(session[thirdclass] == fourthclass))
        right_stim_trials_right_block = np.intersect1d(right_stim_trials_right_block, \
                                            np.where(session[thirdclass] == fourthclass))
        

    # Concatenate neutral L and R trials
    L = binned_firing_rate[left_stim_trials_neutral_block,:,:]
    R = binned_firing_rate[right_stim_trials_neutral_block,:,:]
    L_blockL = binned_firing_rate[left_stim_trials_left_block,:,:]
    R_blockL = binned_firing_rate[right_stim_trials_left_block,:,:]
    L_blockR = binned_firing_rate[left_stim_trials_right_block,:,:]
    R_blockR = binned_firing_rate[right_stim_trials_right_block,:,:]
        
        
        
    # Obtain firing rates
    N_L = mean_binned_firing_rate(L, left_stim_trials_neutral_block, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    N_R = mean_binned_firing_rate(R, right_stim_trials_neutral_block, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    L_L = mean_binned_firing_rate(L_blockL, left_stim_trials_left_block, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    L_R = mean_binned_firing_rate(R_blockL, right_stim_trials_left_block, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    R_L = mean_binned_firing_rate(L_blockR, left_stim_trials_right_block, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    R_R = mean_binned_firing_rate(R_blockR, right_stim_trials_right_block, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    
    # Concatenate
    
    concatenated_mean_firing_rates = np.concatenate((N_L, N_R, L_L, L_R, R_L, R_R)
                                              , axis = 1)
    
        
    # Sort based by neutral block
    
    sorte, order = z_score_and_order(concatenated_mean_firing_rates, (0,56),
                                     order = None )
    
    # Apply sort
    
    sorte_left_block = \
        z_score_and_order(concatenated_mean_firing_rates, (56,112),
                                     order = order )
        
    sorte_right_block = \
        z_score_and_order(concatenated_mean_firing_rates, (112,168),
                                     order = order )
    
    # Plot
    
    fig, ax  = plt.subplots(4,3, figsize=(25,30), sharex=True)
    plt.sca(ax[0,0])
    sns.heatmap(sorte[:,:int(np.shape(sorte)[1]/2)], vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[0,0].set_ylabel('NEUTRAL BLOCK', rotation='vertical',x=-0.1,y=0.5)
    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if epoch=='goCue_times':
        ax[0,0].set_title("LEFT STIM TRIALS")
    elif epoch=='feedback_times':
        ax[0,0].set_title("LEFT CHOICE TRIALS")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte[:,int(np.shape(sorte)[1]/2):], vmin=-5, vmax=5, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    if epoch=='goCue_times':
        ax[0,1].set_title("RIGHT STIM TRIALS")
    elif epoch=='feedback_times':
        ax[0,1].set_title("RIGHT CHOICE TRIALS")
    
    plt.sca(ax[0,2])
    sns.heatmap((sorte[:,:int(np.shape(sorte)[1]/2)] - sorte[:,int(np.shape(sorte)[1]/2):]), 
                vmin=-3, vmax=7, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,2].set_xlabel('Time from event (ms)')
    ax[0,2].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[0,2].set_title(r"$\Delta$ LEFT TRIALS - RIGHT TRIALS")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_block[:,:int(np.shape(sorte_left_block)[1]/2)], vmin=-5, center=0,  cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[1,0].set_ylabel('LEFT BLOCK')
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_left_block[:,int(np.shape(sorte_left_block)[1]/2):], vmin=-5,  center=0, cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[1,2])
    stimulus_delta_left_block =  sorte_left_block[:,:int(np.shape(sorte_left_block)[1]/2)] \
                                - sorte_left_block[:,int(np.shape(sorte_left_block)[1]/2):]
    sns.heatmap(stimulus_delta_left_block, vmin=-5, center=0,  cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,2].set_xlabel('Time from event (ms)')
    ax[1,2].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_right_block[:,:int(np.shape(sorte_right_block)[1]/2)], center=0,  cmap="bwr",
                vmin=-5, vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[2,0].set_ylabel('RIGHT BLOCK')
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_block[:,int(np.shape(sorte_right_block)[1]/2):], center=0,  cmap="bwr",
                vmin=-5, vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[2,2])
    stimulus_delta_right_block =  sorte_right_block[:,:int(np.shape(sorte_right_block)[1]/2)] \
        - sorte_right_block[:,int(np.shape(sorte_right_block)[1]/2):]
    
    sns.heatmap(stimulus_delta_right_block, vmin=-5, center=0, cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,2].set_xlabel('Time from event (ms)')
    ax[2,2].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[3,0])
    block_delta_right_block =  sorte_left_block[:,:int(np.shape(sorte_left_block)[1]/2)] \
                                - sorte_right_block[:,:int(np.shape(sorte_right_block)[1]/2)]
    sns.heatmap(block_delta_right_block, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,0].set_xlabel('Time from event (ms)')
    ax[3,0].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,0].set_ylabel(r'$\Delta$' + 'LEFT BLOCK -  RIGHT BLOCK')
   
    plt.sca(ax[3,1])
    block_delta_left_block = sorte_left_block[:,int(np.shape(sorte_left_block)[1]/2):] \
                            - sorte_right_block[:,int(np.shape(sorte_right_block)[1]/2):]
    sns.heatmap(block_delta_left_block, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,1].set_xlabel('Time from event (ms)')
    ax[3,1].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,1].set_ylabel('Location')

    plt.sca(ax[3,2])
    delta_delta =  stimulus_delta_left_block - stimulus_delta_right_block
    sns.heatmap(delta_delta, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,2].set_xlabel('Time from event (ms)')
    ax[3,2].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,2].set_title(r'$\Delta$ STIM SIDE (L-R) $\Delta$ BLOCK (L-R)')
    plt.tight_layout()
    
    if extra_ref is not None:
        if '/' in extra_ref: # Changes / from e.g layer 2/3 to avoid error at saving
           l_region = list(extra_ref) 
           l_region[region.find('/')] = '_'
           extra_ref = ''.join(l_region)

           plt.savefig(epoch + '_' + extra_ref +'_heatmap_common_L_2_R_order.png')
           plt.savefig(epoch + '_' + extra_ref +'_heatmap_common_L_2_R_order.svg')
           
           

        



    
def sort_non_opto_plot_by_block_and_subtraction_per_region_classified_by_contrast(
        spikeclusters, spiketimes,session, epoch, cluster_select, clusters_depths,
        bin_size=0.025, extra_ref = None, thirdclass = None, fourthclass = None, 
        smoothing_win = 0):
    
    
    # Select bigger window note pre_time and post_time.
    # Smooths and the cuts edges to standard -200 to +500 window
    matplotlib.rcParams.update({'font.size': 22})
    
    binned_firing_rate = bb.singlecell.calculate_peths(
            spiketimes, spikeclusters, cluster_select, session[epoch],
            bin_size=bin_size,  smoothing = 0, pre_time= round(0.2 + smoothing_win, 3), 
            post_time=round(0.5 + smoothing_win, 3))[1]
    
    # Cluster are added 10000 in every session to identify them when pooling, this
    # retrieves session identity
    cluster_session = np.floor(cluster_select/10000)

    
    # Divide by stimulus side and opto block
    left_stim_trials_neutral_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                        np.where(session['opto_probability_left'] == -1))
    right_stim_trials_neutral_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                        np.where(session['opto_probability_left'] == -1))
    left_stim_trials_left_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                        np.where(session['opto_probability_left'] == 1))
    right_stim_trials_left_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                        np.where(session['opto_probability_left'] == 1))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                        np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                        np.where(session['opto_probability_left'] == 0))
    if thirdclass != None:
        left_stim_trials_neutral_block = np.intersect1d(left_stim_trials_neutral_block, \
                                            np.where(session[thirdclass] == fourthclass))
        right_stim_trials_neutral_block = np.intersect1d(right_stim_trials_neutral_block, \
                                            np.where(session[thirdclass] == fourthclass))
        left_stim_trials_left_block = np.intersect1d(left_stim_trials_left_block, \
                                            np.where(session[thirdclass] == fourthclass))
        right_stim_trials_left_block = np.intersect1d(right_stim_trials_left_block, \
                                            np.where(session[thirdclass] == fourthclass))
        left_stim_trials_right_block = np.intersect1d(left_stim_trials_right_block, \
                                            np.where(session[thirdclass] == fourthclass))
        right_stim_trials_right_block = np.intersect1d(right_stim_trials_right_block, \
                                            np.where(session[thirdclass] == fourthclass))
        
    
    # Divide by contrast
    left_stim_trials_neutral_block_0 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0), \
                       left_stim_trials_neutral_block) 
    right_stim_trials_neutral_block_0 = \
        np.intersect1d(np.where(session['contrastRight'] == 0), \
                       right_stim_trials_neutral_block)
    left_stim_trials_neutral_block_06 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0.0625), \
                       left_stim_trials_neutral_block) 
    right_stim_trials_neutral_block_06 = \
        np.intersect1d(np.where(session['contrastRight'] == 0.0625), \
                       right_stim_trials_neutral_block)
    left_stim_trials_neutral_block_25 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0.25), \
                       left_stim_trials_neutral_block) 
    right_stim_trials_neutral_block_25 = \
        np.intersect1d(np.where(session['contrastRight'] == 0.25), \
                       right_stim_trials_neutral_block)
    
    
    left_stim_trials_left_block_0 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0), \
                       left_stim_trials_left_block) 
    right_stim_trials_left_block_0 = \
        np.intersect1d(np.where(session['contrastRight'] == 0), \
                       right_stim_trials_left_block)
    left_stim_trials_left_block_06 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0.0625), \
                       left_stim_trials_left_block) 
    right_stim_trials_left_block_06 = \
        np.intersect1d(np.where(session['contrastRight'] == 0.0625), \
                       right_stim_trials_left_block)
    left_stim_trials_left_block_25 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0.25), \
                       left_stim_trials_left_block) 
    right_stim_trials_left_block_25 = \
        np.intersect1d(np.where(session['contrastRight'] == 0.25), \
                       right_stim_trials_left_block)
            
    left_stim_trials_right_block_0 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0), \
                       left_stim_trials_right_block) 
    right_stim_trials_right_block_0 = \
        np.intersect1d(np.where(session['contrastRight'] == 0), \
                       right_stim_trials_right_block)
    left_stim_trials_right_block_06 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0.0625), \
                       left_stim_trials_right_block) 
    right_stim_trials_right_block_06 = \
        np.intersect1d(np.where(session['contrastRight'] == 0.0625), \
                       right_stim_trials_right_block)
    left_stim_trials_right_block_25 = \
        np.intersect1d(np.where(session['contrastLeft'] == 0.25), \
                       left_stim_trials_right_block) 
    right_stim_trials_right_block_25 = \
        np.intersect1d(np.where(session['contrastRight'] == 0.25), \
                       right_stim_trials_right_block)

    # Determine groups
    L_0 = binned_firing_rate[left_stim_trials_neutral_block_0,:,:]
    R_0 = binned_firing_rate[right_stim_trials_neutral_block_0,:,:]
    L_blockL_0 = binned_firing_rate[left_stim_trials_left_block_0,:,:]
    R_blockL_0 = binned_firing_rate[right_stim_trials_left_block_0,:,:]
    L_blockR_0 = binned_firing_rate[left_stim_trials_right_block_0,:,:]
    R_blockR_0 = binned_firing_rate[right_stim_trials_right_block_0,:,:]
    
    L_06 = binned_firing_rate[left_stim_trials_neutral_block_06,:,:]
    R_06 = binned_firing_rate[right_stim_trials_neutral_block_06,:,:]
    L_blockL_06 = binned_firing_rate[left_stim_trials_left_block_06,:,:]
    R_blockL_06 = binned_firing_rate[right_stim_trials_left_block_06,:,:]
    L_blockR_06 = binned_firing_rate[left_stim_trials_right_block_06,:,:]
    R_blockR_06 = binned_firing_rate[right_stim_trials_right_block_06,:,:]
    
    L_25 = binned_firing_rate[left_stim_trials_neutral_block_25,:,:]
    R_25 = binned_firing_rate[right_stim_trials_neutral_block_25,:,:]
    L_blockL_25 = binned_firing_rate[left_stim_trials_left_block_25,:,:]
    R_blockL_25 = binned_firing_rate[right_stim_trials_left_block_25,:,:]
    L_blockR_25 = binned_firing_rate[left_stim_trials_right_block_25,:,:]
    R_blockR_25 = binned_firing_rate[right_stim_trials_right_block_25,:,:]

    # Obtain firing rates
    N_L_0 = mean_binned_firing_rate(L_0, left_stim_trials_neutral_block_0, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    N_R_0 = mean_binned_firing_rate(R_0, right_stim_trials_neutral_block_0, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    L_L_0 = mean_binned_firing_rate(L_blockL_0, left_stim_trials_left_block_0, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    L_R_0 = mean_binned_firing_rate(R_blockL_0, right_stim_trials_left_block_0, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    R_L_0 = mean_binned_firing_rate(L_blockR_0, left_stim_trials_right_block_0, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    R_R_0 = mean_binned_firing_rate(R_blockR_0, right_stim_trials_right_block_0, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    
    N_L_06 = mean_binned_firing_rate(L_06, left_stim_trials_neutral_block_06, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    N_R_06 = mean_binned_firing_rate(R_06, right_stim_trials_neutral_block_06, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    L_L_06 = mean_binned_firing_rate(L_blockL_06, left_stim_trials_left_block_06, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    L_R_06 = mean_binned_firing_rate(R_blockL_06, right_stim_trials_left_block_06, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    R_L_06 = mean_binned_firing_rate(L_blockR_06, left_stim_trials_right_block_06, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    R_R_06 = mean_binned_firing_rate(R_blockR_06, right_stim_trials_right_block_06, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    
    N_L_25 = mean_binned_firing_rate(L_25, left_stim_trials_neutral_block_25, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    N_R_25 = mean_binned_firing_rate(R_25, right_stim_trials_neutral_block_25, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    L_L_25 = mean_binned_firing_rate(L_blockL_25, left_stim_trials_left_block_25, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    L_R_25 = mean_binned_firing_rate(R_blockL_25, right_stim_trials_left_block_25, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    R_L_25 = mean_binned_firing_rate(L_blockR_25, left_stim_trials_right_block_25, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    R_R_25 = mean_binned_firing_rate(R_blockR_25, right_stim_trials_right_block_25, 
                                  session, cluster_session, order = None , 
                                  bin_size = 0.025, smoothing = smoothing_win)
    
    # Concatenate
    
    concatenated_mean_firing_rates = np.concatenate((N_L_25, N_R_25, L_L_25, L_R_25, 
                                                     R_L_25, R_R_25, N_L_06, N_R_06, 
                                                     L_L_06, L_R_06, R_L_06, R_R_06,
                                                     N_L_0, N_R_0, L_L_0, L_R_0, 
                                                     R_L_0, R_R_0), axis = 1)
    
        
    # Sort based by neutral block 25
    
    sorte, order = z_score_and_order(concatenated_mean_firing_rates, (0,56),
                                     order = None )
    
    # Apply sort
    
    sorte_left_block_25 = \
        z_score_and_order(concatenated_mean_firing_rates, (56,112),
                                     order = order )
        
    sorte_right_block_25 = \
        z_score_and_order(concatenated_mean_firing_rates, (112,168),
                                     order = order )
        
    sorte_neutral_block_06 = \
        z_score_and_order(concatenated_mean_firing_rates, (168,224),
                                     order = order )
    
    sorte_left_block_06 = \
        z_score_and_order(concatenated_mean_firing_rates, (224,280),
                                     order = order )
        
    sorte_right_block_06 = \
        z_score_and_order(concatenated_mean_firing_rates, (280,336),
                                     order = order )
    
    sorte_neutral_block_0 = \
        z_score_and_order(concatenated_mean_firing_rates, (336,392),
                                     order = order )
    
    sorte_left_block_0 = \
        z_score_and_order(concatenated_mean_firing_rates, (392,448),
                                     order = order )
        
    sorte_right_block_0 = \
        z_score_and_order(concatenated_mean_firing_rates, (448,504),
                                     order = order )
    
    # Plot
    
    fig, ax  = plt.subplots(4,9, figsize=(75,30), sharex=True)
    plt.sca(ax[0,0])
    sns.heatmap(sorte[:,:int(np.shape(sorte)[1]/2)], vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,0].set_xlabel('Time from event (ms)')
    ax[0,0].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[0,0].set_ylabel('NEUTRAL BLOCK', rotation='vertical',x=-0.1,y=0.5)
    ax[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if epoch=='goCue_times':
        ax[0,0].set_title("LEFT STIM TRIALS")
    elif epoch=='feedback_times':
        ax[0,0].set_title("LEFT CHOICE TRIALS")
    
    plt.sca(ax[0,1])
    sns.heatmap(sorte[:,int(np.shape(sorte)[1]/2):], vmin=-5, vmax=5, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,1].set_xlabel('Time from event (ms)')
    ax[0,1].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    if epoch=='goCue_times':
        ax[0,1].set_title("RIGHT STIM TRIALS")
    elif epoch=='feedback_times':
        ax[0,1].set_title("RIGHT CHOICE TRIALS")
    
    plt.sca(ax[0,2])
    sns.heatmap((sorte[:,:int(np.shape(sorte)[1]/2)] - sorte[:,int(np.shape(sorte)[1]/2):]), 
                vmin=-3, vmax=7, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,2].set_xlabel('Time from event (ms)')
    ax[0,2].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[0,2].set_title(r"$\Delta$ LEFT TRIALS - RIGHT TRIALS")
    
    
    plt.sca(ax[1,0])
    sns.heatmap(sorte_left_block_25[:,:int(np.shape(sorte_left_block_25)[1]/2)], vmin=-5, center=0,  cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,0].set_xlabel('Time from event (ms)')
    ax[1,0].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[1,0].set_ylabel('LEFT BLOCK')
    
    plt.sca(ax[1,1])
    sns.heatmap(sorte_left_block_25[:,int(np.shape(sorte_left_block_25)[1]/2):], vmin=-5,  center=0, cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,1].set_xlabel('Time from event (ms)')
    ax[1,1].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[1,2])
    stimulus_delta_left_block_25 =  sorte_left_block_25[:,:int(np.shape(sorte_left_block_25)[1]/2)] \
                                - sorte_left_block_25[:,int(np.shape(sorte_left_block_25)[1]/2):]
    sns.heatmap(stimulus_delta_left_block_25, vmin=-5, center=0,  cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,2].set_xlabel('Time from event (ms)')
    ax[1,2].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[2,0])
    sns.heatmap(sorte_right_block_25[:,:int(np.shape(sorte_right_block_25)[1]/2)], center=0,  cmap="bwr",
                vmin=-5, vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,0].set_xlabel('Time from event (ms)')
    ax[2,0].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[2,0].set_ylabel('RIGHT BLOCK')
    
    plt.sca(ax[2,1])
    sns.heatmap(sorte_right_block_25[:,int(np.shape(sorte_right_block_25)[1]/2):], center=0,  cmap="bwr",
                vmin=-5, vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,1].set_xlabel('Time from event (ms)')
    ax[2,1].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[2,2])
    stimulus_delta_right_block_25 =  sorte_right_block_25[:,:int(np.shape(sorte_right_block_25)[1]/2)] \
        - sorte_right_block_25[:,int(np.shape(sorte_right_block_25)[1]/2):]
    
    sns.heatmap(stimulus_delta_right_block_25, vmin=-5, center=0, cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,2].set_xlabel('Time from event (ms)')
    ax[2,2].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[3,0])
    block_delta_right_block_25 =  sorte_left_block_25[:,:int(np.shape(sorte_left_block_25)[1]/2)] \
                                - sorte_right_block_25[:,:int(np.shape(sorte_right_block_25)[1]/2)]
    sns.heatmap(block_delta_right_block_25, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,0].set_xlabel('Time from event (ms)')
    ax[3,0].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,0].set_ylabel(r'$\Delta$' + 'LEFT BLOCK -  RIGHT BLOCK')
   
    plt.sca(ax[3,1])
    block_delta_left_block_25 = sorte_left_block_25[:,int(np.shape(sorte_left_block_25)[1]/2):] \
                            - sorte_right_block_25[:,int(np.shape(sorte_right_block_25)[1]/2):]
    sns.heatmap(block_delta_left_block_25, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,1].set_xlabel('Time from event (ms)')
    ax[3,1].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,1].set_ylabel('Location')

    plt.sca(ax[3,2])
    delta_delta_25 =  stimulus_delta_left_block_25 - stimulus_delta_right_block_25
    sns.heatmap(delta_delta_25, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,2].set_xlabel('Time from event (ms)')
    ax[3,2].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,2].set_title(r'$\Delta$ STIM SIDE (L-R) $\Delta$ BLOCK (L-R)')
    plt.tight_layout()
    
    #2/3 - 06
    
    plt.sca(ax[0,3])
    sns.heatmap(sorte_neutral_block_06[:,:int(np.shape(sorte_neutral_block_06)[1]/2)], vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,3].set_xlabel('Time from event (ms)')
    ax[0,3].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[0,3].set_ylabel('NEUTRAL BLOCK', rotation='vertical',x=-0.1,y=0.5)
    ax[0,3].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if epoch=='goCue_times':
        ax[0,3].set_title("LEFT STIM TRIALS")
    elif epoch=='feedback_times':
        ax[0,3].set_title("LEFT CHOICE TRIALS")
    
    plt.sca(ax[0,4])
    sns.heatmap(sorte_neutral_block_06[:,int(np.shape(sorte_neutral_block_06)[1]/2):], vmin=-5, vmax=5, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,5)
    ax[0,4].set_xlabel('Time from event (ms)')
    ax[0,4].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    if epoch=='goCue_times':
        ax[0,4].set_title("RIGHT STIM TRIALS")
    elif epoch=='feedback_times':
        ax[0,4].set_title("RIGHT CHOICE TRIALS")
    
    plt.sca(ax[0,5])
    sns.heatmap((sorte_neutral_block_06[:,:int(np.shape(sorte_neutral_block_06)[1]/2)] 
                 - sorte_neutral_block_06[:,int(np.shape(sorte_neutral_block_06)[1]/2):]), 
                vmin=-3, vmax=7, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,5].set_xlabel('Time from event (ms)')
    ax[0,5].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[0,5].set_title(r"$\Delta$ LEFT TRIALS - RIGHT TRIALS")
    
    
    plt.sca(ax[1,3])
    sns.heatmap(sorte_left_block_06[:,:int(np.shape(sorte_left_block_06)[1]/2)], vmin=-5, center=0,  cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,3].set_xlabel('Time from event (ms)')
    ax[1,3].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[1,3].set_ylabel('LEFT BLOCK')
    
    plt.sca(ax[1,4])
    sns.heatmap(sorte_left_block_06[:,int(np.shape(sorte_left_block_06)[1]/2):], vmin=-5,  center=0, cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,4].set_xlabel('Time from event (ms)')
    ax[1,4].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[1,5])
    stimulus_delta_left_block_06 =  sorte_left_block_06[:,:int(np.shape(sorte_left_block_06)[1]/2)] \
                                - sorte_left_block_06[:,int(np.shape(sorte_left_block_06)[1]/2):]
    sns.heatmap(stimulus_delta_left_block_06, vmin=-5, center=0,  cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,5].set_xlabel('Time from event (ms)')
    ax[1,5].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[2,3])
    sns.heatmap(sorte_right_block_06[:,:int(np.shape(sorte_right_block_06)[1]/2)], center=0,  cmap="bwr",
                vmin=-5, vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,3].set_xlabel('Time from event (ms)')
    ax[2,3].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[2,3].set_ylabel('RIGHT BLOCK')
    
    plt.sca(ax[2,4])
    sns.heatmap(sorte_right_block_06[:,int(np.shape(sorte_right_block_06)[1]/2):], center=0,  cmap="bwr",
                vmin=-5, vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,4].set_xlabel('Time from event (ms)')
    ax[2,4].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[2,5])
    stimulus_delta_right_block_06 =  sorte_right_block_06[:,:int(np.shape(sorte_right_block_06)[1]/2)] \
        - sorte_right_block_06[:,int(np.shape(sorte_right_block_06)[1]/2):]
    
    sns.heatmap(stimulus_delta_right_block_06, vmin=-5, center=0, cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,5].set_xlabel('Time from event (ms)')
    ax[2,5].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[3,3])
    block_delta_right_block_06 =  sorte_left_block_06[:,:int(np.shape(sorte_left_block_06)[1]/2)] \
                                - sorte_right_block_06[:,:int(np.shape(sorte_right_block_06)[1]/2)]
    sns.heatmap(block_delta_right_block_06, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,3].set_xlabel('Time from event (ms)')
    ax[3,3].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,3].set_ylabel(r'$\Delta$' + 'LEFT BLOCK -  RIGHT BLOCK')
   
    plt.sca(ax[3,4])
    block_delta_left_block_06 = sorte_left_block_06[:,int(np.shape(sorte_left_block_06)[1]/2):] \
                            - sorte_right_block_06[:,int(np.shape(sorte_right_block_06)[1]/2):]
    sns.heatmap(block_delta_left_block_06, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,4].set_xlabel('Time from event (ms)')
    ax[3,4].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,4].set_ylabel('Location')

    plt.sca(ax[3,5])
    delta_delta =  stimulus_delta_left_block_06 - stimulus_delta_right_block_06
    sns.heatmap(delta_delta, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,5].set_xlabel('Time from event (ms)')
    ax[3,5].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,5].set_title(r'$\Delta$ STIM SIDE (L-R) $\Delta$ BLOCK (L-R)')
    plt.tight_layout()
    
    #3/3
    
    plt.sca(ax[0,6])
    sns.heatmap(sorte_neutral_block_0[:,:int(np.shape(sorte_neutral_block_0)[1]/2)], vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,6].set_xlabel('Time from event (ms)')
    ax[0,6].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[0,6].set_ylabel('NEUTRAL BLOCK', rotation='vertical',x=-0.1,y=0.5)
    ax[0,6].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    if epoch=='goCue_times':
        ax[0,6].set_title("LEFT STIM TRIALS")
    elif epoch=='feedback_times':
        ax[0,6].set_title("LEFT CHOICE TRIALS")
    
    plt.sca(ax[0,7])
    sns.heatmap(sorte_neutral_block_0[:,int(np.shape(sorte_neutral_block_0)[1]/2):], vmin=-5, vmax=5, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,7].set_xlabel('Time from event (ms)')
    ax[0,7].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    if epoch=='goCue_times':
        ax[0,7].set_title("RIGHT STIM TRIALS")
    elif epoch=='feedback_times':
        ax[0,7].set_title("RIGHT CHOICE TRIALS")
    
    plt.sca(ax[0,8])
    sns.heatmap((sorte_neutral_block_0[:,:int(np.shape(sorte_neutral_block_0)[1]/2)] - \
                 sorte_neutral_block_0[:,int(np.shape(sorte_neutral_block_0)[1]/2):]), 
                vmin=-3, vmax=7, center=0, cmap="bwr",
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[0,8].set_xlabel('Time from event (ms)')
    ax[0,8].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[0,8].set_title(r"$\Delta$ LEFT TRIALS - RIGHT TRIALS")
    
    
    plt.sca(ax[1,6])
    sns.heatmap(sorte_left_block_0[:,:int(np.shape(sorte_left_block_0)[1]/2)], vmin=-5, center=0,  cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,6].set_xlabel('Time from event (ms)')
    ax[1,6].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[1,6].set_ylabel('LEFT BLOCK')

    plt.sca(ax[1,7])
    sns.heatmap(sorte_left_block_0[:,int(np.shape(sorte_left_block_0)[1]/2):], vmin=-5,  center=0, cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,7].set_xlabel('Time from event (ms)')
    ax[1,7].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[1,8])
    stimulus_delta_left_block_0 =  sorte_left_block_0[:,:int(np.shape(sorte_left_block_0)[1]/2)] \
                                - sorte_left_block_0[:,int(np.shape(sorte_left_block_0)[1]/2):]
    sns.heatmap(stimulus_delta_left_block_0, vmin=-5, center=0,  cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[1,8].set_xlabel('Time from event (ms)')
    ax[1,8].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[2,6])
    sns.heatmap(sorte_right_block_0[:,:int(np.shape(sorte_right_block_0)[1]/2)], center=0,  cmap="bwr",
                vmin=-5, vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,6].set_xlabel('Time from event (ms)')
    ax[2,6].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[2,6].set_ylabel('RIGHT BLOCK')
    
    plt.sca(ax[2,7])
    sns.heatmap(sorte_right_block_0[:,int(np.shape(sorte_right_block_0)[1]/2):], center=0,  cmap="bwr",
                vmin=-5, vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,7].set_xlabel('Time from event (ms)')
    ax[2,7].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[2,8])
    stimulus_delta_right_block_0 =  sorte_right_block_0[:,:int(np.shape(sorte_right_block_0)[1]/2)] \
        - sorte_right_block_0[:,int(np.shape(sorte_right_block_0)[1]/2):]
    
    sns.heatmap(stimulus_delta_right_block_0, vmin=-5, center=0, cmap="bwr",
                vmax=5, cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[2,8].set_xlabel('Time from event (ms)')
    ax[2,8].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    
    plt.sca(ax[3,6])
    block_delta_right_block_0 =  sorte_left_block_0[:,:int(np.shape(sorte_left_block_0)[1]/2)] \
                                - sorte_right_block_0[:,:int(np.shape(sorte_right_block_0)[1]/2)]
    sns.heatmap(block_delta_right_block_0, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,6].set_xlabel('Time from event (ms)')
    ax[3,6].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,6].set_ylabel(r'$\Delta$' + 'LEFT BLOCK -  RIGHT BLOCK')
   
    plt.sca(ax[3,7])
    block_delta_left_block_0 = sorte_left_block_0[:,int(np.shape(sorte_left_block_0)[1]/2):] \
                            - sorte_right_block_0[:,int(np.shape(sorte_right_block_0)[1]/2):]
    sns.heatmap(block_delta_left_block_0, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,7].set_xlabel('Time from event (ms)')
    ax[3,7].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,7].set_ylabel('Location')

    plt.sca(ax[3,8])
    delta_delta_0 =  stimulus_delta_left_block_0 - stimulus_delta_right_block_0
    sns.heatmap(delta_delta_0, vmin=-5, vmax=5, center=0, cmap="bwr", 
                cbar_kws={'label': ' mean  z-scored firing rate (Hz)'})
    plt.axvline(8, 0,1)
    ax[3,8].set_xlabel('Time from event (ms)')
    ax[3,8].set_xticklabels(np.arange(-200,500, bin_size*1000*3)
        , rotation='vertical')
    ax[3,8].set_title(r'$\Delta$ STIM SIDE (L-R) $\Delta$ BLOCK (L-R)')
    plt.tight_layout()
    
    
    if extra_ref is not None:
        if '/' in extra_ref: # Changes / from e.g layer 2/3 to avoid error at saving
           l_region = list(extra_ref) 
           l_region[region.find('/')] = '_'
           extra_ref = ''.join(l_region)

           plt.savefig(epoch + '_' + extra_ref +'_heatmap_common_L_2_R_order.png')
           plt.savefig(epoch + '_' + extra_ref +'_heatmap_common_L_2_R_order.svg')