#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:26:44 2020

@author: alex
"""

import numpy as np
import pandas as pd
import npy2pd
import brainbox as bb
from brainbox.plot import *
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os
from os import path
from alf_functions import *
import sys
from path import Path
import random
import seaborn as sns
### Developing test ###
session_folder = '/Volumes/LaCie/dop_4_ephys_data/2020-01-15/001'
session = load_behavior(session_folder)
clusters_depths, cluster_metrics, \
spikeclusters, spiketimes, \
good_clusters = load_neural_data(session_folder)

###

# Loading data

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
    session['opto.npy'][:] = 0
    session['real_opto'] = np.load(session_folder + '/alf/_ibl_trials.laser_epoch_in.npy')
    session['opto.npy'] = (~np.isnan(session['real_opto']) == 1)*1
    ##
    session['response_times'] = session['response_times'][:727]
    session = pd.DataFrame.from_dict(session)
    session = add_trial_within_block(session)
    session['after_opto'] = session['opto.npy'].shift(periods=1)
    session['after_reward'] = session['feedbackType'].shift(periods=1)
    return session

def load_neural_data(session_folder):
    '''
    Parameters
    ----------
    session_folder : folder of ephy session (str)

    Returns
    -------
    arrays with cluster measures
    '''
    clusters_depths = np.load(session_folder + '/alf/probe00/clusters.depths.npy')
    cluster_metrics = pd.read_csv(session_folder + '/alf/probe00/clusters.metrics.csv')
    spikeclusters = np.load(session_folder + '/alf/probe00/spikes.clusters.npy')
    location = np.load()
    spiketimes = np.load(session_folder + '/alf/probe00/spikes.times.npy')
    good_clusters = cluster_metrics.loc[cluster_metrics['ks2_label']=='good', 'cluster_id']
    allen = np.load('/Users/alex/Downloads/hist.regions_dop04_15.npy', allow_pickle=True) #Not stable
    return clusters_depths, cluster_metrics, spikeclusters, spiketimes, good_clusters, allen




# Run plots
heatmap_sorted_by_pool_stimulus(spikeclusters, spiketimes,session, i, epoch,
                           bin_size=0.025)

heatmap_sorted_by_pool_choice(spikeclusters, spiketimes,session, i, epoch,
                           bin_size=0.025)
heatmap_per_session_stimulus(spikeclusters, spiketimes,session, i, bin_size=0.025)
heatmap_per_session_choice(spikeclusters, spiketimes,session, i, bin_size=0.025)
# Plot heatmaps

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
                                    np.where(session['opto_probability_left'] == 0))
    left_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 0))
    right_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == -1), \
                                    np.where(session['opto_probability_left'] == 1))
    
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
                                    np.where(session['opto_probability_left'] == 0))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == 1))
    
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
    


def heatmap_per_session_stimulus(spikeclusters, spiketimes,session, i, bin_size=0.025, epoch):

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
                                    np.where(session['opto_probability_left'] == 0))
    left_stim_trials_right_block = np.intersect1d(np.where(session['contrastLeft'] >= 0), \
                                    np.where(session['opto_probability_left'] == 0))
    right_stim_trials_right_block = np.intersect1d(np.where(session['contrastRight'] >= 0), \
                                    np.where(session['opto_probability_left'] == 1))
    
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
    
    
    plt.savefig('choice_heatmap.png')
    plt.savefig('choice_heatmap.svg')
    
def heatmap_per_session_choice(spikeclusters, spiketimes,session, i, bin_size=0.025, epoch):

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
                                    np.where(session['opto_probability_left'] == 0))
    left_choice_trials_right_block = np.intersect1d(np.where(session['choice'] == 1), \
                                    np.where(session['opto_probability_left'] == 0))
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

    plt.savefig('stimulus_heatmap.png')
    plt.savefig('stimulus_heatmap.svg')
    


def heatmap_cross_validated(binned_firing_rate):
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
    
    z_score_binned_spikes = \
        ((mean_binned_fr.T-mean_binned_fr.mean(axis =1))\
        /mean_binned_fr.std(axis = 1)).T
    
    # Replace np.nan by 0
    z_score_binned_spikes = np.nan_to_num(z_score_binned_spikes)
    
    order = np.argmax(z_score_binned_spikes, 1)
    
    # make heatmap matrix
    mean_binned_fr_test =  np.mean(test_set/bin_size, axis =0)
    z_score_binned_spikes_test = \
        ((mean_binned_fr_test.T-mean_binned_fr_test.mean(axis =1))\
        /mean_binned_fr_test.std(axis = 1)).T
    z_score_binned_spikes_test = np.nan_to_num(z_score_binned_spikes_test)
        
    sorte = z_score_binned_spikes_test[order.argsort()]
    
    # map new order to cluster ids base on sum of bins in row to be extra sure
    ordered_ids = order.argsort()                     
    
    return sorte, ordered_ids


def heatmap_w_external_sorting(binned_firing_rate, order = None ):
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
    mean_binned_fr_test =  np.mean(binned_firing_rate/bin_size, axis =0)
    z_score_binned_spikes_test = \
        ((mean_binned_fr_test.T-mean_binned_fr_test.mean(axis =1))\
        /mean_binned_fr_test.std(axis = 1)).T
    z_score_binned_spikes_test = np.nan_to_num(z_score_binned_spikes_test)
    
    if order is not None:
        sorte = z_score_binned_spikes_test[order]
        return sorte
    else:
        order = np.argmax(z_score_binned_spikes, 1)
        sorte = z_score_binned_spikes_test[order.argsort()]
    
        return sorte, order.argsort()
    

    



# Ploting PSTH

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
            
    except:
        print('error in feedback after opto figure')


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
        
if __name__ == '__main__':
     session_folder = sys.argv[0]
     session = load_behavior(session_folder)
     clusters_depths, cluster_metrics, \
     spikeclusters, spiketimes, \
     good_clusters = load_neural_data(session_folder)
     session_single_neuron_psth_summary(spikeclusters, spiketimes,session, session_folder)
    
     
   

    
    