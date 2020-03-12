#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:26:44 2020

@author: alex
"""

import numpy as np
import pandas as pd
import npy2pd
from brainbox.plot import *
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import os
from os import path

session_folder = '/Volumes/LaCie/dop_4_ephys_data/2020-01-15/001'
    
def load_behavior(session_folder):
    variables = npy2pd.pybpod_vars()
    session = npy2pd.session_loader(session_folder, variables)
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
    

def add_trial_within_block(session):
    '''
    session: dataframe with behavioral data
    '''
    session['trial_within_block'] = np.nan
    block_breaks = np.diff(session['opto_probability_left'])
    block_breaks = np.where(block_breaks != 0)
    for i, t in enumerate(block_breaks[0]):
        if i == 0:
            for l in range(t+1):
                session.iloc[l, session.columns.get_loc('trial_within_block')] = l
        else:
            for x, l in enumerate(range(block_breaks[0][i-1]+1,t+1)):
                session.iloc[l, session.columns.get_loc('trial_within_block')] = x
    return session
                
                

def load_neural_data(session_folder):
    clusters_depths = np.load(session_folder + '/alf/probe00/clusters.depths.npy')
    cluster_metrics = pd.read_csv(session_folder + '/alf/probe00/clusters.metrics.csv')
    spikeclusters = np.load(session_folder + '/alf/probe00/spikes.clusters.npy')
    spiketimes = np.load(session_folder + '/alf/probe00/spikes.times.npy')
    good_clusters = cluster_metrics.loc[cluster_metrics['ks2_label']=='good', 'cluster_id']

    


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