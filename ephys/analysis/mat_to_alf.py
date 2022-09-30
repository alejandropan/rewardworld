from scipy.io import loadmat
import pathlib
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import sem
import copy
from pathlib import Path
import time
import seaborn as sns
# Function
def calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster):
    right_is_ipsi = encoding_df_cluster.hemisphere   
    if right_is_ipsi==1:
        ses_of_interest_behavior.Qipsi = ses_of_interest_behavior.QR
        ses_of_interest_behavior.QLaseripsi = ses_of_interest_behavior.QRlaser
        ses_of_interest_behavior.QWateripsi = ses_of_interest_behavior.QRreward
        ses_of_interest_behavior.QStayipsi = ses_of_interest_behavior.QRstay
        ses_of_interest_behavior.Qcontra = ses_of_interest_behavior.QL
        ses_of_interest_behavior.QLasercontra = ses_of_interest_behavior.QLlaser
        ses_of_interest_behavior.QWatercontra = ses_of_interest_behavior.QLreward
        ses_of_interest_behavior.QStaycontra = ses_of_interest_behavior.QLstay      
    if right_is_ipsi==0:
        ses_of_interest_behavior.Qipsi = ses_of_interest_behavior.QL
        ses_of_interest_behavior.QLaseripsi = ses_of_interest_behavior.QLlaser
        ses_of_interest_behavior.QWateripsi = ses_of_interest_behavior.QLreward
        ses_of_interest_behavior.QStayipsi = ses_of_interest_behavior.QLstay
        ses_of_interest_behavior.Qcontra = ses_of_interest_behavior.QR
        ses_of_interest_behavior.QLasercontra = ses_of_interest_behavior.QRlaser
        ses_of_interest_behavior.QWatercontra = ses_of_interest_behavior.QRreward
        ses_of_interest_behavior.QStaycontra = ses_of_interest_behavior.QRstay
    ses_of_interest_behavior.Qdelta = (ses_of_interest_behavior.Qcontra - ses_of_interest_behavior.Qipsi)*100 # To avoid numpy precision problem
    ses_of_interest_behavior.QLaserdelta = (ses_of_interest_behavior.QLasercontra - ses_of_interest_behavior.QLaseripsi)*100 # To avoid numpy precision problem
    ses_of_interest_behavior.QWaterdelta = (ses_of_interest_behavior.QWatercontra - ses_of_interest_behavior.QWateripsi)*100 # To avoid numpy precision problem
    ses_of_interest_behavior.QStaydelta = (ses_of_interest_behavior.QStaycontra - ses_of_interest_behavior.QStayipsi)*100 # To avoid numpy precision problem
    return ses_of_interest_behavior

def calculate_qchosen(ses_of_interest_behavior):
    ses_of_interest_behavior.qchosen = ses_of_interest_behavior.QR
    ses_of_interest_behavior.qchosen[np.where(ses_of_interest_behavior.choice==-1)] = \
        ses_of_interest_behavior.QL[np.where(ses_of_interest_behavior.choice==-1)]
    ses_of_interest_behavior.qchosen_w = ses_of_interest_behavior.QRreward
    ses_of_interest_behavior.qchosen_w[np.where(ses_of_interest_behavior.choice==-1)] = \
        ses_of_interest_behavior.QLreward[np.where(ses_of_interest_behavior.choice==-1)]
    ses_of_interest_behavior.qchosen_l = ses_of_interest_behavior.QRlaser
    ses_of_interest_behavior.qchosen_l[np.where(ses_of_interest_behavior.choice==-1)] = \
        ses_of_interest_behavior.QLlaser[np.where(ses_of_interest_behavior.choice==-1)]
    ses_of_interest_behavior.qchosen_s = ses_of_interest_behavior.QRstay
    ses_of_interest_behavior.qchosen_s[np.where(ses_of_interest_behavior.choice==-1)] = \
        ses_of_interest_behavior.QLstay[np.where(ses_of_interest_behavior.choice==-1)]
    return ses_of_interest_behavior

def calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster):
    right_is_ipsi = encoding_df_cluster.hemisphere   
    ses_of_interest_behavior.choice_ipsi = np.zeros(len(ses_of_interest_behavior.choice))
    ses_of_interest_behavior.choice_contra = np.zeros(len(ses_of_interest_behavior.choice))
    if right_is_ipsi==1:
        ses_of_interest_behavior.choice_ipsi = 1*(ses_of_interest_behavior.choice==1)
        ses_of_interest_behavior.choice_contra = 1*(ses_of_interest_behavior.choice==-1)
    if right_is_ipsi==0:
        ses_of_interest_behavior.choice_ipsi = 1*(ses_of_interest_behavior.choice==-1)
        ses_of_interest_behavior.choice_contra = 1*(ses_of_interest_behavior.choice==1)
    return ses_of_interest_behavior

def get_peth_from_struc(peth_struc):
    n_trial = peth_struc.shape[0]
    t_lens = []
    for i in np.arange(n_trial):
        t_lens.append(peth_struc[i].shape[1])
    max_len = np.max(t_lens)
    peth = np.zeros([n_trial,max_len])
    peth[:] = np.nan
    for i in np.arange(n_trial):
        if peth_struc[i][0].size>0:
            peth[i,:peth_struc[i][0].size] = peth_struc[i][0]
    return peth

def psth_plot(encoding_df_cluster, trial_selection, centering_list, alpha=1, pre_time = 500, 
            post_time = 500, color='dodgerblue', bin_size=5):
    centering_list = centering_list[trial_selection]
    trial_selection = trial_selection[np.where(~np.isnan(centering_list))[0].tolist()]
    centering_list = centering_list[np.where(~np.isnan(centering_list))[0].tolist()]
    pre_time=pre_time/bin_size
    post_time=post_time/bin_size
    peth_selection = np.array(encoding_df_cluster.peth[trial_selection])

    selection = []
    for i, c in enumerate(centering_list):
        selection.append(peth_selection[i,int(c-pre_time):int(c+post_time)])
    selection = np.vstack(selection)
    y = np.nanmean(selection, axis=0)
    yerr = sem(selection, axis=0, nan_policy='omit')
    x = np.arange(-pre_time*bin_size,post_time*bin_size,bin_size)
    plt.plot(x,y, alpha=alpha/2, color=color)
    plt.fill_between(x, y-yerr, y+yerr, alpha=alpha/2 ,color=color)
    plt.vlines(0,(y-yerr).min(),(y+yerr).max(), linestyles='dashed', color='k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Normalized Firing Rate')

def kernel_plot(model, trial_selection, centering_list, alpha=1, pre_time = 500, 
            post_time = 500, color='k', bin_size=5):
    centering_list = centering_list[trial_selection]
    trial_selection = trial_selection[np.where(~np.isnan(centering_list))[0].tolist()]
    centering_list = centering_list[np.where(~np.isnan(centering_list))[0].tolist()]
    pre_time=pre_time/bin_size
    post_time=post_time/bin_size
    peth_selection = np.array(model[trial_selection])
    selection = []
    for i, c in enumerate(centering_list):
        selection.append(peth_selection[i,int(c-pre_time):int(c+post_time)])
    selection = np.vstack(selection)
    y = np.nanmean(selection, axis=0)
    x = np.arange(-pre_time*bin_size,post_time*bin_size,bin_size)
    plt.plot(x,y, alpha=alpha/2, color=color, linestyle='dashed')

def plot_qchosen_summary(encoding_df_cluster):
    fig, ax = plt.subplots(3,5,figsize=(15,10))
    fig.suptitle('alpha=Qchosen' +' '+ 
                ' all_p '+str(round(encoding_df_cluster.p_values.value,3))
                )
    # Water at Cue
    plt.sca(ax[0,0])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.qchosen_w[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_w)),water)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_w<(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_w>=np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333))&
        (ses_of_interest_behavior.qchosen_w<=(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_w>(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='dodgerblue')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='k')
    plt.title('Water at cue' + '\nwater_p '+str(round(encoding_df_cluster.p_values.value_water,3)) + 
             ' c_water_g '+str(round(encoding_df_cluster.gains.cue.value_water,3)))
    plt.xlabel('Time from cue(ms)')

    # Laser at Cue
    plt.sca(ax[1,0])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.qchosen_l[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_l)),laser)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_l<(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_l>=np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333))&
        (ses_of_interest_behavior.qchosen_l<=(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_l>(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='orange')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='k')
    plt.title('Laser at cue' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.value_laser,3)) + 
             ' c_laser_g '+str(round(encoding_df_cluster.gains.cue.value_laser,3)))    
    plt.xlabel('Time from cue(ms)')

    # Stay at Cue
    plt.sca(ax[2,0])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
 
    low_value = np.where(ses_of_interest_behavior.qchosen_s<(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_s>=np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333))&
        (ses_of_interest_behavior.qchosen_s<=(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_s>(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='gray')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='gray')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='gray')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='k')
    plt.title('Stay at cue' + '\nstay_p '+str(round(encoding_df_cluster.p_values.value_stay,3)) + 
             ' c_stay_g '+str(round(encoding_df_cluster.gains.cue.value_stay,3)))     
    plt.xlabel('Time from cue(ms)')

    # Water at ipsi choice
    plt.sca(ax[0,1])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    ipsi = np.where((1*(ses_of_interest_behavior.choice>0))==encoding_df_cluster.hemisphere)[0]
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = np.intersect1d(ipsi,water)
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.qchosen_w[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_w)),water)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_w<(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_w>=np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333))&
        (ses_of_interest_behavior.qchosen_w<=(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_w>(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='dodgerblue')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Water at ipsi choice' + '\nwater_p '+str(round(encoding_df_cluster.p_values.value_water,3)) + 
             ' mc_water_g '+str(round(encoding_df_cluster.gains.movement_contra.value_water,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Laser at ipsi choice
    plt.sca(ax[1,1])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    ipsi = np.where((1*(ses_of_interest_behavior.choice>0))==encoding_df_cluster.hemisphere)[0]
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = np.intersect1d(ipsi,laser)    
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.qchosen_l[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_l)),laser)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_l<(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_l>=np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333))&
        (ses_of_interest_behavior.qchosen_l<=(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_l>(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='orange')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Laser at ipsi choice' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.value_laser,3)) + 
             ' mc_laser_g '+str(round(encoding_df_cluster.gains.movement_contra.value_laser,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Stay at ipsi choice
    plt.sca(ax[2,1])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    ipsi = np.where((1*(ses_of_interest_behavior.choice>0))==encoding_df_cluster.hemisphere)[0]
    ses_of_interest_behavior.qchosen_s[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_s)),ipsi)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_s<(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_s>=np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333))&
        (ses_of_interest_behavior.qchosen_s<=(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_s>(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='gray')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='gray')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='gray')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Stay at ipsi choice' + '\nstay_p '+str(round(encoding_df_cluster.p_values.value_stay,3)) + 
             ' mc_stay_g '+str(round(encoding_df_cluster.gains.movement_contra.value_stay,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Water at contra choice
    plt.sca(ax[0,2])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    contra = np.where((1*(ses_of_interest_behavior.choice>0))!=encoding_df_cluster.hemisphere)[0]
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = np.intersect1d(contra,water)    
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.qchosen_w[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_w)),water)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_w<(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_w>=np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333))&
        (ses_of_interest_behavior.qchosen_w<=(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_w>(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='dodgerblue')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Water at contra choice' + '\nwater_p '+str(round(encoding_df_cluster.p_values.value_water,3)) + 
             ' mi_water_g '+str(round(encoding_df_cluster.gains.movement_ipsi.value_water,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Laser at contra choice
    plt.sca(ax[1,2])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    contra = np.where((1*(ses_of_interest_behavior.choice>0))!=encoding_df_cluster.hemisphere)[0]
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = np.intersect1d(contra,laser)       
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.qchosen_l[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_l)),laser)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_l<(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_l>=np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333))&
        (ses_of_interest_behavior.qchosen_l<=(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_l>(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='orange')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Laser at contra choice' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.value_laser,3)) + 
             ' mi_laser_g '+str(round(encoding_df_cluster.gains.movement_ipsi.value_laser,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Stay at contra choice
    plt.sca(ax[2,2])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    contra = np.where((1*(ses_of_interest_behavior.choice>0))!=encoding_df_cluster.hemisphere)[0]
    ses_of_interest_behavior.qchosen_s[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_s)),contra)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_s<(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_s>=np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333))&
        (ses_of_interest_behavior.qchosen_s<=(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_s>(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='gray')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='gray')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='gray')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Stay at contra choice' + '\nstay_p '+str(round(encoding_df_cluster.p_values.value_stay,3)) + 
             ' mi_stay_g '+str(round(encoding_df_cluster.gains.movement_ipsi.value_stay,3)))
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Water at outcome correct
    plt.sca(ax[0,3])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    outcome_water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    correct_water = np.intersect1d(outcome_water,outcome)
    correct_water = correct_water[correct_water>10]
    correct_water = correct_water[correct_water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.qchosen_w[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_w)),correct_water)]=np.nan

    low_value = np.where(ses_of_interest_behavior.qchosen_w<(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_w>=np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333))&
        (ses_of_interest_behavior.qchosen_w<=(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_w>(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='dodgerblue', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='dodgerblue', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='dodgerblue', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Water at reward' + '\nwater_p '+str(round(encoding_df_cluster.p_values.value_water,3)) + 
             '\now_water_g '+str(round(encoding_df_cluster.gains.outcome_water.value_water,3)) + 
             ' ow_laser_g '+str(round(encoding_df_cluster.gains.outcome_water.value_laser,3))) 
    plt.xlabel('Time from water(ms)')
    plt.ylabel(' ')

    # Laser at outcome correct
    plt.sca(ax[1,3])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    outcome_laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    correct_laser = np.intersect1d(outcome_laser,outcome)
    correct_laser = correct_laser[correct_laser>10]
    correct_laser = correct_laser[correct_laser<len(ses_of_interest_behavior.outcome)-150]


    ses_of_interest_behavior.qchosen_l[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_l)),correct_laser)]=np.nan
    low_value = np.where(ses_of_interest_behavior.qchosen_l<(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_l>=np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333))&
        (ses_of_interest_behavior.qchosen_l<=(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_l>(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='orange', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='orange', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='orange', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Laser at reward' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.value_laser,3)) + 
             '\nol_water_g '+str(round(encoding_df_cluster.gains.outcome_laser.value_water,3)) + 
             ' ol_laser_g '+str(round(encoding_df_cluster.gains.outcome_laser.value_laser,3)))  
    plt.xlabel('Time from laser(ms)')
    plt.ylabel(' ')

    # Stay at outcome correct
    plt.sca(ax[2,3])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]


    ses_of_interest_behavior.qchosen_s[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_s)),outcome)]=np.nan
    low_value = np.where(ses_of_interest_behavior.qchosen_s<(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_s>=np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333))&
        (ses_of_interest_behavior.qchosen_s<=(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_s>(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='gray', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='gray', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='gray', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Stay at reward' + '\nstay_p '+str(round(encoding_df_cluster.p_values.value_stay,3)) + 
             '\now_water_g '+str(round(encoding_df_cluster.gains.outcome_water.value_stay,3)) + 
             ' ol_laser_g '+str(round(encoding_df_cluster.gains.outcome_laser.value_stay,3)))   
    plt.xlabel('Time from reward(ms)')
    plt.ylabel(' ')

    # Water at outcome incorrect
    plt.sca(ax[0,4])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    outcome_water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    incorrect_water = np.intersect1d(outcome_water,outcome)
    incorrect_water = incorrect_water[incorrect_water>10]
    incorrect_water = incorrect_water[incorrect_water<len(ses_of_interest_behavior.outcome)-150]

    ses_of_interest_behavior.qchosen_w[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_w)),incorrect_water)]=np.nan
    low_value = np.where(ses_of_interest_behavior.qchosen_w<(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_w>=np.nanquantile(ses_of_interest_behavior.qchosen_w,0.333))&
        (ses_of_interest_behavior.qchosen_w<=(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_w>(np.nanquantile(ses_of_interest_behavior.qchosen_w,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='dodgerblue', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='dodgerblue', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='dodgerblue', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Water at error' + '\nwater_p '+str(round(encoding_df_cluster.p_values.value_water,3)) + 
             ' on_water_g '+str(round(encoding_df_cluster.gains.outcome_none.value_water,3)))   
    plt.xlabel('Time from error(ms)')
    plt.ylabel(' ')

    # Laser at outcome incorrect
    plt.sca(ax[1,4])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    outcome_laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    incorrect_laser = np.intersect1d(outcome_laser,outcome)
    incorrect_laser = incorrect_laser[incorrect_laser>10]
    incorrect_laser = incorrect_laser[incorrect_laser<len(ses_of_interest_behavior.outcome)-150]

    ses_of_interest_behavior.qchosen_l[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_l)),incorrect_laser)]=np.nan
    low_value = np.where(ses_of_interest_behavior.qchosen_l<(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_l>=np.nanquantile(ses_of_interest_behavior.qchosen_l,0.333))&
        (ses_of_interest_behavior.qchosen_l<=(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_l>(np.nanquantile(ses_of_interest_behavior.qchosen_l,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='orange', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='orange', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='orange', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Laser at error' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.value_laser,3)) + 
             ' on_laser_g '+str(round(encoding_df_cluster.gains.outcome_none.value_laser,3)))       
    plt.xlabel('Time from error(ms)')
    plt.ylabel(' ')

    # Stay at outcome incorrect
    plt.sca(ax[2,4])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]

    ses_of_interest_behavior.qchosen_s[np.setxor1d(np.arange(len(ses_of_interest_behavior.qchosen_s)),outcome)] = np.nan
    low_value = np.where(ses_of_interest_behavior.qchosen_s<(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_s>=np.nanquantile(ses_of_interest_behavior.qchosen_s,0.333))&
        (ses_of_interest_behavior.qchosen_s<=(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_s>(np.nanquantile(ses_of_interest_behavior.qchosen_s,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='gray', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='gray', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='gray', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Stay at error' + '\nstay_p '+str(round(encoding_df_cluster.p_values.value_stay,3)) + 
             ' on_stay_g '+str(round(encoding_df_cluster.gains.outcome_none.value_stay,3)))         
    plt.xlabel('Time from error(ms)')
    plt.tight_layout(w_pad = 0, h_pad=0)
    plt.ylabel(' ')
    sns.despine()

def plot_deltaq_summary(encoding_df_cluster):
    fig, ax = plt.subplots(3,5,figsize=(15,10))
    fig.suptitle('alpha=DeltaQ' +' '+ 
                ' all_p '+str(round(encoding_df_cluster.p_values.policy,3))
                )
    # Water at Cue
    plt.sca(ax[0,0])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.QWaterdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QWaterdelta)),water)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QWaterdelta<(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QWaterdelta>=np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333))&
        (ses_of_interest_behavior.QWaterdelta<=(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QWaterdelta>(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='dodgerblue')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='k')
    plt.title('Water at cue' + '\nwater_p '+str(round(encoding_df_cluster.p_values.policy_water,3)) + 
            ' c_water_g '+str(round(encoding_df_cluster.gains.cue.policy_water,3)))
    plt.xlabel('Time from cue(ms)')

    # Laser at Cue
    plt.sca(ax[1,0])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.QLaserdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QLaserdelta)),laser)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QLaserdelta<(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QLaserdelta>=np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333))&
        (ses_of_interest_behavior.QLaserdelta<=(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QLaserdelta>(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='orange')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='k')
    plt.title('Laser at cue' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.policy_laser,3)) + 
              ' c_laser_g '+str(round(encoding_df_cluster.gains.cue.policy_laser,3)))    
    plt.xlabel('Time from cue(ms)')

    # Stay at Cue
    plt.sca(ax[2,0])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
 
    low_value = np.where(ses_of_interest_behavior.QStaydelta<(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QStaydelta>=np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333))&
        (ses_of_interest_behavior.QStaydelta<=(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QStaydelta>(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='gray')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='gray')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='gray')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='k')
    plt.title('Stay at cue' + '\nstay_p '+str(round(encoding_df_cluster.p_values.policy_stay,3)) + 
             ' c_stay_g '+str(round(encoding_df_cluster.gains.cue.policy_stay,3)))     
    plt.xlabel('Time from cue(ms)')

    # Water at ipsi choice
    plt.sca(ax[0,1])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    ipsi = np.where((1*(ses_of_interest_behavior.choice>0))==encoding_df_cluster.hemisphere)[0]
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = np.intersect1d(ipsi,water)
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.QWaterdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QWaterdelta)),water)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QWaterdelta<(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QWaterdelta>=np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333))&
        (ses_of_interest_behavior.QWaterdelta<=(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QWaterdelta>(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='dodgerblue')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Water at ipsi choice' + '\nwater_p '+str(round(encoding_df_cluster.p_values.policy_water,3)) + 
             ' mc_water_g '+str(round(encoding_df_cluster.gains.movement_contra.policy_water,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Laser at ipsi choice
    plt.sca(ax[1,1])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    ipsi = np.where((1*(ses_of_interest_behavior.choice>0))==encoding_df_cluster.hemisphere)[0]
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = np.intersect1d(ipsi,laser)    
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.QLaserdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QLaserdelta)),laser)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QLaserdelta<(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QLaserdelta>=np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333))&
        (ses_of_interest_behavior.QLaserdelta<=(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QLaserdelta>(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='orange')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Laser at ipsi choice' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.policy_laser,3)) + 
             ' mc_laser_g '+str(round(encoding_df_cluster.gains.movement_contra.policy_laser,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Stay at ipsi choice
    plt.sca(ax[2,1])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    ipsi = np.where((1*(ses_of_interest_behavior.choice>0))==encoding_df_cluster.hemisphere)[0]
    ses_of_interest_behavior.QStaydelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QStaydelta)),ipsi)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QStaydelta<(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QStaydelta>=np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333))&
        (ses_of_interest_behavior.QStaydelta<=(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QStaydelta>(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='gray')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='gray')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='gray')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Stay at ipsi choice' + '\nstay_p '+str(round(encoding_df_cluster.p_values.policy_stay,3)) + 
             ' mc_stay_g '+str(round(encoding_df_cluster.gains.movement_contra.policy_stay,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Water at contra choice
    plt.sca(ax[0,2])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    contra = np.where((1*(ses_of_interest_behavior.choice>0))!=encoding_df_cluster.hemisphere)[0]
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = np.intersect1d(contra,water)    
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.QWaterdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QWaterdelta)),water)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QWaterdelta<(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QWaterdelta>=np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333))&
        (ses_of_interest_behavior.QWaterdelta<=(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QWaterdelta>(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='dodgerblue')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Water at contra choice' + '\nwater_p '+str(round(encoding_df_cluster.p_values.policy_water,3)) + 
             ' mi_water_g '+str(round(encoding_df_cluster.gains.movement_ipsi.policy_water,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Laser at contra choice
    plt.sca(ax[1,2])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    contra = np.where((1*(ses_of_interest_behavior.choice>0))!=encoding_df_cluster.hemisphere)[0]
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = np.intersect1d(contra,laser)       
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.QLaserdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QLaserdelta)),laser)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QLaserdelta<(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QLaserdelta>=np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333))&
        (ses_of_interest_behavior.QLaserdelta<=(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QLaserdelta>(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='orange')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Laser at contra choice' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.policy_laser,3)) + 
             ' mi_laser_g '+str(round(encoding_df_cluster.gains.movement_ipsi.policy_laser,3)))   
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Stay at contra choice
    plt.sca(ax[2,2])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    contra = np.where((1*(ses_of_interest_behavior.choice>0))!=encoding_df_cluster.hemisphere)[0]
    ses_of_interest_behavior.QStaydelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QStaydelta)),contra)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QStaydelta<(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QStaydelta>=np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333))&
        (ses_of_interest_behavior.QStaydelta<=(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QStaydelta>(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='gray')
    psth_plot(encoding_df_cluster, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='gray')
    psth_plot(encoding_df_cluster, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='gray')
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.66, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0]+1, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Stay at contra choice' + '\nstay_p '+str(round(encoding_df_cluster.p_values.policy_stay,3)) + 
             ' mi_stay_g '+str(round(encoding_df_cluster.gains.movement_ipsi.policy_stay,3)))
    plt.xlabel('Time from choice(ms)')
    plt.ylabel(' ')

    # Water at outcome correct
    plt.sca(ax[0,3])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    outcome_water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    correct_water = np.intersect1d(outcome_water,outcome)
    correct_water = correct_water[correct_water>10]
    correct_water = correct_water[correct_water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.QWaterdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QWaterdelta)),correct_water)]=np.nan

    low_value = np.where(ses_of_interest_behavior.QWaterdelta<(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QWaterdelta>=np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333))&
        (ses_of_interest_behavior.QWaterdelta<=(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QWaterdelta>(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='dodgerblue', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='dodgerblue', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='dodgerblue', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Water at reward' + '\nwater_p '+str(round(encoding_df_cluster.p_values.policy_water,3)) + 
             '\now_water_g '+str(round(encoding_df_cluster.gains.outcome_water.policy_water,3)) + 
             ' ow_laser_g '+str(round(encoding_df_cluster.gains.outcome_water.policy_laser,3))) 
    plt.xlabel('Time from water(ms)')
    plt.ylabel(' ')

    # Laser at outcome correct
    plt.sca(ax[1,3])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    outcome_laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    correct_laser = np.intersect1d(outcome_laser,outcome)
    correct_laser = correct_laser[correct_laser>10]
    correct_laser = correct_laser[correct_laser<len(ses_of_interest_behavior.outcome)-150]


    ses_of_interest_behavior.QLaserdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QLaserdelta)),correct_laser)]=np.nan
    low_value = np.where(ses_of_interest_behavior.QLaserdelta<(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QLaserdelta>=np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333))&
        (ses_of_interest_behavior.QLaserdelta<=(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QLaserdelta>(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='orange', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='orange', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='orange', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Laser at reward' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.policy_laser,3)) + 
             '\nol_water_g '+str(round(encoding_df_cluster.gains.outcome_laser.policy_water,3)) + 
             ' ol_laser_g '+str(round(encoding_df_cluster.gains.outcome_laser.policy_laser,3)))  
    plt.xlabel('Time from laser(ms)')
    plt.ylabel(' ')

    # Stay at outcome correct
    plt.sca(ax[2,3])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]


    ses_of_interest_behavior.QStaydelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QStaydelta)),outcome)]=np.nan
    low_value = np.where(ses_of_interest_behavior.QStaydelta<(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QStaydelta>=np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333))&
        (ses_of_interest_behavior.QStaydelta<=(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QStaydelta>(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='gray', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='gray', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='gray', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Stay at reward' + '\nstay_p '+str(round(encoding_df_cluster.p_values.policy_stay,3)) + 
             '\now_water_g '+str(round(encoding_df_cluster.gains.outcome_water.policy_stay,3)) + 
             ' ol_laser_g '+str(round(encoding_df_cluster.gains.outcome_laser.policy_stay,3)))   
    plt.xlabel('Time from reward(ms)')
    plt.ylabel(' ')

    # Water at outcome incorrect
    plt.sca(ax[0,4])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    outcome_water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    incorrect_water = np.intersect1d(outcome_water,outcome)
    incorrect_water = incorrect_water[incorrect_water>10]
    incorrect_water = incorrect_water[incorrect_water<len(ses_of_interest_behavior.outcome)-150]

    ses_of_interest_behavior.QWaterdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QWaterdelta)),incorrect_water)]=np.nan
    low_value = np.where(ses_of_interest_behavior.QWaterdelta<(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QWaterdelta>=np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.333))&
        (ses_of_interest_behavior.QWaterdelta<=(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QWaterdelta>(np.nanquantile(ses_of_interest_behavior.QWaterdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='dodgerblue', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='dodgerblue', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='dodgerblue', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Water at error' + '\nwater_p '+str(round(encoding_df_cluster.p_values.policy_water,3)) + 
             ' on_water_g '+str(round(encoding_df_cluster.gains.outcome_none.policy_water,3)))   
    plt.xlabel('Time from error(ms)')
    plt.ylabel(' ')

    # Laser at outcome incorrect
    plt.sca(ax[1,4])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    outcome_laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    incorrect_laser = np.intersect1d(outcome_laser,outcome)
    incorrect_laser = incorrect_laser[incorrect_laser>10]
    incorrect_laser = incorrect_laser[incorrect_laser<len(ses_of_interest_behavior.outcome)-150]

    ses_of_interest_behavior.QLaserdelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QLaserdelta)),incorrect_laser)]=np.nan
    low_value = np.where(ses_of_interest_behavior.QLaserdelta<(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QLaserdelta>=np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.333))&
        (ses_of_interest_behavior.QLaserdelta<=(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QLaserdelta>(np.nanquantile(ses_of_interest_behavior.QLaserdelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='orange', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='orange', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='orange', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Laser at error' + '\nlaser_p '+str(round(encoding_df_cluster.p_values.policy_laser,3)) + 
             ' on_laser_g '+str(round(encoding_df_cluster.gains.outcome_none.policy_laser,3)))       
    plt.xlabel('Time from error(ms)')
    plt.ylabel(' ')

    # Stay at outcome incorrect
    plt.sca(ax[2,4])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_deltaq(ses_of_interest_behavior, encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]

    ses_of_interest_behavior.QStaydelta[np.setxor1d(np.arange(len(ses_of_interest_behavior.QStaydelta)),outcome)] = np.nan
    low_value = np.where(ses_of_interest_behavior.QStaydelta<(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QStaydelta>=np.nanquantile(ses_of_interest_behavior.QStaydelta,0.333))&
        (ses_of_interest_behavior.QStaydelta<=(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QStaydelta>(np.nanquantile(ses_of_interest_behavior.QStaydelta,0.666)))

    psth_plot(encoding_df_cluster, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='gray', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='gray', pre_time = 200, post_time = 800)
    psth_plot(encoding_df_cluster, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='gray', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, high_value[0], encoding_df_cluster.epoch_times.outcome, alpha=1, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, medium_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.66, color='k', pre_time = 200, post_time = 800)
    kernel_plot(encoding_df_cluster.model_peth.full_model, low_value[0], encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k', pre_time = 200, post_time = 800)
    plt.title('Stay at error' + '\nstay_p '+str(round(encoding_df_cluster.p_values.policy_stay,3)) + 
             ' on_stay_g '+str(round(encoding_df_cluster.gains.outcome_none.policy_stay,3)))         
    plt.xlabel('Time from error(ms)')
    plt.tight_layout(w_pad = 0, h_pad=0)
    plt.ylabel(' ')
    sns.despine()

def plot_choice_summary(encoding_df_cluster):
    fig, ax = plt.subplots(2,4,figsize=(15,10))
    fig.suptitle('alpha=DeltaQ' +' '+ 
                ' all_p '+str(round(encoding_df_cluster.p_values.choice,3))
                )
    plt.sca(ax[0,0])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster)
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]

    ipsi = np.intersect1d(np.where(ses_of_interest_behavior.choice_ipsi==1), water)
    contra = np.intersect1d(np.where(ses_of_interest_behavior.choice_contra==1), water)

    psth_plot(encoding_df_cluster, ipsi, encoding_df_cluster.epoch_times.goCue, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, contra, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, ipsi, encoding_df_cluster.epoch_times.goCue, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, contra, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='k')
    plt.title('Water at cue' + '\nwater_p '+str(round(encoding_df_cluster.p_values.policy_water,3)) + 
            ' c_water_g '+str(round(encoding_df_cluster.gains.cue.policy_water,3)))
    plt.xlabel('Time from cue(ms)')


    plt.sca(ax[0,1])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster) 
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]

    ipsi = np.intersect1d(np.where(ses_of_interest_behavior.choice_ipsi==1), water)
    contra = np.intersect1d(np.where(ses_of_interest_behavior.choice_contra==1), water)

    psth_plot(encoding_df_cluster, ipsi, encoding_df_cluster.epoch_times.choice, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, contra, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, ipsi, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, contra, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Water at choice')
    plt.xlabel('Time from choice(s)')


    plt.sca(ax[0,2])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster) 
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    outcome_water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    correct_water = np.intersect1d(outcome_water,outcome)
    correct_water = correct_water[correct_water>10]
    correct_water = correct_water[correct_water<len(ses_of_interest_behavior.outcome)-150]

    ipsi = np.intersect1d(np.where(ses_of_interest_behavior.choice_ipsi==1), correct_water)
    contra = np.intersect1d(np.where(ses_of_interest_behavior.choice_contra==1), correct_water)

    psth_plot(encoding_df_cluster, ipsi, encoding_df_cluster.epoch_times.outcome, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, contra, encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, ipsi, encoding_df_cluster.epoch_times.outcome, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, contra, encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k')
    plt.title('Correct Water at outcome')
    plt.xlabel('Time from outcome(s)')

    plt.sca(ax[1,0])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster)
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]

    ipsi = np.intersect1d(np.where(ses_of_interest_behavior.choice_ipsi==1), laser)
    contra = np.intersect1d(np.where(ses_of_interest_behavior.choice_contra==1), laser)

    psth_plot(encoding_df_cluster, ipsi, encoding_df_cluster.epoch_times.goCue, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, contra, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, ipsi, encoding_df_cluster.epoch_times.goCue, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, contra, encoding_df_cluster.epoch_times.goCue, alpha=0.33, color='k')
    plt.title('Laser at cue')
    plt.xlabel('Time from cue(s)')

    plt.sca(ax[1,1])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster)
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]

    ipsi = np.intersect1d(np.where(ses_of_interest_behavior.choice_ipsi==1), laser)
    contra = np.intersect1d(np.where(ses_of_interest_behavior.choice_contra==1), laser)

    psth_plot(encoding_df_cluster, ipsi, encoding_df_cluster.epoch_times.choice, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, contra, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, ipsi, encoding_df_cluster.epoch_times.choice, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, contra, encoding_df_cluster.epoch_times.choice, alpha=0.33, color='k')
    plt.title('Laser at choice')
    plt.xlabel('Time from choice(s)')

    plt.sca(ax[1,2])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    outcome_laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    correct_laser = np.intersect1d(outcome_laser,outcome)
    correct_laser = correct_laser[correct_laser>10]
    correct_laser = correct_laser[correct_laser<len(ses_of_interest_behavior.outcome)-150]

    ipsi = np.intersect1d(np.where(ses_of_interest_behavior.choice_ipsi==1), correct_laser)
    contra = np.intersect1d(np.where(ses_of_interest_behavior.choice_contra==1), correct_laser)

    psth_plot(encoding_df_cluster, ipsi, encoding_df_cluster.epoch_times.outcome, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, contra, encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, ipsi, encoding_df_cluster.epoch_times.outcome, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, contra, encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k')
    plt.title('Correct Laser at outcome')
    plt.xlabel('Time from outcome(s)')
    plt.tight_layout()

    plt.sca(ax[0,3])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    outcome_water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    incorrect_water = np.intersect1d(outcome_water,outcome)
    incorrect_water = incorrect_water[incorrect_water>10]
    incorrect_water = incorrect_water[incorrect_water<len(ses_of_interest_behavior.outcome)-150]

    ipsi = np.intersect1d(np.where(ses_of_interest_behavior.choice_ipsi==1), incorrect_water)
    contra = np.intersect1d(np.where(ses_of_interest_behavior.choice_contra==1), incorrect_water)

    psth_plot(encoding_df_cluster, ipsi, encoding_df_cluster.epoch_times.outcome, alpha=1, color='dodgerblue')
    psth_plot(encoding_df_cluster, contra, encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='dodgerblue')
    kernel_plot(encoding_df_cluster.model_peth.full_model, ipsi, encoding_df_cluster.epoch_times.outcome, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, contra, encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k')
    plt.title('Incorrect water at outcome')
    plt.xlabel('Time from outcome(s)')

    plt.sca(ax[1,3])
    ses_of_interest_behavior = copy.deepcopy(encoding_df_cluster.alf)
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior,encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    outcome_laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    incorrect_laser = np.intersect1d(outcome_laser,outcome)
    incorrect_laser = incorrect_laser[incorrect_laser>10]
    incorrect_laser = incorrect_laser[incorrect_laser<len(ses_of_interest_behavior.outcome)-150]

    ipsi = np.intersect1d(np.where(ses_of_interest_behavior.choice_ipsi==1), incorrect_laser)
    contra = np.intersect1d(np.where(ses_of_interest_behavior.choice_contra==1), incorrect_laser)

    psth_plot(encoding_df_cluster, ipsi, encoding_df_cluster.epoch_times.outcome, alpha=1, color='orange')
    psth_plot(encoding_df_cluster, contra, encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='orange')
    kernel_plot(encoding_df_cluster.model_peth.full_model, ipsi, encoding_df_cluster.epoch_times.outcome, alpha=1, color='k')
    kernel_plot(encoding_df_cluster.model_peth.full_model, contra, encoding_df_cluster.epoch_times.outcome, alpha=0.33, color='k')
    plt.title('Incorrect Laser at outcome')
    plt.xlabel('Time from outcome(s)')

def get_epochs_from_struc(time_struc):
    n_trial = time_struc.shape[0]
    goCue = np.zeros([n_trial])
    choice = np.zeros([n_trial])
    outcome = np.zeros([n_trial])
    goCue[:] = np.nan
    choice[:] = np.nan
    outcome[:] =np.nan
    for i in np.arange(n_trial):
        if time_struc[i].size>0:
            goCue[i] = time_struc[i][0][0]
            choice[i] = time_struc[i][0][1]
            outcome[i] = time_struc[i][0][2]
    return goCue, choice, outcome

# Object
class neuron:
    def __init__(self,data_dict):
        class alf:
            def __init__(self, path):
                ROOT = '/Volumes/witten/Alex/Data/Subjects/'
                path= ROOT + path[:-4]
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

            def to_df(self):
                return pd.DataFrame.from_dict(self.__dict__)
        class gains:
            def __init__(self):
                class gain_values:
                    def __init__(self, gain_array):
                        self.constant = gain_array[0]
                        self.policy_water = gain_array[1]
                        self.policy_laser = gain_array[2]
                        self.policy_stay = gain_array[3]
                        self.value_water = gain_array[4]
                        self.value_laser = gain_array[5]
                        self.value_stay = gain_array[6]
                self.cue = gain_values(data_dict['EncodingModels']['gains'][0][0][0][0][0][0][0][0][0])
                self.movement_contra = gain_values(data_dict['EncodingModels']['gains'][0][0][0][0][0][0][0][1][0])
                self.movement_ipsi = gain_values(data_dict['EncodingModels']['gains'][0][0][0][0][0][0][0][2][0])
                self.outcome_water = gain_values(data_dict['EncodingModels']['gains'][0][0][0][0][0][0][0][3][0])
                self.outcome_laser = gain_values(data_dict['EncodingModels']['gains'][0][0][0][0][0][0][0][4][0])
                self.outcome_none = gain_values(data_dict['EncodingModels']['gains'][0][0][0][0][0][0][0][5][0])

        class p_values:
            def __init__(self):
                self.choice = data_dict['EncodingModels']['pvalues'][0][0][0][0][0][0][0]
                self.outcome = data_dict['EncodingModels']['pvalues'][0][0][0][0][1][0][0]
                self.policy = data_dict['EncodingModels']['pvalues'][0][0][0][0][2][0][0]
                self.policy_water = data_dict['EncodingModels']['pvalues'][0][0][0][0][3][0][0]
                self.policy_laser = data_dict['EncodingModels']['pvalues'][0][0][0][0][4][0][0]
                self.policy_stay = data_dict['EncodingModels']['pvalues'][0][0][0][0][5][0][0]
                self.value = data_dict['EncodingModels']['pvalues'][0][0][0][0][6][0][0]
                self.value_water = data_dict['EncodingModels']['pvalues'][0][0][0][0][7][0][0]
                self.value_laser = data_dict['EncodingModels']['pvalues'][0][0][0][0][8][0][0]
                self.value_stay = data_dict['EncodingModels']['pvalues'][0][0][0][0][9][0][0]

        class kernels:
            def __init__(self):
                self.cue = data_dict['EncodingModels']['kernels'][0][0][0][0][0][0]
                self.movement_contra = data_dict['EncodingModels']['kernels'][0][0][0][0][1][0]
                self.movement_ipsi = data_dict['EncodingModels']['kernels'][0][0][0][0][2][0]
                self.outcome_water = data_dict['EncodingModels']['kernels'][0][0][0][0][3][0]
                self.outcome_laser = data_dict['EncodingModels']['kernels'][0][0][0][0][4][0]
                self.outcome_none = data_dict['EncodingModels']['kernels'][0][0][0][0][5][0]
        
        class epoch_times:
            def __init__(self):
                self.goCue, self.choice, self.outcome = get_epochs_from_struc(data_dict['EncodingModels']['trials'][0][0][0][0][0][0])
        class model_peths:
            def __init__(self):
                self.full_model = get_peth_from_struc(data_dict['EncodingModels']['trials'][0][0][0][0][2][0][0][0][0])
                self.policy_model = get_peth_from_struc(data_dict['EncodingModels']['trials'][0][0][0][0][2][0][0][1][0])
                self.stay_model = get_peth_from_struc(data_dict['EncodingModels']['trials'][0][0][0][0][2][0][0][2][0])

        self.path  = pathlib.PureWindowsPath(data_dict['EncodingModels']['recording'][0][0][0]).as_posix()
        self.cluster = data_dict['EncodingModels']['cluster'][0][0][0]
        self.location = data_dict['EncodingModels']['location'][0][0][0]
        self.R2 = data_dict['EncodingModels']['R2'][0][0][0]
        self.p_values = p_values()
        self.kernels = kernels()
        self.gains = gains()
        self.epoch_times = epoch_times()
        self.peth = get_peth_from_struc(data_dict['EncodingModels']['trials'][0][0][0][0][1][0])
        self.model_peth = model_peths()
        self.alf = alf(self.path)
        self.hemisphere = data_dict['EncodingModels']['hemisphere'][0][0][0][0]

def plot_PETHs(encoding_df_cluster):
    #qChosen
    reg = encoding_df_cluster.location
    significant = (encoding_df_cluster.p_values.value<=0.01)| \
                (encoding_df_cluster.p_values.value_water<=0.01)| \
                (encoding_df_cluster.p_values.value_laser<=0.01)| \
                (encoding_df_cluster.p_values.value_stay<=0.01)
    SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/qchosen'
    plot_qchosen_summary(encoding_df_cluster)
    if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))==False:
        os.mkdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))
    if os.path.isdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))==False:
        os.mkdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))
    if significant: 
        plt.savefig(SAVE_PATH+'/significant/'+reg+'/'+str(encoding_df_cluster.cluster)+'.pdf')
    else: 
        plt.savefig(SAVE_PATH+'/nonsignificant/'+reg+'/'+str(encoding_df_cluster.cluster)+'.pdf')
    plt.close()
    #qPolicy
    significant = (encoding_df_cluster.p_values.policy<=0.01)| \
                (encoding_df_cluster.p_values.policy_water<=0.01)| \
                (encoding_df_cluster.p_values.policy_laser<=0.01)| \
                (encoding_df_cluster.p_values.policy_stay<=0.01)
    SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/deltaq'
    plot_deltaq_summary(encoding_df_cluster)
    if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))==False:
        os.mkdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))
    if os.path.isdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))==False:
        os.mkdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))
    if significant: 
        plt.savefig(SAVE_PATH+'/significant/'+reg+'/'+str(encoding_df_cluster.cluster)+'.pdf')
    else: 
        plt.savefig(SAVE_PATH+'/nonsignificant/'+reg+'/'+str(encoding_df_cluster.cluster)+'.pdf')
    plt.close()
    #choice
    significant = (encoding_df_cluster.p_values.choice<=0.01)
    SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/choice'
    plot_choice_summary(encoding_df_cluster)
    if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))==False:
        os.mkdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))
    if os.path.isdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))==False:
        os.mkdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))
    if significant: 
        plt.savefig(SAVE_PATH+'/significant/'+reg+'/'+str(encoding_df_cluster.cluster)+'.pdf')
    else: 
        plt.savefig(SAVE_PATH+'/nonsignificant/'+reg+'/'+str(encoding_df_cluster.cluster)+'.pdf')
    plt.close()

source ='/Volumes/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
os.chdir(source)
# Find all clusters
clusters = []
for mouse in np.os.listdir(source):
    if 'dop' not in mouse: 
        continue
    for date in np.os.listdir(source+'/'+mouse):
        if os.path.isdir(source+'/'+mouse+'/'+date)!=True:
            continue
        for ses in np.os.listdir(source+'/'+mouse+'/'+date):
            if os.path.isdir(source+'/'+mouse+'/'+date+'/'+ses)!=True:
                continue
            probes = source+'/'+mouse+'/'+date+'/'+ses+'/alf/EncodingModelSummary'
            for probe in np.os.listdir(probes):
                if os.path.isdir(probes+'/'+probe)!=True:
                    continue
                probe_path = probes+'/'+probe
                clusters.append(list(Path(probe_path).rglob('*.mat')))
                
t = time.time()
errors = []
for clu in clusters:
    for path in clu:
        try:
            data_dict = loadmat(path)
            encoding_df_cluster = neuron(data_dict)
            significant = (encoding_df_cluster.p_values.choice<=0.01)
            plot_PETHs(encoding_df_cluster)
        except:
            errors.append(path)
elapsed = time.time() - t
