#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:09:19 2019
Behavior statistics
Some based on @anneurai and IBL datajoint pipeline
Also functions copied from alex_psy
@author: ibladmin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:11:57 2019

@author: Alejandro
"""
import numpy as np
from ibl_pipeline.utils import psychofit as psy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def ibl_psychometric (psy_df, ax=None, **kwargs):
    """Calculates and plot psychometic curve from dataframe
    datajoint independent
    assumes that there is not signed contrast in dataframe
    INPUTS: Dataframe where index is the trial number
    OUTPUTS:  psychometic fit using IBL function from Miles and fit parameters"""
        
    #1st calculate some useful data...
    psy_df['contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df['contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df['signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])*100
    unique_signed_contrasts  = sorted(psy_df['signed_contrasts'].unique())
    
    right_choices = psy_df['choice']== -1
    psy_df['right_choices'] = right_choices
    total_trials = []
    right_trials = []
            
    for cont in unique_signed_contrasts:
        matching = (psy_df['signed_contrasts'] == cont)
        total_trials.append(np.sum(matching))
        right_trials.append(np.sum(right_choices[matching]))

    prop_right_trials = np.divide(right_trials, total_trials)
    
    pars, L = psy.mle_fit_psycho(
            np.vstack([unique_signed_contrasts, total_trials, prop_right_trials]),
            P_model='erf_psycho_2gammas',
            parstart=np.array([np.mean(unique_signed_contrasts), 20., 0.05, 0.05]),
            parmin=np.array([np.min(unique_signed_contrasts), 0., 0., 0.]),
            parmax=np.array([np.max(unique_signed_contrasts), 100., 1, 1]))

    sns.lineplot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)))
    
    
    sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", linewidth=0, linestyle='None', mew=0.5,
        marker='.', ci=68, data= psy_df)
    
    

    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(['-100', '-50', '0', '50', '100'])
    ax.set_xlim([-110, 110])

    ax.set_yticks([0, .5, 1])
    ax.set_ylim([-0.03, 1.03])
    ax.set_xlabel('Contrast (%)')
    
    return ax , pars,  L
    
def plot_psych_block (psy_df , block_variable, ax):
    """Plots psychometric using ibl_psychometric
    INPUT:  Dataframe where index = trial and block variable = hue"""
    blocks  = psy_df[block_variable].unique()
    
    #First get fits for each block
    for i in blocks:
        psy_df_block  = psy_df.loc[psy_df[block_variable] == i]
        ax , pars,  L  =  ibl_psychometric (psy_df_block, ax)
        #Get sns.lineplot for raw data for each contrast per session
    
    #plot last three session
    
    for date in sorted(psy_df['ses'].unique())[-3:]:
        for i in blocks:
            psy_df_block  = psy_df.loc[psy_df[block_variable] == i]
            psy_df_block_date = psy_df_block.loc[psy_df_block['ses'] == date]
            ax , pars,  L  =  ibl_psychometric (psy_df_block_date, ax)
        #Still need to modify so that it fills subplots
        
def block_qc (psy_df):
    """Quality control for block structure
    INPUT:  dataframe with trial as index
    OUTPUT: summary statistics of each block
    left choice is -1 , right choice 1 , feedback -2 is correct_unrewarded, 1 correct_rewarded"""
    #Select successful trials (reward and unrewarded)
    qc = psy_df.loc[(psy_df['feedbackType']== 1 ) | ( psy_df['feedbackType']  == -2)]
    #Right choice rewarded rate
    qc['correct_rewarded'] = qc['feedbackType'] == 1
    qc['correct_unrewarded'] = qc['feedbackType'] == -2

    block_stats  =  qc.groupby(['rewprobabilityLeft','choice'])['correct_rewarded','correct_unrewarded'].mean()
    
    return block_stats