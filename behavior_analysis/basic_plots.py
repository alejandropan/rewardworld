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
import matplotlib.patches as mpatches

def ibl_psychometric (psy_df, ax=None, **kwargs):
    """Calculates and plot psychometic curve from dataframe
    datajoint independent
    assumes that there is not signed contrast in dataframe
    INPUTS: Dataframe where index is the trial number
    OUTPUTS:  psychometic fit using IBL function from Miles and fit parameters"""
        
    #1st calculate some useful data...
    psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df.loc[:,'signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])*100
    unique_signed_contrasts  = sorted(psy_df['signed_contrasts'].unique())

    
    right_choices = psy_df['choice']== -1
    psy_df.loc[:,'right_choices'] = right_choices
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
        marker='.', color = 'black',  ci=68, data= psy_df)

    return pars,  L
    
def plot_psych_block (psy_df , block_variable):
    """Plots psychometric using ibl_psychometric
    INPUT:  Dataframe where index = trial and block variable = hue
    OUTPUT:  Average of all sessions, Average of Last three sessions, Last 5 sessions"""
    blocks  = psy_df[block_variable].unique()
    
    #First get fits for each block
    #Set frame for plots
    block_summary, axes = plt.subplots(1,2)
    plt.sca(axes[0])
    
    #First get fits for each block
    for i in blocks:
        psy_df_block  = psy_df.loc[psy_df[block_variable] == i]
        pars,  L  =  ibl_psychometric (psy_df_block)
        #Get sns.lineplot for raw data for each contrast per session
    axes[0].set_xlabel('Signed contrast (%)')
    axes[0].set_ylabel('% Right')
    axes[0].set_title('All sessions')
    #plot average last three session
    dates =  sorted(psy_df['ses'].unique())
    psy_df_last3  =  psy_df.loc[(psy_df['ses'] == dates[-1]) | (psy_df['ses'] == dates[-2]) | (psy_df['ses'] == dates[-3])]
    plt.sca(axes[1])
    for i in blocks:
        psy_df_block  = psy_df_last3.loc[psy_df_last3[block_variable] == i]
        pars,  L  =  ibl_psychometric (psy_df_block)
    axes[1].set_xlabel('Signed contrast (%)')
    axes[1].set_title('Last 3 sessions')
    
    #Now plot last 5 sessions
    plots = len(dates)
    rows = int(np.ceil(plots/3))
    cols = int(np.ceil(plots/rows))
    all_sessions =  plt.figure(figsize=(12, 10))
    for j, date in enumerate(dates):
                ax  = all_sessions.add_subplot(rows, cols,1+j)
                ax.set_title(date)
                ax.label_outer()
                ax.set_xlabel('Signed contrast (%)')
                ax.set_ylabel('Right choices (%)')
                for i in blocks:
                    psy_df_block  = psy_df.loc[psy_df[block_variable] == i]
                    psy_df_block_date = psy_df_block.loc[psy_df_block['ses'] == date]
                    pars,  L  =  ibl_psychometric (psy_df_block_date,ax)
                    ax.set_label("block %s" %i)
    blue_patch = mpatches.Patch(color='blue', label='Right P(rew) = 0.8')
    orange_patch  =  mpatches.Patch(color='orange', label='Right P(rew) = 0.4')
    plt.legend(handles=[blue_patch, orange_patch])                
    
    return block_summary, all_sessions
        
def block_qc (psy_df):
    """Quality control for block structure
    INPUT:  dataframe with trial as index
    OUTPUT: summary statistics of each block
    left choice is -1 , right choice 1 , feedback -2 is correct_unrewarded, 1 correct_rewarded"""
    #Select successful trials (reward and unrewarded)
    qc = psy_df.loc[(psy_df['feedbackType']== 1 ) | ( psy_df['feedbackType']  == -2)]
    #Right choice rewarded rate
    qc.loc[:,'correct_rewarded'] = qc['feedbackType'] == 1
    qc.loc[:,'correct_unrewarded'] = qc['feedbackType'] == -2
    #mean of boolean gives you percentage correct
    block_stats  =  qc.groupby(['rewprobabilityLeft','choice'])['correct_rewarded','correct_unrewarded'].mean()
    
    return block_stats