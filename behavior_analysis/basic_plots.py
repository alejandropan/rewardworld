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
from glm import *

def signed_contrast(psy_df):
    psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df.loc[:,'signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])*100
    
    return psy_df
    
def ibl_psychometric (psy_df, ax=None, **kwargs):
    """Calculates and plot psychometic curve from dataframe
    datajoint independent
    assumes that there is not signed contrast in dataframe
    INPUTS: Dataframe where index is the trial number
    OUTPUTS:  psychometic fit using IBL function from Miles and fit parameters"""
        
    #1st calculate some useful data...
    
    psy_df = signed_contrast(psy_df)
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


    return pars,  L




def plot_psych_var_block(psy_df , block_variable, blocks, block2_variable, blocks2):
    """Plots psychometric using ibl_psychometric
    INPUT:  Dataframe where index = trial and block variable = hue
    block_variable =   name of block struct column 
    blocks =  blovks that I want plotted (np.array) (e.g np.array([1, 0])),
    ax= axes to place the plot
    block2 variable and blocks if for adding an extra block eg prob or opto
    OUTPUT:  Average of all sessions, Average of Last three sessions, Last 5 sessions"""
    
    #First get fits for each block
    #Set frame for plots
    fig , axes =  plt.subplots(1,1)
    plt.sca(axes)
    if block2_variable:
        sns.set()
        lines = ['dashed','solid']
        colors =['green','black','blue']
        for j,i in enumerate(blocks):
            for p, bl2 in enumerate(blocks2):
                psy_df_block  = psy_df.loc[(psy_df[block2_variable] == bl2) & (psy_df[block_variable] == i)] 
                pars,  L  =  ibl_psychometric (psy_df_block)
                plt.plot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), linewidth=2,\
                         color = colors[p], linestyle = lines[j])
                sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", \
                             linewidth=0, linestyle='None', mew=0.5,marker='.',
                             color = colors[p],  ci=68, data= psy_df_block, ax =axes)
            
    else:       
        colors = ['blue','black']
        sns.set()
        #First get fits for each block
        for j,i in enumerate(blocks):
            psy_df_block  = psy_df.loc[psy_df[block_variable] == i]
            pars,  L  =  ibl_psychometric (psy_df_block) 
            ln = sns.lineplot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), color = colors[p], ax = axes, linewidth=2)
            ln  = sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", linewidth=0, linestyle='None', mew=0.5,
                         marker='.', color = colors[j],  ci=68, data= psy_df_blocks)
            #Get sns.lineplot for raw data for each contrast per session
    
    
    axes.set_xlim([-50,50])
    green_patch = mpatches.Patch(color='green', label='P(L) = 0.8')
    black_patch  =  mpatches.Patch(color='black', label='P(L) = 0.5')
    blue_patch  =  mpatches.Patch(color='blue', label='P(L) = 0.2')
    dashed  =  plt.Line2D([0], [0], color='black', lw=2, label='Laser on', ls = '--')
    solid  = plt.Line2D([0], [0], color='black', lw=2, label='Laser off', ls = '-')
    plt.legend(handles=[dashed, solid,green_patch, black_patch, blue_patch], loc = 'lower right')
    
    return fig, axes

    
def plot_psych_block_sessions (psy_df , block_variable, blocks):
    """Plots psychometric using ibl_psychometric
    INPUT:  Dataframe where index = trial and block variable = hue
    block_variable =   name of block struct column (e.g rewprobabilityLeft)
    blocks =  blovks that I want plotted (np.array) (e.g np.array([1, 0.7]))
    OUTPUT:  Average of all sessions, Average of Last three sessions, Last 5 sessions"""
    
    #First get fits for each block
    #Set frame for plots
    block_summary, axes = plt.subplots(1,2, sharex=True)
    block_summary.set_figheight(7)
    block_summary.set_figwidth(15)
    plt.sca(axes[0])
    colors = ['blue','green']
    sns.set()
    #First get fits for each block
    for j,i in enumerate(blocks):
        psy_df_block  = psy_df.loc[psy_df[block_variable] == i]
        pars,  L  =  ibl_psychometric (psy_df_block)
        
        sns.lineplot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), color = colors[j])
        sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", linewidth=0, linestyle='None', mew=0.5,
                     marker='.', color = colors[j],  ci=68, data= psy_df_block)
        #Get sns.lineplot for raw data for each contrast per session
    axes[0].set_xlabel('Signed contrast (%)')
    axes[0].set_ylabel('% Right')
    axes[0].set_title('All sessions')
    #plot average last three session
    dates =  sorted(psy_df['ses'].unique())
    psy_df_last3  =  psy_df.loc[(psy_df['ses'] == dates[-1]) | (psy_df['ses'] == dates[-2]) | (psy_df['ses'] == dates[-3])]
    plt.sca(axes[1])
    for j,i in  enumerate(blocks):
        psy_df_block  = psy_df_last3.loc[psy_df_last3[block_variable] == i]
        pars,  L  =  ibl_psychometric (psy_df_block)
        sns.lineplot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), color = colors[j])
        sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", linewidth=0, linestyle='None', mew=0.5,
                     marker='.', color = colors[j],  ci=68, data= psy_df_block)
    axes[1].set_xlabel('Signed contrast (%)')
    axes[1].set_title('Last 3 sessions')
    axes[1].set_ylabel('')
    #Now plot last 5 sessions
    plots = len(dates)
    rows = int(np.ceil(plots/3))
    cols = int(np.ceil(plots/rows))
    all_sessions =  plt.figure(figsize=(12, 10))
    sns.set()
    for j, date in enumerate(dates):
                ax  = all_sessions.add_subplot(rows, cols,1+j)
                ax.set_title(date)
                ax.label_outer()
                
                for l,i in enumerate(blocks):
                    psy_df_block  = psy_df.loc[psy_df[block_variable] == i]
                    psy_df_block_date = psy_df_block.loc[psy_df_block['ses'] == date]
                    pars,  L  =  ibl_psychometric (psy_df_block_date,ax)
                    sns.lineplot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), color = colors[l])
                    sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", linewidth=0, linestyle='None', mew=0.5,
                     marker='.', color = colors[l],  ci=68, data= psy_df_block_date)
                    ax.set_label("block %s" %i)
                    ax.set_xlabel('Signed contrast (%)')
                    ax.set_ylabel('Right choices (%)')
    blue_patch = mpatches.Patch(color='blue', label='P(rew|Left): 1 P(rew|Right): 0.7')
    orange_patch  =  mpatches.Patch(color='green', label='P(rew|Left): 0.7 P(rew|Right): 1')
    plt.legend(handles=[blue_patch, orange_patch], loc = 'lower right')
    ax.set_xlabel('Signed contrast (%)')        
    
    return block_summary, all_sessions
        
def zerobias(psy_df, block_variable, blocks):
    """
    Description : Calculates bias at 0 contrast and drift between two blocks 
    INPUT:  dataframe with data for all trials, name of the block variable and identity of blockj (in porobability of left)
    OUTPUT: Bias drift at 0
    WARNING: ONLY accepts two blocks at the moment, should be ordered from higest to lower (e.g [1 , 0.7])
    """
    
    psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df.loc[:,'signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])*100
    unique_signed_contrasts  = sorted(psy_df['signed_contrasts'].unique())
    
    bias = []
    
    for i in blocks:
       psy_df_block = psy_df.loc[psy_df[block_variable] == i]
       right_choices = psy_df_block['choice']== -1
       psy_df_block.loc[:,'right_choices'] = right_choices
       total_trials = []
       right_trials = []
       
       for cont in unique_signed_contrasts:
            matching = (psy_df_block['signed_contrasts'] == cont)
            total_trials.append(np.sum(matching))
            right_trials.append(np.sum(right_choices[matching]))
       bias.append (right_trials[4]/total_trials[4])
    
    bias_drift  = bias[1]  - bias[0]
    
    return bias_drift

def bias_per_session(psy_df, block_variable, blocks):
    """
    Uses zero bias to calculate and plot bias per session
    INPUT:  psy_df, block_variable, blocks (as described before)
    OUTPUT: bias drift per sesssion and plot 
    """
    
    dates =  sorted(psy_df['ses'].unique())
    
    bias = pd.DataFrame(columns = ['dates', 'bias'])
    bias.loc[:,'dates']  = dates
    bias['bias']  = np.nan
    
    for date in dates:
                    psy_df_date = psy_df.loc[psy_df['ses'] == date]       
                    bias.loc[(bias['dates'] == date), 'bias'] =  zerobias(psy_df_date, block_variable, blocks)
    
    
    fig, ax  = plt.subplots()
    sns.pointplot(x = 'dates' , y = 'bias', data  = bias, color='black' )
    ax.set_xlabel('Dates')
    ax.set_ylabel('Block bias shift')
    ax.set_ylim([-0.5,0.5])
    plt.xticks(rotation=70)
    
    return bias

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

def plot_glm(psy_df, result, r2):
    """
    INPUT:  psy_df, result of regression, r2 of regressions
    OUTPUT: Dataframe with data for plotting  + significance
    """

    results  =  pd.DataFrame({"Predictors": result.model.exog_names , "Coef" : result.params.values,\
                              "SEM": result.bse.values, "Significant": result.pvalues < 0.05/len(result.model.exog_names)})
    
     
    #Plotting
    fig, ax = plt.subplots(figsize=(12, 9))
    ax  = sns.barplot(x = 'Predictors', y = 'Coef', data=results, yerr= results['SEM'])    
    ax.set_xticklabels(results['Predictors'], rotation=-90)
    ax.set_ylabel('coef')
    ax.axhline(y=0, linestyle='--', color='black', linewidth=2)
    fig.suptitle ('GLM Biased Blocks')
    
    return results

def ibl_rt (psy_df):
    """
    INPUT dataframe with trials
    OUTPUT dataframe with RTs and signed contrast
    """
    psy_df['RT']  = psy_df['response_times'] -  psy_df['stimOn_times']
    psy_df = signed_contrast(psy_df)
    
    return psy_df
