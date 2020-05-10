#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:32:37 2020

@author: alex
"""

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Meta plots

def plot_choice_prob_per_animal(psy_df, mouse_name, save=True):
    psy_df_select =  psy_df.loc[psy_df['mouse_name']== mouse_name].copy()
    num_sessions = int(len(psy_df_select.loc[psy_df_select['mouse_name'] == mouse_name,\
                   'ses'].unique()))
    fig, ax = plt.subplots(int(np.ceil(num_sessions/4)),4,figsize=[10,10])
    for i in range(num_sessions):
        plt.sca(ax[i//4,i%4])
        try:
            plot_choice_prob_opto_block(psy_df_select, i, mouse_name, save =False, axes = True)
            ax[i//4,i%4].tick_params(axis='y', labelcolor='g')
            ax[i//4,i%4].set_ylabel('P(choice = right)', color='g')
            ax[i//4,i%4].set_xlabel('Trial', color='black') 
        except:
            continue
    plt.tight_layout()
    green_patch = mpatches.Patch(color='green', label='right opto block')
    black_patch  =  mpatches.Patch(color='black', label='non-op opto block')
    blue_patch  =  mpatches.Patch(color='blue', label='left opto block')
    plt.legend(handles=[green_patch, blue_patch, black_patch],
               loc = 'lower right')
    if save == True:
        plt.savefig(str(mouse_name) + 'p_choice.svg')
        plt.savefig(str(mouse_name) + 'p_choice.jpeg')
        
def plot_choice_per_animal(psy_df, mouse_name, save=True):
    psy_df_select =  psy_df.loc[psy_df['mouse_name']== mouse_name].copy()
    num_sessions = int(len(psy_df_select.loc[psy_df_select['mouse_name'] == mouse_name,\
                   'ses'].unique()))
    fig, ax = plt.subplots(int(np.ceil(num_sessions/4)),4,figsize=[10,10])
    for i in range(num_sessions):
        plt.sca(ax[i//4,i%4])
        try:
            plot_choice_opto_block(psy_df_select, i, mouse_name, save =False, axes = True)
            ax[i//4,i%4].tick_params(axis='y', labelcolor='g')
            ax[i//4,i%4].set_ylabel('P(choice = right)', color='g')
            ax[i//4,i%4].set_xlabel('Trial', color='black') 
        except:
            continue
    plt.tight_layout()
    green_patch = mpatches.Patch(color='green', label='right opto block')
    black_patch  =  mpatches.Patch(color='black', label='non-op opto block')
    blue_patch  =  mpatches.Patch(color='blue', label='left opto block')
    plt.legend(handles=[green_patch, blue_patch, black_patch],
               loc = 'lower right')
    if save == True:
        plt.savefig(str(mouse_name) + 'choice.svg')
        plt.savefig(str(mouse_name) + 'choice.jpeg')
        
        
def plot_q_per_animal(psy_df, mouse_name, save=True):
    psy_df_select =  psy_df.loc[psy_df['mouse_name']== mouse_name].copy()
    num_sessions = int(len(psy_df_select.loc[psy_df_select['mouse_name'] == mouse_name,\
                   'ses'].unique()))
    fig, ax = plt.subplots(int(np.ceil(num_sessions/4)),4,figsize=[10,10])
    for i in range(num_sessions):
        plt.sca(ax[i//4,i%4])
        try:
            plot_QR_QL_opto_block(output,psy_df)
            ax[i//4,i%4].tick_params(axis='y', labelcolor='g')
            ax[i//4,i%4].set_ylabel('P(choice = right)', color='g')
            ax[i//4,i%4].set_xlabel('Trial', color='black') 
        except:
            continue
    plt.tight_layout()
    green_patch = mpatches.Patch(color='green', label='right opto block')
    black_patch  =  mpatches.Patch(color='black', label='non-op opto block')
    blue_patch  =  mpatches.Patch(color='blue', label='left opto block')
    plt.legend(handles=[green_patch, blue_patch, black_patch],
               loc = 'lower right')
    if save == True:
        plt.savefig(str(mouse_name) + 'choice.svg')
        plt.savefig(str(mouse_name) + 'choice.jpeg')
        
        
        
        
# Individual plots

def plot_q_trial_whole_dataset(psy_df):
    psy_select = psy_df.copy()
    
    fig, ax = plt.subplots(2,2, figsize = [10,10])
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0,0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QR',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-ChR2')
    ax[0,0].set_xlabel('Trial in block')
    ax[0,0].set_ylabel('QR')
    
    plt.sca(ax[0,1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QL',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-ChR2')
    ax[0,1].set_xlabel('Trial in block')
    ax[0,1].set_ylabel('QL')
    
    plt.sca(ax[1,0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QR',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-NphR')
    ax[1,0].set_xlabel('Trial in block')
    ax[1,0].set_ylabel('QR')
    
    plt.sca(ax[1,1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QL',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-NpHR')
    ax[1,1].set_xlabel('Trial in block')
    ax[1,1].set_ylabel('QL')
    plt.tight_layout()
    
    plt.savefig('q_across_trials.svg')
    plt.savefig('q_across_trials.jpeg')


def plot_choice_prob_opto_block(psy_df, ses_number, mouse_name, save =False, 
                           axes = False):
    '''
    plot p choice right over trials
    Parameters
    ----------
    psy_df : dataframe with real data
    ses_number :number of  session (int)
    mouse_name : mouse name
    save : whether to save the figure
    Returns
    -------
    Figure with p choice over time, excludes firs 100 trials

    '''
    #
    # neutral block
    
    
    psy_df['right_block'] = np.nan
    psy_df['left_block'] = np.nan
    psy_df['right_block'] = (psy_df['opto_block'] == 'R')*1
    psy_df['left_block'] = (psy_df['opto_block'] == 'L')*1
    psy_df['non_opto_block'] = (psy_df['opto_block'] == 'non_opto')*1
    
    
    
    if axes ==  False:
        fig, ax1 = plt.subplots()
    
    psy_subset =  psy_df.loc[psy_df['mouse_name'] == mouse_name].copy()
    psy_subset = psy_subset.loc[psy_subset['ses'] == psy_subset['ses'].unique()[ses_number]]
    psy_subset['choice'] = psy_subset['choice']*-1
    psy_subset.loc[psy_subset['choice']==-1,'choice'] = 0
    p_choice = (psy_subset['choice'].cumsum() / (np.arange(psy_subset['choice'].count())+1))
    
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['left_block'], color = 'blue', alpha =0.35)

    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['right_block'], color = 'green', alpha =0.35)
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['non_opto_block'],
                     color ='black', alpha =0.35)
    # Probability of rightward choice
    plt.plot((np.arange(psy_subset['choice'].count())+1), 
             p_choice,
                      color = 'k')
    
    plt.xlim(25,psy_subset['choice'].count())
    plt.ylim(min(p_choice[100:])*0.9,max(p_choice[100:])*1.1)
    #plt.ylim(auto = True)
    if  axes ==  False:
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.set_ylabel('P(choice = right)', color='g')
        ax1.set_xlabel('Trial', color='black') 
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        green_patch = mpatches.Patch(color='green', label='right opto block')
        black_patch  =  mpatches.Patch(color='black', label='non-opto block')
        blue_patch  =  mpatches.Patch(color='blue', label='left opto block')
        plt.legend(handles=[green_patch, blue_patch, black_patch],
                   loc = 'lower right')
    plt.savefig('p_choice.svg')
    plt.savefig('p_choice.jpeg')




def plot_choice_prob_opto_block(psy_df, ses_number, mouse_name, save =False, 
                           axes = False):
    '''
    plot p choice right over trials
    Parameters
    ----------
    psy_df : dataframe with real data
    ses_number :number of  session (int)
    mouse_name : mouse name
    save : whether to save the figure
    Returns
    -------
    Figure with p choice over time, excludes firs 100 trials

    '''
    #
    # neutral block
     
    psy_df['right_block'] = np.nan
    psy_df['left_block'] = np.nan
    psy_df['right_block'] = (psy_df['opto_block'] == 'R')*1
    psy_df['left_block'] = (psy_df['opto_block'] == 'L')*1
    psy_df['non_opto_block'] = (psy_df['opto_block'] == 'non_opto')*1
    
    
    if axes ==  False:
        fig, ax1 = plt.subplots()
    
    psy_subset =  psy_df.loc[psy_df['mouse_name'] == mouse_name].copy()
    psy_subset = psy_subset.loc[psy_subset['ses'] == psy_subset['ses'].unique()[ses_number]]
    psy_subset['choice'] = psy_subset['choice']*-1
    psy_subset.loc[psy_subset['choice']==-1,'choice'] = 0
    p_choice = psy_subset['choice'].rolling(30).mean()
    
    #Calculate_rightchoice_probability_model
    psy_subset['pModelRight'] = 0
    psy_subset.loc[psy_subset['choice'] == 1, 'pModelRight'] =  \
        psy_subset.loc[psy_subset['choice'] == 1, 'pChoice'][:]#right choices are 1 know see 2 lines above, useful for other code
    psy_subset.loc[psy_subset['choice'] == 0, 'pModelRight'] =  \
        1 - psy_subset.loc[psy_subset['choice'] == 0, 'pChoice']#right choices are 1 know see 2 lines above, useful for other code
    
    p_choice_model = psy_subset['pModelRight'].rolling(30).mean()
    
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['left_block'], color = 'blue', alpha =0.35)

    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['right_block'], color = 'green', alpha =0.35)
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['non_opto_block'],
                     color ='black', alpha =0.35)
    # Probability of rightward choice
    plt.plot(np.arange(len(p_choice)), 
             p_choice, color = 'r')
    plt.plot(np.arange(len(p_choice_model)), 
             p_choice_model, color = 'purple')
    
    plt.xlim(25,psy_subset['choice'].count())
    plt.ylim(min(p_choice[100:])*0.9,max(p_choice[100:])*1.1)
    #plt.ylim(auto = True)
    if  axes ==  False:
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.set_ylabel('P(choice = right)', color='k')
        ax1.set_xlabel('Trial', color='black') 
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        green_patch = mpatches.Patch(color='green', label='right opto block')
        black_patch  =  mpatches.Patch(color='black', label='non-opto block')
        blue_patch  =  mpatches.Patch(color='blue', label='left opto block')
        red_patch  =  mpatches.Patch(color='red', label='Real Data')
        purple_patch  =  mpatches.Patch(color='purple', label='Model')
        plt.legend(handles=[green_patch, blue_patch, black_patch, red_patch, purple_patch],
                   loc = 'lower right')
    plt.savefig(mouse_name + str(ses_number) + 'p_choice.svg')
    plt.savefig(mouse_name + str(ses_number) + 'p_choice.jpeg')

def plot_choice_opto_block(psy_df, ses_number, mouse_name, save =False, 
                           axes = False):
    '''
    plot choices over trials
    Parameters
    ----------
    psy_df : dataframe with real data
    ses_number :number of  session (int)
    mouse_name : mouse name
    save : whether to save the figure
    Returns
    -------
    Figure with p choice over time, excludes firs 100 trials

    '''
    #
    # neutral block
    
    
    psy_df['right_block'] = np.nan
    psy_df['left_block'] = np.nan
    psy_df['right_block'] = (psy_df['opto_block'] == 'R')*1
    psy_df['left_block'] = (psy_df['opto_block'] == 'L')*1
    psy_df['non_opto_block'] = (psy_df['opto_block'] == 'non_opto')*1

    
    if axes ==  False:
        fig, ax1 = plt.subplots()
    
    psy_subset =  psy_df.loc[psy_df['mouse_name'] == mouse_name].copy()
    psy_subset = psy_subset.loc[psy_subset['ses'] == psy_subset['ses'].unique()[ses_number]]
    psy_subset['choice'] = psy_subset['choice']*-1
    
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), 
                     psy_subset['left_block'], psy_subset['left_block']*-1, 
                     color = 'blue', alpha =0.35)

    plt.fill_between((np.arange(psy_subset['choice'].count())+1), 
                     psy_subset['right_block'],
                     psy_subset['right_block']*-1 , 
                     color = 'green', alpha =0.35)
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), 
                     psy_subset['non_opto_block'],psy_subset['non_opto_block']*-1,
                     color ='black', alpha =0.35)
    # Probability of rightward choice
    plt.plot((np.arange(psy_subset['choice'].count())+1), 
             psy_subset['choice'].rolling(window=20).mean(),
                      color = 'k')
    
    plt.xlim(25,psy_subset['choice'].count())
    #plt.ylim(auto = True)
    if  axes ==  False:
        ax1.tick_params(axis='y', labelcolor='g')
        ax1.set_ylabel('Choice: L(-1), R(1)', color='g')
        ax1.set_xlabel('Trial', color='black') 
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        green_patch = mpatches.Patch(color='green', label='right opto block')
        black_patch  =  mpatches.Patch(color='black', label='non-opto block')
        blue_patch  =  mpatches.Patch(color='blue', label='left opto block')
        plt.legend(handles=[green_patch, blue_patch, black_patch],
                   loc = 'lower right')
    plt.savefig('choice.svg')
    plt.savefig('choice.jpeg')



def plot_QR_QL_opto_block(output,psy_df, mouse_name, ses_number):
    '''
    Parameters
    ----------
    output : Output from RunPOMDP
    psy_df : dataframe with real data

    Returns
    -------
    Figure  QLs across times

    '''
    #
    # neutral block
    psy_df['QR'] = output['QR']
    psy_df['QL'] = output['QL']
    psy_subset =  psy_df.loc[psy_df['mouse_name'] == mouse_name].copy()
    psy_subset = psy_subset.loc[psy_subset['ses'] == psy_subset['ses'].unique()[ses_number]]
    psy_subset['right_block'] = np.nan
    psy_subset['left_block'] = np.nan
    psy_subset['right_block'] = (psy_subset['opto_block'] == 'R')*1
    psy_subset['left_block'] = (psy_subset['opto_block'] == 'L')*1
    psy_subset['non_opto_block'] = (psy_subset['opto_block'] == 'non_opto')*1
    
    fig, ax1 = plt.subplots()
    
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), 
                     psy_subset['left_block'], psy_subset['left_block']*-1, 
                     color = 'blue', alpha =0.35)

    plt.fill_between((np.arange(psy_subset['choice'].count())+1), 
                     psy_subset['right_block'],
                     psy_subset['right_block']*-1 , 
                     color = 'green', alpha =0.35)
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), 
                     psy_subset['non_opto_block'],psy_subset['non_opto_block']*-1,
                     color ='black', alpha =0.35)

    sns.lineplot(data = psy_subset, x = psy_subset.index, 
                      y = psy_subset['QR'].rolling(window=20).mean(),
                      color = 'g')
    ax1.tick_params(axis='y', labelcolor='g')
    ax1.set_ylabel('QR', color='g')
    ax1.set_xlabel('Trial', color='black') 
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    sns.lineplot(data = psy_subset, x = psy_subset.index, 
                      y = psy_subset['QL'].rolling(window=20).mean(),
                      color = 'deepskyblue')
    ax2.tick_params(axis='y', labelcolor='deepskyblue')
    ax2.set_ylabel('QL', color='deepskyblue')  # we already handled the x-label with ax1
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    plt.xlim(9431,10400)
    green_patch = mpatches.Patch(color='green', label='right opto block')
    black_patch  =  mpatches.Patch(color='black', label='non-op opto block')
    blue_patch  =  mpatches.Patch(color='blue', label='left opto block')
    plt.legend(handles=[green_patch, blue_patch, black_patch],
               loc = 'lower right')
    plt.savefig('QL_QR.svg')
    plt.savefig('QL_QR.jpeg')


def plot_barplot_model_types(output_chr2, output_nphr, output_chr2_stay, output_nphr_stay):
    output_chr2['model'] = 'standard'
    output_nphr['model'] = 'standard'
    output_chr2['virus'] = 'chr2'
    output_nphr['virus'] = 'nphr'
    output_chr2_stay['model'] = 'standard + stay parameter'
    output_nphr_stay['model'] = 'standard + stay parameter'
    output_chr2_stay['virus'] = 'chr2'
    output_nphr_stay['virus'] = 'nphr'
    outputs = pd.concat([output_chr2, output_nphr, output_chr2_stay, output_nphr_stay], sort=False)
    fig, ax =  plt.subplots(1,2 , figsize = (10,10))
    plt.sca(ax[0])
    sns.pointplot(data = outputs.loc[outputs['virus'] == 'chr2'], x = 'model', 
                  y = 'pChoice', ci =68)
    ax[0].set_xlabel('Model type')
    ax[0].set_ylabel('Mean accuracy')
    plt.ylim(0,1)
    ax[0].set_title('Model comparison ChR2')
    plt.sca(ax[1])
    sns.pointplot(data = outputs.loc[outputs['virus'] == 'nphr'], x = 'model', 
                  y = 'pChoice', ci =68)
    ax[1].set_xlabel('Model type')
    ax[1].set_ylabel('Mean accuracy')
    ax[1].set_title('Model comparison NpHR')
    plt.ylim(0,1)
    
    plt.savefig('model_performance.svg')
    plt.savefig('model_performance.jpeg')





def plot_action_opto_block(output,psy_df):
    '''
    Parameters
    ----------
    output : Output from runPOMDP
    psy_df : dataframe with real data

    Returns
    -------
    Figure model vs real data.

    '''
    psy_df = psy_df.copy()
    psy_df['action'] = output['action'].copy()
    psy_df['right_choice'] = psy_df['choice'] == -1
    
    ax = sns.lineplot(data = psy_df, x = 'signed_contrasts', y = 'action',
                      hue = 'opto_block', ci=68, legend=False)
    ax.lines[0].set_linestyle("--")
    ax.lines[1].set_linestyle("--")
    ax.lines[2].set_linestyle("--")
    psy_df.loc[psy_df['choice'] == -1, 'right_choice'] = True
    sns.lineplot(data = psy_df, x = 'signed_contrasts', y = 'right_choice', 
                 hue = 'opto_block', ci=68)
    ax.set_ylabel('Fraction of Right Choices')
    
    
def plot_choice_trial_whole_dataset(psy_df):
    psy_select = psy_df.copy()
    psy_select['choice'] = psy_select['choice']*-1
    
    fig, ax = plt.subplots(1,2, sharey=True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'choice',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('Choice: Left(-1) Right(1)')
    plt.sca(ax[1])
    psy_nphr = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_nphr, x = 'trial_within_block', y = 'choice',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    ax[1].set_xlabel('Trial in block')
    plt.title('VTA-NpHR')
    

