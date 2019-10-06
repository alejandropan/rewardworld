#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:06:23 2019
Need to reinstall module, at the moment I need to be in npy2pd module, same for basic plots
@author: ibladmin
"""
from npy2pd import *
from basic_plots import *
from opto_plots import *
from glm import *
import numpy as np
import pandas as pd


#Input folder with raw npy files
psy_raw = load_data('/Volumes/witten/Alex/server_backup/Subjects_personal_project/opto_blocks/')

#For non random block
#Only for random blocks
if psy_raw.isnull().values.any():
    psy_raw  = psy_raw.dropna()
    print ('Warning: sessions deleted due to entire variables with NaN')

psy_df  = unpack(psy_raw)

#Stable block assigner
def opto_block_assigner (psy_df):
    psy_df['opto_block'] = np.nan
    psy_df.loc[(psy_df['opto_probability_left'] == 1), 'opto_block'] = 'L'
    psy_df.loc[(psy_df['opto_probability_left'] == 0), 'opto_block'] = 'R'
    psy_df.loc[(psy_df['opto_probability_left'] == -1), 'opto_block'] = 'non_opto'
    return psy_df





#repair spaces
psy_df = opto_block_assigner (psy_df)
psy_df.loc[(psy_df['hem_stim']== ' L '), 'hem_stim']= 'L'

#Shift opto to slice through next trial
for mouse in psy_df['mouse_name'].unique():
    for day in psy_df.loc[psy_df['mouse_name']== mouse,'ses'].unique():
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_opto'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'opto.npy'].shift(periods=1)

for mouse in psy_df['mouse_name'].unique():
    for day in psy_df.loc[psy_df['mouse_name']== mouse,'ses'].unique():
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'previous_choice'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'choice'].shift(periods=1)

for mouse in psy_df['mouse_name'].unique():
    for day in psy_df.loc[psy_df['mouse_name']== mouse,'ses'].unique():
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_win'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'feedbackType'].shift(periods=1)
        
#1st Plot After laser trials across groups
#Grouping variables
block_variable = 'opto_block'
blocks = ['L', 'R', 'non_opto']
        

#Plot across different trial history groups
long = psychometric_summary_opto_blocks(psy_df, block_variable, blocks)
psy_df_short = shorten_df(psy_df, 0,200)
first = psychometric_summary_opto_blocks(psy_df_short , block_variable, blocks)
psy_df_last = shorten_df(psy_df, -200,-1)
last = psychometric_summary_opto_blocks(psy_df_last , block_variable, blocks)

#Save figs
long.savefig('long.pdf')
first.savefig('first.pdf')
last.savefig('last.pdf')


#Plot glm with opto as a regressor for the different stimulation types

glms = opto_laser_glm(psy_df)
glms.savefig('glms.pdf')



#Plotting functions

summary_opto(psy_df.loc[(psy_df['hem_stim']=='B')] , block_variable, blocks)



def summary_opto(psy_df , block_variable, blocks):
    """
    INPUTS:
    block_variable: string with name of block variable 1,  in this case L or R block
    conditions:  List of potential hemisphere
    viruses: List of viruses
    blocks: Has to be [L,R] in this order, it defines the laser on and off
    OUTPUTS:
    Figure with psychometrics (Info on psychometric measures):
    pars  : bias    = pars[0]
   threshold    = pars[1]
   gamma1    = pars[2]
   gamma2    = pars[3]
    """
    #Clarify variables:
    conditions  = psy_df['hem_stim'].unique()
    mice = psy_df['mouse_name'].unique()
    viruses =  psy_df['virus'].unique()
    figure,ax = plt.subplots(1,2, figsize=(15,5))
    #psy_measures_rows = np.nan(len(blocks)*len(blocks2)*len(conditions)*len(mice))
    #Plot summaries divided by hem stimulated and virus
    for v, virus in enumerate(viruses):
        for c , hem in enumerate(conditions):
            plt.sca(ax[v])
            sns.set()
            colors =['blue', 'green']
            if len(blocks) > 2:
                colors =['blue', 'green', 'black']
            for j,i in enumerate(blocks):
                    psy_df_block  = psy_df.loc[(psy_df['hem_stim'] == hem) & (psy_df['virus'] == virus) & (psy_df[block_variable] == i)] 
                    pars,  L  =  ibl_psychometric (psy_df_block)                    
                    plt.plot(np.arange(-25,25), psy.erf_psycho_2gammas( pars, np.arange(-25,25)), linewidth=2,\
                             color = colors[j])
                    sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", \
                                 linewidth=0, linestyle='None', mew=0.5,marker='.',
                                 color = colors[j],  ci=68, data= psy_df_block)
                    
            ax[v].set_xlim([-25,25])
            ax[v].set_title(virus +' '+ hem)
            ax[v].set_ylabel('Fraction of CW choices')
            ax[v].set_xlabel('Signed contrast')
            green_patch = mpatches.Patch(color='green', label='Stim right = Laser on')
            blue_patch  =  mpatches.Patch(color='blue', label='Stim left = Laser on')
            plt.legend(handles=[green_patch, blue_patch], loc = 'lower right')
        
    return figure 

def summary_opto_permouse(psy_df , block_variable, blocks):
    """
    INPUTS:
    block_variable: string with name of block variable 1,  in this case L or R block
    conditions:  List of potential hemisphere
    viruses: List of viruses
    blocks: Has to be [L,R] in this order, it defines the laser on and off
    OUTPUTS:
    Figure with psychometrics (Info on psychometric measures):
    pars  : bias    = pars[0]
   threshold    = pars[1]
   gamma1    = pars[2]
   gamma2    = pars[3]
    """
    #Clarify variables:
    conditions  = psy_df['hem_stim'].unique()
    viruses =  psy_df['virus'].unique()
    figure,ax = plt.subplots(2,3, figsize=(15,15))
    #psy_measures_rows = np.nan(len(blocks)*len(blocks2)*len(conditions)*len(mice))
    psy_measures =  pd.DataFrame(columns = ['mouse_name','virus','laser_on','probabilityLeft','conditions','bias','threshold', 'gamma1', 'gamma2'  ])
    #Plot summaries divided by hem stimulated and virus
    for v , virus in enumerate(viruses):
        for m, mouse in enumerate(psy_df.loc[psy_df['virus'] == virus, 'mouse_name'].unique()):
        
            plt.sca(ax[v,m])
            sns.set()
            colors =['blue', 'green']
            if len(blocks) > 2:
                colors =['blue', 'green', 'black']
            for j,i in enumerate(blocks):
                    psy_df_block  = psy_df.loc[(psy_df['hem_stim'] == hem) & (psy_df['mouse_name'] == mouse) & (psy_df[block_variable] == i)] 
                    pars,  L  =  ibl_psychometric (psy_df_block)
                    plt.plot(np.arange(-25,25), psy.erf_psycho_2gammas( pars, np.arange(-25,25)), linewidth=2,\
                             color = colors[j])
                    sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", \
                                 linewidth=0, linestyle='None', mew=0.5,marker='.',
                                 color = colors[j],  ci=68, data= psy_df_block)
                    psy_measures = psy_measures.append({'mouse_name': mouse, 'virus':virus, 'block': i,\
                                                        'conditions':hem,'threshold': pars[1],'bias':pars[0],\
                                                       'gamma1':pars[2],'gamma2':pars[3]}, ignore_index=True)
            ax[v,m].set_xlim([-25,25])
            ax[v,m].set_title(mouse +' ' + virus +' '+ hem)
            ax[v,m].set_ylabel('Fraction of CW choices')
            ax[v,m].set_xlabel('Signed contrast')
            green_patch = mpatches.Patch(color='green', label='Stim right = Laser on')
            blue_patch  =  mpatches.Patch(color='blue', label='Stim left = Laser on')
            plt.legend(handles=[green_patch, blue_patch], loc = 'lower right')
        
    return figure 


def choice_at_zero_2blocks (psy_df):
    """ Calculate choice at 0 contrast across blocks
    INPUTS:
    psy_df,  general dataframe with all the information about each trials
    block_variable, dividing blovk variable (E.g opto stim block)
    blocks, identity of the blocks in block variable
    """
    
    #  Reduce dataframe to trials with 0
    psy_0 = \
    psy_df.loc[(psy_df['contrastRight'] == 0) | (psy_df['contrastLeft'] == 0 )]
    
    #  Calculate percentage of right choices
    right_choices = psy_0['choice']== -1
    psy_0.loc[:,'right_choices'] = right_choices
    
    #  Start plotting
    sns.set()
    figure, ax  = plt.subplots(2,2, figsize = [6,10])
    sns.barplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[psy_0['virus'] == 'chr2'],\
                facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["b", "g"], ax = ax[0,0])
    sns.lineplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[psy_0['virus'] == 'chr2'], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,\
                linewidth=2,  alpha = 0.75, ax = ax[0,0])
    ax[0,0].set_ylim([0.2,0.6])
    ax[0,0].legend_.remove()
    ax[0,0].set_xlabel('Opto block')
    ax[0,0].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[psy_0['virus'] == 'nphr'] ,linewidth=2.5, \
                facecolor=(1, 1, 1, 0),
                edgecolor=["b", "g"],  ax = ax[0,1])
    sns.lineplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[psy_0['virus'] == 'nphr'], hue = 'mouse_name', \
                ci=None, palette = "Greys" ,linewidth=2,ax = ax[0,1], \
                alpha = 0.75)
    ax[0,1].set_ylim([0.2,0.6])
    ax[0,1].legend_.remove()
    ax[0,1].set_xlabel('Opto block')
    ax[0,1].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'chr2') & \
                                 (psy_0['previous_choice'] == -1)],\
                                 facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["k", "k"], ax = ax[1,0])
    sns.lineplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'chr2') & \
                                 (psy_0['previous_choice'] == -1)], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,linewidth=2, \
                ax = ax[1,0], alpha = 0.75)
    ax[1,0].set_ylim([0.2,0.7])
    ax[1,0].legend_.remove()
    ax[1,0].set_xlabel('Laser ON on previous trial')
    ax[1,0].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr') & \
                                 (psy_0['previous_choice'] == -1)],\
                                 facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["k", "k"], ax = ax[1,1])
    sns.lineplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr') & \
                                 (psy_0['previous_choice'] == -1)], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,linewidth=2, \
                ax = ax[1,1], alpha = 0.75)
    ax[1,1].set_ylim([0.2,0.7])
    ax[1,1].legend_.remove()
    ax[1,1].set_xlabel('Laser ON on previous trial')
    ax[1,1].set_ylabel('% CCW (Right) choices')
    plt.tight_layout()
    
def choice_at_zero_3blocks (psy_df):
    """ Calculate choice at 0 contrast across blocks
    INPUTS:
    psy_df,  general dataframe with all the information about each trials
    block_variable, dividing blovk variable (E.g opto stim block)
    blocks, identity of the blocks in block variable
    """
    
    #  Reduce dataframe to trials with 0
    psy_0 = \
    psy_df.loc[(psy_df['contrastRight'] == 0) | (psy_df['contrastLeft'] == 0 )]
    
    #  Calculate percentage of right choices
    right_choices = psy_0['choice']== -1
    psy_0.loc[:,'right_choices'] = right_choices
    
    #  Start plotting
    sns.set()
    figure, ax  = plt.subplots(2,2, figsize = [6,10])
    sns.barplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[psy_0['virus'] == 'chr2'],\
                facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["b","g", 'k'], ax = ax[0,0])
    sns.lineplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[psy_0['virus'] == 'chr2'], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,\
                linewidth=2,  alpha = 0.75, ax = ax[0,0])
    ax[0,0].set_ylim([0.2,0.6])
    ax[0,0].legend_.remove()
    ax[0,0].set_xlabel('Opto block')
    ax[0,0].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[psy_0['virus'] == 'nphr'] ,linewidth=2.5, \
                facecolor=(1, 1, 1, 0),
                edgecolor=["b", "g", 'k'],  ax = ax[0,1])
    sns.lineplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[psy_0['virus'] == 'nphr'], hue = 'mouse_name', \
                ci=None, palette = "Greys" ,linewidth=2,ax = ax[0,1], \
                alpha = 0.75)
    ax[0,1].set_ylim([0.2,0.6])
    ax[0,1].legend_.remove()
    ax[0,1].set_xlabel('Opto block')
    ax[0,1].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'chr2') & \
                                 (psy_0['previous_choice'] == -1)],\
                                 facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["b", "g", 'k'], ax = ax[1,0])
    sns.lineplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'chr2') & \
                                 (psy_0['previous_choice'] == -1)], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,linewidth=2, \
                ax = ax[1,0], alpha = 0.75)
    ax[1,0].set_ylim([0.2,0.7])
    ax[1,0].legend_.remove()
    ax[1,0].set_xlabel('Laser ON on previous trial')
    ax[1,0].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr') & \
                                 (psy_0['previous_choice'] == -1)],\
                                 facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["b", "g", 'k'], ax = ax[1,1])
    sns.lineplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr') & \
                                 (psy_0['previous_choice'] == -1)], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,linewidth=2, \
                ax = ax[1,1], alpha = 0.75)
    ax[1,1].set_ylim([0.2,0.7])
    ax[1,1].legend_.remove()
    ax[1,1].set_xlabel('Laser ON on previous trial')
    ax[1,1].set_ylabel('% CCW (Right) choices')
    plt.tight_layout()
    


def opto_chronometric(psy_df):
    """
    DESCRIPTION:  Plot chronometric for laser on and off
    INPUT: psy_df dataframe with trials
    OUTPUT: figure with chronometric for laser on / off
    """
    psy_df  = ibl_rt(psy_df)
    conditions  = psy_df['hem_stim'].unique()
    mice = psy_df['mouse_name'].unique()
    viruses =  psy_df['virus'].unique()
    figure,ax = plt.subplots(1,2, figsize=(15,5))

    for v, virus in enumerate(viruses):
        for c , hem in enumerate(conditions):
            plt.sca(ax[v])
            sns.set()
            colors =['blue', 'green']
            if len(blocks) > 2:
                colors =['blue', 'green']
            for j,i in enumerate(blocks):
                    psy_df_block  = psy_df.loc[(psy_df['hem_stim'] == hem) & \
                                               (psy_df['virus'] == virus) & \
                                               (psy_df[block_variable] == i)] 
                    sns.lineplot(x='signed_contrasts', y='RT', marker = 'o',\
                                 err_style="bars", ci=68, style='after_opto',\
                                 color = colors[j], data= psy_df_block)
                    
            ax[v].set_xlim([-25,25])
            ax[v].set_title(virus +' '+ hem)
            ax[v].set_ylabel('Reaction Times (s)')
            ax[v].set_xlabel('Signed contrast')
            green_patch = mpatches.Patch(color='green', \
                                         label='Stim right = Laser on')
            blue_patch  =  mpatches.Patch(color='blue', \
                                          label='Stim left = Laser on')
            dashed  =  plt.Line2D([0], [0], color='black', lw= 1.5, \
                                 label='Laser ON t-1', ls = '--')
            solid  = plt.Line2D([0], [0], color='black', lw=1.5, \
                                label='Laser OFF t-1', ls = '-')
            
            plt.legend(handles=[green_patch, blue_patch, dashed, solid], \
                       loc = 'lower right')
            
         
def opto_chronometric_permouse (psy_df):
    psy_df  = ibl_rt(psy_df)
    conditions  = psy_df['hem_stim'].unique()
    viruses =  psy_df['virus'].unique()
    figure,ax = plt.subplots(2,3, figsize=(15,15))

    #Plot summaries divided by hem stimulated and virus
    for v , virus in enumerate(viruses):
        for m, mouse in enumerate(psy_df.loc[psy_df['virus'] == virus, \
                                             'mouse_name'].unique()):
        
            plt.sca(ax[v,m])
            sns.set()
            colors =['blue', 'green']
            if len(blocks) > 2:
                colors =['blue', 'green', 'black']
            for j,i in enumerate(blocks):
                    psy_df_block  = psy_df.loc[(psy_df['hem_stim'] == hem) & \
                                    (psy_df['mouse_name'] == mouse) & \
                                    (psy_df[block_variable] == i)] 
                    sns.lineplot(x='signed_contrasts', y='RT', \
                                 err_style="bars", \
                                 linestyle='None',marker='o',
                                 color = colors[j], ci=68, style='after_opto',\
                                 data= psy_df_block)
   
            ax[v,m].set_xlim([-25,25])
            ax[v,m].set_title(mouse +' ' + virus +' '+ hem)
            ax[v,m].set_ylabel('Fraction of CW choices')
            ax[v,m].set_xlabel('Signed contrast')
            green_patch = mpatches.Patch(color='green', \
                                         label='Stim right = Laser on')
            blue_patch  =  mpatches.Patch(color='blue', \
                                          label='Stim left = Laser on')
            dashed  =  plt.Line2D([0], [0], color='black', lw= 1.5, \
                                 label='Laser ON t-1', ls = '--')
            solid  = plt.Line2D([0], [0], color='black', lw=1.5, \
                                label='Laser OFF t-1', ls = '-')
            plt.legend(handles=[green_patch, blue_patch, \
                                dashed, solid], loc = 'lower right')
        
    return figure 

def block_qc(psy_df):
    """
    INPUT: dataframe with all trial information
    OUTPUT: 
    percetange_left : percetange of left stimuli
    percentage_left_0: percentage of 0 that are rewarded
    with a left choice
    """
    
    percentage_left = pd.isnull(psy_df['contrastRight']).sum() / \
    pd.isnull(psy_df['contrastRight']).count()
    
    percentage_left_0 = \
    psy_df.loc[psy_df['contrastLeft'] == 0, 'choice'].count() / \
    psy_df.loc[(psy_df['contrastLeft'] == 0) | \
               (psy_df['contrastRight'] == 0), 'choice'].count()
    
    return percentage_left, percentage_left_0
    
    
    
    


    