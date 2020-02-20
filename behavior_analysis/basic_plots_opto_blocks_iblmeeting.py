#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:06:23 2019
Need to reinstall module, at the moment I need to be in npy2pd module, 
same for basic plots
@author: ibladmin
"""
from npy2pd import *
from basic_plots import *
from opto_plots import *
from glm import *
import numpy as np
import pandas as pd
import seaborn as sns

#Input folder with raw npy files
psy_raw = load_data('/Volumes/witten/Alex/server_backup/Subjects_personal_project/opto_blocks_random/')

#For non random block
#Only for random blocks
if psy_raw.isnull().values.any():
    psy_raw  = psy_raw.dropna()
    print ('Warning: sessions deleted due to entire variables with NaN')

psy_df  = unpack(psy_raw)


#Remove the last 100 trials from every session

def remove_last100(psy_df):
    psy_df_new  = pd.DataFrame(columns = psy_df.columns)
    for mouse in psy_df['mouse_name'].unique():
            for ses in psy_df.loc[(psy_df['mouse_name']==mouse), 'ses'].unique():
                session =  psy_df.loc[(psy_df['mouse_name']==mouse) & \
                                  (psy_df['ses']==ses)]
                new_session = session.iloc[:-100]  
                psy_df_new = psy_df_new.append(new_session, ignore_index = True)
    return psy_df_new


psy_df  = remove_last100(psy_df)
#Stable block assigner
def opto_block_assigner (psy_df):
    psy_df['opto_block'] = np.nan
    psy_df.loc[(psy_df['opto_probability_left'] == 1), 'opto_block'] = 'L'
    psy_df.loc[(psy_df['opto_probability_left'] == 0), 'opto_block'] = 'R'
    psy_df.loc[(psy_df['opto_probability_left'] == -1), 'opto_block'] = 'non_opto'
    return psy_df

'''
 In the task there is no unrewarded opto trials, even if opto.npy is 1, if the 
state machine does not reach the state 'reward', the laser never goes off. Therfore:
'''
#  Correct for lack of unrewarded laser

psy_df.loc[(psy_df['opto.npy'] == 1) & (psy_df['feedbackType'] == -1), \
           'opto.npy'] = 0


#repair spaces
psy_df = opto_block_assigner (psy_df)
psy_df.loc[(psy_df['hem_stim']== ' L '), 'hem_stim']= 'L'

#Shift opto to slice through next trial
for mouse in psy_df['mouse_name'].unique():
    for day in psy_df.loc[psy_df['mouse_name']== mouse,'ses'].unique():
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_opto'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'opto.npy'].shift(periods=1)
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_opto_2'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'opto.npy'].shift(periods=2)
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_opto_3'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'opto.npy'].shift(periods=3)
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_opto_4'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'opto.npy'].shift(periods=4)
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'previous_choice'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'choice'].shift(periods=1)
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_win'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'feedbackType'].shift(periods=1)
        


#Check two trials after opto vs non-opto
    
psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day) & \
                   (psy_df['after_opto']== 1) & \
                   (psy_df['opto.npy']== 1),'after_double_opto'] = 1
           
psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day) & \
                   (psy_df['after_opto']== 0) & \
                   (psy_df['opto.npy']== 0),'after_double_nonopto'] = 1
        


#1st Plot After laser trials across groups
#Grouping variables
block_variable = 'opto_block'
blocks = ['L', 'R',]
        

#Plot across different trial history groups
long = psychometric_summary_opto_blocks(psy_df, block_variable, blocks)
psy_df_short = shorten_df(psy_df, 0,400)
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

summary_opto(psy_df.loc[(psy_df['after_opto']==1)] , block_variable, blocks)



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
            sns.set_style('ticks')
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
                    
            ax[v].set_xlim([-30,30])
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
    psy_measures =  pd.DataFrame(columns = ['mouse_name','virus','laser_on','probabilityLeft','bias','threshold', 'gamma1', 'gamma2'  ])
    #Plot summaries divided by hem stimulated and virus
    for v , virus in enumerate(viruses):
        for m, mouse in enumerate(psy_df.loc[psy_df['virus'] == virus, 'mouse_name'].unique()):
        
            plt.sca(ax[v,m])
            sns.set()
            colors =['blue', 'green']
            if len(blocks) > 2:
                colors =['blue', 'green', 'black']
            for j,i in enumerate(blocks):
                    psy_df_block  = psy_df.loc[ (psy_df['mouse_name'] == mouse) & (psy_df[block_variable] == i)] 
                    pars,  L  =  ibl_psychometric (psy_df_block)
                    plt.plot(np.arange(-25,25), psy.erf_psycho_2gammas( pars, np.arange(-25,25)), linewidth=2,\
                             color = colors[j])
                    sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", \
                                 linewidth=0, linestyle='None', mew=0.5,marker='.',
                                 color = colors[j],  ci=68, data= psy_df_block)
                    psy_measures = psy_measures.append({'mouse_name': mouse, 'virus':virus, 'block': i,\
                                                        'threshold': pars[1],'bias':pars[0],\
                                                       'gamma1':pars[2],'gamma2':pars[3]}, ignore_index=True)
            ax[v,m].set_xlim([-25,25])
            ax[v,m].set_title(mouse +' ' + virus +' ')
            ax[v,m].set_ylabel('Fraction of CW choices')
            ax[v,m].set_xlabel('Signed contrast')
            green_patch = mpatches.Patch(color='green', label='Stim right = Laser on')
            blue_patch  =  mpatches.Patch(color='blue', label='Stim left = Laser on')
            plt.legend(handles=[green_patch, blue_patch], loc = 'lower right')
        
    return figure, psy_measures


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
    psy_0.loc[:,'right_choices'] = 0
    psy_0.loc[psy_0['choice'] == -1,'right_choices'] = 1
    
    #  Start plotting
    sns.set_style('ticks')
    figure, ax  = plt.subplots(2,2, figsize = [6,10])
    sns.barplot( x = 'opto_block', y = 'right_choices', \
                data = (psy_0.loc[(psy_0['virus'] == 'chr2')]),\
                facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["b", "g"], ax = ax[0,0], \
                order= ['L','R'])
    sns.lineplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'chr2')], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,\
                linewidth=2,  alpha = 0.75, ax = ax[0,0])
    #ax[0,0].set_ylim([0,0.6])
    ax[0,0].legend_.remove()
    ax[0,0].set_xlabel('Opto block')
    ax[0,0].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr')] ,linewidth=2.5, \
                facecolor=(1, 1, 1, 0),
                edgecolor=["b", "g"],  ax = ax[0,1], order= ['L','R'])
    sns.lineplot( x = 'opto_block', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr')], hue = 'mouse_name', \
                ci=None, palette = "Greys" ,linewidth=2,ax = ax[0,1], \
                alpha = 0.75)
    ax[0,1].set_ylim([0,0.6])
    ax[0,1].legend_.remove()
    ax[0,1].set_xlabel('Opto block')
    ax[0,1].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'chr2') & \
                                 (psy_0['after_win'] == 1) & \
                                 (psy_0['previous_choice'] == -1)],\
                                 facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["k", "k"], ax = ax[1,0])
    sns.lineplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'chr2') & \
                                 (psy_0['after_win'] == 1) & \
                                 (psy_0['previous_choice'] == -1)], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,linewidth=2, \
                ax = ax[1,0], alpha = 0.75)
    ax[1,0].set_ylim([0,0.7])
    ax[1,0].legend_.remove()
    ax[1,0].set_xlabel('Laser ON on previous trial')
    ax[1,0].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr') & \
                                 (psy_0['after_win'] == 1) & \
                                 (psy_0['previous_choice'] == -1)],\
                                 facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["k", "k"], ax = ax[1,1])
    sns.lineplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr') & \
                                 (psy_0['after_win'] == 1) &\
                                 (psy_0['previous_choice'] == -1)], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,linewidth=2, \
                ax = ax[1,1], alpha = 0.75)
    ax[1,1].set_ylim([0,0.7])
    ax[1,1].legend_.remove()
    ax[1,1].set_xlabel('Laser ON on previous trial')
    ax[1,1].set_ylabel('% CCW (Right) choices')
    plt.tight_layout()
    
    return figure
    
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
                linewidth=2.5, edgecolor=["b",'k', "g"], ax = ax[0,0])
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
                edgecolor=["b",  'k', "g"],  ax = ax[0,1])
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
                                 (psy_0['previous_choice'] == -1)& (psy_0['opto_block'] != 'non_opto')],\
                                 facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["b",  'k', "g"], ax = ax[1,0])
    sns.lineplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'chr2') & \
                                 (psy_0['previous_choice'] == -1) & (psy_0['opto_block'] != 'non_opto')], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,linewidth=2, \
                ax = ax[1,0], alpha = 0.75)
    ax[1,0].set_ylim([0.2,0.7])
    ax[1,0].legend_.remove()
    ax[1,0].set_xlabel('Laser ON on previous trial')
    ax[1,0].set_ylabel('% CCW (Right) choices')
    
    sns.barplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr') & \
                                 (psy_0['previous_choice'] == -1)& (psy_0['opto_block'] != 'non_opto')],\
                                 facecolor=(1, 1, 1, 0),\
                linewidth=2.5, edgecolor=["b",  'k', "g"], ax = ax[1,1])
    sns.lineplot( x = 'after_opto', y = 'right_choices', \
                data = psy_0.loc[(psy_0['virus'] == 'nphr') & \
                                 (psy_0['previous_choice'] == -1)& (psy_0['opto_block'] != 'non_opto')], \
                hue = 'mouse_name', ci=None, palette = "Greys" ,linewidth=2, \
                ax = ax[1,1], alpha = 0.75)
    ax[1,1].set_ylim([0.2,0.7])
    ax[1,1].legend_.remove()
    ax[1,1].set_xlabel('Laser ON on previous trial')
    ax[1,1].set_ylabel('% CCW (Right) choices')
    plt.tight_layout()
    


def opto_chronometric(psy_df, blocks):
    """
    DESCRIPTION:  Plot chronometric for laser on and off
    INPUT: psy_df dataframe with trials
    OUTPUT: figure with chronometric for laser on / off
    """
    if not 'signed_contrasts' in psy_df:
        psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
        psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
        psy_df.loc[:,'signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])
    
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
                colors =['blue', 'green', 'black']
            for j,i in enumerate(blocks):
                
                    if i == 'non_opto':
                        psy_df_block  = psy_df.loc[(psy_df['hem_stim'] == hem) & \
                                               (psy_df['virus'] == virus) & \
                                               (psy_df[block_variable] == i)] 
                        sns.lineplot(x='signed_contrasts', y='RT', marker = 'o',\
                                 err_style="bars", ci=68, \
                                 color = colors[j], data= psy_df_block)
                    else: 
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
            black_patch = mpatches.Patch(color='black', \
                                         label='Non-opto block')
            green_patch = mpatches.Patch(color='green', \
                                         label='Stim right = Laser on')
            blue_patch  =  mpatches.Patch(color='blue', \
                                          label='Stim left = Laser on')
            dashed  =  plt.Line2D([0], [0], color='black', lw= 1.5, \
                                 label='Laser ON t-1', ls = '--')
            solid  = plt.Line2D([0], [0], color='black', lw=1.5, \
                                label='Laser OFF t-1', ls = '-')
            
            plt.legend(handles=[black_patch, green_patch, blue_patch, dashed, solid], \
                       loc = 'lower right')
            
            return figure
            
         
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
                    psy_df_block  = psy_df.loc[ \
                                    (psy_df['mouse_name'] == mouse) & \
                                    (psy_df[block_variable] == i)] 
                    if i == 'non_opto' :
                        sns.lineplot(x='signed_contrasts', y='RT', \
                                 err_style="bars", \
                                 linestyle='None',marker='o',
                                 color = colors[j], ci=68,\
                                 data= psy_df_block)
                    else:    
                        sns.lineplot(x='signed_contrasts', y='RT', \
                                     err_style="bars", \
                                     linestyle='None',marker='o',
                                     color = colors[j], ci=68, style='after_opto',\
                                     data= psy_df_block)
   
            ax[v,m].set_xlim([-25,25])
            ax[v,m].set_title(mouse +' ' + virus +' ')
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


def opto_laser_glm(psy_df_global):
    viruses =  psy_df_global['virus'].unique()
    figure,ax = plt.subplots(5,3, figsize=(24,80))
    for v, virus in enumerate(viruses):
        conditions  = psy_df_global.loc[(psy_df_global['virus']== virus),'hem_stim'].unique()
        for c , hem in enumerate(conditions): #(conditions)
            psy_df = psy_df_global.loc[(psy_df_global['virus']== virus) & \
                                       (psy_df_global['hem_stim']== hem)]
            plt.sca(ax[v,c])
            
            mouse_result, mouse_r2  = glm_logit_opto(psy_df, sex_diff = False)
                
            mouse_result  =  pd.DataFrame({"Predictors": mouse_result.model.exog_names , "Coef" : mouse_result.params.values,\
                              "SEM": mouse_result.bse.values, "Significant": mouse_result.pvalues < 0.05/len(mouse_result.model.exog_names)})
    
            #Drop current evidence
            mouse_result = mouse_result.iloc[2:]
                
            #Plotting
                
            ax[v,c]  = sns.barplot(x = 'Predictors', y = 'Coef', \
              data=mouse_result, yerr= mouse_result['SEM'])    
            ax[v,c].set_xticklabels(mouse_result['Predictors'], rotation=-90)
            ax[v,c].set_ylabel('coef')
            ax[v,c].axhline(y=0, linestyle='--', color='black', linewidth=2)
            ax[v,c].set_title(virus +' '+ hem)
            
            
            #Have to cahnge glm function so that it has the optopn to run normal glm only on opto.npy trials
            for i,mouse in enumerate(psy_df['mouse_name'].unique()):
                mouse_result, mouse_r2  = glm_logit_opto(psy_df.loc[(psy_df['mouse_name']==mouse)], sex_diff = False)
                
                mouse_result  =  pd.DataFrame({"Predictors": mouse_result.model.exog_names , "Coef" : mouse_result.params.values,\
                              "SEM": mouse_result.bse.values, "Significant": mouse_result.pvalues < 0.05/len(mouse_result.model.exog_names)})
    
                #Drop current evidence
                mouse_result = mouse_result.iloc[2:]    
                
                #Plotting
                plt.sca(ax[v+i+2,c])
                ax[v+i+2,c]  = sns.barplot(x = 'Predictors', y = 'Coef', data=mouse_result, yerr= mouse_result['SEM'])    
                ax[v+i+2,c].set_xticklabels(mouse_result['Predictors'], rotation=-90)
                ax[v+i+2,c].set_ylabel('coef')
                ax[v+i+2,c].axhline(y=0, linestyle='--', color='black', linewidth=2)
                ax[v+i+2,c].set_title(virus +' '+mouse +' '+ hem)
                
                
    return figure




def block_qc(psy_df):
    """
    INPUT: dataframe with all trial information
    OUTPUT: 
    percetange_left : percetange of left stimuli
    percentage_left_0: percentage of 0 that are rewarded
    with a left choice
    unrewarded_opto
    """
    
    percentage_left = pd.isnull(psy_df['contrastRight']).sum() / \
    pd.isnull(psy_df['contrastRight']).count()
    
    percentage_left_0 = \
    psy_df.loc[psy_df['contrastLeft'] == 0, 'choice'].count() / \
    psy_df.loc[(psy_df['contrastLeft'] == 0) | \
               (psy_df['contrastRight'] == 0), 'choice'].count()
    
    unrewarded_opto = psy_df.loc[(psy_df['opto.npy'] == 1) &\
                                          (psy_df['feedbackType'] == \
                                           -1)].count()[0]
    
    return percentage_left, percentage_left_0, unrewarded_opto
    
    
psy_measures = pd.DataFrame(columns = ['mouse_name', 'ses',\
                                       'bias','threshold', 'gamma1', 'gamma2'])

for mouse in psy_df['mouse_name'].unique():
    for ses in psy_df.loc[(psy_df['mouse_name']==mouse), 'ses'].unique():
        psy_R =  psy_df.loc[(psy_df['mouse_name']==mouse) & \
                          (psy_df['ses']==ses) & \
                          (psy_df['opto_block']=='R')]
        psy_L =  psy_df.loc[(psy_df['mouse_name']==mouse) & \
                          (psy_df['ses']==ses) & \
                          (psy_df['opto_block']=='L')]
        pars_R,  L_R  =  ibl_psychometric (psy_R)
        pars_L,  L_L  =  ibl_psychometric (psy_L)
        
        pars = pars_R- pars_L
        psy_measures = psy_measures.append({'mouse_name': mouse, 'ses' : ses,\
                    'threshold': pars[1],'bias':pars[0],\
                    'gamma1':pars[2],'gamma2':pars[3]}, ignore_index=True)
    
sns.lineplot(x = 'ses', y = 'bias', hue = 'mouse_name', data =psy_measures)


def trials_per_min(psy_df):
    ses_rate = pd.DataFrame(columns = ['mouse_name','virus','rate'])
    for mouse in psy_df['mouse_name'].unique():
        for ses in psy_df.loc[(psy_df['mouse_name']==mouse), 'ses'].unique():
            session =  psy_df.loc[(psy_df['mouse_name']==mouse) & \
                              (psy_df['ses']==ses)]
            rate = len(session)/(max(session['stimOn_times']))
            
            ses_rate = ses_rate.append({'mouse_name': mouse, \
                             'virus': psy_df.loc[(psy_df['mouse_name'] \
                            == mouse),'virus'].unique()[0],\
                            'rate':rate}, ignore_index=True)
    sns.barplot(x = 'virus', y = 'rate', data = ses_rate)
    
    
    
def zerodelta_trials_back(psy_df, trials_back):
    
    right_choices = psy_df['choice']== -1
    psy_df.loc[:,'right_choices'] = right_choices
    deltas = pd.DataFrame(columns = ['on','virus','delta','trials_back'])
    
    if not 'signed_contrasts' in psy_df:
        psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
        psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
        psy_df.loc[:,'signed_contrasts'] =  \
        (psy_df['contrastRight'] - psy_df['contrastLeft'])
    for v, virus in enumerate(viruses):
            plt.sca(ax[v])
            sns.set_style('ticks')
            colors =['blue', 'green']
            if len(blocks) > 2:
                colors =['blue', 'green', 'black']
            
            for on in [0,1]:
           
                #  This could be made into a loop eventually(afteropto 
                #  iterations)
                #  Super ugly function 
                #delta 1
                psy_df_block_r = psy_df.loc[(psy_df['virus'] == virus) \
                                & (psy_df['after_opto'] == on)\
                                & (psy_df[block_variable] == 'R')\
                                & (psy_df['signed_contrasts'] == 0), \
                                'right_choices'].mean()
                psy_df_block_l = psy_df.loc[(psy_df['virus'] == virus) \
                                & (psy_df['after_opto'] == on)\
                                & (psy_df[block_variable] == 'L')\
                                & (psy_df['signed_contrasts'] == 0),\
                                'right_choices'].mean()
                delta  = psy_df_block_r - psy_df_block_l
                delta  = delta*100 #  Change to percentage
                
                deltas = deltas.append({'on': on, 'virus' : virus,\
                    'delta': delta,'trials_back':1}, ignore_index=True) 
                
                #delta 2
                psy_df_block_r = psy_df.loc[(psy_df['virus'] == virus) \
                                & (psy_df['after_opto_2'] == on)\
                                & (psy_df[block_variable] == 'R')\
                                & (psy_df['signed_contrasts'] == 0), \
                                'right_choices'].mean()
                psy_df_block_l = psy_df.loc[(psy_df['virus'] == virus) \
                                & (psy_df['after_opto_2'] == on)\
                                & (psy_df[block_variable] == 'L')\
                                & (psy_df['signed_contrasts'] == 0),\
                                'right_choices'].mean()
                delta2  = psy_df_block_r - psy_df_block_l
                delta2  = delta2*100 #  Change to percentage
                
                deltas = deltas.append({'on': on, 'virus' : virus,\
                    'delta': delta2,'trials_back':2}, ignore_index=True) 
                
                #delta 3
                psy_df_block_r = psy_df.loc[(psy_df['virus'] == virus) \
                                & (psy_df['after_opto_3'] == on)\
                                & (psy_df[block_variable] == 'R')\
                                & (psy_df['signed_contrasts'] == 0), \
                                'right_choices'].mean()
                psy_df_block_l = psy_df.loc[(psy_df['virus'] == virus) \
                                & (psy_df['after_opto_3'] == on)\
                                & (psy_df[block_variable] == 'L')\
                                & (psy_df['signed_contrasts'] == 0),\
                                'right_choices'].mean()
                delta3  = psy_df_block_r - psy_df_block_l
                delta3  = delta3*100 #  Change to percentage
                
                deltas = deltas.append({'on': on, 'virus' : virus,\
                    'delta': delta3,'trials_back':3}, ignore_index=True) 
                
                #delta 4
                psy_df_block_r = psy_df.loc[(psy_df['virus'] == virus) \
                                & (psy_df['after_opto_4'] == on)\
                                & (psy_df[block_variable] == 'R')\
                                & (psy_df['signed_contrasts'] == 0), \
                                'right_choices'].mean()
                psy_df_block_l = psy_df.loc[(psy_df['virus'] == virus) \
                                & (psy_df['after_opto_4'] == on)\
                                & (psy_df[block_variable] == 'L')\
                                & (psy_df['signed_contrasts'] == 0),\
                                'right_choices'].mean()
                delta4  = psy_df_block_r - psy_df_block_l
                delta4  = delta4*100 #  Change to percentage
                
                deltas = deltas.append({'on': on, 'virus' : virus,\
                    'delta': delta4,'trials_back':4}, ignore_index=True)                
                
    
                sns.set()
                sns.lineplot(x=  'trials_back', y = 'delta', \
                             data = deltas, style = 'on')
                



after_opto_chr2.savefig('after_opto_chr2.pdf')
after_opto_chr2_1.savefig('after_opto_chr2_2.pdf')
after_opto_chr2_3.savefig('after_opto_chr2_3.pdf')
after_opto_chr2_4.savefig('after_opto_chr2_4.pdf')

after_nonopto_chr2.savefig('after_nonopto_chr2.pdf')
after_nonopto_chr2_2.savefig('after_nonopto_chr2_2.pdf')
after_nonopto_chr2_3.savefig('after_nonopto_chr2_3.pdf')
after_nonopto_chr2_4.savefig('after_nonopto_chr2_4.pdf')

after_opto_nphr.savefig('after_opto_nphr.pdf')
after_opto_nphr_2.savefig('after_opto_nphr_2.pdf')
after_opto_nphr_3.savefig('after_opto_nphr_3.pdf')
after_opto_nphr_4.savefig('after_opto_nphr_4.pdf')

after_nonopto_nphr.savefig('after_nonopto_nphr.pdf')
after_nonopto_nphr_2.savefig('after_nonopto_nphr_2.pdf')
after_nonopto_nphr_3.savefig('after_nonopto_nphr_3.pdf')
after_nonopto_nphr_4.savefig('after_nonopto_nphr_4.pdf')
                    
                    