#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:15:23 2020

@author: alex


Plots demonstrating RL behavior

"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Load Alex's actual data from folder, with desired Q values
psy = pd.read_pickle('all_behav.pkl')
mice = np.array(['dop_8', 'dop_9', 'dop_11', 'dop_4'])
psy = psy.loc[np.isin(psy['mouse_name'], mice)]

# Figure 1 #

'''
First we show the psychometric curves post correct, post error right and left
4 psychometrics in total. No fitting
'''

# Set difficulty of trials
hard = [0, 0.0625, -0.0625]
easy = [0.125, 0.25, -0.125, -0.25]
psy['difficulty'] = np.nan
psy.loc[np.isin(psy['signed_contrasts'], hard), 'difficulty'] = 'hard'
psy.loc[np.isin(psy['signed_contrasts'], easy), 'difficulty'] = 'easy'

# Set side
psy['side'] = np.nan
psy.loc[psy['signed_contrasts'] > 0, 'side'] = 'right'
psy.loc[psy['signed_contrasts'] < 0, 'side'] = 'left'
psy.loc[(psy['signed_contrasts'] == 0) & (psy['choice'] == -1) & \
        (psy['feedbackType'] == 1)  , 'side'] = 'right'
psy.loc[(psy['signed_contrasts'] == 0) & (psy['choice'] == 1) & \
        (psy['feedbackType'] == 1)  , 'side'] = 'left'


# Change choice to right==1
psy['choice'] = psy['choice']*-1

# Drop no-go
psy = psy.loc[psy['choice']!=0]

# First calculate previous choice, opto and previous outcome, previous diffficulty
blocks = ['left', 'neutral', 'right']
psy['block'] = np.nan
psy['prev_choice'] = np.nan
psy['prev_outcome'] = np.nan
psy['prev_opto'] = np.nan
psy['prev_difficulty'] = np.nan
psy['prev_stim'] = np.nan
psy['prev_side'] = np.nan
psy['next_side'] = np.nan
psy['next_outcome'] = np.nan
psy['next_difficulty'] = np.nan
psy['next_choice'] = np.nan


for name in psy['mouse_name'].unique():
    psy.loc[psy['mouse_name']==name, 'prev_choice'] = \
       psy.loc[psy['mouse_name']==name, 'choice'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_outcome'] = \
       psy.loc[psy['mouse_name']==name, 'feedbackType'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_opto'] = \
       psy.loc[psy['mouse_name']==name, 'opto.npy'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_difficulty'] = \
       psy.loc[psy['mouse_name']==name, 'difficulty'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_stim'] = \
       psy.loc[psy['mouse_name']==name, 'signed_contrasts'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_side'] = \
       psy.loc[psy['mouse_name']==name, 'side'].shift(1)
    # Calculate future as well
    psy.loc[psy['mouse_name']==name, 'next_side'] = \
       psy.loc[psy['mouse_name']==name, 'side'].shift(-1)
    psy.loc[psy['mouse_name']==name, 'next_difficulty'] = \
       psy.loc[psy['mouse_name']==name, 'difficulty'].shift(-1)
    psy.loc[psy['mouse_name']==name, 'next_outcome'] = \
       psy.loc[psy['mouse_name']==name, 'feedbackType'].shift(-1)
    psy.loc[psy['mouse_name']==name, 'next_choice'] = \
       psy.loc[psy['mouse_name']==name, 'choice'].shift(-1)
       
# Make prev_opto into boolean
psy.loc[psy['prev_opto']==1, 'prev_opto'] = True
psy.loc[psy['prev_opto']==0, 'prev_opto'] = False
      
# Cahnge prev choice and outcome to string
psy.loc[psy['prev_outcome'] == -1, 'prev_outcome'] = 'Error'
psy.loc[psy['prev_outcome'] == 1, 'prev_outcome'] = 'Reward'
psy.loc[psy['prev_choice'] == 1, 'prev_choice'] = 'Right'
psy.loc[psy['prev_choice'] == -1, 'prev_choice'] = 'Left'
psy.loc[psy['next_choice'] == 1, 'next_choice'] = 'Right'
psy.loc[psy['next_choice'] == -1, 'next_choice'] = 'Left'
psy.loc[psy['next_outcome'] == -1, 'next_outcome'] = 'Error'
psy.loc[psy['next_outcome'] == 1, 'next_outcome'] = 'Reward'

# Change choice to 0 to 1 range
psy['choice'] = (psy['choice']>0)*1

#Assign blocks
blocks = ['left', 'neutral', 'right']
psy.loc[psy['opto_probability_left']==-1, 'block']  = 'neutral'
psy.loc[psy['opto_probability_left']== 1, 'block']  = 'left'
psy.loc[psy['opto_probability_left']==0, 'block']  = 'right'
       

# Plot by block based on previous reward

pal ={"Right":"r","Left":"b"}

fig, ax =  plt.subplots(3, figsize=(5,10))
for i, b in enumerate(blocks):
    plt.sca(ax[i])
    sns.lineplot(data = psy.loc[(psy['virus']=='chr2') & (psy['block']== b)], \
                 x='signed_contrasts', 
                 y='choice', hue='prev_choice', style='prev_outcome', ci=68,
                 legend='brief', palette=pal)
    plt.title(b)
    plt.ylabel('% Right Choices')
plt.tight_layout()

fig, ax =  plt.subplots(3, figsize=(5,10))
for i, b in enumerate(blocks):
    plt.sca(ax[i])
    sns.lineplot(data = psy.loc[(psy['virus']=='nphr') & (psy['block']==b)], \
                 x='signed_contrasts', 
                 y='choice', hue='prev_choice', style='prev_outcome', ci=68,
                 legend='brief', palette=pal)
    plt.title(b)
    plt.ylabel('% Right Choices')
plt.tight_layout()


# Plot by previous reward
fig, ax =  plt.subplots(2, figsize=(5,10))
for i, v in enumerate (psy['virus'].unique()):
    plt.sca(ax[i])
    sns.lineplot(data = psy.loc[(psy['virus']==v)], \
                 x='signed_contrasts', 
                 y='choice', hue ='prev_choice', style='prev_outcome', ci=68,
                 legend='brief', palette=pal)
    plt.title(v)
plt.tight_layout()

# Plot by opto

fig, ax =  plt.subplots(2, figsize=(5,10))
for i, v in enumerate(psy['virus'].unique()):
    plt.sca(ax[i])
    sns.lineplot(data = psy.loc[(psy['virus']== v)], \
                     x='signed_contrasts', 
                     y='choice', hue ='prev_choice', style = 'prev_opto', ci = 68,
                     legend='brief', palette=pal)
    plt.title(v)
plt.tight_layout()


# Figure 2 -  Divide by choice difficulty #

#Previous All
fig, ax =  plt.subplots(2, figsize=(5,10))
for i, v in enumerate(psy['virus'].unique()):
    plt.sca(ax[i])
    pal2 = {"hard":"solid","easy":"dashed"}
    sns.lineplot(data = psy.loc[(psy['virus']== v)], \
                     x='signed_contrasts', 
                     y='choice', hue ='prev_choice', style = 'prev_difficulty', 
                     style_order=['easy', 'hard'], ci = 68,
                     legend='brief', palette=pal)
    plt.title(v)
plt.tight_layout()

#Previous correct
fig, ax =  plt.subplots(2, figsize=(5,10))
for i, v in enumerate(psy['virus'].unique()):
    plt.sca(ax[i])
    sns.lineplot(data = psy.loc[(psy['virus']== v) & (psy['prev_outcome']== 'Reward')], \
                     x='signed_contrasts', 
                     y='choice', hue ='prev_choice', style = 'prev_difficulty', ci = 68,
                     legend='brief', palette=pal,  style_order=['easy', 'hard'])
    plt.title(v)
plt.tight_layout()

#Previous error
fig, ax =  plt.subplots(2, figsize=(5,10))
for i, v in enumerate(psy['virus'].unique()):
    plt.sca(ax[i])
    sns.lineplot(data = psy.loc[(psy['virus']== v) & (psy['prev_outcome']== 'Error')], \
                     x='signed_contrasts', 
                     y='choice', hue ='prev_choice', style = 'prev_difficulty', ci = 68,
                     legend='brief', palette=pal,  style_order=['easy', 'hard'])
    plt.title(v)
plt.tight_layout()


# Figure 2 -  Divide by choice difficulty and current stimulus#

#Previous All


fig, ax =  plt.subplots(2,2, figsize=(10,10))
for i, v in enumerate(psy['virus'].unique()):
    hp = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['prev_side'], index=['difficulty', 
                                'prev_difficulty', 'prev_outcome']).reset_index()
    hp['bias_shift'] = hp.choice.right - hp.choice.left
    
    for j, c in enumerate(hp['prev_outcome'].unique()):
        hp1= pd.pivot_table(hp.loc[hp['prev_outcome']==c], values='bias_shift', 
                            columns='prev_difficulty', index = 'difficulty')
        plt.sca(ax[i,j])
        sns.heatmap(hp1, cmap="YlGnBu", cbar_kws={'label': '% updating'})
        plt.title(v + ' after ' + c)
        plt.xlabel('Previous choice difficulty')
        plt.ylabel('Current choice difficulty')   
plt.tight_layout()

# Figure 2 -  with correction
fig, ax =  plt.subplots(2,2, figsize=(10,10))
for i, v in enumerate(psy['virus'].unique()):
    hp = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['prev_side'], index=['difficulty', 
                                'prev_difficulty', 'prev_outcome']).reset_index()
    hp['bias_shift'] = hp.choice.right - hp.choice.left
    
    hp_future = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['next_side'], index=['difficulty', 
                                'next_difficulty', 'next_outcome']).reset_index()
    hp_future['bias_shift'] = hp_future.choice.right - hp_future.choice.left
    
    for j, c in enumerate(hp['prev_outcome'].unique()):
        hp1= pd.pivot_table(hp.loc[hp['prev_outcome']==c], values='bias_shift', 
                            columns='prev_difficulty', index = 'difficulty')
        hp2= pd.pivot_table(hp_future.loc[hp_future['next_outcome']==c], values='bias_shift', 
                            columns='next_difficulty', index = 'difficulty')
        hp3 = hp1-hp2
        plt.sca(ax[i,j])
        sns.heatmap(hp3, cmap="YlGnBu", cbar_kws={'label': '% updating'})
        plt.title(v + ' after ' + c)
        plt.xlabel('Previous choice difficulty')
        plt.ylabel('Current choice difficulty')   
plt.tight_layout()

#Figure 2 replicatein lake figure 1f


fig, ax =  plt.subplots(2,2, figsize=(10,10))
for i, v in enumerate(psy['virus'].unique()):
    hp = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['prev_side'], index=['signed_contrasts', 
                                'prev_difficulty', 'prev_outcome',
                                'mouse_name']).reset_index()
    hp['bias_shift'] = (hp.choice.right - hp.choice.left) *100
    
    for j, c in enumerate(hp['prev_outcome'].unique()):
        hp1= pd.pivot_table(hp.loc[hp['prev_outcome']==c], values='bias_shift', 
                            columns='prev_difficulty', 
                            index = 'signed_contrasts').reset_index()
        hp1= hp.loc[hp['prev_outcome']==c]
        
        
        plt.sca(ax[i,j])
        sns.lineplot(data = hp1, x = 'signed_contrasts', y = 'bias_shift',
                     hue = 'prev_difficulty', ci=68)
        plt.title(v + ' after ' + c)
        plt.ylabel('% $\Delta Right Choices')
        plt.xlabel('Trial contrast')   
plt.tight_layout()


#Figure 2 replicatein lake figure 1f based on prev choice


fig, ax =  plt.subplots(2,2, figsize=(10,10))
for i, v in enumerate(psy['virus'].unique()):
    hp = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['prev_choice'], index=['signed_contrasts', 
                                'prev_difficulty', 'prev_outcome',
                                'mouse_name']).reset_index()
    hp['bias_shift'] = (hp.choice.Right - hp.choice.Left) *100
    
    for j, c in enumerate(hp['prev_outcome'].unique()):
        hp1= pd.pivot_table(hp.loc[hp['prev_outcome']==c], values='bias_shift', 
                            columns='prev_difficulty', 
                            index = 'signed_contrasts').reset_index()
        hp1= hp.loc[hp['prev_outcome']==c]
        
        
        plt.sca(ax[i,j])
        sns.lineplot(data = hp1, x = 'signed_contrasts', y = 'bias_shift',
                     hue = 'prev_difficulty', ci=68)
        plt.title(v + ' after ' + c)
        plt.ylabel('% $Delta$, Right Choices')
        plt.xlabel('Trial contrast')   
plt.tight_layout()



#Figure 2 replicatein lake figure 1f with correction

fig, ax =  plt.subplots(2,2, figsize=(10,10))
for i, v in enumerate(psy['virus'].unique()):
    hp = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['prev_side'], index=['signed_contrasts', 
                                'prev_difficulty', 'prev_outcome',
                                'mouse_name']).reset_index()
    hp['bias_shift'] = (hp.choice.right - hp.choice.left) *100
    
    hp_future = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['next_side'], index=['signed_contrasts', 
                                'next_difficulty', 'next_outcome', 'mouse_name']).reset_index()
    hp_future['bias_shift'] = (hp_future.choice.right - hp_future.choice.left)*100
    
    for j, c in enumerate(hp['prev_outcome'].unique()):
       
        hp1= hp.loc[hp['prev_outcome']==c]
        hp2= hp_future.loc[hp_future['next_outcome']==c]
        hp3= hp1.copy()
        hp3['bias_shift_corrected'] = hp1['bias_shift'] - hp2['bias_shift']
        plt.sca(ax[i,j])
        sns.lineplot(data = hp3, x = 'signed_contrasts', y = 'bias_shift_corrected',
                     hue = 'prev_difficulty', ci=68)
        plt.title(v + ' after ' + c)
        plt.ylabel('% $\Delta Right Choices')
        plt.xlabel('Trial contrast')   
plt.tight_layout()


#Figure 2 replicatein lake figure 1f with correction and prev choice

fig, ax =  plt.subplots(2,2, figsize=(10,10))
for i, v in enumerate(psy['virus'].unique()):
    hp = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['prev_choice'], index=['signed_contrasts', 
                                'prev_difficulty', 'prev_outcome',
                                'mouse_name']).reset_index()
    hp['bias_shift'] = (hp.choice.Right - hp.choice.Left) *100
    
    hp_future = pd.pivot_table( psy.loc[psy['virus']== v], values=['choice'], columns= ['next_choice'], index=['signed_contrasts', 
                                'next_difficulty', 'next_outcome', 'mouse_name']).reset_index()
    hp_future['bias_shift'] = (hp_future.choice.Right - hp_future.choice.Left)*100
    
    for j, c in enumerate(hp['prev_outcome'].unique()):
       
        hp1= hp.loc[hp['prev_outcome']==c]
        hp2= hp_future.loc[hp_future['next_outcome']==c]
        hp3= hp1.copy()
        hp3['bias_shift_corrected'] = hp1['bias_shift'] - hp2['bias_shift']
        plt.sca(ax[i,j])
        sns.lineplot(data = hp3, x = 'signed_contrasts', y = 'bias_shift_corrected',
                     hue = 'prev_difficulty', ci=68)
        plt.title(v + ' after ' + c)
        plt.ylabel('% $\Delta Right Choices')
        plt.xlabel('Trial contrast')   
plt.tight_layout()


# GLM results



