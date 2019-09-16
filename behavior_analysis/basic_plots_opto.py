#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:06:23 2019
Need to reinstall module, at the moment I need to be in npy2pd module, same for basic plots
@author: ibladmin
"""
from npy2pd import *
from basic_plots import *
from glm import *



#Input folder with raw npy files
psy_raw = load_data('/Volumes/witten/Alex/server_backup/Subjects_personal_project/opto_blocks/')
psy_df  = unpack(psy_raw)

#hot_fix for optoblocks until extractor is integrated
def opto_block_assigner (psy_df):
    psy_df['opto_block'] = np.nan
    psy_df.loc[(psy_df['feedbackType'] == 1) & (psy_df['opto.npy'] == 1) & (psy_df['contrastLeft'] >= 0), 'opto_block'] = 'L'
    psy_df.loc[(psy_df['feedbackType'] == 1) & (psy_df['opto.npy'] == 0) & (psy_df['contrastRight'] >= 0), 'opto_block'] = 'L'
    psy_df.loc[(psy_df['feedbackType'] == 1) & (psy_df['opto.npy'] == 1) & (psy_df['contrastRight'] >= 0), 'opto_block'] = 'R'
    psy_df.loc[(psy_df['feedbackType'] == 1) & (psy_df['opto.npy'] == 0) & (psy_df['contrastLeft'] >= 0), 'opto_block'] = 'R'
    psy_df['opto_block'] = psy_df['opto_block'].fillna(method='ffill') #propagate last valid block assignment for nan (incorrect trials)
    return psy_df



#repair spaces
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
block_variable = 'after_opto'
blocks = [1, 0]
block2_variable ='s.probabilityLeft'
blocks2 = [0.8, 0.5, 0.2]
        

#Plot across different trial history groups (after win and loses)
general = psychometric_summary(psy_df , block_variable, blocks, block2_variable, blocks2)
winner = psychometric_summary(psy_df.loc[psy_df['after_win']==1] , block_variable, blocks, block2_variable, blocks2)
loser = psychometric_summary(psy_df.loc[psy_df['after_win']==-1] , block_variable, blocks, block2_variable, blocks2)

#Save figs
general.savefig('general.pdf')
winner.savefig('winner.pdf')
loser.savefig('loser.pdf')


#Select probability blocks only
psy_df_global  = psy_df.loc[(psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8)]

#Plot glm with opto as a regressor for the different stimulation types

glms = opto_laser_glm(psy_df_global)
glms.savefig('glms.pdf')

#after opto vs non after opto
regressors_pre_reward = opto_glm(psy_df_global)
regressors_pre_reward.savefig('regressors_pre_reward.pdf')




