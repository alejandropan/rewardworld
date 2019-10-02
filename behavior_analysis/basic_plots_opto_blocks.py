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


#Input folder with raw npy files
psy_raw = load_data('/Volumes/witten/Alex/server_backup/Subjects_personal_project/opto_blocks/')

if psy_raw.isnull().values.any():
    psy_raw  = psy_raw.dropna()
    print ('Warning: sessions deleted due to entire variables with NaN')

psy_df  = unpack(psy_raw)

#Stable block assigner
def opto_block_assigner (psy_df):
    psy_df['opto_block'] = np.nan
    psy_df.loc[(psy_df['opto_probability_left'] == 1), 'opto_block'] = 'L'
    psy_df.loc[(psy_df['opto_probability_left'] == 0), 'opto_block'] = 'L'
    psy_df.loc[(psy_df['opto_probability_left'] == -1), 'opto_block'] = 'R'
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
blocks = ['L', 'R']
        

#Plot across different trial history groups
long = psychometric_summary_opto_blocks(psy_df , block_variable, blocks)
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



