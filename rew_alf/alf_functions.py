#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:41:27 2020

@author: alex
"""

import numpy as np
import pandas as pd


# Stable block assigner
def opto_block_assigner (psy_df):
    psy_df['opto_block'] = np.nan
    psy_df.loc[(psy_df['opto_probability_left'] == 1), 'opto_block'] = 'L'
    psy_df.loc[(psy_df['opto_probability_left'] == 0), 'opto_block'] = 'R'
    psy_df.loc[(psy_df['opto_probability_left'] == -1), 'opto_block'] = 'non_opto'
    return psy_df

# Signed contrast calculator
def add_signed_contrasts (psy_df):
    psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df.loc[:,'signed_contrasts'] =  \
    (psy_df['contrastRight'] - psy_df['contrastLeft'])
    return psy_df

def add_trial_within_block(df):
    '''
    df: dataframe with behavioral data
    '''
    df['trial_within_block'] = np.nan
    for mouse in df['mouse_name'].unique():
        for ses in df.loc[df['mouse_name']==mouse,'ses'].unique():
            session= df.loc[(df['mouse_name']
                             ==mouse) & (df['ses']==ses)]
            block_breaks = np.diff(session['opto_probability_left'])
            block_breaks = np.where(block_breaks != 0)
            for i, t in enumerate(block_breaks[0]):
                if i == 0:
                    for l in range(t+1):
                        session.iloc[l, session.columns.get_loc('trial_within_block')] = l
                
                else:
                    for x, l in enumerate(range(block_breaks[0][i-1]+1,t+1)):
                        session.iloc[l, session.columns.get_loc('trial_within_block')] = x
                if i + 1 == len(block_breaks[0]):
                    session.iloc[t+1:, session.columns.get_loc('trial_within_block')] = np.arange(len(session)-t -1)
            df.loc[(df['mouse_name']
                             ==mouse) & (df['ses']==ses)] =  session
    return df