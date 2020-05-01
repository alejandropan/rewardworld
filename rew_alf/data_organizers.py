#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 16:10:19 2020

@author: alex
"""
from npy2pd import *
import numpy as np
import pandas as pd




def load_behavior_data_from_root(root_data_folder, remove_last_100 = False):
    
    '''
    
    Parameters
    ----------
    root_data_folder : Location of the root fdat folder for the opto task, the 
    location should be uproot from the virus folder. (str)
    remove_last_100 : Whether to remove the last 100 trials from each session
    since the animal might disengaged (Boolean)
    
    Returns
    -------
    psy_df ready to use in model with extra variable such as: signed_contrast,
    trial within block
    
    Important Notes
    ---------------
    Under this version,  action are are already revearsed to 
    ensure that -1 actions are left and right action are 1. 
    Current code drops NOGO.
    
    '''
    
    psy_raw = load_data(root_data_folder)
    
    if psy_raw.isnull().values.any():
        psy_raw  = psy_raw.dropna()
        print ('Warning: sessions deleted due to entire variables with NaN')
    
        psy_df  = unpack(psy_raw)
        
    if remove_last_100 == True:
        psy_df = remove_last100(psy_df)
    
    psy_df = opto_block_assigner(psy_df)
    psy_df = add_signed_contrasts(psy_df)
    psy_df = add_trial_within_block(psy_df)
    psy_df = psy_df.drop(psy_df[psy_df['choice'] == 0].index)
    psy_df = psy_df.reset_index()
    
    return psy_df
    
    
def psy_df_to_Q_learning_model_format(psy_dataframe, virus = 'chr2'):

    '''
    
    Parameters
    ----------
    psy_df : datafram ewith all the information for every trial in every animal,
    is the product of load_behavior_data_from_root
    
    virus : virus to urn the model on
    
    Returns
    -------
    simulate_data : required data for simulating the POMDP
    model_data: required for running the POMDP model

    '''
    
    
    # Select virus and make dataframe
    
    psy_df = psy_dataframe.loc[psy_df['virus'] == virus].copy()
    psy_df = psy_df.reset_index()
    
    # Add opto_side variable (signal what kind of action led to opto in a trial)
    
    opto_side_num = np.zeros([psy_df.shape[0],1])
    opto_side_num[psy_df.loc[(psy_df['opto_block'] == 'L')|
                             (psy_df['opto_block'] == 'R')].index, 0] = \
                             psy_df.loc[(psy_df['opto_block'] == 'L')| \
                                        (psy_df['opto_block'] == 'R'),'choice'] * \
                             psy_df.loc[(psy_df['opto_block'] == 'L')| \
                                        (psy_df['opto_block'] == 'R'),'opto.npy']
    opto_side = np.empty([len(opto_side_num),1],dtype=object)
    opto_side[:] = 'none'
    opto_side[opto_side_num == 1] = 'left'
    opto_side[opto_side_num == -1] = 'right'
    

    # Signed contrast
    signed_contrasts = np.zeros([len(opto_side_num),1])
    signed_contrasts[:,0] = psy_df['signed_contrasts'].to_numpy()
    
    # Make dataframes for t
    model_data = pd.DataFrame({'stimTrials': signed_contrasts[:,0],'extraRewardTrials': psy_df['opto.npy'], 
                              'choice': psy_df['choice'], 'reward': psy_df['feedbackType'], 'ses':psy_df['ses']})
    model_data.loc[model_data['reward'] == -1, 'reward'] = 0
    
    simulate_data = pd.DataFrame({'stimTrials': signed_contrasts[:,0],'extraRewardTrials': psy_df['opto_block'], 
                              'choice': psy_df['choice'], 'ses':psy_df['ses']})
    simulate_data.loc[simulate_data['extraRewardTrials'] == 'non_opto','extraRewardTrials'] = 'none'
    simulate_data.loc[simulate_data['extraRewardTrials'] == 'L','extraRewardTrials'] = 'left'
    simulate_data.loc[simulate_data['extraRewardTrials'] == 'R','extraRewardTrials'] = 'right'

    return model_data,  simulate_data
    

    
def remove_last100(psy_df):
    psy_df_new  = pd.DataFrame(columns = psy_df.columns)
    for mouse in psy_df['mouse_name'].unique():
            for ses in psy_df.loc[(psy_df['mouse_name']==mouse), 'ses'].unique():
                session =  psy_df.loc[(psy_df['mouse_name']==mouse) & \
                                  (psy_df['ses']==ses)]
                new_session = session.iloc[:-100]  
                psy_df_new = psy_df_new.append(new_session, ignore_index = True)
    return psy_df_new 
    
    

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
