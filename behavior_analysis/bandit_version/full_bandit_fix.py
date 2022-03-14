#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:56:47 2020

@author: alex
"""

from ibllib.io import raw_data_loaders as raw
import sys
import numpy as np
# Fix choices
def full_bandit_fix(ses): 
    data = raw.load_data(ses)
    choices = np.zeros(len(data))
    choices[:] = np.nan
    opto = np.zeros(len(data))
    opto[:] = np.nan
    left_reward = np.zeros(len(data))
    left_reward[:] = np.nan
    right_reward = np.zeros(len(data))
    right_reward[:] = np.nan
    for i in np.arange(len(data)):
        # Extract choices
        if 'RotaryEncoder1_1' in data[i]['behavior_data']['Events timestamps']:
            choices[i] = -1   
        if 'RotaryEncoder1_2' in data[i]['behavior_data']['Events timestamps']:
            choices[i] = 1
        if ('RotaryEncoder1_2' in data[i]['behavior_data']['Events timestamps'])\
                & ('RotaryEncoder1_1' in data[i]['behavior_data']['Events timestamps']):
            # Choose the earlier event, since that one caused reward
            left_time = data[i]['behavior_data']['Events timestamps']['RotaryEncoder1_2'][0]
            right_time = data[i]['behavior_data']['Events timestamps']['RotaryEncoder1_1'][0]
            if left_time > right_time:
                choices[i] = -1
            if right_time > left_time:
                choices[i] = 1
        # Extract opto
        try:
            opto[i] = data[i]['opto']
        except:
            print('No laser info')
        #Extract left and right potential rewards
        left_reward[i] = data[i]['left_reward']
        left_reward[i] = data[i]['left_reward']
        right_reward[i] = data[i]['right_reward']
        right_reward[i] = data[i]['right_reward']

    np.save(ses+'/alf/_ibl_trials.choice.npy',choices)
    np.save(ses+'/alf/_ibl_trials.opto.npy',opto)
    np.save(ses+'/alf/_ibl_trials.left_reward.npy',left_reward)
    np.save(ses+'/alf/_ibl_trials.right_reward.npy',right_reward)

if __name__ == "__main__":
    full_bandit_fix(*sys.argv[1:])
    