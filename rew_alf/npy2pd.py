#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 17:33:22 2019

@alejandro
Functions for extraction and organisation of pybpod npy files.

pybpod_vars is hard coded list of variables: 
Current list of variables
 
'choice'
	'trials.contrastLeft',
	'trials.contrastRight',
	'trials.feedback_times',
	'trials.feedbackType',
	'trials.goCue_times',
	'trials.goCueTrigger_times',
	'trials.intervals',
	'trials.itiDuration',
	'trials.probabilityLeft',
	'trials.response_times',
	'trials.rewardVolume',
	'trials.rewprobabilityLeft',
	'trials.stimOn_times',
	'wheel.position',
	'wheel.timestamps',
	'wheel.velocity']
    
@author: ibladmin
"""

import os
import re
import numpy as np
import pandas as pd

def pybpod_vars():
    pybpod_vars   = [
	'choice',
	'contrastLeft',
	'contrastRight',
	'feedback_times',
	'feedbackType',
	'goCue_times',
	'goCueTrigger_times',
	'intervals',
	'itiDuration',
	's.probabilityLeft', #hack for avoiding confusion with rewproability
	'response_times',
	'rewardVolume',
	'rewprobabilityLeft',
	'stimOn_times'
	#'position',
	#'timestamps',
	#'velocity'
    ]
    
    return pybpod_vars

def session_loader(path,variables):
        """returns dictionary with data for a given a day"""
        #merge sessions from the same day
        raw_data = []
        for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith(".npy"):
                        raw_data.append ((os.path.join(root, file)))
        data = {}
        for var in variables:            
            #list equivalent npys
            merge = [x for x in raw_data if re.search(var, x)]
            if not merge:
                continue
            dat = []
            for x, file in enumerate(merge):
                dat.append(np.load(file))
            dat = np.concatenate(dat)
            data [var]  = dat
            
        return data
    
def load_data (subject_folder):
    """Generates a dataframe with all available information in project folder"""
    #subject_folder =  '/mnt/s0/Data/Subjects_personal_project/rewblocks8040/'
    mice = sorted(os.listdir (subject_folder))
    variables  = pybpod_vars()
    col = variables
    col.append('ses')
    col.append('mouse_name')
    
    macro = pd.DataFrame(columns = col)
    for mouse in mice:
        dates =  sorted(os.listdir (subject_folder + mouse))
        df = pd.DataFrame(index=dates, columns = col)
        for day in dates:
            #merge sessions from the same day
            path = subject_folder + mouse + '/' + day
            data  = session_loader(path, variables)
            df.loc[day]  = data
        df['ses'] = dates
        df['mouse_name'] =mouse
        df = df.set_index(['mouse_name'], drop=False)
        macro = macro.append(df)
    
    return macro