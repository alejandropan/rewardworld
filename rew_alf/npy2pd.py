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
	#'itiDuration',
    's.probabilityLeft', #hack for avoiding confusion with rewproability
    'opto_probability_left',
	'response_times',
	'rewardVolume',
	'opto.npy', 
    'opto_dummy',
    'hem_stim',
    #'rewprobabilityLeft',
	'stimOn_times'
    ]
    
    return pybpod_vars

def clean_str(string):
    """
    Cleans str with extra spaces e.g ' l ' transforms to 'l'
    """



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
           # Code below used to merge variables with similar name,
           # obsolete function in current version, choose first intead
           # to avoid merger of goCue_trigger_times and goCue_trigger_times_bpod
           merge = [x for x in raw_data if re.search(var, x)]

           if not merge:
               continue
           
           untrimmed = np.load(merge[0])
           if (var == 'opto_probability_left' or var == 'opto_dummy' or var == 'hem_stim'):
                data[var]  = untrimmed[:len(data['choice'])] # these files are not automatically trimmed if ephys_trials do not much bpod trials. The extra trials are at the end [confirmed], this happens when ephys finishes before bpod.
           else:
                data [var]  = untrimmed
        return data
    
def load_data (subject_folder):
    """Generates a dataframe with all available information in project folder
    INPUT: root folder include several subjects and viruses
    OUTPUT:  macro (dataframe per animal per session)"""
    #subject_folder =  '/mnt/s0/Data/Subjects_personal_project/rewblocks8040/'
    viruses = sorted([x for x in (os.listdir (subject_folder)) if ".DS_Store" not in x])
    variables  = pybpod_vars()
    col = variables
    col.append('ses')
    col.append('mouse_name')
    col.append('virus')    
    macro = pd.DataFrame(columns = col)
    for virus in viruses:
        mice = sorted([x for x in (os.listdir (subject_folder + virus +'/')) if ".DS_Store" not in x])
        for mouse in mice:
            dates =  sorted([x for x in (os.listdir (subject_folder + virus + '/' + mouse)) if ".DS_Store" not in x])
            df = pd.DataFrame(index=dates, columns = col)
            for day in dates:
                #merge sessions from the same day
                path = subject_folder + virus + '/' + mouse + '/' + day +'/'
                data  = session_loader(path, variables)
                df.loc[day]  = data
            df['virus'] =  virus
            df['ses'] = dates
            df['mouse_name'] =mouse
            df = df.set_index(['mouse_name'], drop=False)
            macro = macro.append(df)
    
    return macro

def unpack (macro):
    """pools dataframe with index= session into dataframe with index = trial
    INPUT": macro (dataframe from load data)
    OUTPUT: trial_macro
    EXCEPTION: Raise exception in nan trials (note: these are not 0 trials, 
    these are trials where both contrastLeft and contrastRight = 0)
    """
    variable_name = list(macro.columns.values)
    ses_num = macro.shape[0]
    #change sessions measing all RTs to nan 
    for i in range(ses_num):
        if  np.isnan(np.nanmean(macro.iloc[i, macro.columns.get_loc('stimOn_times')])):
            macro.iloc[i, macro.columns.get_loc('stimOn_times')]  =  np.full( macro.iloc[i, macro.columns.get_loc('choice')].size, np.nan)
        #copy name and session date to all trials
        macro.iloc[i, macro.columns.get_loc('ses')] = np.full( macro.iloc[i, macro.columns.get_loc('choice')].size, macro.iloc[i, macro.columns.get_loc('ses')])
        macro.iloc[i, macro.columns.get_loc('mouse_name')] = np.full(macro.iloc[i, macro.columns.get_loc('choice')].size, macro.iloc[i, macro.columns.get_loc('mouse_name')])
        macro.iloc[i, macro.columns.get_loc('virus')]= np.full( macro.iloc[i, macro.columns.get_loc('choice')].size, macro.iloc[i, macro.columns.get_loc('virus')])
    #Initialize unpacked dataframe
    trial_macro =  pd.DataFrame(columns = variable_name)
    # ... and fill it 
    for i in variable_name:
       flat_list = [item for test in macro[i] for item in test] #Flattens column of each variable
       trial_macro[i] = flat_list
    #trial_macro at this stage might have nan rows that need to be removed. 
    for trial in range(trial_macro.shape[0]): 
        if np.isnan(trial_macro['contrastLeft'][trial]) & np.isnan(trial_macro['contrastRight'][trial]):
            print('Error in %s trial' %trial)
            raise Exception('Error in trial: {}'.format(trial))
    
    return trial_macro

    