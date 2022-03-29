#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:33:06 2020

@author: alex
"""


import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns
import sys
import json
import rewardworld.behavior_analysis.bandit_version.block_bandit_summary as bd


def get_tree(mouse_folder, extract=False):
    alf_folders = []
    dates = [e for e in mouse_folder.iterdir() if e.is_dir()]
    for date in dates:
        sessions = [e for e in date.iterdir() if e.is_dir()]
        for ses in sessions:
            alf = ses / 'alf'
            if alf.exists():
                alf_folders.append(alf)
            elif extract==True:
                try:
                    bd.extract_all(ses, save=True)
                    #bd.extract_all_wheel(ses, save=True)
                    bd.full_bandit_fix(ses)
                except:
                    print('No data for '+ str(ses))
    return alf_folders

def data_from_mouse(mouse, extract=True):
    mouse_folder = Path('/Volumes/witten/Alex/Data/Subjects/' + mouse)
    alfs = get_tree(mouse_folder, extract=extract)
    performance = []
    protocols = []
    date = []
    iti=[]
    stay=[]
    
    for alf in alfs:
        performance.append(np.mean(np.load(alf / '_ibl_trials.feedbackType.npy')==1))
        jfile = alf.parent / 'raw_behavior_data/_iblrig_taskSettings.raw.json'
        with open(jfile) as json_file: 
            settings = json.load(json_file)
        protocols.append(settings['PYBPOD_PROTOCOL'])
        iti.append(settings['ITI_ERROR'])
        date.append(alf.parent.parent.stem + '-' + alf.parent.stem)
        #bd.full_bandit_fix(str(alf.parent.resolve()))
        dt_mouse = pd.DataFrame()
        dt_mouse['choices']=np.load(alf / '_ibl_trials.choice.npy')
        dt_mouse['rewarded']= np.load(alf / '_ibl_trials.feedbackType.npy')==1
        dt_mouse['rewarded_1']=dt_mouse['rewarded'].shift(1)
        dt_mouse['choices_1']=dt_mouse['choices'].shift(1)
        p_error = dt_mouse.loc[dt_mouse['rewarded_1']==False]
        stay.append(np.mean(p_error['choices']==p_error['choices_1']))

    data = pd.DataFrame()
    data['date'] = date
    data['protocol'] = protocols
    data['performance'] = performance
    data['iti']=iti
    data['stay']=stay
    data  = data.sort_values('date')
    data['protocol'] = data['protocol'].map({'_bandit_1stday_biasedChoiceWorld':'100/100',
                            '_bandit_shaping_biasedChoiceWorld':'80/20',
                            '_bandit_shaping_90_10_biasedChoiceWorld':'90/10',
                            '_bandit_shaping_100_0_biasedChoiceWorld':'100/0',
                            '_bandit_laser_blocks_ephysChoiceWorld':'laser_water_bandit',
                            '_bandit_100_0_biasedChoiceWorld':'100/0',
                            '_bandit_onlylaser_blocks_ephysChoiceWorld':'laser_only_bandit',
                            '_bandit_biasedChoiceWorld':'full'})
    return data

def plot_performance(mouse):
    data = data_from_mouse(mouse)
    sns.pointplot(data=data,x='date',y='performance',hue='protocol')
    plt.xticks(rotation=90)
    sns.despine()
    plt.tight_layout()
    plt.hlines(0.50, 0, len(data))
    plt.ylim(0.2,0.8)
    plt.savefig('/Users/alex/Downloads/performance.png')
    


data = pd.DataFrame()
for mouse in ['dop_40','dop_45','dop_46']:
    mouse_df = data_from_mouse(mouse)
    mouse_df['mouse_name']=mouse
    data = pd.concat([data,mouse_df])
sns.barplot(data=data.loc[data['protocol']=='full'],x='iti',y='performance',
                  hue='mouse_name')


