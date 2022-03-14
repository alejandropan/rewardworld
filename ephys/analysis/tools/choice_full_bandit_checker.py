import pandas as pd
import numpy as np
import model_comparison_accu as mc
from ibllib.io import raw_data_loaders as raw

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
        if 'RotaryEncoder1_1' in data[i]['behavior_data']['Events timestamps']:
            choices[i] = -1   
        if 'RotaryEncoder1_2' in data[i]['behavior_data']['Events timestamps']:
            choices[i] = 1
        if ('RotaryEncoder1_2' in data[i]['behavior_data']['Events timestamps']) \
                & ('RotaryEncoder1_1' in data[i]['behavior_data']['Events timestamps']):
            # Choose the earlier event, since that one caused reward
            left_time = data[i]['behavior_data']['Events timestamps']['RotaryEncoder1_2'][0]
            right_time = data[i]['behavior_data']['Events timestamps']['RotaryEncoder1_1'][0]
            if left_time > right_time:
                choices[i] = -1
            if right_time > left_time:
                choices[i] = 1
    np.save(ses+'/alf/_ibl_trials.choice.npy',choices)

#check that the full bandit fix has been applied
psy=mc.load_data()
alf_list=[]
for ns, mouse in enumerate(psy['mouse'].unique()):
    animal = psy.loc[psy['mouse']==mouse]
    counter=0
    for day in animal['date'].unique():
        day_s = animal.loc[animal['date']==day]
        for ses in day_s['ses'].unique():
            session = day_s.loc[day_s['ses']==ses]
            alf = ROOT_FOLDER+'/'+mouse+'/'+day+'/'+ses+'/alf'
            ses_path = ROOT_FOLDER+'/'+mouse+'/'+day+'/'+ses
            og_choices = np.load(alf+'/_ibl_trials.choice.npy')
            full_bandit_fix(ses_path)
            new_choices = np.load(alf+'/_ibl_trials.choice.npy')
            result = np.array_equal(og_choices, new_choices, equal_nan=True)
            print (ses_path+' '+str(result))
            alf_list.append(alf)

