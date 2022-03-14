from logging import PlaceHolder
import pandas as pd
import numpy as np
from pathlib import Path
import os
from ibllib.io.extractors.biased_trials import extract_all
from full_bandit_fix import full_bandit_fix 
from session_summary_10 import *
import one.alf as alf
from ibllib.io.raw_data_loaders import load_settings
import zipfile


def mouse_data_loader(rootdir):
    '''
    rootdir (str): mouse directory
    variables (list): list containing the keys of the variables of interest
    Will extract and load data from the whole life of animal
    '''
    mouse_df = pd.DataFrame()
    counter = 0
    for file in sorted(os.listdir(rootdir)):
        d = os.path.join(rootdir, file)
        if os.path.isdir(d):
            print(d)
            day_df = pd.DataFrame()
            counter += 1
            if counter>70:
                break
            for ses in sorted(os.listdir(d)):
                s = os.path.join(d, ses)
                if os.path.isdir(s):
                    try:
                        if Path(s+'/alf').is_dir()==False:
                            extract_all(s, save=True)
                            if Path(s+'/alf/probe00').is_dir()==False:    
                                full_bandit_fix(s)
                        ses_df= pd.DataFrame()
                        sesdata  = alf.io.load_object(s+'/alf', 'trials')
                        del sesdata['intervals']
                        ses_df= pd.DataFrame.from_dict(sesdata)
                        protocol = load_settings(s)['_PROTOCOL']
                        if protocol=='_bandit_100_0_biasedChoiceWorld': # check GABOR file for shaping step
                            with zipfile.ZipFile(s+'/raw_behavior_data/_iblrig_taskCodeFiles.raw.zip') as gaborzip:
                                with gaborzip.open('GaborIBLTask/Gabor2D.bonsai') as bonsaifile:
                                    bonsaicode = bonsaifile.read().find(b'it * 5<') #Line of code that makes the GABORs different
                            if bonsaicode==-1:
                                protocol='_bandit_100_0_biasedChoiceWorld'+'_equal_stim'
                            else:
                                protocol='_bandit_100_0_biasedChoiceWorld'+'_different_stim'
                        ses_df['protocol'] = protocol
                        day_df = pd.concat([day_df,ses_df])
                    except:
                        continue
            day_df['day'] = counter
            mouse_df = pd.concat([mouse_df,day_df])
    return mouse_df

ROOT = '/Volumes/witten/Alex/Data/Subjects/'
MICE = ['DChR2_1','DChR2_2','dop_40','dop_45','dop_46','dop_36','dop_31','dop_30']
data=pd.DataFrame()
for mouse in MICE:
    mouse_df = mouse_data_loader(ROOT+mouse)
    mouse_df['mouse'] = mouse
    data = pd.concat([data, mouse_df])
data['feedbackType'] = 1*(data['feedbackType']>0)
data = data.reset_index()

sub_data = data.loc[(data['protocol']=='_bandit_100_0_biasedChoiceWorld_equal_stim') |
    (data['protocol']=='_bandit_100_0_biasedChoiceWorld_different_stim')]
sns.lineplot(data=sub_data,x='day',y='feedbackType',hue='mouse', ci=0)
plt.xlabel('Training Day')
plt.ylabel('Fraction Correct')
plt.ylim(0,1)
plt.hlines(0.75, 0, 30,linestyles='--')




## analysis of bandit step

select_data = data.loc[(data['protocol']=='_bandit_biasedChoiceWorld')]
select_data['high_probability_choice'] = 0
select_data.loc[(select_data['choice']==-1)&(select_data['probabilityLeft']==0.1),'high_probability_choice']=1
select_data.loc[(select_data['choice']==1)&(select_data['probabilityLeft']==0.7),'high_probability_choice']=1


#######
norm_data = pd.DataFrame()
for mouse in select_data.mouse.unique():
    day = select_data.loc[(select_data['mouse']==mouse)].groupby(['day']).mean().reset_index()
    days = day['day'].iloc[np.where(day['day'].diff()==1)[0]].to_numpy()
    sub_data = select_data.loc[(select_data['mouse']==mouse)&(np.isin(select_data['day'],days))]
    sub_data['day'] = sub_data['day'] - days[0]
    norm_data=  pd.concat([norm_data,sub_data])


sns.lineplot(data=norm_data,x='day',y='high_probability_choice',hue='mouse', ci=0)
plt.xlabel('Training Day')
plt.ylabel('High Prob Choice')
plt.ylim(0,1)
plt.xlim(0,80)
plt.hlines(0.5, 0, 30,linestyles='--')
#########

select_data['norm_day'] = np.nan
for mouse in select_data.mouse.unique():
    for i, ses in enumerate(sorted(select_data.loc[select_data['mouse']==mouse,'day'].unique())):
        select_data.loc[(select_data['mouse']==mouse)&(select_data['day']==ses),'norm_day'] = i


sns.lineplot(data = select_data, x='norm_day',y='high_probability_choice', hue='mouse', ci=0)
plt.xlabel('Training Day')
plt.ylabel('High Prob Choice')
plt.ylim(0,1)
plt.xlim(0,40)
plt.hlines(0.5, 0, 30,linestyles='--')


MICE = ['dop_38','dop_24','dop_36','dop_31','dop_30']
