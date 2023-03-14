from logging import PlaceHolder
from tkinter.ttk import Style
import pandas as pd
import numpy as np
from pathlib import Path
import os
from ibllib.io.extractors.biased_trials import extract_all
import rewardworld.behavior_analysis.bandit_version.full_bandit_fix as full_bandit_fix
from rewardworld.behavior_analysis.bandit_version.session_summary_10 import *
import one.alf as alf
from ibllib.io.raw_data_loaders import load_settings
import zipfile
#from investigating_laser_expdecaymodel import *




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
            if counter>80:
                break
            for ses in sorted(os.listdir(d)):
                s = os.path.join(d, ses)
                if os.path.isdir(s):
                    try:
                        if Path(s+'/raw_ephys_data').is_dir()==False:
                            extract_all(s, save=True, settings={'IBLRIG_VERSION_TAG': '4.9.0'})
                            try:
                                os.remove(s+'/alf/_ibl_trials.table.pqt') 
                            except:
                                print('np pq table')
                            if Path(s+'/alf/probe00').is_dir()==False:    
                                full_bandit_fix.full_bandit_fix(s)     
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
                        if len(ses_df)<100:
                            continue
                        else:
                            day_df = pd.concat([day_df,ses_df])
                    except:
                        continue
            day_df['day'] = counter
            mouse_df = pd.concat([mouse_df,day_df])
    return mouse_df

ROOT = '/Volumes/witten/Alex/Data/Subjects/'
#MICE = ['dop_54']
MICE = ['dop_47', 'dop_50','dop_51','dop_52','dop_53']
data=pd.DataFrame()
for mouse in MICE:
    mouse_df = mouse_data_loader(ROOT+mouse)
    mouse_df['mouse'] = mouse
    data = pd.concat([data, mouse_df])
data['feedbackType'] = 1*(data['feedbackType']>0)
data = data.reset_index()

# Palette
fig,ax = plt.subplots(1,2)
# Analysis 100_0 step
plt.sca(ax[0])
sub_data = data.loc[(data['protocol']=='_bandit_100_0_biasedChoiceWorld_different_stim')]
sns.lineplot(data=sub_data,x='day',y='feedbackType',hue='mouse', errorbar=None, palette='Greys')
plt.xlabel('Training Day')
plt.ylabel('Fraction Correct')
plt.xlim(0,8)
plt.ylim(0,1)
sns.despine()
plt.hlines(0.70, 0, sub_data.day.max(),linestyles='--')
plt.sca(ax[1])
## analysis of bandit step
select_data = data.loc[(data['protocol']=='_bandit_biasedChoiceWorld')]
select_data['high_probability_choice'] = 0
select_data.loc[(select_data['choice']==-1)&(select_data['probabilityLeft']==0.1),'high_probability_choice']=1
select_data.loc[(select_data['choice']==1)&(select_data['probabilityLeft']==0.7),'high_probability_choice']=1
select_data['norm_day'] = np.nan
for mouse in select_data.mouse.unique():
    for i, ses in enumerate(sorted(select_data.loc[select_data['mouse']==mouse,'day'].unique())):
        select_data.loc[(select_data['mouse']==mouse)&(select_data['day']==ses),'norm_day'] = i
sns.lineplot(data = select_data, x='norm_day',y='high_probability_choice', hue='mouse', ci=0, palette=pal)
plt.xlabel('Training Day')
plt.ylabel('High Prob Choice')
plt.ylim(0,1)
plt.hlines(0.5, 0, select_data.day.max(),linestyles='--')

###############
###############
###############

# Analysis of laser versions
laser_data = data.loc[(data['protocol']=='_bandit_laser_blocks_ephysChoiceWorld') |
    (data['protocol']=='_bandit_laser_blocks_cued_ephysChoiceWorld')]

laser_data['laser_type'] = '8mW'
for animal in laser_data.mouse.unique():
    cued_days =  laser_data.loc[(laser_data['mouse']==animal) & (laser_data['protocol']=='_bandit_laser_blocks_ephysChoiceWorld')].day.unique()
    low_power_days = laser_data.loc[(laser_data['mouse']==animal) & (laser_data['protocol']=='_bandit_laser_blocks_cued_ephysChoiceWorld')].day.unique()
    laser_data.loc[(laser_data['mouse']==animal) & np.isin(laser_data['day'], cued_days), 'laser_type'] = '4mW+cue'
    laser_data.loc[(laser_data['mouse']==animal) & np.isin(laser_data['day'], low_power_days), 'laser_type'] = '4mW'

laser_data['high_probability_choice'] = 0
laser_data.loc[(laser_data['choice']==-1)&(laser_data['probabilityLeft']==0.1),'high_probability_choice']=1
laser_data.loc[(laser_data['choice']==1)&(laser_data['probabilityLeft']==0.7),'high_probability_choice']=1

laser_data['repeat'] = 0
laser_data.loc[laser_data['choice']==laser_data['choice'].shift(1),'repeat']=1
laser_data['after_reward'] = laser_data['feedbackType'].shift(1)


laser_data['norm_day'] = np.nan
laser_data['period'] = np.nan
laser_data_idx = pd.DataFrame()
for mouse in laser_data.mouse.unique():
    for i, ses in enumerate(sorted(laser_data.loc[laser_data['mouse']==mouse,'day'].unique())):
        ses = laser_data.loc[(laser_data['mouse']==mouse)&(laser_data['day']==ses)].reset_index()
        ses['norm_day'] = i
        to_remove = np.concatenate([np.where(ses.probabilityLeft.diff())[0],
                    np.where(ses.probabilityLeft.diff())[0]+1,
                    np.where(ses.probabilityLeft.diff())[0]+2,
                    np.where(ses.probabilityLeft.diff())[0]+3,
                    np.where(ses.probabilityLeft.diff())[0]+4,
                    np.where(ses.probabilityLeft.diff())[0]+5])
        ses = ses.loc[~(np.isin(ses.index,to_remove))]
        ses.loc[(ses['index']<150),'period'] = 'start'
        ses.loc[(ses['index']>ses['index'].max()-150),'period'] = 'end'
        laser_data_idx = pd.concat([laser_data_idx,ses])
laser_data = laser_data_idx

### Mouse progress

# Plot all
fig,ax=plt.subplots(1,3)
plt.sca(ax[0])
reduced_laser_data = laser_data.groupby(['mouse','norm_day','opto_block','laser_type']).mean()['high_probability_choice'].reset_index()
reduced_laser_data = reduced_laser_data.loc[reduced_laser_data['opto_block']==1]

for animal in reduced_laser_data.mouse.unique():
         reduced_laser_data.loc[(reduced_laser_data['mouse']==animal), 'norm_day'] = np.arange(len(reduced_laser_data.loc[
         (reduced_laser_data['mouse']==animal), 'norm_day']))

pals = ['blue', 'red', 'gray']
for i, laser_type in enumerate(reduced_laser_data.laser_type.unique()):
        for mouse in reduced_laser_data.mouse.unique():
            reduced_laser_data_type =  reduced_laser_data.loc[(reduced_laser_data['laser_type']==laser_type) &
                                                                (reduced_laser_data['mouse']==mouse)]
            sns.lineplot(data=reduced_laser_data.loc[(reduced_laser_data['mouse']==mouse)], x='norm_day', y='high_probability_choice', 
                        color='k', ci=0)
            sns.scatterplot(data=reduced_laser_data_type, x='norm_day',y='high_probability_choice', color = pals[i], ci=0, s=100)
            plt.ylabel('Fraction of correct choices')
            plt.ylim(0,1)

plt.hlines(0.5,0,11,linestyles='dashed', color='k')
plt.title('All trials')
sns.despine()

plt.sca(ax[1])
reduced_laser_data = laser_data.loc[laser_data['period']=='start'].groupby(['mouse','norm_day','opto_block','laser_type']).mean()['high_probability_choice'].reset_index()
reduced_laser_data = reduced_laser_data.loc[reduced_laser_data['opto_block']==1]

for animal in reduced_laser_data.mouse.unique():
         reduced_laser_data.loc[(reduced_laser_data['mouse']==animal), 'norm_day'] = np.arange(len(reduced_laser_data.loc[
         (reduced_laser_data['mouse']==animal), 'norm_day']))

pals = ['blue', 'red', 'gray']
for i, laser_type in enumerate(reduced_laser_data.laser_type.unique()):
        for mouse in reduced_laser_data.mouse.unique():
            reduced_laser_data_type =  reduced_laser_data.loc[(reduced_laser_data['laser_type']==laser_type) &
                                                                (reduced_laser_data['mouse']==mouse)]
            sns.lineplot(data=reduced_laser_data.loc[(reduced_laser_data['mouse']==mouse)], x='norm_day', y='high_probability_choice', 
                        color='k', ci=0)
            sns.scatterplot(data=reduced_laser_data_type, x='norm_day',y='high_probability_choice', color = pals[i], ci=0, s=100)
            plt.ylabel('Fraction of correct choices')
            plt.ylim(0,1)
plt.hlines(0.5,0,11,linestyles='dashed', color='k')
plt.title('First 150')
sns.despine()
plt.sca(ax[2])
reduced_laser_data = laser_data.loc[laser_data['period']=='end'].groupby(['mouse','norm_day','opto_block','laser_type']).mean()['high_probability_choice'].reset_index()
reduced_laser_data = reduced_laser_data.loc[reduced_laser_data['opto_block']==1]

for animal in reduced_laser_data.mouse.unique():
         reduced_laser_data.loc[(reduced_laser_data['mouse']==animal), 'norm_day'] = np.arange(len(reduced_laser_data.loc[
         (reduced_laser_data['mouse']==animal), 'norm_day']))

pals = ['blue', 'red', 'gray']
for i, laser_type in enumerate(reduced_laser_data.laser_type.unique()):
        for mouse in reduced_laser_data.mouse.unique():
            reduced_laser_data_type =  reduced_laser_data.loc[(reduced_laser_data['laser_type']==laser_type) &
                                                                (reduced_laser_data['mouse']==mouse)]
            sns.lineplot(data=reduced_laser_data.loc[(reduced_laser_data['mouse']==mouse)], x='norm_day', y='high_probability_choice', 
                        color='k', ci=0)
            sns.scatterplot(data=reduced_laser_data_type, x='norm_day',y='high_probability_choice', color = pals[i], ci=0, s=100)
            plt.ylabel('Fraction of correct choices')
            plt.ylim(0,1)
plt.hlines(0.5,0,11,linestyles='dashed', color='k')
plt.title('Last 150')
sns.despine()



### Mouse comparisons
reduced_laser_data = laser_data.groupby(['mouse','norm_day','opto_block','laser_type','period']).mean()['high_probability_choice'].reset_index()
reduced_laser_data = reduced_laser_data.loc[reduced_laser_data['opto_block']==1]
reduced_laser_data_plot = reduced_laser_data.loc[reduced_laser_data['period']=='start']
reduced_laser_data_plot['end_performance'] = reduced_laser_data.loc[reduced_laser_data['period']=='end','high_probability_choice'].to_numpy()

sns.scatterplot(data=reduced_laser_data_plot, x='high_probability_choice', y='end_performance', hue='laser_type', style='mouse', palette=pals)
plt.plot([0,1],[0,1], linestyle='dashed', color='k')
plt.xlim(0.1,0.9)
plt.ylim(0.1,0.9)
plt.ylabel('End Laser Performance')
plt.xlabel('Start Laser Performance')
sns.despine()
plt.fill([0,0.5,0.5,0],[0,0,0.5,0.5], color='k', alpha=0.1)


# Transition analysis
laser_data = data.loc[(data['protocol']=='_bandit_laser_blocks_cued_ephysChoiceWorld')]
laser_data['laser_block'] = laser_data['opto_block'] 
laser_data = trial_within_block(laser_data.reset_index())
laser_data = add_transition_info(laser_data, trials_forward=10)

fig,ax  =plt.subplots(1,3)
plt.sca(ax[0])
plot_transition(laser_data.iloc[:,1:])
plt.sca(ax[1])
plot_transition(laser_data.loc[laser_data['index']<150].iloc[:,1:])
plt.sca(ax[2])
plot_transition(laser_data.loc[laser_data['index']>150].iloc[:,1:])


# Performance by bin
pal = ['dodgerblue', 'orange']
laser_data['bin'] = pd.cut(laser_data['index'],13)
sns.pointplot(data = laser_data, x='bin', y='high_probability_choice', hue='opto_block', palette=pal, ci=66)
plt.hlines(0.5,0,13, linestyles='--', color='k')
plt.ylim(0.25,0.75)
plt.xticks(np.arange(13),np.arange(13))
plt.xlabel('50 trial bins')



# 
sub_data = data.loc[np.isin(data['mouse'],MICE)]
sns.pointplot(data=sub_data,x='day',y='feedbackType',hue='mouse', errorbar=None, palette='Reds')
plt.xlabel('Training Day')
plt.ylabel('Fraction Correct')
plt.xlim(0,9)
plt.ylim(0,1)
sns.despine()
plt.hlines(0.695, 0, sub_data.day.max(),linestyles='--', color='r')
plt.hlines(0.5, 0, sub_data.day.max(),linestyles='--', color= 'k')

