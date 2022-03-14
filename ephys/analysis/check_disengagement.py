from ephys_alf_summary import *
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as mpatches

def label_keep_trials(data):
    new_data = pd.DataFrame()
    data['ITI_times'] = data['goCue_trigger_times']-data['start_time']
    data['ITI_times_r'] = data['ITI_times'].rolling(10).median()
    for idx in data.id.unique():
        subdata = data.loc[data['id']==idx]
        subdata['include'] = 1
        try:
            ITIs= np.where((subdata['ITI_times_r']/subdata['ITI_times'][10:100].median())>=2)[0]
            cut_off_ITI = np.array(ITIs)[np.array(ITIs)>100][0]-10
            subdata.loc[subdata['index']>=cut_off_ITI, 'include'] = 0 
        except:
            print('all_trials_included')
        new_data = pd.concat([new_data, subdata])
    return new_data

sessions = ephys_behavior_dataset(LIST_OF_SESSIONS_CHR2_GOOD_REC)
data = sessions.sessions
data['rt'] = data['response_times'] - data['goCue_trigger_times']
data['fm'] = data['first_move'] - data['goCue_trigger_times']
data['ITI_times'] =  data['goCue_trigger_times'] - data['start_time'].shift(1)
data.loc[data['ITI_times']<0, 'ITI_times'] = np.nan
data['id'] = data['mouse']+data['date']
data['rt_r'] = data['rt'].rolling(10).median()
data['fm_r'] = data['fm'].rolling(10,min_periods=1).median()
data['ITI_times_r'] = data['ITI_times'].rolling(10).median()
fig,ax=plt.subplots(5,5, sharey=True, sharex=True)
for i, idx in enumerate(data.id.unique()):
    ax1 = ax[int(i/5),i%5]
    ax2 = ax[int(i/5),i%5].twinx()
    subdata = data.loc[data['id']==idx]
    sns.lineplot(data=subdata, x='index',y='rt_r', ax=ax1)
    sns.lineplot(data=subdata, x='index',y='fm_r', ax=ax1, color='magenta')
    ax1.set_ylabel('Decision Time (s)', color='b')
    plt.sca(ax2)
    sns.lineplot(data=subdata, x='index',y='ITI_times_r', ax=ax2, color='red')
    if i%5==4:
        ax2.set_ylabel('Initiation Time (s)', color='r')
    else:
        ax2.set_ylabel(' ')
    try:
        ITIs= np.where((subdata['ITI_times_r']/subdata['ITI_times'][10:100].median())>=2)[0]
        cut_off_ITI = np.array(ITIs)[np.array(ITIs)>100][0]-10
        plt.vlines(cut_off_ITI, subdata['ITI_times_r'].min(), subdata['ITI_times_r'].max(), linestyles='dashed', color='k')
    except:
        continue

plt.sca(ax[4,4])
colors = ["b", "r", "magenta"]
texts = ["Decision Time", "Initiation Time", "Reaction Time"]
patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]
plt.legend(handles=patches)


fig,ax=plt.subplots(5,5, sharex=True)
for i, idx in enumerate(data.id.unique()):
    ax1 = ax[int(i/5),i%5]
    subdata = data.loc[data['id']==idx]
    sns.lineplot(data=subdata, x='index',y='fm_r', ax=ax1, color='magenta')
    ax1.set_ylabel('Reaction Time (s)', color='k')
plt.tight_layout()
