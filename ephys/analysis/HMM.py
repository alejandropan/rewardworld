# HMM
from os import name
import autograd.numpy as np
import autograd.numpy.random as npr
npr.seed(0)
import ephys_alf_summary as es
import ssm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import matplotlib.patches as mpatches
import pandas as pd

def fit_hmm (obs, num_states=2, obs_dim=1, obs_dist = 'gaussian'):
    # Make an HMM
    hmm = ssm.HMM(num_states, obs_dim, observations=obs_dist)
    hmm_lls = hmm.fit(obs, method="em")
    return hmm, hmm_lls[-1]

def add_relative_trial_time(data):
    data['relative_trial_times'] = np.nan
    for mouse in data.mouse.unique():
        data_m = data.loc[data['mouse']==mouse]
        for date in data_m.date.unique():
            data_s = data_m.loc[data_m['date']==date]
            data.loc[(data['mouse']==mouse)&(data['date']==date),
            'relative_trial_times'] = data_s['index'].to_numpy()/data_s['index'].max()
    return data

def save_states(data,prefix='_ibl_trials.'):
    for mouse in data['mouse'].unique():
        animal = data.loc[data['mouse']==mouse]
        for day in animal['date'].unique():
            day_s = animal.loc[animal['date']==day]
            for ses in day_s['ses'].unique():
                session = day_s.loc[day_s['ses']==ses]
                alf = session['path'].to_list()[0] + '/alf'
                n_trials = len(np.load(alf+'/_ibl_trials.feedbackType.npy'))
                assert n_trials == len(session)
                np.save(alf+'/'+prefix+'states.npy', session['state'].to_numpy())

def calculate_states(data, N_DIMENSIONS=2, METRIC='rank', SINGLE_SESSION_FIT= False, SAVE_STATES=False):
    data = data.copy()
    data['idx'] = data['mouse']+str(data['date'])
    data['rt'] = data['response_times'] - data['goCue_trigger_times']
    data['it'] = data['goCue_trigger_times'] - data['start_time']
    data['mt'] = data['first_move'] - data['goCue_trigger_times']
    data['high_prob'] = np.nan
    data.loc[(data['probabilityLeft']==0.1) & (data['choice_1']==0), 'high_prob'] = 0
    data.loc[(data['probabilityLeft']==0.1) & (data['choice_1']==1), 'high_prob'] = 1
    data.loc[(data['probabilityLeft']==0.7) & (data['choice_1']==1), 'high_prob'] = 0
    data.loc[(data['probabilityLeft']==0.7) & (data['choice_1']==0), 'high_prob'] = 1

    # Z-score Rts and fit hmm
    data['rtz'] = np.nan
    data['itz'] = np.nan
    data['mtz'] = np.nan
    
    if METRIC=='rank':
        for mouse in data.mouse.unique():
            print(mouse)
            for date in  data.loc[data['mouse']==mouse,'date'].unique():
                data.loc[(data['mouse']==mouse)&(data['date']==date),'rtz'] = (data.loc[(data['mouse']==mouse)&(data['date']==date),'rt'].rank().to_numpy()-1) / data.loc[(data['mouse']==mouse)&(data['date']==date),'rt'].rank().max()
                data.loc[(data['mouse']==mouse)&(data['date']==date),'itz'] = (data.loc[(data['mouse']==mouse)&(data['date']==date),'it'].rank().to_numpy()-1) / data.loc[(data['mouse']==mouse)&(data['date']==date),'it'].rank().max()
                print(date)

    if METRIC=='zscore':
        for mouse in data.mouse.unique():
            print(mouse)
            for date in  data.loc[data['mouse']==mouse,'date'].unique():
                data.loc[(data['mouse']==mouse)&(data['date']==date),'rtz'] = zscore(data.loc[(data['mouse']==mouse)&(data['date']==date),'rt'].to_numpy(), nan_policy='omit')
                data.loc[(data['mouse']==mouse)&(data['date']==date),'itz'] = zscore(data.loc[(data['mouse']==mouse)&(data['date']==date),'it'].to_numpy(), nan_policy='omit')
                print(date)

    if SINGLE_SESSION_FIT==True:
        learned_transition_mat = np.zeros([2,2])
        data2=pd.DataFrame()
        counter=0
        true_ll = 0
        for mouse in data.mouse.unique():
            for date in  data.loc[data['mouse']==mouse,'date'].unique():
                data1=data.loc[(data['mouse']==mouse)&(data['date']==date)]
                if N_DIMENSIONS==1:
                    obs = data1.loc[(~np.isnan(data1['rtz']))&(~np.isnan(data1['itz'])),['rtz']].to_numpy()
                    obs = obs.reshape(len(obs),1)
                    hmm,ll = fit_hmm (obs,obs_dim=1)
                if N_DIMENSIONS==2:
                    obs = data1.loc[(~np.isnan(data1['rtz']))&(~np.isnan(data1['itz'])),['rtz','itz']].to_numpy()
                    obs = obs.reshape(len(obs),2)
                    hmm,ll = fit_hmm (obs,obs_dim=2)    
                #Analysis of performence across states
                obs_states = hmm.most_likely_states(obs)
                data1['state']=np.nan
                data1.loc[(~np.isnan(data1['rtz']))&(~np.isnan(data1['itz'])),'state']=obs_states
                learned_transition_mat1 = hmm.transitions.transition_matrix
                if data1.loc[data1['state']==1,'rt'].median()<data1.loc[data1['state']==0,'rt'].median():
                    data1.loc[data1['state']==1,'state'] = 'Engaged'
                    data1.loc[data1['state']==0,'state'] = 'Disengaged'
                    engaged_state=1
                else:
                    data1.loc[data1['state']==1,'state'] = 'Disengaged'
                    data1.loc[data1['state']==0,'state'] = 'Engaged'
                    engaged_state=0
                    learned_transition_mat1 = np.flip(learned_transition_mat1)
                data2=pd.concat([data2,data1])
                learned_transition_mat = learned_transition_mat1 + learned_transition_mat
                counter+=1
                true_ll += hmm.log_probability(obs)
        plot_state_statistics(data2)
        plt.show()
        plot_transition_matrix(learned_transition_mat/counter,engaged_state)
        plt.show()
        plot_transitions(data2)
        plt.show()
        if SAVE_STATES==True:
            save_states(data2)
        return data2, true_ll
    if SINGLE_SESSION_FIT==False:
        if N_DIMENSIONS==1:
            obs = data.loc[(~np.isnan(data['rtz']))&(~np.isnan(data['itz'])),['rtz']].to_numpy()
            obs = obs.reshape(len(obs),1)
            hmm,ll = fit_hmm (obs,obs_dim=1)
        if N_DIMENSIONS==2:
            obs = data.loc[(~np.isnan(data['rtz']))&(~np.isnan(data['itz'])),['rtz','itz']].to_numpy()
            obs = obs.reshape(len(obs),2)
            hmm,ll = fit_hmm (obs,obs_dim=2)
        #Analysis of performence across states
        obs_states = hmm.most_likely_states(obs)
        data['state']=np.nan
        data.loc[(~np.isnan(data['rtz']))&(~np.isnan(data['itz'])),'state']=obs_states
        if data.loc[data['state']==1,'rt'].median()<data.loc[data['state']==0,'rt'].median():
            data.loc[data['state']==1,'state'] = 'Engaged'
            data.loc[data['state']==0,'state'] = 'Disengaged'
            engaged_state=1
        else:
            data.loc[data['state']==1,'state'] = 'Disengaged'
            data.loc[data['state']==0,'state'] = 'Engaged'
            engaged_state=0
        data.groupby(['state']).mean()['outcome']
        learned_transition_mat = hmm.transitions.transition_matrix
        true_ll = hmm.log_probability(obs)
        plot_state_statistics(data)
        plt.show()
        plot_transition_matrix(learned_transition_mat,engaged_state)
        plt.show()
        plot_transitions(data)
        plt.show()
        if SAVE_STATES==True:
            save_states(data)
        return data, true_ll

def plot_state_statistics(data):
    fig, ax = plt.subplots(2,3)
    plt.sca(ax[0,0])
    dataplot = data.groupby(['state','opto_block']).count()['high_prob'].reset_index()
    sns.barplot(data=dataplot,
                x='state', y='high_prob',
                hue='opto_block', palette=['dodgerblue','orange'], alpha=0.75)
    plt.xlabel('State')
    plt.ylabel('No Trials')
    plt.sca(ax[0,1])
    sns.histplot(data=data.loc[data['state']=='Engaged'],
                x='rt', 
                stat='percent', binwidth=0.4, color='k')
    sns.histplot(data=data.loc[data['state']=='Disengaged'],
                x='rt', 
                stat='percent', binwidth=0.4, color='gray')
    plt.xlabel('Decision Time')
    plt.ylabel('Relative % Trials')
    plt.xlim(0,10)
    colors = ['k', 'gray']
    texts = ['Engaged', 'Disengaged']
    patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]
    plt.legend(handles=patches)
    plt.sca(ax[0,2])
    data=add_relative_trial_time(data)
    sns.histplot(data=data.loc[data['state']=='Engaged'],
                x='relative_trial_times', 
                stat='percent', bins=100, color='k')
    sns.histplot(data=data.loc[data['state']=='Disengaged'],
                x='relative_trial_times', 
                stat='percent', bins=100, color='gray')
    plt.xlabel('Relative Trial Time (%)')
    plt.ylabel('Relative % Trials')
    colors = ['k', 'gray']
    texts = ['Engaged', 'Disengaged']
    patches = [ mpatches.Patch(color=colors[i], label="{:s}".format(texts[i]) ) for i in range(len(texts)) ]
    plt.legend(handles=patches)
    sns.despine()
    plt.sca(ax[1,0])
    sns.pointplot(data=data, x='state', y='high_prob', hue='opto_block', ci=68,
                    palette=['dodgerblue','orange'], alpha=0.75, order=['Disengaged', 'Engaged'])
    plt.ylabel('Fraction of High Probability Choices')
    plt.sca(ax[1,1])
    sns.histplot(data=data.loc[data['opto_block']==0],
                x='trial_within_block_real', 
                hue='state',stat="percent", discrete=True,
                palette='Blues')
    plt.xlabel('Trial within block')
    plt.ylabel('Trials (%)')
    plt.sca(ax[1,2])
    sns.histplot(data=data.loc[data['opto_block']==1],
                x='trial_within_block_real', 
                hue='state',stat="percent", discrete=True,
                palette='YlOrBr')
    plt.xlabel('Trial within block')
    plt.ylabel('Trials (%)')

def plot_transition_matrix(learned_transition_mat,engaged_state=1):
    fig = plt.figure()
    im = plt.imshow(learned_transition_mat, cmap='gray')
    plt.title("Transition Matrix")
    if engaged_state==1:
        plt.xticks([0,1],['Disengaged', 'Engaged'])
        plt.yticks([0,1],['Disengaged', 'Engaged'])      
    else:
        plt.xticks([0,1],['Disengaged', 'Engaged'])
        plt.yticks([0,1],['Disengaged', 'Engaged'])
    fig.colorbar(im)

def plot_transitions(data):
    data['id'] = data['mouse']+data['date']
    fig,ax=plt.subplots(5,5, sharey=True, sharex=True)
    for i, idx in enumerate(data.id.unique()):
        ax1 = ax[int(i/5),i%5]
        subdata = data.loc[data['id']==idx]
        sns.scatterplot(data=subdata, x='index',y='rt', hue='state', ax=ax1, palette={'Engaged':'magenta','Disengaged':'k'})
        plt.ylim(0,5)

if __name__=="__main__":
    data = es.ephys_behavior_dataset(es.LIST_OF_SESSIONS_CHR2_GOOD_REC).sessions
    # RANK
    _, rank_2_multi = calculate_states(data, N_DIMENSIONS=2, METRIC='rank', SINGLE_SESSION_FIT= False, SAVE_STATES=False)
    _, rank_1_multi = calculate_states(data, N_DIMENSIONS=1, METRIC='rank', SINGLE_SESSION_FIT= False, SAVE_STATES=False)
    _, rank_2_single = calculate_states(data, N_DIMENSIONS=2, METRIC='rank', SINGLE_SESSION_FIT= True, SAVE_STATES=False)
    _, rank_1_single = calculate_states(data, N_DIMENSIONS=1, METRIC='rank', SINGLE_SESSION_FIT= True, SAVE_STATES=False)
    # ZSCORE
    _, zscore_2_multi = calculate_states(data, N_DIMENSIONS=2, METRIC='zscore', SINGLE_SESSION_FIT= False, SAVE_STATES=False)
    _, zscore_1_multi = calculate_states(data, N_DIMENSIONS=1, METRIC='zscore', SINGLE_SESSION_FIT= False, SAVE_STATES=False)
    _, zscore_2_single = calculate_states(data, N_DIMENSIONS=2, METRIC='zscore', SINGLE_SESSION_FIT= True, SAVE_STATES=False)
    _, zscore_1_single = calculate_states(data, N_DIMENSIONS=1, METRIC='zscore', SINGLE_SESSION_FIT= True, SAVE_STATES=False)

scores = pd.DataFrame()
scores['LL'] = [rank_2_multi,rank_1_multi,rank_2_single,rank_1_single,zscore_2_multi,zscore_1_multi,zscore_2_single,zscore_1_single]
scores['metric'] = ['rank','rank','rank','rank','zscore','zscore','zscore','zscore']
scores['name'] = ['2_multi','1_multi','2_single','1_single','2_multi','1_multi','2_single','1_single']

fig, ax = plt.subplots(1,2, sharey=True)
plt.sca(ax[0])
sns.pointplot(data=scores.loc[scores['metric']=='rank'].sort_values(by=['LL']),x='name', y='LL', color='k')
plt.title('percentile')
plt.sca(ax[1])
sns.pointplot(data=scores.loc[scores['metric']=='zscore'].sort_values(by=['LL']),x='name', y='LL', color='k')
plt.title('zscore')
