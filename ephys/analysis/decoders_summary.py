import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


# Hardcoded parameters
decoders_path = '/Volumes/witten/Alex/decoder_output'
n_neuron_combos_tried = np.array([10,20,30,50])
pre_time = -0.5
post_time = 3
bin_size = 0.1
# First list all the "real"sessions
files = []
f=[]
for i in os.listdir(decoders_path):
    if os.path.isfile(os.path.join(decoders_path,i)) and 'real' in i:
        files.append(i)
for file in files:
    if 'p_summary' in file:
        f.append(file)
# generate big dataframe
os.chdir(decoders_path)
summary = pd.DataFrame()
for f_path in tqdm(f):
    p_summary = np.load(f_path)
    mse_summary = np.load(f_path[:-13]+'mse_summary.npy')
    # Find optimal lambda 
    l_performance = np.nanmean(np.nanmean(np.nanmean(p_summary, axis=0), axis=1), axis=2)
    l_all = np.argmax(l_performance, axis=0)
    # Get summary with optimal lambda
    acc = pd.DataFrame()
    for c in np.arange(np.shape(p_summary)[3]):
        l = l_all[c]
        predict = []
        predict_mse = []
        for b in np.arange(np.shape(p_summary)[2]):
            predict.append(np.nanmean(p_summary[:,l,b,c,:]))
            predict_mse.append(np.nanmean(mse_summary[:,l,b,c,:]))
        acc_combo = pd.DataFrame()
        acc_combo['r'] = predict
        acc_combo['mse'] = predict_mse
        acc_combo['time_bin'] = np.arange(np.shape(p_summary)[2])
        acc_combo['n_neurons'] = n_neuron_combos_tried[c]
        acc = pd.concat([acc,acc_combo])
    acc['region'] = f_path[5:-34]
    acc['mouse'] = f_path[-33:-27]
    acc['hemisphere'] = f_path[-15:-14]
    acc['date'] = f_path[-26:-16]
    acc['type'] = 'real'
    acc['ses_n'] = f_path[0]
    acc['id'] =   acc['region'] + acc['type'] + \
        acc['date'] +acc['mouse'] + acc['hemisphere'] + acc['ses_n']
    summary = pd.concat([summary,acc])
summary  = summary.reset_index()

# Start plotting

# Plot by region  at 30 cells
not_in_summary = ['Other','Thalamus','ZI','SSs','Amygdala','MO', 'Pallidum']
neuron_summary = summary.loc[summary['n_neurons']==10]
to_exclude = neuron_summary.loc[neuron_summary['time_bin']==0]
to_exclude = to_exclude.groupby(['region']).count()['id'].reset_index()
to_exclude = to_exclude.loc[to_exclude['id']==1, 'region'].tolist() # exclude where there is only one recording
to_exclude = np.concatenate([not_in_summary,to_exclude])

fig, ax = plt.subplots(2,1)
plt.sca(ax[0])
sns.lineplot(data=neuron_summary.loc[~np.isin(neuron_summary['region'],to_exclude)], x='time_bin', y='r', hue='region', errorbar='se',
            palette='tab10')
# sns.lineplot(data=neuron_summary, x='time_bin', y='r', hue='region', errorbar='se')
plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[::5],2), rotation=90)
plt.xlabel('Time from epoch')
plt.ylabel('Pearson - r')
stats = neuron_summary.loc[~np.isin(neuron_summary['region'],to_exclude)].groupby(['region']).max()['r'].reset_index()
winner = stats.loc[stats['r'] == stats['r'].max(),'region'].tolist()[0]
print('The r winner is ' + winner)
plt.sca(ax[1])
sns.lineplot(data=neuron_summary.loc[~np.isin(neuron_summary['region'],to_exclude)], x='time_bin', y='mse', hue='region', errorbar='se',
            palette='tab10')
# sns.lineplot(data=neuron_summary, x='time_bin', y='r', hue='region', errorbar='se')
plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[::5],2), rotation=90)
plt.xlabel('Time from epoch')
plt.ylabel('MSE')
stats = neuron_summary.loc[~np.isin(neuron_summary['region'],to_exclude)].groupby(['region']).min()['mse'].reset_index()
winner = stats.loc[stats['mse'] == stats['mse'].min(),'region'].tolist()[0]
print('The mse winner is ' + winner)
sns.despine()


# By region
select_regions = neuron_summary.loc[~np.isin(neuron_summary['region'],to_exclude), 'region'].unique()
pal = dict(zip(neuron_summary.id.unique(), sns.color_palette('cubehelix', len(neuron_summary.id.unique()))))
fig,ax  = plt.subplots(5,2, sharey=True, sharex=True)
for i, reg in enumerate(select_regions):
    plt.sca(ax[i%5,int(i/5)])
    region_data = neuron_summary.loc[neuron_summary['region']==reg]
    sns.lineplot(data=region_data, x='time_bin', y='r', hue='id', palette=pal)
    plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[::5],2), rotation=90)
    plt.xlabel('Time from epoch')
    plt.ylabel('Pearson - r')    
    plt.title(reg)
    plt.legend().remove()
    sns.despine()
plt.tight_layout()

fig,ax  = plt.subplots(5,2, sharey=True, sharex=True)
for i, reg in enumerate(select_regions):
    plt.sca(ax[i%5,int(i/5)])
    region_data = neuron_summary.loc[neuron_summary['region']==reg]
    sns.lineplot(data=region_data, x='time_bin', y='r', errorbar='se')
    plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[::5],2), rotation=90)
    plt.xlabel('Time from epoch')
    plt.ylabel('Pearson - r')    
    plt.title(reg)
    plt.legend().remove()
    sns.despine()
plt.tight_layout()



# Plot n_neurons vs max_p for regions that have 50 cells
# Plot n_neurons vs max_p for regions that have 50 cells
not_in_summary = ['Other','Thalamus','ZI','SSs','Amygdala', 'Pallidum']
to_exclude = neuron_summary.loc[neuron_summary['time_bin']==0]
to_exclude = to_exclude.groupby(['region']).count()['id'].reset_index()
to_exclude = to_exclude.loc[to_exclude['id']==1, 'region'].tolist() # exclude where there is only one recording
to_exclude = np.concatenate([not_in_summary,to_exclude])

summ = neuron_summary.loc[~np.isin(neuron_summary['region'],to_exclude)]
summ = summ.loc[(summ['time_bin']>5) & (summ['time_bin']<15)]

fig, ax = plt.subplots(2)
plt.sca(ax[0])
nsum = summ.groupby(['region','n_neurons','id']).mean()['r'].reset_index()
sns.pointplot(data=nsum, x='n_neurons', y='r', hue='region', errorbar='se')
sns.pointplot(data=nsum, x='n_neurons', y='r', errorbar='se', color='k')
plt.ylabel('Mean r')
plt.xlabel('decoding neurons')
plt.legend().remove()
plt.sca(ax[1])
nsum = summ.groupby(['region','n_neurons','id']).max()['r'].reset_index()
sns.pointplot(data=nsum, x='n_neurons', y='r', hue='region', errorbar='se')
sns.pointplot(data=nsum, x='n_neurons', y='r', errorbar='se', color='k')
plt.ylabel('Max r')
plt.xlabel('decoding neurons')
sns.despine()

fig, ax = plt.subplots(2)
plt.sca(ax[0])
nsum = summ.groupby(['region','n_neurons','id']).mean()['r'].reset_index()
sns.pointplot(data=nsum, x='n_neurons', y='r', errorbar='se', color='k')
plt.ylabel('Mean r')
plt.xlabel('decoding neurons')
plt.legend().remove()
plt.sca(ax[1])
nsum = summ.groupby(['region','n_neurons','id']).max()['r'].reset_index()
sns.pointplot(data=nsum, x='n_neurons', y='r', errorbar='se', color='k')
plt.ylabel('Max r')
plt.xlabel('decoding neurons')
sns.despine()


# Lambdas summary
l = np.logspace(-3,-0.5,100)
os.chdir(decoders_path)
lambdas_selected = []
for f_path in tqdm(f):
    p_summary = np.load(f_path[:-13]+'mse_summary.npy')
    # Find optimal lambda 
    l_performance = np.nanmean(np.nanmean(np.nanmean(p_summary, axis=0), axis=1), axis=2)
    l_all = np.argmin(l_performance, axis=0)
    lambdas_selected.append(l_all)
lambdas_selected = np.concatenate(lambdas_selected)

# Plot null sessions against OFC for 30 neurons
files_null = []
f_null=[]
for i in os.listdir(decoders_path):
    if os.path.isfile(os.path.join(decoders_path,i)) and 'null' in i:
        files_null.append(i)
for file in files_null:
    if 'p_summary' in file:
        if 'OFC' in file:
            f_null.append(file)

nsummary = pd.DataFrame()
for f_path in tqdm(f_null):
    p_summary = np.load(f_path)
    acc = pd.DataFrame()
    for c in np.arange(np.shape(p_summary)[3]):
        predict = []
        for b in np.arange(np.shape(p_summary)[2]):
            predict.append(np.nanmean(p_summary[:,:,b,c,:]))
        acc_combo = pd.DataFrame()
        acc_combo['r'] = predict
        acc_combo['time_bin'] = np.arange(np.shape(p_summary)[2])
        acc_combo['n_neurons'] = n_neuron_combos_tried[c]
        acc = pd.concat([acc,acc_combo])
    acc['region'] = f_path[5:-34]
    acc['mouse'] = f_path[-33:-27]
    acc['hemisphere'] = f_path[-15:-14]
    acc['date'] = f_path[-26:-16]
    acc['type'] = 'real'
    acc['ses_n'] = f_path[0:3]
    acc['id'] =   acc['region'] + acc['type'] + \
        acc['date'] +acc['mouse'] + acc['hemisphere'] + acc['ses_n']
    nsummary = pd.concat([nsummary,acc])
nsummary  = nsummary.reset_index()


# Plot null vs real
ROI='OFC'
ROI_summary = summary.loc[(summary['n_neurons']==30) & (summary['region']==ROI)]
null_summary = nsummary.loc[(nsummary['n_neurons']==30) ]
sns.lineplot(data=ROI_summary, x='time_bin', y='r', errorbar='se', color='r')
sns.lineplot(data=null_summary, x='time_bin', y='r', errorbar='se', color='k')
plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[::5],2), rotation=90)
plt.xlabel('Time from outcome')
plt.ylabel('Pearson - r')
sns.despine()