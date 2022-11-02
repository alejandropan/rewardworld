import pandas as pd
import glob
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from decoders_summary import load_decoders
from scipy.stats import pearsonr

pals = dict({'OFC':'#6495ED',
    'NAcc':'#7FFF00', 
    'PFC':'#7AC5CD',
    'DMS':'#76EEC6',
    'VPS':'#3D9140',
    'VP':'#F08080',
    'SNr':'#8B1A1A',
    'Olfactory':'#838B8B',
    'DLS':'#9BCD9B',
    'GPe':'#FF3030'})


decoder_path = '/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget'
d = load_decoders(decoder_path, var = 'qchosen_pre', epoch = 'cue', x_type = 'residuals')

os.chdir('/Volumes/witten/Alex/decoders_residuals_results/accuracy_by_region')
files = glob.glob('*.csv')
concat_data = pd.DataFrame()
for f in files:
    concat_data=pd.concat([concat_data, pd.read_csv(f).reset_index()])

neuron_selections = [10,20,30,50]
fig,ax = plt.subplots(1,len(neuron_selections))
for i in np.arange(len(neuron_selections)):
    plt.sca(ax[i])
    sns.barplot(data=concat_data.loc[concat_data['n_neurons']>=neuron_selections[i]], x = 'region', y = 'choice_accuracy', errorbar='se')
    plt.xticks(rotation=90)
    sns.despine()
    plt.ylim(0.5,1)
    plt.ylabel('Model Accuracy')
    plt.title('n_neurons %s' %str(neuron_selections[i]))


# Make correlation
accus = []
for i in np.arange(len(d)):
    select = d.iloc[i,:]
    accus.append(concat_data.loc[(concat_data['mouse']==select['mouse'])& (concat_data['ses']==select['date']),'choice_accuracy'].to_numpy()[0])
d['accu'] = accus


# Make selection and average of r for important period
d_select = d.loc[d['time_bin']>5]
d_select = d_select.groupby(['id','region','n_neurons']).mean()[['accu', 'r']].reset_index()
selected_regions = np.array(['OFC', 'NAcc', 'PFC', 'DMS', 'VPS', 'VP', 'SNr','Olfactory', 'DLS', 'GPe'])
d_select =  d_select.loc[np.isin(d_select['region'], selected_regions)]


fig,ax = plt.subplots(1,len(neuron_selections))
for i in np.arange(len(neuron_selections)):
    plt.sca(ax[i])
    sns.scatterplot(data=d_select.loc[d_select['n_neurons']==neuron_selections[i]], x = 'accu', y = 'r', hue='region',  palette=pals)
    r,p=pearsonr(d_select.loc[d_select['n_neurons']==neuron_selections[i],'r'].to_numpy(), 
            d_select.loc[d_select['n_neurons']==neuron_selections[i],'accu'].to_numpy())
    plt.xticks(rotation=90)
    sns.despine()
    if i!=0:
        plt.legend().remove()    
    plt.xlabel('Q-learning Accuracy')
    plt.ylabel('Decoder Accuracy')
    plt.title('n_neurons %s' %str(neuron_selections[i]) + ' ' + 'r=%s' %str(r)[:4] + ' ' + 'p=%s' %str(p)[:4])



d_select = d.loc[d['time_bin']>5]
d_select = d_select.groupby(['region','n_neurons']).mean()[['accu', 'r']].reset_index()
selected_regions = np.array(['OFC', 'NAcc', 'PFC', 'DMS', 'VPS', 'VP', 'SNr','Olfactory', 'DLS', 'GPe'])
d_select =  d_select.loc[np.isin(d_select['region'], selected_regions)]

fig,ax = plt.subplots(1,len(neuron_selections))
for i in np.arange(len(neuron_selections)):
    plt.sca(ax[i])
    sns.scatterplot(data=d_select.loc[d_select['n_neurons']==neuron_selections[i]], x = 'accu', y = 'r', hue='region', palette=pals)
    r,p=pearsonr(d_select.loc[d_select['n_neurons']==neuron_selections[i],'r'].to_numpy(), 
            d_select.loc[d_select['n_neurons']==neuron_selections[i],'accu'].to_numpy())
    plt.xticks(rotation=90)
    sns.despine()
    if i!=0:
        plt.legend().remove()
    plt.xlabel('Q-learning Accuracy')
    plt.ylabel('Decoder Accuracy')
    plt.title('n_neurons %s' %str(neuron_selections[i]) + ' ' + 'r=%s' %str(r)[:4] + ' ' + 'p=%s' %str(p)[:4])