import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ephys_alf_summary import LASER_ONLY
data = pd.read_pickle('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/analysis/da_connected_neurons.pkl')
pal = {
'Olfactory' : '#0c52ad',
'Amygdala': '#e67e55',
'DLS': '#1c4d23',
'DMS': '#08871b',
'DP': '#634beb',
'GP': '#fa20e8',
'Hypothal': '#963712',
'MO': '#1a3252',
'NAc': '#0ceb2c',
'OFC': '#649de8',
'Septum': '#cc4008',
'Thalamus': '#fff200',
'VP': '#f587ec',
'Tracts': '#6a696e'
}

dict_reg = pd.read_csv('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
dict_reg = dict(zip(dict_reg.original, dict_reg.group))

cell_loc_grouped  = pd.DataFrame()
for ses  in LASER_ONLY:
    print(ses)
    for r in Path(ses).glob('alf/probe0*'):
        rec = str(r)
        selection = np.load(rec + '/pykilosort/clusters_selection.npy')
        ch_loc = np.load(rec + '/pykilosort/channels.locations.npy', allow_pickle=True)
        c_ch = np.load(rec + '/pykilosort/clusters.channels.npy')
        regions_selection=[]
        for r in list(ch_loc[c_ch[selection].astype(int)]):
            if r in dict_reg.keys():
                regions_selection.append(dict_reg[r])
            else:
                regions_selection.append('NaN')
        cell_loc = pd.DataFrame(columns=['cluster', 'cell_type'])
        cell_loc['region'] = regions_selection
        cell_loc['rec'] = rec
        cell_loc_grouped = pd.concat([cell_loc_grouped, cell_loc])

cell_loc_grouped.loc[cell_loc_grouped['region']=='GPe', 'region'] = 'GP'
cell_loc_grouped.loc[cell_loc_grouped['region']=='GPi', 'region'] = 'GP'

totals = cell_loc_grouped.groupby(['region']).count().iloc[:,-1].reset_index()
totals = totals.rename(columns={'rec':'count'})
data.loc[data['grouped_region']=='AON', 'grouped_region'] = 'Olfactory'
data.loc[data['grouped_region']=='DP', 'grouped_region'] = 'Olfactory'
data_stats = data.groupby(['grouped_region']).count().reset_index().iloc[:,:2]
data_stats.loc[data_stats['grouped_region']=='axon', 'grouped_region'] = 'Tracts'
data_stats.loc[data_stats['grouped_region']=='Hyp', 'grouped_region'] = 'Hypothal'
data_stats.loc[data_stats['grouped_region']=='M2', 'grouped_region'] = 'MO'

tot = np.concatenate([totals.loc[totals['region']==reg,'count'].to_numpy() for reg in data_stats['grouped_region']])
data_stats['percent'] = data_stats.iloc[:,1]/tot
data_stats = data_stats.rename(columns={'cluster_id':'count'})

fig, ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.barplot(data=data_stats.sort_values('count'), x='grouped_region', y='count', palette=pal)
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Total connected neurons')
plt.sca(ax[1])
sns.barplot(data=data_stats.sort_values('percent'), x='grouped_region', y='percent', palette=pal)
plt.xticks(rotation=90)
plt.xlabel('Region')
plt.ylabel('Fraction of recorded neurons connected')
sns.despine()




cell_loc_grouped  = pd.DataFrame()
for ses  in LASER_ONLY:
    print(ses)
    for r in Path(ses).glob('alf/probe0*'):
        rec = str(r)
        selection = np.load(rec + '/pykilosort/clusters_selection.npy')
        ch_loc = np.load(rec + '/pykilosort/channels.locations.npy', allow_pickle=True)
        c_ch = np.load(rec + '/pykilosort/clusters.channels.npy')
        regions_selection=[]
        for r in list(ch_loc[c_ch[selection].astype(int)]):
            if r in dict_reg.keys():
                regions_selection.append(dict_reg[r])
            else:
                regions_selection.append('NaN')
        cell_loc = pd.DataFrame(columns=['cluster', 'cell_type'])
        cell_loc['region'] = regions_selection
        cell_loc['rec'] = rec
        cell_loc_grouped = pd.concat([cell_loc_grouped, cell_loc])


totals = cell_loc_grouped.groupby(['region']).count().iloc[:,-1].reset_index()
totals = totals.loc[totals['region']!='NaN']
sns.barplot(data=totals.sort_values('rec'), x='region', y='rec', color='k')
plt.ylabel('n_neurons (Single)')
plt.xticks(rotation=90)
plt.title('n_total = %s' % totals['rec'].sum())
sns.despine()
