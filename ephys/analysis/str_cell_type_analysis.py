import numpy as np 
import pandas as pd
from brainbox.singlecell import acorr
from ephys_alf_summary import LASER_ONLY
from pathlib import Path

def classify_cell_type_str(sp_t, c_wav, cluster):
    '''
    sp_t: /spikes.times.npy'
    c_wav: /clusters.waveforms.npy'
    cluster: cluster index
    '''    
    # Calculate metrics
    cluster_p_2_t = abs(c_wav[cluster, :, 0].argmin() - c_wav[cluster, :, 0].argmax())*1e6/sampling_rate
    cluster_isi = np.diff(sp_t[np.where(sp_c==cluster)])
    fraction_long_isis = np.sum(cluster_isi[cluster_isi>2])/(sp_t[np.where(sp_c==cluster)][-1] - sp_t[np.where(sp_c==cluster)][0])
    spike_times =  sp_t[np.where(sp_c==cluster)]
    xc = acorr(spike_times,bin_size=0.001, window_size=2)
    xc = xc[int(xc.shape[-1]/2):]
    xc_rate = xc/(len(spike_times)*0.001) # Moshe Abeles 1982
    avg_fr = np.mean(xc_rate[600:900]) # Peters 2022
    spike_supression = np.where(xc_rate>=avg_fr)[0][0]# Peters 2022

    if cluster_p_2_t<350:
        if fraction_long_isis<0.1:
            return 'FSI', cluster_p_2_t, fraction_long_isis, spike_supression
        else:
            return 'LTSI', cluster_p_2_t, fraction_long_isis, spike_supression
    else:
        if spike_supression>40:
            return 'TAN', cluster_p_2_t, fraction_long_isis, spike_supression
        else:
            return 'MSI', cluster_p_2_t, fraction_long_isis, spike_supression


STR_REGIONS = ['DLS', 'DMS', 'NAc','TS']
dict_reg = pd.read_csv('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
dict_reg = dict(zip(dict_reg.original, dict_reg.group))

cell_types_grouped  = pd.DataFrame()
for ses  in LASER_ONLY:
    print(ses)
    for r in Path(ses).glob('alf/probe0*'):
        rec = str(r)
        selection = np.load(rec + '/pykilosort/clusters_selection.npy')
        ch_loc = np.load(rec + '/pykilosort/channels.locations.npy', allow_pickle=True)
        sp_t = np.load(rec + '/pykilosort/spikes.times.npy')
        sp_c = np.load(rec + '/pykilosort/spikes.clusters.npy')
        c_ch = np.load(rec + '/pykilosort/clusters.channels.npy')
        c_wav = np.load(rec + '/pykilosort/clusters.waveforms.npy') #[n_clusters x n_timepoints x n_channel_subset] [first ch waverform is the ch with center of mass]
        #t_wav = np.load(REC + '/templates.waveforms.npy') #[n_clusters x n_timepoints x n_channel_subset] [first ch waverform is the ch with center of mass]
        sampling_rate = 30000
        #with open(rec + '/pykilosort/params.py') as f:
        #    lines = f.readlines()
        #sampling_rate = int(lines[-2][14:-1])

        # Select striatum clusters
        regions_selection=[]
        for r in list(ch_loc[c_ch[selection].astype(int)]):
            if r in dict_reg.keys():
                regions_selection.append(dict_reg[r])
            else:
                regions_selection.append('NaN')
        str_selected_clusters = selection[np.where(np.isin(regions_selection, STR_REGIONS))]
        str_selected_clusters_regions = np.array(regions_selection)[np.where(np.isin(regions_selection, STR_REGIONS))]

        cell_types = pd.DataFrame(columns=['cluster', 'cell_type'])
        cell_types['cluster'] = str_selected_clusters
        ctypes = []
        p2ts = []
        long_isis = []
        spike_supressions = []
        for cluster in str_selected_clusters:
            try:
                ct, cluster_p_2_t, fraction_long_isis, spike_supression = classify_cell_type_str(sp_t, c_wav, cluster)
                ctypes.append(ct)
                p2ts.append(cluster_p_2_t)
                long_isis.append(fraction_long_isis)
                spike_supressions.append(spike_supression)
            except:
                ctypes.append('NaN')
                p2ts.append(np.nan)
                long_isis.append(np.nan)
                spike_supressions.append(np.nan)

        cell_types['cell_type'] = ctypes
        cell_types['p2t'] = p2ts
        cell_types['fraction_long_isis'] = long_isis
        cell_types['spike_supression'] = spike_supressions
        cell_types['region'] = str_selected_clusters_regions
        cell_types['rec'] = rec
        cell_types_grouped = pd.concat([cell_types_grouped, cell_types])


# Creating figure
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


cell_types_grouped_store = cell_types_grouped.copy()


ROI = 'TS'
cell_types_grouped = cell_types_grouped_store.loc[cell_types_grouped_store['spike_supression']<160]
cell_types_grouped = cell_types_grouped.loc[cell_types_grouped['region']==ROI]
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
# Creating plot
FSI = cell_types_grouped[cell_types_grouped.cell_type=='FSI']
LTSI = cell_types_grouped[cell_types_grouped.cell_type=='LTSI']
TAN = cell_types_grouped[cell_types_grouped.cell_type=='TAN']
MSI = cell_types_grouped[cell_types_grouped.cell_type=='MSI']

ax.scatter3D(FSI['spike_supression'], FSI['p2t'], FSI['fraction_long_isis'], color = "steelblue")
ax.scatter3D(LTSI['spike_supression'], LTSI['p2t'], LTSI['fraction_long_isis'], color = "darkorange")
ax.scatter3D(MSI['spike_supression'], MSI['p2t'], MSI['fraction_long_isis'], color = "magenta")
ax.scatter3D(TAN['spike_supression'], TAN['p2t'], TAN['fraction_long_isis'], color = "sienna")

ax.set_xlabel('Post-spike supression (ms)', fontweight ='bold')
ax.set_ylabel('Spike width (us)', fontweight ='bold')
ax.set_zlabel('Fraction of >2s ISI', fontweight ='bold')  

ax.set_ylim([0, 900])
plt.title("Cell types")

red_patch = mpatches.Patch(color='steelblue', label='FSI')
gold_patch = mpatches.Patch(color='darkorange', label='UIN')
blue_patch = mpatches.Patch(color='magenta', label='MSN')
magenta_patch = mpatches.Patch(color='sienna', label='TAN')
plt.legend(handles=[red_patch,gold_patch,blue_patch,magenta_patch])

plt.show()