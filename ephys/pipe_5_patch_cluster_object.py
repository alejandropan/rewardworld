import pandas as pd
import numpy as np
from pathlib import Path
import sys

"""
Patch clusters object so that splits are allowed. To be used post-splitting
"""

def ammend_cluster_files(probe_path):
    """
    Probe path it should be a string
    e.g.
    probe_path = '/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/004/alf/probe00'
    """
    prev = pd.read_csv(probe_path+'/clusters.metrics.csv')
    # Fix for potential movement of columns
    index_cols = np.array(['id', 'Amplitude', 'ContamPct', 'KSLabel', 'amp', 'ch',
              'depth', 'fr', 'group', 'n_spikes', 'sh'])
    metrics = pd.DataFrame()
    for var in index_cols:
        try:
            metrics[var]=prev[var]
        except:
            metrics[var]=np.nan
    assert prev.shape[0] == metrics.shape[0]
    # Fix for splits
    metrics = metrics.rename(columns={"KSLabel": "ks2_label"})
    metrics.loc[metrics['ks2_label'].isnull(),'ks2_label']='noise'
    len_npy = len(np.load(probe_path+'/clusters.channels.npy'))
    len_csv = len(metrics)
    len_diff = len_csv - len_npy
    if len_diff != 0:
        #First eliminate 0 amp units from cluster files
        print('Splits have been made or 0 amp units')
        #check if sorter skipped any units and remove from the list (0 amplitude units are removed from Phy)
        original_units_del = np.where(np.isnan(np.load(probe_path+'/clusters.amps.npy')))[0]
        print(str(len(original_units_del)) + ' 0 amp units')
        assert len(metrics.loc[np.isin(metrics['id'], original_units_del)])==0
        # Load cluster objects
        amps = np.load(probe_path+'/clusters.amps.npy')
        channels = np.load(probe_path+'/clusters.channels.npy')
        depths = np.load(probe_path+'/clusters.depths.npy')
        peakToTrough = np.load(probe_path+'/clusters.peakToTrough.npy')
        waveforms = np.load(probe_path+'/clusters.waveforms.npy')
        waveformsChannels = np.load(probe_path+'/clusters.waveformsChannels.npy')
        # Ammend files, create arrays
        amps_new = np.delete(amps, original_units_del)
        channels_new = np.delete(channels, original_units_del)
        depths_new = np.delete(depths, original_units_del)
        peakToTrough_new = np.delete(peakToTrough, original_units_del)
        waveforms_new = np.delete(waveforms, original_units_del, axis=0)
        waveformsChannels_new = np.delete(waveformsChannels, original_units_del, axis=0)
        metrics['old_id'] = metrics['id'].copy()
        assert len(metrics)>=len(amps_new)
        if len(metrics)>len(amps_new):
            print('Splitted or Merged units present')
            # Now go for splits, first check if last unit was deleted for idx purposes
            i = 1
            while not len(metrics.loc[metrics['id']==len(channels)-i])>0:
                    i += 1
            last_unit_idx = metrics.loc[metrics['id']==len(channels)-i].index[0]
            # last_unit_idx+1 - for indexing so that selection includes last unit
            m_units = metrics['id'][:last_unit_idx+1]
            m_units_w_zero_amp = np.sort(np.concatenate([m_units, original_units_del]))
            splitted_units = m_units_w_zero_amp[np.where(np.diff(m_units_w_zero_amp)>1)[0]]+1
            # splitted_units = metrics['id'][:last_unit_idx+1][metrics[:last_unit_idx+1].id.diff()>1].to_numpy()-1
            # Check if first unit  was changed
            if metrics['id'].min() != 0:
                splitted_units = np.concatenate([np.array([0]),splitted_units])
            #remove from splitted units, units that are already dealed with due to amp 0
            common = np.intersect1d(splitted_units,original_units_del)
            splitted_units = np.setdiff1d(np.union1d(splitted_units, common), np.intersect1d(splitted_units, common))
            print(str(len(splitted_units)) + ' splitted units')
            # Ammend files, create arrays
            to_remove =  np.concatenate([splitted_units,original_units_del])
            to_remove = np.unique(to_remove)
            amps_new = np.delete(amps, to_remove)
            channels_new = np.delete(channels, to_remove)
            depths_new = np.delete(depths,to_remove)
            peakToTrough_new = np.delete(peakToTrough, to_remove)
            waveforms_new = np.delete(waveforms, to_remove, axis=0)
            waveformsChannels_new = np.delete(waveformsChannels, to_remove, axis=0)
            # Add data for new clusters where possible if not add nan
            channels_new = np.append(channels_new,metrics[last_unit_idx+1:]['ch'].to_numpy())
            depths_new = np.append(depths_new,metrics[last_unit_idx+1:]['depth'].to_numpy())
            nan_peak = np.empty(len(metrics[last_unit_idx+1:]))
            nan_peak[:] = np.nan
            nan_waveform = np.empty((len(metrics[last_unit_idx+1:]),waveforms.shape[1], waveforms.shape[2]))
            nan_waveform[:] = np.nan
            nan_waveform_channel = np.empty((len(metrics[last_unit_idx+1:]),waveformsChannels_new.shape[1]))
            nan_waveform_channel[:] = np.nan
            nan_amp = np.empty(len(metrics[last_unit_idx+1:]))
            nan_amp[:] = np.nan
            peakToTrough_new = np.append(peakToTrough_new,nan_peak)
            waveforms_new = np.append(waveforms_new,nan_waveform, axis=0)
            waveformsChannels_new = np.append(waveformsChannels_new,nan_waveform_channel, axis=0)
            amps_new = np.append(amps_new,nan_amp)
            # Do a few assertions to check the consistency of the data
            assert np.array_equal(metrics['depth'],depths_new)
            assert np.array_equal(metrics['ch'],channels_new)
        # Rename spikes
        spikes = np.load(probe_path + '/spikes.clusters.npy')
        np.save(probe_path+'/spikes.clusters_old_id.npy', spikes)
        for n in np.unique(spikes):
            spikes[np.where(spikes==n)] = metrics.loc[metrics['old_id']==n].index[0]
        np.save(probe_path+'/spikes.clusters.npy', spikes)
        # Overwrite arrays
        np.save(probe_path+'/clusters.amps.npy', amps_new)
        np.save(probe_path+'/clusters.channels.npy', channels_new)
        np.save(probe_path+'/clusters.depths.npy', depths_new)
        np.save(probe_path+'/clusters.peakToTrough.npy', peakToTrough_new)
        np.save(probe_path+'/clusters.waveforms.npy', waveforms_new)
        np.save(probe_path+'/clusters.waveformsChannels.npy', waveformsChannels_new)
        # Update metrics file for homogeonization
        metrics['id'] = np.arange(len(metrics))
        metrics.to_csv(probe_path+'/clusters.metrics.csv')
if __name__=="__main__":
    ammend_cluster_files(*sys.argv[1:])
