import pandas as pd
import numpy as np

def fix_cluster_object(probe_path):
    prev = pd.read_csv(probe_path+'/clusters.metrics.csv')
    old_waveformsChannels = np.load(probe_path+'/clusters.waveformsChannels.npy')
    old_amps = np.load(probe_path+'/clusters.amps.npy')
    old_peakToTrough = np.load(probe_path+'/clusters.peakToTrough.npy')
    old_waveforms = np.load(probe_path+'/clusters.waveforms.npy')
    np.save(probe_path+'/clusters_old_waveformsChannels.npy', old_waveformsChannels)
    np.save(probe_path+'/clusters_old_amps.npy',old_amps)
    np.save(probe_path+'/clusters_old_peakToTrough.npy',old_peakToTrough)
    np.save(probe_path+'/clustersold_waveforms.npy',old_waveforms)
    depths = np.zeros(prev.cluster_id.max()+1)
    depths[:]=np.nan
    ch = np.zeros(prev.cluster_id.max()+1)
    ch[:]=-1
    depths[prev.cluster_id.to_numpy()]=prev['depth'].to_numpy()
    ch[prev.cluster_id.to_numpy()]=prev['ch'].to_numpy()
    waveformsChannels = np.zeros(prev.cluster_id.max()+1)
    waveformsChannels[:]=np.nan
    amps = np.zeros(prev.cluster_id.max()+1)
    amps[:]=np.nan
    amps[:]=np.nan
    peakToTrough = np.zeros(prev.cluster_id.max()+1)
    peakToTrough[:]=np.nan
    waveforms = np.zeros(prev.cluster_id.max()+1)
    waveforms[:]=np.nan
    np.save(probe_path+'/clusters.depths.npy', depths)
    np.save(probe_path+'/clusters.channels.npy',ch.astype(int))
    np.save(probe_path+'/clusters.waveformsChannels.npy', waveformsChannels)
    np.save(probe_path+'/clusters.amps.npy',amps)
    np.save(probe_path+'/clusters.peakToTrough.npy',peakToTrough)
    np.save(probe_path+'/clusters.waveforms.npy',waveforms)