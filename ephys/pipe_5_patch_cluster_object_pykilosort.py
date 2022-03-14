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

    Index(['Unnamed: 0', 'id', 'Amplitude', 'ContamPct', 'old_group', 'amp',
       'old_ch', 'depth', 'fr', 'group', 'n_spikes', 'sh', 'KSLabel', 'ch'],
      dtype='object')
    """
    prev = pd.read_csv(probe_path+'/clusters.metrics.csv')
    clusters = np.load(probe_path+'/clusters.channels.npy')
    np.save(probe_path+'/clusters.channels_old.npy', clusters)
    np.save(probe_path+'/clusters.depths.npy', prev['depth'].to_numpy())
    np.save(probe_path+'/clusters.channels.npy', prev['ch'].to_numpy())

if __name__=="__main__":
    ammend_cluster_files(*sys.argv[1:])
