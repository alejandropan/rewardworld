import glob
import os
import pandas as pd
from brainbox.metrics.single_units import quick_unit_metrics
from pathlib import Path
import one.alf.io
import numpy as np
import sys

def run_metrics(probe_path):
    """
    Launch phy given an eid and probe name.
    TODO calculate metrics and save as .tsvs to include in GUI when launching?
    """
    # This is a first draft, no error handling and a draft dataset list.
    probe_path = Path(probe_path)
    cluster_metrics_path = probe_path.joinpath('clusters_metrics.ibl.pqt')
    print('computing metrics this may take a bit of time')
    spikes = one.alf.io.load_object(probe_path, 'spikes',
                                attribute=['depths', 'times', 'amps', 'clusters'])
    clusters = one.alf.io.load_object(probe_path, 'clusters', attribute=['channels',''])
    r = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths)
    r = pd.DataFrame(r)
    r.to_parquet(cluster_metrics_path)

def make_selection(probe, min_amp=30):
    """
    probe is a string variable with the path to the probe/clusters data
    """
    iblmetrics = pd.read_parquet(probe+'/clusters_metrics.ibl.pqt')
    labels = pd.read_csv(probe+'/clusters.metrics.csv')
    try:
        labels_select = labels.loc[np.isin(labels['group'],['good','mua']),'id'].to_numpy()
    except:
        labels_select = labels.loc[np.isin(labels['group'],['good','mua']),'cluster_id'].to_numpy()
    fr_select = iblmetrics.loc[iblmetrics['firing_rate']>=0.01, 'cluster_id'].to_numpy()
    RP_select = iblmetrics.loc[iblmetrics['contamination']<=0.1, 'cluster_id'].to_numpy()
    #RP_select = iblmetrics.loc[iblmetrics['slidingRP_viol']==1,'cluster_id'].to_numpy()
    amp_select = iblmetrics.loc[iblmetrics['amp_median']>(min_amp/1e6),'cluster_id'].to_numpy()
    selection = np.intersect1d(labels_select,RP_select)
    selection = np.intersect1d(selection, amp_select)
    selection = np.intersect1d(selection, fr_select)
    P_select = iblmetrics.loc[iblmetrics['presence_ratio']>=0.75, 'cluster_id'].to_numpy()
    mua_selection = np.intersect1d(amp_select, P_select) # for decoders only use single or mua present for at least 75% of the recording
    mua_selection = np.intersect1d(mua_selection, fr_select)
    np.save(probe+'/clusters_selection.npy', selection)
    np.save(probe+'/clusters_goodmua_selection.npy', mua_selection)

def run_cluster_selection(probe):
    run_metrics(probe) 
    make_selection(probe)


if __name__ == "__main__":
    probe = sys.argv[1]
    run_cluster_selection(probe) 


