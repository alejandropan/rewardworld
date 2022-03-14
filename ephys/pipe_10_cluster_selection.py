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
    if probe_path.name == 'pykilosort':
        ses_path = probe_path.parent.parent.parent
        probe_name = probe_path.parent.name
    else:
        ses_path = probe_path.parent.parent
        probe_name = probe_path.name
    ephys_file_dir = ses_path.joinpath('raw_ephys_data', probe_name)
    raw_files = glob.glob(os.path.join(ephys_file_dir, '*ap.*bin'))
    raw_file = [raw_files[0]] if raw_files else None
    cluster_metrics_path = probe_path.joinpath('clusters_metrics.ibl.pqt')
    if not cluster_metrics_path.exists():
        print('computing metrics this may take a bit of time')
        spikes = one.alf.io.load_object(probe_path, 'spikes',
                                 attribute=['depths', 'times', 'amps', 'clusters'])
        clusters = one.alf.io.load_object(probe_path, 'clusters', attribute=['channels',''])
        r = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths)
        r = pd.DataFrame(r)
        r.to_parquet(cluster_metrics_path)

def make_selection(probe, min_amp=35):
    """
    probe is a string variable with the path to the probe/clusters data
    """
    iblmetrics = pd.read_parquet(probe+'/clusters_metrics.ibl.pqt')
    labels = pd.read_csv(probe+'/clusters.metrics.csv')
    try:
        labels_select = labels.loc[labels['group']=='good','id'].to_numpy()
    except:
        labels_select = labels.loc[labels['group']=='good','cluster_id'].to_numpy()
    RP_select = iblmetrics.loc[iblmetrics['slidingRP_viol']==1,'cluster_id'].to_numpy()
    amp_select = iblmetrics.loc[iblmetrics['amp_median']>(min_amp/1e6),'cluster_id'].to_numpy()
    selection = np.intersect1d(labels_select,RP_select)
    selection = np.intersect1d(selection, amp_select)
    np.save(probe+'/clusters_selection.npy', selection)


if __name__ == "__main__":
    probe = sys.argv[1]
    run_metrics(probe) 
    make_selection(probe)