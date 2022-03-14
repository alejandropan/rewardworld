import glob
import os
import pandas as pd
from brainbox.metrics.single_units import quick_unit_metrics
from pathlib import Path
import one.alf.io
import numpy as np

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

penetrations = [
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-12/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-30/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-05-05/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-14/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-14/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-20/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-20/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-19/002/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-17/002/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-19/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-21/002/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-21/002/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-23/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-23/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-02/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-02/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-02/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-04/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-04/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-04/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-07/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-07/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-07/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-07/002/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-09/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-09/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-09/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-11/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-11/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-11/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-13/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-13/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-13/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-15/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-15/001/alf/probe01/pykilosort']

for probe in penetrations:
    #run_metrics(probe) 
    make_selection(probe)

