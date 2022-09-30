import pandas as pd
import sys
import numpy as np

def complete_labels(path):
    original_labels = pd.read_csv(path+'/cluster_group.tsv', sep='\t')
    cluster_no = len(np.load(path+'/clusters.channels.npy'))
    new_labels = pd.DataFrame()
    new_labels['cluster_id'] = np.arange(cluster_no)
    new_labels['Group'] = 'nan'
    new_labels.loc[np.isin(new_labels['cluster_id'], original_labels['cluster_id']),
                    'Group'] = 'good'
    new_labels.set_index('cluster_id').to_csv(path+'/cluster_cgroup.tsv', sep='\t')

if __name__=='__main__':
    path = sys.argv[1]
    complete_labels(path)
