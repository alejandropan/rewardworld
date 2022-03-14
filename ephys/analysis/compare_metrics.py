from pathlib import Path
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

LIST_OF_SESSIONS_CHR2_GOOD_REC = \
['/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-12/002',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001',
'/Volumes/witten/Alex/Data/Subjects/dop_13/2021-03-19/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/004',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-19/002',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-05-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-30/001',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-20/001',
'/Volumes/witten/Alex/Data/Subjects/dop_22/2021-06-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_22/2021-06-18/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-17/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-19/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-21/002']

all_metrics = pd.DataFrame()
for ses in LIST_OF_SESSIONS_CHR2_GOOD_REC:
    ses_path = Path(ses)
    alf_path = ses_path.joinpath('alf')
    for p_i in np.arange(4):
        if alf_path.joinpath('probe0%d' %p_i).exists():
            probe_path = alf_path.joinpath('probe0%d' %p_i)
            if probe_path.joinpath('pykilosort').exists():
                probe_path = probe_path.joinpath('pykilosort')
            try:
                mymetrics = pd.read_csv(probe_path.joinpath('clusters.metrics.csv'))
                if 'ks2_label' not in mymetrics.columns:
                    print('no ks2_label')
                    mymetrics['ks2_label'] = mymetrics['KSLabel']
                iblmetrics = pd.read_parquet(probe_path.joinpath('clusters_metrics.ibl.pqt'))
                try:
                    assert np.sum(mymetrics['cluster_id']-iblmetrics['cluster_id'])==0
                except:
                    assert np.sum(mymetrics['id']-iblmetrics['cluster_id'])==0
                merged_metrics = pd.concat([mymetrics,iblmetrics[['amp_median',
                        'slidingRP_viol', 'noise_cutoff', 'label']]] , axis=1)
                all_metrics = pd.concat([all_metrics, merged_metrics], axis=0)
            except:
                print('ERROR in ' +  probe_path)

n_ibl_good = len(np.where(all_metrics['label']==1)[0])
n_me_good = len(np.where(all_metrics['ks2_label']=='good')[0])
not_ibl_good = len(np.where(all_metrics['label']!=1)[0])
not_me_good = len(np.where(all_metrics['ks2_label']!='good')[0])
both_good = len(np.where((all_metrics['ks2_label']=='good') & (all_metrics['label']==1))[0])
rest = len(all_metrics) - both_good

# Percentage of good units
fig, ax = plt.subplots(1,3)
plt.sca(ax[0])
plt.pie([n_ibl_good, not_ibl_good], labels = ['IBL good %d' %n_ibl_good,
    'Rest %d' %not_ibl_good], autopct='%1.0f%%')
plt.sca(ax[1])
plt.pie([n_me_good, not_me_good], labels = ['Alex good %d' %n_me_good,
    'Rest %d' %not_me_good], autopct='%1.0f%%')
plt.sca(ax[2])
plt.pie([both_good, rest], labels = ['Both good %d' %both_good,
    'Rest %d' %rest], autopct='%1.0f%%')

# Where do they fail
selection = all_metrics.loc[all_metrics['ks2_label']=='good']

fig, ax = plt.subplots(1,3)
plt.sca(ax[0])
plt.title('slidingRP_viol')
slidingRP_viol_0 = len(np.where(selection['slidingRP_viol']!=1)[0])
slidingRP_viol_1 = len(np.where(selection['slidingRP_viol']==1)[0])
plt.pie([slidingRP_viol_1, slidingRP_viol_0], labels = ['RPV<0.1 %d' %slidingRP_viol_1,
    'RPV>0.1 %d' %slidingRP_viol_0], autopct='%1.0f%%')
plt.sca(ax[1])
plt.title('noise_cutoff')
noise_viol_0 = len(np.where(selection['noise_cutoff']>20)[0])
noise_viol_1 = len(np.where(selection['noise_cutoff']<20)[0])
plt.pie([noise_viol_1, noise_viol_0], labels = ['Noise<0.2 %d' %noise_viol_1,
    'Noise>0.2 %d' %noise_viol_0], autopct='%1.0f%%')
plt.sca(ax[2])
amp_viol_0 = len(np.where(selection['amp_median']*1e6<50)[0])
amp_viol_1 = len(np.where(selection['amp_median']*1e6>50)[0])
plt.pie([noise_viol_1, noise_viol_0], labels = ['Amplitude>50 %d' %amp_viol_1,
    'Amplitude<50 %d' %amp_viol_0], autopct='%1.0f%%')
plt.title('Amplitude')

fig, ax = plt.subplots(1,2)
plt.sca(ax[0])
plt.title('noise_cutoff')
sns.histplot(selection['noise_cutoff'], stat='percent')
plt.vlines(20, 0, 40, linestyles='dashed', color = 'r')
plt.xlim(0,100)
plt.sca(ax[1])
plt.title('Amplitude')
sns.histplot(selection['amp_median']*1e6, stat='percent')
plt.vlines(50, 0, 10, linestyles='dashed', color='r')
