import numpy as np
from skimage import io
import pandas as pd
#import original atlas
atlas = io.imread('/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/allen_subdivisions.tif')
atlas_values = pd.read_csv('/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/41467_2019_13057_MOESM4_ESM.csv')
dict_for_our_labels = pd.read_csv('/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
######################################################################################################################################################
#search pixels for caudoputamen regions we are interested
atlas_new = np.zeros(np.shape(atlas))
atlas_new[np.where(atlas!=0)] =1000
our_subdivision = ['DMS', 'DLS', 'TS', 'NAc', 'VPS']
values = ['100', '200', '300', '400', '500']
for i, sub in enumerate(our_subdivision):
    lumped_striatal_regions = dict_for_our_labels.loc[dict_for_our_labels['group']==sub,'original']
    values_to_change = atlas_values.loc[np.isin(atlas_values['Franklin-Paxinos Full name'], lumped_striatal_regions), 'Structural ID'].to_numpy()
    atlas_new[np.where(np.isin(atlas,values_to_change))] = values[i]
io.imsave('/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/allen_subdivisions_limited.tif', atlas_new)
######################################################################################################################################################
atlas_values = pd.read_csv('/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/41467_2019_13057_MOESM4_ESM.csv')
atlas_new[np.where(atlas!=0)] =100
cp_ids = atlas_values.loc[atlas_values['Franklin-Paxinos Full name'].str.contains('Caudoputamen')==True,  'Structural ID'].tolist()
[cp_ids.append(i) for i in atlas_values.loc[atlas_values['Franklin-Paxinos Full name'].str.contains('Accumbens')==True,  'Structural ID'].tolist()]
cp_ids_colors = (np.arange(len(cp_ids))+2) * 100
for i, cp in enumerate(cp_ids):
     atlas_new[np.where(atlas==cp)] = cp_ids_colors[i]
io.imsave('/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/allen_subdivisions_only_str.tif', atlas_new)
