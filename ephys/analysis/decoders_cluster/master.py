
import os
os.chdir('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
from ephys_alf_summary import alf
import numpy as np
import subprocess
import pandas as pd

# Parameters
work_directory = '/jukebox/witten/Alex/decoder_output'

# Get areas in recording
SESSIONS = ['/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
#'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-16/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',  
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-19/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-27/003', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-20/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-10/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001']

AREAS = []

for ses in SESSIONS:
    alfio = alf(ses, ephys=True)
    areas = []
    for p in np.arange(4): # Max 4 probes
        try:
            areas.append(alfio.probe[p].cluster_group_locations.unique()[~pd.isna(alfio.probe[p].cluster_group_locations.unique())])
        except:
            continue
    areas  = np.unique(np.concatenate(areas))
    AREAS.append(areas)

np.save('/jukebox/witten/Alex/decoder_output/areas_summary.npy', AREAS)

for ses_n,ses in enumerate(SESSIONS):
    for area_n, area in enumerate(AREAS[ses_n]):
        print(area_n*1000 + ses_n)


        print(subprocess.run(['/Volumes/witten-1/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/decoder_job_outcome_chosen_value_all_trials.cmd'+ ' ' + str(ses_n) + ' ' + str(area_n)], shell=True))
