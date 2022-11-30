
import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
from ephys_alf_summary import alf
import numpy as np
import subprocess
import pandas as pd
import os

# Get areas in recording
ROOT='/jukebox/witten/Alex/Data/Subjects/'
SESSIONS = [
 'dop_47/2022-06-05/001',
 'dop_47/2022-06-06/001',
 'dop_47/2022-06-07/001',
 #'dop_47/2022-06-09/003',
 'dop_47/2022-06-10/002',
 'dop_47/2022-06-11/001',
 'dop_48/2022-06-20/001',
 'dop_48/2022-06-27/002',
 'dop_48/2022-06-28/001',
 'dop_49/2022-06-14/001',
 'dop_49/2022-06-15/001',
 'dop_49/2022-06-16/001',
 'dop_49/2022-06-17/001',
 'dop_49/2022-06-18/002',
 'dop_49/2022-06-19/001',
 'dop_49/2022-06-20/001',
 'dop_49/2022-06-27/003',
 'dop_50/2022-09-12/001',
 'dop_50/2022-09-13/001',
 'dop_50/2022-09-14/003',
 'dop_53/2022-10-02/001',
 'dop_53/2022-10-03/001',
 'dop_53/2022-10-04/001',
 'dop_53/2022-10-05/001',
 'dop_53/2022-10-07/001']

AREAS = []

for ses in SESSIONS:
    alfio = alf(ROOT+ses, ephys=True)
    areas = []
    for p in np.arange(4): # Max 4 probes
        try:
            areas.append(alfio.probe[p].cluster_group_locations.unique()[~pd.isna(alfio.probe[p].cluster_group_locations.unique())])
        except:
            continue
    areas  = np.unique(np.concatenate(areas))
    AREAS.append(areas)
np.save('/jukebox/witten/Alex/decoders_raw_results/decoder_output_rpe_outcome_forget/areas_summary.npy', AREAS)
id_dict = pd.DataFrame()
counter = 0
for ses_n,ses in enumerate(SESSIONS):
    for area_n, area in enumerate(AREAS[ses_n]):
        ids = pd.DataFrame()
        ids['ses'] = [ses]
        ids['area'] = [area]
        ids['id'] = [counter]
        id_dict=pd.concat([id_dict,ids])
        counter+=1
id_dict.to_csv('/jukebox/witten/Alex/decoders_raw_results/decoder_output_rpe_outcome_forget/id_dict.csv') # Translate slurm ids to regions
# counter-1 since last counter is not used in dict
print(subprocess.run(['sbatch --array=0-'+str(counter-1)+' /jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/raw_data/standard_w_forgetting/decoder_RPE_outcome/decoder_job_rpe_outcome_forget.cmd'], shell=True))