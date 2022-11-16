import pipe_9_cluster_selection as p9
import pipe_8_full_bandit_fix as p8
import pipe_3_relabel_pykilosort as p3
import pipe_7_json_to_alf as p7
import pipe_12_extract_licks as p12
from check_sync import check_sync
import shutil
from pathlib import Path
import numpy as np
from tqdm import tqdm

SESSIONS = \
['/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-22/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-23/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-26/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-28/001',
'/Volumes/witten/Alex/Data/Subjects/dop_52/2022-10-02/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-04/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-03/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-02/001']

for ses in tqdm(SESSIONS):
    p8.full_bandit_fix(ses)
    check_sync(ses)
    p12.extract_licks(ses)
    for i in np.arange(3):
        probe = ses + '/alf/probe0' + str(i) + '/pykilosort'
        p3.run_relabeling(probe)
        p7.run_get_location(probe)
        p9.run_cluster_selection(probe)









errors=[]
for ses in SESSIONS:
    for i in np.arange(4):
        probe = ses + '/alf/probe0' + str(i) + '/pykilosort'
        if Path(probe).exists():
            try:
                p5.get_xyz(probe)
            except:    
                errors.append(probe)



errors=[]
for ses in SESSIONS:
    for i in np.arange(3):
        probe = ses + '/alf/probe0' + str(i) + '/pykilosort'
        if Path(probe).exists():
            try:
                file = '/STD_ds_dop_%s_RD.nrrd' % probe[-40:-38]
                src = '/Users/alexpan/Downloads'+file
                shutil.copyfile(src, probe+file)
                file = '/STD_ds_dop_%s_GR.nrrd' % probe[-40:-38]
                src = '/Users/alexpan/Downloads'+file
                shutil.copyfile(src, probe+file)
            except:    
                errors.append(probe)

unconsumed=[]
errors=[]
for ses in SESSIONS:
    try:
        unconsumed.append(p12.extract_licks(ses))
    except:    
        errors.append(ses)
