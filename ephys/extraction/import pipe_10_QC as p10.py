import pipe_10_QC as p10
import pipe_9_cluster_selection as p9
import pipe_8_full_bandit_fix as p8
import pipe_3_relabel_pykilosort as p3
import pipe_4_patch_cluster_object_pykilosort as p4
import pipe_5_getAlyxpenetration as p5
import pipe_7_json_to_alf as p7
import pipe_11_make_splits_alignable as p11
import pipe_12_extract_licks as p12
import shutil
from pathlib import Path
import numpy as np

SESSIONS = ['/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-16/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',  
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-19/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-27/003', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-10/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-05/001']


SESSIONS = ['/mnt/s0/Data/Subjects/dop_48/2022-06-20/001',
'/mnt/s0/Data/Subjects/dop_48/2022-06-19/002',
'/mnt/s0/Data/Subjects/dop_48/2022-06-28/001',
'/mnt/s0/Data/Subjects/dop_48/2022-06-27/002',
'/mnt/s0/Data/Subjects/dop_49/2022-06-14/001',
'/mnt/s0/Data/Subjects/dop_49/2022-06-15/001',
'/mnt/s0/Data/Subjects/dop_49/2022-06-16/001',
'/mnt/s0/Data/Subjects/dop_49/2022-06-17/001',
'/mnt/s0/Data/Subjects/dop_49/2022-06-18/002',
'/mnt/s0/Data/Subjects/dop_49/2022-06-19/001',
'/mnt/s0/Data/Subjects/dop_49/2022-06-27/003',
'/mnt/s0/Data/Subjects/dop_49/2022-06-20/001',
'/mnt/s0/Data/Subjects/dop_47/2022-06-11/001',
'/mnt/s0/Data/Subjects/dop_47/2022-06-10/002',
'/mnt/s0/Data/Subjects/dop_47/2022-06-09/003',
'/mnt/s0/Data/Subjects/dop_47/2022-06-05/001']



good_units = []
errors  = []
for ses in SESSIONS:
    p8.full_bandit_fix(ses)
    p10.run_QC(ses)
    for i in np.arange(4):
        probe = ses + '/alf/probe0' + str(i) + '/pykilosort'
        if Path(probe).exists():
            try:
                p3.run_relabeling(probe)
                p4.ammend_cluster_files(probe)
                p9.run_cluster_selection(probe)
                good_units.append(len(np.load(probe+'/clusters_selection.npy')))
            except:
                errors.append(probe)





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
    for i in np.arange(4):
        probe = ses + '/alf/probe0' + str(i) + '/pykilosort'
        if Path(probe).exists():
            try:
                p7.run_get_location(probe)
            except:    
                errors.append(probe)



errors=[]
for ses in SESSIONS:
    for i in np.arange(4):
        probe = ses + '/alf/probe0' + str(i) + '/pykilosort'
        if Path(probe).exists():
            try:
                p11.fix_cluster_object(probe)
            except:    
                errors.append(probe)



errors=[]
for ses in SESSIONS:
    for i in np.arange(4):
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
