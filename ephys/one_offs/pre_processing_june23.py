from ephys_alf_summary import LASER_ONLY

for ses in ['/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-10/002',
    '/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
    '/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-05/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-06/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001']:
    update_params_file(ses)
    for p_path in Path(ses).glob('alf/probe0*'):
        try:
            extract_waveforms(p_path.as_posix())
        except:
            print('error extracting waveforms from ' + p_path.as_posix())

import pipe_3_relabel_pykilosort as p3
import pipe_4_patch_cluster_object_pykilosort as p4
import pipe_9_cluster_selection as p9

for ses in LASER_ONLY:
    for p_path in Path(ses).glob('alf/probe0*'):
        p_path_str = p_path.as_posix() + '/pykilosort'
        #run_relabeling(p_path_str)
        ammend_cluster_files(p_path_str)
        #run_cluster_selection(p_path_str) 


counter = 0 
counter_mua = 0
for ses in LASER_ONLY:
    for p_path in Path(ses).glob('alf/probe0*'):
        p_path_str = p_path.as_posix() + '/pykilosort'
        counter+=len(np.load(p_path_str+'/clusters_selection.npy'))
        counter_mua+=len(np.load(p_path_str+'/clusters_goodmua_selection.npy'))
        #run_relabeling(p_path_str)
        #run_cluster_selection(p_path_str) 


for ses in ['/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-10/002',
    '/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
    '/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-05/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-06/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
    '/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001']:
    for p_path in Path(ses).glob('alf/probe0*'):
        p_path_str = p_path.as_posix() + '/pykilosort'
        p3.run_relabeling(p_path_str)
        p4.ammend_cluster_files(p_path_str)
        p9.run_cluster_selection(p_path_str) 