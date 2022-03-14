from pathlib import Path
from calculate_metrics import run_metrics

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


for ses in LIST_OF_SESSIONS_CHR2_GOOD_REC:
    ses_path = Path(ses)
    alf_path = ses_path.joinpath('alf')
    for p_i in np.arange(4):
        if alf_path.joinpath('probe0%d' %p_i).exists():
            probe_path = alf_path.joinpath('probe0%d' %p_i)
            if probe_path.joinpath('pykilosort').exists():
                probe_path = probe_path.joinpath('pykilosort')
            try:
                probe_path = probe_path.as_posix()
                run_metrics(probe_path)
            except:
                print('ERROR in ' +  probe_path)
