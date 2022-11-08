LASER_SESSIONS = ['/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-27/003',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-20/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-19/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-16/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-15/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-28/001',
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-27/002',
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-20/001',
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-19/002',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-10/002',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-09/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-06/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-05/001']

LASER_SESSIONS = ['/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-04/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-03/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-02/001'
]

STAN_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_laser_only'

# Copy data to stan folder

import shutil

for ses in LASER_SESSIONS:
    mouse_path = ses[-22:-15]
    ses_path  = ses[-15:]
    src = ses + '/raw_behavior_data'
    dst = STAN_FOLDER + mouse_path + ses_path + '/raw_behavior_data'
    shutil.copytree(src, dst)

LASER_WATER_SESSIONS = [
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-22/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-23/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-26/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-28/001',
'/Volumes/witten/Alex/Data/Subjects/dop_52/2022-10-02/001'
]

STAN_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_waterlaser'

for ses in LASER_WATER_SESSIONS:
    mouse_path = ses[-22:-15]
    ses_path  = ses[-15:]
    src = ses + '/raw_behavior_data'
    dst = STAN_FOLDER + mouse_path + ses_path + '/raw_behavior_data'
    shutil.copytree(src, dst)
