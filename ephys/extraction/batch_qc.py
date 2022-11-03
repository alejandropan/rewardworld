from pipe_8_full_bandit_fix import full_bandit_fix
from check_sync import check_sync


probes = [
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001/alf/probe02/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001/alf/probe00/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001/alf/probe01/pykilosort',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001/alf/probe02/pykilosort',
]

ses = \
['/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001']

for s in ses:
    check_sync(s)
    full_bandit_fix(s)