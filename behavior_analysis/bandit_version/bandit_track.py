import numpy as np
from matplotlib import pyplot as plt
import psytrack as psy
import seaborn
import pandas as pd
from pathlib import Path

# Hard coded paths

ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects/ephys_bandit/'
TRIALS_BACK =  5
REEXTRACT = False

# First load data in ROOT_FOLDER
root_path = Path(ROOT_FOLDER)
data = pd.DataFrame()
for animal in root_path.iterdir():
            if animal.is_dir():
                for day in animal.iterdir():
                    if day.is_dir():
                        for ses in day.iterdir():
                            if ses.is_dir():
                                if REEXTRACT==True:
                                    full_bandit_fix(ses.as_posix())
                                mouse_psy = pd.DataFrame()
                                mouse_psy['feedback'] = \
                                    1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)
                                mouse_psy['choices'] = \
                                    -1*np.load(ses.joinpath('alf', '_ibl_trials.choice.npy'))
                                mouse_psy['laser_block'] = \
                                    np.load(ses.joinpath('alf', '_ibl_trials.opto_block.npy'))
                                mouse_psy['laser'] = mouse_psy['feedback']*mouse_psy['laser_block']
                                mouse_psy['reward'] = mouse_psy['feedback']*(1*(mouse_psy['laser_block']==0))
                                mouse_psy['mouse'] = animal.name
                                mouse_psy['ses'] = ses.name
                                mouse_psy['date'] = day.name
                                data = pd.concat([data, mouse_psy])
                            else:
                                continue
                    else:
                        continue
            else:
                continue

data['unrewarded'] = 1*(data['reward']==0)
data['unlasered'] = 1*(data['laser']==0)



# Make prediction matrices
for mouse in data.mouse.unique():
    mouse_df=data.loc[data['mouse']==mouse]
    D=dict() # Dict for psytrack
    mouse_df = mouse_df[~np.isnan(mouse_df['choices'])] # Drop nan trials
    day_lens = []
    for m,date in enumerate(mouse_df.date.unique()):
        date_df = mouse_df.loc[mouse_df['date']==date]
        template = np.zeros([len(date_df), TRIALS_BACK])
        template[:] = np.nan
        rewarded_choices_d = template
        unrewarded_choices_d = template
        lasered_choices_d =  template
        unlasered_choices_d =  template
        choices_d = date_df['choices'].to_numpy()


        for i in np.arange(TRIALS_BACK):
            rewarded_choices_d[:,i] = date_df['choices'].shift(i+1)* \
                                      date_df['reward'].shift(i+1)
            unrewarded_choices_d[:,i] = date_df['choices'].shift(i+1)* \
                                      date_df['unrewarded'].shift(i+1)
            lasered_choices_d[:,i] = date_df['choices'].shift(i+1)* \
                                      date_df['laser'].shift(i+1)
            unlasered_choices_d[:,i] = date_df['choices'].shift(i+1)* \
                                      date_df['unlasered'].shift(i+1)
        choices_d = choices_d[~np.isnan(rewarded_choices_d).any(axis=1)]
        rewarded_choices_d = rewarded_choices_d[~np.isnan(rewarded_choices_d).any(axis=1), :]
        unrewarded_choices_d = unrewarded_choices_d[~np.isnan(unrewarded_choices_d).any(axis=1), :]
        lasered_choices_d = lasered_choices_d[~np.isnan(lasered_choices_d).any(axis=1), :]
        unlasered_choices_d = unlasered_choices_d[~np.isnan(unlasered_choices_d).any(axis=1), :]
        day_lens.append(len(choices_d))

        if m==0:
            rewarded_choices_m = rewarded_choices_d
            unrewarded_choices_m = unrewarded_choices_d
            lasered_choices_m = lasered_choices_d
            unlasered_choices_m = unlasered_choices_d
            choices_m = choices_d

        else:
            rewarded_choices_m = np.concatenate([rewarded_choices_m, rewarded_choices_d])
            unrewarded_choices_m = np.concatenate([unrewarded_choices_m, unrewarded_choices_d])
            lasered_choices_m = np.concatenate([lasered_choices_m, lasered_choices_d])
            unlasered_choices_m = np.concatenate([unlasered_choices_m, unlasered_choices_d])
            choices_m = np.concatenate([choices_m, choices_d])



    D['inputs'] = {'rc':rewarded_choices_m,
                    'urc':unrewarded_choices_m,
                    'lc':lasered_choices_m ,
                    'ulc':unlasered_choices_m}
    D['name'] = mouse
    D['y'] = 1*(choices_m>0)
    D['dayLength'] = np.array(day_lens)

    weights = {'bias': 1,  # a special key
           'rc': 3,    # use only the first column of s1 from inputs
           'urc': 3,
           'lc': 3,    # use only the first column of s1 from inputs
           'ulc': 3}    # use only the first column of s2 from inputs
    K = np.sum([weights[i] for i in weights.keys()])
    hyper= {'sigInit': 2**4.,      # Set to a single, large value for all weights. Will not be optimized further.
        'sigma': [2**-4.]*K,   # Each weight will have it's own sigma optimized, but all are initialized the same
        'sigDay': [2**-4.]*K}        # Indicates that session boundaries will be ignored in the optimization
    optList = ['sigma', 'sigDay']
    hyp, evd, wMode, hess_info = psy.hyperOpt(D, hyper, weights, optList)






# Create data dictionary
for mouse in data.mouse.unique():
    D_raw = data.loc[data['mouse']==mouse].to_dict()
    D=dict()
    D['y'] = D_raw['choices']
    # Prepare rewarded trials


    D['inputs']['r1']
