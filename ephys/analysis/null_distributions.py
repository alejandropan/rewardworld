import numpy as np
from ephys_alf_summary import alf, LASER_ONLY
import random
import pandas as pd
from pathlib import Path
random.seed(10)

# PATHS
destination = '/Volumes/witten/Alex/null_sessions/laser_only'

# 0 Load all the data in random order and concatenate and reset index
data = pd.DataFrame()
for ses in LASER_ONLY:
        ses_data = alf(ses, ephys=False)
        ses_data.mouse = Path(ses).parent.parent.name
        ses_data.date = Path(ses).parent.name
        ses_data.ses = Path(ses).name
        data = pd.concat([data,ses_data.to_df()[10:]])
data=data.reset_index()
# 1 Start the for loop
for n_null in np.arange(200):
    # 2 Take value for how much to circularly shift the data and shift the data
    shift = random.randint(0, len(data))
    data_t = data.reindex(index=np.roll(data.index,shift)).reset_index().copy()
    null = data_t.iloc[:1000, :]
    # 3 Save the data
    null.to_csv(destination+'/%s.csv' %n_null)

