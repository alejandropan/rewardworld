import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as npm
from encoding_model_summary_to_df import load_all_residuals, common_trials, common_neural_data
from decoding_debugging import *
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt

#Parameters
ROOT='/jukebox/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
id_dict = pd.read_csv('/jukebox/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget/id_dict.csv')
sessions = id_dict.ses.unique()

#Storing variables
total =[]
used =[]

for ses in sessions:
    potential_trials =  len(alf(ROOT + ses, ephys=False).choice)-10
    encoding_res_path = ROOT_NEURAL+'/' + ses + \
                        '/alf/encodingmodels/inputs/neurons/' 
    neural_data = load_all_residuals(encoding_res_path)
    areas = id_dict.loc[id_dict['ses']==ses,'area'].unique()

    for area in areas:
        region_data = neural_data.loc[neural_data['location']==area]
        if region_data.shape[0]>=10:
            trials_included, _ = common_trials(region_data)
            total.append(potential_trials)
            used.append(len(trials_included))

# Calculate percent and plot
data = used/total
sns.histplot(data, stat='percent')
plt.xlabel('Fraction of trials used')
plt.ylabel('%')