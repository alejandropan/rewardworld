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

##########################
####### Parameters #######
##########################

ROOT='/jukebox/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
id_dict = pd.read_csv('/jukebox/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget/id_dict.csv')
n_neurons_minimum = 10
alignment_time = 'goCue_time'
pre_time = 0.5
post_time  = 4
smoothing=0
bin_size=0.1
output_folder = '/jukebox/witten/Alex/decoders_residuals_results/accuracy_by_region'
temp_folder = '/jukebox/witten/Alex/decoder_wd'

##########################
####### Load Data ########
##########################
ses = ROOT+id_dict.loc[id_dict['id']==int(sys.argv[1]),'ses'].to_string(index=False)
area = id_dict.loc[id_dict['id']==int(sys.argv[1]),'area'].to_string(index=False)

# Load behavior
alfio = alf(ses, ephys=False)
alfio.mouse = Path(ses).parent.parent.name
alfio.date = Path(ses).parent.name
alfio.ses = Path(ses).name
alfio.path = ses

choice_accuracy  = (1*(alfio.fchoice_prediction>0.5)) == (1*(alfio.choice>0))

# Load neurons
encoding_res_path = ROOT_NEURAL+'/'+ \
                    id_dict.loc[id_dict['id']==int(sys.argv[1]),'ses'].to_string(index=False)+\
                    '/alf/encodingmodels/inputs/neurons/' 
neural_data = load_all_residuals(encoding_res_path)
neural_data = neural_data.loc[neural_data['location']==area]

# Trials used
trials_included, neural_data = common_trials(neural_data)
c_neural_data = common_neural_data(neural_data, trials_included)
choice_accuracy = np.mean(choice_accuracy[trials_included.astype(int)])

# Divide in hemispheres

for h in c_neural_data['hem'].unique():
    hem_neural_data = c_neural_data.loc[c_neural_data['hem']==h]
    accu_summary = pd.DataFrame()
    accu_summary['choice_accuracy'] = [choice_accuracy]
    accu_summary['id'] = int(sys.argv[1])
    accu_summary['mouse'] = [alfio.mouse]
    accu_summary['ses'] = [alfio.date]
    accu_summary['region'] = [area]
    accu_summary['n_neurons'] = [len(hem_neural_data['cluster_id'].unique())]
    accu_summary['hemisphere'] = [h]
    accu_summary.to_csv(output_folder+'/'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_accu_summary.csv')