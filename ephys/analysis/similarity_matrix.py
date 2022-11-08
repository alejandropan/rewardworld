# network similarity matrix

import sys
sys.path.append('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/analysis')
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as npm
from encoding_model_summary_to_df import load_all_residuals, common_trials, common_neural_data
import warnings
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

##########################
####### Parameters #######
##########################

ROOT='/Volumes/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/Volumes/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
sessions = pd.read_csv('/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget/id_dict.csv').ses.unique() #fast way of loading all session ids
alignment_time = 'goCue_time'
pre_time = 0.5
post_time  = 4
smoothing=0
bin_size=0.1
output_folder = '/Volumes/witten/Alex/similarity_matrix/'
##########################
####### Load Data ########
##########################

for ses in sessions:
    # Load behavior
    ses_path = ROOT+ses
    alfio = alf(ses_path, ephys=False)
    alfio.mouse = Path(ses_path).parent.parent.name
    alfio.date = Path(ses_path).parent.name
    alfio.ses = Path(ses_path).name
    alfio.path = ses_path

    # Load neurons
    encoding_res_path = ROOT_NEURAL+'/'+ \
                       ses+\
                        '/alf/encodingmodels/inputs/neurons/' 
    neural_data = load_all_residuals(encoding_res_path, filetype='real')

    trials_included, h_neural_data = common_trials(neural_data)
    h_neural_data = common_neural_data(h_neural_data, trials_included)
    for h in h_neural_data.hem.unique():
        c_neural_data = h_neural_data.loc[h_neural_data['hem']==h].copy()
        c_neural_data = c_neural_data['residuals_goCue']
        similarity_matrix = np.zeros([len(trials_included),len(trials_included)])
        similarity_matrix[:]=np.nan
        for i in np.arange(len(trials_included)):
            for j in np.arange(len(trials_included)):
                x=[]
                y=[]
                for neuron in np.arange(len(c_neural_data)):
                    x.append(c_neural_data.iloc[neuron][i,:5].flatten()) # for iti analysis c_neural_data.iloc[neuron][i,:5]
                    y.append(c_neural_data.iloc[neuron][j,:5].flatten())
                x = np.concatenate(x)
                y = np.concatenate(y)
                r,_=pearsonr(x,y)
                similarity_matrix[i,j]=r

        #qChosen data
        alfio.fQRreward_cue = np.copy(np.roll(alfio.fQRreward,1))
        alfio.fQLreward_cue = np.copy(np.roll(alfio.fQLreward,1))
        alfio.fQRreward_cue[0] = 0
        alfio.fQLreward_cue[0] = 0
        regressed_variable = np.copy(alfio.fQRreward_cue) #For now qchosen
        regressed_variable[np.where(alfio.choice==-1)] = alfio.fQLreward_cue[np.where(alfio.choice==-1)] #For now qchosen
        regressed_variable = regressed_variable[trials_included.astype(int)]

        #All trials
        fig,ax = plt.subplots(1,4, figsize=(20, 15))
        plt.sca(ax[0])
        block_changes = np.where(np.diff(alfio.probabilityLeft)!=0)[0]+1
        block_changes = np.searchsorted(trials_included, block_changes)
        sns.heatmap(similarity_matrix, cmap='RdGy', center=0, square=True, vmin=similarity_matrix.min(), vmax=np.percentile(similarity_matrix,99), cbar=False)
        [plt.axvline(_x, linewidth=1.2, color='blue', linestyle='--') for _x in block_changes]
        [plt.axhline(_x, linewidth=1.2, color='blue', linestyle='--') for _x in block_changes]
        plt.title(ses + ' all trials')
        #Rewarded trials
        plt.sca(ax[1])
        rewarded_trials = np.intersect1d(trials_included,np.where(alfio.outcome==1)[0]).astype(int)
        rewarded_trials = np.searchsorted(trials_included, rewarded_trials)
        block_changes = np.where(np.diff(alfio.probabilityLeft)!=0)[0]+1
        block_changes = np.searchsorted(trials_included, block_changes)
        block_changes = np.searchsorted(rewarded_trials, block_changes)
        similarity_matrix_r = similarity_matrix[rewarded_trials, :]
        similarity_matrix_r = similarity_matrix_r[:,rewarded_trials]
        sns.heatmap(similarity_matrix_r, cmap='RdGy', center=0, square=True,vmin=similarity_matrix.min(), vmax=np.percentile(similarity_matrix,99),  cbar=False)
        [plt.axvline(_x,  color='blue', linestyle='--') for _x in block_changes]
        [plt.axhline(_x,  color='blue', linestyle='--') for _x in block_changes]
        plt.title(ses + ' R trials')
        #Rewarded and right trials
        plt.sca(ax[2])

        rewarded_right_trials = np.intersect1d(np.where(alfio.choice==1)[0], np.where(alfio.outcome==1)[0]).astype(int)
        rewarded_trials = np.intersect1d(trials_included,rewarded_right_trials).astype(int)
        rewarded_trials = np.searchsorted(trials_included, rewarded_trials)
        block_changes = np.where(np.diff(alfio.probabilityLeft)!=0)[0]+1
        block_changes = np.searchsorted(trials_included, block_changes)
        block_changes = np.searchsorted(rewarded_trials, block_changes)
        similarity_matrix_r = similarity_matrix[rewarded_trials, :]
        similarity_matrix_r = similarity_matrix_r[:,rewarded_trials]
        sns.heatmap(similarity_matrix_r, cmap='RdGy', center=0, square=True,vmin=similarity_matrix.min(), vmax=np.percentile(similarity_matrix,99),  cbar=False)
        [plt.axvline(_x,  color='blue', linestyle='--') for _x in block_changes]
        [plt.axhline(_x,  color='blue', linestyle='--') for _x in block_changes]
        plt.title(ses + ' right R trials')
        plt.sca(ax[3])
        rewarded_left_trials = np.intersect1d(np.where(alfio.choice==-1)[0], np.where(alfio.outcome==1)[0]).astype(int)
        rewarded_trials = np.intersect1d(trials_included,rewarded_left_trials).astype(int)
        rewarded_trials = np.searchsorted(trials_included, rewarded_trials)
        block_changes = np.where(np.diff(alfio.probabilityLeft)!=0)[0]+1
        block_changes = np.searchsorted(trials_included, block_changes)
        block_changes = np.searchsorted(rewarded_trials, block_changes)
        similarity_matrix_r = similarity_matrix[rewarded_trials, :]
        similarity_matrix_r = similarity_matrix_r[:,rewarded_trials]
        sns.heatmap(similarity_matrix_r, cmap='RdGy', center=0, square=True,vmin=similarity_matrix.min(), vmax=np.percentile(similarity_matrix,99))
        [plt.axvline(_x, color='blue', linestyle='--') for _x in block_changes]
        [plt.axhline(_x, color='blue', linestyle='--') for _x in block_changes]
        plt.title(ses + ' left R trials')
        ses_str = ses[:6]+'_'+ses[7:17]
        plt.tight_layout()
        plt.savefig(output_folder+ses_str+'_hem_'+str(h)+'.pdf')


