import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as np
from encoding_model_summary_to_df import load_all_residuals, common_neural_data, common_trials
from decoding_debugging import *

##########################
####### Residuals #######
##########################
ROOT='/jukebox/witten/Alex/Data/Subjects/'
ROOT_NEURAL = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data/Subjects'
id_dict = pd.read_csv('/jukebox/witten/Alex/decoders_residuals_results/decoder_output_choice_cue_forget/id_dict.csv')
ses = ROOT+id_dict.loc[id_dict['id']==int(6),'ses'].to_string(index=False)
area = id_dict.loc[id_dict['id']==int(6),'area'].to_string(index=False)
alfio = alf(ses, ephys=True)
encoding_res_path = ROOT_NEURAL+'/'+ \
                    id_dict.loc[id_dict['id']==int(6),'ses'].to_string(index=False)+\
                    '/alf/encodingmodels/inputs/neurons/' 
neural_data = load_all_residuals(encoding_res_path)
neural_data = neural_data.loc[neural_data['location']==area]
c_neural_data_res = common_neural_data(neural_data, n_trials_minimum=100)
neural_data = load_all_residuals(encoding_res_path, filetype='real')
neural_data = neural_data.loc[neural_data['location']==area]
c_neural_data_raw = common_neural_data(neural_data, n_trials_minimum=100)
regressed_variable = [1*(np.copy(alfio.choice)>0), 1*(np.copy(alfio.choice)<1)] #right_choice is contra, left_choice is contra
regressed_variable = regressed_variable[1]
# Process residuals
reduced_residuals, trials_included = common_trials(c_neural_data_res, np.arange(len(c_neural_data_res)))
binned_spikes = reduced_residuals['residuals_goCue']
choices = regressed_variable[trials_included.astype(int)]
xs_s = reshape_psth_array(binned_spikes) # turn into array
#Order by trials by choice==1
order = np.argmax(np.mean(xs_s[np.where(choices==1),:,:][0], axis = 0), 1)
xs_sorted = xs_s[:,order.argsort(),:]
#Get the two arrays
xs_s_contra = xs_sorted[np.where(choices==1),:,:][0]
xs_s_ipsi = xs_sorted[np.where(choices==0),:,:][0]
#Mean by trial
xs_s_contra = np.mean(xs_s_contra,0)
xs_s_ipsi =  np.mean(xs_s_ipsi,0)

# Process raw
reduced_residuals, trials_included = common_trials(c_neural_data_raw, np.arange(len(c_neural_data_res)))
binned_spikes = reduced_residuals['residuals_goCue']
xs_s = reshape_psth_array(binned_spikes) # turn into array
#Order by trials by choice==1
xs_sorted = xs_s[:,order.argsort(),:]
#Get the two arrays
xs_s_contra_raw = xs_sorted[np.where(choices==1),:,:][0]
xs_s_ipsi_raw = xs_sorted[np.where(choices==0),:,:][0]
#Mean by trial
xs_s_contra_raw = np.mean(xs_s_contra_raw,0)
xs_s_ipsi_raw =  np.mean(xs_s_ipsi_raw,0)


fig, ax = plt.subplots(3,3, sharey=True,  sharex=True)
plt.sca(ax[0,0])
sns.heatmap(xs_s_contra, vmin=-0.5, vmax=0.5, center=0, cmap="bwr", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.ylabel('Residuals')
plt.title('Contra choices')
plt.sca(ax[0,1])
sns.heatmap(xs_s_ipsi, vmin=-0.5, vmax=0.5, center=0, cmap="bwr", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.title('Ipsi choices')
plt.sca(ax[0,2])
sns.heatmap(xs_s_contra - xs_s_ipsi, vmin=-0.5, vmax=0.5, center=0, cmap="bwr",
                cbar_kws={'label': 'sigma'})
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.title('Delta c-i')
plt.sca(ax[1,0])
sns.heatmap(xs_s_contra_raw, vmin=-0.5, vmax=0.5, center=0, cmap="bwr", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.ylabel('Raw')
plt.sca(ax[1,1])
sns.heatmap(xs_s_ipsi_raw, vmin=-0.5, vmax=0.5, center=0, cmap="bwr", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.sca(ax[1,2])
sns.heatmap(xs_s_contra_raw - xs_s_ipsi_raw, vmin=-0.5, vmax=0.5, center=0, cmap="bwr",
                cbar_kws={'label': 'sigma'})
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.sca(ax[2,0])
sns.heatmap((xs_s_contra-xs_s_contra_raw), vmin=-0.5, vmax=0.5, center=0, cmap="bwr", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.ylabel('Pred=Raw-Res')
plt.xlabel('Time from action')
plt.sca(ax[2,1])
sns.heatmap((xs_s_ipsi_raw-xs_s_ipsi), vmin=-0.5, vmax=0.5, center=0, cmap="bwr", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from action')
plt.sca(ax[2,2])
sns.heatmap(((xs_s_contra_raw - xs_s_ipsi_raw)-(xs_s_contra - xs_s_ipsi)), vmin=-0.5, vmax=0.5, center=0, cmap="bwr",
                cbar_kws={'label': 'sigma'})
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from action')
fig.savefig('summary.svg')

####
c = c_neural_data_res.cluster_id.to_numpy()[order.argsort()]

neural_data = load_all_residuals(encoding_res_path, filetype='prediction')
neural_data = neural_data.loc[neural_data['location']==area]
c_neural_data_pred = common_neural_data(neural_data, n_trials_minimum=100)

def plot_psth(psth, color='k', alpha=1):
    n_sample_size = psth.shape[0]
    psth_means = psth.mean(axis=0)
    psth_sem = psth.std(axis=0).T/np.sqrt(n_sample_size)
    psth_tscale = np.arange(-0.5,1,0.1)
    plt.plot(psth_tscale, psth_means.T, color=color, alpha=alpha)
    for m in np.arange(psth_means.shape[0]):
        plt.fill_between(psth_tscale,
                        psth_means.T - psth_sem,
                        psth_means.T + psth_sem,
                        alpha=alpha/2, color=color)
    plt.ylabel('Firing Rate')
    plt.xlabel('Time (s)')


neuron_res = c_neural_data_res.loc[c_neural_data_res['cluster_id']==20093].residuals_goCue.to_numpy()[0][trials_included.astype(int),:]
neuron_raw = c_neural_data_raw.loc[c_neural_data_raw['cluster_id']==20093].residuals_goCue.to_numpy()[0][trials_included.astype(int),:]
neuron_pred = c_neural_data_pred.loc[c_neural_data_pred['cluster_id']==20093].residuals_goCue.to_numpy()[0][trials_included.astype(int),:]

fig, ax = plt.subplots(1,3)
plt.sca(ax[0])
plot_psth(neuron_raw[np.where(choices==1)], color='r')
plot_psth(neuron_raw[np.where(choices==0)],  color='k')
plt.title('Raw')
sns.despine()
plt.sca(ax[1])
plot_psth(neuron_res[np.where(choices==1)], color='r')
plot_psth(neuron_res[np.where(choices==0)],  color='k')
plt.title('Residual')
sns.despine()
plt.sca(ax[2])
plot_psth(neuron_pred[np.where(choices==1)], color='r')
plot_psth(neuron_pred[np.where(choices==0)],  color='k')
plt.title('Prediction')
sns.despine()
plt.tight_layout()
fig.savefig('trial.svg')

old_path = '/jukebox/witten/Chris/data/ibl_da_neuropixels/Data_20221118/Subjects/dop_47/2022-06-05/001/alf/encodingmodels/inputs/neurons'
neural_data = load_all_residuals(old_path, filetype='prediction')
neural_data = neural_data.loc[neural_data['location']==area]
c_neural_data_pred_old = common_neural_data(neural_data, n_trials_minimum=100)
neural_data = load_all_residuals(old_path, filetype='real')
neural_data = neural_data.loc[neural_data['location']==area]
c_neural_data_raw_old = common_neural_data(neural_data, n_trials_minimum=100)
neural_data = load_all_residuals(old_path)
neural_data = neural_data.loc[neural_data['location']==area]
c_neural_data_res_old = common_neural_data(neural_data, n_trials_minimum=100)

neuron_res = c_neural_data_res_old.loc[c_neural_data_res_old['cluster_id']==20093].residuals_goCue.to_numpy()[0][trials_included.astype(int),:]
neuron_raw = c_neural_data_raw_old.loc[c_neural_data_raw_old['cluster_id']==20093].residuals_goCue.to_numpy()[0][trials_included.astype(int),:]
neuron_pred = c_neural_data_pred_old.loc[c_neural_data_pred_old['cluster_id']==20093].residuals_goCue.to_numpy()[0][trials_included.astype(int),:]

fig, ax = plt.subplots(1,3)
plt.sca(ax[0])
plot_psth(neuron_raw[np.where(choices==1)], color='r')
plot_psth(neuron_raw[np.where(choices==0)],  color='k')
plt.title('Raw')
sns.despine()
plt.sca(ax[1])
plot_psth(neuron_res[np.where(choices==1)], color='r')
plot_psth(neuron_res[np.where(choices==0)],  color='k')
plt.title('Residual')
sns.despine()
plt.sca(ax[2])
plot_psth(neuron_pred[np.where(choices==1)], color='r')
plot_psth(neuron_pred[np.where(choices==0)],  color='k')
plt.title('Prediction')
sns.despine()
plt.tight_layout()
fig.savefig('trial_old.svg')