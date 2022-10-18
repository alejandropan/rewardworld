# Mat to alf for 2022 global encoding model ssumary

from scipy.io import loadmat
import pandas as pd
import mat73
import glob

ENCODING_MODEL_MAT_FILE = '/Volumes/witten/Chris/matlab/cz/ibl_da_neuropixels_2022/archive/20220913/encoding-model/encoding-model-output-2022-08-03-12-31-PM.mat'

def load_encoding_model(model_path = ENCODING_MODEL_MAT_FILE):
    model_dict = loadmat(model_path)
    encoding_model = pd.DataFrame()
    mouse=[]
    date=[]
    ses=[]
    id=[]
    probe=[]
    [mouse.append(i[0][:6]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [date.append(i[0][7:17]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [ses.append(i[0][18:21]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [id.append(i[0]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [probe.append(int(i[0][6])) for i in model_dict['EncodingSummary']['probe'][0][0][0]]
    encoding_model['pretest'] = model_dict['EncodingSummary']['pvalues'][0][0]['pretest'][0][0][0]
    encoding_model['choice'] = model_dict['EncodingSummary']['pvalues'][0][0]['choice'][0][0][0]
    encoding_model['outcome'] = model_dict['EncodingSummary']['pvalues'][0][0]['outcome'][0][0][0]
    encoding_model['policy'] = model_dict['EncodingSummary']['pvalues'][0][0]['policy'][0][0][0]
    encoding_model['value'] = model_dict['EncodingSummary']['pvalues'][0][0]['value'][0][0][0]
    encoding_model['value_laser'] = model_dict['EncodingSummary']['pvalues'][0][0]['value_laser'][0][0][0]
    encoding_model['value_water'] = model_dict['EncodingSummary']['pvalues'][0][0]['value_water'][0][0][0]
    encoding_model['value_stay'] = model_dict['EncodingSummary']['pvalues'][0][0]['value_stay'][0][0][0]
    encoding_model['policy_laser'] = model_dict['EncodingSummary']['pvalues'][0][0]['policy_laser'][0][0][0]
    encoding_model['policy_water'] = model_dict['EncodingSummary']['pvalues'][0][0]['policy_water'][0][0][0]
    encoding_model['policy_stay'] = model_dict['EncodingSummary']['pvalues'][0][0]['policy_stay'][0][0][0]
    encoding_model['cluster_id'] = model_dict['EncodingSummary']['cluster'][0][0][0].astype(int)
    encoding_model['laser_cue_gain'] = model_dict['EncodingSummary']['gains'][0][0]['cue'][0][0][:,5]
    encoding_model['laser_contra_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_contra'][0][0][:,5]
    encoding_model['laser_ipsi_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_ipsi'][0][0][:,5]
    encoding_model['outcome_laser_gain'] = model_dict['EncodingSummary']['gains'][0][0]['outcome_laser'][0][0][:,5]
    encoding_model['dlaser_cue_gain'] = model_dict['EncodingSummary']['gains'][0][0]['cue'][0][0][:,2]
    encoding_model['dlaser_contra_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_contra'][0][0][:,2]
    encoding_model['dlaser_ipsi_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_ipsi'][0][0][:,2]
    encoding_model['doutcome_laser_gain'] = model_dict['EncodingSummary']['gains'][0][0]['outcome_laser'][0][0][:,2]
    encoding_model['firing_rate'] =  model_dict['EncodingSummary']['firing_rate'][0][0][0]
    encoding_model['region'] =  model_dict['EncodingSummary']['region_key'][0][0][0][model_dict['EncodingSummary']['region_id'][0][0][0]-1]
    encoding_model['mouse'] = mouse
    encoding_model['date'] = date
    encoding_model['ses'] = ses
    encoding_model['probe'] = probe
    encoding_model['id'] = id
    return encoding_model

def summary_firing_rate(encoding_model):
    fig,ax = plt.subplots(5,5, sharex=True, sharey=True)
    for i, reg in enumerate(np.unique(encoding_model['region'])):
        plt.sca(ax[int(i/5),i%5])
        encoding_model_reg = encoding_model.loc[encoding_model['region']==reg[0]]
        value = encoding_model_reg.loc[encoding_model_reg['policy_laser']<=0.01, 'firing_rate']
        non_value = encoding_model_reg.loc[encoding_model_reg['policy_laser']>0.01, 'firing_rate']
        sns.histplot(non_value,stat='percent', bins=np.arange(0,100,5))
        sns.histplot(value,stat='percent',bins=np.arange(0,100,5), color='orange')
        plt.title(reg[0])
        sns.despine()

def resample (psth_array,final_bins, model_bin_size):
    resampling_factor = final_bins/model_bin_size
    new_n_col = int(psth_array.shape[1]/resampling_factor)
    new_psth_array = np.zeros([psth_array.shape[0], new_n_col])
    for j in np.arange(new_n_col):
        new_psth_array[:,j] = np.mean(psth_array[:,int(resampling_factor*j):int(resampling_factor*(j+1))], 
                                        axis = 1)
    return new_psth_array

def load_residual(neuron_file, model_bin_size=5, final_bins=100, pre_time = -500, post_time = 2000): # bin sizes in ms
    residual_struct = mat73.loadmat(neuron_file)
    # Start_dataframe
    neuron = pd.DataFrame()
    # General info
    neuron['n_trials'] = len(residual_struct['data']['Y_pred'])
    neuron['trials_included'] = residual_struct['data']['trials']
    neuron['cluster_id'] =  int(residual_struct['data']['cluster']['clusterid'])
    neuron['animal'] =  residual_struct['data']['cluster']['session'][:6]
    neuron['date'] = residual_struct['data']['cluster']['session'][7:17]
    neuron['session'] = residual_struct['data']['cluster']['session'][18:21]
    # Make matrix
    pre_window = int(abs(pre_time/model_bin_size))
    post_window = int(abs(post_time/model_bin_size))
    residuals_goCue = np.zeros([int(neuron['n_trials']), pre_window+post_window])
    residuals_choice =  np.zeros([int(neuron['n_trials']), pre_window+post_window])
    residuals_outcome = np.zeros([int(neuron['n_trials']), pre_window+post_window])
    for i in range(n_trials):
        go = int(residual_struct['data']['times'][i][0] - 1) # -1 to account for matlab indexing
        choice =  int(residual_struct['data']['times'][i][1] - 1) # -1 to account for matlab indexing
        outcome = int(residual_struct['data']['times'][i][2] - 1) # -1 to account for matlab indexing
        go_data = residual_struct['data']['Y_pred'][i][go-pre_window:go+post_window]
        choice_data = residual_struct['data']['Y_pred'][i][choice-pre_window:choice+post_window]
        outcome_data = residual_struct['data']['Y_pred'][i][outcome-pre_window:outcome+post_window]
        assert len(go_data) == len(residuals_goCue[i,:])
        assert len(choice_data) == len(residuals_choice[i,:])
        assert len(outcome_data) == len(residuals_outcome[i,:])
        residuals_goCue[i,:] = go_data
        residuals_choice[i,:] = choice_data
        residuals_outcome[i,:] = outcome_data
    # Resample
    neuron['residuals_goCue'] = resample(residuals_goCue,final_bins, model_bin_size)
    neuron['residuals_choice'] =  resample(residuals_choice,final_bins, model_bin_size)
    neuron['residuals_outcome'] = resample(residuals_outcome,final_bins, model_bin_size)
    return neuron

def load_all_residuals(root_path):
    path_to_all_residuals = glob.glob(root_path+'/*_residuals.mat')
    residuals = pd.DataFrame()
    for p in path_to_all_residuals:
        residuals = pd.concat([residuals, load_residual(p)])
    return residuals
    
    

