# Mat to alf for 2022 global encoding model ssumary

from scipy.io import loadmat
import pandas as pd
import mat73
import glob
import numpy as np

ENCODING_MODEL_MAT_FILE = '/Volumes/witten/Chris/matlab/cz/ibl_da_neuropixels_2022/encoding-model/encoding-model-output-2023-01-04-02-00-PM.mat'

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
    #encoding_model['pretest'] = model_dict['EncodingSummary']['pvalues'][0][0]['pretest'][0][0][0]
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
    encoding_model['region_raw'] = model_dict['EncodingSummary']['region_name'][0][0][0]
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

def load_residual(neuron_file, model_bin_size=5, final_bins=100, pre_time = -500, post_time = 1000, filetype='residual', criterion='good'): # bin sizes in ms
    if criterion == 'good':
        acceptable_labels = 1
    elif criterion == 'mua':
        acceptable_labels = 2
    else:
        acceptable_labels = [1,2]
    residual_struct = mat73.loadmat(neuron_file)
    if ~np.isin(residual_struct['data']['cluster']['label'], acceptable_labels):
        return None
    # Start_dataframe
    neuron = pd.DataFrame()
    n_trials = int(len(residual_struct['data']['Y_pred']))
    # General info
    try:
        neuron['lambd'] = [residual_struct['data']['lambda'].tolist()]
    except:
        neuron['lambd'] = np.nan    
    neuron['trials_included'] = [residual_struct['data']['trials'] - 1]
    neuron['cluster_id_org'] =  int(residual_struct['data']['cluster']['clusterid'])
    neuron['probe_id'] = int(residual_struct['data']['cluster']['probe'][6])
    neuron['cluster_id'] =  (neuron['probe_id']*10000)+ neuron['cluster_id_org']
    neuron['animal'] =  residual_struct['data']['cluster']['session'][:6]
    neuron['date'] = residual_struct['data']['cluster']['session'][7:17]
    neuron['session'] = residual_struct['data']['cluster']['session'][18:21]
    neuron['area'] = residual_struct['data']['cluster']['location']
    neuron['hem'] = residual_struct['data']['cluster']['hem']
    # Make matrix
    pre_window = int(abs(pre_time/model_bin_size))
    post_window = int(abs(post_time/model_bin_size))
    residuals_goCue = np.zeros([n_trials, pre_window+post_window])
    residuals_choice =  np.zeros([n_trials, pre_window+post_window])
    residuals_outcome = np.zeros([n_trials, pre_window+post_window])
    for i in range(n_trials):
        go = int(residual_struct['data']['times'][i][0] - 1) # -1 to account for matlab indexing
        choice =  int(residual_struct['data']['times'][i][1] - 1) # -1 to account for matlab indexing
        outcome = int(residual_struct['data']['times'][i][2] - 1) # -1 to account for matlab indexing
        if filetype=='residual':
            go_data = residual_struct['data']['Y_resd'][i][go-pre_window:go+post_window] #
            choice_data = residual_struct['data']['Y_resd'][i][choice-pre_window:choice+post_window] #
            outcome_data = residual_struct['data']['Y_resd'][i][outcome-pre_window:outcome+post_window] #
        if filetype=='real':
            go_data = residual_struct['data']['Y_real'][i][go-pre_window:go+post_window] #
            choice_data = residual_struct['data']['Y_real'][i][choice-pre_window:choice+post_window] #
            outcome_data = residual_struct['data']['Y_real'][i][outcome-pre_window:outcome+post_window] # 
        if filetype=='prediction':
            go_data = residual_struct['data']['Y_pred'][i][go-pre_window:go+post_window] #
            choice_data = residual_struct['data']['Y_pred'][i][choice-pre_window:choice+post_window] #
            outcome_data = residual_struct['data']['Y_pred'][i][outcome-pre_window:outcome+post_window] #         
        assert len(go_data) == len(residuals_goCue[i,:])
        assert len(choice_data) == len(residuals_choice[i,:])
        assert len(outcome_data) == len(residuals_outcome[i,:])
        residuals_goCue[i,:] = go_data
        residuals_choice[i,:] = choice_data
        residuals_outcome[i,:] = outcome_data
    # Resample
    neuron['residuals_goCue'] = [resample(residuals_goCue,final_bins, model_bin_size)]
    neuron['residuals_choice'] =  [resample(residuals_choice,final_bins, model_bin_size)]
    neuron['residuals_outcome'] = [resample(residuals_outcome,final_bins, model_bin_size)]
    return neuron

def homogenize_neural_data(residuals, trials_included):
    reduced_residuals = pd.DataFrame()
    for i in np.arange(len(residuals)):
        new_neuron = pd.DataFrame()
        neuron = residuals.iloc[i].copy()
        assert np.array_equal(neuron['trials_included'], sorted(neuron['trials_included']))
        trials_idx = np.searchsorted(neuron['trials_included'], trials_included)
        new_neuron['cluster_id'] = [neuron['cluster_id']]
        new_neuron['animal'] = [neuron['animal']]
        new_neuron['date'] = [neuron['date']]
        new_neuron['session'] = [neuron['session']]
        new_neuron['area'] = [neuron['area']]
        new_neuron['hem'] = [neuron['hem']]
        new_neuron['location'] = [neuron['location']]
        new_neuron['trials_included'] = [neuron['trials_included'][trials_idx].copy()]
        new_neuron['residuals_goCue'] = [neuron['residuals_goCue'][trials_idx].copy()]
        new_neuron['residuals_choice'] = [neuron['residuals_choice'][trials_idx].copy()]
        new_neuron['residuals_outcome'] = [neuron['residuals_outcome'][trials_idx].copy()]
        assert new_neuron['residuals_goCue'][0].shape[0]==len(trials_included)
        reduced_residuals = pd.concat([reduced_residuals,new_neuron])
    return reduced_residuals

def load_all_residuals(root_path, filetype='residual', criterion='good'):
    path_to_all_residuals = glob.glob(root_path+'/*_residuals.mat')
    residuals = pd.DataFrame()
    for p in path_to_all_residuals:
        residuals = pd.concat([residuals, load_residual(p, filetype=filetype, criterion=criterion)])
    try:
        groups = pd.read_csv('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
    except:
        groups = pd.read_csv('/volumes/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
    groups = groups.iloc[:,1:3]
    groups = groups.set_index('original')
    group_dict = groups.to_dict()['group']
    residuals['location'] = pd.Series(residuals.area).map(group_dict)
    return residuals

def common_neural_data(residuals,n_trials_minimum=100):
    included = []
    for i in np.arange(len(residuals)):
        if len(residuals.iloc[i]['trials_included'])>=n_trials_minimum:
             included.append(i)
    return residuals.iloc[included,:]

def common_trials(neural_data, subsample):
    hem_neural_data = neural_data.iloc[subsample].copy()
    for i in np.arange(len(hem_neural_data)-1):    
        if i == 0:
            trials_included = hem_neural_data['trials_included'].iloc[i]
        t = hem_neural_data['trials_included'].iloc[i+1]
        trials_included = np.intersect1d(trials_included,t)
    reduced_residuals = homogenize_neural_data(hem_neural_data, trials_included)
    return reduced_residuals, trials_included

def common_trials_old(residuals):
    # First it excludes neurons with less than 2 std the number of trials
    select = []
    ts = []
    for i in np.arange(len(residuals)):    
        ts.append(len(residuals['trials_included'].iloc[i]))
    ts = np.array(ts)
    t_range = [int(np.median(ts)-np.std(ts)*2), int(np.median(ts)+np.std(ts)*2)]
    residuals = residuals.iloc[np.where(ts>=t_range[0])]
    for i in np.arange(len(residuals)-1):    
        if i == 0:
            select = residuals['trials_included'].iloc[i]
        t = residuals['trials_included'].iloc[i+1]
        select = np.intersect1d(select,t)
    return select, residuals

