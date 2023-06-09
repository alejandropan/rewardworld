# What information is DLS receiving

from ephys_alf_summary import *
from scipy.stats import zscore
from sklearn.model_selection import KFold
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.io import loadmat
import scipy.stats as sps
import scipy.ndimage
from sklearn.metrics import r2_score
from psths_per_region import *

# Parameters
ROIS = np.array(['MO', 'DLS'])
MIN_N = 15
BIN_SIZE = 0.1 # in seconds

def flatten_variable(variable, n_neural_data_bins = 101600):
    trial_window = int(n_neural_data_bins/len(variable))
    new_variable = []
    for i in variable:
        new_variable.append([i]*trial_window)
    return np.concatenate(new_variable)

def flatten_trials(array):
    farray = np.zeros([array.shape[0]* array.shape[2], array.shape[1]])
    for i in np.arange(array.shape[1]):
        farray_temp = []
        for j in np.arange(array.shape[0]):
            farray_temp.append(array[j,i,:])
        farray[:,i] = np.concatenate(farray_temp) 
    return farray

def halfgaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Half-Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(0, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    return phi_x

def halfgaussian_filter1d(input, sigma, axis=-1, output=None,
                      mode="constant", cval=0.0, truncate=4.0):
    """
    Convolves a 1-D Half-Gaussian convolution kernel.
    """
    sd = float(sigma)
    # make the radius of the filter equal to truncate standard deviations
    lw = int(truncate * sd + 0.5)
    weights = halfgaussian_kernel1d(sigma, lw)
    origin = -lw // 2
    return scipy.ndimage.convolve1d(input, weights, axis, output, mode, cval, origin)

def filterneurons(X, zscoring=False, zscoring_params=None, bins =0.1, bin_size=0.025):
    if zscoring==True:
        X = (X - zscoring_params[:,0][:,None])/(zscoring_params[:,1][:,None])
    resampling_factor = int(bins/bin_size)
    new_n_col = int(X.shape[2]/resampling_factor)
    new_psth_array = np.zeros([X.shape[0], X.shape[1], new_n_col])
    for j in np.arange(new_n_col):
        new_psth_array[:,:,j] = np.sum(X[:,:,int(resampling_factor*j):int(resampling_factor*(j+1))], 
                                        axis = 2)
    return new_psth_array 

def zscore_set_params(array, params): # array myst be neuron X time_bins
    norm_array = (array - params[:,0][:,None])/(params[:,1][:,None])
    return norm_array

def get_connectivity_summaries(sessions, ROIS, allow_callosal=False):
    if allow_callosal==True:
        connected_summary  = pd.DataFrame()
        for i in np.arange(len(sessions[:])):
            ses = sessions[i]
            connected_ses_hem = pd.DataFrame()
            source = []
            destination = []
            for j in np.arange(len(ses.probe[:])):
                select = ses.probe[j].cluster_goodmuaselection
                locations = pd.Series(ses.probe[j].cluster_locations).map(group_dict)
                source.append(np.intersect1d(np.where(locations==ROIS[0]), select))
                destination.append(np.intersect1d(np.where(locations==ROIS[1]), select))
            source = np.concatenate(source)
            destination = np.concatenate(destination)
            if (len(source)>=1) & (len(destination)>=1):
                connected_ses_hem['mouse'] = [ses.mouse]
                connected_ses_hem['date'] = [ses.date]
                connected_ses_hem['ses'] = [ses.ses]
                connected_ses_hem['i'] = [i]
                connected_ses_hem['hem'] = [np.nan]
                connected_ses_hem['n_source'] = [len(source)]
                connected_ses_hem['n_destination'] = [len(destination)]
                connected_summary = pd.concat([connected_summary,connected_ses_hem])
        connected_summary = connected_summary.reset_index()
    else:
        connected_summary  = pd.DataFrame()
        for i in np.arange(len(sessions[:])):
            ses = sessions[i]
            for hem in np.arange(2):
                connected_ses_hem = pd.DataFrame()
                source = []
                destination = []
                for j in np.arange(len(ses.probe[:])):
                    select = ses.probe[j].cluster_goodmuaselection
                    locations = pd.Series(ses.probe[j].cluster_locations[ses.probe[j].cluster_hem==hem]).map(group_dict)
                    source.append(np.intersect1d(np.where(locations==ROIS[0]), select))
                    destination.append(np.intersect1d(np.where(locations==ROIS[1]), select))
                source = np.concatenate(source)
                destination = np.concatenate(destination)
                if (len(source)>=1) & (len(destination)>=1):
                    connected_ses_hem['mouse'] = [ses.mouse]
                    connected_ses_hem['date'] = [ses.date]
                    connected_ses_hem['ses'] = [ses.ses]
                    connected_ses_hem['i'] = [i]
                    connected_ses_hem['hem'] = [hem]
                    connected_ses_hem['n_source'] = [len(source)]
                    connected_ses_hem['n_destination'] = [len(destination)]
                    connected_summary = pd.concat([connected_summary,connected_ses_hem])
        connected_summary = connected_summary.reset_index()
    return connected_summary

sessions = ephys_ephys_dataset(len(LASER_ONLY))
for i, ses in enumerate(LASER_ONLY):
        print(ses)
        ses_data = alf(ses, ephys=True)
        ses_data.mouse = Path(ses).parent.parent.name
        ses_data.date = Path(ses).parent.name
        ses_data.ses = Path(ses).name
        sessions[i] = ses_data

# Look for any unreferenced regions
groups = pd.read_csv('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
groups = groups.iloc[:,1:3]
groups = groups.set_index('original')
group_dict = groups.to_dict()['group']
current_regions = groups.reset_index().original.unique()
[group_dict[r] for r in current_regions] # This will error if dictionary is not complete

# Check relevant sessions, currently focusing on same hemisphere communication (min n=20 in each area)
connected_summary = get_connectivity_summaries(sessions, ROIS, allow_callosal=False) # Allow_callosal determines whether across hemisphere communication is allowed for analysis
usable = connected_summary.loc[(connected_summary['n_source']>=MIN_N)&(connected_summary['n_destination']>=MIN_N)].reset_index()

# Example session to work with is i=22, hem=1
# Eventually make this into a for loop to go thorugh every session for now , example session
# bin[i-1]<spike time<bin[i]
ses_of_interest = sessions[22]
hems = np.array([1])
connected_ses_hem = pd.DataFrame()
source_matrix = []
destination_matrix = []
bins = np.arange(ses_of_interest.start_time[10],ses_of_interest.start_time[-1], BIN_SIZE)
for j in np.arange(len(ses.probe[:])):
    select = ses_of_interest.probe[j].cluster_goodmuaselection
    locations = pd.Series(ses_of_interest.probe[j].cluster_locations[np.isin(ses_of_interest.probe[j].cluster_hem,hems)]).map(group_dict)
    source_clusters = np.intersect1d(np.where(locations==ROIS[0]), select)
    destination_clusters = np.intersect1d(np.where(locations==ROIS[1]), select)
    for s_c in source_clusters:
        spt = ses_of_interest.probe[j].spike_times[ses_of_interest.probe[j].spike_clusters==s_c]
        # trial 10 to trial -1
        spt = spt[np.where(spt>=ses_of_interest.start_time[10])]
        spt = spt[np.where(spt<=ses_of_interest.start_time[-1])]
        binned = np.bincount(np.digitize(spt,bins),minlength=len(bins)+1)/BIN_SIZE #divide by bin_size for firing rate
        source_matrix.append(binned)
    for d_c in destination_clusters:
        spt = ses_of_interest.probe[j].spike_times[ses_of_interest.probe[j].spike_clusters==d_c]
        # trial 10 to trial -1
        spt = spt[np.where(spt>=ses_of_interest.start_time[10])]
        spt = spt[np.where(spt<=ses_of_interest.start_time[-1])]
        binned = np.bincount(np.digitize(spt,bins),minlength=len(bins)+1)/BIN_SIZE #divide by bin_size for firing rate
        destination_matrix.append(binned)
source_matrix = np.stack(source_matrix)
destination_matrix = np.stack(destination_matrix)


# Look at information encoded
source_region = ROIS[0]
dest_region =  ROIS[1]
min_number_of_neurons = 10
Ncrossval = 5
lam = 0

# Filter by receiving region
# Make session_hem unique id, i.e. separate by id but group probes in same hemisphere

region_df_grouped['hem_id'] = region_df_grouped['mouse'] + '_' + region_df_grouped['date'] + '_' + \
                              region_df_grouped['ses'] + '_' + region_df_grouped['hem'].astype(str)

sim_recorded_regions = region_df_grouped.groupby('hem_id')['region'].apply(lambda x: list(np.unique(x))).reset_index()

# Filter ids from session with simultaneously recorded neurons of interest
sessions_of_interest = sim_recorded_regions.loc[ 
    sim_recorded_regions['region'].apply(lambda x: (np.isin(source_region, x)) & (np.isin(dest_region, x))),
    'hem_id'].to_list()

# Now exclude sessions with not enough neurons in the area of interest, unit quality selection already goeas in the making of region_df_grouped
curated_simultaneous_recordings= []
for ses in sessions_of_interest:
    recording = region_df_grouped.loc[region_df_grouped['hem_id']==ses]
    n_source = recording.loc[recording['region']==source_region, 'binned_spikes_gocue'].iloc[0].shape[1]
    n_dest = recording.loc[recording['region']==dest_region, 'binned_spikes_gocue'].iloc[0].shape[1]
    if (n_source>=min_number_of_neurons) & (n_dest>=min_number_of_neurons):
        curated_simultaneous_recordings.append(ses)

# for every session in simultaneous
communication = pd.DataFrame()
for rec in curated_simultaneous_recordings:
    print(rec)
    sim_recording = region_df_grouped[region_df_grouped['hem_id'] ==rec]
    X = sim_recording.loc[sim_recording['region']==source_region, 'binned_spikes_gocue'].iloc[0]  # source region
    Y = sim_recording.loc[sim_recording['region']==dest_region, 'binned_spikes_gocue'].iloc[0] # dest region
    X =  X[:,:,20:] # Get rid of pre time + outcome, just use last 2 seconds of ITI
    Y =  Y[:,:,20:] # Get rid of pre time + outcome just use last 2 seconds of ITI
    X  = halfgaussian_filter1d(X, 1.5, axis=2, output=None,
                      mode="constant", cval=0.0, truncate=4.0)
    Y  = halfgaussian_filter1d(Y, 1.5, axis=2, output=None,
                      mode="constant", cval=0.0, truncate=4.0)
    # Filter as wanted (i.e, 100ms bins with/without zscoring)
    #X = filterneurons(X, zscoring=False, zscoring_params=None, bins = 0.1, bin_size=0.025)
    #Y = filterneurons(Y, zscoring=False, zscoring_params=None, bins = 0.1, bin_size=0.025)
    # Divide in folds
    kf = KFold(n_splits=Ncrossval)
    error = np.zeros([len(np.arange(2,10)), Ncrossval])
    max_pred = np.zeros([Ncrossval])
    for i, indexes in enumerate(kf.split(np.arange(X.shape[0]))):
        train_index , test_index= indexes
        X_train , X_test = X[train_index,:,:], X[test_index,:,:]
        y_train , y_test = Y[train_index,:,:], Y[test_index,:,:]
        #for j,m in enumerate(np.arange(1,np.min([X.shape[1],Y.shape[1]]))):  # Find communication subspace dimension
        for j,m in enumerate(np.arange(2,10)):  # Find communication subspace dimension
            try:
                # Reshape (i.e concatenate trials)
                X_train_f = flatten_trials(X_train)
                y_train_f = flatten_trials(y_train)
                X_test_f = flatten_trials(X_test)
                y_test_f = flatten_trials(y_test)
                n, p = X_train_f.shape
                _, q = y_train_f.shape
                # RRR
                Bols = np.linalg.inv(X_train_f.T @ X_train_f + lam*np.eye(p)) @ (X_train_f.T @ y_train_f)  # Estimate for B with ordinary least squares
                Yestimate = X_train_f @ Bols
                pca = PCA(n_components=m)
                pca.fit(Yestimate)
                V = pca.components_.T
                Brrr = Bols @ V @ V.T
                Yestimate = X_test_f @ Brrr
                Yestimate_lr = X_test_f @ Bols
                error[j,i] = 1 - np.var(Yestimate - y_test_f) / np.var(y_test_f)
            except:
                continue
            #error[j,i] = sps.pearsonr(Yestimate.flatten(), y_test_f.flatten())[0]
        try:
            max_pred[i] = 1 - np.var(Yestimate_lr - y_test_f) / np.var(y_test_f)
        except:
            continue
    error_f = np.mean(error, axis=1)
    max_pred = np.mean(max_pred) #upper bound? 
    # Find lowest dimension within one SD from peak performance (As in Semedo et al., 2019)
    best_pred = np.argmax(error_f)
    selected_n_dimensions= np.where(error_f>=(error_f[best_pred]-np.std(error[best_pred,:])))[0][0]+2
    # Now that we now n_dimension project data and decode variables of interest
    choice = 1*(sim_recording['choice'].iloc[0]>0)
    outcome = sim_recording['outcome'].iloc[0]
    qchosen = sim_recording['qchosen'].iloc[0]
    delta = sim_recording['delta'].iloc[0]
    state_value = sim_recording['state_value'].iloc[0]
    # Generate estimates from selected dimensionality and decode varibles
    error_choice = np.zeros([Ncrossval,3])
    error_outcome = np.zeros([Ncrossval,3])
    error_qchosen = np.zeros([Ncrossval,3])
    error_delta = np.zeros([Ncrossval,3])
    error_state_value = np.zeros([Ncrossval,3])
    for i, indexes in enumerate(kf.split(np.arange(X.shape[0]))):
        try:
            train_index , test_index= indexes
            X_train , X_test = X[train_index,:,:], X[test_index,:,:]
            y_train , y_test = Y[train_index,:,:], Y[test_index,:,:]
            # Reshape (i.e concatenate trials)
            X_train_f = flatten_trials(X_train)
            y_train_f = flatten_trials(y_train)
            X_test_f = flatten_trials(X_test)
            y_test_f = flatten_trials(y_test)
            n, p = X_train_f.shape
            _, q = y_train_f.shape
            # RRR
            Bols = np.linalg.inv(X_train_f.T @ X_train_f + lam*np.eye(p)) @ (X_train_f.T @ y_train_f)  # Estimate for B with ordinary least squares
            Yestimate = X_train_f @ Bols
            pca = PCA(n_components=selected_n_dimensions)
            pca.fit(Yestimate)
            V = pca.components_.T
            Brrr = Bols @ V @ V.T
            Yestimate_train = X_train_f @ Brrr
            Yestimate_test = X_test_f @ Brrr
            # Run on subspace
            # TODO CHECK DIMENSION OF ERROR CALCULATIONS AND FLATTEN VARIABLES BEYON Yestimate
            model_subspace_choice = LogisticRegression().fit(Yestimate_train, flatten_variable(choice[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_outcome = LogisticRegression().fit(Yestimate_train, flatten_variable(outcome[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_qchosen = LinearRegression().fit(Yestimate_train, flatten_variable(qchosen[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_delta = LinearRegression().fit(Yestimate_train, flatten_variable(delta[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_state_value = LinearRegression().fit(Yestimate_train, flatten_variable(state_value[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            error_choice[i,0] = np.sum(model_subspace_choice.predict(Yestimate_test)==
                                flatten_variable(choice[test_index],n_neural_data_bins=Yestimate_test.shape[0]))/len(Yestimate_test)
            error_outcome[i,0] = np.sum(model_subspace_outcome.predict(Yestimate_test)==
                                flatten_variable(outcome[test_index],n_neural_data_bins=Yestimate_test.shape[0]))/len(Yestimate_test)
            error_qchosen[i,0] = 1 - np.var(model_subspace_qchosen.predict(Yestimate_test) -  flatten_variable(qchosen[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(qchosen[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
            error_delta[i,0] = 1 - np.var(model_subspace_delta.predict(Yestimate_test) -  flatten_variable(delta[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(delta[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
            error_state_value[i,0] = 1 - np.var(model_subspace_state_value.predict(Yestimate_test) -  flatten_variable(state_value[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(state_value[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
            # Run on full space of source
            model_subspace_choice = LogisticRegression().fit(X_train_f, flatten_variable(choice[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_outcome = LogisticRegression().fit(X_train_f, flatten_variable(outcome[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_qchosen = LinearRegression().fit(X_train_f, flatten_variable(qchosen[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_delta = LinearRegression().fit(X_train_f, flatten_variable(delta[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_state_value = LinearRegression().fit(X_train_f,  flatten_variable(state_value[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            error_choice[i,1] = np.sum(model_subspace_choice.predict(X_test_f)==
                                flatten_variable(choice[test_index],n_neural_data_bins=X_test_f.shape[0]))/len(X_test_f)
            error_outcome[i,1] = np.sum(model_subspace_outcome.predict(X_test_f)==
                                flatten_variable(outcome[test_index],n_neural_data_bins=X_test_f.shape[0]))/len(X_test_f)
            error_qchosen[i,1] = 1 - np.var(model_subspace_qchosen.predict(X_test_f) - flatten_variable(qchosen[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(qchosen[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
            error_delta[i,1] = 1 - np.var(model_subspace_delta.predict(X_test_f) - flatten_variable(delta[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(delta[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
            error_state_value[i,1] = 1 - np.var(model_subspace_state_value.predict(X_test_f) - flatten_variable(state_value[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(state_value[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
            # Run on full space of destination
            model_subspace_choice = LogisticRegression().fit(y_train_f, flatten_variable(choice[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_outcome = LogisticRegression().fit(y_train_f, flatten_variable(outcome[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_qchosen = LinearRegression().fit(y_train_f,  flatten_variable(qchosen[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_delta = LinearRegression().fit(y_train_f, flatten_variable(delta[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            model_subspace_state_value = LinearRegression().fit(y_train_f,  flatten_variable(state_value[train_index],n_neural_data_bins=Yestimate_train.shape[0]))
            error_choice[i,2] = np.sum(model_subspace_choice.predict(y_test_f)==
                                flatten_variable(choice[test_index],n_neural_data_bins=y_test_f.shape[0]))/len(y_test_f)
            error_outcome[i,2] = np.sum(model_subspace_outcome.predict(y_test_f)==
                                flatten_variable(outcome[test_index],n_neural_data_bins=y_test_f.shape[0]))/len(y_test_f)   
            error_qchosen[i,2] = 1 - np.var(model_subspace_qchosen.predict(y_test_f) - flatten_variable(qchosen[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(qchosen[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
            error_delta[i,2] = 1 - np.var(model_subspace_delta.predict(y_test_f) - flatten_variable(delta[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(delta[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
            error_state_value[i,2] = 1 - np.var(model_subspace_state_value.predict(y_test_f) - flatten_variable(state_value[test_index],n_neural_data_bins=Yestimate_test.shape[0])) / np.var(flatten_variable(state_value[test_index],n_neural_data_bins=Yestimate_test.shape[0]))
        except:
            error_choice[i,:]=np.nan
            error_outcome[i,:]=np.nan
            error_qchosen[i,:]=np.nan
            error_delta[i,:]=np.nan
            error_state_value[i,:]=np.nan
            continue
    rec_communication = pd.DataFrame()
    rec_communication['choice_communicated'] = [np.nanmean(error_choice[:,0])]
    rec_communication['choice_at_source'] = [np.nanmean(error_choice[:,1])]
    rec_communication['choice_at_destination'] = [np.nanmean(error_choice[:,2])]
    rec_communication['outcome_communicated'] = [np.nanmean(error_outcome[:,0])]
    rec_communication['outcome_at_source'] = [np.nanmean(error_outcome[:,1])]
    rec_communication['outcome_at_destination'] = [np.nanmean(error_outcome[:,2])]
    rec_communication['qchosen_communicated'] = [np.nanmean(error_qchosen[:,0])]
    rec_communication['qchosen_at_source'] = [np.nanmean(error_qchosen[:,1])]
    rec_communication['qchosen_at_destination'] = [np.nanmean(error_qchosen[:,2])]
    rec_communication['delta_communicated'] = [np.nanmean(error_delta[:,0])]
    rec_communication['delta_at_source'] = [np.nanmean(error_delta[:,1])]
    rec_communication['delta_at_destination'] = [np.nanmean(error_delta[:,2])]
    rec_communication['state_value_communicated'] = [np.nanmean(error_state_value[:,0])]
    rec_communication['state_value_at_source'] = [np.nanmean(error_state_value[:,1])]
    rec_communication['state_value_at_destination'] = [np.nanmean(error_state_value[:,2])]
    rec_communication['optimal_RRR_d'] = [selected_n_dimensions]
    rec_communication['lr_performance'] = [max_pred]
    rec_communication['RRR_performance'] = [error_f[best_pred]]
    rec_communication['id'] = [rec]
    rec_communication['n_source'] = [X.shape[1]]
    rec_communication['n_dest'] = [Y.shape[1]]
    communication = pd.concat([communication,rec_communication])


# Dimensionality in high q chosen vs low q chosen

coms_val = pd.DataFrame()
for rec in curated_simultaneous_recordings:
    rec_communication = pd.DataFrame()
    print(rec)
    sim_recording = region_df_grouped[region_df_grouped['hem_id'] ==rec]
    X = sim_recording.loc[sim_recording['region']==source_region, 'binned_spikes_outcome_reward_value_upper'].iloc[0]  # source region
    Y = sim_recording.loc[sim_recording['region']==dest_region, 'binned_spikes_outcome_reward_value_lower'].iloc[0] # dest region
    X =  X[:,:,-80:] # Get rid of pre time + outcome, just use last 2 seconds of ITI
    Y =  Y[:,:,-80:] # Get rid of pre time + outcome just use last 2 seconds of ITI
    X  = halfgaussian_filter1d(X, 1.5, axis=2, output=None,
                      mode="constant", cval=0.0, truncate=4.0)
    Y  = halfgaussian_filter1d(Y, 1.5, axis=2, output=None,
                      mode="constant", cval=0.0, truncate=4.0)
    # Filter as wanted (i.e, 100ms bins with/without zscoring)
    #X = filterneurons(X, zscoring=False, zscoring_params=None, bins = 0.1, bin_size=0.025)
    #Y = filterneurons(Y, zscoring=False, zscoring_params=None, bins = 0.1, bin_size=0.025)
    # Divide in folds
    kf = KFold(n_splits=Ncrossval)
    error = np.zeros([len(np.arange(2,10)), Ncrossval])
    max_pred = np.zeros([Ncrossval])
    for i, indexes in enumerate(kf.split(np.arange(X.shape[0]))):
        train_index , test_index= indexes
        X_train , X_test = X[train_index,:,:], X[test_index,:,:]
        y_train , y_test = Y[train_index,:,:], Y[test_index,:,:]
        #for j,m in enumerate(np.arange(1,np.min([X.shape[1],Y.shape[1]]))):  # Find communication subspace dimension
        for j,m in enumerate(np.arange(2,10)):  # Find communication subspace dimension
            try:
                # Reshape (i.e concatenate trials)
                X_train_f = flatten_trials(X_train)
                y_train_f = flatten_trials(y_train)
                X_test_f = flatten_trials(X_test)
                y_test_f = flatten_trials(y_test)
                n, p = X_train_f.shape
                _, q = y_train_f.shape
                # RRR
                Bols = np.linalg.inv(X_train_f.T @ X_train_f + lam*np.eye(p)) @ (X_train_f.T @ y_train_f)  # Estimate for B with ordinary least squares
                Yestimate = X_train_f @ Bols
                pca = PCA(n_components=m)
                pca.fit(Yestimate)
                V = pca.components_.T
                Brrr = Bols @ V @ V.T
                Yestimate = X_test_f @ Brrr
                Yestimate_lr = X_test_f @ Bols
                error[j,i] = 1 - np.var(Yestimate - y_test_f) / np.var(y_test_f)
            except:
                continue
            #error[j,i] = sps.pearsonr(Yestimate.flatten(), y_test_f.flatten())[0]
        try:
            max_pred[i] = 1 - np.var(Yestimate_lr - y_test_f) / np.var(y_test_f)
        except:
            continue
    error_f = np.nanmean(error, axis=1)
    max_pred = np.nanmean(max_pred) #upper bound? 
    # Find lowest dimension within one SD from peak performance (As in Semedo et al., 2019)
    best_pred = np.argmax(error_f)
    selected_n_dimensions= np.where(error_f>=(error_f[best_pred]-np.std(error[best_pred,:])))[0][0]+2
    rec_communication['optimal_RRR_d'] = [selected_n_dimensions]
    rec_communication['lr_performance'] = [max_pred]
    rec_communication['RRR_performance'] = [error_f[best_pred]]
    rec_communication['id'] = [rec]
    rec_communication['type'] = 'high_q_chosen'
    coms_val = pd.concat([coms_val,rec_communication])
    # Low val
    rec_communication = pd.DataFrame()
    print(rec)
    sim_recording = region_df_grouped[region_df_grouped['hem_id'] ==rec]
    X = sim_recording.loc[sim_recording['region']==source_region, 'binned_spikes_gocue_qchosen_lower'].iloc[0]  # source region
    Y = sim_recording.loc[sim_recording['region']==dest_region, 'binned_spikes_gocue_qchosen_lower'].iloc[0] # dest region
    X =  X[:,:,-80:] # Get rid of pre time + outcome, just use last 2 seconds of ITI
    Y =  Y[:,:,-80:] # Get rid of pre time + outcome just use last 2 seconds of ITI
    X  = halfgaussian_filter1d(X, 1.5, axis=2, output=None,
                      mode="constant", cval=0.0, truncate=4.0)
    Y  = halfgaussian_filter1d(Y, 1.5, axis=2, output=None,
                      mode="constant", cval=0.0, truncate=4.0)
    # Filter as wanted (i.e, 100ms bins with/without zscoring)
    #X = filterneurons(X, zscoring=False, zscoring_params=None, bins = 0.1, bin_size=0.025)
    #Y = filterneurons(Y, zscoring=False, zscoring_params=None, bins = 0.1, bin_size=0.025)
    # Divide in folds
    kf = KFold(n_splits=Ncrossval)
    error = np.zeros([len(np.arange(2,10)), Ncrossval])
    max_pred = np.zeros([Ncrossval])
    for i, indexes in enumerate(kf.split(np.arange(X.shape[0]))):
        train_index , test_index= indexes
        X_train , X_test = X[train_index,:,:], X[test_index,:,:]
        y_train , y_test = Y[train_index,:,:], Y[test_index,:,:]
        #for j,m in enumerate(np.arange(1,np.min([X.shape[1],Y.shape[1]]))):  # Find communication subspace dimension
        for j,m in enumerate(np.arange(2,10)):  # Find communication subspace dimension
            try:
                # Reshape (i.e concatenate trials)
                X_train_f = flatten_trials(X_train)
                y_train_f = flatten_trials(y_train)
                X_test_f = flatten_trials(X_test)
                y_test_f = flatten_trials(y_test)
                n, p = X_train_f.shape
                _, q = y_train_f.shape
                # RRR
                Bols = np.linalg.inv(X_train_f.T @ X_train_f + lam*np.eye(p)) @ (X_train_f.T @ y_train_f)  # Estimate for B with ordinary least squares
                Yestimate = X_train_f @ Bols
                pca = PCA(n_components=m)
                pca.fit(Yestimate)
                V = pca.components_.T
                Brrr = Bols @ V @ V.T
                Yestimate = X_test_f @ Brrr
                Yestimate_lr = X_test_f @ Bols
                error[j,i] = 1 - np.var(Yestimate - y_test_f) / np.var(y_test_f)
            except:
                continue
            #error[j,i] = sps.pearsonr(Yestimate.flatten(), y_test_f.flatten())[0]
        try:
            max_pred[i] = 1 - np.var(Yestimate_lr - y_test_f) / np.var(y_test_f)
        except:
            continue
    error_f = np.nanmean(error, axis=1)
    max_pred = np.nanmean(max_pred) #upper bound? 
    # Find lowest dimension within one SD from peak performance (As in Semedo et al., 2019)
    best_pred = np.argmax(error_f)
    selected_n_dimensions= np.where(error_f>=(error_f[best_pred]-np.std(error[best_pred,:])))[0][0]+2
    rec_communication['optimal_RRR_d'] = [selected_n_dimensions]
    rec_communication['lr_performance'] = [max_pred]
    rec_communication['RRR_performance'] = [error_f[best_pred]]
    rec_communication['id'] = [rec]
    rec_communication['type'] = 'low_q_chosen'
    coms_val = pd.concat([coms_val,rec_communication])

dls_coms_outcome['epoch'] = 'ITI'
dls_coms_cue['epoch'] = 'Trial'
nac_coms_outcome['epoch'] = 'ITI'
nac_coms_cue['epoch'] = 'Trial'
DLS = pd.concat([dls_coms_cue, dls_coms_outcome])
DLS['type'] = 'Dorsal'
NAcc = pd.concat([nac_coms_cue, nac_coms_outcome])
NAcc['type'] = 'Ventral'
RRR = pd.concat([NAcc, DLS])

sns.barplot(data = RRR, x='type', y = 'RRR_performance', hue = 'epoch', errorbar = 'se',
            palette = ['red', 'k'])

RRR.groupby(['type', 'epoch']).mean()
sns.despine()
plt.ylabel('Variance Explained')
plt.title('Optimal Dimensions = 2')



# Communication Ration
RRR['outcome_communicated_ratio']  = RRR['outcome_communicated'] / RRR['outcome_at_source']
RRR['choice_communicated_ratio']  = RRR['choice_communicated'] / RRR['choice_at_source']
RRR['communicated_ratio']  = RRR['choice_communicated'] / RRR['outcome_communicated']



fig, ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.barplot(data = RRR, x='type', y = 'choice_communicated', hue = 'epoch', errorbar = 'se',
            palette = ['red', 'k'])
sns.despine()
plt.ylabel('Communication Ratio')
plt.title('Choice')
plt.sca(ax[1])
sns.barplot(data = RRR, x='type', y = 'outcome_communicated', hue = 'epoch', errorbar = 'se',
            palette = ['red', 'k'])
sns.despine()
plt.ylabel('Accuracy')
plt.title('Outcome')

fig, ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.barplot(data = RRR, x='type', y = 'choice_communicated_ratio', hue = 'epoch', errorbar = 'se',
            palette = ['red', 'k'])
sns.despine()
plt.ylabel('Accuracy')
plt.title('Choice')
plt.sca(ax[1])
sns.barplot(data = RRR, x='type', y = 'outcome_communicated_ratio', hue = 'epoch', errorbar = 'se',
            palette = ['red', 'k'])
sns.despine()
plt.ylabel('Communication Ratio')
plt.title('Outcome')


fig, ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.barplot(data = RRR, x='type', y = 'communicated_ratio', hue = 'epoch', errorbar = 'se',
            palette = ['red', 'k'])
sns.despine()
plt.ylabel('Accuracy')
plt.title('Choice')
plt.sca(ax[1])
sns.barplot(data = RRR, x='type', y = 'communicated_ratio', hue = 'epoch', errorbar = 'se',
            palette = ['red', 'k'])
sns.despine()
plt.ylabel('Communication Ratio')
plt.title('Outcome')