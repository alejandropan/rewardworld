# What information is DLS receiving

from ephys_alf_summary import *
from scipy.stats import zscore
from sklearn.model_selection import KFold
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from scipy.io import loadmat
import scipy.stats as sps
import scipy.ndimage
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from psths_per_region import *

# Parameters
ROIS = np.array(['MO', 'DLS'])
MIN_N = 20
BIN_SIZE = 0.1 # in seconds
N_COMBINATIONS = 1
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

# Example session to work with is i=22, hem=1, ses_id=22
# Eventually make this into a for loop to go thorugh every session for now , example session
# bin[i-1]<spike time<bin[i]
results = pd.DataFrame()
for ses_id in usable.i.unique():
    ses_of_interest = sessions[ses_id]
    hems = usable.loc[usable['i']==ses_id, 'hem'].to_numpy()
    source_matrix = []
    destination_matrix = []
    bins = np.arange(ses_of_interest.start_time[10],ses_of_interest.start_time[-1], BIN_SIZE)
    for j in np.arange(len(ses_of_interest.probe[:])):
        select = ses_of_interest.probe[j].cluster_goodmuaselection
        locations = pd.Series(ses_of_interest.probe[j].cluster_locations[np.isin(ses_of_interest.probe[j].cluster_hem,hems)]).map(group_dict)
        source_clusters = np.intersect1d(np.where(locations==ROIS[0]), select)
        destination_clusters = np.intersect1d(np.where(locations==ROIS[1]), select)
        for s_c in source_clusters:
            spt = ses_of_interest.probe[j].spike_times[ses_of_interest.probe[j].spike_clusters==s_c]
            # trial 10 to trial -1
            spt = spt[np.where(spt>=bins[0])]
            spt = spt[np.where(spt<=bins[-1])]
            binned = np.bincount(np.digitize(spt,bins),minlength=len(bins))/BIN_SIZE #divide by bin_size for firing rate
            source_matrix.append(binned)
        for d_c in destination_clusters:
            spt = ses_of_interest.probe[j].spike_times[ses_of_interest.probe[j].spike_clusters==d_c]
            # trial 10 to trial -1
            spt = spt[np.where(spt>=bins[0])]
            spt = spt[np.where(spt<=bins[-1])]
            binned = np.bincount(np.digitize(spt,bins),minlength=len(bins))/BIN_SIZE #divide by bin_size for firing rate
            destination_matrix.append(binned)
    source_matrix = np.stack(source_matrix)
    destination_matrix = np.stack(destination_matrix)

    # Look at information encoded
    Ncrossval = 10
    lam = 0

    # for every session in simultaneous
    X_total = source_matrix.T  # source region
    Y_total = destination_matrix.T # dest region
    X_total = zscore(X_total)
    Y_total = zscore(Y_total)

    total_fold_results = pd.DataFrame()
    for selection_fold in np.arange(N_COMBINATIONS):
        s_fold_results = pd.DataFrame()
        s_fold_results['s_fold'] = [selection_fold]
        s_fold_results['mouse'] = usable.loc[usable['i']==ses_id,'mouse'].to_numpy()
        s_fold_results['date'] = usable.loc[usable['i']==ses_id,'date'].to_numpy()
        s_fold_results['n_source'] = usable.loc[usable['i']==ses_id,'n_source'].to_numpy()
        s_fold_results['n_destination'] = usable.loc[usable['i']==ses_id,'n_destination'].to_numpy()
        s_fold_results['source'] = [ROIS[0]]
        s_fold_results['destination'] = [ROIS[1]]

        # make selection of N neurons
        Xselection =  np.random.choice(np.arange(X_total.shape[1]), MIN_N, replace=False)
        Yselection = np.random.choice(np.arange(Y_total.shape[1]), MIN_N, replace=False)
        if N_COMBINATIONS == 1:
            X = X_total
            Y = Y_total
        else:
            X = X_total[:,Xselection]
            Y = Y_total[:,Yselection]
        # Divide in folds
        kf = KFold(n_splits=Ncrossval)
        error = np.zeros([len(np.arange(2,10)), Ncrossval])
        for i, indexes in enumerate(kf.split(np.arange(X.shape[0]))):
            train_index , test_index= indexes
            X_train , X_test = X[train_index,:], X[test_index,:]
            y_train , y_test = Y[train_index,:], Y[test_index,:]
            #for j,m in enumerate(np.arange(1,np.min([X.shape[1],Y.shape[1]]))):  # Find communication subspace dimension
            for j,m in enumerate(np.arange(2,10)):  # Find communication subspace dimension
                try:
                    # Reshape (i.e concatenate trials)
                    n, p = X_train.shape
                    _, q = y_train.shape
                    # RRR
                    Bols = np.linalg.inv(X_train.T @ X_train + lam*np.eye(p)) @ (X_train.T @ y_train)  # Estimate for B with ordinary least squares
                    Yestimate = X_train @ Bols
                    pca = PCA(n_components=m)
                    pca.fit(Yestimate)
                    V = pca.components_.T
                    Brrr = Bols @ V @ V.T
                    Yestimate = X_test @ Brrr
                    #error[j,i]  = r2_score(y_test, Yestimate)
                    error[j,i] = 1 - np.var(Yestimate - y_test) / np.var(y_test)
                except:
                    continue
        error_f = np.mean(error, axis=1)
        # Find lowest dimension within one SD from peak performance (As in Semedo et al., 2019)
        best_pred = np.argmax(error_f)
        selected_n_dimensions= np.where(error_f>=(error_f[best_pred]-np.std(error_f)))[0][0]+2 # +2 because that is the min of dimension tried

        s_fold_results['dimensionality'] = [selected_n_dimensions]
        s_fold_results['communication'] = [error_f[best_pred]]

        # Now that we now n_dimension project data and decode variables of interest
        # first assing trial number for each bin
        bin_trial = np.zeros(X.shape[0])
        for t in np.arange(10,len(ses_of_interest.start_time)-1):
            t_start = ses_of_interest.start_time[t]
            t_end = ses_of_interest.start_time[t+1]
            bin_trial[np.where(np.logical_and(bins>=t_start, bins<=t_end))]=t
        bin_trial = bin_trial.astype(int)

        # First make variable vectors
        bin_outcome = ses_of_interest.outcome[bin_trial]
        state = ses_of_interest.fQRreward.copy() + ses_of_interest.fQLreward.copy()
        bin_state = state[bin_trial]
        if hems==0: #hems==0 means that right is contra
            choice = 1*(ses_of_interest.choice==1)
            bin_choice =choice[bin_trial]
            delta_q = ses_of_interest.fQRreward.copy() - ses_of_interest.fQLreward.copy()
        else:
            choice = 1*(ses_of_interest.choice==-1)
            delta_q = ses_of_interest.fQLreward.copy() - ses_of_interest.fQRreward.copy() 
        bin_choice = choice[bin_trial]
        bin_deltaq = delta_q[bin_trial]
        q_chosen = np.array([ses_of_interest.fQLreward, ses_of_interest.fQRreward])[choice,np.arange(len(choice))]
        bin_qchosen = q_chosen[bin_trial]

        #Only outcome 
        outimes = np.where(np.diff(bin_trial)!=0)[0] - 44
        error_choice = np.zeros([Ncrossval,3])
        error_outcome = np.zeros([Ncrossval,3])
        error_qchosen = np.zeros([Ncrossval,3])
        error_delta = np.zeros([Ncrossval,3])
        error_state_value = np.zeros([Ncrossval,3])
        for i, indexes in enumerate(kf.split(outimes)):
            train_index , test_index= indexes
            train_index = outimes[train_index]
            test_index = outimes[test_index]
            X_train , X_test = X[train_index,:], X[test_index,:]
            y_train , y_test = Y[train_index,:], Y[test_index,:]
            # Reshape (i.e concatenate trials)
            n, p = X_train.shape
            _, q = y_train.shape
            # RRR
            Bols = np.linalg.inv(X_train.T @ X_train + lam*np.eye(p)) @ (X_train.T @ y_train)  # Estimate for B with ordinary least squares
            Yestimate = X_train @ Bols
            pca = PCA(n_components=selected_n_dimensions)
            pca.fit(Yestimate)
            V = pca.components_.T
            Brrr = Bols @ V @ V.T
            Yestimate_train = X_train @ Brrr
            Yestimate_test = X_test @ Brrr
            # Run on subspace, source and destination # class_weight = None, class_weight = 'balanced'
            model_subspace_choice = LogisticRegression(class_weight = 'balanced',C=1).fit(Yestimate_train, bin_choice[train_index])
            error_choice[i,0] =  np.sum(model_subspace_choice.predict(Yestimate_test)==bin_choice[test_index])/len(Yestimate_test)
            model_subspace_choice = LogisticRegression(class_weight = 'balanced',C=1).fit(X_train, bin_choice[train_index])
            error_choice[i,1] =  np.sum(model_subspace_choice.predict(X_test)==bin_choice[test_index])/len(X_test)
            model_subspace_choice = LogisticRegression(class_weight = 'balanced',C=1).fit(y_train, bin_choice[train_index])
            error_choice[i,2] = np.sum(model_subspace_choice.predict(y_test)==bin_choice[test_index])/len(y_test)

            model_subspace_outcome = LogisticRegression(class_weight = 'balanced',C=1).fit(Yestimate_train, bin_outcome[train_index])
            error_outcome[i,0] =  np.sum(model_subspace_outcome.predict(Yestimate_test)==bin_outcome[test_index])/len(Yestimate_test)
            model_subspace_outcome = LogisticRegression(class_weight = 'balanced',C=1).fit(X_train, bin_outcome[train_index])
            error_outcome[i,1] =  np.sum(model_subspace_outcome.predict(X_test)==bin_outcome[test_index])/len(X_test)
            model_subspace_outcome = LogisticRegression(class_weight = 'balanced',C=1).fit(y_train, bin_outcome[train_index])
            error_outcome[i,2] = np.sum(model_subspace_outcome.predict(y_test)==bin_outcome[test_index])/len(y_test)

        s_fold_results['choice_communicated'] = [error_choice.mean(axis=0)[0]]
        s_fold_results['outcome_communicated'] = [error_outcome.mean(axis=0)[0]]
        s_fold_results['choice_source'] = [error_choice.mean(axis=0)[1]]
        s_fold_results['outcome_source'] = [error_outcome.mean(axis=0)[1]]
        s_fold_results['choice_dest'] = [error_choice.mean(axis=0)[2]]
        s_fold_results['outcome_dest'] = [error_outcome.mean(axis=0)[2]]

        #Only rewarded choices
        outimes = np.where(np.diff(bin_trial)!=0)[0] - 44
        bin_rewarded =  np.where(bin_outcome==1)
        outcome_rewarded = np.intersect1d(outimes,bin_rewarded)
        error_choice = np.zeros([Ncrossval,3])
        error_outcome = np.zeros([Ncrossval,3])
        error_qchosen = np.zeros([Ncrossval,3])
        error_delta = np.zeros([Ncrossval,3])
        error_state_value = np.zeros([Ncrossval,3])
        for i, indexes in enumerate(kf.split(outcome_rewarded)):
            train_index , test_index= indexes
            train_index = outimes[train_index]
            test_index = outimes[test_index]
            X_train , X_test = X[train_index,:], X[test_index,:]
            y_train , y_test = Y[train_index,:], Y[test_index,:]
            # Reshape (i.e concatenate trials)
            n, p = X_train.shape
            _, q = y_train.shape
            # RRR
            Bols = np.linalg.inv(X_train.T @ X_train + lam*np.eye(p)) @ (X_train.T @ y_train)  # Estimate for B with ordinary least squares
            Yestimate = X_train @ Bols
            pca = PCA(n_components=selected_n_dimensions)
            pca.fit(Yestimate)
            V = pca.components_.T
            Brrr = Bols @ V @ V.T
            Yestimate_train = X_train @ Brrr
            Yestimate_test = X_test @ Brrr
            # Run on subspace, source and destination # class_weight = None, class_weight = 'balanced'
            model_subspace_choice = LogisticRegression(class_weight = 'balanced',C=1).fit(Yestimate_train, bin_choice[train_index])
            error_choice[i,0] =  np.sum(model_subspace_choice.predict(Yestimate_test)==bin_choice[test_index])/len(Yestimate_test)
            model_subspace_choice = LogisticRegression(class_weight = 'balanced',C=1).fit(X_train, bin_choice[train_index])
            error_choice[i,1] =  np.sum(model_subspace_choice.predict(X_test)==bin_choice[test_index])/len(X_test)
            model_subspace_choice = LogisticRegression(class_weight = 'balanced',C=1).fit(y_train, bin_choice[train_index])
            error_choice[i,2] = np.sum(model_subspace_choice.predict(y_test)==bin_choice[test_index])/len(y_test)

            model_subspace_qchosen = LinearRegression().fit(Yestimate_train, bin_qchosen[train_index])
            error_qchosen[i,0] =  pearsonr(model_subspace_qchosen.predict(Yestimate_test), bin_qchosen[test_index])[0]
            model_subspace_qchosen = LinearRegression().fit(X_train, bin_qchosen[train_index])
            error_qchosen[i,1] =  pearsonr(model_subspace_qchosen.predict(X_test), bin_qchosen[test_index])[0]
            model_subspace_qchosen = LinearRegression().fit(y_train, bin_qchosen[train_index])
            error_qchosen[i,2] =  pearsonr(model_subspace_qchosen.predict(y_test), bin_qchosen[test_index])[0]

            model_state_value = LinearRegression().fit(Yestimate_train, bin_state[train_index])
            error_state_value[i,0] =  pearsonr(model_state_value.predict(Yestimate_test), bin_qchosen[test_index])[0]
            model_state_value = LinearRegression().fit(X_train, bin_state[train_index])
            error_state_value[i,1] =  pearsonr(model_state_value.predict(X_test), bin_qchosen[test_index])[0]
            model_state_value = LinearRegression().fit(y_train, bin_state[train_index])
            error_state_value[i,2] =  pearsonr(model_state_value.predict(y_test), bin_qchosen[test_index])[0]

        s_fold_results['choice_rewarded_communicated'] = [error_choice.mean(axis=0)[0]]
        s_fold_results['qchosen_rewarded_communicated'] = [error_qchosen.mean(axis=0)[0]]
        s_fold_results['state_rewarded_communicated'] = [error_state_value.mean(axis=0)[0]]
        s_fold_results['choice_rewarded_source'] = [error_choice.mean(axis=0)[1]]
        s_fold_results['qchosen_rewarded_source'] = [error_qchosen.mean(axis=0)[1]]
        s_fold_results['state_rewarded_source'] = [error_state_value.mean(axis=0)[1]]
        s_fold_results['choice_rewarded_dest'] = [error_choice.mean(axis=0)[2]]
        s_fold_results['qchosen_rewarded_dest'] = [error_qchosen.mean(axis=0)[2]]
        s_fold_results['state_rewarded_dest'] = [error_state_value.mean(axis=0)[2]]

        #Only contra choices
        outimes = np.where(np.diff(bin_trial)!=0)[0] - 44
        bin_contra =  np.where(bin_choice==1)
        outcome_contra = np.intersect1d(outimes,bin_contra)
        error_outcome = np.zeros([Ncrossval,3])
        error_qchosen = np.zeros([Ncrossval,3])
        error_delta = np.zeros([Ncrossval,3])
        error_state_value = np.zeros([Ncrossval,3])
        for i, indexes in enumerate(kf.split(outcome_contra)):
            train_index , test_index= indexes
            train_index = outimes[train_index]
            test_index = outimes[test_index]
            X_train , X_test = X[train_index,:], X[test_index,:]
            y_train , y_test = Y[train_index,:], Y[test_index,:]
            # Reshape (i.e concatenate trials)
            n, p = X_train.shape
            _, q = y_train.shape
            # RRR
            Bols = np.linalg.inv(X_train.T @ X_train + lam*np.eye(p)) @ (X_train.T @ y_train)  # Estimate for B with ordinary least squares
            Yestimate = X_train @ Bols
            pca = PCA(n_components=selected_n_dimensions)
            pca.fit(Yestimate)
            V = pca.components_.T
            Brrr = Bols @ V @ V.T
            Yestimate_train = X_train @ Brrr
            Yestimate_test = X_test @ Brrr
            # Run on subspace, source and destination # class_weight = None, class_weight = 'balanced'
            model_subspace_outcome = LogisticRegression(class_weight = 'balanced',C=1).fit(Yestimate_train, bin_outcome[train_index])
            error_outcome[i,0] =  np.sum(model_subspace_outcome.predict(Yestimate_test)==bin_outcome[test_index])/len(Yestimate_test)
            model_subspace_outcome = LogisticRegression(class_weight = 'balanced',C=1).fit(X_train, bin_outcome[train_index])
            error_outcome[i,1] =  np.sum(model_subspace_outcome.predict(X_test)==bin_outcome[test_index])/len(X_test)
            model_subspace_outcome = LogisticRegression(class_weight = 'balanced',C=1).fit(y_train, bin_outcome[train_index])
            error_outcome[i,2] = np.sum(model_subspace_outcome.predict(y_test)==bin_outcome[test_index])/len(y_test)

            model_subspace_delta = LinearRegression().fit(Yestimate_train, bin_deltaq[train_index])
            error_delta[i,0] =  pearsonr(model_subspace_delta.predict(Yestimate_test), bin_qchosen[test_index])[0]
            model_subspace_delta = LinearRegression().fit(X_train, bin_deltaq[train_index])
            error_delta[i,1] =  pearsonr(model_subspace_delta.predict(X_test), bin_qchosen[test_index])[0]
            model_subspace_delta = LinearRegression().fit(y_train, bin_deltaq[train_index])
            error_delta[i,2] =  pearsonr(model_subspace_delta.predict(y_test), bin_qchosen[test_index])[0]

        s_fold_results['outcome_contra_communicated'] = [error_outcome.mean(axis=0)[0]]
        s_fold_results['delta_contra_communicated'] = [error_delta.mean(axis=0)[0]]
        s_fold_results['outcome_contra_source'] = [error_outcome.mean(axis=0)[1]]
        s_fold_results['delta_contra_source'] = [error_delta.mean(axis=0)[1]]
        s_fold_results['outcome_contra_dest'] = [error_outcome.mean(axis=0)[2]]
        s_fold_results['delta_contra_dest'] = [error_delta.mean(axis=0)[2]]

        # High vs low_value only look at rewarded trials
        #outimes = np.where(np.diff(bin_trial)!=0)[0] - 44
        bin_rewarded =  np.where(bin_outcome==1)
        #outcome_rewarded = np.intersect1d(outimes,bin_rewarded)
        outcome_rewarded =  bin_rewarded
        outcome_rewarded_high_value = np.intersect1d(np.where(bin_qchosen>=np.quantile(q_chosen,0.66)), outcome_rewarded)
        outcome_rewarded_low_value = np.intersect1d(np.where(bin_qchosen<=np.quantile(q_chosen,0.66)), outcome_rewarded)
        error_high_value = np.zeros([len(np.arange(2,10)), Ncrossval])
        error_low_value = np.zeros([len(np.arange(2,10)), Ncrossval])

        for i, indexes in enumerate(kf.split(outcome_rewarded_high_value)):
            train_index , test_index= indexes
            X_train , X_test = X[train_index,:], X[test_index,:]
            y_train , y_test = Y[train_index,:], Y[test_index,:]
            #for j,m in enumerate(np.arange(1,np.min([X.shape[1],Y.shape[1]]))):  # Find communication subspace dimension
            for j,m in enumerate(np.arange(2,10)):  # Find communication subspace dimension
                try:
                    # Reshape (i.e concatenate trials)
                    n, p = X_train.shape
                    _, q = y_train.shape
                    # RRR
                    Bols = np.linalg.inv(X_train.T @ X_train + lam*np.eye(p)) @ (X_train.T @ y_train)  # Estimate for B with ordinary least squares
                    Yestimate = X_train @ Bols
                    pca = PCA(n_components=m)
                    pca.fit(Yestimate)
                    V = pca.components_.T
                    Brrr = Bols @ V @ V.T
                    Yestimate = X_test @ Brrr
                    #error[j,i]  = r2_score(y_test, Yestimate)
                    error_high_value[j,i] = 1 - (np.var(Yestimate - y_test) / np.var(y_test))
                except:
                    continue
        error_error_high_value_mean = np.mean(error_high_value, axis=1)
        # Find lowest dimension within one SD from peak performance (As in Semedo et al., 2019)
        best_pred_high_value = np.argmax(error_error_high_value_mean)
        selected_n_dimensions_high_value= np.where(error_error_high_value_mean>=(error_error_high_value_mean[best_pred_high_value]-np.std(error_error_high_value_mean)))[0][0]+2 # +2 because that is the min of dimension tried
        high_value_pred = error_error_high_value_mean[selected_n_dimensions_high_value-2]

        # Now for low value
        for i, indexes in enumerate(kf.split(outcome_rewarded_low_value)):
            train_index , test_index= indexes
            X_train , X_test = X[train_index,:], X[test_index,:]
            y_train , y_test = Y[train_index,:], Y[test_index,:]
            #for j,m in enumerate(np.arange(1,np.min([X.shape[1],Y.shape[1]]))):  # Find communication subspace dimension
            for j,m in enumerate(np.arange(2,10)):  # Find communication subspace dimension
                try:
                    # Reshape (i.e concatenate trials)
                    n, p = X_train.shape
                    _, q = y_train.shape
                    # RRR
                    Bols = np.linalg.inv(X_train.T @ X_train + lam*np.eye(p)) @ (X_train.T @ y_train)  # Estimate for B with ordinary least squares
                    Yestimate = X_train @ Bols
                    pca = PCA(n_components=m)
                    pca.fit(Yestimate)
                    V = pca.components_.T
                    Brrr = Bols @ V @ V.T
                    Yestimate = X_test @ Brrr
                    #error[j,i]  = r2_score(y_test, Yestimate)
                    error_low_value[j,i] = 1 - np.var(Yestimate - y_test) / np.var(y_test)
                except:
                    continue
        error_error_low_value_mean = np.mean(error_low_value, axis=1)
        # Find lowest dimension within one SD from peak performance (As in Semedo et al., 2019)
        best_pred_low_value = np.argmax(error_error_low_value_mean)
        selected_n_dimensions_low_value= np.where(error_error_low_value_mean>=(error_error_low_value_mean[best_pred_low_value]-np.std(error_error_low_value_mean)))[0][0]+2 # +2 because that is the min of dimension tried
        low_value_pred = error_error_low_value_mean[selected_n_dimensions_low_value-2]

        s_fold_results['dimensionality_high_value_rewarded'] = [selected_n_dimensions_high_value]
        s_fold_results['dimensionality_low_value_rewarded'] = [selected_n_dimensions_low_value]
        s_fold_results['communication_high_value_rewarded'] = [high_value_pred]
        s_fold_results['communication_low_value_rewarded'] = [low_value_pred] 
        total_fold_results=pd.concat([total_fold_results, s_fold_results])
    results = pd.concat([results, total_fold_results])

results['id'] = results['mouse'] + results['date']
results_summary = results.groupby(['id']).mean().reset_index()
results_summary['choice_source'] -  results_summary['choice_dest'] 

######## Plotting ########
results_dorsal.to_pickle('results_dorsal.pkl')
results_ventral.to_pickle('results_ventral.pkl')