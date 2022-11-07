import os
os.chdir('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
from brainbox.singlecell import calculate_peths
from sklearn.linear_model import Lasso as LR
from sklearn.linear_model import LogisticRegression as LLR
from scipy.stats import pearsonr
#import multiprocess as mp
import sys
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

def z_score_peth(binned_spikes):
    zdata=np.zeros(binned_spikes.shape)
    zdata[:]=np.nan
    u = np.mean(np.mean(binned_spikes, axis=0), axis=1)
    v = np.std(np.std(binned_spikes, axis=0), axis=1)
    for i in np.arange(binned_spikes.shape[1]):
        if  binned_spikes[:,i,:].sum()==0:
            zdata[:,i,:] = 0
        else:
            zdata[:,i,:] = (binned_spikes[:,i,:] - u[i])/v[i]
    return zdata

def run_decoder_for_session(area, alfio, regressed_variable, weights, alignment_times, etype = 'real', 
                            output_folder='/jukebox/witten/Alex/decoder_output', n=None, bin_s=0.1, z_data=True,
                            n_neurons_minimum = 10, pre_time=0.5, post_time=4, smoothing=0):
    hem=[]
    binned_spikes = []
    for probe_id in np.arange(4):
        try:
            selection = np.isin(alfio.probe[probe_id].cluster_metrics,['good','mua'])
            c_selection = 1*(alfio.probe[probe_id].cluster_group_locations==area) * selection
            if np.sum(c_selection)<n_neurons_minimum:
                continue
            cluster_ids = np.where(c_selection==1)[0]
            spike_times = np.copy(alfio.probe[probe_id].spike_times)
            spike_clusters = np.copy(alfio.probe[probe_id].spike_clusters)
            hem.append(alfio.probe[probe_id].cluster_hem[cluster_ids])
            _, xs = calculate_peths(
                    spike_times, spike_clusters, 
                    cluster_ids,  alignment_times,
                    pre_time=pre_time,post_time=post_time, bin_size=bin_s,
                    smoothing=smoothing, return_fr=True)
            binned_spikes.append(xs)      
        except:
            continue
    if len(binned_spikes)==0:
        return print('Not enough neurons in area')
    binned_spikes = np.concatenate(binned_spikes,axis=1)
    binned_spikes = binned_spikes/bin_s
    if z_data==True:
        binned_spikes = z_score_peth(binned_spikes)
    hem = np.concatenate(hem)
    for h in np.unique(hem): # Only analyze ensambles from a single hemisphere, even if from multiple probes, they should be from the same hem
        cluster_selection = binned_spikes[:, np.where(hem==h)[0],:]
        if etype=='real':
            p_summary, mse_summary = run_decoder(cluster_selection, regressed_variable, weights)
            np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_p_summary.npy', p_summary)
            np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_mse_summary.npy', mse_summary)
        else:
            p_summary = np.load(output_folder+'/'+'real'+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_p_summary.npy')
            l_performance  = np.nanmean(np.nanmean(np.nanmean(np.nanmean(p_summary, axis=0), axis=1), axis=1), axis=1)
            l = np.logspace(-5,-0.5,100)[np.argmax(l_performance)]
            p_summary, mse_summary = run_decoder(cluster_selection, regressed_variable, weights, lambdas=np.array([l]))
            np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_p_summary.npy', p_summary)
            np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_mse_summary.npy', mse_summary)


def get_session_sample_weights(df, categories = ['choice','probabilityLeft', 'outcome']):
        df = df.reset_index()
        summary = df.groupby(categories)['stimOn_times'].count().reset_index()
        summary['weight'] = get_sample_weighting(len(summary),summary.iloc[:,-1])
        dfm = df.merge(summary, on=categories).sort_values('index')
        weights = dfm['weight'].to_numpy()
        return weights

def get_sample_weighting(n_classes,samples_per_class):
    w = 1/samples_per_class
    w = w/np.sum(w) * n_classes
    return w

def suffled_Xval(n_trials, k_folds):
    x = np.arange(n_trials)
    # outer_folds
    k = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    outer_traning=[]
    outer_test=[]
    for otrainingi, otesti in k.split(x):
        outer_traning.append(otrainingi)
        outer_test.append(otesti)
    outer_folds=[outer_traning,outer_test]
    # inner_folds
    all_inner_traning=[]
    all_inner_test=[]
    for f in outer_folds[0]: # go over outer traning folds
        inner_traning=[]
        inner_test=[]
        for itrainingi, itesti in k.split(f):
            inner_traning.append(f[itrainingi]) # is already an index, we dont want the index of the index hence f[it..i]
            inner_test.append(f[itesti])
        all_inner_traning.append(inner_traning) # is already an index, we dont want the index of the index hence f[it..i]
        all_inner_test.append(inner_test)        
    inner_folds = [all_inner_traning,all_inner_test]
    return outer_folds, inner_folds

def makeXvalpartitions(trials,number_folds):
    folds = np.floor(trials/number_folds) * np.ones([number_folds,1])
    if trials%number_folds> 0:
        folds[:trials%number_folds] = folds[:trials%number_folds]+1
    fold_indices = [[] for t in np.arange(number_folds)]
    for i in np.arange(number_folds):
        fold_indices[i] = np.arange(folds[i])
        if i>0:
            fold_indices[i]=fold_indices[i] + sum(folds[:i])
    fold_indices = np.array(fold_indices, dtype=object)
    partition_test = [[] for t in np.arange(number_folds)]
    partition_train = [[] for t in np.arange(number_folds)]
    for i in np.arange(number_folds):
        partition_test[i] = [fold_indices[i]]
        if i==0:
            partition_train[i] = fold_indices[np.setdiff1d(np.arange(number_folds-1),[number_folds-1,i,i+1])]
        elif i == number_folds-1:
            partition_train[i] = fold_indices[np.setdiff1d(np.arange(number_folds-1),[i-1,i,0])]
        else:
            partition_train[i] = fold_indices[np.setdiff1d(np.arange(number_folds), np.arange(i-1,i+2))]
    return partition_train, partition_test

def run_decoder(xs, regressed_variable, weights, number_folds=5, decoder = LR, 
                n_neurons_minimum = 10, n_neurons_max = 50, lambdas = None, max_n_combinations=100, xval_type='shuffled'):
    '''
    Params:
    xs : This is the binned firing rates for the population of interest
    regressed_variable : the y
    weights: weights if avaiable (might be None)
    max_n_combinations: Number of combination run for each set of (10,20 etc neurons)
    xval_type: shuffled (ibl style Xval) or gap (conservative cross validation with 50t gap between train and test)
    '''
    # center data
    if decoder == LR:
        regressed_variable = zscore(regressed_variable)
    if xval_type == 'gap':
        folds  = makeXvalpartitions(len(regressed_variable),number_folds)
    elif xval_type == 'shuffled': 
        folds  = suffled_Xval(len(regressed_variable),number_folds)
    # Prepare training sets
    if lambdas is  None:
        lambdas = np.array([0.001,0.01,0.1,1,10])
        if decoder == LLR:
            lambdas = np.array([0.001,0.01,0.1,1,10])
            lambdas = 1/(2*lambdas) #To match alphs of linear regressions 
    if lambdas == 'omit':
        folds  = makeXvalpartitions(len(regressed_variable),number_folds) #shuffled Xval makes no sense if not optimizing lambda
        neuron_samples_n = np.array([n_neurons_minimum, n_neurons_minimum*2, n_neurons_minimum*3, n_neurons_max])
        neuron_samples_n = neuron_samples_n[np.where(neuron_samples_n<=xs.shape[1])] # Ignore samples with higher n that possible neurons
        neuron_combinations = []
        for samples in neuron_samples_n:
            neuron_selections = []
            for i in np.arange(max_n_combinations):
                neuron_selections.append(np.random.choice(np.arange(xs.shape[1]), samples, replace=False))
            neuron_combinations.append(neuron_selections) # neuron_combinations[neuron_samples_n][n_combinations_until_using_every_neurons]
        n_bins = xs.shape[2]
        pearson_summary = np.zeros([number_folds,1, n_bins, len(neuron_combinations), max_n_combinations])
        mse_summary = np.zeros([number_folds,1, n_bins, len(neuron_combinations), max_n_combinations]) 
        # pearson_summary  = [n_folds x n_lambdas x n_timebins x n_neurons_samples x n_combinations]
        pearson_summary[:] = np.nan
        mse_summary[:] = np.nan
        # Start workers and define multiprocessing function
        #pool = mp.Pool(processes=12)    
        for nc, nsample in enumerate(neuron_combinations):
            for s, subsample in enumerate(nsample):
                for f in np.arange(number_folds):
                    training_trials = np.concatenate(folds[0][:][f]).astype(int)
                    testing_trials = np.concatenate(folds[1][:][f]).astype(int)
                    for b in np.arange(n_bins):
                        fit_qc = weighted_decoder(b, regressed_variable=regressed_variable, xs=xs, 
                                                subsample=subsample, training_trials=training_trials, 
                                                testing_trials=testing_trials, 
                                                weights=weights, decoder=decoder)
                        pearson_summary[f, 0, b, nc, s] = fit_qc[0]
                        mse_summary[f, 0, b, nc, s] = fit_qc[1]
    else:
        neuron_samples_n = np.array([n_neurons_minimum, n_neurons_minimum*2, n_neurons_minimum*3, n_neurons_max])
        neuron_samples_n = neuron_samples_n[np.where(neuron_samples_n<=xs.shape[1])] # Ignore samples with higher n that possible neurons
        neuron_combinations = []
        for samples in neuron_samples_n:
            neuron_selections = []
            for i in np.arange(max_n_combinations):
                neuron_selections.append(np.random.choice(np.arange(xs.shape[1]), samples, replace=False))
            neuron_combinations.append(neuron_selections) # neuron_combinations[neuron_samples_n][n_combinations_until_using_every_neurons]
        n_bins = xs.shape[2]
        # Start workers and define multiprocessing function
        #pool = mp.Pool(processes=12)    
        if xval_type == 'gap':
            pearson_summary = np.zeros([number_folds,len(lambdas), n_bins, len(neuron_combinations), max_n_combinations])
            mse_summary = np.zeros([number_folds,len(lambdas), n_bins, len(neuron_combinations), max_n_combinations]) 
            # pearson_summary  = [n_folds x n_lambdas x n_timebins x n_neurons_samples x n_combinations]
            pearson_summary[:] = np.nan
            mse_summary[:] = np.nan
            for nc, nsample in enumerate(neuron_combinations):
                for s, subsample in enumerate(nsample):
                    for f in np.arange(number_folds):
                            training_trials = np.concatenate(folds[0][:][f]).astype(int)
                            testing_trials = np.concatenate(folds[1][:][f]).astype(int)
                            for i_l, l in enumerate(lambdas):
                                for b in np.arange(n_bins):
                                    fit_qc = weighted_decoder(b, regressed_variable=regressed_variable, xs=xs, 
                                                            subsample=subsample, training_trials=training_trials, 
                                                            testing_trials=testing_trials, 
                                                            weights=weights, decoder=decoder, l=l)
                                    pearson_summary[f, i_l, b, nc, s] = fit_qc[0]
                                    mse_summary[f, i_l, b, nc, s] = fit_qc[1]

        if xval_type == 'shuffled':
            pearson_summary = np.zeros([number_folds,len(lambdas), n_bins, len(neuron_combinations), max_n_combinations])
            mse_summary = np.zeros([number_folds,len(lambdas), n_bins, len(neuron_combinations), max_n_combinations]) 
            # pearson_summary  = [n_folds x n_lambdas x n_timebins x n_neurons_samples x n_combinations]
            pearson_summary[:] = np.nan
            mse_summary[:] = np.nan
            for nc, nsample in enumerate(neuron_combinations): # loop over combos of 10, 20 ... neurons
                for f in np.arange(number_folds):
                    outer_training = folds[0][0][f] 
                    outer_test = folds[0][1][f] 
                    inner_training = folds[1][0][f]
                    inner_test = folds[1][1][f]
                    inner_l_performance = np.zeros([len(inner_training),len(lambdas), n_bins, max_n_combinations])
                    inner_l_performance[:] = np.nan
                    # Start of inner XVal
                    for i,inner_f in enumerate(np.arange(len(inner_training))):
                        training_trials = inner_training[inner_f].astype(int)
                        testing_trials = inner_test[inner_f].astype(int)
                        for s, subsample in enumerate(nsample):
                            for i_l, l in enumerate(lambdas):
                                for b in np.arange(n_bins):
                                    ifit_qc = weighted_decoder(b, regressed_variable=regressed_variable, xs=xs, 
                                                            subsample=subsample, training_trials=training_trials, 
                                                            testing_trials=testing_trials, 
                                                            weights=weights, decoder=decoder, l=l)
                                    inner_l_performance[i,i_l, b, s] = ifit_qc[0] #currently pearson r for scoring
                    fold_lambda = np.argmax(np.nanmean(np.nanmean(np.nanmean(inner_l_performance, axis=0),axis=1),axis=1)) # select best lambda
                    # End of inner XVal
                    for s, subsample in enumerate(nsample): # Now trainf and test on outer folds with best lambdas, run through all bins and samples of neurons
                        for b in np.arange(n_bins):
                            ofit_qc = weighted_decoder(b, regressed_variable=regressed_variable, xs=xs, 
                                                    subsample=subsample, training_trials=outer_training.astype(int), 
                                                    testing_trials=outer_test.astype(int), 
                                                    weights=weights, decoder=decoder, l=lambdas[fold_lambda])
                            pearson_summary[f, fold_lambda, b, nc, s] = ofit_qc[0]
                            mse_summary[f, fold_lambda, b, nc, s] = ofit_qc[1]                        
    return pearson_summary, mse_summary

def weighted_decoder(b, regressed_variable=None, xs=None, 
    subsample=None, training_trials=None, 
    testing_trials=None, 
    weights=None, decoder=None, l=None):
    spike_data = xs[:,subsample,b]
    X_train = spike_data[training_trials]
    X_test = spike_data[testing_trials]
    y_train = regressed_variable[training_trials]
    y_test = regressed_variable[testing_trials]
    if decoder == LR:
        if l is None:
            reg = decoder().fit(X_train, y_train) #sample_weight    
        else:
            if weights!=None:
                reg = decoder(alpha=l).fit(X_train, y_train, sample_weight=weights[training_trials]) #sample_weight
            else:
                reg = decoder(alpha=l).fit(X_train, y_train) #sample_weight    
    if decoder == LLR:
        if l is None:
            reg = decoder(class_weight='balanced').fit(X_train, y_train) #sample_weight      
        else:
            if weights!=None:
                reg = decoder(penalty='l1', solver='liblinear', C=l).fit(X_train, y_train, sample_weight=weights[training_trials]) #sample_weight currently not functional for LLR
            else:
                reg = decoder(penalty='l1', solver='liblinear', C=l, class_weight='balanced').fit(X_train, y_train) #sample_weight      
    y_pred = reg.predict(X_test)
    p = pearsonr(y_test, y_pred)[0] #pearson correlation with y-test
    if np.isnan(p)==True:
        p=0
    if decoder == LLR:
        mse = np.mean(1*(y_test==y_pred))  #store accuracy instead
    else:
        mse = r2_score(y_test, y_pred, multioutput='uniform_average')
    return np.array([p, mse])

def reshape_psth_array(binned_spikes):
    # n_trials x n_neurons x n_bins
    spike_data = np.zeros([binned_spikes.iloc[0].shape[0],len(binned_spikes),binned_spikes.iloc[0].shape[1]])
    for n in np.arange(spike_data.shape[1]):
        assert spike_data[:,n,:].shape ==  binned_spikes.iloc[n].shape
        spike_data[:,n,:] = binned_spikes.iloc[n]
    return spike_data

def run_decoder_for_session_residual(c_neural_data, area, alfio, regressed_variable, weights, alignment_time, etype = 'real', 
                            output_folder='/jukebox/witten/Alex/decoder_output', n_neurons_minimum = 10, n=None, decoder = 'lasso', lambdas=None):
    
    if decoder == 'lasso':
        decoder_type = LR
    if decoder == 'logistic':
        decoder_type = LLR     
    # Regressed variable can be a list of arrays in the case of delta q (QR-QL and QL-QR)
    hem = c_neural_data['hem']
    if alignment_time=='response_time':
        binned_spikes = c_neural_data['residuals_outcome']
    if alignment_time=='goCue_time':
        binned_spikes = c_neural_data['residuals_goCue']
    if alignment_time=='action_time':
        binned_spikes = c_neural_data['residuals_choice']
    binned_spikes = reshape_psth_array(binned_spikes) # turn into array
    for h in np.unique(hem): # Only analyze ensambles from a single hemisphere, even if from multiple probes, they should be from the same hem
        cluster_selection = np.copy(binned_spikes[:, np.where(hem==h)[0],:])
        if cluster_selection.shape[1]<n_neurons_minimum:
            return print("Not enough neurons")
        else:
            if len(regressed_variable)==2: #then we are decoding deltaq, need to choose R-L or L-R so that it matches contra-ipsi 
                regressed_variable = regressed_variable[int(h)]
            if etype=='real':
                p_summary, mse_summary = run_decoder(cluster_selection, regressed_variable, weights, n_neurons_minimum=n_neurons_minimum, decoder = decoder_type, lambdas=lambdas)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_p_summary.npy', p_summary)
                np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_mse_summary.npy', mse_summary)
            else:
                p_summary = np.load(output_folder+'/'+'real'+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_p_summary.npy')
                l_performance  = np.nanmean(np.nanmean(np.nanmean(np.nanmean(p_summary, axis=0), axis=1), axis=1), axis=1)
                l = np.array([0.001,0.01,0.1,1,10])[np.argmax(l_performance)]
                p_summary, mse_summary = run_decoder(cluster_selection, regressed_variable, weights, lambdas=np.array([l]), n_neurons_minimum=n_neurons_minimum, decoder = decoder_type)
                np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_p_summary.npy', p_summary)
                np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_'+str(int(h))+'_mse_summary.npy', mse_summary)



