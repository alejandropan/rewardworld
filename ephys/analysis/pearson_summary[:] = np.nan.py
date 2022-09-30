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
                for b in np.arange(n_bins):  #iterate by bins
                        spike_data = zscore(xs[:,subsample,b])
                        spike_data[np.where(np.isnan(spike_data))] = 0
                        X_train = spike_data[training_trials]
                        X_test = spike_data[testing_trials]
                        y_train = regressed_variable[training_trials]
                        y_test = regressed_variable[testing_trials]
                        reg = decoder(alpha=0.02).fit(X_train, y_train, sample_weight=weights[training_trials]) #sample_weight
                        y_pred =  reg.predict(X_test)
                        p = pearsonr(y_test, y_pred)[0] #pearson correlation with y-test
                        mse = np.square(np.subtract(y_test,y_pred)).mean()
                        pearson_summary[f, 3, b, nc, s] = p
                        mse_summary[f, 3, b, nc, s] = mse