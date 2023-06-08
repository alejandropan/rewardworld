import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
from ephys_alf_summary import *
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso as LR
from sklearn.linear_model import LogisticRegression as LLR
from scipy.stats import pearsonr
import numpy as np
from scipy.stats import zscore
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

ROOT='/jukebox/witten/Alex/Data/Subjects/'
SESSIONS = [
 'dop_47/2022-06-05/001',
 'dop_47/2022-06-06/001',
 'dop_47/2022-06-07/001',
 'dop_47/2022-06-09/003',
 'dop_47/2022-06-10/002',
 'dop_47/2022-06-11/001',
 #'dop_48/2022-06-20/001', Some issue with the video synching
 'dop_48/2022-06-27/002',
 'dop_48/2022-06-28/001',
 'dop_49/2022-06-14/001',
 'dop_49/2022-06-15/001',
 'dop_49/2022-06-16/001',
 'dop_49/2022-06-17/001',
 'dop_49/2022-06-18/002',
 'dop_49/2022-06-19/001',
 'dop_49/2022-06-20/001',
 'dop_49/2022-06-27/003',
 'dop_50/2022-09-12/001',
 'dop_50/2022-09-13/001',
 'dop_50/2022-09-14/003',
 'dop_53/2022-10-02/001',
 'dop_53/2022-10-03/001',
 'dop_53/2022-10-04/001',
 'dop_53/2022-10-05/001',
 'dop_53/2022-10-07/001']


def resample (varss, frame_rate, video_timestamps, bin_size=0.1):
    resampling_factor = bin_size/(1/frame_rate)
    new_array = np.zeros([int(len(varss)/resampling_factor), varss.shape[1]])
    for j in np.arange(int(len(varss)/resampling_factor)):
        new_array[j,:] = np.mean(varss[int(resampling_factor*j):int(resampling_factor*(j+1)),:], 
                                        axis = 0)
    new_video_timestamps = video_timestamps[::int(resampling_factor)]
    return new_array, new_video_timestamps[:len(new_array)]

def reshape_into_bins(varss,epoch_time, n_trials, frame_rate,video_timestamps, bin_size = 0.1): #reshape into -0.5 to 3s from epoch i.e. 35 bins for 100ms bins
    varss, video_timestamps = resample (varss, frame_rate, video_timestamps, bin_size=bin_size)
    frame_trials = np.digitize(video_timestamps,epoch_time)-1
    trial_size = int(3.5/bin_size)
    ordered_data = np.zeros([n_trials,varss.shape[1], trial_size])
    for i in np.arange(n_trials):
        ordered_data[i,:,:] = varss[frame_trials==i,:][:trial_size,:].T
    return ordered_data
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

def run_decoder(video_data, regressed_variable, number_folds=5, decoder = LR,  
                lambdas = None, weights=None):
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
    if lambdas is  None:
        lambdas = np.array([0.0001,0.001,0.01]) #lambdas = np.array([0.0001,0.001,0.01,0.1,1])
        if decoder == LLR:
            lambdas = np.array([0.0001,0.001,0.01]) #lambdas = np.array([0.0001,0.001,0.01,0.1,1])
            lambdas = 1/(2*lambdas) #To match alphs of linear regressions 
    n_bins = video_data.shape[-1]
    pearson_summary = np.zeros([number_folds,len(lambdas), n_bins])
    mse_summary = np.zeros([number_folds,len(lambdas), n_bins]) 
    # pearson_summary  = [n_folds x n_lambdas x n_timebins x n_neurons_samples x n_combinations]
    pearson_summary[:] = np.nan
    mse_summary[:] = np.nan
    folds  = suffled_Xval(len(regressed_variable),number_folds)
    for f in np.arange(number_folds):
        outer_training = folds[0][0][f] 
        outer_test = folds[0][1][f] 
        inner_training = folds[1][0][f]
        inner_test = folds[1][1][f]
        inner_l_performance = np.zeros([len(inner_training),len(lambdas), n_bins])
        inner_l_performance[:] = np.nan
        # Start of inner XVal
        for i,inner_f in enumerate(np.arange(len(inner_training))):
            training_trials = inner_training[inner_f].astype(int)
            testing_trials = inner_test[inner_f].astype(int)
            for i_l, l in enumerate(lambdas):
                for b in np.arange(n_bins):
                    ifit_qc = weighted_decoder(b, regressed_variable=regressed_variable, xs=video_data,
                                            training_trials=training_trials, 
                                            testing_trials=testing_trials, 
                                            weights=weights, decoder=decoder, l=l)
                    inner_l_performance[i,i_l, b] = ifit_qc[0] #currently pearson r for scoring
        fold_lambda = np.argmax(np.nanmean(np.nanmean(inner_l_performance, axis=0),axis=1)) # select best lambda
        # End of inner XVal
        for b in np.arange(n_bins):
            ofit_qc = weighted_decoder(b, regressed_variable=regressed_variable, xs=video_data, 
                                    training_trials=outer_training.astype(int), 
                                    testing_trials=outer_test.astype(int), 
                                    weights=weights, decoder=decoder, l=lambdas[fold_lambda])
            pearson_summary[f, fold_lambda, b] = ofit_qc[0]
            mse_summary[f, fold_lambda, b] = ofit_qc[1]                        
    return pearson_summary, mse_summary

def weighted_decoder(b, regressed_variable=None, xs=None, 
    training_trials=None, 
    testing_trials=None, 
    weights=None, decoder=None, l=None):
    spike_data = xs[:,:,b]
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

def run_decoder_for_session_residual(video_data, area, alfio, regressed_variable, weights, alignment_time, etype = 'real', 
                            output_folder='/jukebox/witten/Alex/decoder_output', n_neurons_minimum = 10, n=None, decoder = 'lasso', lambdas=None):
    if decoder == 'lasso':
        decoder_type = LR
    if decoder == 'logistic':
        decoder_type = LLR     
    # Regressed variable can be a list of arrays in the case of delta q (QR-QL and QL-QR)
    if etype=='real':
        p_summary, mse_summary = run_decoder(video_data, regressed_variable, decoder = decoder_type, lambdas=lambdas)
        np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_0'+'_p_summary.npy', p_summary)
        np.save(output_folder+'/'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_0'+'_mse_summary.npy', mse_summary)
    else:
        p_summary, mse_summary = run_decoder(video_data, regressed_variable, decoder = decoder_type,  lambdas=lambdas)
        np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_0'+'_p_summary.npy', p_summary)
        np.save(output_folder+'/'+str(n)+'_'+str(etype)+'_'+str(area)+'_'+str(alfio.mouse)+'_'+str(alfio.date)+'_0'+'_mse_summary.npy', mse_summary)

##########################
####### Load Data ########
##########################
ses = ROOT+SESSIONS[int(sys.argv[1])]
output_folder = '/jukebox/witten/Alex/decoders_video_results/decoder_output_deltaq_cue_forget'

# Load behavior
alfio = alf(ses, ephys=False)
alfio.mouse = Path(ses).parent.parent.name
alfio.date = Path(ses).parent.name
alfio.ses = Path(ses).name
alfio.path = ses
epoch_time = alfio.goCue_trigger_times-0.5
n_trials = len(epoch_time)


#Load video data
video_data =np.load(ses+'/raw_video_data/_iblrig_rightCamera.raw_proc.npy',allow_pickle=True).item()
video_timestamps = np.load(ses+'/alf/_ibl_rightCamera.times.npy')

if (epoch_time[-1]<video_timestamps[-1]) & (epoch_time[0]>video_timestamps[0]): # QC for videoframes sync
    frame_rate = np.round(1/((video_timestamps[-1] - video_timestamps[0]) / len(video_timestamps)))
    mePC = zscore(video_data['motSVD'][0])
    mPC = zscore(video_data['movSVD'][0])
    # reasmple video reshape variables into n_trials vs bins
    processed_mePC = reshape_into_bins(mePC,epoch_time, n_trials, frame_rate,video_timestamps, bin_size = 0.1)
    processed_mPC = reshape_into_bins(mPC, epoch_time, n_trials, frame_rate,video_timestamps, bin_size = 0.1)

    # Only trials included in analysis
    weights = None

    # Load variable to be decoded and aligment times
    alfio.fQRreward_cue = np.copy(np.roll(alfio.fQRreward,1))
    alfio.fQLreward_cue = np.copy(np.roll(alfio.fQLreward,1))
    alfio.fQRreward_cue[0] = 0
    alfio.fQLreward_cue[0] = 0
    regressed_variable = alfio.fQRreward_cue - alfio.fQLreward_cue

    # Only trials included in analysis
    #weights = get_session_sample_weights(alfio.to_df(), categories = ['choice','probabilityLeft', 'outcome'])
    weights = None

    # Load and run null distributions
    null_sesssions = []
    for i in np.arange(200):
        n_temp =  pd.read_csv('/jukebox/witten/Alex/null_sessions/laser_only/'+str(i)+'.csv')
        n_temp = n_temp.iloc[:, np.where(n_temp.columns=='choice')[0][0]:]
        qr = np.roll(np.copy(n_temp['fQRreward'].to_numpy()),1)
        ql = np.roll(np.copy(n_temp['fQLreward'].to_numpy()),1)
        delta = qr - ql
        delta = delta[:len(regressed_variable[0])]
        null_sesssions.append(delta)
    ##########################
    ## Run decoder (linear) ##
    ##########################
    run_decoder_for_session_residual(processed_mePC, 'motSVD', alfio, regressed_variable, weights, epoch_time, 
                                    etype = 'real', output_folder=output_folder)
    run_decoder_for_session_residual(processed_mPC, 'movSVD', alfio, regressed_variable, weights, epoch_time, 
                                    etype = 'real', output_folder=output_folder)

    #for n, null_ses in enumerate(null_sesssions):
    #    run_decoder_for_session_residual(processed_mePC, 'motSVD', alfio, null_ses, weights, alignment_time, etype = 'null', n=n, output_folder=output_folder, decoder = 'logistic')
    #for n, null_ses in enumerate(null_sesssions):
    #    run_decoder_for_session_residual(processed_mPC, 'movSVD', alfio, null_ses, weights, alignment_time, etype = 'null', n=n, output_folder=output_folder, decoder = 'logistic')




