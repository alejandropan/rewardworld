# Imports
import os
os.chdir('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/')
from sklearn import svm
import sklearn.utils as utils
from ephys_alf_summary import *
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.stats import sem
from dict_region import *
import multiprocess as mp

# Functions currently not in use 

def balance_training_test_set(alfio_sample, train_size = 0.8):
    '''
    Params:
    alfio_sample (object) :  Subsample from matched_subsample
    train_size (float):  Size of the training set
    Return:
    train_set, test_set: indexes of the trials in alfio (general dataframe, i.e actual trial numbers 
    for test and train set)
    '''
    n_trials_train = int(train_size*len(alfio_sample))
    combined_weight = alfio_sample['combined'].value_counts(normalize=True)
    alfio_sample['combined_weight'] = alfio_sample['combined'].apply(lambda x: combined_weight[x])
    train_set = alfio_sample.sample(n_trials_train, weights=alfio_sample['combined_weight']).index
    test_set = np.setxor1d(alfio_sample.index, train_set)
    test_set_labels = alfio_sample.loc[np.isin(alfio_sample.index,test_set),'opto_block'].to_numpy()
    return train_set, test_set, test_set_labels

def matched_subsample(alfio, categories):
    '''
    Params:
    alfio (object) :  This is an object with all the session behavioral and ephys data
    alfio_sample: Balanced sample, where minimum category is fully sample and rest undersampled
    '''
    # Look for smallest category
    alfio_df = alfio.to_df()
    alfio_df = alfio_df.dropna(subset = categories)# Remove nan trials
    min_category_count = alfio_df.groupby(categories).count().iloc[:,0].min()
    combined=[]
    for i in np.arange(alfio_df.shape[0]):
        combined.append(str(alfio_df[categories].iloc[i,:].tolist()))
    alfio_df['combined']=combined
    alfio_sample = pd.DataFrame()
    for type in alfio_df['combined'].unique():
        alfio_sample = \
            pd.concat([alfio_sample,
                alfio_df.loc[alfio_df['combined']==type].sample(min_category_count)])
    return alfio_sample




# Functions


def labeler(alfio, categories):
    # Look for smallest category
    alfio_df = alfio.to_df()
    combined=[]
    for i in np.arange(alfio_df.shape[0]):
        combined.append(str(alfio_df[categories].iloc[i,:].tolist()))
    alfio.label=combined
    class_weight = utils.class_weight.compute_class_weight('balanced', classes = np.unique(alfio.label), y = alfio.label)
    class_dict = dict(zip(np.unique(alfio.label), class_weight))
    weights = [class_dict[l] for l in alfio.label]
    alfio.weights  = weights
    return alfio

def run_decoder(alfio,test_set_labels, train_trials=None, test_trials=None, 
                probe_id=None, cluster_ids=None, target_category=None, 
                alignment_times=None, fr_bin = 0.05, 
                decoder_type='svm',pre_time=1,post_time=1, smoothing=0.025):
    '''
    Params:
    alfio (object) :  This is an object with all the session behavioral and ephys data
    train_trials(list of int): index of train trials
    test_trials(list of int): index of test trials
    cluster_ids(np.array):  unit to run the decoder on
    target_category(str): y for decoder (e.g. choice)
    fr_bin (float) :  size of the bins for the decoder
    alignment_times (np.array): times of the variable of interest (e.g. Go Cue)
    decoder_type(str): family of decoder support by scikitlearn 
    Return: 
    acc (np.array) : accuracy from pre_time to post_time in fr_bins
    '''
    # Prepare training sets
    _, xs = calculate_peths(
            alfio.probe[probe_id].spike_times, alfio.probe[probe_id].spike_clusters, 
            cluster_ids,  alignment_times,
            pre_time=pre_time,post_time=post_time, bin_size=fr_bin,
            smoothing=smoothing, return_fr=True)
    y_train = alfio.to_df()[target_category][train_trials].to_numpy()
    y_test = alfio.to_df()[target_category][test_trials].to_numpy()
    acc = []
    acc_laser = []
    acc_water = []
    for i in np.arange(xs.shape[2]): 
        X_train = xs[train_trials,:,i]
        X_test = xs[test_trials,:,i]            
        classifier = svm.SVC(kernel='linear', class_weight='balanced') # Linear Kernel
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        acc.append(metrics.accuracy_score(y_test, y_pred))
        acc_laser.append(metrics.accuracy_score(y_test[test_set_labels==1], y_pred[test_set_labels==1]))
        acc_water.append(metrics.accuracy_score(y_test[test_set_labels==0], y_pred[test_set_labels==0]))
    return np.array([acc]),np.array([acc_water]),np.array([acc_laser])

def run_decoding_experiment(alfio,categories,probe_number,cluster_ids, target_category, aligment_timepoint):
    acc = []
    acc_water = []
    acc_laser = []
    for m in np.arange(10):
        alfio_sample = matched_subsample(alfio, categories)
        for i in np.arange(10):
            train_trials, test_trials, test_set_labels = balance_training_test_set(alfio_sample)
            ac, ac_water, ac_laser = run_decoder(alfio, test_set_labels, train_trials=train_trials, test_trials=test_trials, 
                                        probe_id=probe_number, cluster_ids=cluster_ids, target_category=target_category, 
                                        alignment_times=aligment_timepoint)       
            acc.append(ac)
            acc_water.append(ac_water)
            acc_laser.append(ac_laser)
    return acc, acc_water, acc_laser

def main_decoder(ses, categories, target_category, regions_of_interest, reg):
    alfio = alf(ses, ephys=True)
    probe_no = len(sorted(glob(ses +'/alf/*[0-9]*/')))
    aligment_timepoint  = np.array(alfio.goCue_trigger_times)
    rec_decoder_pool = pd.DataFrame()
    print(ses)
    for probe_id in np.arange(probe_no):
        rec_decoder = pd.DataFrame()
        cluster_ids_uncurated = np.array(np.where(np.isin(alfio.probe[probe_id].cluster_locations,regions_of_interest))[0])
        good_clusters = np.where(alfio.probe[probe_id].cluster_metrics=='good')
        cluster_ids = np.intersect1d(cluster_ids_uncurated,good_clusters)
        if len(cluster_ids)<10:
            continue
        acc, acc_water, acc_laser = run_decoding_experiment(alfio,categories,probe_id,cluster_ids, target_category, aligment_timepoint)
        m_acc = np.mean(np.vstack(acc),axis=0)
        m_acc_water = np.mean(np.vstack(acc_water),axis=0)
        m_acc_laser = np.mean(np.vstack(acc_laser),axis=0)
        rec_decoder['Region'] = [regions_of_interest]
        rec_decoder['Region Pool'] = reg
        rec_decoder['Mouse'] = ses[35:41]
        rec_decoder['Date'] = ses[42:52]
        rec_decoder['Accuracy'] = [m_acc]
        rec_decoder['Accuracy Water'] = [m_acc_water]
        rec_decoder['Accuracy Laser'] = [m_acc_laser]
        rec_decoder['Path'] = ses
        rec_decoder['Probe_id'] = probe_id
        rec_decoder_pool = pd.concat([rec_decoder_pool,rec_decoder])
    return rec_decoder_pool

def plot_decoding_accuracy(acc, pre=-1, post=1, bin_size=0.05, color='k'):
    plt.plot(np.mean(np.vstack(acc),axis=0), color=color)
    plt.fill_between(np.arange(len(np.mean(np.vstack(acc),axis=0))), 
                        np.mean(np.vstack(acc),axis=0) -  sem(np.vstack(acc),axis=0), 
                        np.mean(np.vstack(acc),axis=0) + sem(np.vstack(acc),axis=0),
                        color=color, alpha=0.2)
    plt.xticks(np.arange(0,(post-pre)/bin_size,4),np.round(np.arange(pre,post,4*bin_size),2))
    plt.xlabel('Time')
    plt.ylabel('Accuracy')

#########################
#########################

LIST_OF_SESSIONS_ALEX = \
['/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/001',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-19/002',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-14/001',
'/Volumes/witten/Alex/Data/Subjects/dop_21/2021-06-20/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-17/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-19/001',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-23/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-02/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-04/001',
'/Volumes/witten/Alex/Data/Subjects/dop_30/2021-11-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-07/002',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-09/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-11/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_38/2021-12-15/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-05-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-30/001',
'/Volumes/witten/Alex/Data/Subjects/dop_16/2021-04-26/001']

# 0 Settings
target_category = 'choice'
categories = ['choice', 'outcome', 'opto_block'] # matching categories

# 1 Run
choice_decoder = pd.DataFrame()
for reg in dict_region['Summary_Region'].unique():
    regions_of_interest = dict_region.loc[dict_region['Summary_Region']==reg, 'Allen'].to_list()
    info = list(zip(LIST_OF_SESSIONS_ALEX, 
                    len(LIST_OF_SESSIONS_ALEX)*[categories],
                    len(LIST_OF_SESSIONS_ALEX)*[target_category],
                    len(LIST_OF_SESSIONS_ALEX)*[regions_of_interest],
                    len(LIST_OF_SESSIONS_ALEX)*[reg]))
    with mp.Pool() as p:
      region_decoder = p.starmap(main_decoder,info)
    region_sum = pd.DataFrame()
    for reg_pd in region_decoder:
        region_sum = pd.concat([region_sum, reg_pd])
    choice_decoder = pd.concat([choice_decoder, region_sum])

# 2 Plot
fig, ax = plt.subplots(int(np.ceil(len(dict_region['Summary_Region'].unique())/4)),4, sharey=True)
for i, reg in enumerate(dict_region['Summary_Region'].unique()):
    plt.sca(ax[int(i/4),i%4])
    plt.title(reg)
    region = choice_decoder.loc[choice_decoder['Region Pool']==reg]
    if len(region)==0:
        continue
    plot_decoding_accuracy(region['Accuracy'].to_list(), color='black')
    plot_decoding_accuracy(region['Accuracy Water'].to_list(), color='dodgerblue')
    plot_decoding_accuracy(region['Accuracy Laser'].to_list(), color='orange')
plt.tight_layout(h_pad=-2,w_pad=-0.5)
sns.despine()




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
