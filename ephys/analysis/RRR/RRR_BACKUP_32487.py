# RRR from https://www.biorxiv.org/content/10.1101/302208v3.full.pdf
# 
# 

<<<<<<< HEAD
plt.plot(np.arange(1,7), NAcc_RRR, color='b')
plt.scatter(np.arange(1,7), NAcc_RRR, color='b')
plt.plot(np.arange(1,7), DLS_RRR, color='r')
plt.scatter(np.arange(1,7), DLS_RRR, color='r')
plt.ylim(0,0.05)
plt.xlabel('Predicitive Dimensions')
plt.ylabel('Performance(r2)')
red_patch = mpatches.Patch(color='red', label='OFC->DLS')
blue_patch = mpatches.Patch(color='blue', label='OFC->NAcc')
plt.legend(handles=[red_patch,blue_patch])
sns.despine()
'''
from itertools import permutations
=======
>>>>>>> 6f212170eb5d631222ff703c9098f20e8151c3cc
import sys
sys.path.insert(0,'/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
from brainbox.singlecell import calculate_peths
from sklearn.linear_model import Lasso as LR
from scipy.stats import zscore
import multiprocess as mp
from ephys_alf_summary import alf
from pathlib import Path
import pandas as pd
import numpy as np
import random
from chord import Chord


def qc(x_data_p,trials_edges = 10):
    fr_qc = np.where(np.mean(np.mean(x_data_p,axis=0),axis=1)>0.1)[0]
    presence_qc = []
    for i in fr_qc:
        linearized = x_data_p[:,i,:].reshape(x_data_p.shape[0]*x_data_p.shape[2])
        start=np.where(linearized>0)[0][0]
        end=np.where(linearized>0)[0][-1]
        if (start<trials_edges*x_data_p.shape[2]) and (end>((x_data_p.shape[0]-trials_edges)*x_data_p.shape[2])):
            presence_qc.append(i)
    presence_qc = np.array(presence_qc)
    return presence_qc


def run_RRR(alfio, x_area, y_area, lambdas=None, ranks=None, criterion=['good','mua']):
    # Center the data 
    for probe_id in np.arange(4):
        try:
            x_clusters = np.where(alfio.probe[probe_id].cluster_group_locations==x_area)[0]
            y_clusters  = np.where(alfio.probe[probe_id].cluster_group_locations==y_area)[0]
            x_good_clusters = np.where(np.isin(alfio.probe[probe_id].cluster_metrics,criterion))[0]
            y_good_clusters = np.where(np.isin(alfio.probe[probe_id].cluster_metrics,criterion))[0]     
            x_clusters = np.intersect1d(x_clusters,x_good_clusters)
            y_clusters = np.intersect1d(y_clusters,y_good_clusters)
            x_hem = alfio.probe[probe_id].cluster_hem[x_clusters]
            y_hem = alfio.probe[probe_id].cluster_hem[y_clusters]
            x_probe = np.ones(len(x_clusters))* probe_id
            y_probe = np.ones(len(y_clusters))* probe_id
            if probe_id==0:
                x_idx = x_clusters
                y_idx = y_clusters
                x_hem_idx = x_hem
                y_hem_idx = y_hem
                x_probe_idx = x_probe    
                y_probe_idx = y_probe              
            else:
                x_idx = np.concatenate([x_idx, x_clusters])
                y_idx = np.concatenate([y_idx, y_clusters])
                x_hem_idx = np.concatenate([x_hem_idx, x_hem])
                y_hem_idx = np.concatenate([y_hem_idx, y_hem])
                x_probe_idx = np.concatenate([x_probe_idx, x_probe])
                y_probe_idx = np.concatenate([y_probe_idx, y_probe])
            print(probe_id)
        except:
            continue
    
    if len(x_idx)<alfio.n_neurons_minimum or len(y_idx)<alfio.n_neurons_minimum: # Check if enought cells in either area
        return
    if ~np.any(np.isin(x_hem_idx,np.unique(y_hem_idx))):  # Check if there are cells on the same hemisphere
        return
    rrr_result=[]
    full_model_result=[]
    for h in np.unique(x_hem_idx)[np.isin(np.unique(x_hem_idx),np.unique(y_hem_idx))]: # in case we have connected recordings in each hemisphere, iterate through them indepednently
        x_select = x_idx[x_hem_idx==h]
        y_select = y_idx[y_hem_idx==h]
        if len(x_select)<alfio.n_neurons_minimum or len(y_select)<alfio.n_neurons_minimum:
            continue
        x_probe_select = x_probe_idx[x_hem_idx==h]
        y_probe_select = y_probe_idx[y_hem_idx==h]
        for i, p in enumerate(np.unique(np.concatenate([x_probe_select,y_probe_select]))):
            p=int(p)
            x_ids = x_select[x_probe_select==p]
            y_ids = y_select[y_probe_select==p]
            spike_times = alfio.probe[p].spike_times
            spike_clusters = alfio.probe[p].spike_clusters 
            _, x_data_p = calculate_peths(
                    spike_times, spike_clusters, 
                    x_ids,  alfio.alignment_times,
                    pre_time=alfio.pre_time,post_time=alfio.post_time, bin_size=alfio.bin_size,
                    smoothing=alfio.smoothing, return_fr=False)
            _, y_data_p = calculate_peths(
                    spike_times, spike_clusters, 
                    y_ids,  alfio.alignment_times,
                    pre_time=alfio.pre_time,post_time=alfio.post_time, bin_size=alfio.bin_size,
                    smoothing=alfio.smoothing, return_fr=False)

            x_data_p = x_data_p/alfio.bin_size
            y_data_p = y_data_p/alfio.bin_size
            
            if i==0:
                x_data = x_data_p[:,qc(x_data_p),:]
                y_data = y_data_p[:,qc(y_data_p),:]
            else:
                x_data = np.concatenate([x_data,x_data_p],axis=1)
                y_data = np.concatenate([y_data,y_data_p],axis=1)

        # Balance data set
        n_x = x_data.shape[1]
        n_y = y_data.shape[1]
        sample_from = np.argmax([n_x,n_y])
        n_samples = 10
        if lambdas is  None:
            lambdas = [0.0001,0.001,0.01,0.1,1]
        if ranks is  None:
            ranks = np.arange(1,8)
        prediction = np.zeros([len(lambdas), len(ranks), n_samples])
        full_prediction = np.zeros([len(lambdas), n_samples])
        for s in np.arange(n_samples):
            if sample_from==0:
                x_data_sample = x_data[:,random.sample(range(0,n_x),n_y),:].copy()
                y_data_sample = y_data.copy()
            else:
                x_data_sample = x_data.copy()
                y_data_sample = y_data[:,random.sample(range(0,n_y),n_x),:].copy()
            # Reshape data
            x_data_sample = x_data_sample.reshape(x_data_sample.shape[0]*x_data_sample.shape[2], x_data_sample.shape[1])
            y_data_sample = y_data_sample.reshape(y_data_sample.shape[0]*y_data_sample.shape[2], y_data_sample.shape[1])
            for l, lambdau in enumerate(lambdas):
                full_r2 =cv_full_model(x_data_sample, y_data_sample, lambdau=lambdau, cv_folds=10)
                full_prediction[l,s] = np.mean(full_r2)
                for r, rank in enumerate(ranks):
                    r2 = cv_rrr(x_data_sample, y_data_sample, rank=rank, lambdau=lambdau, cv_folds=10)
                    prediction[l,r,s] = np.mean(r2)
        prediction_ms = np.mean(prediction, axis=2)
        lambdas_select = np.argmax(prediction_ms, axis=0)
        r2s = []
        for i in np.arange(len(lambdas_select)):
            r2s.append(prediction_ms[lambdas_select[i],i])
        rrr_result.append(r2s)
        full_prediction_ms = np.mean(full_prediction, axis=1)
        full_lambdas_select = np.argmax(full_prediction_ms)
        fullr2s = full_prediction_ms[full_lambdas_select]
        full_model_result.append(fullr2s)

    return rrr_result, full_model_result

def cv_full_model(X, Y, lambdau=1, cv_folds=10):
    r2 = np.zeros(cv_folds)
    zX = zscore(X)
    zY = zscore(Y)
    for cvfold in np.arange(cv_folds):
        n = X.shape[0]
        indtest  = np.arange(cvfold*int(n/cv_folds), (cvfold+1)*int(n/cv_folds))
        indtrain = np.setdiff1d(np.arange(n), indtest)
        Xtrain = zX[indtrain,:].copy()
        Ytrain =  zY[indtrain,:].copy()
        Xtest  =  zX[indtest,:].copy()
        Ytest  =  zY[indtest,:].copy()
        w = OLS_alt(Xtrain, Ytrain, lambdau=lambdau)
        r2[cvfold] = 1 - (np.sum((Ytest - (Xtest @ w) )**2) / np.sum((Ytest-np.mean(Ytest))**2))
    return r2

def cv_rrr(X, Y, rank=2, lambdau=1, cv_folds=10):
    r2 = np.zeros(cv_folds)
    zX = zscore(X)
    zY = zscore(Y)
    for cvfold in np.arange(cv_folds):
        n = X.shape[0]
        indtest  = np.arange(cvfold*int(n/cv_folds), (cvfold+1)*int(n/cv_folds))
        indtrain = np.setdiff1d(np.arange(n), indtest)
        Xtrain = zX[indtrain,:].copy()
        Ytrain =  zY[indtrain,:].copy()
        Xtest  =  zX[indtest,:].copy()
        Ytest  =  zY[indtest,:].copy()
        w,v= rrr(Xtrain, Ytrain, rank=rank, lambdau=lambdau)
        r2[cvfold] = 1 - (np.sum((Ytest - Xtest @ w @ v.T)**2) / np.sum((Ytest-np.mean(Ytest))**2))
    return r2

def OLS_alt(X, Y,lambdau=1):
    B = np.linalg.inv(X.T @ X + lambdau*np.diag(np.ones(X.shape[1]))) @ X.T @ Y
    return B

def OLS(X, Y,lambdau=1):
    U,s,V = np.linalg.svd(X, full_matrices=False)
    B = V.T @ np.diag(s/(s**2 + lambdau*X.shape[0])) @ U.T @ Y
    return B

def rrr(X, Y, rank=2, lambdau=1):
    U,s,V = np.linalg.svd(X, full_matrices=False)
    B = V.T @ np.diag(s/(s**2 + lambdau*X.shape[0])) @ U.T @ Y # OLS svd solution
    U,s,V = np.linalg.svd(X@B, full_matrices=False)
    w = B @ V.T[:,:rank]
    v = V.T[:,:rank]
    # This section is to standardize the signs, the results stay the same, is just to standardize whether w or v should have - signs
    pos = np.argmax(np.abs(v), axis=0) 
    flips = np.sign(v[pos, range(v.shape[1])])
    v = v * flips
    w = w * flips
    return (w,v)

def rrr_alt(X, Y, rank=2, lambdau=1):
    B =  OLS(X, Y,lambdau=1) # OLS svd solution
    U,s,V = np.linalg.svd(X@B, full_matrices=False)
    w = B @ V.T[:,:rank]
    v = V.T[:,:rank]
    # This section is to standardize the signs, the results stay the same, is just to standardize whether w or v should have - signs
    pos = np.argmax(np.abs(v), axis=0) 
    flips = np.sign(v[pos, range(v.shape[1])])
    v = v * flips
    w = w * flips
    return (w,v)

##########################
####### Parameters #######
##########################
SESSIONS = ['/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-16/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',  
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-19/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-27/003', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-20/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-10/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001']

alignment_time = 'stimOn_times'
x_area = 'OFC'
y_area = 'NAcc'
results=[]
results_full=[]
for ses in SESSIONS:
    ##########################
    ####### Load Data ########
    ##########################
    print(ses)
    alfio = alf(ses, ephys=True)
    print(len(alfio.choice))
    alfio.mouse = Path(ses).parent.parent.name
    alfio.date = Path(ses).parent.name
    alfio.ses = Path(ses).name
    alfio.path = ses
    alfio.alignment_times = getattr(alfio, alignment_time)
    alfio.n_neurons_minimum = 10
    alfio.pre_time = 0
    alfio.post_time  = 4
    alfio.smoothing=0
    alfio.bin_size=0.1
    alfio.output_folder = '/jukebox/witten/Alex/decoder_output'
    alfio.temp_folder = '/jukebox/witten/Alex/decoder_wd'
    r = run_RRR(alfio, x_area, y_area, lambdas=None, ranks=None)
    results.append(r)

<<<<<<< HEAD

# Plot paired connectivity map

criterion=['good','mua']
n_neurons_minimum = 20

pooled_region_info = pd.DataFrame()
for ses in SESSIONS:
    print(ses)
    alfio = alf(ses, ephys=True)
    region_info = pd.DataFrame()
    for hemisphere in np.array([0,1]):
        regions = pd.DataFrame()
        for probe_id in np.arange(len(alfio.probe.probes)):
                    unique_regions = alfio.probe[probe_id].cluster_group_locations[np.where(
                        np.isin(alfio.probe[probe_id].cluster_metrics,criterion) & 
                        (alfio.probe[probe_id].cluster_hem==hemisphere))[0]].value_counts()
                    unique_regions = unique_regions[unique_regions>=n_neurons_minimum]
                    regions = pd.concat([regions,unique_regions])
        regions = regions.reset_index().groupby('index').sum().reset_index()
        regions['hemisphere'] = hemisphere
        region_info=pd.concat([region_info, regions])
    region_info['mouse'] = Path(ses).parent.parent.name
    region_info['date'] = Path(ses).parent.name
    region_info['ses'] = Path(ses).name
    region_info['id'] = region_info['mouse']+region_info['date']+region_info['ses']+ region_info['hemisphere'].astype(str)
    pooled_region_info = pd.concat([pooled_region_info,region_info])

# Plot connectivity map
chord_data = pooled_region_info.copy()
selected_regions = np.array(['OFC', 'NAcc', 'PFC', 'DMS', 'VPS', 'VP', 'SNr','Olfactory', 'DLS', 'GPe'])
summary = np.zeros([len(selected_regions),len(selected_regions)])
chord_data = chord_data.loc[np.isin(chord_data['index'],selected_regions)]
for id in chord_data.id.unique():
    s_chord_data_r = chord_data.loc[chord_data['id']==id,'index']
    idx = [np.where(selected_regions==r)[0][0] for r in s_chord_data_r]
    lidx = np.array(list(permutations(idx,2)))
    for l in lidx:        
        summary[l[0],l[1]]+=1

chord_diagram(summary, names=selected_regions, rotate_names=True, cmap='Dark2')
plt.show()

=======
>>>>>>> 6f212170eb5d631222ff703c9098f20e8151c3cc
