#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:57:34 2020

@author: alex
"""


import pystan
import numpy as np
import scipy.optimize as so
import pickle
import pandas as pd
import os
import glob
from os import path
import os
os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'

psy = pd.read_pickle('all_behav.pkl')

psy = psy.loc[((psy['ses']>'2020-01-13') & (psy['mouse_name'] == 'dop_4')) | 
              ((psy['ses']>'2020-03-13') & (psy['mouse_name'] != 'dop_9')) |
              ((psy['ses']>'2020-03-13') & (psy['ses']<'2020-03-19') 
               & (psy['mouse_name'] == 'dop_9'))]

# Make laser into int not array
psy.loc[psy['opto.npy']==1, 'opto.npy'] = 1
psy.loc[psy['opto.npy']==0, 'opto.npy'] = 0
psy['opto.npy'] = pd.to_numeric(psy['opto.npy'])


# Prepara data for Stan​
# 1) optomization variables
NS = 10
NA = len(psy['mouse_name'].unique())
NS_all  = np.zeros(NA)
NT_all = np.zeros([NA,NS])
length_info = psy.groupby(['mouse_name','ses']).count()['index'].reset_index()
for i, mouse in enumerate(length_info['mouse_name'].unique()):
    animal = length_info.loc[length_info['mouse_name']==mouse]
    NS_all[i] = len(animal)
    NT_all[i,:len(animal)] = animal['index']
NT = int(max(NT_all.ravel()))# Maximum number of trials per session

# 2) trial variables
r = np.zeros([NA,NS,NT])
c = np.zeros([NA,NS,NT])
l = np.zeros([NA,NS,NT]) 
sc = np.zeros([NA,NS,NT])
for na, mouse in enumerate(length_info['mouse_name'].unique()):
    animal = psy.loc[psy['mouse_name']==mouse]
    for ns, ses in enumerate(animal['ses'].unique()):
        session = animal.loc[animal['ses']==ses]
        r[na, ns, :len(session)] =  session['feedbackType'].map({-1:0, 1:1})
        choices = session['choice']*-1 #.map({1:0, -1:1}) # -1 =  right choice
        choices[choices==-1] = 0
        c[na, ns, :len(session)] = choices
        sc[na, ns, :len(session)] = session['signed_contrasts']
        l[na, ns, :len(session)] = session['opto.npy']
        

standata = {'NA':NA,'NS':NS ,'NT':NT,'NT_all':NT_all.astype(int), 
           'r':r.astype(int), 'c':c.astype(int), 'l':l.astype(int), 'sc':sc ,'NS_all':NS_all.astype(int)}


# Compile the model 
try:
    with open('model.pkl', 'rb') as fhand: 
        sm = pickle.load(fhand) 

except:
	sm = pystan.StanModel('/Users/alex/Documents/PYTHON/rewardworld/behavior_analysis/models/no_stay_stan_model.stan')
	with open('model.pkl', 'wb') as f:pickle.dump(sm, f)
​
# Fit model 
fit = sm.sampling(data=standata,iter=1000,warmup=250,chains=4,control=dict(adapt_delta = 0.99))
​
#import stan_utility
#stan_utility.check_all_diagnostics(fit)
print(fit)
​
print(pystan.check_hmc_diagnostics(fit))
​
# Save data
summary = fit.summary()
summary = pd.DataFrame(summary['summary'], columns=summary['summary_colnames'], index=summary['summary_rownames'])
summary.to_csv(os.path.join('/jukebox/witten/Julia/QLearning/DMS_all/', 'summary_int_pc.csv'))
​
extract = fit.extract()
for k, v in extract.items(): extract[k] = v
with open(os.path.join('/jukebox/witten/Julia/QLearning/DMS_all/' 'StanFit_int_pc.pickle'), 'wb') as fn: cPickle.dump(extract, fn)
​
samplerParams = fit.get_sampler_params()
samplerParams = pd.DataFrame(samplerParams)
summary.to_csv(os.path.join('/jukebox/witten/Julia/QLearning/DMS_all/', 'samplerParams.csv'))
​
with open(os.path.join('/jukebox/witten/Julia/QLearning/DMS_all/' 'fit.pkl'), 'wb') as f:
    pickle.dump({'model' : sm, 'fit' : fit}, f, protocol=-1)
    
with open(os.path.join('/jukebox/witten/Julia/QLearning/DMS_all/' 'fit2.pkl'), 'wb') as f:
    pickle.dump([sm, fit], f, protocol=-1)