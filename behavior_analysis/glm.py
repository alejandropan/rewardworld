#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 16:55:17 2019


Predictors:  Rewarded choice and unrewarded choice
These two predictors include  correct choice and reward i.e

R (Right_Choice = True, Correct_choice = True, Reward = True) = 1
R (Right_Choice = True, Correct_choice = False, Reward = True) = 1 - This never happens
U(Right_Choice = True, Correct_choice = False, Reward = False) = 0
U(Right_Choice = True, Correct_choice = True, Reward = False) = 0

R (Right_Choice = False, Correct_choice = True, Reward = True) =  -1
R (Right_Choice = False, Correct_choice = False, Reward = True) =  -1 - This never happens
U(Right_Choice = False, Correct_choice = False, Reward = False) = 0 /
U(Right_Choice = False, Correct_choice = True, Reward = False) =  0 /


@author: ibladmin
"""

import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
import sklearn as sk
from math import sqrt
from sklearn import preprocessing
import time


##calculate useful variables

#Calculate signed contrast
if not 'signed_contrasts' in psy_df:
    psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df.loc[:,'signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])*100
    unique_signed_contrasts  = sorted(psy_df['signed_contrasts'].unique())

##Add choice -1 etc

##

start = time.time()
for date in sorted(psy_df['ses'].unique()):
    for i in range(psy_df.loc[psy_df['ses'] == date,'choice'].shape[0]):
        psy_df.loc[psy_df['ses'] == date,'choice%s' %str(i+1)] =  psy_df.loc[psy_df['ses'] == date,'choice'].shift(i+1) #no point in 0 shift

end = time.time()
print(end - start)
      
    
#Add sex if not present
if not 'sex' in psy_df:
    psy_df.loc[:,'sex'] = np.empty([psy_df.shape[0],1])
    mice  = sorted(psy_df['mouse_name'].unique())
    for mouse in mice:
        sex = input('Sex of animal ')
        psy_df.loc[ psy_df['mouse_name'] == mouse, ['sex']]  = sex

#make separate datafrme 
data =  psy_df.loc[ :, ['sex', 'feedbackType', 'signed_contrasts', 'choice','ses']]

## Build predictor matrix
data.loc[(data['choice'] == -1) & (data['feedbackType'] == -2) , ['choice']]  = 0
data.loc[(data['choice'] == -1) & (data['feedbackType'] == -1) , ['choice']]  = 0
data.loc[(data['choice'] == -1) & (data['feedbackType'] == 1) , ['choice']]  = -1
data.loc[(data['choice'] == 1) & (data['feedbackType'] == 1) , ['choice']]  = 1
data.loc[(data['choice'] == 1) & (data['feedbackType'] == -2) , ['choice']]  = 0
data.loc[(data['choice'] == 1) & (data['feedbackType'] == -1) , ['choice']]  = 0


start = time.time()
for date in sorted(data['ses'].unique()):
    for i in range(data.loc[data['ses'] == date,'choice'].shape[0]):
        data.loc[data['ses'] == date,'choice%s' %str(i+1)] =  data.loc[data['ses'] == date,'choice'].shift(i+1) #no point in 0 shift

end = time.time()
print(end - start)
      



data =  data.drop(columns  = ['feedbackType'])

## construct our model, with contrast as a variable
md = smf.mixedlm("choice ~ reward", data, groups =  data['signed_contrasts']) # data['sex'],
mdf = md.fit()
print(mdf.summary())
