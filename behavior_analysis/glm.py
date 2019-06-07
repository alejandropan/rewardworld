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

""""






"""
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
import statsmodels.genmod.bayes_mixed_glm as  bayes

##calculate useful variables

#Calculate signed contrast
if not 'signed_contrasts' in psy_df:
    psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df.loc[:,'signed_contrasts'] =  (psy_df['contrastRight'] - psy_df['contrastLeft'])
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
data =  psy_df.loc[ :, ['sex', 'mouse_name', 'feedbackType', 'signed_contrasts', 'choice','ses']]

## Build predictor matrix

#Rewardeded choices: 
data.loc[(data['choice'] == -1) & (data['feedbackType'] == -2) , 'rchoice']  = 0
data.loc[(data['choice'] == -1) & (data['feedbackType'] == -1) , 'rchoice']  = 0
data.loc[(data['choice'] == -1) & (data['feedbackType'] == 1) , 'rchoice']  = -1
data.loc[(data['choice'] == 1) & (data['feedbackType'] == 1) , 'rchoice']  = 1
data.loc[(data['choice'] == 1) & (data['feedbackType'] == -2) , 'rchoice']  = 0
data.loc[(data['choice'] == 1) & (data['feedbackType'] == -1) , 'rchoice']  = 0

#Unrewarded choices: 
data.loc[(data['choice'] == -1) & (data['feedbackType'] == -2) , 'uchoice']  = -1
data.loc[(data['choice'] == -1) & (data['feedbackType'] == -1) , 'uchoice']  = -1
data.loc[(data['choice'] == -1) & (data['feedbackType'] == 1) , 'uchoice']  = 0
data.loc[(data['choice'] == 1) & (data['feedbackType'] == 1) , 'uchoice']  = 0
data.loc[(data['choice'] == 1) & (data['feedbackType'] == -2) , 'uchoice']  = 1
data.loc[(data['choice'] == 1) & (data['feedbackType'] == -1) , 'uchoice']  = 1

#Drop nogos
data = data.drop(data.index[data['feedbackType'] == 0],axis=0)
## Change -1 for 0 in choice 
data.loc[(data['choice'] == -1), 'choice'] = 0

#make Revidence and LEvidence
data.loc[(data['signed_contrasts'] >= 0), 'Revidence'] = data.loc[(data['signed_contrasts'] >= 0), 'signed_contrasts'].abs()
data.loc[(data['signed_contrasts'] <= 0), 'Revidence'] = 0
data.loc[(data['signed_contrasts'] <= 0), 'Levidence'] = data.loc[(data['signed_contrasts'] <= 0), 'signed_contrasts'].abs()
data.loc[(data['signed_contrasts'] >= 0), 'Levidence'] = 0


#previous choices and evidence

no_tback = 5 #no of trials back

start = time.time()
for date in sorted(data['ses'].unique()):
    for i in range(no_tback):
        data.loc[data['ses'] == date,'rchoice%s' %str(i+1)] =  data.loc[data['ses'] == date,'rchoice'].shift(i+1) #no point in 0 shift
        data.loc[data['ses'] == date,'uchoice%s' %str(i+1)] =  data.loc[data['ses'] == date,'uchoice'].shift(i+1) #no point in 0 shift
        data.loc[data['ses'] == date,'Levidence%s' %str(i+1)] =  data.loc[data['ses'] == date,'Levidence'].shift(i+1) #no point in 0 shift
        data.loc[data['ses'] == date,'Revidence%s' %str(i+1)] =  data.loc[data['ses'] == date,'Revidence'].shift(i+1) #no point in 0 shift
end = time.time()
print(end - start)


#Remove first 5 trials from each ses
for date in sorted(data['ses'].unique()):
    data  = data.drop(data.index[data['ses'] == date][0:5] ,axis=0)
    
data = data.reset_index()
#Drop unnecessary elements
data =  data.drop(columns  = ['feedbackType','sex', 'ses','signed_contrasts', 'index'])

model = 'choice' + '~' + 'rchoice + uchoice + Revidence + Levidence + \
 rchoice1 + uchoice1 + Levidence1 + Revidence1 + rchoice2 + uchoice2 + Levidence2 + \
 Revidence2 + rchoice3 + uchoice3 + Levidence3 + Revidence3 + rchoice4 + uchoice4 + \
 Levidence4 + Revidence4 + rchoice5 + uchoice5 + Levidence5 + Revidence5'

## construct our model, with contrast as a variable

##Bayeasian
 
endog  = pd.DataFrame(data['choice'])
exog  = data[[ 'Revidence', 'Levidence', 'rchoice1', 'uchoice1', 'Levidence1', 'Revidence1', 'rchoice2', 'uchoice2', 'Levidence2', 'Revidence2', 'rchoice3', 'uchoice3', 'Levidence3', 'Revidence3', 'rchoice4', 'uchoice4', 'Levidence4', 'Revidence4', 'rchoice5', 'uchoice5', 'Levidence5', 'Revidence5']]
exog_vc =  np.ones((7508, 1))
ident = np.ones(1)
model1 = bayes.BinomialBayesMixedGLM(endog, exog, exog_vc, ident)
result = model1.fit_map()
result.summary()

##Frequentist
logit_model = sm.Logit(endog,exog)
result=logit_model.fit()
print(result.summary2())
#Remove from fit non significant predictors
pval =  result.pvalues
coff = result.fittedvalues()

#cross validate  with sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(exog, np.ravel(endog), test_size=0.3)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))


#Regression stats

def logit2prob(coef):
    
