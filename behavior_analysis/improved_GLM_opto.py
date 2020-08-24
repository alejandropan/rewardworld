#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:47:42 2020

@author: alex
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
# for modelling
import patsy # to build design matrix
import statsmodels.api as sm
from matplotlib import pyplot as plt
import seaborn as sns

# DEFINE THE MODEL
# Assumes right choice  = 1, left choice = -1
def fit_glm(behav, prior_blocks=False, folds=5):
    
    # use patsy to easily build design matrix
    endog, exog = patsy.dmatrices('choice ~ 1 + stimulus_side:C(contrast, Treatment)'
                                      '+ previous_choice:C(previous_outcome)'
                                      '+ previous_opto'
                                      '+ previous_side:C(previous_contrast, Treatment)',
                               data=behav.dropna(subset=['feedbackType', 'choice',
                                  'previous_choice', 'previous_outcome', 
                                  'previous_opto', 'previous_choice']).reset_index(),
                                      return_type='dataframe')

    # remove the one column (with 0 contrast) that has no variance
    if 'stimulus_side:C(contrast, Treatment)[0.0]' in exog.columns:
        exog.drop(columns=['stimulus_side:C(contrast, Treatment)[0.0]'], inplace=True)
    if 'previous_side:C(previous_contrast, Treatment)[0.0]' in exog.columns:
        exog.drop(columns=['previous_side:C(previous_contrast, Treatment)[0.0]' ], inplace=True)

    
    # recode choices for logistic regression
    endog['choice'] = endog['choice'].map({-1:0, 1:1})

    # rename columns
    exog.rename(columns={'Intercept': 'bias',
              'stimulus_side:C(contrast, Treatment)[0.0625]': '6.25',
             'stimulus_side:C(contrast, Treatment)[0.125]': '12.5',
             'stimulus_side:C(contrast, Treatment)[0.25]': '25',
             'previous_choice:C(previous_outcome)[-1.0]': 'unrewarded',
             'previous_choice:C(previous_outcome)[1.0]': 'rewarded',
             'previous_opto': 'After opto',
             'previous_side:C(previous_contrast, Treatment)[0.0625]': 'After 0.0625', 
             'previous_side:C(previous_contrast, Treatment)[0.125]':'After 0.125', 
             'previous_side:C(previous_contrast, Treatment)[0.25]': 'After 0.25', 
             },
    
             inplace=True)

    # NOW FIT THIS WITH STATSMODELS - ignore NaN choices
    logit_model = sm.Logit(endog.iloc[1:], exog[1:])
    res = logit_model.fit_regularized(disp=False) # run silently

    # what do we want to keep?
    params = pd.DataFrame(res.params).T
    params['pseudo_rsq'] = res.prsquared # https://www.statsmodels.org/stable/generated/statsmodels.discrete.discrete_model.LogitResults.prsquared.html?highlight=pseudo

    # ===================================== #
    # ADD MODEL ACCURACY - cross-validate

    kf = KFold(n_splits=folds, shuffle=True)
    acc = np.array([])
    for train, test in kf.split(endog):
        X_train, X_test, y_train, y_test = exog.loc[train], exog.loc[test], \
                                           endog.loc[train], endog.loc[test]
        # fit again
        logit_model = sm.Logit(y_train, X_train)
        res = logit_model.fit_regularized(disp=False)  # run silently

        # compute the accuracy on held-out data [from Luigi]:
        # suppose you are predicting Pr(Left), let's call it p,
        # the % match is p if the actual choice is left, or 1-p if the actual choice is right
        # if you were to simulate it, in the end you would get these numbers
        y_test['pred'] = res.predict(X_test)
        y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
        acc = np.append(acc, y_test['pred'].mean())

    # average prediction accuracy over the K folds
    params['accuracy'] = np.mean(acc)

    return params  # wide df



def psy_for_glm(psy, choice_left=-1):
    #psy = pd.read_pickle('all_behav.pkl')
    mice = [ 'dop_8', 'dop_9', 'dop_11', 'dop_4']
    psy = psy.loc[np.isin(psy['mouse_name'], mice)]
    
    if choice_left!=0:
        psy['choice'] = psy['choice']*(choice_left*-1)
    else:
        psy['choice'] = psy['choice'].map({0:-1,1:1})
    # Drop no-go
    psy = psy.loc[psy['choice']!=0]
    
    # First calculate previous choice, opto and previous outcome, previous diffficulty
    
    psy['previous_choice'] = np.nan
    psy['previous_outcome'] = np.nan
    psy['previous_opto'] = np.nan
    psy['previous_stim'] = np.nan
    
    #Set side of stimulus
    psy['stimulus_side'] = np.sign(psy.signed_contrasts)
    psy.loc[(psy['stimulus_side']==0) & (psy['feedbackType']==1) &
            (psy['choice']==1) ,'stimulus_side'] = 1
    psy.loc[(psy['stimulus_side']==0) & (psy['feedbackType']!=1) &
            (psy['choice']==-1) ,'stimulus_side'] = 1
    psy.loc[(psy['stimulus_side']==0) & (psy['feedbackType']!=1) &
            (psy['choice']==1) ,'stimulus_side'] = -1
    psy.loc[(psy['stimulus_side']==0) & (psy['feedbackType']==1) &
            (psy['choice']==-1) ,'stimulus_side'] = -1
    
    for name in psy['mouse_name'].unique():
        psy.loc[psy['mouse_name']==name, 'previous_choice'] = \
           psy.loc[psy['mouse_name']==name, 'choice'].shift(1)
        psy.loc[psy['mouse_name']==name, 'previous_outcome'] = \
           psy.loc[psy['mouse_name']==name, 'feedbackType'].shift(1)
        psy.loc[psy['mouse_name']==name, 'previous_opto'] = \
           psy.loc[psy['mouse_name']==name, 'opto.npy'].shift(1)
        psy.loc[psy['mouse_name']==name, 'previous_stim'] = \
           psy.loc[psy['mouse_name']==name, 'signed_contrasts'].shift(1)
        psy.loc[psy['mouse_name']==name, 'previous_side'] = \
           psy.loc[psy['mouse_name']==name, 'stimulus_side'].shift(1)
    
    
    # Make previous opto a number not an array
    psy.loc[psy['previous_opto']==1, 'previous_opto'] = 1
    psy.loc[psy['previous_opto']==0, 'previous_opto'] = 0
    psy['previous_opto'] = pd.to_numeric(psy['previous_opto'])
    psy['previous_opto'] = psy['previous_opto']*psy['previous_choice']
    
    
    
    psy['contrast'] = np.abs(psy.signed_contrasts)
    psy['previous_contrast'] =np.abs(psy.previous_stim)
    psy['block'] = psy['opto_probability_left'].map({-1:0,0:-1, 1:1})
    
    return psy

mice = psy['mouse_name'].unique() # mice = np.array(['dop_8', 'dop_9', 'dop_11', 'dop_4'])
params = pd.DataFrame()
for mouse in mice:
    behav = psy.loc[psy['mouse_name']==mouse]
    m_params = fit_glm(behav, prior_blocks=False, folds=5)
    m_params['mouse_name'] = mouse
    m_params['virus'] = behav['virus'].to_numpy()[0] #All trials will have the same virus so this works 
    params =  pd.concat([params, m_params])
params['mouse_name'] = mice
# Plot results
predictors = params.columns[~np.isin(params.columns,['accuracy', 'pseudo_rsq'])]
# Melt to use column names as variables
params_plot = params[predictors].melt(id_vars=['mouse_name','virus'])
fig, ax =  plt.subplots(2, figsize=(5,10))
plt.sca(ax[0])
sns.swarmplot(data=params_plot, x='variable', y='value',
              order=['25','12.5','6.25',
                     'After 0.25','After 0.125',
                     'After 0.0625', 'rewarded',
                     'unrewarded', 'After opto', 'bias'] , hue='virus')
plt.ylabel('Coefficient')
plt.xlabel('Variable')
plt.xticks(rotation=90)
ax[0].axhline(0, ls='--', color = 'k')
plt.sca(ax[1])
sns.swarmplot(data=params, x='virus', y='accuracy')
plt.ylim(0.5,1)
plt.tight_layout()
sns.despine()



