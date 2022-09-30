#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 23:08:38 2020

@author: alex
"""

import numpy as np
import pandas as pd
import seaborn as sns
import patsy
from sklearn.model_selection import KFold
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sys
from ibllib.io.extractors.training_wheel import extract_all as extract_all_wheel



def plot_all(behav, params, acc, ses, *args, save=False):
    fig, ax = plt.subplots(2,2, figsize = (10,10))
    plt.sca(ax[0,0])
    plot_choice_summary(behav)
    plt.sca(ax[0,1])
    plot_stay_prob(behav)
    plt.sca(ax[1,0])
    plot_block_evolution(behav)
    plt.sca(ax[1,1])
    plot_GLM (params, acc)
    plt.tight_layout()
    if save == True:
        plt.savefig(ses + '/summary_10.png')

def plot_block_evolution(behav):
    # Plot block evolution
    sns.lineplot(data=behav, x='trial_within_block',
                 y = 1*(behav['choice']>0),
                 hue = behav['Probability of reward on left'], errorbar='se',
                 err_style="bars", palette = 'colorblind')
    plt.xlim(-5,10)
    plt.xlabel('Trial within block')
    plt.ylabel('Fraction of right choices')
    plt.xticks(np.arange(-5,10))
    plt.vlines(0,0,1,linestyles='dashed')
    sns.despine()

def plot_stay_prob(behav):
    # Plot probability of staying
    sns.barplot(x=behav['previous_outcome_1'],  y=behav['choice']==behav['previous_choice_1'],
                palette = {0:'r', 1:'g'}, errorbar='se')
    plt.ylim(0,1)
    plt.ylabel('Fraction of repeated choices')
    plt.xticks([0,1],['Previous Unrewarded', 'Rrevious Rewarded'])
    sns.despine()

def plot_choice_summary(behav):
    # Plot fraction of left choices per block
    sns.barplot(data=behav, x='probabilityLeft',
                y = 1*(behav['choice']*-1>0), errorbar='se')
    plt.ylabel('Fraction Left Choices')
    plt.ylim(0,1)
    sns.despine()

def plot_GLM(params, acc):
    # Plot GLM coefficients
    try:
        sns.pointplot(data = params, x = 'trials_back',
                     y = 'coefficient', hue = 'type',
                     palette = {'rewarded':'r', 'unrewarded': 'b',
                            'bias' : 'k', 'previous_laser_1' : 'g'})
        plt.hlines(0,0,10,linestyles='dashed')
        plt.errorbar(np.array([0]), params.loc[params['type']=='bias',
                                               'coefficient'],
                     yerr= params.loc[params['type']=='bias',
                                               'ci_95'][0], color='k')
        plt.errorbar(np.array([11]), params.loc[params['type']=='previous_laser_1',
                                               'coefficient'],
                     yerr= params.loc[params['type']=='previous_laser_1',
                                               'ci_95'], color='g')
        plt.errorbar(np.array([1,2,3,4,5,6,7,8,9,10]), params.loc[params['type']=='rewarded',
                                               'coefficient'],
                     yerr= params.loc[params['type']=='rewarded',
                                               'ci_95'], color='r')
        plt.errorbar(np.array([1,2,3,4,5,6,7,8,9,10]), params.loc[params['type']=='unrewarded',
                                               'coefficient'],
                     yerr= params.loc[params['type']=='unrewarded',
                                               'ci_95'], color='b')
    except:
        sns.pointplot(data = params, x = 'trials_back',
                     y = 'coefficient', hue = 'type',
                     palette = {'rewarded':'r', 'unrewarded': 'b',
                            'bias' : 'k'})
        plt.hlines(0,0,10,linestyles='dashed')
        plt.errorbar(np.array([0]), params.loc[params['type']=='bias',
                                               'coefficient'],
                     yerr= params.loc[params['type']=='bias',
                                               'ci_95'][0], color='k')
        plt.errorbar(np.array([1,2,3,4,5,6,7,8,9,10]), params.loc[params['type']=='rewarded',
                                               'coefficient'],
                     yerr= params.loc[params['type']=='rewarded',
                                               'ci_95'], color='r')
        plt.errorbar(np.array([1,2,3,4,5,6,7,8,9,10]), params.loc[params['type']=='unrewarded',
                                               'coefficient'],
                     yerr= params.loc[params['type']=='unrewarded',
                                               'ci_95'], color='b')
    plt.annotate('Accuracy:' + str(np.round(acc,2)), xy=[0,-0.2])
    # Statistical annotation
    for coef in params['index']:
        pvalue = params.loc[params['index'] == coef, 'pvalues']
        xy = params.loc[params['index'] == coef,
                    ['trials_back', 'coefficient']].to_numpy() + [0,0.05]
        if pvalue.to_numpy()[0] < 0.05:
            plt.annotate(num_star(pvalue.to_numpy()[0]),
                             xy= xy[0] ,
                             fontsize=20)
    sns.despine()

def fit_GLM(data, model=False):
    '''

    Parameters
    ----------
    behav : pandas dataframe
        dataframe with choice and outcome data, up to 5 trials back
        choice is signed, feedback = 1 for reward and 0 for error.

    Returns
    -------
    params : GLM coefficients
    acc: average crossvalidated accuracy

    '''
    #Remove no go trials and make choices in unit interval
    behav = data.copy()
    behav['choice']= behav['choice'].map({-1:0, 0:np.nan, 1:1})
    if np.isnan(behav['laser'].to_numpy()[0]) == True:
            behav = behav.drop(['laser', 'previous_laser_1'], axis=1)
            try:
                behav = behav[['choice', 'previous_choice_1', 'previous_choice_2',
                                'previous_choice_3', 'previous_choice_4', 'previous_choice_5',
                                'previous_choice_6', 'previous_choice_7', 'previous_choice_8',
                                'previous_choice_9', 'previous_choice_10', 'previous_outcome_1',
                                'previous_outcome_2','previous_outcome_3', 'previous_outcome_4',
                                'previous_outcome_5','previous_outcome_6','previous_outcome_7',
                                'previous_outcome_8','previous_outcome_9', 'previous_outcome_10']]
            except:
                print('No first movement')
            behav = behav.dropna()
            idx = behav.index
            behav = behav.reset_index()
            endog, exog = patsy.dmatrices('choice ~ 1 + previous_choice_1:C(previous_outcome_1)'
                                      '+ previous_choice_2:C(previous_outcome_2)'
                                      '+ previous_choice_3:C(previous_outcome_3)'
                                      '+ previous_choice_4:C(previous_outcome_4)'
                                      '+ previous_choice_5:C(previous_outcome_5)'
                                      '+ previous_choice_6:C(previous_outcome_6)'
                                      '+ previous_choice_7:C(previous_outcome_7)'
                                      '+ previous_choice_8:C(previous_outcome_8)'
                                      '+ previous_choice_9:C(previous_outcome_9)'
                                      '+ previous_choice_10:C(previous_outcome_10)',
                                       data=behav,return_type='dataframe')

    else:
        behav = behav.dropna().reset_index()
        endog, exog = patsy.dmatrices('choice ~ 1 + previous_laser_1 + previous_choice_1:C(previous_outcome_1)'
                                      '+ previous_choice_2:C(previous_outcome_2)'
                                      '+ previous_choice_3:C(previous_outcome_3)'
                                      '+ previous_choice_4:C(previous_outcome_4)'
                                      '+ previous_choice_5:C(previous_outcome_5)'
                                      '+ previous_choice_6:C(previous_outcome_6)'
                                      '+ previous_choice_7:C(previous_outcome_7)'
                                      '+ previous_choice_8:C(previous_outcome_8)'
                                      '+ previous_choice_9:C(previous_outcome_9)'
                                      '+ previous_choice_10:C(previous_outcome_10)',
                                       data=behav,return_type='dataframe')


    exog.rename(columns={'Intercept': 'bias',
                        'previous_choice_1:C(previous_outcome_1)[0.0]': 'unrewarded_1',
                        'previous_choice_1:C(previous_outcome_1)[1.0]': 'rewarded_1',
                        'previous_choice_2:C(previous_outcome_2)[0.0]': 'unrewarded_2',
                        'previous_choice_2:C(previous_outcome_2)[1.0]': 'rewarded_2',
                        'previous_choice_3:C(previous_outcome_3)[0.0]': 'unrewarded_3',
                        'previous_choice_3:C(previous_outcome_3)[1.0]': 'rewarded_3',
                        'previous_choice_4:C(previous_outcome_4)[0.0]': 'unrewarded_4',
                        'previous_choice_4:C(previous_outcome_4)[1.0]': 'rewarded_4',
                        'previous_choice_5:C(previous_outcome_5)[0.0]': 'unrewarded_5',
                        'previous_choice_5:C(previous_outcome_5)[1.0]': 'rewarded_5',
                        'previous_choice_6:C(previous_outcome_6)[0.0]': 'unrewarded_6',
                        'previous_choice_6:C(previous_outcome_6)[1.0]': 'rewarded_6',
                        'previous_choice_7:C(previous_outcome_7)[0.0]': 'unrewarded_7',
                        'previous_choice_7:C(previous_outcome_7)[1.0]': 'rewarded_7',
                        'previous_choice_8:C(previous_outcome_8)[0.0]': 'unrewarded_8',
                        'previous_choice_8:C(previous_outcome_8)[1.0]': 'rewarded_8',
                        'previous_choice_9:C(previous_outcome_9)[0.0]': 'unrewarded_9',
                        'previous_choice_9:C(previous_outcome_9)[1.0]': 'rewarded_9',
                        'previous_choice_10:C(previous_outcome_10)[0.0]': 'unrewarded_10',
                        'previous_choice_10:C(previous_outcome_10)[1.0]': 'rewarded_10'},
                 inplace = True)
    # Fit model
    logit_model = sm.Logit(endog, exog)
    res = logit_model.fit_regularized(disp=False) # run silently
    pred = res.predict(exog)
    params = pd.DataFrame({'coefficient':res.params, 'pvalues': res.pvalues,
                           'ci_95':res.conf_int()[0]-res.params}).reset_index()
    params['trials_back'] = np.nan
    params['type'] = np.nan
    params.loc[params['index']=='unrewarded_1', ['trials_back', 'type']] = [1, 'unrewarded']
    params.loc[params['index']=='unrewarded_2', ['trials_back', 'type']] = [2, 'unrewarded']
    params.loc[params['index']=='unrewarded_3', ['trials_back', 'type']] = [3, 'unrewarded']
    params.loc[params['index']=='unrewarded_4', ['trials_back', 'type']] = [4, 'unrewarded']
    params.loc[params['index']=='unrewarded_5', ['trials_back', 'type']] = [5, 'unrewarded']
    params.loc[params['index']=='rewarded_1', ['trials_back', 'type']] = [1, 'rewarded']
    params.loc[params['index']=='rewarded_2', ['trials_back', 'type']] = [2, 'rewarded']
    params.loc[params['index']=='rewarded_3', ['trials_back', 'type']] = [3, 'rewarded']
    params.loc[params['index']=='rewarded_4', ['trials_back', 'type']] = [4, 'rewarded']
    params.loc[params['index']=='rewarded_5', ['trials_back', 'type']] = [5, 'rewarded']
    params.loc[params['index']=='unrewarded_6', ['trials_back', 'type']] = [6, 'unrewarded']
    params.loc[params['index']=='unrewarded_7', ['trials_back', 'type']] = [7, 'unrewarded']
    params.loc[params['index']=='unrewarded_8', ['trials_back', 'type']] = [8, 'unrewarded']
    params.loc[params['index']=='unrewarded_9', ['trials_back', 'type']] = [9, 'unrewarded']
    params.loc[params['index']=='unrewarded_10', ['trials_back', 'type']] = [10, 'unrewarded']
    params.loc[params['index']=='rewarded_6', ['trials_back', 'type']] = [6, 'rewarded']
    params.loc[params['index']=='rewarded_7', ['trials_back', 'type']] = [7, 'rewarded']
    params.loc[params['index']=='rewarded_8', ['trials_back', 'type']] = [8, 'rewarded']
    params.loc[params['index']=='rewarded_9', ['trials_back', 'type']] = [9, 'rewarded']
    params.loc[params['index']=='rewarded_10', ['trials_back', 'type']] = [10, 'rewarded']
    params.loc[params['index']=='bias', ['trials_back', 'type']] = [0, 'bias']
    try:
        params.loc[params['index']=='previous_laser_1',
                   ['trials_back', 'type']] = [11, 'previous_laser_1']
    except:
        print('No laser coefficient')
    # Calculate accuracy in crossvalidated version
    acc = np.array([])
    kf = KFold(n_splits=5, shuffle=True)
    for train, test in kf.split(endog):
            X_train, X_test, y_train, y_test = exog.loc[train], exog.loc[test], \
                                               endog.loc[train], endog.loc[test]
            # fit again
            logit_model = sm.Logit(y_train, X_train)
            res = logit_model.fit_regularized(disp=False) # run silently
            # compute the accuracy on held-out data [from Luigi]:
            # suppose you are predicting Pr(Left), let's call it p,
            # the % match is p if the actual choice is left, or 1-p if the actual choice is right
            # if you were to simulate it, in the end you would get these numbers
            y_test['pred'] = res.predict(X_test)
            y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
            acc = np.append(acc, y_test['pred'].mean())
    if model==True:
        return params, np.mean(acc), pred, idx
    return params, np.mean(acc)


def load_session_dataframe(ses):
    choice = np.load(ses + '/alf/_ibl_trials.choice.npy')*-1
    reward = (np.load(ses + '/alf/_ibl_trials.feedbackType.npy')>0)*1
    block = np.load(ses + '/alf/_ibl_trials.probabilityLeft.npy')
    response_time = np.load(ses + '/alf/_ibl_trials.response_times.npy')
    cue_on_trigger = np.load(ses + '/alf/_ibl_trials.goCueTrigger_times.npy')
    cue_on = np.load(ses + '/alf/_ibl_trials.goCue_times.npy')
    potential_reward_r = np.load(ses + '/alf/_ibl_trials.right_reward.npy')
    potential_reward_l = np.load(ses + '/alf/_ibl_trials.left_reward.npy')
    try:
        firstmove = np.load(ses + '/alf/_ibl_trials.firstMovement_times.npy')
    except:
        print('Extracting wheel movement')
        extract_all_wheel(ses, save=True)
        firstmove = np.load(ses + '/alf/_ibl_trials.goCueTrigger_times.npy')
    behav = pd.DataFrame({'choice' : choice, 'outcome' : reward,
                          'probabilityLeft' : block, 'first_movement': firstmove,
                          'response_time': response_time, 'cue_on_trigger': cue_on_trigger,
                          'cue_on':cue_on, 'potential_reward_l':potential_reward_l,
                          'potential_reward_r': potential_reward_r})
    try:
        laser = np.load(ses + '/alf/_ibl_trials.opto.npy')
    except:
        print('No laser info')
        laser = np.zeros(len(choice))
        laser[:] = np.nan
    try:
        laser_block = np.load(ses + '/alf/_ibl_trials.opto_block.npy')
        behav['laser_block'] = laser_block
    except:
        print('No laser block')
    behav['laser'] = laser
    behav['previous_laser_1'] = behav['laser'].shift(1)
    behav['previous_outcome_1'] = behav['outcome'].shift(1)
    behav['previous_choice_1'] =  behav['choice'].shift(1)
    behav['previous_outcome_2'] = behav['outcome'].shift(2)
    behav['previous_choice_2'] =  behav['choice'].shift(2)
    behav['previous_outcome_3'] = behav['outcome'].shift(3)
    behav['previous_choice_3'] =  behav['choice'].shift(3)
    behav['previous_outcome_4'] = behav['outcome'].shift(4)
    behav['previous_choice_4'] =  behav['choice'].shift(4)
    behav['previous_outcome_5'] = behav['outcome'].shift(5)
    behav['previous_choice_5'] =  behav['choice'].shift(5)
    behav['previous_outcome_6'] = behav['outcome'].shift(6)
    behav['previous_choice_6'] =  behav['choice'].shift(6)
    behav['previous_outcome_7'] = behav['outcome'].shift(7)
    behav['previous_choice_7'] =  behav['choice'].shift(7)
    behav['previous_outcome_8'] = behav['outcome'].shift(8)
    behav['previous_choice_8'] =  behav['choice'].shift(8)
    behav['previous_outcome_9'] = behav['outcome'].shift(9)
    behav['previous_choice_9'] =  behav['choice'].shift(9)
    behav['previous_outcome_10'] = behav['outcome'].shift(10)
    behav['previous_choice_10'] =  behav['choice'].shift(10)
    behav = trial_within_block(behav)
    return behav

def trial_within_block(behav):
    behav['Probability of reward on left'] = np.nan
    behav['trial_within_block'] = np.nan
    behav['block_number'] = np.nan
    behav['block_change'] = np.concatenate([np.zeros(1),
                                            1*(np.diff(behav['probabilityLeft'])!=0)])
    block_switches = np.concatenate([np.zeros(1),
                                     behav.loc[behav['block_change']==1].index]).astype(int)
    for i in np.arange(len(block_switches)):
        if i == 0:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]:block_switches[i+1], -4] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]:block_switches[i+1], -3] = \
            np.arange(block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], -2] = i
        elif i == len(block_switches)-1:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:, -4] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]-5:, -3] = \
                np.arange(-5, len(behav) - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:, -2] = i
        else:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:block_switches[i+1], -4] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]-5:block_switches[i+1], -3] = \
                np.arange(-5, block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], -2] = i
    return behav

def num_star(pvalue):
    if pvalue < 0.05:
        stars = '*'
    if pvalue < 0.01:
        stars = '**'
    if pvalue < 0.001:
        stars = '***'
    if pvalue < 0.0001:
        stars = '****'
    return stars


def plot_session_wo_laser(ses_df):
    example = ses_df.copy()
    ses_df_c = ses_df.copy()
    params, _, value_choice, idx = fit_GLM(ses_df_c, model=True)
    example['value_choice']=np.nan
    example['value_choice'][idx] = value_choice.to_numpy()
    example['choice_l'] = (example['choice']==-1)*1
    example['choice_r'] = (example['choice']==1)*1
    example['reward_r'] = example['outcome']*example['choice_r']
    example['reward_l'] = example['outcome']*example['choice_l']
    example['probabilityRight']=0.1
    example.loc[example['probabilityLeft']==0.1, 'probabilityRight'] = 0.7
    fig, ax =plt.subplots(1,figsize=(10,5))
    plt.sca(ax)
    plt.plot(example['choice_r'].rolling(10, center=True).mean(),color='orange')
    #plt.plot(example['choice_l'].rolling(10, center=True).mean(),color='blue')
    plt.plot(example['value_choice'].rolling(10, center=True).mean(),color='k', linestyle='dashed')
    plt.plot(example['probabilityRight']/5+1.11,color='k',
             linestyle='--', alpha =0.5)
    #plt.xlim(0,400)
    plt.yticks(ticks=[0.0,0.25,0.5,0.75, 1.0],labels=[0.0,0.25,0.5,0.75,1.0])
    #plt.vlines(np.where((example['potential_reward_l']==1)),1.25,1.30, color='green')
    #plt.vlines(np.where((example['potential_reward_r']==1)),1.37,1.43, color='green')
    plt.vlines(np.where(example['choice_r']==1),1.01,1.11, color='black')
    plt.vlines(np.where(example['choice_l']==1),-0.11,-0.01, color='black')
    plt.vlines(np.where(example['reward_r']==1),1.01,1.11, color='green')
    plt.vlines(np.where(example['reward_l']==1),-0.11,-0.01, color='green')
    sns.despine()
    plt.xlabel('Trial')
    plt.ylabel('Choice probability')

if __name__ == "__main__":
    ses = sys.argv[1]
    data = load_session_dataframe(ses)
    params, acc = fit_GLM(data)
    plot_all(data, params, acc, ses, save=True)
