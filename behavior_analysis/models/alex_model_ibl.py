#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 21:40:20 2020

@author: alex
"""


import os
import datajoint as dj
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
import pandas as pd
import numpy as np
os.chdir('paper-behavior')
from paper_behavior_functions import (query_sessions_around_criterion,
                                      EXAMPLE_MOUSE, institution_map,
                                      dj2pandas, fit_psychfunc)
os.chdir('..')
os.chdir('rewardworld')
from ibl_pipeline import behavior, subject, reference
import os
from tqdm.auto import tqdm
from sklearn.model_selection import KFold

# for modelling
import numpy as np
import scipy.optimize as so
import time
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from rew_alf.data_organizers import *
import seaborn as sns
from scipy.stats.distributions import chi2
from scipy.stats import norm
import random
from matplotlib.lines import Line2D
import os
import glob
from os import path
from scipy.integrate import quad

def ibl_df_to_Q_learning_model_format(psy_dataframe, virus = None):

    '''
    Parameters
    ----------
    psy_df : datafram ewith all the information for every trial in every animal,
    is the product of load_behavior_data_from_root
    
    virus : virus to urn the model on
    
    Returns
    -------
    simulate_data : required data for simulating the POMDP
    model_data: required for running the POMDP model

    '''
    
    
    # Select virus and make dataframe
    psy_df = psy_dataframe.loc[psy_dataframe['virus'] == virus].copy()
    psy_df = psy_df.reset_index()
    
    # Signed contrast
    signed_contrasts = np.zeros([len(psy_dataframe),1])
    signed_contrasts[:,0] = psy_df['signed_contrasts'].to_numpy()
    
    # Make dataframes for t
    model_data = pd.DataFrame({'stimTrials': signed_contrasts[:,0], 
                              'choice': psy_df['choice'], 'reward': psy_df['feedbackType'], 'ses':psy_df['session_start_time']})
    model_data.loc[model_data['reward'] == -1, 'reward'] = 0
    
    simulate_data = pd.DataFrame({'stimTrials': signed_contrasts[:,0],
                              'choice': psy_df['choice'], 'ses':psy_df['session_start_time']})

    return model_data,  simulate_data

def true_stim_posterior(true_contrast, beliefSTD):

    def st_sp_0(percieve_contrast, beliefSTD=beliefSTD):
        all_contrasts = np.array([-1,-0.5,-0.25, -0.125, -0.0625,\
                                  0.0625, 0.125, 0.25, 0.5, 1])
        a = 0
        b = 0
        for i in all_contrasts[all_contrasts>0]:
            a += norm.pdf((percieve_contrast - i)/ beliefSTD)
        for i in all_contrasts:
            b += norm.pdf((percieve_contrast - i)/ beliefSTD)

        return a/b * norm.pdf(percieve_contrast,true_contrast,beliefSTD)

    bs_right = quad(st_sp_0,-2, 2)

    return [1-bs_right[0],bs_right[0]]

def compute_QL_QR(Q, contrast_posterior):
    Q_L = 0
    Q_R = 0

    # We compute Q_L, Q_R as in the Cell paper, by taking the weighted average of the various perceived stimuli,
    # according to the probability that they were perceived
    for i in range(len(contrast_posterior)):
        Q_L += contrast_posterior[i] * Q[i, 0]
        Q_R += contrast_posterior[i] * Q[i, 1]

    return Q_L, Q_R

def softmax_stay(Q_L, Q_R, beta, l_stay, r_stay, stay):
    p = [np.exp(Q_L / beta + stay*l_stay),
      np.exp(Q_R / beta + stay*r_stay)]
    p /= np.sum(p)

    return p

def trial_log_likelihood_stay(params, trial_data, Q, all_contrasts, all_posteriors,
                              previous_trial, trial_num, fitting = False):
    # Get relevant parameters
    trial_contrast, trial_choice, reward = trial_data
    learning_rate, beliefSTD, beta, stay = params
    Q =  Q.copy()

    # Compute the log-likelihood of the actual mouse choice
    if all_posteriors is None:
        contrast_posterior = true_stim_posterior(trial_contrast, beliefSTD)
    else:
        posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
        contrast_posterior = all_posteriors[posterior_idx, :]

    Q_L, Q_R = compute_QL_QR(Q, contrast_posterior)
    # Weighted Q values for encoding/decoding
    posterior = np.array([[contrast_posterior[0]],[contrast_posterior[1]]])
    Q_w = Q * posterior
    Q_LL = Q_w[0,0]
    Q_RL = Q_w[1,0]
    Q_LR = Q_w[0,1]
    Q_RR = Q_w[1,1]

    if trial_num == 0:
        (l_stay, r_stay) = [0,0]
    else:
        previous_choice= [0,0]
        previous_choice[previous_trial] = 1
        (l_stay, r_stay) = previous_choice

    choice_dist = softmax_stay(Q_L, Q_R, beta,l_stay, r_stay, stay)
    LL = np.log(choice_dist[trial_choice])

    # Learning
    if trial_choice == 0:
        Q_chosen = Q_L
    else:
        Q_chosen = Q_R

    # Laser-modulation
    received_reward = reward

    # Update Q-values according to the aggregate reward + laser value
    for i in range(2):
        Q[i, trial_choice] += contrast_posterior[i] * learning_rate * (received_reward - Q_chosen)


    if fitting == False:
        return LL, Q, Q_L, Q_R, choice_dist[1] #  choice_dist[1] = pChoice_right

    else:
        return LL, Q



def session_neg_log_likelihood_stay(params, *data, pregen_all_posteriors=True,
                                    fitting = True):
    # Unpack the arguments
    learning_rate, beliefSTD, beta, stay = params
    rewards, true_contrasts, choices, ses_switch = data
    num_trials = len(rewards)
    
    # Start data holders
    Q_L = []
    Q_R = []
    Q_R_ITI =[]
    Q_L_ITI = []
    pRight = []

    # Generate the possible contrast list
    all_contrasts = np.array([-1,-0.5,-0.25, -0.125, -0.0625,\
                                  0.0625, 0.125, 0.25, 0.5, 1])
    num_contrasts = len(all_contrasts)

    # If True, generate all posterior distributions ahead of time to save time
    if pregen_all_posteriors:
        all_posteriors = np.zeros((num_contrasts, 2))
        for idx, contrast in enumerate(all_contrasts):
            all_posteriors[idx, :] = true_stim_posterior(contrast, beliefSTD)
    else:
        all_posteriors = None

    # Compute the log-likelihood
    acc = 0
    LL = 0
    Q = np.zeros([2,2])
    
    for i in range(num_trials):
            if i == 0:
                trial_LL, newQ, Q_Lt, Q_Rt, pright = trial_log_likelihood_stay(params,
                                            [true_contrasts[i], choices[i], rewards[i]],
                                            Q, all_contrasts, all_posteriors,
                                            np.nan, i)

            else:
                if ses_switch[i] == 1:
                    Q = np.zeros([2,2])

                trial_LL, newQ, Q_Lt, Q_Rt, pright = trial_log_likelihood_stay(params,
                                            [true_contrasts[i], choices[i], rewards[i]],
                                            Q, all_contrasts, all_posteriors,
                                            choices[i-1], i)
            LL += trial_LL
            Q = newQ

            acc += (np.exp(trial_LL)>0.5)*1
            

            Q_L.append(Q_Lt)
            Q_R.append(Q_Rt)
            Q_L_ITI.append(np.sum(Q, axis=0)[0])
            Q_R_ITI.append(np.sum(Q, axis=0)[1])
            pRight.append(pright)
            
    acc = acc/num_trials
    
    if fitting==True:
        return -LL
    else:
        return -LL,  acc, Q_L, Q_R, Q_L_ITI, Q_R_ITI, pRight
# Optimize several times with different initializations and return the best fit parameters, and negative log likelihood

def optimizer_stay(data, num_fits = 3, initial_guess=[0.1, 1, 1, 1]):
    # Accounting variables
    best_NLL = np.Inf
    best_x = [None, None, None, None]
    buffer_NLL = []
    buffer_x = np.empty([num_fits,len(initial_guess)])
    # Do our fit with several different initializations
    for i in range(num_fits):
        print('Starting fit %d' % i)

        # For every fit other than the first, construct a new initial guess
        if i != 0:
            lr_guess = np.random.uniform(0, 1)
            beliefSTD_guess = np.random.uniform(0.03, 1)
            beta_guess = np.random.uniform(0.01, 1)
            stay = np.random.uniform(-1, 1)
            initial_guess = [lr_guess, beliefSTD_guess, beta_guess, stay]

        # Run the fit
        res = so.minimize(session_neg_log_likelihood_stay, initial_guess, args=data,
                    method='L-BFGS-B', bounds=[(0, 2), (0.03, 1), (0.01, 1),
                                    (-1,1)])

        # If this fit is better than the previous best, remember it, otherwise toss
        buffer_x[i,:] = res.x
        buffer_NLL.append(res.fun)

        if res.fun <= best_NLL:
            best_NLL = res.fun
            best_x = res.x

    return best_x, best_NLL, buffer_NLL, buffer_x


def transform_model_struct_2_POMDP(model_data, simulate_data):
        obj = model_data
        obj['choice'] = obj['choice'] * -1
        obj.loc[obj['choice'] == -1, 'choice'] = 0
        return obj

def aic(LL,n_param):
    # Calculates Akaike Information Criterion
    aic =  2*n_param - 2*LL
    return aic

def generate_data_stay(data, all_contrasts, learning_rate=0.3,
                       beliefSTD=0.1, beta=0.2,
                       stay = 1, is_verbose=False, propagate_errors = True):

    rewards = []
    true_contrasts = []
    choices = []
    
    all_posteriors = np.zeros((num_contrasts, 2))
    for idx, contrast in enumerate(all_contrasts):
            all_posteriors[idx, :] = true_stim_posterior(contrast, beliefSTD)

    if propagate_errors == False:
        prop = 3
    else:
        prop = 4

    # Simulate the POMDP model
    Q = np.zeros([2, 2])
    for t in range(len(data[0])):
        if is_verbose:
            print(t)

        # Pick a true stimulus and store
        trial_contrast = data[1][t]
        true_contrasts.append(trial_contrast)
        # Add noise
        posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
        contrast_posterior = all_posteriors[posterior_idx, :]

        Q_L, Q_R = compute_QL_QR(Q, contrast_posterior)

        if t == 0:
            (l_stay, r_stay) = [0,0]
        else:
            previous_choice= [0,0]
            previous_choice[choices[t-1]] = 1
            (l_stay, r_stay) = previous_choice
        choice_dist = softmax_stay(Q_L, Q_R, beta,l_stay, r_stay, stay)
        choice = np.random.choice(2, p = [float(choice_dist[0]), float(choice_dist[1])])
        choices.append(choice)

        # Get reward and store it
        if np.sign(trial_contrast) == -1 and choice == 0:
            reward = 1
        elif np.sign(trial_contrast) == 1 and choice == 1:
            reward = 1
        elif np.sign(trial_contrast) == 0:
            reward = random.choice([0,1])
        else:
            reward = 0

        rewards.append(reward)

        # Add laser value on the correct condition
        if propagate_errors == False:
            reward = data[0][t]
        # Learn (update Q-values)
        if choice == 0:
            Q_chosen = Q_L
        else:
            Q_chosen = Q_R

        for i in range(2):
            Q[i, choice] += contrast_posterior[i] * learning_rate * (reward - Q[i, choice])

    return rewards, true_contrasts, choices

#%% Script 

# progress bar
tqdm.pandas(desc="model fitting")

# whether to query data from DataJoint (True), or load from disk (False)
query = True
institution_map, col_names = institution_map()

# ========================================== #
#%% 1. LOAD DATA
# ========================================== #

# Query sessions: before and after full task was first introduced
if query is True:
    use_sessions, _ = query_sessions_around_criterion(criterion='biased',
                                                      days_from_criterion=[2, 3],
                                                      as_dataframe=False,
                                                      force_cutoff=True)

    trial_fields = ('trial_stim_contrast_left', 'trial_stim_contrast_right',
                    'trial_response_time', 'trial_stim_prob_left',
                    'trial_feedback_type', 'trial_stim_on_time', 'trial_response_choice')

    # query trial data for sessions and subject name and lab info
    trials = use_sessions.proj('task_protocol') * behavior.TrialSet.Trial.proj(*trial_fields)
    subject_info = subject.Subject.proj('subject_nickname') * \
                   (subject.SubjectLab * reference.Lab).proj('institution_short')

    # Fetch, join and sort data as a pandas DataFrame
    behav = dj2pandas(trials.fetch(format='frame')
                             .join(subject_info.fetch(format='frame'))
                             .sort_values(by=['institution_short', 'subject_nickname',
                                              'session_start_time', 'trial_id'])
                             .reset_index())
    behav['institution_code'] = behav.institution_short.map(institution_map)
    # split the two types of task protocols (remove the pybpod version number)
    behav['task'] = behav['task_protocol'].str[14:20].copy()

    # RECODE SOME THINGS JUST FOR PATSY
    behav['contrast'] = np.abs(behav.signed_contrast)
    behav['stimulus_side'] = np.sign(behav.signed_contrast)
    behav['block_id'] = behav['probabilityLeft'].map({80:-1, 50:0, 20:1})
    
behav = behav.loc[behav['lab_name']=='wittenlab']
behav['signed_contrasts'] = behav['signed_contrast']/100
behav['choice'] = behav['choice']*-1
behav['feedbackType'] = 1
behav.loc[behav['correct']==0, 'feedbackType'] = 0
behav = behav.loc[behav['task']=='traini']

# Select mice
behav = behav.rename(columns={'subject_nickname':'mouse_name'})
psy = behav.copy()
mice = behav['mouse_name'].unique()

# Create columns that we are about to calculate
psy['QL'] = np.nan
psy['QR'] = np.nan
psy['ITIQL'] = np.nan
psy['ITIQR'] = np.nan
psy['pRight'] = np.nan
psy['virus'] = 'stupidity'
# Cross validation parameters
train_set_size = 1
cross_validate = False

# Dataframes for output
model_parameters = pd.DataFrame()

#%% Fitting
# Generate the possible contrast list
all_contrasts = np.array([-1,-0.5,-0.25, -0.125, -0.0625,\
                                  0.0625, 0.125, 0.25, 0.5, 1])
num_contrasts = len(all_contrasts)

# Fitting and calculating
for i, mouse in enumerate(mice):
    
        # Transform the data
        model_data_nphr, simulate_data_nphr  = \
            ibl_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse],
                                              virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0])
        # Obtain sessions switches
        session_switches = np.zeros(len(model_data_nphr))
        for session in model_data_nphr['ses'].unique():
             session_switches[model_data_nphr.ses.ge(session).idxmax()]=1

        obj = transform_model_struct_2_POMDP(model_data_nphr, simulate_data_nphr)
                
        choices = list(obj['choice'].to_numpy())
        contrasts = list(obj['stimTrials'].to_numpy())
        rewards = list(obj['reward'].to_numpy())

        data = (rewards[:int(len(rewards)*train_set_size)],
                contrasts[:int(len(rewards)*train_set_size)],
                choices[:int(len(rewards)*train_set_size)],
                session_switches[:int(len(rewards)*train_set_size)])
        simulate_data = (rewards[:int(len(rewards)*train_set_size)],
                         contrasts[:int(len(rewards)*train_set_size)],
                         choices[:int(len(rewards)*train_set_size)],
                      session_switches[:int(len(rewards)*train_set_size)])
        
        if train_set_size == 1:
            data_test = data
            simulate_data_test = simulate_data
        
        # else: Need to find a good cross validation option, I was dividing 
        # in order but that might create artifacts
        
        # Fit:
        (best_x_stay, train_NLL_stay, buffer_NLL_stay,
         buffer_x_stay) = optimizer_stay(data, initial_guess=[0.3, 0.05, 0.2,1])
        
        # Calculate variables
        neg_LL, acc, Q_L, Q_R, Q_L_ITI, Q_R_ITI, pRight = session_neg_log_likelihood_stay(best_x_stay,
                  *data_test, pregen_all_posteriors=True,fitting =False)
        
        psy.loc[psy['mouse_name']==mouse, 'QL']=Q_L
        psy.loc[psy['mouse_name']==mouse, 'QR']=Q_R
        psy.loc[psy['mouse_name']==mouse, 'ITIQL']=Q_L_ITI
        psy.loc[psy['mouse_name']==mouse, 'ITIQR']=Q_R_ITI
        psy.loc[psy['mouse_name']==mouse, 'pRight']=pRight
        
        # Model QC
        LL = neg_LL*-1
        cv_acc_stay = acc
        
        # Add to storing DataFrames
        model_parameters_mouse = pd.DataFrame()
        model_parameters_mouse['x'] = [best_x_stay]
        model_parameters_mouse['LL'] = (LL/len(data_test[0])) # Log likelihood per trial
        model_parameters_mouse['accu'] = cv_acc_stay
        model_parameters_mouse['model_name'] = 'Nathaniel'
        model_parameters_mouse['mouse'] = mouse
        model_parameters_mouse['virus'] = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0]
        model_parameters = pd.concat([model_parameters, model_parameters_mouse])

        # Add simulatio0n data
        sim_data = generate_data_stay(simulate_data_test, all_contrasts, learning_rate=best_x_stay[0],
                                       beliefSTD=best_x_stay[1], beta=best_x_stay[2], stay=best_x_stay[3])
        sim_data = pd.DataFrame(sim_data).T
        sim_data = sim_data.rename(columns={0: "model_rewards", 
                                            1: "signed_contrast", 
                                            2: "simulated_choices"})
        
        psy.loc[psy['mouse_name']==mouse,'model_rewards']=sim_data['model_rewards']
        psy.loc[psy['mouse_name']==mouse,'simulated_choices']=sim_data['simulated_choices']
        
        

# Analysis
psy['QRQL'] = psy['QR'] - psy['QL']
psy['ses'] = psy['session_start_time'] 
psy = psy.reset_index()
psy['trial'] = psy['index']
psy = add_trial_within_block(psy)

def add_trial_within_block(df):
    '''
    df: dataframe with behavioral data
    '''
    df['trial_within_block'] = np.nan
    for mouse in df['mouse_name'].unique():
        for ses in df.loc[df['mouse_name']==mouse,'ses'].unique():
            session= df.loc[(df['mouse_name']
                             ==mouse) & (df['ses']==ses)]
            block_breaks = np.diff(session['probabilityLeft'])
            block_breaks = np.where(block_breaks != 0)
            for i, t in enumerate(block_breaks[0]):
                if i == 0:
                    for l in range(t+1):
                        session.iloc[l, session.columns.get_loc('trial_within_block')] = l
                
                else:
                    for x, l in enumerate(range(block_breaks[0][i-1]+1,t+1)):
                        session.iloc[l, session.columns.get_loc('trial_within_block')] = x
                if i + 1 == len(block_breaks[0]):
                    session.iloc[t+1:, session.columns.get_loc('trial_within_block')] = np.arange(len(session)-t -1)
            df.loc[(df['mouse_name']
                             ==mouse) & (df['ses']==ses)] =  session
    return df

def plot_q_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    #Get index for trials of block start
    index = psy_select.loc[psy['trial_within_block'] == 0, 'trial']
    index = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
    for i in range(10):
        for idx in index-(i+1):
            psy_select.loc[psy_select['trial'] == idx, 'trial_within_block'] = \
                -(i+1)
            psy_select.loc[psy_select['trial'] == idx, 'block_id'] = \
                psy_select.loc[psy_select['trial'] == idx+(i+1), 'block_id'].to_numpy()[0]
    
    fig, ax = plt.subplots(figsize = [5,5])
    palette ={'R':'g','L':'b','non_opto':'k'}
    sns.lineplot(data = psy_select, x = 'trial_within_block', y = 'QRQL',
                     hue = 'block_id', ci=68)
    plt.xlim(-10,50)
    plt.ylim(-1,1)
    plt.title('VTA-ChR2')
    ax.set_xlabel('Trial in block')
    ax.set_ylabel('QR-QL')

    plt.tight_layout()
    sns.despine()
    
    if save ==True:
        plt.savefig('q_across_trials.svg', dpi =300)
        plt.savefig('q_across_trials.jpeg',  dpi =300)
    
def plot_choice_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    psy_select['choice'] = psy_select['choice'] * - 1
    psy_select.loc[psy_select['choice'] == -1, 'choice']  = 0 
    sns.set(style = 'white')
    #Get index for trials of block start
    index = psy_select.loc[psy['trial_within_block'] == 0, 'index']
    index = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
    for i in range(10):
        for idx in index-(i+1):
            psy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
                -(i+1)
            psy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
                psy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]
    
    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'choice',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(0,1)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('Choice')
    
    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'choice',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(0,1)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()
    
    if save ==True:
        plt.savefig('choice_across_trials.svg', dpi =300)
        plt.savefig('choice_across_trials.jpeg',  dpi =300)    
        
        
plt.plot(s['trial_id'][200:350], s['QRQL'][200:350])
plt.plot(s['trial_id'][200:350], s['QRQL'][200:350].diff())

plt.plot(s['trial_id'][200:350], (s['choice'][200:350]*-1))
plt.plot(s['trial_id'][200:350], abs(s['probabilityLeft'][200:350].diff())>0)

    