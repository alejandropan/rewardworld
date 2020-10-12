#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:52:51 2020

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:13:18 2020

@author: alex
"""
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
 

def true_stim_posterior(true_contrast, beliefSTD):

    def st_sp_0(percieve_contrast, beliefSTD=beliefSTD):
        all_contrasts = np.array([-0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25])
        a = 0
        b = 0
        for i in all_contrasts[all_contrasts>0]:
            a += norm.pdf((percieve_contrast - i)/ beliefSTD)
        for i in all_contrasts:
            b += norm.pdf((percieve_contrast - i)/ beliefSTD)

        return a/b * norm.pdf(percieve_contrast,true_contrast,beliefSTD)

    bs_right = quad(st_sp_0,-1, 1)

    return [1-bs_right[0],bs_right[0]]


# Given all of the Q values (a matrix of size num_contrasts x 2), compute the overall Q_left and Q_right
# (i.e., the overall value of choosing left or right) given the perceived stimulus
def compute_QL_QR(Q, contrast_posterior):  
    Q_L = contrast_posterior[0] * Q[0]
    Q_R = contrast_posterior[1] * Q[1]
    
    return Q_L, Q_R

def softmax_stay(Q_L, Q_R, beta, l_stay, r_stay, stay):
    p = [np.exp(Q_L / beta + stay*l_stay),
      np.exp(Q_R / beta + stay*r_stay)]
    p /= np.sum(p)

    return p

def trial_log_likelihood_stay(params, trial_data, Q, all_contrasts, all_posteriors,
                              previous_trial, trial_num):
    # Get relevant parameters
    trial_contrast, trial_choice, reward, laser = trial_data
    learning_rate, beliefSTD, extraVal, beta, stay = params
    Q =  Q.copy()

    # Compute the log-likelihood of the actual mouse choice
    if all_posteriors is None:
        contrast_posterior = true_stim_posterior(trial_contrast, beliefSTD)
    else:
        posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
        contrast_posterior = all_posteriors[posterior_idx, :]

    Q_L, Q_R = compute_QL_QR(Q, contrast_posterior)

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
    if laser == 1:
        received_reward = reward + extraVal
    else:
        received_reward = reward

    # Update Q-values according to the aggregate reward + laser value
    Q[trial_choice] += contrast_posterior[trial_choice] * learning_rate * (received_reward - Q_chosen)

    return LL, Q, Q_L, Q_R, choice_dist[1] #  choice_dist[1] = pChoice_right


def session_neg_log_likelihood_stay(params, *data, pregen_all_posteriors=True,
                                    fitting = True):
    # Unpack the arguments
    learning_rate, beliefSTD, extraVal, beta, stay = params
    rewards, true_contrasts, choices, lasers, ses_switch = data
    num_trials = len(rewards)
    
    # Start data holders
    Q_L = []
    Q_R = []
    Q_R_ITI =[]
    Q_L_ITI = []
    pRight = []

    # Generate the possible contrast list
    all_contrasts = np.array([-0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25])
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
    Q = np.zeros(2)
    
    for i in range(num_trials):
            if i == 0:
                trial_LL, newQ, Q_Lt, Q_Rt, pright = trial_log_likelihood_stay(params,
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]],
                                            Q, all_contrasts, all_posteriors,
                                            np.nan, i)

            else:
                if ses_switch[i] == 1:
                    Q = np.zeros(2)

                trial_LL, newQ, Q_Lt, Q_Rt, pright = trial_log_likelihood_stay(params,
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]],
                                            Q, all_contrasts, all_posteriors,
                                            choices[i-1], i)
            LL += trial_LL
            Q = newQ

            acc += (np.exp(trial_LL)>0.5)*1
            

            Q_L.append(Q_Lt)
            Q_R.append(Q_Rt)
            Q_L_ITI.append(Q[0])
            Q_R_ITI.append(Q[1])
            pRight.append(pright)
            
    acc = acc/num_trials
    
    if fitting==True:
        return -LL
    else:
        return -LL,  acc, Q_L, Q_R, Q_L_ITI, Q_R_ITI, pRight

# Optimize several times with different initializations and return the best fit parameters, and negative log likelihood

def optimizer_stay(data, num_fits = 3, initial_guess=[0.1, 1, 0, 1, 1]):
    # Accounting variables
    best_NLL = np.Inf
    best_x = [None, None, None, None, None]
    buffer_NLL = []
    buffer_x = np.empty([num_fits,len(initial_guess)])
    # Do our fit with several different initializations
    for i in range(num_fits):
        print('Starting fit %d' % i)

        # For every fit other than the first, construct a new initial guess
        if i != 0:
            lr_guess = np.random.uniform(0, 1)
            beliefSTD_guess = np.random.uniform(0.03, 1)
            extraVal_guess = np.random.uniform(-2,2)
            beta_guess = np.random.uniform(0.01, 1)
            stay = np.random.uniform(-1, 1)
            initial_guess = [lr_guess, beliefSTD_guess, extraVal_guess, beta_guess, stay]

        # Run the fit
        res = so.minimize(session_neg_log_likelihood_stay, initial_guess, args=data,
                    method='L-BFGS-B', bounds=[(0, 2), (0.03, 1), (-2, 2), (0.01, 1),
                                    (-1,1)])

        # If this fit is better than the previous best, remember it, otherwise toss
        buffer_x[i,:] = res.x
        buffer_NLL.append(res.fun)

        if res.fun <= best_NLL:
            best_NLL = res.fun
            best_x = res.x

    return best_x, best_NLL, buffer_NLL, buffer_x

def generate_data_stay(data, all_contrasts, learning_rate=0.3,
                       beliefSTD=0.1, extraVal=1, beta=0.2,
                       stay = 1, is_verbose=False, propagate_errors = True,
                       pregen_all_posteriors=True):

    rewards = []
    true_contrasts = []
    choices = []
    lasers = []

    if propagate_errors == False:
        prop = 3
    else:
        prop = 4
        
    if pregen_all_posteriors:
        all_posteriors = np.zeros((num_contrasts, 2))
        for idx, contrast in enumerate(all_contrasts):
            all_posteriors[idx, :] = true_stim_posterior(contrast, beliefSTD)
    else:
        all_posteriors = None

    # Simulate the POMDP model
    Q = np.zeros(2)
    for t in range(len(data[0])):
        if is_verbose:
            print(t)

        # Pick a true stimulus and store
        trial_contrast = data[1][t]
        true_contrasts.append(trial_contrast)
        # Add noise
        # Compute the log-likelihood of the actual mouse choice
        if all_posteriors is None:
            contrast_posterior = true_stim_posterior(trial_contrast, beliefSTD)
        else:
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
        if propagate_errors == True:
            if choice == data[prop][t]:
                reward += extraVal
                lasers.append(1)
            else:
                lasers.append(-1)
        else:
            reward = data[0][t]
            reward += extraVal*data[prop][t]
            lasers.append(data[prop][t])
        # Learn (update Q-values)
        if choice == 0:
            Q_chosen = Q_L
        else:
            Q_chosen = Q_R
            
        Q[choice] =  learning_rate * (reward - Q_chosen)

    return rewards, true_contrasts, choices, lasers

def transform_model_struct_2_POMDP(model_data, simulate_data):
        simulate_data.loc[simulate_data['extraRewardTrials'] == 'right', 'extraRewardTrials' ] = 1
        simulate_data.loc[simulate_data['extraRewardTrials'] == 'left', 'extraRewardTrials' ] = 0
        simulate_data.loc[simulate_data['extraRewardTrials'] == 'none', 'extraRewardTrials' ] = -1
        obj = model_data
        obj['choice'] = obj['choice'] * -1
        obj.loc[obj['choice'] == -1, 'choice'] = 0
        obj['laser_side'] = simulate_data['extraRewardTrials']
        return obj

def aic(LL,n_param):
    # Calculates Akaike Information Criterion
    aic =  2*n_param - 2*LL
    return aic

# Script

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

# Fitting and calculating
for i, mouse in enumerate(mice):
    
        # Transform the data
        model_data_nphr, simulate_data_nphr  = \
            psy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse],
                                              virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0])
        # Obtain sessions switches
        session_switches = np.zeros(len(model_data_nphr))
        for session in model_data_nphr['ses'].unique():
             session_switches[model_data_nphr.ses.ge(session).idxmax()]=1

        obj = transform_model_struct_2_POMDP(model_data_nphr, simulate_data_nphr)
        
        opto = obj['extraRewardTrials'].to_numpy()
        
        # Opto comes in a weird array this fixes it
        lasers = []
        for i in range(len(opto)):
            try:
                lasers.append(int(opto[i][0]))
            except:
                lasers.append(int(opto[i]))
                
        choices = list(obj['choice'].to_numpy())
        contrasts = list(obj['stimTrials'].to_numpy())
        rewards = list(obj['reward'].to_numpy())
        laser_side = list(obj['laser_side'].to_numpy())

        data = (rewards[:int(len(rewards)*train_set_size)],
                contrasts[:int(len(rewards)*train_set_size)],
                choices[:int(len(rewards)*train_set_size)],
                lasers[:int(len(rewards)*train_set_size)],
                session_switches[:int(len(rewards)*train_set_size)])
        simulate_data = (rewards[:int(len(rewards)*train_set_size)],
                         contrasts[:int(len(rewards)*train_set_size)],
                         choices[:int(len(rewards)*train_set_size)],
                      lasers[:int(len(rewards)*train_set_size)],
                      laser_side[:int(len(rewards)*train_set_size)],
                      session_switches[:int(len(rewards)*train_set_size)])
        
        if train_set_size == 1:
            data_test = data
            simulate_data_test = simulate_data
        
        # else: Need to find a good cross validation option, I was dividing 
        # in order but that might create artifacts
        
        # Fit:
        (best_x_stay, train_NLL_stay, buffer_NLL_stay,
         buffer_x_stay) = optimizer_stay(data, initial_guess=[0.3, 0.05, -1, 0.2,1])
        
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
        cv_aic_stay = aic(LL, len(best_x_stay))
        cv_acc_stay = acc
        
        # Add to storing DataFrames
        model_parameters_mouse = pd.DataFrame()
        model_parameters_mouse['x'] = [best_x_stay]
        model_parameters_mouse['LL'] = (LL/len(data_test[0])) # Log likelihood per trial
        model_parameters_mouse['aic'] = cv_aic_stay
        model_parameters_mouse['accu'] = cv_acc_stay
        model_parameters_mouse['model_name'] = '2q2RPE'
        model_parameters_mouse['mouse'] = mouse
        model_parameters_mouse['virus'] = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0]
        model_parameters = pd.concat([model_parameters, model_parameters_mouse])

        # Add simulatio0n data
        sim_data = generate_data_stay(simulate_data_test, all_contrasts, learning_rate=best_x_stay[0],
                                       beliefSTD=best_x_stay[1], extraVal=best_x_stay[2], beta=best_x_stay[3], stay=best_x_stay[4])
        sim_data = pd.DataFrame(sim_data).T
        sim_data = sim_data.rename(columns={0: "model_rewards", 
                                            1: "signed_contrast", 
                                            2: "simulated_choices",
                                            3: "model_laser"})
        
        psy.loc[psy['mouse_name']==mouse,'model_rewards']=sim_data['model_rewards']
        psy.loc[psy['mouse_name']==mouse,'simulated_choices']=sim_data['simulated_choices']
        psy.loc[psy['mouse_name']==mouse,'model_laser']=sim_data['model_laser']
        
        

# Analysis


boxplot_model_parameters_per_mouse(model_parameters,
                                       model_type= 'w_stay',
                                       save = True)
plot_q_trial_whole_dataset(psy)
plot_qmotivation_trial_whole_dataset(psy, save= True)

