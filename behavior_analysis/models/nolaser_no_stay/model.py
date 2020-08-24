#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:58:29 2020

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
    # Compute distribution over perceived contrast
    # start_time = time.time()

    def f(x):
        return norm.cdf(x,0,beliefSTD) * norm.pdf(x,true_contrast,beliefSTD)
    
    bs_right = quad(f,-np.inf, +np.inf)
    return [1-bs_right[0],bs_right[0]]


# Given all of the Q values (a matrix of size num_contrasts x 2), compute the overall Q_left and Q_right 
# (i.e., the overall value of choosing left or right) given the perceived stimulus
def compute_QL_QR(Q, trial_contrast, contrast_posterior):
    Q_L = 0
    Q_R = 0

    # We compute Q_L, Q_R as in the Cell paper, by taking the weighted average of the various perceived stimuli,
    # according to the probability that they were perceived
    for i in range(len(contrast_posterior)):
        Q_L += contrast_posterior[i] * Q[i, 0]
        Q_R += contrast_posterior[i] * Q[i, 1]

    return Q_L, Q_R

def softmax_stay(Q_L, Q_R, beta):
    p = [np.exp(Q_L / beta),
      np.exp(Q_R / beta)]
    p /= np.sum(p)

    return p

def trial_log_likelihood_stay(params, trial_data, Q, all_contrasts, all_posteriors, 
                              previous_trial, trial_num, retrieve_Q = False, retrieve_ITIQ = False):
    # Get relevant parameters
    trial_contrast, trial_choice, reward, laser = trial_data
    x = 1
    learning_rate, beliefSTD, beta = params
    Q =  Q.copy()

    # Compute the log-likelihood of the actual mouse choice
    if all_posteriors is None:
        contrast_posterior = true_stim_posterior(trial_contrast, beliefSTD)
    else:
        posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
        contrast_posterior = all_posteriors[posterior_idx, :]

    Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)
    
    
    choice_dist = softmax_stay(Q_L, Q_R, beta)
    LL = np.log(choice_dist[trial_choice])    

    # Learning
    if trial_choice == 0:
        Q_chosen = Q_L
    else:
        Q_chosen = Q_R

    # Laser-modulation
    if laser == 1:
        received_reward = reward
    else:
        received_reward = reward

    # Update Q-values according to the aggregate reward + laser value
    for i in range(2):
        Q[i, trial_choice] += contrast_posterior[i] * \
            learning_rate * (received_reward - Q_chosen)

    if retrieve_ITIQ == True:
        Q_L = np.sum(Q, axis=0)[0]
        Q_R = np.sum(Q, axis=0)[1]


    if retrieve_Q==True:
        return LL, Q, Q_L, Q_R, choice_dist[1] #  choice_dist[1] = pChoice_right
    
    else:
        return LL, Q



def session_neg_log_likelihood_stay(params, *data, pregen_all_posteriors=True, 
                                    accu=False, retrieve_Q =  False, 
                                    retrieve_ITIQ = False):
    # Unpack the arguments
    learning_rate, beliefSTD, beta = params
    rewards, true_contrasts, choices, lasers, ses_switch = data
    num_trials = len(rewards)
    
    if retrieve_Q==True:
        Q_L = []
        Q_R = []
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
    if accu == True:
        acc = 0
    LL = 0
    Q = np.zeros([2, 2])
    
    if retrieve_Q == True:
            
        for i in range(num_trials):
            if i == 0:
                if ses_switch[i] == 1:
                    Q =  np.zeros([2,2])
                trial_LL, newQ, Q_Lt, Q_Rt, pright = trial_log_likelihood_stay(params, 
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]], 
                                            Q, all_contrasts, all_posteriors, 
                                            np.nan, i, retrieve_Q=retrieve_Q, retrieve_ITIQ=retrieve_ITIQ)
            else:
                if ses_switch[i] == 1:
                    Q =  np.zeros([2,2])
                trial_LL, newQ, Q_Lt, Q_Rt, pright = trial_log_likelihood_stay(params, 
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]], 
                                            Q, all_contrasts, all_posteriors, 
                                            choices[i-1], i, retrieve_Q=retrieve_Q, retrieve_ITIQ=retrieve_ITIQ)

            if (i != 0) & (np.sum(Q, axis=0)[0] != np.sum(newQ, axis=0)[0]) & (np.sum(Q, axis=0)[1] != np.sum(newQ, axis=0)[1]):
                print('Warning, double update error in trial %d'%i)
            LL += trial_LL
            Q = newQ
            
            if accu == True:
                acc += (np.exp(trial_LL)>0.5)*1
            
            Q_L.append(Q_Lt)
            Q_R.append(Q_Rt)
            pRight.append(pright)
        
        
    else:        
        for i in range(num_trials):
            if i == 0:
                if ses_switch[i] == 1:
                    Q =  np.zeros([2,2])
                trial_LL, newQ = trial_log_likelihood_stay(params, 
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]], 
                                            Q, all_contrasts, all_posteriors, np.nan, i)
            else:
                if ses_switch[i] == 1:
                    Q =  np.zeros([2,2])
                trial_LL, newQ = trial_log_likelihood_stay(params, 
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]], 
                                            Q, all_contrasts, all_posteriors, choices[i-1], i)
            LL += trial_LL
            Q = newQ
            
            if accu == True:
                acc += (np.exp(trial_LL)>0.5)*1
                

    if retrieve_Q == True:   
        if accu == True:
            acc = acc/num_trials
            return -LL,  acc, Q_L, Q_R, pRight
        else:
            return -LL, Q_L, Q_R, pRight

    else:
        if accu == True:
            acc = acc/num_trials
            return -LL,  acc
        else:
            return -LL





# Optimize several times with different initializations and return the best fit parameters, and negative log likelihood

def optimizer_stay(data, num_fits = 20, initial_guess=[0.1, 1, 0, 1]):
    # Accounting variables
    best_NLL = np.Inf
    best_x = [None, None, None]
    buffer_NLL = []
    buffer_x = np.empty([num_fits,len(initial_guess)])
    # Do our fit with several different initializations
    for i in range(num_fits):
        print('Starting fit %d' % i)

        # For every fit other than the first, construct a new initial guess
        if i != 0:
            lr_guess = np.random.uniform(0, 2)
            beliefSTD_guess = np.random.uniform(0.03, 1)
            beta_guess = np.random.uniform(0.01, 1)
            initial_guess = [lr_guess, beliefSTD_guess, beta_guess]

        # Run the fit
        res = so.minimize(session_neg_log_likelihood_stay, initial_guess, args=data, 
                    method='L-BFGS-B', bounds=[(0, 2), (0.03, 1), (0.01, 5)])

        # If this fit is better than the previous best, remember it, otherwise toss
        buffer_x[i,:] = res.x
        buffer_NLL.append(res.fun)
        
        if res.fun <= best_NLL:
            best_NLL = res.fun
            best_x = res.x

    return best_x, best_NLL, buffer_NLL, buffer_x



# hardcode this for speed
def normal_pdf(x, loc, scale):
    factor = 1 / (np.sqrt(2 * np.pi) * scale)
    power = -0.5 * (((x - loc) / scale) ** 2)

    return factor * np.exp(power)