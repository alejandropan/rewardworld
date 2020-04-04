#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:23:04 2020

Lak 2016 reinforcement model

@author: alex
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from npy2pd import *

import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import fmin
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import least_squares
from scipy.special import softmax



#Input folder with raw npy files'
psy_df = opto_block_assigner(psy_df)
psy_df = add_signed_contrasts(psy_df)
psy_df = add_trial_within_block(psy_df)


# Select dataset
psy_df1  = psy_df

psy_df  = psy_df.loc[psy_df['virus']=='nphr']
psy_df = psy_df.reset_index()

# Get sesion information

# Opto stimuli
opto_side_num = np.zeros([psy_df.shape[0],1])
opto_side_num[psy_df.loc[(psy_df['opto_block'] == 'L')|
                         (psy_df['opto_block'] == 'R')].index, 0] = \
                         psy_df.loc[(psy_df['opto_block'] == 'L')| \
                                    (psy_df['opto_block'] == 'R'),'choice'] * \
                         psy_df.loc[(psy_df['opto_block'] == 'L')| \
                                    (psy_df['opto_block'] == 'R'),'opto.npy']
opto_side = np.empty([len(opto_side_num),1],dtype=object)
opto_side[:] = 'none'
opto_side[opto_side_num == 1] = 'left'
opto_side[opto_side_num == -1] = 'right'

# Signed contrast
signed_contrasts = np.zeros([len(opto_side_num),1])
signed_contrasts[:,0] = psy_df['signed_contrasts'].to_numpy()

# Make dataframe
model_data = pd.DataFrame({'stimTrials': signed_contrasts[:,0],'extraRewardTrials': opto_side[:,0], 
                          'choice': psy_df['choice']})

# RUN


# Set variables (for initiliaziton)
learningRate = 0.15     # learning rate
extraRewardVal = - 1    # value for opto
beliefNoiseSTD = 0.075			# noise in belief
Temperature = 0.1 #Temperature for softmax


params = [learningRate, extraRewardVal, beliefNoiseSTD, Temperature]



#Least squares
result1 = minimize(session_log_likelihood, params, args=(model_data), 
                   method = 'Nelder-Mead', callback=print_callback,
                   options ={'disp':True, 'xtol' : 0.1})

result1 = fmin(session_log_likelihood, params,
               callback=print_callback, disp =True,  xtol = 0.01)



params = result1['x']
output = RunPOMDP(params, model_data)

#Plt
plot_action_opto_block(output,psy_df)



###########################################################
##########################Functions########################
###########################################################

# Stable block assigner
def opto_block_assigner (psy_df):
    psy_df['opto_block'] = np.nan
    psy_df.loc[(psy_df['opto_probability_left'] == 1), 'opto_block'] = 'L'
    psy_df.loc[(psy_df['opto_probability_left'] == 0), 'opto_block'] = 'R'
    psy_df.loc[(psy_df['opto_probability_left'] == -1), 'opto_block'] = 'non_opto'
    return psy_df

# Signed contrast calculator
def add_signed_contrasts (psy_df):
    psy_df.loc[:,'contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df.loc[:,'contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df.loc[:,'signed_contrasts'] =  \
    (psy_df['contrastRight'] - psy_df['contrastLeft'])
    return psy_df

def add_trial_within_block(session):
    '''
    session: dataframe with behavioral data
    '''
    session['trial_within_block'] = np.nan
    block_breaks = np.diff(session['opto_probability_left'])
    block_breaks = np.where(block_breaks != 0)
    for i, t in enumerate(block_breaks[0]):
        if i == 0:
            for l in range(t+1):
                session.iloc[l, session.columns.get_loc('trial_within_block')] = l
        else:
            for x, l in enumerate(range(block_breaks[0][i-1]+1,t+1)):
                session.iloc[l, session.columns.get_loc('trial_within_block')] = x
    return session

def plot_action_opto_block(output,psy_df):
    '''
    Parameters
    ----------
    output : Output from runPOMDP
    psy_df : dataframe with real data

    Returns
    -------
    Figure model vs real data.

    '''
    #
    # neutral block
    psy_df['action'] = output['action']
    ax = sns.lineplot(data = psy_df, x = 'signed_contrasts', y = 'action',
                      hue = 'opto_block', ci=None, legend=False)
    ax.lines[0].set_linestyle("--")
    ax.lines[1].set_linestyle("--")
    ax.lines[2].set_linestyle("--")
    psy_df['right_choice'] = psy_df['choice'] == -1
    sns.lineplot(data = psy_df, x = 'signed_contrasts', y = 'right_choice', 
                 hue = 'opto_block')
    ax.set_ylabel('Fraction of Right Choices')
    


def session_log_likelihood(params, model_data = model_data):
        output = RunPOMDP(params, model_data)
        negLL = -np.sum(np.log(output['pChoice']))
        return negLL


def print_callback(xs):
    """
    Callback called after every iteration.

    xs is the estimated location of the optimum.
    """
    
    return(print (xs))






###########################################################

# Calculate Q values
def RunPOMDP(params, model_data):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    
    stimTrials = model_data['stimTrials']
    extraReward = model_data['extraRewardTrials']
    choices = model_data['choice']
    
    # set run numbers
    iterN = 1  # model values are averaged over iterations (odd number)
    trialN = len(stimTrials)
    
    
    # initialise variables, for speed
    action = np.empty([trialN,iterN],dtype=object)
    QL = np.zeros([trialN,iterN])
    QR = np.zeros([trialN,iterN])
    pL1 = np.zeros([trialN,iterN])
    pR1 = np.zeros([trialN,iterN])
    predictionError = np.zeros([trialN,iterN])


    for i in range(iterN):
        # initalise Q values for each iteration
        QLL = 1
        QRR = 1
        QLR = 0
        QRL = 0
        # start model
        for trial in range(trialN):
            # set contrast
            currentStim = stimTrials[trial]
    		# calculate belief
            Belief_L, Belief_R = true_stim_posterior(currentStim)
            Belief_R = 1 - Belief_L
            # initialise Q values for this iteration
            QL[trial,i] = Belief_L*QLL + Belief_R*QRL # need explanation for 4 q values
            QR[trial,i] = Belief_L*QLR + Belief_R*QRR
            
            #switiching for a softmax
            pL1[trial,i] , pR1[trial,i]  = q_softmax(QL[trial,i], QR[trial,i], Temperature)
            
            if choices[trial] == -1:
                action[trial,i] = 'right'
            else:
                action[trial,i] = 'left'
       
    		# trial reward for action chosen by agent
            if currentStim<0 and (action[trial,i] == 'left'): # If action and stim pair, reward
                if extraReward[trial] == 'left':
                    reward = 1 + extraRewardVal # Add extra reward in our case the laser, if it is a laser trial
                elif extraReward[trial] == 'right':
                    reward = 1
                elif extraReward[trial] == 'none':
                    reward = 1
    			
            elif currentStim>0 and (action[trial,i] == 'right'): # If action and stim pair, reward
                
                if extraReward[trial] == 'left':
                    reward = 1
                elif extraReward[trial] == 'right': # Add extra reward in our case the laser, if it is a laser trial
                    reward = 1 + extraRewardVal
                elif extraReward[trial] == 'none':
                    reward = 1
    			
            elif currentStim==0:
                if np.random.rand() > 0.5:	# no evidence up to chance, if correct
                    if action[trial,i] == 'left':      
                        if extraReward[trial] == 'left':
                            reward = 1 + extraRewardVal
                        elif extraReward[trial] == 'right':
                            reward = 1
                        elif extraReward[trial] == 'none':
                            reward = 1
    					
                    elif action[trial,i] =='right':
                        if extraReward[trial] == 'left':
                            reward = 1
                        elif extraReward[trial] == 'right':
                            reward = 1 + extraRewardVal
                        elif extraReward[trial] == 'none':
                            reward = 1
    				
                else: # if 0 evidence incorrect
                    if action[trial,i] == 'left':      
                        if extraReward[trial] == 'left':
                            reward = 0 + extraRewardVal
                        elif extraReward[trial] == 'right':
                            reward = 0
                        elif extraReward[trial] == 'none':
                            reward = 0
    					
                    elif action[trial,i] =='right':
                        if extraReward[trial] == 'left':
                            reward = 0
                        elif extraReward[trial] == 'right':
                            reward = 0 + extraRewardVal
                        elif extraReward[trial] == 'none':
                            reward = 0
    			
            else: #  cases with errors
                if currentStim>0 and (action[trial,i] == 'left'): # erroneous choice
                    if extraReward[trial] == 'left':
                        reward = 0 + extraRewardVal # Add extra reward in our case the laser, if it is a laser trial
                    elif extraReward[trial] == 'right':
                        reward = 0
                    elif extraReward[trial] == 'none':
                        reward = 0
        			
                elif currentStim<0 and (action[trial,i] == 'right'): # erroneous choice
                    if extraReward[trial] == 'left':
                        reward = 0
                    elif extraReward[trial] == 'right': # Add extra reward in our case the laser, if it is a laser trial
                        reward = 0 + extraRewardVal
                    elif extraReward[trial] == 'none':
                        reward = 0
    			
    		
    		
    		# calculate the prediction error, and update Q values
            if action[trial,i] == 'left':
                predictionError[trial, i] = reward - QL[trial,i] 
                QLL	= QLL + learningRate * predictionError[trial,i] * Belief_L #Belief is left, left action
                QRL	= QRL + learningRate * predictionError[trial,i] * Belief_R #Belief is left, right action
    			
            else:   # right
                predictionError[trial, i] = reward - QR[trial,i]
                QLR		= QLR + learningRate * predictionError[trial,i] * Belief_L #Belief is left, right action
                QRR		= QRR + learningRate * predictionError[trial,i] * Belief_R #Belief is right, right action
    
    actionLeft = (action == 'left').astype(int)
    actionRight = (action == 'right').astype(int)
    meanActionNum = np.mean(actionRight,1)
    #meanActionNum = np.mean(actionRight-actionLeft,1)
    
    
    # set output
    output=pd.DataFrame()
    output['action'] = meanActionNum
    output['QL'] = np.mean(QL,1)
    output['QR'] = np.mean(QR,1)
    output['action_sd']  = sdActionNum
    output['pL1'] = np.mean(pL1,1)
    output['pR1'] = np.mean(pR1,1)
    output.loc[(output['action'] == 0), 'pChoice'] = output.loc[(output['action'] == 0), 'pL1']
    output.loc[(output['action'] == 1), 'pChoice'] = output.loc[(output['action'] == 1), 'pR1']
    return (output)

def simulatePOMDP(params, model_data):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    
    stimTrials = model_data['stimTrials']
    extraReward = model_data['extraRewardTrials']
    choices = model_data['choice']
    
    # set run numbers
    iterN = 1  # model values are averaged over iterations (odd number)
    trialN = len(stimTrials)
    
    
    # initialise variables, for speed
    action = np.empty([trialN,iterN],dtype=object)
    QL = np.zeros([trialN,iterN])
    QR = np.zeros([trialN,iterN])
    pL1 = np.zeros([trialN,iterN])
    pR1 = np.zeros([trialN,iterN])
    predictionError = np.zeros([trialN,iterN])


    for i in range(iterN):
        # initalise Q values for each iteration
        QLL = 1
        QRR = 1
        QLR = 0
        QRL = 0
        # start model
        for trial in range(trialN):
            # set contrast
            currentStim = stimTrials[trial]
    		# calculate belief
            Belief_L, Belief_R = true_stim_posterior(currentStim)
            Belief_R = 1 - Belief_L
            # initialise Q values for this iteration
            QL[trial,i] = Belief_L*QLL + Belief_R*QRL # need explanation for 4 q values
            QR[trial,i] = Belief_L*QLR + Belief_R*QRR
            
            #switiching for a softmax
            pL1[trial,i] , pR1[trial,i]  = q_softmax(QL[trial,i], QR[trial,i], Temperature)
            
            #Exception for when softmax returns 0
            if (np.isnan(pL1[trial,i]) | np.isnan(pR1[trial,i]) ) :
                action[trial,i] = np.random.choice(['left',        'right'], p=[0.5, 0.5])
            # Make a choice
            else:
                action[trial,i] = np.random.choice(['left', 'right'], p=[pL1[trial,i], pR1[trial,i]])
       
    		# trial reward for action chosen by agent
            if currentStim<0 and (action[trial,i] == 'left'): # If action and stim pair, reward
                if extraReward[trial] == 'left':
                    reward = 1 + extraRewardVal # Add extra reward in our case the laser, if it is a laser trial
                elif extraReward[trial] == 'right':
                    reward = 1
                elif extraReward[trial] == 'none':
                    reward = 1
    			
            elif currentStim>0 and (action[trial,i] == 'right'): # If action and stim pair, reward
                
                if extraReward[trial] == 'left':
                    reward = 1
                elif extraReward[trial] == 'right': # Add extra reward in our case the laser, if it is a laser trial
                    reward = 1 + extraRewardVal
                elif extraReward[trial] == 'none':
                    reward = 1
    			
            elif currentStim==0:
                if np.random.rand() > 0.5:	# no evidence up to chance, if correct
                    if action[trial,i] == 'left':      
                        if extraReward[trial] == 'left':
                            reward = 1 + extraRewardVal
                        elif extraReward[trial] == 'right':
                            reward = 1
                        elif extraReward[trial] == 'none':
                            reward = 1
    					
                    elif action[trial,i] =='right':
                        if extraReward[trial] == 'left':
                            reward = 1
                        elif extraReward[trial] == 'right':
                            reward = 1 + extraRewardVal
                        elif extraReward[trial] == 'none':
                            reward = 1
    				
                else: # if 0 evidence incorrect
                    if action[trial,i] == 'left':      
                        if extraReward[trial] == 'left':
                            reward = 0 + extraRewardVal
                        elif extraReward[trial] == 'right':
                            reward = 0
                        elif extraReward[trial] == 'none':
                            reward = 0
    					
                    elif action[trial,i] =='right':
                        if extraReward[trial] == 'left':
                            reward = 0
                        elif extraReward[trial] == 'right':
                            reward = 0 + extraRewardVal
                        elif extraReward[trial] == 'none':
                            reward = 0
    			
            else: #  cases with errors
                if currentStim>0 and (action[trial,i] == 'left'): # erroneous choice
                    if extraReward[trial] == 'left':
                        reward = 0 + extraRewardVal # Add extra reward in our case the laser, if it is a laser trial
                    elif extraReward[trial] == 'right':
                        reward = 0
                    elif extraReward[trial] == 'none':
                        reward = 0
        			
                elif currentStim<0 and (action[trial,i] == 'right'): # erroneous choice
                    if extraReward[trial] == 'left':
                        reward = 0
                    elif extraReward[trial] == 'right': # Add extra reward in our case the laser, if it is a laser trial
                        reward = 0 + extraRewardVal
                    elif extraReward[trial] == 'none':
                        reward = 0
    					
    		# calculate the prediction error, and update Q values
            if action[trial,i] == 'left':
                predictionError[trial, i] = reward - QL[trial,i] 
                QLL	= QLL + learningRate * predictionError[trial,i] * Belief_L #Belief is left, left action
                QRL	= QRL + learningRate * predictionError[trial,i] * Belief_R #Belief is left, right action
    			
            else:   # right
                predictionError[trial, i] = reward - QR[trial,i]
                QLR		= QLR + learningRate * predictionError[trial,i] * Belief_L #Belief is left, right action
                QRR		= QRR + learningRate * predictionError[trial,i] * Belief_R #Belief is right, right action
    
    actionLeft = (action == 'left').astype(int)
    actionRight = (action == 'right').astype(int)
    meanActionNum = np.mean(actionRight,1)
    #meanActionNum = np.mean(actionRight-actionLeft,1)
    
    
    # set output
    output=pd.DataFrame()
    output['action'] = meanActionNum
    output['QL'] = np.mean(QL,1)
    output['QR'] = np.mean(QR,1)
    output['action_sd']  = sdActionNum
    output['pL1'] = np.mean(pL1,1)
    output['pR1'] = np.mean(pR1,1)
    output.loc[(output['action'] == 0), 'pChoice'] = output.loc[(output['action'] == 0), 'pL1']
    output.loc[(output['action'] == 1), 'pChoice'] = output.loc[(output['action'] == 1), 'pR1']
    return (output)

########################################################################
#########################Yotam version##################################
########################################################################

# Given all of the Q values (a matrix of size num_contrasts x 2), compute the overall Q_left and Q_right 
# (i.e., the overall value of choosing left or right) given the perceived stimulus
def compute_QL_QR(Q, unique_contrasts):
	Q_L = 0
	Q_R = 0
​
	# We compute Q_L, Q_R as in the Cell paper, by taking the weighted average of the various perceived stimuli,
	# according to the probability that they were perceived
	for contrast in unique_contrasts:
		Q_L += true_stim_posterior(contrast) * Q[contrast, 0]
		Q_R += true_stim_posterior(contrast) * Q[contrast, 1]
​
	return Q_L, Q_R
​
# Given Q_left, Q_right, and the softmax inverse temperature beta, compute the probability of
# turning left or right
def q_softmax(Q_L, Q_R, beta):
	p = [np.exp(Q_L / beta), np.exp(Q_R / beta)]
	p = p / (np.exp(Q_L / beta) + np.exp(Q_R / beta))

	return p
​
# Compute the log likelihood of a given trial under model parameters params,
# and an underlying set of Q values, Q
def trial_log_likelihood(trial_data, params, Q, unique_contrasts):
	# Get relevant parameters
	contrast, actual_choice, reward, opto = trial_data
	learning_rate, extraVal, beliefSTD, beta = params
​
	# Compute the log-likelihood of the actual mouse choice
	Q_L, Q_R = compute_QL_QR(Q, unique_contrasts)
	choice_dist = q_softmax(Q_L, Q_R, beta)
	LL = np.log(choice_dist[actual_choice])
​
	# Update Q-values according to received reward
	for contrast in unique_contrasts:
		if actual_choice == 0:
			Q_chosen = Q_L
		else:
			Q_chosen = Q_R
​
		Q[true_contrast, actual_choice] +=  \
        true_stim_posterior(contrast) \
            * learning_rate * ((reward + opto) - Q_chosen)
​
	return LL, Q
​
# Compute the log likelihood of all the ocncatenated trials
def sesssion_log_likelihood(psy_df, params):
	learning_rate, beliefSTD, extraVal, inverse_temperature = params
    unique_contrasts = np.sort(psy_df['stimTrials'].unique())
	LL = 0
	Q = np.zeros((len(unique_contrasts), 2))
    model_data = model_data['stimTrials', 'choice', 'reward']
    
	for trial in model_data:
        trial_data =  [trial['stimTrials'], trial['choice'], trial['choice']]
		trial_LL, newQ = trial_log_likelihood(trial_data, params, Q, unique_contrasts)
		LL += trial_LL
​
	return LL

def true_stim_posterior(contrast):
    # Prior is (1/7) , likelihood is X ~ N(truestim, sensory_noise)
    # multiplication gives us a posterior = X ~ N(prior*percived_stim,sensory_noise * truestim)
    posteriror_left = \
        norm.cdf(x = 0, loc=contrast * (1/7), scale=beliefNoiseSTD * (1/7))
    posterior_right = 1- posteriror_left
    return posteriror_left, posterior_right
    
# Optimize
def optimizer(data):
	return scipy.optimize.minimize(-session_log_likelihood(data))






