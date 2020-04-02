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
model_data = pd.DataFrame({'stimTrials': signed_contrasts[:,0],'extraRewardTrials': opto_side[:,0]})

# RUN


# Set variables (for initiliaziton)
learningRate = 0.15     # learning rate
extraRewardVal = -1    # value for opto
beliefNoiseSTD = 0.075			# noise in belief
Temperature = 1 #Temperature for softmax


params = [learningRate, extraRewardVal, beliefNoiseSTD, Temperature]



#Least squares
result1 = minimize(session_log_likelihood, params, options ={'disp':True})


params = result1['x']
output = RunPOMDP(params, model_data)

#Plt
plot_action_opto_block(output,psy_df)







###########################################################
##########################Functions########################
###########################################################



def RunPOMDP(params, model_data):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    
    stimTrials = model_data['stimTrials']
    extraReward = model_data['extraRewardTrials']
    
    
    # set run numbers
    iterN = 1  # model values are averaged over iterations (odd number)
    trialN = len(stimTrials)
    
    
    # initialise variables, for speed
    action = np.empty([trialN,iterN],dtype=object)
    QL = np.zeros([trialN,iterN])
    QR = np.zeros([trialN,iterN])
    pL = np.zeros([trialN,iterN])
    pR = np.zeros([trialN,iterN])
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
    		# add sensory noise
            stim_withnoise  = currentStim + beliefNoiseSTD * np.random.randn()
    		# calculate belief
            Belief_L = norm.cdf(x=0, loc=stim_withnoise, scale=beliefNoiseSTD) # Probability that is less than 0 aka left (cdf evaluated at 0)
            Belief_R = 1 - Belief_L
            # initialise Q values for this iteration
            QL[trial,i] = Belief_L*QLL + Belief_R*QRL # need explanation for 4 q values
            QR[trial,i] = Belief_L*QLR + Belief_R*QRR 
    		
            ##### action <-- max(QL,QR)
            
            #if QL[trial,i] > QR[trial,i]: # Action with higher action value (Q) is chosen
            #    action[trial,i] = 'left'
    		#	
            #elif QL[trial,i] < QR[trial,i]:
            #    action[trial,i] = 'right'
            #    
            #else:
            #    if np.random.rand() >= 0.5: # If Q is the same random choice
            #        action[trial,i] = 'right'
            #    else:
            #        action[trial,i] = 'left'
    		
            #####
            
            #switiching for a softmax
            
            #Calculate softmax
            pL[:,i]= softmax(QL[:,i]/T)
            pR[:,i]= softmax(QR[:,i]/T)
            #Choose action
            pL1 = pL[trial,i]/(pL[trial,i]+pR[trial,i])
            pR1 = pR[trial,i]/(pL[trial,i]+pR[trial,i])
            
            if (np.isnan(pL1) | np.isnan(pR1) ) :
                action[trial,i] = np.random.choice(['left', 'right'], p=[0.5, 0.5])
            
            else:
                action[trial,i] = np.random.choice(['left', 'right'], p=[pL1, pR1])
            
            
            #action[trial,i] = np.random.choice(['left', 'right'],
                                               #p=[pL[trial,i], 1-pL[trial,i]])
            
            
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
    sdActionNum = np.std(actionRight-actionLeft,1)
    
    
    # set output
    output=pd.DataFrame()
    output['action'] = meanActionNum
    output['QL'] = np.mean(QL,1)
    output['QR'] = np.mean(QR,1)
    output['action_sd']  = sdActionNum
    output['pL'] = np.mean(pL1,1)
    output['pR'] = np.mean(pR1,1)
    
    
    return (output)

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
    


def session_log_likelihood(params, trial_data = model_data,):
    try:
        output = RunPOMDP(params, model_data)
        negLL = -np.sum(stats.norm.logpdf(psy_df['right_choice'], loc=output['pR1']))\
                                           #scale = params[2]))
    except:
        print(params)
    return negLL


