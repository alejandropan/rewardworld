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


# Set variables 
learningRate    = 0.35     # learning rate
extraRewardVal	= -4.0     # value for opto
beliefNoiseSTD	= 0.18			# noise in belief


params = [learningRate, extraRewardVal, beliefNoiseSTD]

# Get sesion information

opto_side_num = psy_df['choice'] * psy_df['opto.npy']
opto_side = opto_side_num.loc[opto_side_num == 1] = 'left'
opto_side = opto_side_num.loc[opto_side_num == -1] = 'right'

model_data = pd.DataFrame(data = [psy_df['signed_contrast'],opto_side],
                  columns=['stimTrials', 'extraRewardTrials'])

# RUN

output = RunPOMDP(model_data,params);



###########################################################

def RunPOMDP(model_data,params):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    
    stimTrials = model_data['stimTrials']
    extraReward = model_data['extraRewardTrials']
    
    
    # set run numbers
    iterN = 21  # model values are averaged over iterations (odd number)
    trialN = np.length(stimTrials)
    
    
    # initialise variables, for speed
    action = np.empty([trialN,iterN],dtype=object)
    QL = np.zeros([trialN,iterN])
    QR = np.zeros([trialN,iterN])
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
    		currentStim = stimTrials[trial];
    		
    		# add sensory noise
    		stim_withnoise  = currentStim + beliefNoiseSTD * np.random.randn()
    		
    		# calculate belief
    		Belief_L = norm.cdf(x=0, loc=stim_withnoise, scale=beliefNoiseSTD)
    		Belief_R = 1 - Belief_L
    		
    		# initialise Q values for this iteration
    		QL[trial,i] = Belief_L*QLL + Belief_R*QRL
    		QR[trial,i] = Belief_L*QLR + Belief_R*QRR
    		
    		# action <-- max(QL,QR)
    		if QL[trial,i] > QR[trial,i]:
    			action[trial,i] = 'left'
    			
    		elif QL(trial,i) < QR[trial,i]:
    			action[trial,i] = 'right'
                
    		else:
    			if np.random.rand() >= 0.5:
    				action[trial,i] = 'right'
    			else:
    				action[trial,i] = 'left'
    		
    		# trial reward for action chosen by agent
    		if currentStim<0 and (action[trial,i] == 'left'):
                if extraReward[trial] == 'left':
                    reward = 1 + extraRewardVal
                elif extraReward[trial] == 'right':
                    reward = 1
				elif extraReward[trial] == 'none':
    				reward = 1
    			
    		elif currentStim>0 and (action[trial,i] == 'right'):
                
                if extraReward[trial] == 'left':
                    reward = 1
                elif extraReward[trial] == 'right':
                    reward = 1 + extraRewardVal
				elif extraReward[trial] == 'none':
    				reward = 1
    			
    		elif currentStim==0:
    			if np.random.rand() > 0.5:	
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
    				
    			else:
    				reward = 0
    			
    		else:
    			reward = 0
    			
    		
    		
    		# calculate the prediction error, and update Q values
    		if action[trial,i] == 'left':
    			predictionError[trial, i] = reward - QL[trial,i]
    			QLL	= QLL + learningRate * predictionError[trial,i] * Belief_L
    			QRL	= QRL + learningRate * predictionError[trial,i] * Belief_R
    			
    		else:   # right
    			predictionError[trial, i] = reward - QR[trial,i]
    			QLR		= QLR + learningRate * predictionError[trial,i] * Belief_L
    			QRR		= QRR + learningRate * predictionError[trial,i] * Belief_R
    
    actionLeft = (action == 'left').astype(int)
    actionRight = (action == 'right').astype(int)
    meanActionNum = np.mean(actionRight-actionLeft,1)
    
    # set output
    output=pd.DataFrame()
    output['action'] = meanActionNum
    output['QL'] = mean(QL,1)
    output['QR'] = mean(QR,1)
    
    
    return output
