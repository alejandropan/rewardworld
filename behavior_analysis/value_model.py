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
from rew_alf.npy2pd import *
import time
from rew_alf.data_organizers import *
import numpy as np, pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.optimize import fmin
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel
from scipy.optimize import minimize
from scipy.special import softmax
from behavior_analysis.value_model_figures import *



# Notes - 2 iterations nphr no stay -8.424181138661481
# Right choices are -1, left choices 1
# Load the data
psy = load_behavior_data_from_root('/Volumes/witten/Alex/recordings_march_2020_dop/')

#****************************************************************************#
#****************************************************************************#
########################### Optimize models ##################################
#****************************************************************************#
#****************************************************************************#


##############################################################################
###########################  Pooling mice ####################################
##############################################################################

model_data_chr2, simulate_data_chr2  = \
    psy_df_to_Q_learning_model_format(psy, virus = 'chr2')
    
model_data_nphr, simulate_data_nphr  = \
    psy_df_to_Q_learning_model_format(psy, virus = 'nphr')

best_nphr, best_nphr_NLL = optimizer_runPOMDP(model_data_nphr, num_fits = 10, 
                                initial_guess=[0.1, -1, 0.5, 0.6])
best_chr2, best_chr2_NLL = optimizer_runPOMDP(model_data_chr2, num_fits = 10,
                                initial_guess=[0.1, 1, 0.5, 0.6])


best_nphr_stay, best_nphr_NLL_stay = optimizer_runPOMDP_stay(model_data_nphr, 
                                num_fits = 10, initial_guess=[0.1, -1, 0.5, 0.6, 0])
best_chr2_stay, best_chr2_NLL_stay = optimizer_runPOMDP_stay(model_data_chr2, 
                                num_fits = 10, initial_guess=[0.1, 1, 0.5, 0.6, 0])


# Get Q values
output_chr2 = RunPOMDP(best_chr2, model_data_chr2)
output_nphr = RunPOMDP(best_nphr, model_data_nphr)
output_chr2_stay = RunPOMDP_stay(best_chr2_stay, model_data_chr2)
output_nphr_stay = RunPOMDP_stay(best_nphr_stay, model_data_nphr)

# Add Q to dataframe

psy.loc[psy['virus'] == 'chr2', 'QL'] = output_chr2_stay['QL'].to_numpy()
psy.loc[psy['virus'] == 'chr2', 'QR'] = output_chr2_stay['QR'].to_numpy()
psy.loc[psy['virus'] == 'chr2', 'pChoice'] = output_chr2_stay['pChoice'].to_numpy()
 
psy.loc[psy['virus'] == 'nphr', 'QL'] = output_nphr_stay['QL'].to_numpy()
psy.loc[psy['virus'] == 'nphr', 'QR'] = output_nphr_stay['QR'].to_numpy()
psy.loc[psy['virus'] == 'nphr', 'pChoice'] = output_nphr_stay['pChoice'].to_numpy()



##############################################################################
############################ Individual mice #################################
##############################################################################

best_nphr_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),6])
best_nphr_NLL_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])


for i, mouse in enumerate(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()): 
    model_data_nphr, simulate_data_nphr  = \
        psy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse], 
                                                  virus = 'nphr')
    best_nphr_individual[i,:], best_nphr_NLL_individual[i,:] = optimizer_runPOMDP_stay(model_data_nphr, num_fits = 2, 
                                initial_guess=[0.1, -1, 0.5, 0.6, 0.5, 0.5])
    
    output_nphr = RunPOMDP_stay(best_nphr_individual[i,:], model_data_nphr)
    psy.loc[psy['mouse_name'] == mouse, 'QL'] = output_nphr['QL'].to_numpy()
    psy.loc[psy['mouse_name'] == mouse, 'QR'] = output_nphr['QR'].to_numpy()
    psy.loc[psy['mouse_name'] == mouse, 'pChoice'] = output_nphr['pChoice'].to_numpy()
        
for mouse in psy.loc[psy['virus'] == 'chr2', 'mouse_name'].unique(): 
    model_data_chr2, simulate_data_chr2  = \
        psy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse], 
                                                  virus = 'chr2')
    best_chr2, best_chr2_NLL = optimizer_runPOMDP(model_data_chr2, num_fits = 2, 
                                initial_guess=[0.1, -1, 0.5, 0.6])
    output_chr2 = RunPOMDP(best_chr2, model_data_chr2)
    psy.loc[psy['mouse_name'] == mouse, 'QL'] = output_chr2['QL'].to_numpy()
    psy.loc[psy['mouse_name'] == mouse, 'QR'] = output_chr2['QR'].to_numpy()
    psy.loc[psy['mouse_name'] == mouse, 'pChoice'] = output_chr2['pChoice'].to_numpy()

##############################################################################
##############################################################################



#Plot 
simulate_chr2 = simulatePOMDP(best_chr2, simulate_data_chr2)
plot_action_opto_block(simulate_chr2,psy.loc[psy['virus'] == 'chr2'].reset_index())
simulate_nphr = simulatePOMDP(best_nphr, simulate_data_nphr)
plot_action_opto_block(simulate_nphr,psy.loc[psy['virus'] == 'nphr'].reset_index())

simulate_nphr = simulatePOMDP_stay(best_nphr_stay, simulate_data_nphr)
plot_action_opto_block(simulate_nphr,psy.loc[psy['virus'] == 'nphr'].reset_index())

simulate_chr2 = simulatePOMDP_stay(best_chr2_stay, simulate_data_chr2)
plot_action_opto_block(simulate_chr2,psy.loc[psy['virus'] == 'chr2'].reset_index())

###########################################################
##########################Functions########################
###########################################################



def session_log_likelihood1(params, model_data):
        output = RunPOMDP(params, model_data)
        negLL = -np.log(np.sum(output['pChoice']))
        return negLL
    
def session_log_likelihood_stay(params, model_data):
        output = RunPOMDP_stay(params, model_data)
        negLL = -np.log(np.sum(output['pChoice']))
        return negLL


def print_callback(xs):
    """
    Callback called after every iteration.

    xs is the estimated location of the optimum.
    """
    
    return(print (xs))

def optimizer_runPOMDP(data, num_fits = 5, initial_guess=[0.1, -1, 0.5, 0.6]):
    # Accounting variables
    best_NLL = np.Inf
    best_x = [None, None, None, None]

	# Do our fit with several different initializations
    for i in range(num_fits):
        print('Starting fit %d' % i)
        # For every fit other than the first, construct a new initial guess
        if i != 0:
            lr_guess = np.random.uniform(0, 1)
            extraVal_guess = np.random.normal()
            beliefSTD_guess = np.random.uniform(0.005, 5)
            beta_guess = np.random.uniform(0.01, 5)
            initial_guess = [lr_guess, extraVal_guess, beliefSTD_guess, beta_guess]
		# Run the fit
        res = minimize(session_log_likelihood1, initial_guess, 
                       args= data, method='L-BFGS-B', 
                       bounds=[(0, 1), (-1, 1), (0.01, 1), 
                               (0.01, 1)])
		# If this fit is better than the previous best, remember it, otherwise toss
        if res.fun <= best_NLL:
            best_NLL = res.fun
            best_x = res.x
    return best_x, best_NLL

def optimizer_runPOMDP_stay(data, num_fits = 5, initial_guess=[0.1, -1, 0.5, 0.6, 0.5,0.5]):
    # Accounting variables
    best_NLL = np.Inf
    best_x = [None, None, None, None, None]

	# Do our fit with several different initializations
    for i in range(num_fits):
        print('Starting fit %d' % i)
        # For every fit other than the first, construct a new initial guess
        if i != 0:
            lr_guess = np.random.uniform(0, 1)
            extraVal_guess = np.random.normal()
            beliefSTD_guess = np.random.uniform(0.005, 5)
            beta_guess = np.random.uniform(0.01, 5)
            stay_guess = np.random.uniform(0,100)
            bias_guess = np.random.uniform(0,100)
            initial_guess = [lr_guess, extraVal_guess, beliefSTD_guess, beta_guess, stay_guess, bias_guess]
		# Run the fit
        res = minimize(session_log_likelihood_stay, initial_guess, 
                       args= data, method='L-BFGS-B', 
                       bounds=[(None, None), (None, None), (0.01, None), 
                               (None, None), (None,None), (None,None)])
		# If this fit is better than the previous best, remember it, otherwise toss
        if res.fun <= best_NLL:
            best_NLL = res.fun
            best_x = res.x
    return best_x, best_NLL

def cross_validated_optimazation(data,num_fits = 5, initial_guess=[0.1, -1, 0.5, 0.6] ):
    cvv = np.zeros(len(data['ses'].unique()))
    cvv = np.nan
              
    for x, i in enumerate(data['ses'].unique()):
        #Train in all session but one, test on that session
        #train
        train_data = data.loc[data['ses']!=i]
        best_x, _ = optimizer_runPOMDP(train_data, num_fits = num_fits, 
                           initial_guess=initial_guess)
        #test
        test_data = data.loc[data['ses']==i]
        best_x, _ = optimizer_runPOMDP(test_data, num_fits = num_fits, 
                           initial_guess=initial_guess)
        cvv[x] = session_log_likelihood1(best_x, test_data)
    return cvv, best_x


###########################################################

# Calculate Q values
def RunPOMDP(params, model_data):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    
    model_data = model_data.drop(model_data[model_data.choice == 0].index)    
    reward = model_data['reward'].to_numpy()
    choices = model_data['choice'].to_numpy()
    stimTrials = model_data['stimTrials'].to_numpy()
    extraReward = model_data['extraRewardTrials'].to_numpy()
    # set run numbers
    trialN = len(stimTrials)
    # initialise variables, for speed
    all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.    ,  0.0625, 0.125 , 0.25  ])
    QL = np.zeros([trialN])
    QR = np.zeros([trialN])
    pL1 = np.zeros([trialN])
    pR1 = np.zeros([trialN])
    pChoice = np.zeros([trialN])
    predictionError = np.zeros([trialN])
    
    # initalise Q values for each iteration
    Q = np.zeros( [7, 2])
    # start model
    for trial in range(trialN):
            # set contrast
            currentStim = stimTrials[trial]
    		# calculate belief
            contrast_posterior = true_stim_posterior(currentStim, all_contrasts, beliefNoiseSTD)
            # calculate Q values
            QL[trial], QR[trial]  = compute_QL_QR(Q, currentStim, contrast_posterior)

            #switiching for a softmax
            pL1[trial] , pR1[trial]  = q_softmax(QL[trial], QR[trial], Temperature)
            
            if choices[trial] == -1:
                pChoice[trial] = pR1[trial]
                Q_chosen = QR[trial]
                trial_choice  =  1
            if choices[trial] == 1:
                pChoice[trial] = pL1[trial]
                Q_chosen = QL[trial]
                trial_choice  =  0
            
            #Laser modulatation (reward is already 1 by default)
            if extraReward[trial] == 1:
                rew = reward[trial] + extraRewardVal
            else:
                rew = reward[trial]
            
            # calculate the prediction error, and update Q values
            for j in range(len(all_contrasts)):
                Q[j, trial_choice] += contrast_posterior[j] * learningRate * (rew - Q_chosen)
    
    # set output
    output=pd.DataFrame()
    output['QL'] = QL
    output['QR'] = QR
    output['pL1'] = pL1
    output['pR1'] = pR1
    output['pChoice'] = pChoice
    return output

def RunPOMDP_stay(params, model_data):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    stay = params[4]
    bias = params[5]
    
    model_data = model_data.drop(model_data[model_data.choice == 0].index)    
    reward = model_data['reward'].to_numpy()
    choices = model_data['choice'].to_numpy()
    stimTrials = model_data['stimTrials'].to_numpy()
    extraReward = model_data['extraRewardTrials'].to_numpy()
    # set run numbers
    trialN = len(stimTrials)
    # initialise variables, for speed
    all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.,  0.0625, 0.125 , 0.25  ])
    QL = np.zeros([trialN])
    QR = np.zeros([trialN])
    pL1 = np.zeros([trialN])
    pR1 = np.zeros([trialN])
    pChoice = np.zeros([trialN])
    predictionError = np.zeros([trialN])
    
    # initalise Q values for each iteration
    Q = np.zeros( [7, 2])
    # start model
    for trial in range(trialN):
            # set contrast
            currentStim = stimTrials[trial]
    		# calculate belief
            contrast_posterior = true_stim_posterior(currentStim, all_contrasts, beliefNoiseSTD)
            # calculate Q values
            QL[trial], QR[trial]  = compute_QL_QR(Q, currentStim, contrast_posterior)
            # check previous choice and calculate stay parameter
            l_stay, r_stay = calculate_stay(stay, trial, choices)
            # switiching for a softmax
            pL1[trial] , pR1[trial]  = q_softmax_stay(QL[trial], QR[trial], Temperature, 
                                                      l_stay, r_stay, bias)
            
            if choices[trial] == -1:
                pChoice[trial] = pR1[trial]
                Q_chosen = QR[trial]
                trial_choice  =  1
            if choices[trial] == 1:
                pChoice[trial] = pL1[trial]
                Q_chosen = QL[trial]
                trial_choice  =  0
            
            #Laser modulatation (reward is already 1 by default)
            if extraReward[trial] == 1:
                rew = reward[trial] + extraRewardVal
            else:
                rew = reward[trial]
            
            # calculate the prediction error, and update Q values
            for j in range(len(all_contrasts)):
                Q[j, trial_choice] += contrast_posterior[j] * learningRate * (rew - Q_chosen)
    
    # set output
    output=pd.DataFrame()
    output['QL'] = QL
    output['QR'] = QR
    output['pL1'] = pL1
    output['pR1'] = pR1
    output['pChoice'] = pChoice
    output['singed_contrast'] = stimTrials
    return output



def simulatePOMDP(params, model_data):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    
    stimTrials = model_data['stimTrials']
    extraReward = model_data['extraRewardTrials']
    #choices = model_data['choice']
    
    # set run numbers
    iterN = 1  # model values are averaged over iterations (odd number)
    trialN = len(stimTrials)
    
    all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.,  0.0625, 0.125 , 0.25  ])
    # initialise variables, for speed
    action = np.empty([trialN,iterN],dtype=object)
    QL = np.zeros([trialN,iterN])
    QR = np.zeros([trialN,iterN])
    pL1 = np.zeros([trialN,iterN])
    pR1 = np.zeros([trialN,iterN])
    predictionError = np.zeros([trialN,iterN])


    for i in range(iterN):
        # initalise Q values for each iteration
        Q = np.zeros( [7, 2])
        # start model
        for trial in range(trialN):
            # set contrast
            currentStim = stimTrials[trial]
    		# calculate belief
            contrast_posterior = true_stim_posterior(currentStim, all_contrasts, beliefNoiseSTD)
            # calculate Q values
            QL[trial,i], QR[trial,i]  = compute_QL_QR(Q, currentStim, contrast_posterior)

            #switiching for a softmax
            pL1[trial,i] , pR1[trial,i]  = q_softmax(QL[trial,i], QR[trial,i], Temperature)
            
            #Exception for when softmax returns 0
            if (np.isnan(pL1[trial,i]) | np.isnan(pR1[trial,i])):
                action[trial,i] = np.random.choice(['left', 'right'], p=[0.5, 0.5])
            # Make a choice
            else:
                #action[trial,i] = np.random.choice(['left', 'right'], p=[pL1[trial,i], pR1[trial,i]])
                if QL[trial,i] > QR[trial,i]:
                    action[trial,i] = 'left'
                if QL[trial,i] < QR[trial,i]:
                    action[trial,i] = 'right'
                if QL[trial,i] == QR[trial,i]:
                    action[trial,i] = np.random.choice(['left', 'right'], p=[0.5, 0.5])
            if (action[trial,i] == 'left'):
                Q_chosen = QL[trial,i]
                trial_choice  =  0
            else:
                Q_chosen = QR[trial,i]
                trial_choice  =  1
            
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
            for j in range(len(all_contrasts)):
                Q[j, trial_choice] += contrast_posterior[j] * learningRate * ((reward) - Q_chosen)
                
    actionLeft = (action == 'left').astype(int)
    actionRight = (action == 'right').astype(int)
    meanActionNum = np.mean(actionRight,1)
    #meanActionNum = np.mean(actionRight-actionLeft,1)
    
    
    # set output
    output=pd.DataFrame()
    output['action'] = meanActionNum
    output['contrast'] = model_data['stimTrials']
    output['QL'] = np.mean(QL,1)
    output['QR'] = np.mean(QR,1)
    output['pL1'] = np.mean(pL1,1)
    output['pR1'] = np.mean(pR1,1)
    return (output)

def simulatePOMDP_stay(params, model_data):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    stay = params[4]
    bias = params[5]
    
    stimTrials = model_data['stimTrials']
    extraReward = model_data['extraRewardTrials']
    #choices = model_data['choice']
    
    # set run numbers
    iterN = 10  # model values are averaged over iterations (odd number)
    trialN = len(stimTrials)
    
    all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.    ,  0.0625, 0.125 , 0.25  ])
    # initialise variables, for speed
    action = np.empty([trialN,iterN],dtype=object)
    QL = np.zeros([trialN,iterN])
    QR = np.zeros([trialN,iterN])
    pL1 = np.zeros([trialN,iterN])
    pR1 = np.zeros([trialN,iterN])
    predictionError = np.zeros([trialN,iterN])


    for i in range(iterN):
        # initalise Q values for each iteration
        Q = np.zeros( [7, 2])
        # start model
        for trial in range(trialN):
            # set contrast
            currentStim = stimTrials[trial]
    		# calculate belief
            contrast_posterior = true_stim_posterior(currentStim, all_contrasts, beliefNoiseSTD)
            # calculate Q values
            QL[trial,i], QR[trial,i]  = compute_QL_QR(Q, currentStim, contrast_posterior)

            # check previous choice and calculate stay parameter
            l_stay, r_stay = calculate_stay(stay, trial, action[:,i], simulate = True)
            # switiching for a softmax
            pL1[trial,i] , pR1[trial,i]  = q_softmax_stay(QL[trial,i], QR[trial,i], Temperature, l_stay, r_stay,bias)
            
            #Exception for when softmax returns 0
            if (np.isnan(pL1[trial,i]) | np.isnan(pR1[trial,i])):
                action[trial,i] = np.random.choice(['left', 'right'], p=[0.5, 0.5])
            # Make a choice
            else:
                action[trial,i] = np.random.choice(['left', 'right'], p=[pL1[trial,i], pR1[trial,i]])
            
            
            if (action[trial,i] == 'left'):
                Q_chosen = QL[trial,i]
                trial_choice  =  0
            else:
                Q_chosen = QR[trial,i]
                trial_choice  =  1
            
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
            for j in range(len(all_contrasts)):
                Q[j, trial_choice] += contrast_posterior[j] * learningRate * ((reward) - Q_chosen)
    
    actionLeft = (action == 'left').astype(int)
    actionRight = (action == 'right').astype(int)
    meanActionNum = np.mean(actionRight,1)
    #meanActionNum = np.mean(actionRight-actionLeft,1)
    
    
    # set output
    output=pd.DataFrame()
    output['action'] = meanActionNum
    output['contrast'] = model_data['stimTrials']
    output['QL'] = np.mean(QL,1)
    output['QR'] = np.mean(QR,1)
    output['pL1'] = np.mean(pL1,1)
    output['pR1'] = np.mean(pR1,1)
    return (output)
# Given Q_left, Q_right, and the softmax inverse temperature beta, compute the probability of
# turning left or right

def calculate_stay(stay, trial, choices, simulate = False):
    if simulate == True: # For simulation
        choices_int = np.empty(len(choices))
        choices_int[np.where(choices == 'right')] = -1
        choices_int[np.where(choices == 'left')] = 1
        choices = choices_int
    if trial == 0:
        l_stay = 0
        r_stay = 0
    else:
        l_stay = stay * ((choices[trial-1] == 1)*1)
        r_stay = stay * ((choices[trial-1] == -1)*1)
        
    return l_stay, r_stay

def q_softmax(Q_L, Q_R, beta):
	p = [np.exp(Q_L / beta), np.exp(Q_R / beta)]
	p /= np.sum(p)
	return p

def q_softmax_stay(Q_L, Q_R, beta, l_stay, r_stay, bias):
	p = [np.exp((Q_L / beta) + l_stay + bias), np.exp((Q_R / beta) + r_stay + bias)]
	p /= np.sum(p)
	return p

######################
def true_stim_posterior(true_contrast, all_contrasts, beliefSTD):
	# Compute distribution over perceived contrast
	# start_time = time.time()
	p_perceived = normal_pdf(all_contrasts, loc=true_contrast, scale=beliefSTD)
 
	start_time = time.time()
	mat = np.zeros((len(all_contrasts), len(all_contrasts))) # vectorized for speed, but this implements the sum above
	for idx, perceived_contrast in enumerate(all_contrasts):
		mat[:, idx] = normal_pdf(all_contrasts, loc=perceived_contrast, scale=beliefSTD)
 
	posterior = (mat @ p_perceived) * (1/len(all_contrasts)) # multiply by prior
	posterior /= np.sum(posterior)
 
	return posterior


######################

def compute_QL_QR(Q, trial_contrast, contrast_posterior):
	Q_L = 0
	Q_R = 0

	# We compute Q_L, Q_R as in the Cell paper, by taking the weighted average of the various perceived stimuli,
	# according to the probability that they were perceived
	for i in range(len(contrast_posterior)):
		Q_L += contrast_posterior[i] * Q[i, 0]
		Q_R += contrast_posterior[i] * Q[i, 1]

	return Q_L, Q_R


def normal_ (x, loc, scale):
	factor = 1 / (np.sqrt(2 * np.pi) * scale)
	power = -0.5 * (((x - loc) / scale) ** 2)
 
	return factor * np.exp(power)