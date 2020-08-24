#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 18:24:04 2020

@author: alex
"""
def q_softmax(Q_L, Q_R, beta):
	p = [np.exp(Q_L / beta), np.exp(Q_R / beta)]
	p /= np.sum(p)
	return p

def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))

def aic(LL,n_param):
    # Calculates Akaike Information Criterion
    aic =  2*n_param - 2*LL
    return aic
    
def chi2_LLR(L1,L2):
    LR = likelihood_ratio(L1,L2)
    p = chi2.sf(LR, 1) # L2 has 1 DoF more than L1
    return p

def RunPOMDP(params, model_data):

    learningRate = params[0]
    extraRewardVal = params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    iter_n = 21
    model_data = model_data.drop(model_data[model_data.choice == 0].index)    
    reward = model_data['reward'].to_numpy()
    choices = model_data['choice'].to_numpy()
    stimTrials = model_data['stimTrials'].to_numpy()
    extraReward = model_data['extraRewardTrials'].to_numpy()
    # set run numbers
    trialN = len(stimTrials)
    # initialise variables, for speed
    QL = np.zeros([trialN, iter_n])
    QR = np.zeros([trialN, iter_n])
    pL1 = np.zeros([trialN, iter_n])
    pR1 = np.zeros([trialN, iter_n])
    pChoice = np.zeros([trialN, iter_n])
  
    start_time = time.time()
    # initalise Q values for each iteration
    for i in range(iter_n):
        QLL = 1
        QRL =0
        QLR = 0
        QRR = 1
        # start model
        for trial in range(trialN):
                # set contrast
                currentStim = stimTrials[trial]
        		# calculate belief
                perceived_stim = np.random.normal(loc = currentStim, scale = beliefNoiseSTD)
                belief_l = norm.cdf(0, loc = perceived_stim, scale = beliefNoiseSTD)
                belief_r = 1 - belief_l
                
                # Overal action values
                QL[trial, i] = QLL * belief_l + QRL * belief_r
                QR[trial, i] = QLR * belief_l + QRR * belief_r
                
                #switiching for a softmax
                pL1[trial,i] , pR1[trial,i]  = q_softmax(QL[trial, i], QR[trial, i], Temperature)
                
                if choices[trial] == -1:
                    pChoice[trial, i] = pR1[trial, i]
                    Q_chosen = QR[trial,i]

                else:
                    pChoice[trial, i] = pL1[trial, i]
                    Q_chosen = QL[trial,i]
                
                #Laser modulatation (reward is already 1 by default)
                if extraReward[trial] == 1:
                    rew = reward[trial] + extraRewardVal
                else:
                    rew = reward[trial]
                
                # calculate the prediction error, and update Q values
                if choices[trial] == 1:
                    QLL = QLL + learningRate * (rew - Q_chosen)
                    QRL = QRL + learningRate * (rew - Q_chosen)
                else:
                    QLR = QLR + learningRate * (rew - Q_chosen)
                    QRR = QRR + learningRate * (rew - Q_chosen)
    end_time = time.time()
    #print(end_time - start_time)
    
    # set output
    output=pd.DataFrame()
    output['QL'] = np.mean(QL,axis =1)
    output['QR'] = np.mean(QR,axis =1)
    output['pL1'] = np.mean(pL1,axis =1)
    output['pR1'] = np.mean(pR1,axis =1)
    output['pChoice'] = np.mean(pChoice,axis =1)
    output['signed_contrast'] = stimTrials
    return output

def simulatePOMDP(params, model_data):

    learningRate = params[0]
    extraRewardVal =  params[1]
    beliefNoiseSTD = params[2]
    Temperature = params[3]
    iter_n = 21
    model_data = model_data.drop(model_data[model_data.choice == 0].index)    
    stimTrials = model_data['stimTrials'].to_numpy()
    extraReward = model_data['extraRewardTrials'].to_numpy()
    # set run numbers
    trialN = len(stimTrials)
    # initialise variables, for speed
    QL = np.zeros([trialN, iter_n])
    QR = np.zeros([trialN, iter_n])
    pL1 = np.zeros([trialN, iter_n])
    pR1 = np.zeros([trialN, iter_n])
    trial_action = np.zeros([trialN, iter_n])
    action = np.empty([trialN,iter_n],dtype=object)
    pChoice = np.zeros([trialN, iter_n])
  
    start_time = time.time()
    # initalise Q values for each iteration
    for i in range(iter_n):
        QLL = 1
        QRL = 0
        QLR = 0
        QRR = 1
        # start model
        for trial in range(trialN):
                # set contrast
                currentStim = stimTrials[trial]
        		# calculate belief
                perceived_stim = np.random.normal(loc = currentStim, scale = beliefNoiseSTD)
                belief_l = norm.cdf(0, loc = perceived_stim, scale = beliefNoiseSTD)
                belief_r = 1 - belief_l
  
                # Overal action values
                QL[trial, i] = QLL * belief_l + QRL * belief_r
                QR[trial, i] = QLR * belief_l + QRR * belief_r
                
                #switiching for a softmax
                pL1[trial,i] , pR1[trial,i]  = q_softmax(QL[trial, i], QR[trial, i], Temperature)
            # Make a choice
                action[trial,i] = np.random.choice(['left', 'right'], p=[pL1[trial,i], pR1[trial,i]])
                
                if (action[trial,i] == 'left'):
                    Q_chosen = QL[trial,i]
                    trial_action[trial,i] = 1
                else:
                    Q_chosen = QR[trial,i]
                    trial_action[trial,i] = -1
            
    		# trial reward for action chosen by agent
                if currentStim<0 and (action[trial,i] == 'left'): # If action and stim pair, reward
                    if extraReward[trial] == 'left':
                        reward = 1 + extraRewardVal # Add extra reward in our case the laser, if it is a laser trial
                    elif extraReward[trial] == 'right':
                        reward = 1
                    else:
                        reward = 1
        			
                elif currentStim>0 and (action[trial,i] == 'right'): # If action and stim pair, reward
                    if extraReward[trial] == 'left':
                        reward = 1
                    elif extraReward[trial] == 'right': # Add extra reward in our case the laser, if it is a laser trial
                        reward = 1 + extraRewardVal
                    else:
                        reward = 1
        			
                elif currentStim==0:
                    if np.random.rand() > 0.5:	# no evidence up to chance, if correct
                        
                        if action[trial,i] == 'left': 
                            if extraReward[trial] == 'left':
                                reward = 1 + extraRewardVal
                            elif extraReward[trial] == 'right':
                                reward = 1
                            else:
                                reward = 1
        					
                        else:
                            if extraReward[trial] == 'left':
                                reward = 1
                            elif extraReward[trial] == 'right':
                                reward = 1 + extraRewardVal
                            else:
                                reward = 1
        				
                    else: # if 0 evidence incorrect
                        if action[trial,i] == 'left':
                            if extraReward[trial] == 'left':
                                reward = 0 + extraRewardVal
                            elif extraReward[trial] == 'right':
                                reward = 0
                            else:
                                reward = 0
        					
                        else:
                            if extraReward[trial] == 'left':
                                reward = 0
                            elif extraReward[trial] == 'right':
                                reward = 0 + extraRewardVal
                            else:
                                reward = 0
        			
                else: #  cases with errors
                    if currentStim>0 and (action[trial,i] == 'left'): # erroneous choice
                        if extraReward[trial] == 'left':
                            reward = 0 + extraRewardVal # Add extra reward in our case the laser, if it is a laser trial
                        elif extraReward[trial] == 'right':
                            reward = 0
                        else:
                            reward = 0
            			
                    else: # erroneous choice
                        if extraReward[trial] == 'left':
                            reward = 0
                        elif extraReward[trial] == 'right': # Add extra reward in our case the laser, if it is a laser trial
                            reward = 0 + extraRewardVal
                        else:
                            reward = 0
                            
                # calculate the prediction error, and update Q values
                if action[trial,i] == 'left':
                        QLL = QLL +  learningRate * ((reward) - Q_chosen)
                        QRL = QRL +  learningRate * ((reward) - Q_chosen)
                else:
                    QLR = QLR + learningRate * ((reward) - Q_chosen)
                    QRR = QRR + learningRate * ((reward) - Q_chosen)
               
    end_time = time.time()
    #print(end_time - start_time)
    
    # set output
    output=pd.DataFrame()
    output['QL'] = np.mean(QL,axis =1)
    output['QR'] = np.mean(QR,axis =1)
    output['pL1'] = np.mean(pL1,axis =1)
    output['pR1'] = np.mean(pR1,axis =1)
    trial_action = trial_action*-1
    output['action'] = np.mean(trial_action,axis =1)
    output['signed_contrast'] = stimTrials
    return output

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

def session_log_likelihood1(params, model_data):
        output = RunPOMDP(params, model_data)
        negLL = -np.log(np.sum(output['pChoice']))
        return negLL
    
# Fit per animal

best_nphr_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),6])
best_nphr_NLL_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])
best_nphr_accu_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])
best_nphr_accu_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])

for i, mouse in enumerate(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()): 
    model_data_nphr, simulate_data_nphr  = \
        psy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse], 
                                                  virus = 'nphr')
    
    train_data_nphr = model_data_nphr.iloc[:int(len(model_data_nphr)*0.7),:]
    test_data_nphr = model_data_nphr.iloc[int(len(model_data_nphr)*0.7):,:]
    simulate_data_nphr.iloc[int(len(model_data_nphr)*0.7):,:]
    best_nphr_individual[i,:], best_nphr_NLL_individual[i,:] = optimizer_runPOMDP(train_data_nphr, num_fits = 2, 
                                initial_guess=[0.1, -1, 0.5, 0.6])
    
    
    output_nphr = RunPOMDP_stay(best_nphr_individual[i,:], test_data_nphr)
    psy.loc[psy['mouse_name'] == mouse, 'QL'] = output_nphr['QL'].to_numpy()
    psy.loc[psy['mouse_name'] == mouse, 'QR'] = output_nphr['QR'].to_numpy()
    psy.loc[psy['mouse_name'] == mouse, 'pChoice'] = output_nphr['pChoice'].to_numpy()
    best_nphr_accu_individual[i,:] = np.sum(output['pChoice'] > 0.5)/len(output['pChoice'] > 0.5)