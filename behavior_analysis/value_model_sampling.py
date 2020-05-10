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
                # calculate Q values
                QLL = QLL * belief_l
                QRL = QRL * belief_r
                QLR = QLR * belief_l
                QRR = QRR * belief_r
                
                # Overal action values
                QL[trial, i] = QLL + QRL
                QR[trial, i] = QLR + QRR
                
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
                    QLL = QLL + belief_l * learningRate * (rew - Q_chosen)
                    QRL = QRL + belief_r * learningRate * (rew - Q_chosen)
                else:
                    QLR = QLR + belief_l * learningRate * (rew - Q_chosen)
                    QRR = QRR + belief_r * learningRate * (rew - Q_chosen)
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




output = RunPOMDP(params, model_data1)
simulate = simulatePOMDP(params, model_data)
a = psy.loc[psy['mouse_name'] == 'dop_4'].reset_index()
sns.lineplot(data = a , x = 'signed_contrasts', y = (a['choice']==-1)*1)
sns.lineplot(data =output, x = 'signed_contrast', y = (output['pL1']>0.5) *1)
sns.lineplot(data =simulate, x = 'signed_contrast', y = (simulate['action'] >0)*1)

def simulatePOMDP(params, model_data):

    learningRate = 0.5 #params[0]
    extraRewardVal = 0 #params[1]
    beliefNoiseSTD = 0.0001#params[2]
    Temperature = params[3]
    iter_n = 10000
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
                # calculate Q values
                QLL = QLL * belief_l
                QRL = QRL * belief_r
                QLR = QLR * belief_l
                QRR = QRR * belief_r
                
                # Overal action values
                QL[trial, i] = QLL + QRL
                QR[trial, i] = QLR + QRR
                
                #switiching for a softmax
                #pL1[trial,i] , pR1[trial,i]  = q_softmax(QL[trial, i], QR[trial, i], Temperature)
            # Make a choice
                #action[trial,i] = np.random.choice(['left', 'right'], p=[pL1[trial,i], pR1[trial,i]])
                
                if  QL[trial, i] >  QR[trial, i]:
                    action[trial,i] = 'left'
                elif QR[trial, i] > QL[trial, i]: 
                    action[trial,i] = 'right'
                else:
                    action[trial,i] = np.random.choice(['left', 'right'], p =[0.5,0.5])
                
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
                        QLL = QLL + belief_l * learningRate * ((reward) - Q_chosen)
                        QRL = QRL + belief_r * learningRate * ((reward) - Q_chosen)
                else:
                    QLR = QLR + belief_l * learningRate * ((reward) - Q_chosen)
                    QRR = QRR + belief_r * learningRate * ((reward) - Q_chosen)
               
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
