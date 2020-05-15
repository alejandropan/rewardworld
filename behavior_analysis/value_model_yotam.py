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




# Computing by marginalizing over the perceived contrast i.e. 
# p(decode_contrast | true_contrast) = \sum p(decode contrast | perceived contrast) * p(perceived contrast | true contrast)
def true_stim_posterior(true_contrast, all_contrasts, beliefSTD):
	# Compute distribution over perceived contrast
	# start_time = time.time()
	p_perceived = normal_pdf(all_contrasts, loc=true_contrast, scale=beliefSTD)

	start_time = time.time()
	mat = np.zeros((len(all_contrasts), len(all_contrasts))) # vectorized for speed, but this implements the sum above
	for idx, perceived_contrast in enumerate(all_contrasts):
		mat[:, idx] = normal_pdf(all_contrasts, loc=perceived_contrast, scale=beliefSTD)

	posterior = mat @ p_perceived
	posterior /= np.sum(posterior)

	return posterior

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

# Given Q_left, Q_right, and the softmax inverse temperature beta, compute the probability of
# turning left or right
def softmax(Q_L, Q_R, beta):
	p = [np.exp(Q_L / beta), np.exp(Q_R / beta)]
	p /= np.sum(p)

	return p

def softmax_bias(Q_L, Q_R, beta, bias):
	p = [np.exp(Q_L / beta), np.exp(Q_R / beta +bias)]
	p /= np.sum(p)

	return p

def softmax_stay_bias(Q_L, Q_R, beta, bias, l_stay, r_stay, stay):
	p = [np.exp(Q_L / beta + l_stay + stay*l_stay),
      np.exp(Q_R / beta + bias + l_stay + stay*r_stay)]
	p /= np.sum(p)

	return p

def softmax_stay(Q_L, Q_R, beta, l_stay, r_stay, stay):
	p = [np.exp(Q_L / beta + stay*l_stay),
      np.exp(Q_R / beta + stay*r_stay)]
	p /= np.sum(p)

	return p

# Compute the log likelihood of a given trial under model parameters params,
# and an underlying set of Q values, Q
def trial_log_likelihood(params, trial_data, Q, all_contrasts, all_posteriors):
	# Get relevant parameters
	trial_contrast, trial_choice, reward, laser = trial_data
	learning_rate, beliefSTD, extraVal, beta = params

	# Compute the log-likelihood of the actual mouse choice
	if all_posteriors is None:
		contrast_posterior = true_stim_posterior(trial_contrast, all_contrasts, beliefSTD)
	else:
		posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
		contrast_posterior = all_posteriors[posterior_idx, :]

	Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)
	choice_dist = softmax(Q_L, Q_R, beta)
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
	for i in range(len(all_contrasts)):
		Q[i, trial_choice] += contrast_posterior[i] * learning_rate * (received_reward - Q_chosen)

	return LL, Q

def trial_log_likelihood_stay_and_bias(params, trial_data, Q, all_contrasts, all_posteriors, 
                              previous_trial, trial_num):
	# Get relevant parameters
	trial_contrast, trial_choice, reward, laser = trial_data
	learning_rate, beliefSTD, extraVal, beta, bias, stay = params

	# Compute the log-likelihood of the actual mouse choice
	if all_posteriors is None:
		contrast_posterior = true_stim_posterior(trial_contrast, all_contrasts, beliefSTD)
	else:
		posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
		contrast_posterior = all_posteriors[posterior_idx, :]

	Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)
	if trial_num == 0:
		(l_stay, r_stay) = [0,0]
	else:
		previous_choice= [0,0]
		previous_choice[previous_trial] = 1
		(l_stay, r_stay) = previous_choice
    
	choice_dist = softmax_stay_bias(Q_L, Q_R, beta, bias, l_stay, r_stay, stay)
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
	for i in range(len(all_contrasts)):
		Q[i, trial_choice] += contrast_posterior[i] * learning_rate * (received_reward - Q_chosen)
    
	return LL, Q

def trial_log_likelihood_stay(params, trial_data, Q, all_contrasts, all_posteriors, 
                              previous_trial, trial_num):
	# Get relevant parameters
	trial_contrast, trial_choice, reward, laser = trial_data
	learning_rate, beliefSTD, extraVal, beta, stay = params

	# Compute the log-likelihood of the actual mouse choice
	if all_posteriors is None:
		contrast_posterior = true_stim_posterior(trial_contrast, all_contrasts, beliefSTD)
	else:
		posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
		contrast_posterior = all_posteriors[posterior_idx, :]

	Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)
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
	for i in range(len(all_contrasts)):
		Q[i, trial_choice] += contrast_posterior[i] * learning_rate * (received_reward - Q_chosen)

	return LL, Q


def trial_log_likelihood_bias(params, trial_data, Q, all_contrasts, all_posteriors):
	# Get relevant parameters
	trial_contrast, trial_choice, reward, laser = trial_data
	learning_rate, beliefSTD, extraVal, beta, bias = params

	# Compute the log-likelihood of the actual mouse choice
	if all_posteriors is None:
		contrast_posterior = true_stim_posterior(trial_contrast, all_contrasts, beliefSTD)
	else:
		posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
		contrast_posterior = all_posteriors[posterior_idx, :]

	Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)
	choice_dist = softmax_bias(Q_L, Q_R, beta, bias)
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
	for i in range(len(all_contrasts)):
		Q[i, trial_choice] += contrast_posterior[i] * learning_rate * (received_reward - Q_chosen)

	return LL, Q


# Compute the log likelihood of all the concatenated trials
def session_neg_log_likelihood(params, *data, pregen_all_posteriors=True, accu=False):
	# Unpack the arguments
	learning_rate, beliefSTD, extraVal, beta = params
	rewards, true_contrasts, choices, lasers = data
	num_trials = len(rewards)

	# Generate the possible contrast list
	all_contrasts = np.array([-0.25, -0.125, -0.6125, 0, 0.6125, 0.125, 0.25])
	num_contrasts = len(all_contrasts)

	# If True, generate all posterior distributions ahead of time to save time
	if pregen_all_posteriors:
		all_posteriors = np.zeros((num_contrasts, num_contrasts))
		for idx, contrast in enumerate(all_contrasts):
			all_posteriors[idx, :] = true_stim_posterior(contrast, all_contrasts, beliefSTD)
	else:
		all_posteriors = None

	# Compute the log-likelihood
	if accu == True:
		acc = 0
	LL = 0
	Q = np.zeros((len(all_contrasts), 2))
	for i in range(num_trials):
		trial_LL, newQ = trial_log_likelihood(params, [true_contrasts[i], choices[i], rewards[i], lasers[i]], Q, all_contrasts, all_posteriors)
		LL += trial_LL
		Q = newQ
        
		if accu == True:
			acc += (np.exp(trial_LL)>0.5)*1    
            
	if accu == True:
		acc = acc/num_trials
		return -LL,  acc
	else:
		return -LL

def session_neg_log_likelihood_bias(params, *data, pregen_all_posteriors=True, accu=False):
	# Unpack the arguments
	learning_rate, beliefSTD, extraVal, beta, bias = params
	rewards, true_contrasts, choices, lasers = data
	num_trials = len(rewards)

	# Generate the possible contrast list
	all_contrasts = np.array([-0.25, -0.125, -0.6125, 0, 0.6125, 0.125, 0.25])
	num_contrasts = len(all_contrasts)

	# If True, generate all posterior distributions ahead of time to save time
	if pregen_all_posteriors:
		all_posteriors = np.zeros((num_contrasts, num_contrasts))
		for idx, contrast in enumerate(all_contrasts):
			all_posteriors[idx, :] = true_stim_posterior(contrast, all_contrasts, beliefSTD)
	else:
		all_posteriors = None

	# Compute the log-likelihood
	if accu == True:
		acc = 0
	LL = 0
	Q = np.zeros((len(all_contrasts), 2))
	for i in range(num_trials):
		trial_LL, newQ = trial_log_likelihood_bias(params, [true_contrasts[i], choices[i], rewards[i], lasers[i]], Q, all_contrasts, all_posteriors)
		LL += trial_LL
		Q = newQ

		if accu == True:
			acc += (np.exp(trial_LL)>0.5)*1

	if accu == True:
		acc = acc/num_trials
		return -LL,  acc
	else:
		return -LL


def session_neg_log_likelihood_stay(params, *data, pregen_all_posteriors=True, accu=False):
	# Unpack the arguments
	learning_rate, beliefSTD, extraVal, beta, stay = params
	rewards, true_contrasts, choices, lasers = data
	num_trials = len(rewards)

	# Generate the possible contrast list
	all_contrasts = np.array([-0.25, -0.125, -0.6125, 0, 0.6125, 0.125, 0.25])
	num_contrasts = len(all_contrasts)

	# If True, generate all posterior distributions ahead of time to save time
	if pregen_all_posteriors:
		all_posteriors = np.zeros((num_contrasts, num_contrasts))
		for idx, contrast in enumerate(all_contrasts):
			all_posteriors[idx, :] = true_stim_posterior(contrast, all_contrasts, beliefSTD)
	else:
		all_posteriors = None

	# Compute the log-likelihood
	if accu == True:
		acc = 0
	LL = 0
	Q = np.zeros((len(all_contrasts), 2))
	for i in range(num_trials):
		if i == 0:
			trial_LL, newQ = trial_log_likelihood_stay(params, 
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]], 
                                            Q, all_contrasts, all_posteriors, np.nan, i)
		else:
			trial_LL, newQ = trial_log_likelihood_stay(params, 
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]], 
                                            Q, all_contrasts, all_posteriors, choices[i-1], i)
		LL += trial_LL
		Q = newQ
        
		if accu == True:
			acc += (np.exp(trial_LL)>0.5)*1

	if accu == True:
		return -LL,  acc/num_trials
	else:
		return -LL

def session_neg_log_likelihood_stay_and_bias(params, *data, pregen_all_posteriors=True, accu = False):
	# Unpack the arguments
	learning_rate, beliefSTD, extraVal, beta, bias, stay = params
	rewards, true_contrasts, choices, lasers = data
	num_trials = len(rewards)

	# Generate the possible contrast list
	all_contrasts = np.array([-0.25, -0.125, -0.6125, 0, 0.6125, 0.125, 0.25])
	num_contrasts = len(all_contrasts)

	# If True, generate all posterior distributions ahead of time to save time
	if pregen_all_posteriors:
		all_posteriors = np.zeros((num_contrasts, num_contrasts))
		for idx, contrast in enumerate(all_contrasts):
			all_posteriors[idx, :] = true_stim_posterior(contrast, all_contrasts, beliefSTD)
	else:
		all_posteriors = None
    
	if accu == True:
		acc = 0
	LL = 0    
	# Compute the log-likelihood
	Q = np.zeros((len(all_contrasts), 2))
	for i in range(num_trials):
		trial_data = [true_contrasts[i], choices[i], rewards[i], lasers[i]]
		previous_trial = choices[i-1]
		trial_num = i
		if i == 0:
			[true_contrasts[i], choices[i], rewards[i], lasers[i]]
			trial_LL, newQ = trial_log_likelihood_bias(params[:5], trial_data, 
                                            Q, all_contrasts, all_posteriors)
		else:
			trial_LL, newQ = trial_log_likelihood_stay_and_bias(params, trial_data, 
                                            Q, all_contrasts, all_posteriors, previous_trial, trial_num)
		LL += trial_LL
		Q = newQ
        
		if accu == True:
			acc += (np.exp(trial_LL)>0.5)*1

	if accu == True:
		return -LL, acc/num_trials
	else:
		return -LL

# Optimize several times with different initializations and return the best fit parameters, and negative log likelihood
def optimizer(data, num_fits = 5, initial_guess=[0.1, 1, 0, 1]):
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
			beliefSTD_guess = np.random.uniform(0.01, 1)
			extraVal_guess = np.random.uniform(-5,5)
			beta_guess = np.random.uniform(0.01, 5)
			initial_guess = [lr_guess, beliefSTD_guess, extraVal_guess, beta_guess]

		# Run the fit
		res = so.minimize(session_neg_log_likelihood, initial_guess, args=data, method='L-BFGS-B', bounds=[(0, 1), (0.01, 1), (-5, 5), (0.01, 1)])

		# If this fit is better than the previous best, remember it, otherwise toss
		buffer_x[i,:] = res.x
		buffer_NLL.append(res.fun)
        
		if res.fun <= best_NLL:
			best_NLL = res.fun
			best_x = res.x

	return best_x, best_NLL, buffer_NLL, buffer_x

def optimizer_bias(data, num_fits = 5, initial_guess=[0.1, 1, 0, 1, 0]):
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
			beliefSTD_guess = np.random.uniform(0.01, 1)
			extraVal_guess = np.random.uniform(-5,5)
			beta_guess = np.random.uniform(0.01, 5)
			bias = np.random.uniform(-100, 100)
			initial_guess = [lr_guess, beliefSTD_guess, extraVal_guess, beta_guess,bias]

		# Run the fit
		res = so.minimize(session_neg_log_likelihood_bias, initial_guess, args=data, method='L-BFGS-B', bounds=[(0, 1), (0.01, 1), (-5, 5), (0.01, 1), 
                                                                                                     (-100,100)])

		# If this fit is better than the previous best, remember it, otherwise toss
		buffer_x[i,:] = res.x
		buffer_NLL.append(res.fun)
        
		if res.fun <= best_NLL:
			best_NLL = res.fun
			best_x = res.x

	return best_x, best_NLL, buffer_NLL, buffer_x

def optimizer_stay(data, num_fits = 5, initial_guess=[0.1, 1, 0, 1, 1]):
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
			beliefSTD_guess = np.random.uniform(0.01, 1)
			extraVal_guess = np.random.uniform(-5,5)
			beta_guess = np.random.uniform(0.01, 5)
			stay = np.random.uniform(-100, 100)
			initial_guess = [lr_guess, beliefSTD_guess, extraVal_guess, beta_guess, stay]

		# Run the fit
		res = so.minimize(session_neg_log_likelihood_stay, initial_guess, args=data, method='L-BFGS-B', bounds=[(0, 1), (0.01, 1), (-5, 5), (0.01, 1), 
                                                                                                     (-100,100)])

		# If this fit is better than the previous best, remember it, otherwise toss
		buffer_x[i,:] = res.x
		buffer_NLL.append(res.fun)
        
		if res.fun <= best_NLL:
			best_NLL = res.fun
			best_x = res.x

	return best_x, best_NLL, buffer_NLL, buffer_x

def optimizer_stay_and_bias(data, num_fits = 5, initial_guess=[0.1, 1, 0, 1, 0, 1]):
	# Accounting variables
	best_NLL = np.Inf
	best_x = [None, None, None, None, None, None]
	buffer_NLL = []
	buffer_x = np.empty([num_fits,len(initial_guess)])
	# Do our fit with several different initializations
	for i in range(num_fits):
		print('Starting fit %d' % i)

		# For every fit other than the first, construct a new initial guess
		if i != 0:
			lr_guess = np.random.uniform(0, 1)
			beliefSTD_guess = np.random.uniform(0.01, 1)
			extraVal_guess = np.random.uniform(-5,5)
			beta_guess = np.random.uniform(0.01, 5)
			bias = np.random.uniform(-100, 100)
			stay = np.random.uniform(-100, 100)
			initial_guess = [lr_guess, beliefSTD_guess, extraVal_guess, beta_guess, bias, stay]

		# Run the fit
		res = so.minimize(session_neg_log_likelihood_stay_and_bias, initial_guess, args=data, method='L-BFGS-B', bounds=[(0, 1), (0.01, 1), (-5, 5), (0.01, 1), 
                                                                                                     (-100,100),  (-100,100)])

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

##### Analysis functions

def accuracy_per_contrast(data, all_contrasts):
	rewards, contrasts, choices, lasers = data
	num_trials = len(rewards)

	acc = np.zeros(len(all_contrasts))
	trials_per_contrast = np.zeros(len(all_contrasts))
	for i in range(num_trials):
		reward = rewards[i]
		contrast = contrasts[i]
		choice = choices[i]
		laser = lasers[i]

		contrast_idx = np.argmin(np.abs(all_contrasts - contrast))
		
		if reward > 0:
			acc[contrast_idx] += 1

		trials_per_contrast[contrast_idx] += 1

	acc /= trials_per_contrast

	return acc

def psychometric_curve(data, all_contrasts):
	rewards, contrasts, choices, lasers = data
	num_trials = len(rewards)

	p_right = np.zeros(len(all_contrasts))
	trials_per_contrast = np.zeros(len(all_contrasts))
	for i in range(num_trials):
		reward = rewards[i]
		contrast = contrasts[i]
		choice = choices[i]
		laser = lasers[i]

		contrast_idx = np.argmin(np.abs(all_contrasts - contrast))
		
		if choice == 1:
			p_right[contrast_idx] += 1

		trials_per_contrast[contrast_idx] += 1

	p_right /= trials_per_contrast

	return p_right

##### TESTING SCRIPT #####

def simulation_contrast_distribution(mean_contrast, beliefSTD, all_contrasts):
	# Compute distribution of final perceived contrasts
	p = normal_pdf(all_contrasts, loc=mean_contrast, scale=beliefSTD)

	# Renormalize
	p /= np.sum(p)

	return p

def simulation_QL_QR(Q, perceived_contrast, all_contrasts, beliefSTD):
	Q_L = 0
	Q_R = 0

	# We compute Q_L, Q_R as in the Cell paper, by taking the weighted average of the various perceived stimuli,
	# according to the probability that they were perceived
	contrast_posterior = simulation_contrast_distribution(perceived_contrast, beliefSTD, all_contrasts)
	for i in range(len(all_contrasts)):
		Q_L += contrast_posterior[i] * Q[i, 0]
		Q_R += contrast_posterior[i] * Q[i, 1]

	return Q_L, Q_R

def generate_data(data, all_contrasts, learning_rate=0.3, beliefSTD=0.1, 
                  extraVal=1, beta=0.2, is_verbose=False, propagate_errors = False):
	rewards = []
	true_contrasts = []
	choices = []
	lasers = []
    
	if propagate_errors == False:
		prop = 3
	else:
		prop = 4


	# Simulate the POMDP model
	Q = np.zeros((len(all_contrasts), 2))
	for t in range(len(data[0])):
		if is_verbose:
			print(t)

		# Pick a true stimulus and store
		trial_contrast = data[1][t]
		true_contrasts.append(trial_contrast)
		
		# Add noise
		perceived_contrast_distribution = simulation_contrast_distribution(trial_contrast, beliefSTD, all_contrasts)
		perceived_contrast = np.random.choice(all_contrasts, p=perceived_contrast_distribution)

		# Make a decision and store it
		Q_L, Q_R = simulation_QL_QR(Q, trial_contrast, all_contrasts, beliefSTD)
		choice_dist = softmax(Q_L, Q_R, beta)
		choice = np.random.choice(2, p=choice_dist)
		choices.append(choice)

		# Get reward and store it
		if np.sign(trial_contrast) == -1 and choice == 0:
			reward = 1
		elif np.sign(trial_contrast) == 1 and choice == 1:
			reward = 1
		else:
			reward = 0

		rewards.append(reward)
		
		# Add laser value on the correct condition
		if choice == data[prop][t]:
			reward += extraVal
			lasers.append(1)
		else:
			lasers.append(-1)

		# Learn (update Q-values)
		if choice == 0:
			Q_chosen = Q_L
		else:
			Q_chosen = Q_R

		contrast_posterior = simulation_contrast_distribution(perceived_contrast, beliefSTD, all_contrasts)
		for i in range(len(all_contrasts)):
			Q[i, choice] += contrast_posterior[i] * learning_rate * (reward - Q_chosen)

	return rewards, true_contrasts, choices, lasers

def generate_data_bias(data, all_contrasts, learning_rate=0.3, beliefSTD=0.1, 
                       extraVal=1, beta=0.2, bias = 0, is_verbose=False, propagate_errors = False):
	rewards = []
	true_contrasts = []
	choices = []
	lasers = []

	if propagate_errors == False:
		prop = 3
	else:
		prop = 4

	# Simulate the POMDP model
	Q = np.zeros((len(all_contrasts), 2))
	for t in range(len(data[0])):
		if is_verbose:
			print(t)

		# Pick a true stimulus and store
		trial_contrast = data[1][t]
		true_contrasts.append(trial_contrast)
		
		# Add noise
		perceived_contrast_distribution = simulation_contrast_distribution(trial_contrast, beliefSTD, all_contrasts)
		perceived_contrast = np.random.choice(all_contrasts, p=perceived_contrast_distribution)

		# Make a decision and store it
		Q_L, Q_R = simulation_QL_QR(Q, trial_contrast, all_contrasts, beliefSTD)
		choice_dist = softmax_bias(Q_L, Q_R, beta, bias)
		choice = np.random.choice(2, p=choice_dist)
		choices.append(choice)

		# Get reward and store it
		if np.sign(trial_contrast) == -1 and choice == 0:
			reward = 1
		elif np.sign(trial_contrast) == 1 and choice == 1:
			reward = 1
		else:
			reward = 0

		rewards.append(reward)
		
		# Add laser value on the correct condition
		if choice == data[prop][t]:
			reward += extraVal
			lasers.append(1)
		else:
			lasers.append(-1)

		# Learn (update Q-values)
		if choice == 0:
			Q_chosen = Q_L
		else:
			Q_chosen = Q_R

		contrast_posterior = simulation_contrast_distribution(perceived_contrast, beliefSTD, all_contrasts)
		for i in range(len(all_contrasts)):
			Q[i, choice] += contrast_posterior[i] * learning_rate * (reward - Q_chosen)

	return rewards, true_contrasts, choices, lasers

def generate_data_stay(data, all_contrasts, learning_rate=0.3, beliefSTD=0.1, 
                       extraVal=1, beta=0.2, stay = 1, is_verbose=False, propagate_errors = False):
	rewards = []
	true_contrasts = []
	choices = []
	lasers = []
    
	if propagate_errors == False:
		prop = 3
	else:
		prop = 4

	# Simulate the POMDP model
	Q = np.zeros((len(all_contrasts), 2))
	for t in range(len(data[0])):
		if is_verbose:
			print(t)

		# Pick a true stimulus and store
		trial_contrast = data[1][t]
		true_contrasts.append(trial_contrast)
		
		# Add noise
		perceived_contrast_distribution = simulation_contrast_distribution(trial_contrast, beliefSTD, all_contrasts)
		perceived_contrast = np.random.choice(all_contrasts, p=perceived_contrast_distribution)

		# Make a decision and store it
		Q_L, Q_R = simulation_QL_QR(Q, trial_contrast, all_contrasts, beliefSTD)
		if t == 0:
		    (l_stay, r_stay) = [0,0]
		else:
		    previous_choice= [0,0]
		    previous_choice[choices[t-1]]= 1
		    (l_stay, r_stay) = previous_choice
		choice_dist = softmax_stay(Q_L, Q_R, beta,l_stay, r_stay, stay)
		choice = np.random.choice(2, p=choice_dist)
		choices.append(choice)

		# Get reward and store it
		if np.sign(trial_contrast) == -1 and choice == 0:
			reward = 1
		elif np.sign(trial_contrast) == 1 and choice == 1:
			reward = 1
		else:
			reward = 0

		rewards.append(reward)
		
		# Add laser value on the correct condition
		if choice == data[prop][t]:
			reward += extraVal
			lasers.append(1)
		else:
			lasers.append(-1)

		# Learn (update Q-values)
		if choice == 0:
			Q_chosen = Q_L
		else:
			Q_chosen = Q_R

		contrast_posterior = simulation_contrast_distribution(perceived_contrast, beliefSTD, all_contrasts)
		for i in range(len(all_contrasts)):
			Q[i, choice] += contrast_posterior[i] * learning_rate * (reward - Q_chosen)

	return rewards, true_contrasts, choices, lasers

def generate_data_stay_and_bias(data, all_contrasts, learning_rate=0.3, beliefSTD=0.1, 
                                extraVal=1, beta=0.2, bias = 0, stay = 1, is_verbose=False,
                                propagate_errors = False):
	rewards = []
	true_contrasts = []
	choices = []
	lasers = []
    
	if propagate_errors == False:
		prop = 3
	else:
		prop = 4

	# Simulate the POMDP model
	Q = np.zeros((len(all_contrasts), 2))
	for t in range(len(data[0])):
		if is_verbose:
			print(t)

		# Pick a true stimulus and store
		trial_contrast = data[1][t]
		true_contrasts.append(trial_contrast)
		
		# Add noise
		perceived_contrast_distribution = simulation_contrast_distribution(trial_contrast, beliefSTD, all_contrasts)
		perceived_contrast = np.random.choice(all_contrasts, p=perceived_contrast_distribution)

		# Make a decision and store it
		Q_L, Q_R = simulation_QL_QR(Q, trial_contrast, all_contrasts, beliefSTD)
		if t == 0:
		    (l_stay, r_stay) = [0,0]
		else:
		    previous_choice= [0,0]
		    previous_choice[choices[t-1]] = 1
		    (l_stay, r_stay) = previous_choice
		choice_dist = softmax_stay_bias(Q_L, Q_R, beta, bias, l_stay, r_stay, stay)
		choice = np.random.choice(2, p=choice_dist)
		choices.append(choice)

		# Get reward and store it
		if np.sign(trial_contrast) == -1 and choice == 0:
			reward = 1
		elif np.sign(trial_contrast) == 1 and choice == 1:
			reward = 1
		else:
			reward = 0

		rewards.append(reward)
		
		# Add laser value on the correct condition
		if choice == data[prop][t]:
			reward += extraVal
			lasers.append(1)
		else:
			lasers.append(-1)

		# Learn (update Q-values)
		if choice == 0:
			Q_chosen = Q_L
		else:
			Q_chosen = Q_R

		contrast_posterior = simulation_contrast_distribution(perceived_contrast, beliefSTD, all_contrasts)
		for i in range(len(all_contrasts)):
			Q[i, choice] += contrast_posterior[i] * learning_rate * (reward - Q_chosen)

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
    
# Main function, runs all the testing scripts
if __name__ == '__main__':
	# Test the fitting procedure by fitting on arbitrarily generated data
	# num_trials = 30000
	all_contrasts = np.array([-0.6125, -0.25, -0.125, 0, 0.125, 0.25, 0.6125])

	# data = generate_data(num_trials, all_contrasts)
	# x, NLL = optimizer(data, initial_guess=[0.3, 0.1, 1, 0.2])

	# print(x)

	# Load Alex's actual data
    psy = load_behavior_data_from_root('/Volumes/witten/Alex/recordings_march_2020_dop/')

    train_set_size = 1
    cross_validate = False
     
    all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.    ,  0.0625, 0.125 , 0.25  ])
    best_nphr_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),4])
    best_nphr_NLL_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])
    model_parameters = pd.DataFrame()
    modelled_data = pd.DataFrame()
    for i, mouse in enumerate(psy['mouse_name'].unique()): 
        model_data_nphr, simulate_data_nphr  = \
            psy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse], 
                                              virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0])
        
        
        obj = transform_model_struct_2_POMDP(model_data_nphr, simulate_data_nphr)
        
        virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0]
        
        opto = obj['extraRewardTrials'].to_numpy()
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
                lasers[:int(len(rewards)*train_set_size)])
        simulate_data = (rewards[:int(len(rewards)*train_set_size)], 
                         contrasts[:int(len(rewards)*train_set_size)], 
                         choices[:int(len(rewards)*train_set_size)], 
                      lasers[:int(len(rewards)*train_set_size)], 
                      laser_side[:int(len(rewards)*train_set_size)])
        
        if cross_validate == True:
            
            data_test = (rewards[int(len(rewards)*train_set_size):], 
                         contrasts[int(len(rewards)*train_set_size):], 
                         choices[int(len(rewards)*train_set_size):], 
                         lasers[int(len(rewards)*train_set_size):])
            simulate_data_test = (rewards[int(len(rewards)*train_set_size):], 
                                  contrasts[int(len(rewards)*train_set_size):], 
                                  choices[int(len(rewards)*train_set_size):], 
                          lasers[int(len(rewards)*train_set_size):], 
                          laser_side[int(len(rewards)*train_set_size):])
        else:
            data_test = data
            simulate_data_test = simulate_data
        
        (best_x, train_NLL, buffer_NLL, buffer_x) = optimizer(data, 
                           initial_guess=[0.3, 0.01, -1, 0.2])
        (best_x_bias, train_NLL_bias, buffer_NLL_bias, 
         buffer_x_bias) = optimizer_bias(data, initial_guess=[0.3, 0.01, -1, 0.2, 0])
        (best_x_stay, train_NLL_stay, buffer_NLL_stay, 
         buffer_x_stay) = optimizer_stay(data, initial_guess=[0.3, 0.01, -1, 0.2,1])
        (best_x_bias_stay, train_NLL_bias_stay, 
        buffer_NLL_bias_stay, buffer_x_bias_stay) = optimizer_stay_and_bias(data, 
                                    initial_guess=[0.3, 0.01, -1, 0.2, 0, 1])
        
        cv_aic = aic((session_neg_log_likelihood(best_x, 
                                                 *data_test, pregen_all_posteriors=True))*-1, 4)
        cv_aic_bias = aic((session_neg_log_likelihood_bias(best_x_bias, 
                                                           *data_test, pregen_all_posteriors=True))*-1,5)
        cv_aic_stay = aic((session_neg_log_likelihood_stay(best_x_stay, *data_test, pregen_all_posteriors=True))*-1,5)
        cv_aic_bias_stay = aic((session_neg_log_likelihood_stay_and_bias(best_x_bias_stay, *data_test, pregen_all_posteriors=True))*-1,6)
        
        cv_LL = (session_neg_log_likelihood(best_x, *data_test, pregen_all_posteriors=True))*-1
        cv_LL_bias = (session_neg_log_likelihood_bias(best_x_bias, *data_test, pregen_all_posteriors=True))*-1
        cv_LL_stay = (session_neg_log_likelihood_stay(best_x_stay, *data_test, pregen_all_posteriors=True))*-1
        cv_LL_bias_stay = (session_neg_log_likelihood_stay_and_bias(best_x_bias_stay, *data_test, pregen_all_posteriors=True))*-1
        
        _, cv_acc = session_neg_log_likelihood(best_x, *data_test, pregen_all_posteriors=True, accu=True)
        _, cv_acc_bias = session_neg_log_likelihood_bias(best_x_bias, *data_test, pregen_all_posteriors=True, accu=True)
        _, cv_acc_stay = session_neg_log_likelihood_stay(best_x_stay, *data_test, pregen_all_posteriors=True, accu=True)
        _, cv_acc_bias_stay = session_neg_log_likelihood_stay_and_bias(best_x_bias_stay, *data_test, pregen_all_posteriors=True, accu=True)
        
        bias_vs_x = chi2_LLR(cv_LL,cv_LL_bias)
        stay_vs_x = chi2_LLR(cv_LL,cv_LL_stay)
        bias_stay_vs_x = chi2_LLR(cv_LL,cv_LL_bias_stay)
        bias_stay_bias_vs_stay = chi2_LLR(cv_LL_bias_stay,cv_LL_stay)
        bias_stay_bias_vs_bias = chi2_LLR(cv_LL_bias_stay,cv_LL_bias)
        
        model_parameters_mouse = pd.DataFrame()
        model_parameters_mouse['model_name'] = 'standard', 'w_bias', 'w_stay', 'w_bias_n_stay'
        model_parameters_mouse['x'] = best_x, best_x_bias, best_x_stay, best_x_bias_stay
        model_parameters_mouse['LL'] = (cv_LL/len(data_test[0])), (cv_LL_bias/len(data_test[0])), (cv_LL_stay/len(data_test[0])), (cv_LL_bias_stay/len(data_test[0]))
        model_parameters_mouse['aic'] = cv_aic, cv_aic_bias, cv_aic_stay, cv_aic_bias_stay
        model_parameters_mouse['LLR_p_values_cv'] =  np.nan, bias_vs_x, bias_stay_vs_x, bias_stay_vs_x
        model_parameters_mouse['accu'] = cv_acc, cv_acc_bias, cv_acc_stay, cv_acc_bias_stay
        
        sim_data = generate_data(simulate_data_test, all_contrasts, learning_rate=best_x[0], beliefSTD=best_x[1], extraVal=best_x[2], beta=best_x[3])
        sim_data = pd.DataFrame(sim_data)
        sim_data_bias =  generate_data_bias(simulate_data_test, all_contrasts, learning_rate=best_x_bias[0], 
                                       beliefSTD=best_x_bias[1], extraVal=best_x_bias[2], beta=best_x_bias[3], bias=best_x_bias[4])
        sim_data_stay =  generate_data_stay(simulate_data_test, all_contrasts, learning_rate=best_x_stay[0], 
                                       beliefSTD=best_x_stay[1], extraVal=best_x_stay[2], beta=best_x_stay[3], stay=best_x_stay[4])
        sim_data_stay_and_bias =  generate_data_stay_and_bias(simulate_data_test, all_contrasts, learning_rate=best_x_bias_stay[0], 
                                       beliefSTD=best_x_bias_stay[1], extraVal=best_x_bias_stay[2], beta=best_x_bias_stay[3],
                                       bias = best_x_bias_stay[4], stay = best_x_bias_stay[5])
        
        sim_data = sim_data.rename(columns={0: "rewards", 1: "signed_contrast", 2: "simulated_choices", 3: "model"})
        sim_data = np.array(sim_data)
        sim_data = pd.DataFrame(sim_data).T
        sim_data['laser_side'] = laser_side[:int(len(rewards)*train_set_size)]
        sim_data['real_choice'] = choices[:int(len(rewards)*train_set_size)]
        sim_data['choices_w_bias'] =  sim_data_bias[2]
        sim_data['choices_w_stay'] = sim_data_stay[2]
        sim_data['choices_w_bias_n_stay'] = sim_data_stay_and_bias[2]
        sim_data['mouse_name']  = mouse
        sim_data['virus']  = virus
       
        # Concatenate with general dataframes
        
        model_parameters_mouse['mouse'] = mouse
        model_parameters_mouse['virus'] = virus
        
        # Concatenate with general dataframes
        model_parameters = pd.concat([model_parameters, model_parameters_mouse])
        modelled_data = pd.concat([modelled_data, sim_data])


modelled_data = modelled_data.rename(columns={0: "rewards", 
   1: "signed_contrast", 2: "choices_standard", 3: "laser_trials"})

###############################################################################
################################# Figures #####################################
###############################################################################

## Model comparison requires data from the simple model

fig,  ax = plt.subplots(3,2, figsize = [20,20])
sns.set(font_scale=1.5)
plt.sca(ax[0,0])
sns.pointplot(data = model_parameters_complex, x = 'model_name', y = 'LL', 
              hue ='virus', ci = 68, label='large')
ax[0,0].set_xlabel('Model Type')
ax[0,0].set_ylabel('Average LL')
plt.sca(ax[0,1])
sns.pointplot(data = model_parameters_complex, x = 'model_name', 
              y = 'LL', hue ='virus', ci = None, alpha=0.02)
sns.pointplot(data = model_parameters_simple, x = 'model_name', 
              y = 'LL', hue ='virus', ci = 68)
ax[0,1].lines[0].set_linestyle("--")
ax[0,1].lines[1].set_linestyle("--")
ax[0,1].set_xlabel('Model Type')
ax[0,1].set_ylabel('Average LL')
plt.sca(ax[1,0])
sns.pointplot(data = model_parameters_complex, x = 'model_name', 
              y = 'aic', hue ='virus', ci = 68)
ax[1,0].set_xlabel('Model Type')
ax[1,0].set_ylabel('AIC')
plt.sca(ax[1,1])
sns.pointplot(data = model_parameters_complex, x = 'model_name', 
              y = 'aic', hue ='virus', ci = None, alpha=0.02)
sns.pointplot(data = model_parameters_simple, x = 'model_name', 
              y = 'aic', hue ='virus', ci = 68)
ax[1,1].lines[0].set_linestyle("--")
ax[1,1].lines[1].set_linestyle("--")
ax[1,1].set_xlabel('Model Type')
ax[1,1].set_ylabel('AIC')
plt.sca(ax[2,0])
sns.pointplot(data = model_parameters_complex, x = 'model_name', 
              y = 'accu', hue ='virus', ci = 68)
ax[2,0].set_xlabel('Model Type')
ax[2,0].set_ylabel('Accuracy')
plt.sca(ax[2,1])
sns.pointplot(data = model_parameters_complex, x = 'model_name',
              y = 'accu', hue ='virus', ci = None, alpha=0.02)
sns.pointplot(data = model_parameters_simple, x = 'model_name', 
              y = 'accu', hue ='virus', ci = 68)
ax[2,1].lines[0].set_linestyle("--")
ax[2,1].lines[1].set_linestyle("--")
ax[2,1].set_xlabel('Model Type')
ax[2,1].set_ylabel('Accuracy')
plt.tight_layout()

## Comparison of parameters across animals


ax = sns.lineplot(data= sim_data, x = 'signed_contrast', 
                  y = 'simulated_choices', hue = 'laser_side', ci = 0)
ax.lines[0].set_linestyle("--")
ax.lines[1].set_linestyle("--")
ax.lines[2].set_linestyle("--")
sns.lineplot(data= sim_data, x = 'signed_contrast', 
             y = 'real_choice', hue = 'laser_side', ci = 68)

def boxplot_model_parameters_per_mouse(model_parameters, 
                                       model_type= 'w_bias_n_stay', 
                                       save = True):
    '''
    Notes: Plots learned parameters across virus
    '''
    fig, ax =  plt.subplots()
    sns.set(style='white')
    model = model_parameters.loc[model_parameters['model_name'] == model_type, 
                                 ['x', 'virus']]
    params = [r'$\alpha$', r'$\theta$', r'$\psi$',
              r'$\tau$', r'$\gamma$', r'$\phi$']
    mod = pd.DataFrame(columns = ['params', 'virus'])
    for i in range(len(model)):
        temp_mod = pd.DataFrame(model['x'].iloc[i])
        temp_mod['params'] = params[:len(temp_mod)]
        temp_mod['virus'] = model.iloc[i,1]
        mod = mod.append(temp_mod)
    sns.swarmplot(data = mod,  x = 'params', y = 0,  hue = 'virus', 
                  palette = ['dodgerblue', 'orange'], split = False)
    ax.axhline(0, ls='--', color = 'k')
    ax.set_xlabel('Model Parameter')
    ax.set_ylabel('Fitted Coef')
    sns.despine()
    if save == True:
         plt.savefig('learned_parameters.pdf', dpi =300)
    
def model_psychometrics(modelled_data, model_type= 'w_bias_n_stay', 
                        perform_inlet = True,   save = True):
    '''
    Notes: Plots psychometrics
    '''
    
    sns.set(style='white')
    num_mice  = len(modelled_data['mouse_name'].unique())
    fig, ax =  plt.subplots(1, num_mice, figsize = (40,10), sharey=True)
    mice = modelled_data['mouse_name'].unique()
    for i in range(num_mice):
        plt.sca(ax[i])
        # Plot real data on trials that have simulation
        mouse_data = modelled_data.loc[modelled_data['mouse_name'] == mice[i]]
        temp_data  = mouse_data.loc[mouse_data['choices_'+model_type].notnull()]
        sns.lineplot(data = temp_data, x = 'signed_contrast', y = 'real_choice',
                     hue = 'laser_side', palette = ['k','b','g'], ci =68, legend=None)
        # Plot model data with dash line
        sns.lineplot(data = temp_data, x = 'signed_contrast', y = 'choices_'+model_type,
                     hue = 'laser_side', palette = ['k','b','g'], ci =0, legend=None)
        ax[i].lines[3].set_linestyle("--")
        ax[i].lines[4].set_linestyle("--")
        ax[i].lines[5].set_linestyle("--")
        ax[i].set_ylabel('Fraction of Choices', fontsize =20)
        ax[i].set_xlabel('Signed Contrast', fontsize =20)
        ax[i].set_title(mice[i], fontsize =20)
    sns.despine()  
        
