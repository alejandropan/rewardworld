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
from matplotlib.lines import Line2D



    
def plot_q_trial_whole_dataset_per_mouse(psy_df, save=True):
    psy_select = psy_df.copy()
    sns.set(style = 'white', font_scale=3)
    num_mice = len(psy_select['mouse_name'].unique())
    index = psy_select.loc[psy['trial_within_block'] == 0, 'index']
    index = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
    for i in range(10):
        for idx in index-(i+1):
            psy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
                -(i+1)
            psy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
                psy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]
    fig, ax = plt.subplots(1,num_mice, figsize = [60,20], sharey =True)
    for i, mouse in enumerate(psy_select['mouse_name'].unique()):
        palette ={'R':'g','L':'b','non_opto':'k'}
        plt.sca(ax[i])
        psy_mouse = psy_select.loc[psy_select['mouse_name']==mouse]
        sns.lineplot(data = psy_mouse, x = 'trial_within_block', y = 'QRQL',
                         hue = 'opto_block', palette = palette, ci=68)
        plt.xlim(-10,50)
        plt.ylim(-1,1)
        plt.title('VTA-'+str(psy_select.loc[psy_select['mouse_name']==mouse,
                                            'virus'].unique()) + '-' +
                                                                  str(mouse))
        ax[i].set_xlabel('Trial in block')
        ax[i].set_ylabel('QR-QL')
    sns.despine()
    if save ==True:
        plt.savefig('q_across_trials_p_mouse.svg', dpi =300)
        plt.savefig('q_across_trials_p_mouse.jpeg',  dpi =300)

def plot_q_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
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
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QRQL',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(-1,1)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('QR-QL')
    
    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QRQL',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(-1,1)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()
    
    if save ==True:
        plt.savefig('q_across_trials.svg', dpi =300)
        plt.savefig('q_across_trials.jpeg',  dpi =300)
        

def boxplot_model_parameters_per_mouse(model_parameters, 
                                       model_type= 'w_stay', 
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
         
def simulate_and_plot(modelled_data, model_parameters, 
                    model_type= 'w_stay'):
    mice = modelled_data['mouse_name'].unique()
    for mouse in mice:
        data_pd = modelled_data.loc[modelled_data['mouse_name'] == mouse, 
                    ['real_rewards', 'signed_contrast', 'real_choice', 'laser_trials',
                     'laser_side']]
        
        # -1 to 0 for laser
        data_pd.loc[data_pd['laser_trials'] == -1, 'laser_trials'] = 0 
        # Make data into the right format
        data_np = data_pd.to_numpy()
        array_of_tuples = map(tuple, data_np.T)
        data = tuple(array_of_tuples)
        data2 = tuple(tuple(map(int, tup)) for tup in data[2:]) 
        data0 = tuple(tuple(map(int, data[0]))) 
        data1  = data[1]
        data = [data0, data1, data2[0], data2[1], data2[2]]
        params = model_parameters.loc[(model_parameters['mouse'] == mouse)
        & (model_parameters['model_name'] == model_type), 'x'].tolist()[0]
        
        # Multiply data by 1000
        data_m = []
        for i  in range(len(data)):
            data_m.append(data[i]*1)        

        # Calculate Q values
        if model_type == 'standard':
           sim_data = generate_data(data_m, all_contrasts, learning_rate=params[0], 
                                               beliefSTD=params[1], extraVal=params[2], 
                                               beta=params[3], stay=params[4])   
        
        if model_type == 'w_bias':
            sim_data = generate_data_bias(data_m, all_contrasts, learning_rate=params[0], 
                                               beliefSTD=params[1], extraVal=params[2], 
                                               beta=params[3], stay=params[4])   
        
        if model_type == 'w_stay':
            sim_data = generate_data_stay(data_m, all_contrasts, learning_rate=params[0], 
                                               beliefSTD=params[1], extraVal=params[2], 
                                               beta=params[3], stay=params[4])   
        
        if model_type == 'w_bias_n_stay':
           sim_data = generate_data_stay_and_bias(data_m, all_contrasts, learning_rate=params[0], 
                                               beliefSTD=params[1], extraVal=params[2], 
                                               beta=params[3], stay=params[4])
        # Plots
        sim_data = pd.DataFrame(sim_data)
        sim_data = np.array(sim_data)
        sim_data = pd.DataFrame(sim_data).T
        sim_data['laser'] = data_m[3]
        sim_data['laser_side'] = data_m[4]
        sim_data['real_choice'] = data_m[2]
        sim_data['mouse_name']  = mouse
        sim_data['real_rewards']  = data_m[0]
        sim_data = sim_data.rename(columns={0: "rewards", 1: "signed_contrast", 2: "simulated_choices", 3: "model_laser"})
        
        model_psychometrics(sim_data, data_pd, mouse, save = True)
        
    return modelled_data

def calculate_QL_QR(modelled_data, model_parameters, 
                    model_type= 'w_stay'):
    # Also calculates pRight
    
    ACC = []
    mice = modelled_data['mouse_name'].unique()
    modelled_data['QL'] = np.nan
    modelled_data['QR'] = np.nan
    for mouse in mice:
        data_pd = modelled_data.loc[modelled_data['mouse_name'] == mouse, 
                    ['real_rewards', 'signed_contrast', 'real_choice', 'laser_trials']]
        
        # -1 to 0 for laser
        data_pd.loc[data_pd['laser_trials'] == -1, 'laser_trials'] = 0 
        # Make data into the right format
        data_np = data_pd.to_numpy()
        array_of_tuples = map(tuple, data_np.T)
        data = tuple(array_of_tuples)
        data2 = tuple(tuple(map(int, tup)) for tup in data[2:]) 
        data0 = tuple(tuple(map(int, data[0]))) 
        data1  = data[1]
        data = [data0, data1, data2[0], data2[1]]
        params = model_parameters.loc[(model_parameters['mouse'] == mouse)
        & (model_parameters['model_name'] == model_type), 'x'].copy().tolist()[0]
        # Calculate Q values
        if model_type == 'standard':
            _,_,Q_L,Q_R =  session_neg_log_likelihood(params, *data, 
            pregen_all_posteriors=True, accu=True, retrieve_Q=True)
        
        if model_type == 'w_bias':
            _,_,Q_L,Q_R = session_neg_log_likelihood_bias(params, 
        *data, pregen_all_posteriors=True, accu=True, retrieve_Q=True)
        
        if model_type == 'w_stay':
            _,acc,Q_L,Q_R, pRight = session_neg_log_likelihood_stay(params, 
        *data, pregen_all_posteriors=True, accu=True, retrieve_Q=True)
        
        if model_type == 'w_bias_n_stay':
            _,_,Q_L,Q_R = session_neg_log_likelihood_stay_and_bias(params, 
                *data, pregen_all_posteriors=True, accu=True, retrieve_Q=True)
        
        # Return Q values to matrix
        ACC.append(acc)
        modelled_data.loc[modelled_data['mouse_name'] == mouse, 
                    'QL'] = Q_L
        modelled_data.loc[modelled_data['mouse_name'] == mouse, 
                    'QR'] = Q_R
        modelled_data.loc[modelled_data['mouse_name'] == mouse, 
                    'pRight'] = pRight
    # Calculate QR-QL
    modelled_data['QRQL'] = modelled_data['QR'].to_numpy() - \
        modelled_data['QL'].to_numpy()
        
    return modelled_data



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



def softmax_stay(Q_L, Q_R, beta, l_stay, r_stay, stay):
	p = [np.exp(Q_L / beta + stay*l_stay),
      np.exp(Q_R / beta + stay*r_stay)]
	p /= np.sum(p)

	return p



def trial_log_likelihood_stay(params, trial_data, Q, all_contrasts, all_posteriors, 
                              previous_trial, trial_num, retrieve_Q=False):
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

	if retrieve_Q==True:
		return LL, Q, Q_L, Q_R, choice_dist[trial_choice]
	else:
		return LL, Q



def session_neg_log_likelihood_stay(params, *data, pregen_all_posteriors=True, 
                                    accu=False, retrieve_Q=False):
	# Unpack the arguments
	learning_rate, beliefSTD, extraVal, beta, stay = params
	rewards, true_contrasts, choices, lasers = data
	num_trials = len(rewards)
    
    # Retrieve Q
	if retrieve_Q==True:
		Q_L = []
		Q_R = []
		pRight = []

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
    
	if retrieve_Q == True:
            
		for i in range(num_trials):
			if i == 0:
			    trial_LL, newQ, Q_Lt, Q_Rt, pright = trial_log_likelihood_stay(params, 
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]], 
                                            Q, all_contrasts, all_posteriors, np.nan, i, retrieve_Q=True)
			else:
			    trial_LL, newQ, Q_Lt, Q_Rt, pright = trial_log_likelihood_stay(params, 
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]], 
                                            Q, all_contrasts, all_posteriors, choices[i-1], i, retrieve_Q=True)
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



def optimizer_stay(data, num_fits = 10, initial_guess=[0.1, 1, 0, 1, 1]):
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
                    method='L-BFGS-B', bounds=[(0, 1), (0.03, 1), (-1, 1), (0.01, 1), 
                                    (-1,1)])

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


def generate_data_stay(data, all_contrasts, learning_rate=0.3, beliefSTD=0.1, 
                       extraVal=1, beta=0.2, stay = 1, is_verbose=False, propagate_errors = True):
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
            
		reward = 0

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
    psy = pd.read_pickle('all_behav.pkl')
    psy = psy[psy['mouse_name']!= 'dop_3']
    mice = np.array(['dop_8', 'dop_9', 'dop_11', 'dop_4', 'dop_7'])
    train_set_size = 1
    cross_validate = False
     
    all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.    ,  0.0625, 0.125 , 0.25  ])
    best_nphr_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),4])
    best_nphr_NLL_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])
    model_parameters = pd.DataFrame()
    modelled_data = pd.DataFrame()
    for i, mouse in enumerate(mice): 
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
            
            data = (rewards[:int(len(rewards)*train_set_size)],
                contrasts[:int(len(rewards)*train_set_size)], 
                choices[:int(len(rewards)*train_set_size)], 
                lasers[:int(len(rewards)*train_set_size)])
            simulate_data_test = (rewards[int(len(rewards)*train_set_size):], 
                                  contrasts[int(len(rewards)*train_set_size):], 
                                  choices[int(len(rewards)*train_set_size):], 
                          lasers[int(len(rewards)*train_set_size):], 
                          laser_side[int(len(rewards)*train_set_size):])
        else:
            data_test = data
            simulate_data_test = simulate_data
        

        (best_x_stay, train_NLL_stay, buffer_NLL_stay, 
         buffer_x_stay) = optimizer_stay(data, initial_guess=[0.3, 0.05, -1, 0.2,1])
        
        cv_aic_stay = aic((session_neg_log_likelihood_stay(best_x_stay,
                  *data_test, pregen_all_posteriors=True))*-1,5)
        
       
        cv_LL_stay = (session_neg_log_likelihood_stay(best_x_stay, *data_test,
                                                      pregen_all_posteriors=True))*-1
       
        
        
        _, cv_acc_stay = session_neg_log_likelihood_stay(best_x_stay, *data_test, 
                       pregen_all_posteriors=True, accu=True)
       
        
        model_parameters_mouse = pd.DataFrame()
        model_parameters_mouse['x'] = [best_x_stay]
        model_parameters_mouse['LL'] = (cv_LL_stay/len(data_test[0]))
        model_parameters_mouse['aic'] = cv_aic_stay
        model_parameters_mouse['accu'] = cv_acc_stay
        model_parameters_mouse['model_name'] = 'w_stay'

        
        sim_data = generate_data_stay(simulate_data_test, all_contrasts, learning_rate=best_x_stay[0], 
                                       beliefSTD=best_x_stay[1], extraVal=best_x_stay[2], beta=best_x_stay[3], stay=best_x_stay[4])
        sim_data = pd.DataFrame(sim_data)
        
        sim_data = sim_data.rename(columns={0: "rewards", 1: "signed_contrast", 2: "simulated_choices", 3: "model_laser"})
        sim_data = np.array(sim_data)
        sim_data = pd.DataFrame(sim_data).T
        sim_data['laser'] = lasers[:int(len(rewards)*train_set_size)]
        sim_data['laser_side'] = laser_side[:int(len(rewards)*train_set_size)]
        sim_data['real_choice'] = choices[:int(len(rewards)*train_set_size)]
        sim_data['mouse_name']  = mouse
        sim_data['virus']  = virus
        sim_data['real_rewards']  = simulate_data[0]
       
        # Concatenate with general dataframes
        model_parameters_mouse['mouse'] = mouse
        model_parameters_mouse['virus'] = virus
        
        # Concatenate with general dataframes
        model_parameters = pd.concat([model_parameters, model_parameters_mouse])
        modelled_data = pd.concat([modelled_data, sim_data])

# Analysis

modelled_data = modelled_data.rename(columns={0: "rewards", 
   1: "signed_contrast", 2: "choices_standard", 3: "laser_trials"})

modelled_data = calculate_QL_QR(modelled_data, model_parameters, 
                    model_type= 'w_stay')

# Calculate a few things
psy['QL'] = np.nan
psy['QR'] = np.nan
psy['QRQL'] = np.nan
psy['pRight'] = np.nan
for i, mouse in enumerate(mice):
    psy.loc[psy['mouse_name'] == mouse, ['QL', 'QR', 'QRQL','pRight']] =\
    modelled_data.loc[modelled_data['mouse_name'] == mouse,
                          ['QL', 'QR', 'QRQL', 'pRight']].to_numpy()

psy['argmax_choice'] = (psy['pRight']>0.5)*1


boxplot_model_parameters_per_mouse(model_parameters, 
                                       model_type= 'w_stay', 
                                       save = True)
plot_q_trial_whole_dataset(psy, save= True)
plot_q_trial_whole_dataset_per_mouse(psy)
model_performance(model_parameters, modelled_data, model_type= 
                  'w_stay', save = True)

plot_choice_prob_opto_block(psy, 0, 'dop_7', save =True)
plot_choice_prob_opto_block(psy, 20, 'dop_8', save =True)
plot_choice_prob_opto_block(psy, 17, 'dop_9', save =True)
plot_choice_prob_opto_block(psy, 10, 'dop_11', save =True)
plot_choice_prob_opto_block(psy, 19, 'dop_4', save =True)

plot_qmotivation_trial_whole_dataset(psy, save= True)











