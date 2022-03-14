import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
from ssm.util import one_hot, find_permutation
import statsmodels.api as sm
from scipy.stats import bernoulli

npr.seed(0)

# Set the parameters of the GLM-HMM
num_states = 2        # number of discrete states
obs_dim = 1           # number of observed dimensions
num_categories = 2    # number of categories for output
input_dim = 36         # input dimensions

# Make a GLM-HMM
true_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                   observation_kwargs=dict(C=num_categories), transitions="standard")

# Put some toy parameters for simulations
gen_weights = np.array([[[2.6, 1.3, 0.3, 0.2, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01,
                          0.06, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                          1, 0.6, 0.2, 0.1, 0.02,
                          0.8, 0.4, 0.1, 0.05, 0.01,
                          0.01, 0.01, 0.01, 0.01, 0.01,
                          0.3]],
                        [[2.0, 1.6, 1.0, 0.6, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                          2.0, 1.6, 1.0, 0.6, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
                          0.6, 0.2, 0.15, 0.15, 0.05,
                          0.6, 0.2, 0.15, 0.15, 0.05,
                          0.8, 0.6, 0.3, 0.1, 0.01,
                          0.2]]])

gen_log_trans_mat = np.log(np.array([[[0.99, 0.01], [0.03, 0.97]]]))
true_glmhmm.observations.params = gen_weights
true_glmhmm.transitions.params = gen_log_trans_mat

# Plot generative parameters:
fig = plt.figure(figsize=(8, 3), dpi=80, facecolor='w', edgecolor='k')
plt.subplot(1, 2, 1)
cols = ['#ff7f00', '#4daf4a', '#377eb8']
for k in range(num_states):
    plt.plot(range(10), gen_weights[k][0][:10], marker='o',
             color=cols[k], linestyle='-',
             lw=1.5, label="state " + str(k+1) + " rewarded history")
    plt.plot(range(10), gen_weights[k][0][10:20], marker='o',
             color=cols[k], linestyle='--',
             lw=1.5, label="state " + str(k+1) + " unrewarded history")
    plt.plot(range(5), gen_weights[k][0][20:25], marker='*',
             color=cols[k], linestyle='-',
             lw=1.5, label="state " + str(k+1) + " laser rewarded history")
    plt.plot(range(5), gen_weights[k][0][25:30], marker='*',
             color=cols[k], linestyle='--',
             lw=1.5, label="state " + str(k+1) + " laser unrewarded history")
    plt.plot(range(5), gen_weights[k][0][30:35], marker='*',
             color=cols[k], linestyle='-.',
             lw=1.5, label="state " + str(k+1) + " laser bias history")
    plt.plot(10, gen_weights[k][0][35], marker='o',
             color=cols[k], linestyle=':',
             lw=1.5, label="state " + str(k+1) + " bias")
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.legend()
plt.title("Generative weights", fontsize = 15)

plt.subplot(1, 2, 2)
gen_trans_mat = np.exp(gen_log_trans_mat)[0]
plt.imshow(gen_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
for i in range(gen_trans_mat.shape[0]):
    for j in range(gen_trans_mat.shape[1]):
        text = plt.text(j, i, str(np.around(gen_trans_mat[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, num_states), ('1', '2'), fontsize=10)
plt.yticks(range(0, num_states), ('1', '2'), fontsize=10)
plt.ylim(num_states - 0.5, -0.5)
plt.ylabel("state t", fontsize = 15)
plt.xlabel("state t+1", fontsize = 15)
plt.title("Generative transition matrix", fontsize = 15)
plt.show()

# 1. Create input sequences
# Currently first 10 trials of every session are choice 0 (No choice)
num_sess = 6 # number of example sessions
num_trials_per_sess = 800 # number of trials in a session
inpts = np.zeros((num_sess, num_trials_per_sess, input_dim),dtype=int) # initialize inpts array
true_latents = np.zeros((num_sess, num_trials_per_sess,1),dtype=int) # initialize inpts array
true_choices = np.zeros((num_sess, num_trials_per_sess, 1),dtype=int) # initialize inpts array
# Generate correct laser and correct answer
blocks = [0.8,0.2]#Probability of reward on right
laser_prob=  0.2
rewarded_trials_left = np.zeros((num_sess, num_trials_per_sess, 1))
rewarded_trials_right = np.zeros((num_sess, num_trials_per_sess, 1))
laser_trials = np.zeros((num_sess, num_trials_per_sess, 1))
for ses in range(num_sess):
    block_lengths = np.random.choice(np.arange(10,20),int(num_trials_per_sess/10)) # 10, 20 min and max length of a block
    block_switches = block_lengths.cumsum()
    block_right_probs = np.random.choice(blocks, len(block_lengths))
    block_num=0
    for t in range(num_trials_per_sess):
        if t>block_switches[block_num]:
            block_num+=1
        right_prob=block_right_probs[block_num]
        rewarded_trials_right[ses,t]=np.random.choice(2,p=[1-right_prob,right_prob])
        rewarded_trials_left[ses,t]=np.random.choice(2,p=[right_prob,1-right_prob])
        laser_trials[ses,t]=np.random.choice(2, p=[1-laser_prob,laser_prob])

t_latents=[]
t_choices=[]
inpt=[]
#Generate inputs
for ses, input in enumerate(inpts):
    T = input.shape[0]
    print(ses,'started')
    # Collecting variables aside from input
    latent_z = np.zeros(input.shape[0], dtype=int)
    data = np.zeros(input.shape[0], dtype=int)
    # Set 1 for bias predictor (the last one)
    input[:,-1] = 1
    # Now loop through each time and get the input, state and the observation for each time step:
    pi0 = np.exp(gen_log_trans_mat[0][0])
    latent_z[0] = int(npr.choice(2, p=pi0))
    for t in range(0, T):
          Pt = np.exp(gen_log_trans_mat[0])
          # Get observation at current trial (based on state)
          data[t] = true_glmhmm.observations.sample_x(latent_z[t], _, np.expand_dims(input[t], axis=0), tag=None)
                # Get state at next trial
                if t < T-1:
                        latent_z[t+1] = int(npr.choice(num_states, p=Pt[latent_z[t]]))
                        # update past choice and wsls based on sampled y and correct answer
                        choice = 2*data[t] - 1
                        # predictors:
                        if data[t]==0:
                            rewarded = rewarded_trials_left[ses][t]
                        else:
                            rewarded = rewarded_trials_right[ses][t]
                        laser = laser_trials[ses][t]

                        # Rewarded regressors
                        input[t+1, 0] = choice*rewarded
                        input[t+1, 1:10] = input[t, 0:9]
                        # Unrewarded regressors
                        input[t+1, 10] = choice*(rewarded==0)
                        input[t+1, 11:20] = input[t, 10:19]
                        # Rewarded laser
                        input[t+1, 20] = choice*rewarded*laser
                        input[t+1, 21:25] = input[t, 20:24]
                        # Unrewarded laser
                        input[t+1, 25] = choice*(rewarded==0)*laser
                        input[t+1, 26:30] = input[t, 25:29]
                        # Laser bias
                        input[t+1, 30] = laser
                        input[t+1, 31:35] = input[t, 30:34]
          print(t)
          true_choices[ses]=data.reshape(len(data),1)
          true_latents[ses]=latent_z.reshape(len(data),1)
          print(ses, 'ended')
          t_latents.append(latent_z.reshape(len(data),1))
          t_choices.append(data.reshape(len(data),1))
          inpts[ses]=input
          inpt.append(input)

# Calculate true loglikelihood
true_choices=t_choices
true_latents=t_latents
true_ll = true_glmhmm.log_probability(true_choices, inputs=inpt)
print("true ll = " + str(true_ll))
new_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                     observation_kwargs=dict(C=num_categories), transitions="standard")

N_iters = 200 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
fit_ll = new_glmhmm.fit(true_choices, inputs=inpt, method="em", num_iters=N_iters, tolerance=10**-4)
# Plot the log probabilities of the true and fit models. Fit model final LL should be greater
# than or equal to true LL.
fig = plt.figure(figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
plt.plot(fit_ll, label="EM")
plt.plot([0, len(fit_ll)], true_ll * np.ones(2), ':k', label="True")
plt.legend(loc="lower right")
plt.xlabel("EM Iteration")
plt.xlim(0, len(fit_ll))
plt.ylabel("Log Probability")
plt.show()

new_glmhmm.permute(find_permutation(true_latents[0].ravel(),
                                    new_glmhmm.most_likely_states(true_choices[0],
                                    input=inpt[0])))
fig = plt.figure( dpi=80, facecolor='w', edgecolor='k')
cols = ['#ff7f00', '#4daf4a', '#377eb8']
recovered_weights = new_glmhmm.observations.params
covariates =[
    'choice*reward t-1',
    'choice*reward t-2',
    'choice*reward t-3',
    'choice*reward t-4',
    'choice*reward t-5',
    'choice*reward t-6',
    'choice*reward t-7',
    'choice*reward t-8',
    'choice*reward t-9',
    'choice*reward t-10',
    'choice*unreward t-1',
    'choice*unreward t-2',
    'choice*unreward t-3',
    'choice*unreward t-4',
    'choice*unreward t-5',
    'choice*unreward t-6',
    'choice*unreward t-7',
    'choice*unreward t-8',
    'choice*unreward t-9',
    'choice*unreward t-10',
    'choice*laser*reward t-1',
    'choice*laser*reward t-2',
    'choice*laser*reward t-3',
    'choice*laser*reward t-4',
    'choice*laser*reward t-5',
    'choice*laser*unreward t-1',
    'choice*laser*unreward t-2',
    'choice*laser*unreward t-3',
    'choice*laser*unreward t-4',
    'choice*laser*unreward t-5',
    'laser t-1',
    'laser t-2',
    'laser t-3',
    'laser t-4',
    'laser t-5',
    'intercept']
for k in range(num_states):
    if k ==0:
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k], linestyle='-',
                 lw=1.5, label="generative")
        plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                 lw=1.5,  label = "recovered", linestyle = '--')
    else:
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k], linestyle='-',
                 lw=1.5, label="")
        plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
                 lw=1.5,  label = '', linestyle = '--')
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks(np.arange(36), covariates, fontsize=12, rotation=90)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.legend()
plt.title("Weight recovery", fontsize=15)
plt.tight_layout()


# Try MAP
# Instantiate GLM-HMM and set prior hyperparameters
prior_sigma = 2
prior_alpha = 2
map_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                     observation_kwargs=dict(C=num_categories,prior_sigma=prior_sigma),
                     transitions="sticky", transition_kwargs=dict(alpha=prior_alpha,kappa=0))

# Fit GLM-HMM with MAP estimation:
_ = map_glmhmm.fit(true_choices, inputs=inpt, method="em", num_iters=N_iters, tolerance=10**-4)
true_likelihood = true_glmhmm.log_likelihood(true_choices, inputs=inpt)
mle_final_ll = new_glmhmm.log_likelihood(true_choices, inputs=inpt)
map_final_ll = map_glmhmm.log_likelihood(true_choices, inputs=inpt)
# Plot these values
fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
loglikelihood_vals = [true_likelihood, mle_final_ll, map_final_ll]
colors = ['Red', 'Navy', 'Purple']
for z, occ in enumerate(loglikelihood_vals):
    plt.bar(z, occ, width = 0.8, color = colors[z])
plt.ylim((true_likelihood-50, true_likelihood+50))
plt.xticks([0, 1, 2], ['true', 'mle', 'map'], fontsize = 10)
plt.xlabel('model', fontsize = 15)
plt.ylabel('loglikelihood', fontsize=15)

# Create additional input,latents and choice sequences to be used as held-out test data

# Repeat same as before
num_test_sess = 5 # number of example sessions
test_inpts = np.zeros((num_test_sess, num_trials_per_sess, input_dim),dtype=int) # initialize inpts array

# Generate correct laser and correct answer
blocks = [0.8,0.2]#Probability of reward on right
laser_prob=  0.2
rewarded_trials_left = np.zeros((num_sess, num_trials_per_sess, 1))
rewarded_trials_right = np.zeros((num_sess, num_trials_per_sess, 1))
laser_trials = np.zeros((num_sess, num_trials_per_sess, 1))
for ses in range(num_sess):
    block_lengths = np.random.choice(np.arange(10,20),int(num_trials_per_sess/10)) # 10, 20 min and max length of a block
    block_switches = block_lengths.cumsum()
    block_right_probs = np.random.choice(blocks, len(block_lengths))
    block_num=0
    for t in range(num_trials_per_sess):
        if t>block_switches[block_num]:
            block_num+=1
        right_prob=block_right_probs[block_num]
        rewarded_trials_right[ses,t]=np.random.choice(2,p=[1-right_prob,right_prob])
        rewarded_trials_left[ses,t]=np.random.choice(2,p=[right_prob,1-right_prob])
        laser_trials[ses,t]=np.random.choice(2, p=[1-laser_prob,laser_prob])

test_latents=[]
test_choices=[]
test_inpt=[]
#Generate inputs
for ses, input in enumerate(test_inpts):
    T = input.shape[0]
    print(ses,'started')
    # Collecting variables aside from input
    latent_z = np.zeros(input.shape[0], dtype=int)
    data = np.zeros(input.shape[0], dtype=int)
    # Set 1 for bias predictor (the last one)
    input[:,-1] = 1
    # Now loop through each time and get the input, state and the observation for each time step:
    pi0 = np.exp(gen_log_trans_mat[0][0])
    latent_z[0] = int(npr.choice(2, p=pi0))
    for t in range(0, T):
        Pt = np.exp(gen_log_trans_mat[0])
        # Get observation at current trial (based on state)
        data[t] = true_glmhmm.observations.sample_x(latent_z[t], _, np.expand_dims(input[t], axis=0), tag=None)
        # Get state at next trial
        if t < T-1:
            latent_z[t+1] = int(npr.choice(num_states, p=Pt[latent_z[t]]))
            # update past choice and wsls based on sampled y and correct answer
            choice = 2*data[t] - 1
            # predictors:
            if data[t]==0:
                rewarded = rewarded_trials_left[ses][t]
            else:
                rewarded = rewarded_trials_right[ses][t]
            laser = laser_trials[ses][t]

            # Rewarded regressors
            input[t+1, 0] = choice*rewarded
            input[t+1, 1:10] = input[t, 0:9]
            # Unrewarded regressors
            input[t+1, 10] = choice*(rewarded==0)
            input[t+1, 11:20] = input[t, 10:19]
            # Rewarded laser
            input[t+1, 20] = choice*rewarded*laser
            input[t+1, 21:25] = input[t, 20:24]
            # Unrewarded laser
            input[t+1, 25] = choice*(rewarded==0)*laser
            input[t+1, 26:30] = input[t, 25:29]
            # Laser bias
            input[t+1, 30] = laser
            input[t+1, 31:35] = input[t, 30:34]
    print(t)
    true_choices[ses]=data.reshape(len(data),1)
    true_latents[ses]=latent_z.reshape(len(data),1)
    print(ses, 'ended')
    test_latents.append(latent_z.reshape(len(data),1))
    test_choices.append(data.reshape(len(data),1))
    test_inpts[ses]=input
    test_inpt.append(input)
# Compare likelihood of test_choices for model fit with MLE and MAP:
mle_test_ll = new_glmhmm.log_likelihood(test_choices, inputs=test_inpt)
map_test_ll = map_glmhmm.log_likelihood(test_choices, inputs=test_inpt)

fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
loglikelihood_vals = [mle_test_ll, map_test_ll]
colors = ['Navy', 'Purple']
for z, occ in enumerate(loglikelihood_vals):
    plt.bar(z, occ, width = 0.8, color = colors[z])
plt.ylim((mle_test_ll-20, mle_test_ll+50))
plt.xticks([0, 1], ['mle', 'map'], fontsize = 10)
plt.xlabel('model', fontsize = 15)
plt.ylabel('loglikelihood', fontsize=15)

#plot MAP vs MLE
map_glmhmm.permute(find_permutation(true_latents[0].ravel(),
                                    map_glmhmm.most_likely_states(true_choices[0], input=inpt[0])))

fig = plt.figure(dpi=80, facecolor='w', edgecolor='k')
cols = ['#ff7f00', '#4daf4a', '#377eb8']
plt.subplot(1,2,1)
recovered_weights_mle = new_glmhmm.observations.params
for k in range(num_states):
    if k ==0: # show labels only for first state
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k],
                 lw=1.5, label="generative")
        plt.plot(range(input_dim), recovered_weights_mle[k][0], color=cols[k],
                 lw=1.5,  label = 'recovered', linestyle='--')
    else:
        plt.plot(range(input_dim), gen_weights[k][0], marker='o',
                 color=cols[k],
                 lw=1.5, label="")
        plt.plot(range(input_dim), recovered_weights_mle[k][0], color=cols[k],
                 lw=1.5,  label = '', linestyle='--')
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks(np.arange(36), covariates, fontsize=12, rotation=45)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.title("MLE", fontsize = 15)
plt.legend()
plt.legend()
plt.subplot(1,2,2)
recovered_weights = map_glmhmm.observations.params
for k in range(num_states):
    plt.plot(range(input_dim), gen_weights[k][0], marker='o',
             color=cols[k],
             lw=1.5, label="", linestyle = '-')
    plt.plot(range(input_dim), recovered_weights[k][0], color=cols[k],
             lw=1.5,  label = '', linestyle='--')
plt.yticks(fontsize=10)
plt.xticks(np.arange(36), covariates, fontsize=12, rotation=45)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.title("MAP", fontsize = 15)