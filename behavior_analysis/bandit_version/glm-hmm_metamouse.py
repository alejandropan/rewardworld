import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
from ssm.util import one_hot, find_permutation
import pandas as pd
import rewardworld.behavior_analysis.bandit_version.session_summary as ses_sum
from pathlib import Path
import patsy

# GLM-HMM session formater
def glmhmm_formatter(ses_df):
    '''
    Makes df from load_session_dataframe (ses_df) into array for
    glmhmm packgage
    '''
    ses_df['choice']= ses_df['choice'].map({-1:0, 0:np.nan, 1:1})
    ses_df = ses_df.dropna().reset_index()
    endog, exog = patsy.dmatrices('choice ~ previous_choice_1:C(previous_outcome_1)'
                                  '+ previous_choice_2:C(previous_outcome_2)'
                                  '+ previous_choice_3:C(previous_outcome_3)'
                                  '+ previous_choice_4:C(previous_outcome_4)'
                                  '+ previous_choice_5:C(previous_outcome_5)'
                                  '+ previous_choice_6:C(previous_outcome_6)'
                                  '+ previous_choice_7:C(previous_outcome_7)'
                                  '+ previous_choice_8:C(previous_outcome_8)'
                                  '+ previous_choice_9:C(previous_outcome_9)'
                                  '+ previous_choice_10:C(previous_outcome_10)'
                                  '+ previous_choice_1:C(previous_outcome_1):C(previous_laser_1)'
                                  '+ previous_choice_2:C(previous_outcome_2):C(previous_laser_2)'
                                  '+ previous_choice_3:C(previous_outcome_3):C(previous_laser_3)'
                                  '+ previous_choice_4:C(previous_outcome_4):C(previous_laser_4)'
                                  '+ previous_choice_5:C(previous_outcome_5):C(previous_laser_5)'
                                  '+previous_laser_1'
                                  '+previous_laser_2'
                                  '+previous_laser_3'
                                  '+previous_laser_4'
                                  '+previous_laser_5'
                                  '+1',
                                  data=ses_df,return_type='dataframe')

    exog.rename(columns={'previous_choice_1:C(previous_outcome_1)[1.0]': 'rewarded_1',
                         'previous_choice_2:C(previous_outcome_2)[1.0]': 'rewarded_2',
                         'previous_choice_3:C(previous_outcome_3)[1.0]': 'rewarded_3',
                         'previous_choice_4:C(previous_outcome_4)[1.0]': 'rewarded_4',
                         'previous_choice_5:C(previous_outcome_5)[1.0]': 'rewarded_5',
                         'previous_choice_6:C(previous_outcome_6)[1.0]': 'rewarded_6',
                         'previous_choice_7:C(previous_outcome_7)[1.0]': 'rewarded_7',
                         'previous_choice_8:C(previous_outcome_8)[1.0]': 'rewarded_8',
                         'previous_choice_9:C(previous_outcome_9)[1.0]': 'rewarded_9',
                         'previous_choice_10:C(previous_outcome_10)[1.0]': 'rewarded_10',
                         'previous_choice_1:C(previous_outcome_1)[0.0]': 'unrewarded_1',
                         'previous_choice_2:C(previous_outcome_2)[0.0]': 'unrewarded_2',
                         'previous_choice_3:C(previous_outcome_3)[0.0]': 'unrewarded_3',
                         'previous_choice_4:C(previous_outcome_4)[0.0]': 'unrewarded_4',
                         'previous_choice_5:C(previous_outcome_5)[0.0]': 'unrewarded_5',
                         'previous_choice_6:C(previous_outcome_6)[0.0]': 'unrewarded_6',
                         'previous_choice_7:C(previous_outcome_7)[0.0]': 'unrewarded_7',
                         'previous_choice_8:C(previous_outcome_8)[0.0]': 'unrewarded_8',
                         'previous_choice_9:C(previous_outcome_9)[0.0]': 'unrewarded_9',
                         'previous_choice_10:C(previous_outcome_10)[0.0]': 'unrewarded_10',
                         'previous_choice_1:C(previous_outcome_1)[1.0]:C(previous_laser_1)[T.1.0]': 'stim_rewarded_1',
                         'previous_choice_2:C(previous_outcome_2)[1.0]:C(previous_laser_2)[T.1.0]': 'stim_rewarded_2',
                         'previous_choice_3:C(previous_outcome_3)[1.0]:C(previous_laser_3)[T.1.0]': 'stim_rewarded_3',
                         'previous_choice_4:C(previous_outcome_4)[1.0]:C(previous_laser_4)[T.1.0]': 'stim_rewarded_4',
                         'previous_choice_5:C(previous_outcome_5)[1.0]:C(previous_laser_5)[T.1.0]': 'stim_rewarded_5',
                         'previous_choice_1:C(previous_outcome_1)[0.0]:C(previous_laser_1)[T.1.0]': 'stim_unrewarded_1',
                         'previous_choice_2:C(previous_outcome_2)[0.0]:C(previous_laser_2)[T.1.0]': 'stim_unrewarded_2',
                         'previous_choice_3:C(previous_outcome_3)[0.0]:C(previous_laser_3)[T.1.0]': 'stim_unrewarded_3',
                         'previous_choice_4:C(previous_outcome_4)[0.0]:C(previous_laser_4)[T.1.0]': 'stim_unrewarded_4',
                         'previous_choice_5:C(previous_outcome_5)[0.0]:C(previous_laser_5)[T.1.0]': 'stim_unrewarded_5',
                         'previous_laser_1':'laser_1',
                         'previous_laser_2':'laser_2',
                         'previous_laser_3':'laser_3',
                         'previous_laser_4':'laser_4',
                         'previous_laser_5':'laser_5',
                         'Intercept': 'bias',},
                         inplace = True)
    #reorder exog
    exog=exog[['rewarded_1','rewarded_2','rewarded_3','rewarded_4','rewarded_5',
           'rewarded_6','rewarded_7','rewarded_8','rewarded_9','rewarded_10',
           'unrewarded_1','unrewarded_2','unrewarded_3','unrewarded_4','unrewarded_5',
           'unrewarded_6','unrewarded_7','unrewarded_8','unrewarded_9','unrewarded_10',
           'stim_rewarded_1','stim_rewarded_2','stim_rewarded_3','stim_rewarded_4','stim_rewarded_5',
           'stim_unrewarded_1','stim_unrewarded_2','stim_unrewarded_3','stim_unrewarded_4','stim_unrewarded_5',
           'laser_1','laser_2', 'laser_3', 'laser_4', 'laser_5','bias']]

    inpt = exog.to_numpy(dtype=int)
    true_choices = endog.to_numpy(dtype=int)
    return inpt, true_choices

# Load data
gen_path = Path('/Volumes/witten/Alex/Data/Subjects/opto_chr2_bandit/chr2')
meta_inpts = []
meta_true_choices=[]
for animal in sorted(gen_path.iterdir()):
    try:
        for day in sorted(animal.iterdir()):
            try:
                for ses in day.iterdir():
                    try:
                        print(ses)
                        ses_df = ses_sum.load_session_dataframe(ses._str)
                        inpt, true_choices = glmhmm_formatter(ses_df)
                        meta_inpts.append(inpt)
                        meta_true_choices.append(true_choices)
                    except:
                        continue
            except:
                continue
    except:
        continue


# Cross validated state number, held out session
num_sessions = len(meta_true_choices)
log_likelihoods = np.zeros([5,num_sessions])
for fold in np.arange(num_sessions):
    lls=[]
    train_idx = np.array(list(set(np.arange(num_sessions)).symmetric_difference([fold])))
    test_idx = [fold]
    training_inpt = list(map(meta_inpts.__getitem__, train_idx))
    training_choice = list(map(meta_true_choices.__getitem__, train_idx))
    test_inpt = list(map(meta_inpts.__getitem__, test_idx))
    test_choices = list(map(meta_true_choices.__getitem__, test_idx))
    for i in np.arange(1,6):
        num_states = i
        obs_dim = 1
        input_dim = 36
        num_categories = 2
        bandit_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                                observation_kwargs=dict(C=num_categories), transitions="standard")
        fit_ll = bandit_glmhmm.fit(training_choice, inputs=training_inpt, method="em",
                                   num_iters=N_iters, tolerance=10**-4)
        lls.append(bandit_glmhmm.log_likelihood(test_choices, inputs=test_inpt))
    log_likelihoods[:,fold]=lls

num_states = 2
obs_dim = 1
input_dim = 36
num_categories = 2




# Fit model
N_iters = 400 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter

weights_pool=[]
ll=[]
for i in range(20):
    bandit_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                            observation_kwargs=dict(C=num_categories), transitions="standard")
    fit_ll = bandit_glmhmm.fit(meta_true_choices, inputs=meta_inpts, method="em",
                               num_iters=N_iters, tolerance=10**-8)
    ll.append(fir_ll)
    weights_pool.append(bandit_glmhmm.observations.params*-1)
    print(fit_ll[-1])

weights = weights_pool[np.where(np.array(ll)==max(ll))[0][0]]

    # Plot results

# Log likelihood for every iteration

cols = ['blue', 'orange']

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
    plt.plot(range(input_dim), weights[k][0],
                 lw=2,  label = '', linestyle = '--', color=cols[k])
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks(np.arange(36), covariates, fontsize=12, rotation=90)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.tight_layout()

# Get expected states:
posterior_probs = [bandit_glmhmm.expected_states(data=meta_true_choices, input=meta_inpts)[0]
                   for meta_true_choices, meta_inpts
                   in zip(meta_true_choices, meta_inpts)]


fig, ax = plt.subplots(5,4, figsize=(15,20))
for i in range(num_sessions):
    column = int(np.floor(i/5))
    row =i%5
    plt.sca(ax[row, column])
    for k in range(num_states):
        plt.plot(posterior_probs[i][:, k], label="State " + str(k + 1), lw=2,
                 color=cols[k])
    plt.ylim((-0.01, 1.01))
    plt.yticks([0, 0.5, 1], fontsize = 10)
    plt.xlabel("trial #", fontsize = 15)
    plt.ylabel("p(state)", fontsize = 15)
    plt.tight_layout()

plt.plot(log_likelihoods.mean(axis=1))
plt.xticks(np.arange(5), np.arange(1,6))
plt.xlabel('Number of latents')
plt.ylabel('LL')