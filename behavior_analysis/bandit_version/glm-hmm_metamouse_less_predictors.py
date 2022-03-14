import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import ssm
from ssm.util import one_hot, find_permutation
import pandas as pd
import rewardworld.behavior_analysis.bandit_version.session_summary as ses_sum
from pathlib import Path
import patsy
import seaborn as sns

# GLM-HMM session formater
def glmhmm_formatter(ses_df, trials_back=2):
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
                         'previous_choice_1:C(previous_outcome_1)[0.0]:C(previous_laser_1)[T.1.0]': 'stim_unrewarded_1',
                         'Intercept': 'bias',},
                inplace = True)
    #reorder exog
    list_var=[]
    for i in np.arange(1,trials_back+1):
        list_var.append('rewarded_%s' % i)
    for i in np.arange(1,trials_back+1):
        list_var.append('unrewarded_%s' % i)
    list_var.append('stim_rewarded_1')
    list_var.append('stim_unrewarded_1')
    list_var.append('bias')

    exog=exog[list_var]
    inpt = exog.to_numpy(dtype=int)
    true_choices = endog.to_numpy(dtype=int)
    return inpt, true_choices

# Load data
gen_path = Path('/Volumes/witten/Alex/Data/Subjects/opto_chr2_bandit/full_bandit_opto_2s_ITI/chr2')
for i in np.arange(5):
    trials_back=i+1
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
                            inpt, true_choices = glmhmm_formatter(ses_df, trials_back=trials_back)
                            meta_inpts.append(inpt)
                            meta_true_choices.append(true_choices)
                        except:
                            continue
                except:
                    continue
        except:
            continue


    # Cross validated state number, held out session
    if FIND_BEST_PARAM_STRUCT==True:
        num_sessions = len(meta_true_choices)
        log_likelihoods = np.zeros([3,num_sessions])
        for fold in np.arange(num_sessions):
            lls=[]
            lls_var=[]
            train_idx = np.array(list(set(np.arange(num_sessions)).symmetric_difference([fold])))
            test_idx = [fold]
            training_inpt = list(map(meta_inpts.__getitem__, train_idx))
            training_choice = list(map(meta_true_choices.__getitem__, train_idx))
            test_inpt = list(map(meta_inpts.__getitem__, test_idx))
            test_choices = list(map(meta_true_choices.__getitem__, test_idx))
            for i in np.arange(1,4):
                num_states = i
                obs_dim = 1
                input_dim = 5
                num_categories = 2
                i_ll =[]
                for p in np.arange(20):
                    bandit_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                                            observation_kwargs=dict(C=num_categories), transitions="standard")
                    fit_ll = bandit_glmhmm.fit(training_choice, inputs=training_inpt, method="em",
                                               num_iters=N_iters, tolerance=10**-4)
                    i_ll.append(bandit_glmhmm.log_likelihood(test_choices, inputs=test_inpt))
                lls.append(np.mean(i_ll))
                lls_var.append(np.stddev(i_ll))
            log_likelihoods[:,fold]=lls

        # Plot Crossvalidation
        sns.pointplot(x=np.arange(5),y=np.mean(log_likelihoods, axis=1))
        plt.xticks(np.arange(5), np.arange(1,6))
        plt.ylabel('LL')
        plt.xlabel('Number of latents')

    # Fit model
    num_states = 2
    obs_dim = 1
    input_dim = np.shape(meta_inpts[0])[1]
    num_categories = 2
    N_iters = 400 # maximum number of EM iterations. Fitting with stop earlier if increase in LL is below tolerance specified by tolerance parameter
    bandit_glmhmm = ssm.HMM(num_states, obs_dim, input_dim, observations="input_driven_obs",
                                observation_kwargs=dict(C=num_categories), transitions="standard")
    fit_ll = bandit_glmhmm.fit(meta_true_choices, inputs=meta_inpts, method="em",
                                   num_iters=N_iters, tolerance=10**-8)
    weights = bandit_glmhmm.observations.params*-1

# Plot GLM summary
cols = ['blue', 'orange', 'green']
covariates=[]
for i in np.arange(1,trials_back+1):
    covariates.append('rewarded_%s' % i)
for i in np.arange(1,trials_back+1):
    covariates.append('unrewarded_%s' % i)
covariates.append('stim_rewarded_1')
covariates.append('stim_unrewarded_1')
covariates.append('bias')
for k in range(num_states):
    plt.plot(range(input_dim), weights[k][0],
             lw=2,  label = '', linestyle = '--', color=cols[k])
plt.yticks(fontsize=10)
plt.ylabel("GLM weight", fontsize=15)
plt.xlabel("covariate", fontsize=15)
plt.xticks(np.arange(len(covariates)), covariates, fontsize=12, rotation=90)
plt.axhline(y=0, color="k", alpha=0.5, ls="--")
plt.tight_layout()

#Summary of transitions
recovered_trans_mat = np.exp(bandit_glmhmm.transitions.log_Ps)
plt.imshow(recovered_trans_mat, vmin=-0.8, vmax=1, cmap='bone')
for i in range(recovered_trans_mat.shape[0]):
    for j in range(recovered_trans_mat.shape[1]):
        text = plt.text(j, i, str(np.around(recovered_trans_mat[i, j], decimals=2)), ha="center", va="center",
                        color="k", fontsize=12)
plt.xlim(-0.5, num_states - 0.5)
plt.xticks(range(0, num_states), np.arange(num_states), fontsize=10)
plt.yticks(range(0, num_states), np.arange(num_states), fontsize=10)
plt.ylim(num_states - 0.5, -0.5)
plt.subplots_adjust(0, 0, 1, 1)


# Get expected states:
posterior_probs = [bandit_glmhmm.expected_states(data=a, input=b)[0]
                   for a, b
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


#Occupancy
# concatenate posterior probabilities across sessions
posterior_probs_concat = np.concatenate(posterior_probs)
# get state with maximum posterior probability at particular trial:
state_max_posterior = np.argmax(posterior_probs_concat, axis = 1)
# now obtain state fractional occupancies:
_, state_occupancies = np.unique(state_max_posterior, return_counts=True)
state_occupancies = state_occupancies/np.sum(state_occupancies)
fig = plt.figure(figsize=(2, 2.5), dpi=80, facecolor='w', edgecolor='k')
for z, occ in enumerate(state_occupancies):
    plt.bar(z, occ, width = 0.8, color = cols[z])
plt.ylim((0, 1))
plt.xticks([0, 1, 2], ['1', '2', '3'], fontsize = 10)
plt.yticks([0, 0.5, 1], ['0', '0.5', '1'], fontsize=10)
plt.xlabel('state', fontsize = 15)
plt.ylabel('frac. occupancy', fontsize=15)
# Win stay Lose switch dividing by predictor
df = pd.DataFrame()
for ses in range(len(meta_inpts)):
      session= pd.DataFrame()
      states=[]
      for i in range(len(posterior_probs[ses])):
          states.append(np.where(np.max(posterior_probs[ses][i,:])==
                                 posterior_probs[ses][i,:])[0][0])
      session['state'] = states
      session['choice'] = meta_true_choices[ses]
      session['previous_choice'] = session['choice'].shift(1)
      session['repeated'] = 1*(session['choice']==session['previous_choice'])
      session['previous_laser'] = ((meta_inpts[ses][:,-2]+meta_inpts[ses][:,-3])!=0)*1
      session['previous_reward'] =((meta_inpts[ses][:,0]!=0))*1
      session['number'] = ses
      df = pd.concat([df,session])

fig, ax = plt.subplots(1,3, sharey=True)
plt.sca(ax[0])
df_0 = df.loc[df['state']==0]
df_1 = df.loc[df['state']==1]
df_2 = df.loc[df['state']==2]
sns.barplot(x=df_0['previous_reward'], y=df_0['repeated'],
            hue = df_0['previous_laser'], ci =68)
plt.title('State 1')
plt.xlabel('Outcome t-1')
plt.ylabel('% Stay')
plt.sca(ax[1])
plt.title('State 2')
sns.barplot(x=df_1['previous_reward'], y=df_1['repeated'],
            hue = df_1['previous_laser'], ci =68)
plt.xlabel('Outcome t-1')
plt.ylabel('% Stay')
plt.tight_layout()
plt.sca(ax[2])
plt.title('State 3')
sns.barplot(x=df_2['previous_reward'], y=df_2['repeated'],
            hue = df_2['previous_laser'], ci =68)
plt.xlabel('Outcome t-1')
plt.ylabel('% Stay')
plt.tight_layout()

sns.barplot(data=a, x='state', y='previous_choice', hue='choice')
plt.ylabel('Num trials')
plt.xticks(np.arange(2),np.arange(1,3))
