from model_comparison_accu import load_data_reduced, make_stan_data_reduced, load_sim_data_reduced
from model_comparison_accu import q_learning_model_reduced_stay, simulate_q_learning_model_reduced_stay
from model_comparison_accu import reinforce_model_reduced, simulate_reinforce_reduced
from model_comparison_accu import reinforce_model_reduced_stay, simulate_reinforce_reduced_stay
from model_comparison_accu import reduced_uchida_model, simulate_reduced_uchida_model
from model_comparison_accu import double_update_model, simulate_double_update_model
from model_comparison_accu import plot_params_reduced, stan_data_to_df_reduced
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from investigating_laser_expdecaymodel import transform_data_laserdecay, trial_within_block, add_transition_info, laserdecayanalysis, plot_laserdecay_summary
import seaborn as sns

def assign_section (df):
    df['section'] = np.nan
    df.loc[df['index']<100, 'section'] = 0
    df.loc[(df['index']>=100) & (df['index']<200), 'section'] = 1
    df.loc[(df['index']>=200) & (df['index']<300), 'section'] = 2
    df.loc[(df['index']>=300) & (df['index']<400), 'section'] = 3
    df.loc[(df['index']>=400) & (df['index']<500), 'section'] = 4
    df.loc[(df['index']>=500) & (df['index']<600), 'section'] = 5
    df.loc[(df['index']>=600) & (df['index']<700), 'section'] = 6
    df.loc[(df['index']>=700) & (df['index']<800), 'section'] = 7
    return df

def trial_within_block(behav):
    behav  = behav.reset_index()
    behav['trial_within_block'] = np.nan
    behav['block_number'] = np.nan
    behav['trial_within_block_real'] = np.nan
    behav['probabilityLeft_next'] = np.nan # this is for plotting trials before block change
    behav['probabilityLeft_past'] = np.nan # this is for plotting trials before block change
    behav['block_change'] = np.concatenate([np.zeros(1),
                                            1*(np.diff(behav['probabilityLeft'])!=0)])
    block_switches = np.concatenate([np.zeros(1),
                                     behav.loc[behav['block_change']==1].index]).astype(int)
    col_trial_within_block = np.where(behav.columns == 'trial_within_block')[0][0]
    col_probabilityLeft_next = np.where(behav.columns == 'probabilityLeft_next')[0][0]
    col_block_number = np.where(behav.columns == 'block_number')[0][0]
    col_opto_probabilityLeft_past = np.where(behav.columns == 'probabilityLeft_past')[0][0]
    col_trial_within_block_real = np.where(behav.columns == 'trial_within_block_real')[0][0]
    for i in np.arange(len(block_switches)):
        if i == 0:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]:block_switches[i+1], col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i+1]]
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], col_block_number] = i
            behav.iloc[block_switches[i]:block_switches[i+1], col_trial_within_block_real] = \
            np.arange(len(behav.iloc[block_switches[i]:block_switches[i+1], col_block_number]))
        elif i == len(block_switches)-1:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:, col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]:, col_opto_probabilityLeft_past] = \
                behav['probabilityLeft'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:, col_trial_within_block] = \
                np.arange(-5, len(behav) - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:, col_block_number] = i
            behav.iloc[block_switches[i]:, col_trial_within_block_real] = np.arange(len(behav.iloc[block_switches[i]:, col_block_number]))
        else:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_probabilityLeft_past] = \
                behav['probabilityLeft'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_trial_within_block] = \
                np.arange(-5, block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], col_block_number] = i
            behav.iloc[block_switches[i]:block_switches[i+1], col_trial_within_block_real] = \
                np.arange(len(behav.iloc[block_switches[i]:block_switches[i+1], col_block_number]))
    #Assign next block to negative within trial
    behav['block_number_real'] = behav['block_number'].copy()
    behav.loc[behav['trial_within_block']<0,'block_number'] = \
                behav.loc[behav['trial_within_block']<0,'block_number']+1
    behav.loc[behav['trial_within_block']>=0,'probabilityLeft_next'] = np.nan
    behav.loc[behav['trial_within_block']<0,'probabilityLeft_past'] = np.nan
    return behav

def add_transition_info(ses_d, trials_forward=20):
    trials_back=5 # Current cannot be changed
    ses_df = ses_d.copy()
    ses_df['choice_1'] = ses_df['choice']>0
    ses_df['transition_analysis'] = np.nan
    ses_df['transition_type'] = np.nan
    ses_df['transition_analysis_real'] = np.nan
    ses_df['transition_type_real'] = np.nan
    ses_df.loc[ses_df['trial_within_block']<trials_forward,'transition_analysis']=1
    ses_df['transition_analysis_real']=1
    for i in np.arange(len(ses_df['block_number'].unique())):
        if i>0:
            ses_ses_past = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']<0)]

            ses_ses_next = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']>=0)]

            ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']<0) ,'transition_type'] = \
                                    ses_ses_past['probabilityLeft'].astype(str)+ \
                                    ' to '\
                                   +ses_ses_past['probabilityLeft_next'].astype(str)

            ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis']==1) &
            (ses_df['trial_within_block']>=0) ,'transition_type'] = \
                                    ses_ses_next['probabilityLeft_past'].astype(str)+ \
                                     ' to '\
                                   +ses_ses_next['probabilityLeft'].astype(str)

            blocks = np.array([0.1,0.7])
            ses_ses_next_real = ses_df.loc[(ses_df['block_number']==i) &
            (ses_df['transition_analysis_real']==1) &
            (ses_df['trial_within_block_real']>=0)]
            past_block = blocks[blocks!=ses_ses_next_real['probabilityLeft'].iloc[0]]

            ses_df.loc[(ses_df['block_number_real']==i) &
            (ses_df['transition_analysis_real']==1) &
            (ses_df['trial_within_block_real']>=0) ,'transition_type_real'] = \
                                    ses_ses_next_real['probabilityLeft'].iloc[0].astype(str) + \
                                     ' to '\
                                   + str(past_block[0])

    return ses_df

def transition_plot(df, trials_forward=30):
            ses_df = df.copy()
            ses_df['id'] =  ses_df['mouse']*100 + ses_df['ses']
            negative_trials = ses_df.loc[ses_df['trial_within_block']<0].copy()
            positive_trials = ses_df.loc[ses_df['trial_within_block_real']>=0].copy()
            positive_trials['trial_within_block']=positive_trials['trial_within_block_real']
            positive_trials['transition_type']=positive_trials['transition_type_real']
            ses_df = pd.concat([negative_trials,positive_trials])
            ses_df = ses_df.loc[(ses_df['transition_type']=='0.7 to 0.1') |
                (ses_df['transition_type']=='0.1 to 0.7')].reset_index()
            ses_df_ses = ses_df.groupby(['id','transition_type','trial_within_block']).mean()['choice_1'].reset_index()
            sns.lineplot(data = ses_df_ses, x='trial_within_block', y='choice_1',
                ci=68, err_style='bars', style='transition_type')
            plt.ylim(0,1)
            plt.xlim(-5,trials_forward)
            plt.vlines(0,0,1,linestyles='dashed', color='k')
            plt.ylabel('% Right Choices')
            plt.xlabel('Trials from block switch')
            sns.despine()
            plt.title(str(len(df))+' trials')
            plt.show()

def laserdecayanalysis(df):
    # Remove early side block changes for performance
    ses_df_all = df.copy()
    ses_df_all['id'] =  ses_df_all['mouse']*100 + ses_df_all['ses']
    ses_df_all = assign_section(ses_df_all)
    ses_df_all['correct_choices']  = 1
    ses_df_all.loc[(ses_df_all['probabilityLeft']<0.5) & (ses_df_all['choice']<0.5), 'correct_choices' ] = 0
    ses_df_all.loc[(ses_df_all['probabilityLeft']>0.5) & (ses_df_all['choice']>0.5), 'correct_choices' ] = 0
    ses_df_all['side_block_change'] = ses_df_all['probabilityLeft'].diff()
    ses_df_all['late_side_block'] =  1
    ses_df_all_reduced_side = ses_df_all.loc[ses_df_all['tb']>5].copy()
    ses_df_all_reduced_side =  ses_df_all_reduced_side.groupby(['id', 'section']).mean()['correct_choices'].reset_index()
    sns.pointplot(data= ses_df_all_reduced_side, x='section', y='correct_choices', 
               ci=68)
    plt.xlabel('Trial bin (100trials)')
    plt.hlines(0.5,0,ses_df_all['section'].max(), linestyles='--', color='k')
    plt.ylabel('Fraction of High Prob Choices')
    plt.ylim(0.25,0.75)
    plt.xlim(0,5) # Section 5 is the last one with data from the 3 mice
    sns.despine()

# Load data
data=load_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_laser_only', trial_start=0, trial_end=None)
standata = make_stan_data_reduced(data)
standata_recovery = load_sim_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_laser_only', trial_start=0, trial_end=None)
# Q-learning
model_standard = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/standard_reduced_stay/output/summary.csv')
accu_standard = pd.DataFrame()
accu_standard['Accuracy'] = q_learning_model_reduced_stay(standata,saved_params=model_standard).groupby(['id']).mean()['acc']
accu_standard['Model'] = 'Q-Learning'
_, _, sim_standard = simulate_q_learning_model_reduced_stay(standata_recovery,saved_params=model_standard)
# REINFORCE
model_reinforce = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/REINFORCE_reduced/output/summary.csv')
accu_reinforce = pd.DataFrame()
accu_reinforce['Accuracy'] = reinforce_model_reduced(standata,saved_params=model_reinforce).groupby(['id']).mean()['acc']
accu_reinforce['Model'] = 'REINFORCE'
_, _, sim_reinforce = simulate_reinforce_reduced(standata_recovery,saved_params=model_reinforce)
# REINFORCE w stay
model_reinforce_stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/REINFORCE_mixedperseveration/output/summary.csv')
accu_reinforce_stay = pd.DataFrame()
accu_reinforce_stay['Accuracy'] = reinforce_model_reduced_stay(standata,saved_params=model_reinforce_stay).groupby(['id']).mean()['acc']
accu_reinforce_stay['Model'] = 'REINFORCE w stay'
_, _, sim_reinforce_stay = simulate_reinforce_reduced_stay(standata_recovery,saved_params=model_reinforce_stay)
# Uchida
model_uchida = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/standard_uchida/output/summary.csv')
accu_uchida = pd.DataFrame()
accu_uchida['Accuracy'] = reduced_uchida_model(standata,saved_params=model_uchida).groupby(['id']).mean()['acc']
accu_uchida['Model'] = 'Two-Accumulators'
_, _, sim_uchida = simulate_reduced_uchida_model(standata_recovery,saved_params=model_uchida)
# Double-update
double_model = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/double_update/output/summary.csv')
accu_double = pd.DataFrame()
accu_double['Accuracy'] = double_update_model(standata,saved_params=double_model).groupby(['id']).mean()['acc']
accu_double['Model'] = 'Double-update'
_, _, sim_double = simulate_double_update_model(standata_recovery,saved_params=double_model)


# Start summary figures
# 0. Raw data performance (justification for not decay)
realdata = stan_data_to_df_reduced(standata_recovery,standata)
realdata = realdata.rename(columns={'choices':'choice'})
realdata = trial_within_block(realdata)
realdata = add_transition_info(realdata, trials_forward=30)

simulation = sim_reinforce.copy()
simulation = simulation.rename(columns={'choices':'choice'})
simulation = trial_within_block(simulation)
simulation = add_transition_info(simulation, trials_forward=30)

transition_plot(realdata)
laserdecayanalysis(realdata)

transition_plot(simulation)
laserdecayanalysis(simulation)
# 1. Parameters
fig, ax = plt.subplots(2,2)
plt.sca(ax[0,0])
plot_params_reduced(model_standard, standata)
plt.title('Q-Learning')
plt.sca(ax[0,1])
plot_params_reduced(model_reinforce, standata)
plt.title('REINFORCE')
plt.sca(ax[1,0])
plot_params_reduced(model_reinforce_stay, standata)
plt.title('REINFORCE w stay')
plt.sca(ax[1,1])
plot_params_reduced(model_uchida, standata)
plt.title('Two accumulators')
plt.sca(ax[1,1])
plt.tight_layout(h_pad=-4.5, w_pad=-2)


plot_params_reduced(double_model, standata)
plt.title('Q-learning with state inference')

# 2. Accuracy
accu = pd.concat([accu_standard, accu_reinforce, accu_reinforce_stay, accu_uchida]).reset_index()
sns.lineplot(data=accu, x = 'Model', hue='id', y='Accuracy', palette=sns.color_palette(['black'], accu.id.unique().shape[0]))
sns.barplot(data=accu, x = 'Model', y='Accuracy', ci=68, palette='winter')
plt.ylim(0.2,1)
plt.legend().remove()
sns.despine()

# 3. Log-likelihood