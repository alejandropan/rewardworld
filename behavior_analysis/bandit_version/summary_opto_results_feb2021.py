from behavior_analysis.bandit_version.summary_all import listdir_nohidden
from behavior_analysis.bandit_version.block_bandit_summary import *
import scipy
root = '/Volumes/witten/Alex/Data/Subjects/opto_chr2_bandit/50_50_reward_laser_blocks_stable/chr2'
chr2 = load_condition(root)
mice = ['dop_13','dop_15','dop_21', 'dop_24']
chr2= chr2.loc[np.isin(chr2['mouse'],mice)]
# Divide and reconcatenate (avoid session break issues in transition labelling)
dop_13 = addtransition_info_animal(chr2.loc[chr2['mouse']=='dop_13'])
dop_15 = add_transition_info_animal(chr2.loc[chr2['mouse']=='dop_15'])
dop_21 = add_transition_info_animal(chr2.loc[chr2['mouse']=='dop_21'])
dop_24 = add_transition_info_animal(chr2.loc[chr2['mouse']=='dop_24'])
chr2 = pd.concat([dop_13,dop_15,dop_21,dop_24])

# Plot
plot_dataframe_opto(dop_13, root)
plot_dataframe_opto(dop_15, root)
plot_dataframe_opto(dop_21, root)
plot_dataframe_opto(dop_24, root)


###################
#Functions
###################

def plot_dataframe_opto(ses_df, root, individual_anmial=True):
    ###################
    # 1. Plot Stay Prob
    ###################
    ses_df = ses_df.reset_index()
    ses_df = ses_df.iloc[:,1:]
    ses_df['repeated'] = ses_df['choice']==ses_df['previous_choice_1']
    sns.barplot(data=ses_df, x='laser_block', y='repeated', hue='previous_outcome_1', ci=68)
    plt.savefig(root +'/' + str(ses_df['mouse'].unique()[0]) +'/stay_sum.png')
    plt.close()
    ###################
    # 2. Plot GLM
    ###################
    # New GLM
    ses_df['laser']=0
    ses_df.loc[ses_df['laser_block']==1,'laser'] = \
        ses_df.loc[ses_df['laser_block']==1,'outcome']
    ses_df['previous_laser_1']=ses_df['laser'].shift(1)
    ses_df['previous_laser_2']=ses_df['laser'].shift(2)
    ses_df['previous_laser_3']=ses_df['laser'].shift(3)
    ses_df['previous_laser_4']=ses_df['laser'].shift(4)
    ses_df['previous_laser_5']=ses_df['laser'].shift(5)
    ses_df['previous_laser_6']=ses_df['laser'].shift(6)
    ses_df['previous_laser_7']=ses_df['laser'].shift(7)
    ses_df['previous_laser_8']=ses_df['laser'].shift(8)
    ses_df['previous_laser_9']=ses_df['laser'].shift(9)
    ses_df['previous_laser_10']=ses_df['laser'].shift(10)
    ses_df.loc[ses_df['laser_block']==1,'outcome'] = 0
    ses_df['previous_outcome_1'] = ses_df['outcome'].shift(1)
    ses_df['previous_outcome_2'] = ses_df['outcome'].shift(2)
    ses_df['previous_outcome_3'] = ses_df['outcome'].shift(3)
    ses_df['previous_outcome_4'] = ses_df['outcome'].shift(4)
    ses_df['previous_outcome_5'] = ses_df['outcome'].shift(5)
    ses_df['previous_outcome_6'] = ses_df['outcome'].shift(6)
    ses_df['previous_outcome_7'] = ses_df['outcome'].shift(7)
    ses_df['previous_outcome_8'] = ses_df['outcome'].shift(8)
    ses_df['previous_outcome_9'] = ses_df['outcome'].shift(9)
    ses_df['previous_outcome_10'] = ses_df['outcome'].shift(10)
    ses_df['previous_laser_block_1'] = ses_df['laser_block'].shift(1)
    ses_df['previous_laser_block_2'] = ses_df['laser_block'].shift(2)
    ses_df['previous_laser_block_3'] = ses_df['laser_block'].shift(3)
    ses_df['previous_laser_block_4'] = ses_df['laser_block'].shift(4)
    ses_df['previous_laser_block_5'] = ses_df['laser_block'].shift(5)
    ses_df['previous_laser_block_6'] = ses_df['laser_block'].shift(6)
    ses_df['previous_laser_block_7'] = ses_df['laser_block'].shift(7)
    ses_df['previous_laser_block_8'] = ses_df['laser_block'].shift(8)
    ses_df['previous_laser_block_9'] = ses_df['laser_block'].shift(9)
    ses_df['previous_laser_block_10'] = ses_df['laser_block'].shift(10)
    # Fit and plot
    params, acc = fit_GLM_w_laser_10(ses_df)
    plot_GLM_blocks(params,acc)
    plt.savefig(root +'/' + str(ses_df['mouse'].unique()[0]) +'/glm.png')
    plt.close()
    ######################################
    # 3. Plot choices per block
    ######################################
    sns.barplot(data=ses_df, x='probabilityLeft', y=1*(ses_df['choice']>0),
                hue = 'laser_block')
    plt.ylim(0,1)
    plt.savefig(root +'/' + str(ses_df['mouse'].unique()[0]) +'/block_choices.png')
    plt.close()
    ######################################
    # 4. Plot transition
    ######################################
    #Remove right to right / left to left transitions
    ses_df_plot = ses_df.loc[ses_df['block_number']!=0]
    ses_df_plot = ses_df_plot.loc[ses_df_plot['probability_current']!=ses_df_plot['probability_past']]
    sns.lineplot(data=ses_df_plot, x='trial_within_block', y='choice_1',
                     hue='transition', ci=68, err_style='bars')
    plt.xlim(-5,15)
    plt.vlines(0,0,1,linestyles='dashed', color='k')
    plt.xlabel('Trial from block switch')
    plt.ylabel('Fraction of Right Choices')
    L=plt.legend()
    plt.annotate('n = '+ str(len(ses_df_plot['mouse'].unique()))+' mice', (6,0))
    plt.savefig(root +'/' + str(ses_df['mouse'].unique()[0]) +'/transition.png')
    plt.close()
    ######################################
    # 5. Plot choice probability as a function of trial
    ######################################
    fig, ax = plt.subplots(1,2, sharey=True)
    plt.sca(ax[0])
    sns.lineplot(data=ses_df.loc[(ses_df['laser_block']==0)],x='trial_within_block',y='choice_1',
                 hue='probabilityLeft', ci=68,palette='hls')
    plt.xlim(0,15)
    plt.sca(ax[1])
    sns.lineplot(data=ses_df.loc[(ses_df['laser_block']==1)],x='trial_within_block',y='choice_1',
                  hue='probabilityLeft', ci=68, palette='hls')
    plt.xlim(0,15)
    plt.savefig(root +'/' + str(ses_df['mouse'].unique()[0]) +'/transition_2.png')
    plt.close()
    #####################################
    # 5. Plot reaction times
    #####################################
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    plt.sca(ax[0])
    ses_df['rt'] = ses_df['first_movement'] - ses_df['cue_on_trigger']
    sns.histplot(data=ses_df,x='rt', hue='laser_block')
    plt.xlim(0,2)
    plt.xlabel('Reaction Time (s)')
    plt.sca(ax[1])
    ses_df['decision_time'] = ses_df['response_time'] - ses_df['cue_on_trigger']
    sns.histplot(data=ses_df,x='decision_time', hue='laser_block')
    plt.xlim(0,2)
    plt.xlabel('Decision Time (s)')
    plt.tight_layout()
    plt.savefig(root +'/' + str(ses_df['mouse'].unique()[0]) +'/rt.png')
    plt.close()
    #####################################
    # 6. Debiasing by block
    #####################################
    sns.barplot(data = ses_df.loc[ses_df['trial_within_block']>-1],y='trial_within_block',
                x='probabilityLeft', hue='laser_block')
    plt.ylabel('Average number of trial in block (debiasing measure)')
    plt.savefig(root +'/' + str(ses_df['mouse'].unique()[0]) +'/debias.png')
    plt.close()
    #####################################
    # 7. Example of session
    #####################################
    if individual_anmial==True:
        eg_ses=np.random.choice(ses_df['date'].unique())
        example = ses_df.loc[ses_df['date']==eg_ses].copy().reset_index()
        params, _, dm = fit_GLM_w_laser_10(example, save_dm=True)
        assert len(example) == len(dm)
        value_laser, value_reward , value_choice = simulate_from_laser_model(dm, params)
        example['value_laser']=value_laser
        example['value_reward']=value_reward
        example['value_choice']=value_choice
        example['choice_l'] = (example['choice']==-1)*1
        example['choice_r'] = (example['choice']==1)*1
        example['reward_r'] = example['outcome']*example['choice_r']*(1*(example['laser_block']!=1))
        example['reward_l'] = example['outcome']*example['choice_l']*(1*(example['laser_block']!=1))
        example['laser_r'] = example['outcome']*example['choice_r']*example['laser_block']
        example['laser_l'] = example['outcome']*example['choice_l']*example['laser_block']
        example['probabilityRight']=0.1
        example.loc[example['probabilityLeft']==0.1, 'probabilityRight'] = 0.7
        fig, ax =plt.subplots(1,figsize=(10,5))
        plt.sca(ax)
        plt.plot(example['choice_r'].rolling(10, center=True).mean(),color='orange')
        #plt.plot(example['choice_l'].rolling(10, center=True).mean(),color='blue')
        plt.plot(example['value_laser'].rolling(10, center=True).mean(),color='magenta', linestyle='dashed')
        plt.plot(example['value_reward'].rolling(10, center=True).mean(),color='green', linestyle='dashed')
        plt.plot(example['value_choice'].rolling(10, center=True).mean(),color='k', linestyle='dashed')
        plt.plot(example['probabilityRight']/5+1.50,color='k',
                 linestyle='--', alpha =0.5)
        plt.xlim(0,400)
        plt.yticks(ticks=[0.0,0.25,0.5,0.75, 1.0],labels=[0.0,0.25,0.5,0.75,1.0])
        plt.vlines(np.where(example['reward_l']==1),1.37,1.47, color='blue')
        plt.vlines(np.where(example['reward_r']==1),1.25,1.35, color='orange')
        plt.vlines(np.where(example['laser_l']==1),1.37,1.47, color='magenta')
        plt.vlines(np.where(example['laser_r']==1),1.25,1.35, color='magenta')
        plt.vlines(np.where((example['potential_reward_r']==1)&
                            (example['laser_block']==0)),1.25,1.30, color='orange')
        plt.vlines(np.where((example['potential_reward_l']==1)&
                            (example['laser_block']==0)),1.37,1.43, color='blue')
        plt.vlines(np.where((example['potential_reward_r']==1)&
                            (example['laser_block']==1)),1.25,1.30, color='magenta')
        plt.vlines(np.where((example['potential_reward_l']==1)&
                            (example['laser_block']==1)),1.37,1.43, color='magenta')
        sns.despine()
        plt.xlabel('Trial')
        plt.ylabel('Choice probability')
        plt.savefig(root +'/' + str(ses_df['mouse'].unique()[0]) +'/example.png')
        plt.close()


