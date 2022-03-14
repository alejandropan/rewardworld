from behavior_analysis.bandit_version.summary_all import listdir_nohidden
from behavior_analysis.bandit_version.block_bandit_summary import *
chr2 = load_condition('/Volumes/witten/Alex/Data/Subjects/opto_chr2_bandit/50_50_reward_laser_blocks_stable/chr2')
mice = ['dop_13','dop_15','dop_21', 'dop_24']
dop_13 = chr2.loc[chr2['mouse']=='dop_13']
dop_15 = chr2.loc[chr2['mouse']=='dop_15']
dop_21 = chr2.loc[chr2['mouse']=='dop_21']
dop_24 = chr2.loc[chr2['mouse']=='dop_24']

plot_stay_prob(chr2)

def load_condition(root):
    '''Load data from all animals in a condition (e.g all chr2)'''
    behav=pd.DataFrame()
    for mouse in listdir_nohidden(root):
        print(mouse)
        behav_m=pd.DataFrame()
        for day in listdir_nohidden(root+'/'+mouse):
            print(day)
            behav_d=pd.DataFrame()
            for ses in listdir_nohidden(root+'/'+mouse+'/'+day):
                print(ses)
                path = root+'/'+mouse+'/'+day+'/'+ses
                full_bandit_fix(path)
                behav_s = load_session_dataframe(path)
                behav_s['mouse']=mouse
                behav_d = pd.concat([behav_d, behav_s])
            behav_m = pd.concat([behav_m,behav_d])
        behav = pd.concat([behav, behav_m])
    return behav

def add_transition_info(ses_d, trials_back=5):
    ses_df = ses_d.copy()
    ses_df['choice_1'] = ses_df['choice']>0
    ses_df['probability_past'] = np.nan
    ses_df['probability_current'] = ses_df['probabilityLeft'].copy()
    ses_df['opto_past'] = np.nan
    ##
    ses_df.loc[ses_df['trial_within_block']<0,'block_number'] = \
        ses_df.loc[ses_df['trial_within_block']<0,'block_number']+1
    for block in range(ses_df.block_number.max().astype(int)):
        ses_df.loc[(ses_df['block_number']==block)&(ses_df['trial_within_block']<0),
              'probability_current'] = ses_df.loc[ses_df['block_number']==block,
                                                  'probability_current'].to_numpy()[trials_back+1]
        ses_df.loc[(ses_df['block_number']==block)&(ses_df['trial_within_block']<0),
                   'laser_block'] = ses_df.loc[ses_df['block_number']==block,
                                                       'laser_block'].to_numpy()[trials_back+1]
        if block > 0:
            ses_df.loc[ses_df['block_number']==block, 'probability_past']= \
                ses_df.loc[ses_df['block_number']==block-1, 'probabilityLeft'].to_numpy()[trials_back+1]
            ses_df.loc[ses_df['block_number']==block, 'opto_past'] = \
                ses_df.loc[ses_df['block_number']==block-1, 'laser_block'].to_numpy()[trials_back+1]
    ##
    # Define transition types
    ses_df.loc[ses_df['probability_past']>0.5,'probability_past'] = 'Left'
    ses_df.loc[ses_df['probability_past']!='Left','probability_past'] = 'Right'
    ses_df.loc[ses_df['probability_current']>0.5,'probability_current'] = 'Left'
    ses_df.loc[ses_df['probability_current']!='Left','probability_current'] = 'Right'
    ses_df['transition'] = ses_df['probability_past'].astype(str)+ ' to '\
                           +ses_df['probability_current'].astype(str)+ \
                           ' Opto'+ses_df['laser_block'].astype(str)
    return ses_df

# Add variables
dop_24 = pd.concat([add_transition_info(ses_df_2),add_transition_info(ses_df_3),
                    add_transition_info(ses_df_4)])
dop_24['mouse']='dop_24'
dop_21 = pd.concat([add_transition_info(ses_df_5),add_transition_info(ses_df_6)])
dop_21['mouse']='dop_21'
dop_15 = pd.concat([add_transition_info(ses_df_7),add_transition_info(ses_df_8),
                    add_transition_info(ses_df_9),add_transition_info(ses_df_10)])
dop_15['mouse']='dop_15'
dop_13 = pd.concat([add_transition_info(ses_df_11),add_transition_info(ses_df_12),
                    add_transition_info(ses_df_13),add_transition_info(ses_df_14)])
dop_13['mouse']='dop_13'
ses_df = pd.concat([dop_13,dop_15,dop_21,dop_24])

#Remove right to right / left to left transitions
ses_df_plot = ses_df.loc[ses_df['block_number']!=0]
ses_df_plot = ses_df.loc[ses_df['probability_current']!=ses_df['probability_past']]
pool = ses_df_plot.groupby(['trial_within_block', 'mouse','transition']).mean().reset_index()
sns.lineplot(data=ses_df_plot, x='trial_within_block', y='choice_1',
             hue='transition', ci=68, err_style='bars')
plt.xlim(-5,15)
plt.vlines(0,0,1,linestyles='dashed', color='k')
plt.xlabel('Trial from block switch')
plt.ylabel('Fraction of Right Choices')
L=plt.legend()
L.get_texts()[0].set_text('Right to Left Natural Reward')
L.get_texts()[1].set_text('Right to Left Natural Reward')
L.get_texts()[2].set_text('Right to Left Laser Reward')
L.get_texts()[3].set_text('Left to Right Laser Reward')
plt.annotate('n = '+ str(len(ses_df_plot['mouse'].unique()))+' mice', (6,0))
plt.show()

