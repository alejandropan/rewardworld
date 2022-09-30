from model_comparison_accu import *

def trial_within_block(behav):
    behav['trial_within_block'] = np.nan
    behav['block_number'] = np.nan
    behav['trial_within_block_real'] = np.nan
    behav['probabilityLeft_next'] = np.nan # this is for plotting trials before block change
    behav['opto_block_next'] = np.nan # this is for plotting trials before block change
    behav['probabilityLeft_past'] = np.nan # this is for plotting trials before block change
    behav['opto_block_past'] = np.nan # this is for plotting trials before block change
    behav['block_change'] = np.concatenate([np.zeros(1),
                                            1*(np.diff(behav['probabilityLeft'])!=0)])
    block_switches = np.concatenate([np.zeros(1),
                                     behav.loc[behav['block_change']==1].index]).astype(int)
    col_trial_within_block = np.where(behav.columns == 'trial_within_block')[0][0]
    col_probabilityLeft_next = np.where(behav.columns == 'probabilityLeft_next')[0][0]
    col_block_number = np.where(behav.columns == 'block_number')[0][0]
    col_opto_block_next = np.where(behav.columns == 'opto_block_next')[0][0]
    col_opto_block_past = np.where(behav.columns == 'opto_block_past')[0][0]
    col_opto_probabilityLeft_past = np.where(behav.columns == 'probabilityLeft_past')[0][0]
    col_trial_within_block_real = np.where(behav.columns == 'trial_within_block_real')[0][0]
    for i in np.arange(len(block_switches)):
        if i == 0:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]:block_switches[i+1], col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i+1]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_next] = \
                        behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_next] = \
            np.arange(block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], col_block_number] = i
            behav.iloc[block_switches[i]:block_switches[i+1], col_trial_within_block_real] = \
            np.arange(len(behav.iloc[block_switches[i]:block_switches[i+1], col_block_number]))
        elif i == len(block_switches)-1:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:, col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]-5:, col_opto_block_next] = \
                behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:, col_opto_probabilityLeft_past] = \
                behav['probabilityLeft'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:, col_opto_block_past] = \
                behav['opto_block'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:, col_trial_within_block] = \
                np.arange(-5, len(behav) - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:, col_block_number] = i
            behav.iloc[block_switches[i]:, col_trial_within_block_real] = np.arange(len(behav.iloc[block_switches[i]:, col_block_number]))
        else:
            # Trial within block and probability for plotting
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_probabilityLeft_next] = \
                behav['probabilityLeft'][block_switches[i]]
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_opto_block_next] = \
                behav['opto_block'][block_switches[i]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_probabilityLeft_past] = \
                behav['probabilityLeft'][block_switches[i-1]]
            behav.iloc[block_switches[i]:block_switches[i+1], col_opto_block_past] = \
                behav['opto_block'][block_switches[i-1]]
            behav.iloc[block_switches[i]-5:block_switches[i+1], col_trial_within_block] = \
                np.arange(-5, block_switches[i+1] - block_switches[i])
            # Block number
            behav.iloc[block_switches[i]:block_switches[i+1], col_block_number] = i
            behav.iloc[block_switches[i]:block_switches[i+1], col_trial_within_block_real] = np.arange(len(behav.iloc[block_switches[i]:block_switches[i+1], col_block_number]))
    #Assign next block to negative within trial
    behav['block_number_real'] = behav['block_number'].copy()
    behav.loc[behav['trial_within_block']<0,'block_number'] = \
                behav.loc[behav['trial_within_block']<0,'block_number']+1
    behav.loc[behav['trial_within_block']>=0,'opto_block_next'] = np.nan
    behav.loc[behav['trial_within_block']>=0,'probabilityLeft_next'] = np.nan
    behav.loc[behav['trial_within_block']<0,'opto_block_past'] = np.nan
    behav.loc[behav['trial_within_block']<0,'probabilityLeft_past'] = np.nan
    return behav

def transform_data_laserdecay(original, bins = 8):
    '''
    original(df): dataframe with trial data (tested with the output of stan_data_to_df function)
    bins(int): Number of bins for classifying session trials
    '''
    sesdata = original.reset_index()
    sesdata['outcome'] = sesdata['water']+sesdata['laser']
    sesdata['First_last_100'] = np.nan
    sesdata['real_index'] = np.nan
    sesdata.loc[sesdata['index']<100, '0_100']=1
    for mouse in sesdata['mouse'].unique():
            select_1 = sesdata.loc[sesdata['mouse']==mouse]
            for ses in select_1['ses'].unique():
                ses_max = sesdata.loc[(sesdata['mouse']==mouse) & (sesdata['ses']==ses), 'index'].max()
                sesdata.loc[(sesdata['mouse']==mouse) & (sesdata['ses']==ses) & (sesdata['index']>ses_max-100), 'First_last_100']=0
                sesdata.loc[(sesdata['mouse']==mouse) & (sesdata['ses']==ses), 'real_index'] = np.arange(len(sesdata.loc[(sesdata['mouse']==mouse) & (sesdata['ses']==ses), 'real_index']))
                sesdata.loc[(sesdata['mouse']==mouse) & (sesdata['ses']==ses), 'sesid'] = (mouse*100)+ses

    # Bin data
    BINS = bins
    sesdata['ses_id'] = sesdata['mouse']*100 +  sesdata['ses']
    sesdata['section'] = pd.cut(sesdata['real_index'],bins=BINS)
    sesdata['outcome'] = sesdata['water'] + sesdata['laser']
    sesdata['correct_choices'] = 0
    sesdata.loc[(sesdata['probabilityLeft']==0.1) & (sesdata['choice']==1), 'correct_choices'] = 1
    sesdata.loc[(sesdata['probabilityLeft']==0.7) & (sesdata['choice']==0), 'correct_choices'] = 1

    return sesdata

def add_transition_info(ses_d, trials_forward=10):
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

############################
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
############################

    return ses_df

def laserdecayanalysis(df, BINS=8):
    # Remove early side block changes for performance
    ses_df_all = df.copy()
    ses_df_all['side_block_change'] = ses_df_all['probabilityLeft'].diff()
    ses_df_all['late_side_block'] =  1
    ses_df_all_reduced_side = ses_df_all.loc[ses_df_all['tb']>5].copy()

    sns.pointplot(data= ses_df_all_reduced_side, x='section', y='correct_choices', 
                hue='opto_block', ci=68, palette=['dodgerblue','orange'])
    plt.xlabel('Trial bin (100trials)')
    plt.hlines(0.5,0,BINS, linestyles='--', color='k')
    plt.xticks(np.arange(BINS),np.arange(BINS))
    plt.ylabel('Fraction of High Prob Choices')
    plt.ylim(0.25,0.75)
    sns.despine()

    # Repeat plots

    ses_df_all['repeat'] = ses_df_all['choice']==ses_df_all['choice'].shift(1)
    ses_df_all['prev_outcome'] = ses_df_all['outcome'].shift(1)

    fig, ax = plt.subplots(1,2, sharey=True)
    plt.sca(ax[0])
    sns.pointplot(data= ses_df_all.loc[ses_df_all['opto_block']==1], x='prev_outcome', 
                y='repeat', hue='section', ci=68, 
                palette='YlOrBr')
    plt.title('Laser Blocks')
    plt.xlabel('Previous Laser')
    plt.sca(ax[1])
    sns.pointplot(data= ses_df_all.loc[ses_df_all['opto_block']==0], x='prev_outcome', 
                y='repeat', hue='section', ci=68, 
                palette='Blues')
    plt.title('Water Blocks')
    plt.ylabel('Fraction of repeated choices')
    plt.xlabel('Previous water')


    fig, ax = plt.subplots(1,6, sharey=True)
    for i,section in enumerate(ses_df_all.section.unique()[:6]):
        plt.sca(ax[i])
        sns.pointplot(data= ses_df_all.loc[ses_df_all['section']==section], x='prev_outcome', 
                    y='repeat', hue='opto_block', ci=68, 
                    palette=['dodgerblue', 'orange'])
        plt.title('Bin: '+ str(section))
        plt.xlabel('Previous Outcome')
        plt.ylabel('Repeated last choice (Fraction)')


    ses_dl_all_reduced = pd.DataFrame()
    for ses in ses_df_all['ses_id'].unique():
        r = ses_df_all.loc[ses_df_all['ses_id']==ses]
        ses_dl_all_reduced = pd.concat([ses_dl_all_reduced,r[:-150]])

    print(ses_dl_all_reduced.groupby(['ses_id']).mean()['outcome'].median())
    fig, ax = plt.subplots(2,6, sharey=True)
    for i,section in enumerate(ses_dl_all_reduced.section.unique()[:6]):
        plt.sca(ax[0,i])
        sns.pointplot(data= ses_dl_all_reduced.loc[(ses_dl_all_reduced['section']==section) &
                        (ses_dl_all_reduced['opto_block']==0)], x='prev_outcome', 
                        y='repeat', hue='mouse', ci=68, palette='Blues')
        plt.title('Bin: '+ str(section))
        plt.xlabel('Previous Outcome')
        plt.ylabel('Repeated last choice (Fraction)')  
        plt.legend().remove()
        plt.sca(ax[1,i])
        sns.pointplot(data= ses_dl_all_reduced.loc[(ses_dl_all_reduced['section']==section) &
                        (ses_dl_all_reduced['opto_block']==1)], x='prev_outcome', 
                        y='repeat', hue='mouse', ci=68, palette='YlOrBr')
        plt.title('Bin: '+ str(section))
        plt.xlabel('Previous Outcome')
        plt.ylabel('Repeated last choice (Fraction)')  
        plt.legend().remove()

    # Transition plots
    fig, ax = plt.subplots()
    transition_plot(ses_df_all)
    plot_transition_by_section(ses_df_all)
    # Remove early identity block changes #
    ses_df_all['identity_block_change'] = ses_df_all['opto_block'].diff()
    reduced = ses_df_all.loc[(ses_df_all['trial_within_block']<=15)].copy().reset_index()
    reduced['include']=1
    trials_eliminated=[]
    for idx in reduced.loc[reduced['identity_block_change']==1].index:
        twb = reduced.iloc[idx,reduced.columns.get_loc('trial_within_block')]+1
        if twb>-5:
            to_nan = np.arange(idx+1,idx+1+(16-twb)).astype(int)
            trials_eliminated.append(len(to_nan))
            try:
                reduced.iloc[to_nan,reduced.columns.get_loc('include')] = 0
            except:
                continue #Try statement for trials to nan outside of range (i.e block change happened just before the end of the session)
    reduced = reduced.loc[reduced['include']==1]
    reduced = reduced.iloc[:,1:]
    transition_plot(reduced)
    plot_transition_by_section(reduced)

def plot_laserdecay_summary(sesdata, BINS=8):
    sns.pointplot(data= sesdata, x='section', y='correct_choices', hue='opto_block', ci=68, palette=['dodgerblue','orange'])
    plt.legend().remove()
    plt.xlabel('Trial bin (100trials)')
    plt.hlines(0.5,0,BINS, linestyles='--', color='k')
    plt.xticks(np.arange(BINS),np.arange(BINS))
    plt.ylabel('Fraction of High Prob Choices')
    plt.ylim(0.25,0.75)
    sns.despine()

def plot_transition_by_section(df):
    # You need df formatted as the ouput of my alf objects
        for section in  df.section.unique():
            ses_df = df.loc[df['section']==section].copy()
            negative_trials = ses_df.loc[ses_df['trial_within_block']<0].copy()
            positive_trials = ses_df.loc[ses_df['trial_within_block_real']>=0].copy()
            positive_trials['trial_within_block']=positive_trials['trial_within_block_real']
            positive_trials['transition_type']=positive_trials['transition_type_real']
            ses_df = pd.concat([negative_trials,positive_trials])
            sns.lineplot(data = ses_df.loc[(ses_df['transition_type']=='0.7 to 0.1') |
                (ses_df['transition_type']=='0.1 to 0.7')].reset_index(), x='trial_within_block', y='choice_1',
                ci=68, hue='opto_block', err_style='bars', style='transition_type', palette=['dodgerblue','orange'])
            plt.ylim(0,1)    
            plt.xlim(-5,15)
            plt.vlines(0,0,1,linestyles='dashed', color='k')
            plt.ylabel('% Right Choices')
            plt.xlabel('Trials from block switch')
            sns.despine()
            plt.title('Bin: ' +str(section) + ' ' + str(len(df.loc[df['section']==section]))+' trials')
            plt.show()

def transition_plot(df):
            ses_df = df.copy()
            negative_trials = ses_df.loc[ses_df['trial_within_block']<0].copy()
            positive_trials = ses_df.loc[ses_df['trial_within_block_real']>=0].copy()
            positive_trials['trial_within_block']=positive_trials['trial_within_block_real']
            positive_trials['transition_type']=positive_trials['transition_type_real']
            ses_df = pd.concat([negative_trials,positive_trials])
            sns.lineplot(data = ses_df.loc[(ses_df['transition_type']=='0.7 to 0.1') |
                (ses_df['transition_type']=='0.1 to 0.7')].reset_index(), x='trial_within_block', y='choice_1',
                ci=68, hue='opto_block', err_style='bars', style='transition_type', palette=['dodgerblue','orange'])
            plt.ylim(0,1)
            plt.xlim(-5,15)
            plt.vlines(0,0,1,linestyles='dashed', color='k')
            plt.ylabel('% Right Choices')
            plt.xlabel('Trials from block switch')
            sns.despine()
            plt.title(str(len(df))+' trials')
            plt.show()

def alphadecay_plot(data):
    data['id'] =  data['mouse']*100 + data['ses']
    fig, ax= plt.subplots(1,2)
    plt.sca(ax[0])
    sns.lineplot(data=data.reset_index(), x ='index', y='alphalaser', hue='id', palette = 'viridis')
    plt.xlabel('Trial No')
    plt.xlim(0,200)
    plt.legend().remove()
    plt.sca(ax[1])
    sns.lineplot(data=data.reset_index(), x ='index', y='Qlaser', hue='id', palette = 'viridis')
    plt.xlabel('Trial No')
    plt.xlim(0,200)
    plt.legend().remove()
    plt.show()
    fig, ax= plt.subplots(1,2)
    plt.sca(ax[0])
    sns.lineplot(data=data.reset_index(), x ='index', y='alphalaser', palette = 'viridis')
    plt.xlabel('Trial No')
    plt.xlim(0,200)
    plt.legend().remove()
    plt.sca(ax[1])
    sns.lineplot(data=data.reset_index(), x ='index', y='Qlaser', palette = 'viridis')
    plt.xlabel('Trial No')
    plt.xlim(0,200)
    plt.legend().remove()
    plt.show()


if __name__=='__main__':
    standata = make_stan_data(load_data())
    standata_recovery = load_sim_data() # Raw data for simulations

    # Accuracy analysis:
    accu_standard = 0.7064414537611433
    betalaserdecayaccu = 0.7072184584282222

    realdata = stan_data_to_df(standata_recovery,standata)
    realdata = realdata.rename(columns={'choices':'choice'})
    realdata = transform_data_laserdecay(realdata, bins=8)
    realdata = realdata.rename(columns={'laser_block':'opto_block'})
    realdata = trial_within_block(realdata)
    realdata = add_transition_info(realdata, trials_forward=10)





    if LASERDECAY==True:
        standard_laserdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_alphalaserdecay/output/summary.csv')
        laserdecayaccu = q_learning_model_alphalaserdecay(standata,saved_params=standard_laserdecay)['acc'].unique().mean()
        simdata_standard = pd.DataFrame()
        for i in np.arange(5):
            _, _, _, df = simulate_q_learning_model_alphalaserdecay(standata_recovery,saved_params=standard_laserdecay)
            simdata_standard = pd.concat([simdata_standard,df])
        simdata_standard = simdata_standard.rename(columns={'choices':'choice'})
        simdata_standard = transform_data_laserdecay(simdata_standard, bins=8)
        simdata_standard = simdata_standard.rename(columns={'laser_block':'opto_block'})
        simdata_standard = trial_within_block(simdata_standard)
        simdata_standard = add_transition_info(simdata_standard, trials_forward=10)
        plot_laserdecay_summary(simdata_standard, BINS=8)
        laserdecayanalysis(simdata_standard, BINS=8)


    if standard_75==True:
        standard_75 = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/standard_75/output/summary.csv')
        simdata_standard = pd.DataFrame()
        for i in np.arange(10):
            _, _, _, df = simulate_q_learning_model_new(standata_recovery,saved_params=standard_75)
            simdata_standard = pd.concat([simdata_standard,df])
        simdata_standard = simdata_standard.rename(columns={'choices':'choice'})
        simdata_standard = transform_data_laserdecay(simdata_standard, bins=8)
        simdata_standard = simdata_standard.rename(columns={'laser_block':'opto_block'})
        simdata_standard = trial_within_block(simdata_standard)
        simdata_standard = add_transition_info(simdata_standard, trials_forward=10)
        plot_laserdecay_summary(simdata_standard, BINS=8)

    if standard_150rest==True:
        standard_150rest = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/standard_rest/output/summary.csv')
        simdata_standard = pd.DataFrame()
        for i in np.arange(10):
            _, _, _, df = simulate_q_learning_model_new(standata_recovery,saved_params=standard_150rest)
            simdata_standard = pd.concat([simdata_standard,df])
        simdata_standard = simdata_standard.rename(columns={'choices':'choice'})
        simdata_standard = transform_data_laserdecay(simdata_standard, bins=8)
        simdata_standard = simdata_standard.rename(columns={'laser_block':'opto_block'})
        simdata_standard = trial_within_block(simdata_standard)
        simdata_standard = add_transition_info(simdata_standard, trials_forward=10)
        plot_laserdecay_summary(simdata_standard, BINS=8)


    if reinforce_75==True:
        reinforce_75 = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/reinforce_75/output/summary.csv')
        simdata_reinforce = pd.DataFrame()
        for i in np.arange(10):
            _, _, _, df =  simulate_reinforce_winloss_stay(standata_recovery,saved_params=reinforce_75)
            simdata_reinforce = pd.concat([simdata_reinforce,df])
        simdata_reinforce = simdata_reinforce.rename(columns={'choices':'choice'})
        simdata_reinforce = transform_data_laserdecay(simdata_reinforce, bins=8)
        simdata_reinforce = simdata_reinforce.rename(columns={'laser_block':'opto_block'})
        simdata_reinforce = trial_within_block(simdata_reinforce)
        simdata_reinforce = add_transition_info(simdata_reinforce, trials_forward=10)
        plot_laserdecay_summary(simdata_reinforce, BINS=8)

    if reinforce_150rest==True:
        reinforce_150rest = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/reinforce_rest/output/summary.csv')
        simdata_reinforce = pd.DataFrame()
        for i in np.arange(10):
            _, _, _, df =  simulate_reinforce_winloss_stay(standata_recovery,saved_params=reinforce_150rest)
            simdata_reinforce = pd.concat([simdata_reinforce,df])
        simdata_reinforce = simdata_reinforce.rename(columns={'choices':'choice'})
        simdata_reinforce = transform_data_laserdecay(simdata_reinforce, bins=8)
        simdata_reinforce = simdata_reinforce.rename(columns={'laser_block':'opto_block'})
        simdata_reinforce = trial_within_block(simdata_reinforce)
        simdata_reinforce = add_transition_info(simdata_reinforce, trials_forward=10)
        plot_laserdecay_summary(simdata_reinforce, BINS=8)





    if LASERDECAY_everytrial==True:
        standard_laserdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_alphalaserdecay_every_trial/output/summary.csv')
        laserdecayeverytrialaccu = q_learning_model_alphalaserdecay_everytrial(standata,saved_params=standard_laserdecay)['acc'].unique().mean()
        _, _, _, simdata_standard = simulate_q_learning_model_alphalaserdecay_everytrial(standata_recovery,saved_params=standard_laserdecay)
        simdata_standard = simdata_standard.rename(columns={'choices':'choice'})
        simdata_standard = transform_data_laserdecay(simdata_standard, bins=8)
        simdata_standard = simdata_standard.rename(columns={'laser_block':'opto_block'})
        simdata_standard = trial_within_block(simdata_standard)
        simdata_standard = add_transition_info(simdata_standard, trials_forward=10)
        plot_laserdecay_summary(simdata_standard, BINS=8)


    if REINFORCE==True:
        REINFORCEalphalaserdecaystay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywin_mixedstay/output/summary.csv')
        _, _, _, simdata_reinforce = simulate_reinforce_alphalaserdecay_win_stay(standata_recovery,saved_params=REINFORCEalphalaserdecaystay)
        simdata_reinforce = simdata_reinforce.rename(columns={'choices':'choice'})
        simdata_reinforce = transform_data_laserdecay(simdata_reinforce, bins=8)
        simdata_reinforce = simdata_reinforce.rename(columns={'laser_block':'opto_block'})
        simdata_reinforce = trial_within_block(simdata_reinforce)
        simdata_reinforce = add_transition_info(simdata_reinforce, trials_forward=10)
        laserdecayanalysis(simdata_reinforce, BINS=8)
        plot_laserdecay_summary(simdata_reinforce, BINS=8)








    simdata_reinforce1=pd.DataFrame()
    for i in np.arange(10):
        _, _, _, simdata_reinforce = simulate_reinforce_alphalaserdecay_win_stay(standata_recovery,saved_params=REINFORCEalphalaserdecaystay)
        simdata_reinforce = simdata_reinforce.rename(columns={'choices':'choice'})
        simdata_reinforce = transform_data_laserdecay(simdata_reinforce, bins=8)
        simdata_reinforce = simdata_reinforce.rename(columns={'laser_block':'opto_block'})
        simdata_reinforce = trial_within_block(simdata_reinforce)
        simdata_reinforce = add_transition_info(simdata_reinforce, trials_forward=10)
        simdata_reinforce1=pd.concat([simdata_reinforce1,simdata_reinforce])



    # Add transition data to dataframe
    laserdecayanalysis(realdata, BINS=8)
    plot_laserdecay_summary(realdata, BINS=8)