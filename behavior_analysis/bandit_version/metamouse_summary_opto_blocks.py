from ibllib.io.extractors.biased_trials import extract_all
from rewardworld.behavior_analysis.bandit_version.full_bandit_fix_blocks import full_bandit_fix
from rewardworld.behavior_analysis.bandit_version.session_summary import *
from pathlib import Path

################################## Functions ##################################

def plot_GLM_blocks(params, acc):
    # Plot GLM coefficients
    try:
        sns.pointplot(data = params, x = 'trials_back',
                      y = 'coefficient', hue = 'type',
                      palette = {'choice*rewarded':'r', 'choice*unrewarded': 'b',
                                 'bias' : 'k', 'choice*laser' : 'coral',
                                 'choice*unlasered':'g'}, legend=False)
        plt.hlines(0,0,len(params.loc[params['type']=='choice*rewarded',
                                      'coefficient']),linestyles='dashed')
        plt.errorbar(np.arange(len(params.loc[params['type']=='bias',
                                              'coefficient'])), params.loc[params['type']=='bias',
                                                                           'coefficient'],
                     yerr= params.loc[params['type']=='bias',
                                      'ci_95'][0], color='k')
        plt.errorbar(np.arange(1,len(params.loc[params['type']=='choice*rewarded',
                                                'coefficient'])+1), params.loc[params['type']=='choice*rewarded',
                                                                               'coefficient'],
                     yerr= params.loc[params['type']=='choice*rewarded',
                                      'ci_95'], color='r')
        plt.errorbar(np.arange(1,len(params.loc[params['type']=='choice*rewarded',
                                                'coefficient'])+1), params.loc[params['type']=='choice*unrewarded',
                                                                               'coefficient'],
                     yerr= params.loc[params['type']=='choice*unrewarded',
                                      'ci_95'], color='b')
        plt.errorbar(np.arange(1,len(params.loc[params['type']=='choice*rewarded',
                                                'coefficient'])+1), params.loc[params['type']=='choice*laser',
                                                                               'coefficient'],
                     yerr= params.loc[params['type']=='choice*laser',
                                      'ci_95'], color='coral')
        plt.errorbar(np.arange(1,len(params.loc[params['type']=='choice*rewarded',
                                                'coefficient'])+1), params.loc[params['type']=='choice*unlasered',
                                                                               'coefficient'],
                     yerr= params.loc[params['type']=='choice*unlasered',
                                      'ci_95'], color='g')

    except:
        print('Cant plot, probably Param type not found!')

    plt.annotate('Accuracy:' + str(np.round(acc,2)), xy=[0,-0.2])
    # Statistical annotation
    for coef in params['index']:
        pvalue = params.loc[params['index'] == coef, 'pvalues']
        xy = params.loc[params['index'] == coef,
                        ['trials_back', 'coefficient']].to_numpy() + [0,0.05]
        if pvalue.to_numpy()[0] < 0.05:
            plt.annotate(num_star(pvalue.to_numpy()[0]),
                         xy= xy[0] ,
                         fontsize=20)
    sns.despine()

def fit_GLM_w_laser_10(data, trials_back=10):
    '''
    Parameters
    ----------
    behav : pandas dataframe
        dataframe with choice and outcome data, up to 5 trials back
        choice is signed, feedback = 1 for reward and 0 for error.
    Returns
    -------
    params : GLM coefficients
    acc: average crossvalidated accuracy
    '''
    #Remove no go trials and make choices in unit interval
    behav = data.copy()
    behav['choice']= behav['choice'].map({-1:0, 0:np.nan, 1:1})
    if np.isnan(behav['laser'][0]) == True:
        print('No laser data')
    else:
        endog, exog = patsy.dmatrices('choice ~ 1 + previous_choice_1:C(previous_outcome_1):C(previous_laser_1):C(previous_laser_block_1)'
                                      '+ previous_choice_2:C(previous_outcome_2):C(previous_laser_2):C(previous_laser_block_2)'
                                      '+ previous_choice_3:C(previous_outcome_3):C(previous_laser_3):C(previous_laser_block_3)'
                                      '+ previous_choice_4:C(previous_outcome_4):C(previous_laser_4):C(previous_laser_block_4)'
                                      '+ previous_choice_5:C(previous_outcome_5):C(previous_laser_5):C(previous_laser_block_5)'
                                      '+ previous_choice_6:C(previous_outcome_6):C(previous_laser_6):C(previous_laser_block_6)'
                                      '+ previous_choice_7:C(previous_outcome_7):C(previous_laser_7):C(previous_laser_block_7)'
                                      '+ previous_choice_8:C(previous_outcome_8):C(previous_laser_8):C(previous_laser_block_8)'
                                      '+ previous_choice_9:C(previous_outcome_9):C(previous_laser_9):C(previous_laser_block_9)'
                                      '+ previous_choice_10:C(previous_outcome_10):C(previous_laser_10):C(previous_laser_block_10)',
                                      data=behav,return_type='dataframe', NA_action='drop')
        columns_to_drop = []
        idx_to_drop = np.array([3,4,6,8])
        for i in np.arange(trials_back):
            cols_temp = idx_to_drop + 8*i
            columns_to_drop.append(cols_temp.tolist())
        columns_to_drop = np.array(columns_to_drop).flatten()
        exog = exog.drop(columns=exog.columns[columns_to_drop])

        exog.rename(columns={'Intercept': 'bias',
                             'previous_choice_1:C(previous_outcome_1)[0.0]:C(previous_laser_1)[0.0]:C(previous_laser_block_1)[0.0]':'choice*unrewarded_1',
                             'previous_choice_1:C(previous_outcome_1)[1.0]:C(previous_laser_1)[0.0]:C(previous_laser_block_1)[0.0]':'choice*rewarded_1',
                             'previous_choice_1:C(previous_outcome_1)[0.0]:C(previous_laser_1)[0.0]:C(previous_laser_block_1)[1.0]':'choice*unlasered_1',
                             'previous_choice_1:C(previous_outcome_1)[0.0]:C(previous_laser_1)[1.0]:C(previous_laser_block_1)[1.0]':'choice*laser_1',
                             'previous_choice_2:C(previous_outcome_2)[0.0]:C(previous_laser_2)[0.0]:C(previous_laser_block_2)[0.0]':'choice*unrewarded_2',
                             'previous_choice_2:C(previous_outcome_2)[1.0]:C(previous_laser_2)[0.0]:C(previous_laser_block_2)[0.0]':'choice*rewarded_2',
                             'previous_choice_2:C(previous_outcome_2)[0.0]:C(previous_laser_2)[0.0]:C(previous_laser_block_2)[1.0]':'choice*unlasered_2',
                             'previous_choice_2:C(previous_outcome_2)[0.0]:C(previous_laser_2)[1.0]:C(previous_laser_block_2)[1.0]':'choice*laser_2',
                             'previous_choice_3:C(previous_outcome_3)[0.0]:C(previous_laser_3)[0.0]:C(previous_laser_block_3)[0.0]':'choice*unrewarded_3',
                             'previous_choice_3:C(previous_outcome_3)[1.0]:C(previous_laser_3)[0.0]:C(previous_laser_block_3)[0.0]':'choice*rewarded_3',
                             'previous_choice_3:C(previous_outcome_3)[0.0]:C(previous_laser_3)[0.0]:C(previous_laser_block_3)[1.0]':'choice*unlasered_3',
                             'previous_choice_3:C(previous_outcome_3)[0.0]:C(previous_laser_3)[1.0]:C(previous_laser_block_3)[1.0]':'choice*laser_3',
                             'previous_choice_4:C(previous_outcome_4)[0.0]:C(previous_laser_4)[0.0]:C(previous_laser_block_4)[0.0]':'choice*unrewarded_4',
                             'previous_choice_4:C(previous_outcome_4)[1.0]:C(previous_laser_4)[0.0]:C(previous_laser_block_4)[0.0]':'choice*rewarded_4',
                             'previous_choice_4:C(previous_outcome_4)[0.0]:C(previous_laser_4)[0.0]:C(previous_laser_block_4)[1.0]':'choice*unlasered_4',
                             'previous_choice_4:C(previous_outcome_4)[0.0]:C(previous_laser_4)[1.0]:C(previous_laser_block_4)[1.0]':'choice*laser_4',
                             'previous_choice_5:C(previous_outcome_5)[0.0]:C(previous_laser_5)[0.0]:C(previous_laser_block_5)[0.0]':'choice*unrewarded_5',
                             'previous_choice_5:C(previous_outcome_5)[1.0]:C(previous_laser_5)[0.0]:C(previous_laser_block_5)[0.0]':'choice*rewarded_5',
                             'previous_choice_5:C(previous_outcome_5)[0.0]:C(previous_laser_5)[0.0]:C(previous_laser_block_5)[1.0]':'choice*unlasered_5',
                             'previous_choice_5:C(previous_outcome_5)[0.0]:C(previous_laser_5)[1.0]:C(previous_laser_block_5)[1.0]':'choice*laser_5',
                             'previous_choice_6:C(previous_outcome_6)[0.0]:C(previous_laser_6)[0.0]:C(previous_laser_block_6)[0.0]':'choice*unrewarded_6',
                             'previous_choice_6:C(previous_outcome_6)[1.0]:C(previous_laser_6)[0.0]:C(previous_laser_block_6)[0.0]':'choice*rewarded_6',
                             'previous_choice_6:C(previous_outcome_6)[0.0]:C(previous_laser_6)[0.0]:C(previous_laser_block_6)[1.0]':'choice*unlasered_6',
                             'previous_choice_6:C(previous_outcome_6)[0.0]:C(previous_laser_6)[1.0]:C(previous_laser_block_6)[1.0]':'choice*laser_6',
                             'previous_choice_7:C(previous_outcome_7)[0.0]:C(previous_laser_7)[0.0]:C(previous_laser_block_7)[0.0]':'choice*unrewarded_7',
                             'previous_choice_7:C(previous_outcome_7)[1.0]:C(previous_laser_7)[0.0]:C(previous_laser_block_7)[0.0]':'choice*rewarded_7',
                             'previous_choice_7:C(previous_outcome_7)[0.0]:C(previous_laser_7)[0.0]:C(previous_laser_block_7)[1.0]':'choice*unlasered_7',
                             'previous_choice_7:C(previous_outcome_7)[0.0]:C(previous_laser_7)[1.0]:C(previous_laser_block_7)[1.0]':'choice*laser_7',
                             'previous_choice_8:C(previous_outcome_8)[0.0]:C(previous_laser_8)[0.0]:C(previous_laser_block_8)[0.0]':'choice*unrewarded_8',
                             'previous_choice_8:C(previous_outcome_8)[1.0]:C(previous_laser_8)[0.0]:C(previous_laser_block_8)[0.0]':'choice*rewarded_8',
                             'previous_choice_8:C(previous_outcome_8)[0.0]:C(previous_laser_8)[0.0]:C(previous_laser_block_8)[1.0]':'choice*unlasered_8',
                             'previous_choice_8:C(previous_outcome_8)[0.0]:C(previous_laser_8)[1.0]:C(previous_laser_block_8)[1.0]':'choice*laser_8',
                             'previous_choice_9:C(previous_outcome_9)[0.0]:C(previous_laser_9)[0.0]:C(previous_laser_block_9)[0.0]':'choice*unrewarded_9',
                             'previous_choice_9:C(previous_outcome_9)[1.0]:C(previous_laser_9)[0.0]:C(previous_laser_block_9)[0.0]':'choice*rewarded_9',
                             'previous_choice_9:C(previous_outcome_9)[0.0]:C(previous_laser_9)[0.0]:C(previous_laser_block_9)[1.0]':'choice*unlasered_9',
                             'previous_choice_9:C(previous_outcome_9)[0.0]:C(previous_laser_9)[1.0]:C(previous_laser_block_9)[1.0]':'choice*laser_9',
                             'previous_choice_10:C(previous_outcome_10)[0.0]:C(previous_laser_10)[0.0]:C(previous_laser_block_10)[0.0]':'choice*unrewarded_10',
                             'previous_choice_10:C(previous_outcome_10)[1.0]:C(previous_laser_10)[0.0]:C(previous_laser_block_10)[0.0]':'choice*rewarded_10',
                             'previous_choice_10:C(previous_outcome_10)[0.0]:C(previous_laser_10)[0.0]:C(previous_laser_block_10)[1.0]':'choice*unlasered_10',
                             'previous_choice_10:C(previous_outcome_10)[0.0]:C(previous_laser_10)[1.0]:C(previous_laser_block_10)[1.0]':'choice*laser_10'},
                    inplace = True)
    #Reset index
    exog = exog.reset_index(drop=True)
    endog = endog.reset_index(drop=True)
    # Add laser unrewarded
    # Fit model
    logit_model = sm.Logit(endog, exog)
    res = logit_model.fit_regularized(disp=False) # run silently
    # Organise parameters
    params = pd.DataFrame({'coefficient':res.params, 'pvalues': res.pvalues,
                           'ci_95':res.conf_int()[0]-res.params}).reset_index()
    params['trials_back'] = np.nan
    params['type'] = np.nan
    params.loc[params['index']=='choice*unrewarded_1', ['trials_back', 'type']] = [1, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_2', ['trials_back', 'type']] = [2, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_3', ['trials_back', 'type']] = [3, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_4', ['trials_back', 'type']] = [4, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_5', ['trials_back', 'type']] = [5, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_6', ['trials_back', 'type']] = [6, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_7', ['trials_back', 'type']] = [7, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_8', ['trials_back', 'type']] = [8, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_9', ['trials_back', 'type']] = [9, 'choice*unrewarded']
    params.loc[params['index']=='choice*unrewarded_10', ['trials_back', 'type']] = [10, 'choice*unrewarded']
    params.loc[params['index']=='choice*rewarded_1', ['trials_back', 'type']] = [1, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_2', ['trials_back', 'type']] = [2, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_3', ['trials_back', 'type']] = [3, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_4', ['trials_back', 'type']] = [4, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_5', ['trials_back', 'type']] = [5, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_6', ['trials_back', 'type']] = [6, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_7', ['trials_back', 'type']] = [7, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_8', ['trials_back', 'type']] = [8, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_9', ['trials_back', 'type']] = [9, 'choice*rewarded']
    params.loc[params['index']=='choice*rewarded_10', ['trials_back', 'type']] = [10, 'choice*rewarded']
    params.loc[params['index']=='bias', ['trials_back', 'type']] = [0, 'bias']
    params.loc[params['index']=='choice*laser_1', ['trials_back', 'type']] = [1, 'choice*laser']
    params.loc[params['index']=='choice*laser_2', ['trials_back', 'type']] = [2, 'choice*laser']
    params.loc[params['index']=='choice*laser_3', ['trials_back', 'type']] = [3, 'choice*laser']
    params.loc[params['index']=='choice*laser_4', ['trials_back', 'type']] = [4, 'choice*laser']
    params.loc[params['index']=='choice*laser_5', ['trials_back', 'type']] = [5, 'choice*laser']
    params.loc[params['index']=='choice*laser_6', ['trials_back', 'type']] = [6, 'choice*laser']
    params.loc[params['index']=='choice*laser_7', ['trials_back', 'type']] = [7, 'choice*laser']
    params.loc[params['index']=='choice*laser_8', ['trials_back', 'type']] = [8, 'choice*laser']
    params.loc[params['index']=='choice*laser_9', ['trials_back', 'type']] = [9, 'choice*laser']
    params.loc[params['index']=='choice*laser_10', ['trials_back', 'type']] = [10, 'choice*laser']
    params.loc[params['index']=='choice*unlasered_1', ['trials_back', 'type']] = [1, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_2', ['trials_back', 'type']] = [2, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_3', ['trials_back', 'type']] = [3, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_4', ['trials_back', 'type']] = [4, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_5', ['trials_back', 'type']] = [5, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_6', ['trials_back', 'type']] = [6, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_7', ['trials_back', 'type']] = [7, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_8', ['trials_back', 'type']] = [8, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_9', ['trials_back', 'type']] = [9, 'choice*unlasered']
    params.loc[params['index']=='choice*unlasered_10', ['trials_back', 'type']] = [10, 'choice*unlasered']

    # Calculate accuracy in crossvalidated version
    acc = np.array([])
    kf = KFold(n_splits=5, shuffle=True)
    for train, test in kf.split(endog):
        X_train, X_test, y_train, y_test = exog.loc[train], exog.loc[test], \
                                           endog.loc[train], endog.loc[test]
        # fit again
        logit_model = sm.Logit(y_train, X_train)
        res = logit_model.fit_regularized(disp=False) # run silently
        # compute the accuracy on held-out data [from Luigi]:
        # suppose you are predicting Pr(Left), let's call it p,
        # the % match is p if the actual choice is left, or 1-p if the actual choice is right
        # if you were to simulate it, in the end you would get these numbers
        y_test['pred'] = res.predict(X_test)
        y_test.loc[y_test['choice'] == 0, 'pred'] = 1 - y_test.loc[y_test['choice'] == 0, 'pred']
        acc = np.append(acc, y_test['pred'].mean())
    return params , np.mean(acc)

######################################################################################################
gen_path = Path('/Volumes/witten/Alex/Data/Subjects/opto_chr2_bandit/pilot_bandit_laser_blocks/chr2')
ses_df=pd.DataFrame()
for animal in sorted(gen_path.iterdir()):
        try:
            for day in sorted(animal.iterdir()):
                try:
                    for ses in day.iterdir():
                        try:
                            print(ses)
                            #extract_all(ses, save=True)
                            #full_bandit_fix(ses._str)
                            ses_df_t = load_session_dataframe(ses._str)
                            ses_df = pd.concat([ses_df,ses_df_t])
                        except:
                            continue
                except:
                    continue
        except:
            continue

ses_df = ses_df.reset_index()
ses_df = ses_df.iloc[:,1:]

# Create new variables
ses_df['repeated'] = ses_df['choice']==ses_df['previous_choice_1']
sns.barplot(data=ses_df, x='laser_block', y='repeated', hue='previous_outcome_1', ci=68)

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


params, acc = fit_GLM_w_laser_10(ses_df)
plot_GLM_blocks(params,acc)
