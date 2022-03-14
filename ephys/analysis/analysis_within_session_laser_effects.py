import numpy as np
import seaborn as sns
import logistic_regression as lr
from matplotlib import pyplot as plt
import pandas as pd
from model_comparison_accu import *

psy = load_data()
standata = make_stan_data(psy)
standata_recovery = load_sim_data() # Raw data for simulations
original = stan_data_to_df(standata_recovery,standata)

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
    sesdata.loc[(sesdata['probabilityLeft']==0.1) & (sesdata['choices']==1), 'correct_choices'] = 1
    sesdata.loc[(sesdata['probabilityLeft']==0.7) & (sesdata['choices']==0), 'correct_choices'] = 1

    return sesdata

fig, ax  = plt.subplots(1,3)
plt.sca(ax[0])
sns.pointplot(data= sesdata.loc[sesdata['laser_block']==1], x='section', y='outcome', hue='sesid',  ci=0, palette='YlOrBr')
plt.legend().remove()
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.xlabel('Trial bin (50trials)')
plt.ylabel('Fraction rewarded')
plt.title('Laser block performance')
plt.sca(ax[1])

sns.pointplot(data= sesdata.loc[sesdata['laser_block']==0], x='section', y='outcome', hue='sesid',  ci=0, palette='Blues')
plt.legend().remove()
plt.title('Water block performance')
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.xlabel('Trial bin (50trials)')
plt.ylabel('Fraction rewarded')
plt.sca(ax[2])
sns.pointplot(data= sesdata, x='section', y='outcome',hue='laser_block', ci=68, palette=['dodgerblue','orange'])
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.xlabel('Trial bin (50trials)')
plt.title('Summary')
plt.ylabel('Fraction rewarded')
plt.tight_layout()

fig, ax  = plt.subplots(1,3)
plt.sca(ax[0])
sns.pointplot(data= sesdata.loc[(sesdata['probabilityLeft']==0.7)],linestyles='-' , x='section', y='outcome', style='laser_block', hue='sesid',  ci=0, palette='binary')
plt.legend().remove()
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.xlabel('Trial bin (50trials)')
plt.ylabel('Fraction rewarded')
plt.title('L Block')
plt.sca(ax[1])
sns.pointplot(data= sesdata.loc[(sesdata['probabilityLeft']==0.1)],linestyles='-' , x='section', y='outcome', style='laser_block', hue='sesid',  ci=0, palette='binary')
plt.legend().remove()
plt.title('R Block')
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.xlabel('Trial bin (50trials)')
plt.ylabel('Fraction rewarded')
plt.sca(ax[2])
sns.pointplot(data= sesdata.loc[(sesdata['probabilityLeft']==0.1)],linestyles='-', x='section', y='outcome',hue='laser_block', ci=68, palette=['dodgerblue','orange'])
sns.pointplot(data= sesdata.loc[(sesdata['probabilityLeft']==0.7)],linestyles='--', x='section', y='outcome',hue='laser_block', ci=68, palette=['dodgerblue','orange'])
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.xlabel('Trial bin (50trials)')
plt.title('Summary')
plt.ylabel('Fraction rewarded')
plt.legend().remove()
plt.tight_layout()

def plot_glm_sesdata(sesdata):
    ses_df = sesdata.copy()
    ses_df['choice']=ses_df['choices']
    ses_df['choice'] = ses_df['choice'].map({0:-1,1:1})
    ses_df = lr.add_laser_block_regressors(ses_df)
    ses_df = ses_df[['choice', 'laser',
        'previous_choice_1', 'previous_choice_2', 'previous_choice_3',
        'previous_choice_4', 'previous_choice_5', 'previous_choice_6',
        'previous_choice_7', 'previous_choice_8', 'previous_choice_9',
        'previous_choice_10', 'previous_laser_1', 'previous_laser_2',
        'previous_laser_3', 'previous_laser_4', 'previous_laser_5',
        'previous_laser_6', 'previous_laser_7', 'previous_laser_8',
        'previous_laser_9', 'previous_laser_10', 'previous_outcome_1',
        'previous_outcome_2', 'previous_outcome_3', 'previous_outcome_4',
        'previous_outcome_5', 'previous_outcome_6', 'previous_outcome_7',
        'previous_outcome_8', 'previous_outcome_9', 'previous_outcome_10',
        'previous_laser_block_1', 'previous_laser_block_2',
        'previous_laser_block_3', 'previous_laser_block_4',
        'previous_laser_block_5', 'previous_laser_block_6',
        'previous_laser_block_7', 'previous_laser_block_8',
        'previous_laser_block_9', 'previous_laser_block_10']]
    params = lr.fit_GLM_w_laser_10(ses_df,drop=False)
    plot_GLM_blocks(params, stars=False)

def plot_GLM_blocks(params, stars=False):
    # Plot GLM coefficients
    try:
        sns.pointplot(data = params, x = 'trials_back',
                      y = 'coefficient', hue = 'type',
                      palette = {'choice*rewarded':'dodgerblue', 'choice*unrewarded': 'lightskyblue',
                                 'bias' : 'k', 'choice*laser' : 'orange',
                                 'choice*unlasered':'navajowhite'}, legend=False)
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
                                      'ci_95'], color='dodgerblue')
        plt.errorbar(np.arange(1,len(params.loc[params['type']=='choice*rewarded',
                                                'coefficient'])+1), params.loc[params['type']=='choice*unrewarded',
                                                                               'coefficient'],
                     yerr= params.loc[params['type']=='choice*unrewarded',
                                      'ci_95'], color='lightskyblue')
        plt.errorbar(np.arange(1,len(params.loc[params['type']=='choice*rewarded',
                                                'coefficient'])+1), params.loc[params['type']=='choice*laser',
                                                                               'coefficient'],
                     yerr= params.loc[params['type']=='choice*laser',
                                      'ci_95'], color='orange')
        plt.errorbar(np.arange(1,len(params.loc[params['type']=='choice*rewarded',
                                                'coefficient'])+1), params.loc[params['type']=='choice*unlasered',
                                                                               'coefficient'],
                     yerr= params.loc[params['type']=='choice*unlasered',
                                      'ci_95'], color='navajowhite')
        plt.xlabel('Trials back')
        plt.ylabel('Regression Coefficient')
        plt.ylim(-0.3,1.3)

    except:
        print('Cant plot, probably Param type not found!')
    # Statistical annotation
    for coef in params['index']:
        pvalue = params.loc[params['index'] == coef, 'pvalues']
        xy = params.loc[params['index'] == coef,
                        ['trials_back', 'coefficient']].to_numpy() + [0,0.05]
        if stars==True:
            if pvalue.to_numpy()[0] < 0.05:
                plt.annotate(num_star(pvalue.to_numpy()[0]),
                             xy= xy[0] ,
                             fontsize=20)
    sns.despine()

fig, ax = plt.subplots(4,4, sharex=True,sharey=True)
for n,section in enumerate(sesdata.section.unique()):
        ses_mini = sesdata.loc[sesdata['section']==section].reset_index()
        plt.sca(ax[int(np.floor(n/4)),n%4])
        plot_glm_sesdata(ses_mini)
        plt.ylim(-1,2)
        plt.title('nbin='+str(n))
        plt.legend().remove()
        if int(np.floor(n/4))<3:
            ax[int(np.floor(n/4)),n%4].get_xaxis().set_visible(False)


############################################## Laser decay by animal #############################################

def plot_laserdecay_summary(sesdata, BINS=8):
    sns.pointplot(data= sesdata, x='section', y='correct_choices', hue='laser_block', ci=68, palette=['dodgerblue','orange'])
    plt.legend().remove()
    plt.xlabel('Trial bin (100trials)')
    plt.hlines(0.5,0,BINS, linestyles='--', color='k')
    plt.xticks(np.arange(BINS),np.arange(BINS))
    plt.ylabel('Fraction of High Prob Choices')
    plt.ylim(0.25,0.75)
    sns.despine()

#############################################

fig, ax = plt.subplots(7,6, sharex=True,sharey=True)
for mouse in sesdata.mouse.unique():
        ses_mini = sesdata.loc[sesdata['mouse']==mouse]
        for n, ses in enumerate(ses_mini.ses.unique()):
            ses_micro = ses_mini.loc[ses_mini['ses']==ses]
            plt.sca(ax[mouse,ses])
            sns.pointplot(data= ses_micro, x='section', y='correct_choices', hue='laser_block', ci=68, palette=['dodgerblue','orange'])
            plt.xlabel('Bin(100 trials)')
            plt.hlines(0.5,0,BINS, linestyles='--', color='k')
            plt.xticks(np.arange(BINS),np.arange(BINS))
            plt.ylabel(str(mouse_names[mouse]))
            plt.legend().remove()
            if n!=0:
                plt.ylabel(' ')
            if mouse!=6:
                plt.xlabel(' ')

### Group by histology
good_hist =[2, 4, 6]
average_hist=[0, 1]
bad_hist=[5, 3]
fig, ax = plt.subplots(1,3, sharey=True)
ses_bad = sesdata.loc[np.isin(sesdata['mouse'],bad_hist)].reset_index()
ses_average = sesdata.loc[np.isin(sesdata['mouse'],average_hist)].reset_index()
ses_good = sesdata.loc[np.isin(sesdata['mouse'],good_hist)].reset_index()
plt.sca(ax[0])
sns.pointplot(data= ses_bad, x='section', y='correct_choices', hue='laser_block', ci=68, palette=['dodgerblue','orange'])
plt.xlabel('Bin(100 trials)')
plt.hlines(0.5,0,BINS, linestyles='--', color='k')
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.ylabel('Fraction of High Prob Choices')
plt.title('Bad histology')
plt.sca(ax[1])
sns.pointplot(data= ses_average, x='section', y='correct_choices', hue='laser_block', ci=68, palette=['dodgerblue','orange'])
plt.xlabel('Bin(100 trials)')
plt.hlines(0.5,0,BINS, linestyles='--', color='k')
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.title('Average histology')
plt.sca(ax[2])
sns.pointplot(data= ses_good, x='section', y='correct_choices', hue='laser_block', ci=68, palette=['dodgerblue','orange'])
plt.xlabel('Bin(100 trials)')
plt.hlines(0.5,0,BINS, linestyles='--', color='k')
plt.xticks(np.arange(BINS),np.arange(BINS))
plt.title('Good histology')