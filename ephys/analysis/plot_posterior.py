import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
from pathlib import Path
import glob

# LOAD MODEL
SUMMARY_CSV = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output'
CHAIN_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output/chains'
LOAD_DF = False
REEXTRACT = False
ROOT_FOLDER =  '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced'
FIG_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard/figures'
# Load data
if LOAD_DF==False:
    root_path = Path(ROOT_FOLDER)
    psy = pd.DataFrame()
    for animal in root_path.iterdir():
        if animal.is_dir():
            for day in animal.iterdir():
                if day.is_dir():
                    for ses in day.iterdir():
                        if ses.is_dir():
                            if REEXTRACT==True:
                                full_bandit_fix(ses.as_posix())
                            mouse_psy = pd.DataFrame()
                            mouse_psy['feedback'] = \
                                1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)
                            mouse_psy['choices'] = \
                                1*((-1*np.load(ses.joinpath('alf', '_ibl_trials.choice.npy')))>0)
                            mouse_psy['laser_block'] = \
                                np.load(ses.joinpath('alf', '_ibl_trials.opto_block.npy'))
                            mouse_psy['laser'] = mouse_psy['feedback']*mouse_psy['laser_block']
                            mouse_psy['reward'] = mouse_psy['feedback']*(1*(mouse_psy['laser_block']==0))
                            mouse_psy['mouse'] = animal.name
                            mouse_psy['ses'] = ses.name
                            mouse_psy['date'] = day.name
                            psy = pd.concat([psy, mouse_psy])
                        else:
                            continue
                else:
                    continue
        else:
            continue
else:
    psy = pd.read_pickle("psy.pkl")

# Prepara data for Stan​
# 1) optomization variables
NS = len(psy['mouse'].unique())
NSESS = psy.groupby(['mouse','date','ses']).size().reset_index()['mouse'].value_counts().max() # Maximum number of sessions
NT = psy.groupby(['mouse','date','ses']).size().reset_index()[0].max()
NSxNSESS = NS*NSESS
NT_all = np.zeros(NSxNSESS)
sub_idx = np.zeros(NSxNSESS) #  Mapping parameter to subject
for i in np.arange(psy['mouse'].unique().shape[0]):
    sub_idx[i*NSESS:(i+1)*NSESS]=i
sess_idx =  np.zeros(NSxNSESS) # Mapping parameter to session

# 2) trial variables
r = np.zeros([NS,NSESS,NT])
c = np.zeros([NS,NSESS,NT])
l = np.zeros([NS,NSESS,NT])
for ns, mouse in enumerate(psy['mouse'].unique()):
    animal = psy.loc[psy['mouse']==mouse]
    counter=0
    for d, day in enumerate(animal['date'].unique()):
        day_s = animal.loc[animal['date']==day]
        for nsess, ses in enumerate(day_s['ses'].unique()):
            session = day_s.loc[day_s['ses']==ses]
            r[ns, counter, :len(session)] = session['reward']
            c[ns, counter, :len(session)] = session['choices']
            l[ns, counter, :len(session)] = session['laser']
            sess_idx[(ns*NSESS)+counter] = counter
            NT_all[(ns*NSESS)+counter] = len(session)
            counter+=1

standata = {'NS': int(NS) ,'NT': int(NT),'NSESS': int(NSESS),
           'r':r.astype(int), 'c':c.astype(int), 'l':l.astype(int),
           'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
           'sess_idx':np.array(sess_idx).astype(int)+1,
           'sub_idx':np.array(sub_idx).astype(int)+1}



# Functions
def inv_logit(arr):
        '''Elementwise inverse logit (logistic) function.'''
        return 1 / (1 + np.exp(-arr))
def phi_approx(arr):
        '''Elementwise fast approximation of the cumulative unit normal.
        For details, see Bowling et al. (2009). "A logistic approximation
        to the cumulative normal distribution."'''
        return inv_logit(0.07056 * arr ** 3 + 1.5976 * arr)

def load_cmd_stan_output(folder):
    pathlib = folder
    chains = glob.glob(folder + '/*.csv')
    output = pd.DataFrame()
    for c,i in enumerate(chains):
        chain = pd.read_csv(i
            , skiprows= \
            np.concatenate([np.arange(42), np.array([44,45,46,43]),
            np.arange(1547,1552)]))
        chain['chain'] = c+1
        output = pd.concat([output,chain])
    return output

# Load fitting results
N_MOUSE = NS
N_SES = NSESS
SES_LEN = NT_all.reshape(6,6).T
summary = pd.read_csv(SUMMARY_CSV)
level_0 = ['betam', 'laserm', 'staym', 'alpham', 'alphalaserm', 'alphastaym', 'sidem']
level_0_colors = ['blue', 'magenta', 'gray', 'blue', 'magenta', 'gray', 'k']
level_0_names = ['β Water', 'β Laser', 'β Stay', 'α Water', 'α Laser', 'α Stay', 'Side Bias']
level_1 = ['beta_mouse[%s]', 'laser_mouse[%s]', 'stay_mouse[%s]', 'alpha_mouse[%s]',
            'alphalaser_mouse[%s]', 'alphastay_mouse[%s]', 'side_mouse[%s]']
level_2 = ['beta_ses[%s]','laser_ses[%s]', 'stay_ses[%s]', 'alpha_ses[%s]',
            'alphalaser_ses[%s]', 'alphastay_ses[%s]', 'sides[%s]']
level_2_chains = ['beta_ses.%s','laser_ses.%s', 'stay_ses.%s', 'alpha_ses.%s',
            'alphalaser_ses.%s', 'alphastay_ses.%s', 'sides.%s']



# Prepare dataframes
# Level_1
level_1_df = pd.DataFrame()
for m in np.arange(N_MOUSE):
    n=m+1
    for var in level_1:
        var_m = var %n
        temp_df = pd.DataFrame()
        if 'alpha' in var:
            temp_df['center'] = phi_approx(summary.loc[summary['name']==var_m, 'Mean'])
            temp_df['error_l'] = phi_approx(summary.loc[summary['name']==var_m, '5%'])
            temp_df['error_h'] = phi_approx(summary.loc[summary['name']==var_m, '95%'])
        else:
            temp_df['center'] = summary.loc[summary['name']==var_m, 'Mean']
            temp_df['error_l'] = summary.loc[summary['name']==var_m, '5%']
            temp_df['error_h'] = summary.loc[summary['name']==var_m, '95%']
        temp_df['var'] = var
        temp_df['mouse'] = n
        level_1_df = pd.concat([level_1_df, temp_df])
# Level_2
level_2_df = pd.DataFrame()
for m in np.arange(N_MOUSE):
    n=m+1
    for ses in np.arange(N_SES):
        ses_n = ses + 1
        for var in level_2:
            var_s = var % (n*ses_n)
            temp_df = pd.DataFrame()
            if 'alpha' in var:
                temp_df['center'] = phi_approx(summary.loc[summary['name']==var_s, 'Mean'])
                temp_df['error_l'] = phi_approx(summary.loc[summary['name']==var_s, '5%'])
                temp_df['error_h'] = phi_approx(summary.loc[summary['name']==var_s, '95%'])
            else:
                temp_df['center'] = summary.loc[summary['name']==var_s, 'Mean']
                temp_df['error_l'] = summary.loc[summary['name']==var_s, '5%']
                temp_df['error_h'] = summary.loc[summary['name']==var_s, '95%']
            temp_df['var'] = var
            temp_df['mouse'] = n
            temp_df['ses'] = ses_n
            level_2_df = pd.concat([level_2_df, temp_df])

# Plot level_0 and level_1
plt.subplot()
centers = np.zeros(len(level_0))
errors = np.zeros([len(level_0),2])
for i, var in enumerate(level_0):
    if 'alpha' in var:
        errors[i,1] = phi_approx(summary.loc[summary['name']==var, '95%'])
        centers[i]  = phi_approx(summary.loc[summary['name']==var, 'Mean'])
        errors[i,0] = phi_approx(summary.loc[summary['name']==var, '5%'])
    else:
        errors[i,1] = summary.loc[summary['name']==var, '95%']
        centers[i]  = summary.loc[summary['name']==var, 'Mean']
        errors[i,0] = summary.loc[summary['name']==var, '5%']
yerr = np.c_[centers-errors[:,0],errors[:,1]-centers ].T
sns.pointplot(x=level_0, y=centers, join=False, palette=level_0_colors,
                markers='_', scale=2)
plt.errorbar(range(len(level_0)), centers, yerr=yerr, ls = 'none',
            ecolor=level_0_colors, elinewidth=2)
sns.pointplot(data=level_1_df, x='var', y='center', join=False, hue='mouse',
              scale=0.8, legend=False)
plt.xticks(range(len(level_0)), level_0_names, rotation=45)
sns.despine()
plt.savefig(FIG_FOLDER + '/level0.pdf')

# Plot level_1 and level_2
# Determine order
level_1_df['var_order']=np.nan
level_1_df['color']='k'
for i, var in enumerate(level_1):
    level_1_df.loc[level_1_df['var']==var,'var_order']=i
    level_1_df.loc[level_1_df['var']==var,'color']=level_0_colors[i]
level_1_df = level_1_df.sort_values(by=['var_order','mouse'])

level_2_df['var_order']=np.nan
level_2_df['color']='k'
for i, var in enumerate(level_2):
    level_2_df.loc[level_2_df['var']==var,'var_order']=i
    level_2_df.loc[level_2_df['var']==var,'color']=level_0_colors[i]
level_2_df = level_2_df.sort_values(by=['var_order','mouse'])

# Delete data from sessions with 0 data
clean_level_2_df = pd.DataFrame()
clean_level_2_df['ses_order']=np.nan
level_2_df['ses_order']=np.nan

for i in np.arange(len(level_2_df)):
    m = level_2_df.iloc[i,:]['mouse']
    s = level_2_df.iloc[i,:]['ses']
    if SES_LEN[s-1, m-1]!=0:
        level_2_df.iloc[i,level_2_df.columns.get_loc('ses_order')] = \
        (m-1) + ((np.where(np.isin(level_2, level_2_df.iloc[[i]]['var']))[0][0])*N_MOUSE)
        clean_level_2_df = pd.concat([clean_level_2_df, level_2_df.iloc[[i]]])

# Plot
plt.subplot()
plt.errorbar(range(len(level_0)*N_MOUSE), level_1_df.center, yerr = [level_1_df.center-level_1_df.error_l,
            level_1_df.error_h - level_1_df.center],ls = 'none', ecolor=level_1_df.color, elinewidth=2)
sns.pointplot(x=np.arange(len(level_0)*N_MOUSE), y=level_1_df.center, join=False, palette=level_1_df.color,
                markers='_', scale=2)
sns.scatterplot(data=clean_level_2_df, x='ses_order', y='center', hue='ses',
                legend=False, zorder=2, palette='tab10')
plt.xticks(np.arange(N_MOUSE/2,len(level_0)*N_MOUSE,N_MOUSE), level_0_names)
sns.despine()
plt.savefig(FIG_FOLDER + '/level1.pdf')

# Plot sessions posteriors

fit = load_cmd_stan_output(CHAIN_FOLDER)

for v, var in enumerate(level_2_chains):
    fig, ax = plt.subplots(SES_LEN.shape[0], SES_LEN.shape[1], sharex=True, sharey=True)
    for mouse in np.arange(SES_LEN.shape[1]):
        for ses in np.arange(SES_LEN.shape[0]):
            if SES_LEN[ses,mouse]!=0:
                flat_ses = mouse * N_MOUSE + ses + 1
                plt.sca(ax[ses, mouse])
                if 'alpha' in var:
                    sns.kdeplot(phi_approx(fit[var %flat_ses]), color=level_0_colors[v])
                else:
                    sns.kdeplot(fit[var %flat_ses], color=level_0_colors[v])
                plt.ylabel(' ')
                plt.xlabel(' ')
    fig.suptitle(level_0_names[v], fontsize=20)
    fig.text(0.5, 0.01, 'Coefficient', ha='center')
    fig.text(0.01, 0.5, 'Density', va='center', rotation='vertical')
    plt.tight_layout()
    sns.despine()
    fig.savefig(FIG_FOLDER + '/' +level_0_names[v]+'_transformed.pdf')

for v, var in enumerate(level_2_chains):
    fig, ax = plt.subplots(SES_LEN.shape[0], SES_LEN.shape[1], sharex=True, sharey=True)
    for mouse in np.arange(SES_LEN.shape[1]):
        for ses in np.arange(SES_LEN.shape[0]):
            if SES_LEN[ses,mouse]!=0:
                flat_ses = mouse * N_MOUSE + ses + 1
                plt.sca(ax[ses, mouse])
                sns.kdeplot(fit[var %flat_ses], color=level_0_colors[v])
                plt.ylabel(' ')
                plt.xlabel(' ')
    fig.suptitle(level_0_names[v], fontsize=20)
    fig.text(0.5, 0.01, 'Coefficient', ha='center')
    fig.text(0.01, 0.5, 'Density', va='center', rotation='vertical')
    plt.tight_layout()
    sns.despine()
    fig.savefig(FIG_FOLDER + '/' +level_0_names[v]+'.pdf')

for v, var in enumerate(level_2_chains):
    fig, ax = plt.subplots(SES_LEN.shape[0], SES_LEN.shape[1], sharex=True, sharey=True)
    for mouse in np.arange(SES_LEN.shape[1]):
        for ses in np.arange(SES_LEN.shape[0]):
            if SES_LEN[ses,mouse]!=0:
                flat_ses = mouse * N_MOUSE + ses + 1
                plt.sca(ax[ses, mouse])
                sns.histplot(fit[var %flat_ses], color=level_0_colors[v])
                plt.ylabel(' ')
                plt.xlabel(' ')
    fig.suptitle(level_0_names[v], fontsize=20)
    fig.text(0.5, 0.01, 'Coefficient', ha='center')
    fig.text(0.01, 0.5, 'Density', va='center', rotation='vertical')
    plt.tight_layout()
    sns.despine()
    fig.savefig(FIG_FOLDER + '/' +level_0_names[v]+'_histogram.pdf')
