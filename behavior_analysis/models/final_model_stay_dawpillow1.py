#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:52:51 2020

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 19:13:18 2020

@author: alex
"""
import numpy as np
import scipy.optimize as so
import time
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from rew_alf.data_organizers import *
import seaborn as sns
from scipy.stats.distributions import chi2
from scipy.stats import norm
import random
from matplotlib.lines import Line2D
import os
import glob
from os import path
from scipy.integrate import quad



def save_Q(psy, root = '/Volumes/witten/Alex/recordings_march_2020_dop'):
t'''
tParams:
troot: root folder for ephys sessions
tpsy: dataframe with all the data
tDescription:
tSave QL QR QL0 and QR0 files, QL0 and QR0 files are Q values in the absence
tof laser
t'''
tviruses = psy['virus'].unique()
tfor virus in viruses:
ttbyvirus = psy.loc[psy['virus'] == virus]
ttmice = byvirus['mouse_name'].unique()
ttfor mouse in mice:
tttbymouse = byvirus.loc[byvirus['mouse_name'] == mouse]
tttmouse_path = os.path.join(root, virus, mouse )
ttt# First check if mouse has ephys data
tttif os.path.exists(mouse_path):
ttttdates = glob.glob(path.join(mouse_path,  '*'))
tttt# Get all ephys dates
ttttfor d in dates:
tttttsessions = glob.glob(path.join(d, '*'))
tttttif len(sessions)>1:
ttttttprint('No current support for 2 sessions in a day'
ttttttt  + mouse + d)
tttttfor ses in sessions:
ttttttalf  =  os.path.join(ses, 'alf')
ttttttdate = d[-10:]
ttttttQ_files = ['QL', 'QR', 'QL0', 'QR0', 'ITIQL', 'ITIQR',
tttttttt   'ITIQL0', 'ITIQR0']
ttttttfor Q in Q_files:
tttttttqfile = os.path.join(alf,Q+'stay_ephysfit.npy')
tttttttfile = bymouse.loc[bymouse['ses'] == date,
ttttttttttttt Q].to_numpy()
tttttttif len(np.load(os.path.join(alf,'_ibl_trials.choice.npy'))) != len(file):
ttttttttprint('uniqual number of Q and t'
ttttttttt  + mouse + ses)
tttttttelse:
ttttttttnp.save(qfile, file)

def model_choice_raw_prob(psy, mouse, save = True):
t'''
tNotes: Plots psychometrics
t'''
tsns.set(style='white', font_scale = 2, rc={"lines.linewidth": 2.5})
tfig, ax =  plt.subplots( figsize = (10,10))
tplt.sca(ax)
tpsy_select = psy.loc[psy['mouse_name'] == mouse].copy()
tpsy_select['choice'] = psy_select['choice'] * -1
tpsy_select.loc[psy_select['choice'] == -1, 'choice'] = 0
tsns.lineplot(data = psy_select, x = 'signed_contrasts', y = 'choice',
ttttt hue = 'opto_block', hue_order = ['non_opto','L','R'],
ttttt palette = ['k','b','g'], ci =68, legend=None)
t# Plot model data with dash line
tsns.lineplot(data = psy_select, x = 'signed_contrasts', y = psy_select['pRight'],
ttttt hue = 'opto_block', hue_order = ['non_opto','L','R'],
ttttt palette = ['k','b','g'], ci =0, legend=None)
tax.lines[3].set_linestyle("--")
tax.lines[4].set_linestyle("--")
tax.lines[5].set_linestyle("--")
tax.set_ylabel('Fraction of Choices', fontsize =20)
tax.set_xlabel('Signed Contrast', fontsize =20)
tax.set_xlim(-0.3,0.3)
tax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
tax2.tick_params(axis='y', labelcolor='k')
tax2.set_ylabel('pRight', color='k', fontsize =20)  # we alread
tax.set_title(mouse, fontsize =20)
tlines = [Line2D([0], [0], color='black', linewidth=1.5, linestyle='-'),
tLine2D([0], [0], color='black', linewidth=1.5, linestyle='--')]
tlabels = ['Real Data', 'Model']
tax.axvline(0.3, ls='--', color = 'k')
tplt.legend(lines, labels)
tsns.despine()
tif save == True:
tt plt.savefig('model_choice_prob_raw'+mouse+'.png', dpi =300)



def plot_qr_trial_whole_dataset(psy_df, save= True):
tpsy_select = psy_df.copy()
tsns.set(style = 'white')
tfig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
tpalette ={'R':'g','L':'b','non_opto':'k'}
tplt.sca(ax[0])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y = psy_chr2['QR'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-ChR2')
tax[0].set_xlabel('Trial in block')
tax[0].set_ylabel('QR')

tplt.sca(ax[1])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y =  psy_chr2['QR'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-NpHR')
tax[1].set_xlabel('Trial in block')
tplt.tight_layout()
tsns.despine()

tif save ==True:
ttplt.savefig('qr_across_trials.svg', dpi =300)
ttplt.savefig('qr_across_trials.jpeg',  dpi =300)


def plot_ql_trial_whole_dataset(psy_df, save= True):
tpsy_select = psy_df.copy()
tsns.set(style = 'white')
tfig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
tpalette ={'R':'g','L':'b','non_opto':'k'}
tplt.sca(ax[0])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y = psy_chr2['QR'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-ChR2')
tax[0].set_xlabel('Trial in block')
tax[0].set_ylabel('QL')

tplt.sca(ax[1])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y =  psy_chr2['QL'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-NpHR')
tax[1].set_xlabel('Trial in block')
tplt.tight_layout()
tsns.despine()

tif save ==True:
ttplt.savefig('ql_across_trials.svg', dpi =300)
ttplt.savefig('ql_across_trials.jpeg',  dpi =300)


def plot_choice_trial_from_model(psy_df, save= True):
tpsy_select = psy_df.copy()
tsns.set(style = 'white')
t#Get index for trials of block start
tindex = psy_select.loc[psy['trial_within_block'] == 0, 'index']
tindex = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
tfor i in range(10):
ttfor idx in index-(i+1):
tttpsy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
tttt-(i+1)
tttpsy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
ttttpsy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]

tfig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
tpalette ={'R':'g','L':'b','non_opto':'k'}
tplt.sca(ax[0])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'pRight',
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(-10,50)
tplt.ylim(0,1)
tplt.title('VTA-ChR2')
tax[0].set_xlabel('Trial in block')
tax[0].set_ylabel('pRight')

tplt.sca(ax[1])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'pRight',
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(-10,50)
tplt.ylim(0,1)
tplt.title('VTA-NpHR')
tax[1].set_xlabel('Trial in block')
tplt.tight_layout()
tsns.despine()

tif save ==True:
ttplt.savefig('plot_choice_trial_from_model.svg', dpi =300)
ttplt.savefig('plot_choice_trial_from_model.jpeg',  dpi =300)


def plot_qr_trial_whole_dataset(psy_df, save= True):
tpsy_select = psy_df.copy()
tsns.set(style = 'white')
tfig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
tpalette ={'R':'g','L':'b','non_opto':'k'}
tplt.sca(ax[0])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y = psy_chr2['QR'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-ChR2')
tax[0].set_xlabel('Trial in block')
tax[0].set_ylabel('QR')

tplt.sca(ax[1])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y =  psy_chr2['QR'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-NpHR')
tax[1].set_xlabel('Trial in block')
tplt.tight_layout()
tsns.despine()

tif save ==True:
ttplt.savefig('qr_across_trials.svg', dpi =300)
ttplt.savefig('qr_across_trials.jpeg',  dpi =300)


def plot_ql_trial_whole_dataset(psy_df, save= True):
tpsy_select = psy_df.copy()
tsns.set(style = 'white')
tfig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
tpalette ={'R':'g','L':'b','non_opto':'k'}
tplt.sca(ax[0])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y = psy_chr2['QL'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-ChR2')
tax[0].set_xlabel('Trial in block')
tax[0].set_ylabel('QL')

tplt.sca(ax[1])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y =  psy_chr2['QL'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-NpHR')
tax[1].set_xlabel('Trial in block')
tplt.tight_layout()
tsns.despine()

tif save ==True:
ttplt.savefig('ql_across_trials.svg', dpi =300)
ttplt.savefig('ql_across_trials.jpeg',  dpi =300)



def model_performance(model_parameters, modelled_data, model_type= 'w_stay', save = True):
t'''
tNotes: Plots model accuracy and performance
t'''

tsns.set(style='white', font_scale = 2)
tnum_mice  = len(modelled_data['mouse_name'].unique())
tmice = modelled_data['mouse_name'].unique()
tmod_param  = model_parameters.loc[model_parameters['model_name'] == model_type]
tideal = modelled_data.copy()
tideal = ideal.groupby('mouse_name').mean()
tfor mouse in mice:
ttideal.loc[ideal.index == mouse, 'virus'] = \
tttmodel_parameters.loc[model_parameters['mouse'] == mouse, 'virus'][0]

tfig, ax =  plt.subplots(1,3, figsize=(20,10))
tplt.sca(ax[0])
tsns.barplot(data=mod_param, x = 'virus', y = 'accu', palette = ['dodgerblue', 'orange'],
tttt order=['chr2','nphr'])
tsns.swarmplot(data=mod_param, x = 'virus', y = 'accu', color='k', order=['chr2','nphr'])
tax[0].set_ylim(0,1)
tax[0].set_ylabel('Model Accuracy (%)')
tax[0].set_title('Model Accuracy')
tax[0].set_xlabel('Virus')
tplt.sca(ax[1])
tsns.barplot(data=ideal, x = 'virus', y = 'rewards', palette = ['dodgerblue', 'orange'],
tttt order=['chr2','nphr'])
tsns.swarmplot(data=ideal, x = 'virus', y = 'rewards', color='k', order=['chr2','nphr'])
tax[1].set_ylim(0,1)
tax[1].set_title('Model Performance')
tax[1].set_ylabel('Task Performance (%)')
tax[1].set_xlabel('Virus')
tplt.sca(ax[2])
tsns.barplot(data=ideal, x = 'virus', y = 'real_rewards', palette = ['dodgerblue', 'orange'],
tttt order=['chr2','nphr'])
tsns.swarmplot(data=ideal, x = 'virus', y = 'real_rewards', color='k', order=['chr2','nphr'])
tax[2].set_ylim(0,1)
tax[2].set_ylabel('Task Performance (%)')
tax[2].set_title('Mouse Performance')
tax[2].set_xlabel('Virus')
tsns.despine()
tplt.tight_layout()
tif save == True:
tt plt.savefig('performance_model_real.pdf', dpi =300)



def simulate_and_plot(modelled_data, model_parameters,
tttttmodel_type= 'w_stay'):
tmice = modelled_data['mouse_name'].unique()
tfor mouse in mice:
ttdata_pd = modelled_data.loc[modelled_data['mouse_name'] == mouse,
ttttt['real_rewards', 'signed_contrast', 'real_choice', 'laser',
ttttt 'laser_side']]

tt# -1 to 0 for laser
ttdata_pd.loc[data_pd['laser'] == -1, 'laser'] = 0
tt# Make data into the right format
ttdata_np = data_pd.to_numpy()
ttarray_of_tuples = map(tuple, data_np.T)
ttdata = tuple(array_of_tuples)
ttdata2 = tuple(tuple(map(int, tup)) for tup in data[2:])
ttdata0 = tuple(tuple(map(int, data[0])))
ttdata1  = data[1]
ttdata = [data0, data1, data2[0], data2[1], data2[2]]
ttparams = model_parameters.loc[(model_parameters['mouse'] == mouse)
tt& (model_parameters['model_name'] == model_type), 'x'].tolist()[0]

tt# Multiply data by 1000
ttdata_m = []
ttfor i  in range(len(data)):
tttdata_m.append(data[i]*1)

tt# Calculate Q values
ttif model_type == 'standard':
tt   sim_data = generate_data(data_m, all_contrasts, learning_rate=params[0],
ttttttttttt   beliefSTD=params[1], extraVal=params[2],
ttttttttttt   beta=params[3], stay=params[4])

ttif model_type == 'w_bias':
tttsim_data = generate_data_bias(data_m, all_contrasts, learning_rate=params[0],
ttttttttttt   beliefSTD=params[1], extraVal=params[2],
ttttttttttt   beta=params[3], stay=params[4])

ttif model_type == 'w_stay':
tttsim_data = generate_data_stay(data_m, all_contrasts, learning_rate=params[0],
ttttttttttt   beliefSTD=params[1], extraVal=params[2],
ttttttttttt   beta=params[3], stay=params[4])

ttif model_type == 'w_bias_n_stay':
tt   sim_data = generate_data_stay_and_bias(data_m, all_contrasts, learning_rate=params[0],
ttttttttttt   beliefSTD=params[1], extraVal=params[2],
ttttttttttt   beta=params[3], stay=params[4])
tt# Plots
ttsim_data = pd.DataFrame(sim_data)
ttsim_data = np.array(sim_data)
ttsim_data = pd.DataFrame(sim_data).T
ttsim_data['laser'] = data_m[3]
ttsim_data['laser_side'] = data_m[4]
ttsim_data['real_choice'] = data_m[2]
ttsim_data['mouse_name']  = mouse
ttsim_data['real_rewards']  = data_m[0]
ttsim_data = sim_data.rename(columns={0: "rewards", 1: "signed_contrast", 2: "simulated_choices", 3: "model_laser"})

ttmodel_psychometrics(sim_data, data_pd, mouse, save = True)

treturn modelled_data

def calculate_QL_QR(modelled_data, model_parameters,
tttttmodel_type= 'w_stay', zero = False, retrieve_ITIQ = False):
t# Also calculates pRight

tACC = []
tmice = modelled_data['mouse_name'].unique()
tfor mouse in mice:
ttdata_pd = modelled_data.loc[modelled_data['mouse_name'] == mouse,
ttttt['real_rewards', 'signed_contrast', 'real_choice', 'laser',
ttttt 'session_switches']]

tt# -1 to 0 for laser
ttdata_pd.loc[data_pd['laser'] == -1, 'laser'] = 0
tt# Make data into the right format
ttdata_np = data_pd.to_numpy()
ttarray_of_tuples = map(tuple, data_np.T)
ttdata = tuple(array_of_tuples)
ttdata2 = tuple(tuple(map(int, tup)) for tup in data[2:])
ttdata0 = tuple(tuple(map(int, data[0])))
ttdata1  = data[1]
ttdata = [data0, data1, data2[0], data2[1], data2[2]]
ttparams0 = model_parameters.loc[(model_parameters['mouse'] == mouse)
tt& (model_parameters['model_name'] == model_type), 'x'].copy()
ttparams1 = params0.tolist()[0].copy()

ttif zero ==  True:
tttparams1[2] = 0
tt# Calculate Q values
ttif model_type == 'w_stay':
ttt-, acc, Q_L, Q_R, Q_LL, Q_RL, Q_LR, Q_RR, pRight = session_neg_log_likelihood_stay(params1,
tttt*data, pregen_all_posteriors=True, accu=True, retrieve_Q=True,
ttttretrieve_ITIQ=retrieve_ITIQ)


tt# Return Q values to matrix
ttACC.append(acc)
ttif (zero == True) & (retrieve_ITIQ==False):
tttprint('zero no ITI')
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QL0'] = Q_L
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QR0'] = Q_R
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'pRight0'] = pRight
tttmodelled_data['QRQL0'] = modelled_data['QR0'].to_numpy() - \
ttttmodelled_data['QL0'].to_numpy()

tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QLL0'] = Q_LL
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QRL0'] = Q_RL
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QLR0'] = Q_LR
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QRR0'] = Q_RR

ttelif (zero == False) & (retrieve_ITIQ==False):
tttprint('standard')
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QL'] = Q_L
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QR'] = Q_R
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'pRight'] = pRight
ttt# Calculate QR-QL
tttmodelled_data['QRQL'] = modelled_data['QR'].to_numpy() - \
ttttmodelled_data['QL'].to_numpy()

tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QLL'] = Q_LL
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QRL'] = Q_RL
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QLR'] = Q_LR
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'QRR']  = Q_RR

ttelif (zero == True) & (retrieve_ITIQ==True):
tttprint('zero with ITI')
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQL0'] = Q_L
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQR0'] = Q_R
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIpRight0'] = pRight
tttmodelled_data['ITIQRQL0'] = \
ttttmodelled_data['ITIQR0'].to_numpy() - \
ttttmodelled_data['ITIQL0'].to_numpy()

tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQLL0'] = Q_LL
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQRL0'] = Q_RL
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQLR0'] = Q_LR
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQRR0'] = Q_RR

ttelif (zero == False) & (retrieve_ITIQ==True):
tttprint('standard ITI')
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQL'] = Q_L
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQR'] = Q_R
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIpRight'] = pRight
ttt# Calculate QR-QL
tttmodelled_data['ITIQRQL'] = \
ttttmodelled_data['ITIQR'].to_numpy() - \
ttttmodelled_data['ITIQL'].to_numpy()

tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQLL'] = Q_LL
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQRL'] = Q_RL
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQLR'] = Q_LR
tttmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt'ITIQRR'] = Q_RR

treturn modelled_data


def boxplot_model_parameters_per_mouse(model_parameters,
ttttttttt   model_type= 'w_stay',
ttttttttt   save = True):
t'''
tNotes: Plots learned parameters across virus
t'''
tfig, ax =  plt.subplots()
tsns.set(style='white')
tmodel = model_parameters.loc[model_parameters['model_name'] == model_type,
tttttttt ['x', 'virus']]
tparams = [r'$\alpha$', r'$\theta$', r'$\psi$',
ttt  r'$\tau$', r'$\gamma$', r'$\phi$']
tmod = pd.DataFrame(columns = ['params', 'virus'])
tfor i in range(len(model)):
tttemp_mod = pd.DataFrame(model['x'].iloc[i])
tttemp_mod['params'] = params[:len(temp_mod)]
tttemp_mod['virus'] = model.iloc[i,1]
ttmod = mod.append(temp_mod)
tsns.swarmplot(data = mod,  x = 'params', y = 0,  hue = 'virus',
tttt  palette = ['dodgerblue', 'orange'], split = False)
tax.axhline(0, ls='--', color = 'k')
tax.set_xlabel('Model Parameter')
tax.set_ylabel('Fitted Coef')
tsns.despine()
tif save == True:
tt plt.savefig('learned_parameters.pdf', dpi =300)

def model_psychometrics(sim_data, data_pd, mouse, save = True):
t'''
tNotes: Plots psychometrics
t'''

tsns.set(style='white', font_scale = 2, rc={"lines.linewidth": 2.5})
tfig, ax =  plt.subplots( figsize = (10,10))
tplt.sca(ax)
tsns.lineplot(data = data_pd, x = 'signed_contrast', y = 'real_choice',
ttttt hue = 'laser_side', palette = ['k','b','g'], ci =68, legend=None)
t# Plot model data with dash line
tsns.lineplot(data = sim_data, x = 'signed_contrast', y = 'simulated_choices',
ttttt hue = 'laser_side', palette = ['k','b','g'], ci =0, legend=None)
tax.lines[3].set_linestyle("--")
tax.lines[4].set_linestyle("--")
tax.lines[5].set_linestyle("--")
tax.set_ylabel('Fraction of Choices', fontsize =20)
tax.set_xlabel('Signed Contrast', fontsize =20)
tax.set_title(mouse, fontsize =20)
tlines = [Line2D([0], [0], color='black', linewidth=1.5, linestyle='-'),
tLine2D([0], [0], color='black', linewidth=1.5, linestyle='--')]
tlabels = ['Real Data', 'Model']
tplt.legend(lines, labels)
tsns.despine()
tif save == True:
tt plt.savefig('model_psychometric_'+mouse+'.png', dpi =300)

def model_choice_prob(psy, mouse, save = True):
t'''
tNotes: Plots psychometrics
t'''
tsns.set(style='white', font_scale = 2, rc={"lines.linewidth": 2.5})
tfig, ax =  plt.subplots( figsize = (10,10))
tplt.sca(ax)
tpsy_select = psy.loc[psy['mouse_name'] == mouse].copy()
tpsy_select['choice'] = psy_select['choice'] * -1
tpsy_select.loc[psy_select['choice'] == -1, 'choice'] = 0
tsns.lineplot(data = psy_select, x = 'signed_contrasts', y = 'choice',
ttttt hue = 'opto_block', hue_order = ['non_opto','L','R'],
ttttt palette = ['k','b','g'], ci =68, legend=None)
t# Plot model data with dash line
tsns.lineplot(data = psy_select, x = 'signed_contrasts', y = 1*(psy_select['pRight']>0.5),
ttttt hue = 'opto_block', hue_order = ['non_opto','L','R'],
ttttt palette = ['k','b','g'], ci =0, legend=None)
tax.lines[3].set_linestyle("--")
tax.lines[4].set_linestyle("--")
tax.lines[5].set_linestyle("--")
tax.set_ylabel('Fraction of Choices', fontsize =20)
tax.set_xlabel('Signed Contrast', fontsize =20)
tax.set_xlim(-0.3,0.3)
tax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
tax2.tick_params(axis='y', labelcolor='k')
tax2.set_ylabel('pRight', color='k', fontsize =20)  # we alread
tax.set_title(mouse, fontsize =20)
tlines = [Line2D([0], [0], color='black', linewidth=1.5, linestyle='-'),
tLine2D([0], [0], color='black', linewidth=1.5, linestyle='--')]
tlabels = ['Real Data', 'Model']
tax.axvline(0.3, ls='--', color = 'k')
tplt.legend(lines, labels)
tsns.despine()
tif save == True:
tt plt.savefig('model_choice_prob_'+mouse+'.png', dpi =300)




def plot_q_trial_whole_dataset(psy_df, save= True):
tpsy_select = psy_df.copy()
tsns.set(style = 'white')
t#Get index for trials of block start
tindex = psy_select.loc[psy['trial_within_block'] == 0, 'index']
tindex = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
tfor i in range(10):
ttfor idx in index-(i+1):
tttpsy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
tttt-(i+1)
tttpsy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
ttttpsy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]

tfig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
tpalette ={'R':'g','L':'b','non_opto':'k'}
tplt.sca(ax[0])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QRQL',
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(-10,50)
tplt.ylim(-1,1)
tplt.title('VTA-ChR2')
tax[0].set_xlabel('Trial in block')
tax[0].set_ylabel('QR-QL')

tplt.sca(ax[1])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QRQL',
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(-10,50)
tplt.ylim(-1,1)
tplt.title('VTA-NpHR')
tax[1].set_xlabel('Trial in block')
tplt.tight_layout()
tsns.despine()

tif save ==True:
ttplt.savefig('q_across_trials.svg', dpi =300)
ttplt.savefig('q_across_trials.jpeg',  dpi =300)


def plot_choice_trial_whole_dataset(psy_df, save= True):
tpsy_select = psy_df.copy()
tpsy_select['choice'] = psy_select['choice'] * - 1
tpsy_select.loc[psy_select['choice'] == -1, 'choice']  = 0
tsns.set(style = 'white')
t#Get index for trials of block start
tindex = psy_select.loc[psy['trial_within_block'] == 0, 'index']
tindex = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
tfor i in range(10):
ttfor idx in index-(i+1):
tttpsy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
tttt-(i+1)
tttpsy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
ttttpsy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]

tfig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
tpalette ={'R':'g','L':'b','non_opto':'k'}
tplt.sca(ax[0])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'choice',
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(-10,50)
tplt.ylim(0,1)
tplt.title('VTA-ChR2')
tax[0].set_xlabel('Trial in block')
tax[0].set_ylabel('Choice')

tplt.sca(ax[1])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'choice',
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(-10,50)
tplt.ylim(0,1)
tplt.title('VTA-NpHR')
tax[1].set_xlabel('Trial in block')
tplt.tight_layout()
tsns.despine()

tif save ==True:
ttplt.savefig('choice_across_trials.svg', dpi =300)
ttplt.savefig('choice_across_trials.jpeg',  dpi =300)





def plot_qmotivation_trial_whole_dataset(psy_df, save= True):
tpsy_select = psy_df.copy()
tsns.set(style = 'white')
tfig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
tpalette ={'R':'g','L':'b','non_opto':'k'}
tplt.sca(ax[0])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y = psy_chr2['QR'] +  psy_chr2['QL'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-ChR2')
tax[0].set_xlabel('Trial in block')
tax[0].set_ylabel('QR+QL')

tplt.sca(ax[1])
tpsy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
tsns.lineplot(data = psy_chr2, x = 'trial_within_block',
tttt y =  psy_chr2['QR'] +  psy_chr2['QL'],
ttttt hue = 'opto_block', palette = palette, ci=68)
tplt.xlim(0,50)
tplt.title('VTA-NpHR')
tax[1].set_xlabel('Trial in block')
tplt.tight_layout()
tsns.despine()

tif save ==True:
ttplt.savefig('qmotiv_across_trials.svg', dpi =300)
ttplt.savefig('qmotiv_across_trials.jpeg',  dpi =300)



def plot_q_trial_whole_dataset_per_mouse(psy_df, save=True):
tpsy_select = psy_df.copy()
tsns.set(style = 'white', font_scale=3)
tnum_mice = len(psy_select['mouse_name'].unique())
tindex = psy_select.loc[psy['trial_within_block'] == 0, 'index']
tindex = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
tfor i in range(10):
ttfor idx in index-(i+1):
tttpsy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
tttt-(i+1)
tttpsy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
ttttpsy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]
tfig, ax = plt.subplots(1,num_mice, figsize = [60,20], sharey =True)
tfor i, mouse in enumerate(psy_select['mouse_name'].unique()):
ttpalette ={'R':'g','L':'b','non_opto':'k'}
ttplt.sca(ax[i])
ttpsy_mouse = psy_select.loc[psy_select['mouse_name']==mouse]
ttsns.lineplot(data = psy_mouse, x = 'trial_within_block', y = 'QRQL',
tttttt hue = 'opto_block', palette = palette, ci=68)
ttplt.xlim(-10,50)
ttplt.ylim(-2,2)
ttplt.title('VTA-'+str(psy_select.loc[psy_select['mouse_name']==mouse,
ttttttttttt'virus'].unique()) + '-' +
tttttttttttttttt  str(mouse))
ttax[i].set_xlabel('Trial in block')
ttax[i].set_ylabel('QR-QL')
tsns.despine()
tif save ==True:
ttplt.savefig('q_across_trials_p_mouse.svg', dpi =300)
ttplt.savefig('q_across_trials_p_mouse.jpeg',  dpi =300)

def plot_choice_prob_opto_block(psy_df, ses_number, mouse_name, save =False):
t'''
tplot p choice right over trials
tParameters
t----------
tpsy_df : dataframe with real data
tses_number :number of  session (int)
tmouse_name : mouse name
tsave : whether to save the figure
tReturns
t-------
tFigure with p choice over time, excludes firs 100 trials

t'''
t#
t# neutral block

tsns.set(style='white', font_scale=4)
tpsy_df['right_block'] = np.nan
tpsy_df['left_block'] = np.nan
tpsy_df['right_block'] = (psy_df['opto_block'] == 'R')*1
tpsy_df['left_block'] = (psy_df['opto_block'] == 'L')*1
tpsy_df['non_opto_block'] = (psy_df['opto_block'] == 'non_opto')*1

tfig, ax1 = plt.subplots( figsize= (30,10))
tplt.sca(ax1)
tpsy_subset =  psy_df.loc[psy_df['mouse_name'] == mouse_name].copy()
tpsy_subset = psy_subset.loc[psy_subset['ses'] == psy_subset['ses'].unique()[ses_number]].reset_index()
tpsy_subset['choice'] = psy_subset['choice']*-1
tpsy_subset.loc[psy_subset['choice']==-1,'choice'] = 0
tp_choice = ((psy_subset['choice'].rolling(5).mean() +
ttttpsy_subset['choice'].rolling(5).mean().shift(-5))/2)

tsns.lineplot(data = psy_subset, x = psy_subset.index,
ttttt  y = p_choice,
ttttt  color = 'k')

tplt.fill_between((np.arange(psy_subset['choice'].count())+1),
ttttt psy_subset['left_block'], color = 'blue', alpha =0.35)

tplt.fill_between((np.arange(psy_subset['choice'].count())+1),
ttttt psy_subset['right_block'], color = 'green', alpha =0.35)
tplt.fill_between((np.arange(psy_subset['choice'].count())+1),
ttttt psy_subset['non_opto_block'],
ttttt color ='black', alpha =0.35)
tsns.scatterplot(data = psy_subset, x = psy_subset.index,
ttttt  y = 'choice',
ttttt  color = 'k', s=100)
t# Probability of rightward choice

tplt.xlim(25,psy_subset['choice'].count())
tax1.set_ylim(-0.1,1.1)


tax1.tick_params(axis='y', labelcolor='k')
tax1.set_ylabel('% Right Choices', color='k')
tax1.set_xlabel('Trial', color='black')
tax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
tsns.lineplot(data = psy_subset, x = psy_subset.index,
ttttt  y = ((psy_subset['QRQL'].rolling(5).mean() +
tttttt   psy_subset['QRQL'].rolling(5).mean().shift(-5))/2),
ttttt  color = 'red', ax = ax2)

tax2.tick_params(axis='y', labelcolor='red')
tax2.set_ylabel('QR - QL', color='red')  # we already handled the x-label wi
tax2.set_ylim(-1,1)
tplt.tight_layout()

tplt.savefig('choice_and_q.svg')
tplt.savefig('choice_and_q.jpeg')


def plot_choice_40_trials(psy_df, ses_number, mouse_name, save =False):
t'''
tplot p choice right over trials
tParameters
t----------
tpsy_df : dataframe with real data
tses_number :number of  session (int)
tmouse_name : mouse name
tsave : whether to save the figure
tReturns
t-------
tFigure with p choice over time, excludes firs 100 trials

t'''
t#
t# neutral block

tsns.set(style='white', font_scale=4)
tpsy_df['right_block'] = np.nan
tpsy_df['left_block'] = np.nan
tpsy_df['right_block'] = (psy_df['opto_block'] == 'R')*1
tpsy_df['left_block'] = (psy_df['opto_block'] == 'L')*1
tpsy_df['non_opto_block'] = (psy_df['opto_block'] == 'non_opto')*1

tfig, ax1 = plt.subplots( figsize= (30,10))
tplt.sca(ax1)
tpsy_subset =  psy_df.loc[psy_df['mouse_name'] == mouse_name].copy()
tpsy_subset = psy_subset.loc[psy_subset['ses'] == psy_subset['ses'].unique()[ses_number]].reset_index()
tpsy_subset['choice'] = psy_subset['choice']*-1
tpsy_subset.loc[psy_subset['choice']==-1,'choice'] = 0

tplt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['left_block'], color = 'blue', alpha =0.35)

tplt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['right_block'], color = 'green', alpha =0.35)
tplt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['non_opto_block'],
ttttt color ='black', alpha =0.35)
tsns.scatterplot(data = psy_subset, x = psy_subset.index,
ttttt  y = 1,
ttttt  color = 'k',
ttttt  hue = (psy_subset['signed_contrasts'].abs() * (1*(psy_subset['signed_contrasts']>=0))),
ttttt  palette = 'Greys',
ttttt  s = 1000,
ttttt  legend= None, edgecolor='black')
tsns.scatterplot(data = psy_subset, x = psy_subset.index,
ttttt  y = 0,
ttttt  color = 'k',
ttttt  hue = (psy_subset['signed_contrasts'].abs() * (1*(psy_subset['signed_contrasts']<=0))),
ttttttt palette = 'Greys',
ttttt  s = 1000,
ttttt  legend= None, edgecolor='black')
t# Add feedback
tsns.scatterplot(data = psy_subset, x = psy_subset.index,
ttttt  y = psy_subset['choice'],
ttttt  color = 'k',
ttttt  hue = psy_subset['feedbackType'], palette = ['r', 'g'],
ttttt  s = 150,
ttttt  legend= None, edgecolor='black')

t# Probability of rightward choice

tplt.xlim(200,140)
tplt.ylim(-0.1,1.1)


tax1.tick_params(axis='y', labelcolor='k')
tax1.set_ylabel('Choice', color='k')
tax1.set_yticks([0,1])
tax1.set_yticklabels(['Left', 'Right'])
tax1.set_xlabel('Trial', color='black')
tax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
tsns.lineplot(data = psy_subset, x = psy_subset.index,
ttttt  y = psy_subset['QRQL'],
ttttt  color = 'red')
tax2.tick_params(axis='y', labelcolor='red')
tax2.axhline(0, ls='--', color = 'k', lw='4')
tax2.set_ylabel('QR - QL', color='red')  # we already handled the x-label wi
tplt.tight_layout()

tplt.savefig('choice_and_40.svg')
tplt.savefig('choice_and_40.jpeg')

def true_stim_posterior(true_contrast, beliefSTD):

tdef st_sp_0(percieve_contrast, beliefSTD=beliefSTD):
ttall_contrasts = np.array([-0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25])
tta = 0
ttb = 0
ttfor i in all_contrasts[all_contrasts>0]:
ttta += norm.pdf((percieve_contrast - i)/ beliefSTD)
ttfor i in all_contrasts:
tttb += norm.pdf((percieve_contrast - i)/ beliefSTD)

ttreturn a/b * norm.pdf(percieve_contrast,true_contrast,beliefSTD)

tbs_right = quad(st_sp_0,-1, 1)

treturn [1-bs_right[0],bs_right[0]]


# Given all of the Q values (a matrix of size num_contrasts x 2), compute the overall Q_left and Q_right
# (i.e., the overall value of choosing left or right) given the perceived stimulus
def compute_QL_QR(Q, trial_contrast, contrast_posterior):
	Q_L = 0
	Q_R = 0

	# We compute Q_L, Q_R as in the Cell paper, by taking the weighted average of the various perceived stimuli,
	# according to the probability that they were perceived
	for i in range(len(contrast_posterior)):
		Q_L += contrast_posterior[i] * Q[i, 0]
		Q_R += contrast_posterior[i] * Q[i, 1]

	return Q_L, Q_R

def softmax_stay(Q_L, Q_R, beta, l_stay, r_stay, stay):
	p = [np.exp(Q_L / beta + stay*l_stay),
t  np.exp(Q_R / beta + stay*r_stay)]
	p /= np.sum(p)

	return p

def trial_log_likelihood_stay(params, trial_data, Q, all_contrasts, all_posteriors,
ttttttt  previous_trial, trial_num, retrieve_Q = False, retrieve_ITIQ = False):
	# Get relevant parameters
	trial_contrast, trial_choice, reward, laser = trial_data
	learning_rate, beliefSTD, extraVal, beta, stay = params
	Q =  Q.copy()

	# Compute the log-likelihood of the actual mouse choice
	if all_posteriors is None:
		contrast_posterior = true_stim_posterior(trial_contrast, beliefSTD)
	else:
		posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
		contrast_posterior = all_posteriors[posterior_idx, :]

	Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)
t# Weighted Q values for encoding/decoding
tposterior = np.array([[contrast_posterior[0]],[contrast_posterior[1]]])
tQ_w = Q * posterior
tQ_LL = Q[0,0]
tQ_RL = Q[1,0]
tQ_LR = Q[0,1]
tQ_RR = Q[1,1]

	if trial_num == 0:
		(l_stay, r_stay) = [0,0]
	else:
		previous_choice= [0,0]
		previous_choice[previous_trial] = 1
		(l_stay, r_stay) = previous_choice

	choice_dist = softmax_stay(Q_L, Q_R, beta,l_stay, r_stay, stay)
	LL = np.log(choice_dist[trial_choice])

	# Learning
	if trial_choice == 0:
		Q_chosen = Q_L
	else:
		Q_chosen = Q_R

	# Laser-modulation
	if laser == 1:
		received_reward = reward + extraVal
	else:
		received_reward = reward

	# Update Q-values according to the aggregate reward + laser value
	for i in range(2):
		Q[i, trial_choice] += contrast_posterior[i] * learning_rate * (received_reward - Q[i, trial_choice])

	if retrieve_ITIQ == True:
ttQ_LL = Q[0,0]
ttQ_RL = Q[1,0]
ttQ_LR = Q[0,1]
ttQ_RR = Q[1,1]

	if retrieve_Q==True:
		return LL, Q, Q_L, Q_R, Q_LL, Q_RL, Q_LR, Q_RR, choice_dist[1] #  choice_dist[1] = pChoice_right

	else:
		return LL, Q



def session_neg_log_likelihood_stay(params, *data, pregen_all_posteriors=True,
tttttttttaccu=False, retrieve_Q =  False, retrieve_ITIQ = False):
	# Unpack the arguments
	learning_rate, beliefSTD, extraVal, beta, stay = params
	rewards, true_contrasts, choices, lasers, ses_switch = data
	num_trials = len(rewards)

	if retrieve_Q==True:
		Q_L = []
		Q_R = []
ttQ_LL = []
ttQ_RL = []
ttQ_LR = []
ttQ_RR = []
		pRight = []

	# Generate the possible contrast list
	all_contrasts = np.array([-0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25])
	num_contrasts = len(all_contrasts)

	# If True, generate all posterior distributions ahead of time to save time
	if pregen_all_posteriors:
		all_posteriors = np.zeros((num_contrasts, 2))
		for idx, contrast in enumerate(all_contrasts):
			all_posteriors[idx, :] = true_stim_posterior(contrast, beliefSTD)
	else:
		all_posteriors = None

	# Compute the log-likelihood
	if accu == True:
		acc = 0
	LL = 0
	Q = np.zeros([2,2])

	if retrieve_Q == True:
		for i in range(num_trials):
			if i == 0:

				trial_LL, newQ, Q_Lt, Q_Rt,Q_LLt, Q_RLt, Q_LRt, Q_RRt, pright = trial_log_likelihood_stay(params,
ttttttttttt[true_contrasts[i], choices[i], rewards[i], lasers[i]],
tttttttttttQ, all_contrasts, all_posteriors,
tttttttttttnp.nan, i, retrieve_Q=retrieve_Q, retrieve_ITIQ=retrieve_ITIQ)
			else:
				if ses_switch[i] == 1:
					Q = np.zeros([2,2])

				trial_LL, newQ, Q_Lt, Q_Rt,Q_LLt, Q_RLt, Q_LRt, Q_RRt, pright = trial_log_likelihood_stay(params,
ttttttttttt[true_contrasts[i], choices[i], rewards[i], lasers[i]],
tttttttttttQ, all_contrasts, all_posteriors,
tttttttttttchoices[i-1], i, retrieve_Q=retrieve_Q, retrieve_ITIQ=retrieve_ITIQ)
			if (i != 0) & (np.sum(Q, axis=0)[0] != np.sum(newQ, axis=0)[0]) & (np.sum(Q, axis=0)[1] != np.sum(newQ, axis=0)[1]):
				print('Warning, double update error in trial %d'%i)
			LL += trial_LL
			Q = newQ

			if accu == True:
				acc += (np.exp(trial_LL)>0.5)*1

			Q_L.append(Q_Lt)
			Q_R.append(Q_Rt)
tttQ_LL.append(Q_LLt)
tttQ_RL.append(Q_RLt)
tttQ_LR.append(Q_LRt)
tttQ_RR.append(Q_RRt)
			pRight.append(pright)

	else:
		for i in range(num_trials):
			if i == 0:
				trial_LL, newQ = trial_log_likelihood_stay(params,
ttttttttttt[true_contrasts[i], choices[i], rewards[i], lasers[i]],
tttttttttttQ, all_contrasts, all_posteriors, np.nan, i)
			else:
				if ses_switch[i] == 1:
					Q = np.zeros([2,2])

				trial_LL, newQ = trial_log_likelihood_stay(params,
ttttttttttt[true_contrasts[i], choices[i], rewards[i], lasers[i]],
tttttttttttQ, all_contrasts, all_posteriors, choices[i-1], i)
			LL += trial_LL
			Q = newQ

			if accu == True:
				acc += (np.exp(trial_LL)>0.5)*1


	if retrieve_Q == True:
		if accu == True:
			acc = acc/num_trials
			return -LL,  acc, Q_L, Q_R, Q_LL, Q_RL, Q_LR, Q_RR, pRight
		else:
			return -LL, Q_L, Q_R, Q_LL, Q_RL, Q_LR, Q_RR, pRight

	else:
		if accu == True:
			acc = acc/num_trials
			return -LL,  acc
		else:
			return -LL





# Optimize several times with different initializations and return the best fit parameters, and negative log likelihood

def optimizer_stay(data, num_fits = 10, initial_guess=[0.1, 1, 0, 1, 1]):
	# Accounting variables
	best_NLL = np.Inf
	best_x = [None, None, None, None, None]
	buffer_NLL = []
	buffer_x = np.empty([num_fits,len(initial_guess)])
	# Do our fit with several different initializations
	for i in range(num_fits):
		print('Starting fit %d' % i)

		# For every fit other than the first, construct a new initial guess
		if i != 0:
			lr_guess = np.random.uniform(0, 2)
			beliefSTD_guess = np.random.uniform(0.03, 1)
			extraVal_guess = np.random.uniform(-2,2)
			beta_guess = np.random.uniform(0.01, 1)
			stay = np.random.uniform(-1, 1)
			initial_guess = [lr_guess, beliefSTD_guess, extraVal_guess, beta_guess, stay]

		# Run the fit
		res = so.minimize(session_neg_log_likelihood_stay, initial_guess, args=data,
tttttmethod='L-BFGS-B', bounds=[(0, 2), (0.03, 1), (-2, 2), (0.01, 1),
ttttttttt(-1,1)])

		# If this fit is better than the previous best, remember it, otherwise toss
		buffer_x[i,:] = res.x
		buffer_NLL.append(res.fun)

		if res.fun <= best_NLL:
			best_NLL = res.fun
			best_x = res.x

	return best_x, best_NLL, buffer_NLL, buffer_x



# hardcode this for speed
def normal_pdf(x, loc, scale):
	factor = 1 / (np.sqrt(2 * np.pi) * scale)
	power = -0.5 * (((x - loc) / scale) ** 2)

	return factor * np.exp(power)

##### Analysis functions

def accuracy_per_contrast(data, all_contrasts):
	rewards, contrasts, choices, lasers = data
	num_trials = len(rewards)

	acc = np.zeros(len(all_contrasts))
	trials_per_contrast = np.zeros(len(all_contrasts))
	for i in range(num_trials):
		reward = rewards[i]
		contrast = contrasts[i]
		choice = choices[i]
		laser = lasers[i]

		contrast_idx = np.argmin(np.abs(all_contrasts - contrast))

		if reward > 0:
			acc[contrast_idx] += 1

		trials_per_contrast[contrast_idx] += 1

	acc /= trials_per_contrast

	return acc

def psychometric_curve(data, all_contrasts):
	rewards, contrasts, choices, lasers = data
	num_trials = len(rewards)

	p_right = np.zeros(len(all_contrasts))
	trials_per_contrast = np.zeros(len(all_contrasts))
	for i in range(num_trials):
		reward = rewards[i]
		contrast = contrasts[i]
		choice = choices[i]
		laser = lasers[i]

		contrast_idx = np.argmin(np.abs(all_contrasts - contrast))

		if choice == 1:
			p_right[contrast_idx] += 1

		trials_per_contrast[contrast_idx] += 1

	p_right /= trials_per_contrast

	return p_right

##### TESTING SCRIPT #####

def simulation_contrast_distribution(mean_contrast, beliefSTD, all_contrasts):
	# Compute distribution of final perceived contrasts
	p = normal_pdf(all_contrasts, loc=mean_contrast, scale=beliefSTD)

	# Renormalize
	p /= np.sum(p)

	return p


def generate_data_stay(data, all_contrasts, learning_rate=0.3,
ttttt   beliefSTD=0.1, extraVal=1, beta=0.2,
ttttt   stay = 1, is_verbose=False, propagate_errors = True):

	rewards = []
	true_contrasts = []
	choices = []
	lasers = []

	if propagate_errors == False:
		prop = 3
	else:
		prop = 4

	# Simulate the POMDP model
	Q = np.zeros([2, 2])
	for t in range(len(data[0])):
		if is_verbose:
			print(t)

		# Pick a true stimulus and store
		trial_contrast = data[1][t]
		true_contrasts.append(trial_contrast)
tt# Add noise

		contrast_posterior = true_stim_posterior(trial_contrast, beliefSTD)

		Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)

		if t == 0:
		t(l_stay, r_stay) = [0,0]
		else:
		tprevious_choice= [0,0]
		tprevious_choice[choices[t-1]] = 1
		t(l_stay, r_stay) = previous_choice
		choice_dist = softmax_stay(Q_L, Q_R, beta,l_stay, r_stay, stay)
		choice = np.random.choice(2, p = [float(choice_dist[0]), float(choice_dist[1])])
		choices.append(choice)

		# Get reward and store it
		if np.sign(trial_contrast) == -1 and choice == 0:
			reward = 1
		elif np.sign(trial_contrast) == 1 and choice == 1:
			reward = 1
		elif np.sign(trial_contrast) == 0:
			reward = random.choice([0,1])
		else:
			reward = 0

		rewards.append(reward)

		# Add laser value on the correct condition
		if propagate_errors == True:
			if choice == data[prop][t]:
			treward += extraVal
			tlasers.append(1)
			else:
			tlasers.append(-1)
		else:
			reward = data[0][t]
			reward += extraVal*data[prop][t]
			lasers.append(data[prop][t])
		# Learn (update Q-values)
		if choice == 0:
			Q_chosen = Q_L
		else:
			Q_chosen = Q_R


		for i in range(2):
			Q[i, choice] += contrast_posterior[i] * learning_rate * (reward - Q_chosen)

	return rewards, true_contrasts, choices, lasers



def transform_model_struct_2_POMDP(model_data, simulate_data):
ttsimulate_data.loc[simulate_data['extraRewardTrials'] == 'right', 'extraRewardTrials' ] = 1
ttsimulate_data.loc[simulate_data['extraRewardTrials'] == 'left', 'extraRewardTrials' ] = 0
ttsimulate_data.loc[simulate_data['extraRewardTrials'] == 'none', 'extraRewardTrials' ] = -1
ttobj = model_data
ttobj['choice'] = obj['choice'] * -1
ttobj.loc[obj['choice'] == -1, 'choice'] = 0
ttobj['laser_side'] = simulate_data['extraRewardTrials']
ttreturn obj

def likelihood_ratio(llmin, llmax):
treturn(2*(llmax-llmin))

def aic(LL,n_param):
t# Calculates Akaike Information Criterion
taic =  2*n_param - 2*LL
treturn aic

def chi2_LLR(L1,L2):
tLR = likelihood_ratio(L1,L2)
tp = chi2.sf(LR, 1) # L2 has 1 DoF more than L1
treturn p

# Main function, runs all the testing scripts

# Test the fitting procedure by fitting on arbitrarily generated data
	# num_trials = 30000
all_contrasts = np.array([-0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25])

	# data = generate_data(num_trials, all_contrasts)
	# x, NLL = optimizer(data, initial_guess=[0.3, 0.1, 1, 0.2])

	# print(x)
mice = np.array(['dop_8', 'dop_9', 'dop_11', 'dop_4'])

# Load Alex's actual data
psy = pd.read_pickle('all_behav.pkl')

# Select only ephys sessions
#psy = psy.loc[((psy['ses']>'2020-01-13') & (psy['mouse_name'] == 'dop_4')) |
#ttt  ((psy['ses']>'2020-03-13') & (psy['ses']<'2020-03-19')
#ttt  ((psy['ses']>'2020-03-13') & (psy['mouse_name'] != 'dop_9')) |
#ttt   & (psy['mouse_name'] == 'dop_9'))]

train_set_size = 1
cross_validate = False

all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.t,  0.0625, 0.125 , 0.25  ])
best_nphr_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),4])
best_nphr_NLL_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])
model_parameters = pd.DataFrame()
modelled_data = pd.DataFrame()
for i, mouse in enumerate(mice):
ttmodel_data_nphr, simulate_data_nphr  = \
tttpsy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse],
ttttttttttt  virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0])
tt# Obtain sessions switches
ttsession_switches = np.zeros(len(model_data_nphr))
ttfor session in model_data_nphr['ses'].unique():
ttt session_switches[model_data_nphr.ses.ge(session).idxmax()]=1

ttobj = transform_model_struct_2_POMDP(model_data_nphr, simulate_data_nphr)


ttvirus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0]

ttopto = obj['extraRewardTrials'].to_numpy()
ttlasers = []
ttfor i in range(len(opto)):
ttttry:
ttttlasers.append(int(opto[i][0]))
tttexcept:
ttttlasers.append(int(opto[i]))

ttchoices = list(obj['choice'].to_numpy())
ttcontrasts = list(obj['stimTrials'].to_numpy())
ttrewards = list(obj['reward'].to_numpy())
ttlaser_side = list(obj['laser_side'].to_numpy())


ttdata = (rewards[:int(len(rewards)*train_set_size)],
ttttcontrasts[:int(len(rewards)*train_set_size)],
ttttchoices[:int(len(rewards)*train_set_size)],
ttttlasers[:int(len(rewards)*train_set_size)],
ttttsession_switches[:int(len(rewards)*train_set_size)])
ttsimulate_data = (rewards[:int(len(rewards)*train_set_size)],
tttttt contrasts[:int(len(rewards)*train_set_size)],
tttttt choices[:int(len(rewards)*train_set_size)],
ttttt  lasers[:int(len(rewards)*train_set_size)],
ttttt  laser_side[:int(len(rewards)*train_set_size)],
ttttt  session_switches[:int(len(rewards)*train_set_size)])


ttdata_test = data
ttsimulate_data_test = simulate_data


tt(best_x_stay, train_NLL_stay, buffer_NLL_stay,
tt buffer_x_stay) = optimizer_stay(data, initial_guess=[0.3, 0.05, -1, 0.2,1])

ttcv_aic_stay = aic((session_neg_log_likelihood_stay(best_x_stay,
tttt  *data_test, pregen_all_posteriors=True))*-1,5)


ttcv_LL_stay = (session_neg_log_likelihood_stay(best_x_stay, *data_test,
ttttttttttttt  pregen_all_posteriors=True))*-1



tt_, cv_acc_stay = session_neg_log_likelihood_stay(best_x_stay, *data_test,
ttttt   pregen_all_posteriors=True, accu=True)


ttmodel_parameters_mouse = pd.DataFrame()
ttmodel_parameters_mouse['x'] = [best_x_stay]
ttmodel_parameters_mouse['LL'] = (cv_LL_stay/len(data_test[0]))
ttmodel_parameters_mouse['aic'] = cv_aic_stay
ttmodel_parameters_mouse['accu'] = cv_acc_stay
ttmodel_parameters_mouse['model_name'] = 'w_stay'


ttsim_data = generate_data_stay(simulate_data_test, all_contrasts, learning_rate=best_x_stay[0],
ttttttttt   beliefSTD=best_x_stay[1], extraVal=best_x_stay[2], beta=best_x_stay[3], stay=best_x_stay[4])
ttsim_data = pd.DataFrame(sim_data)

ttsim_data = sim_data.rename(columns={0: "rewards", 1: "signed_contrast", 2: "simulated_choices", 3: "model_laser"})
ttsim_data = np.array(sim_data)
ttsim_data = pd.DataFrame(sim_data).T
ttsim_data['laser'] = lasers[:int(len(rewards)*train_set_size)]
ttsim_data['laser_side'] = laser_side[:int(len(rewards)*train_set_size)]
ttsim_data['real_choice'] = choices[:int(len(rewards)*train_set_size)]
ttsim_data['session_switches'] = session_switches[:int(len(rewards)*train_set_size)]
ttsim_data['mouse_name']  = mouse
ttsim_data['virus']  = virus
ttsim_data['real_rewards']  = simulate_data[0]

tt# Concatenate with general dataframes
ttmodel_parameters_mouse['mouse'] = mouse
ttmodel_parameters_mouse['virus'] = virus

tt# Concatenate with general dataframes
ttmodel_parameters = pd.concat([model_parameters, model_parameters_mouse])
ttmodelled_data = pd.concat([modelled_data, sim_data])

# Analysis

modelled_data = modelled_data.rename(columns={0: "rewards",
   1: "signed_contrast", 2: "choices_standard", 3: "model_laser"})

modelled_data = calculate_QL_QR(modelled_data, model_parameters,
tttttmodel_type= 'w_stay')
modelled_data = calculate_QL_QR(modelled_data, model_parameters,
tttttmodel_type= 'w_stay', zero=True)
modelled_data = calculate_QL_QR(modelled_data, model_parameters,
tttttmodel_type= 'w_stay', zero=True, retrieve_ITIQ=  True)
modelled_data = calculate_QL_QR(modelled_data, model_parameters,
tttttmodel_type= 'w_stay', zero=False, retrieve_ITIQ=  True)

# Calculate a few things
psy['QL'] = np.nan
psy['QR'] = np.nan
psy['QLL'] = np.nan
psy['QRL'] = np.nan
psy['QLR'] = np.nan
psy['QRR'] = np.nan
psy['QLL0'] = np.nan
psy['QRL0'] = np.nan
psy['QLR0'] = np.nan
psy['QRR0'] = np.nan
psy['ITIQLL'] = np.nan
psy['ITIQRL'] = np.nan
psy['ITIQLR'] = np.nan
psy['ITIQRR'] = np.nan
psy['ITIQLL0'] = np.nan
psy['ITIQRL0'] = np.nan
psy['ITIQLR0'] = np.nan
psy['ITIQRR0'] = np.nan
psy['QL0'] = np.nan
psy['QR0'] = np.nan
psy['QRQL'] = np.nan
psy['QRQL0'] = np.nan
psy['pRight'] = np.nan
psy['pRight0'] = np.nan
psy['ITIQL'] = np.nan
psy['ITIQR'] = np.nan
psy['ITIQL0'] = np.nan
psy['ITIQR0'] = np.nan
psy['ITIQRQL'] = np.nan
psy['ITIpRight'] = np.nan
psy['ITIpRight0'] = np.nan
psy['ITIQRQL'] = np.nan
psy['ITIQRQL0'] = np.nan

for i, mouse in enumerate(mice):
tpsy.loc[psy['mouse_name'] == mouse, ['QL', 'QR', 'QRQL', 'QLL', 'QRL', 'QLR', 'QRR','pRight',
tttttttttt 'QL0', 'QR0', 'QLL0', 'QRL0', 'QLR0', 'QRR0', 'pRight0', 'QRQL0',
tttttttttt 'ITIQL', 'ITIQR', 'ITIQLL', 'ITIQRL', 'ITIQLR', 'ITIQRR'
tttttttttt 'ITIQL0', 'ITIQR0', 'ITIQRQL', 'ITIpRight',
tttttttttt 'ITIQLL0', 'ITIQRL0', 'ITIQLR0', 'ITIQRR0'
tttttttttt 'ITIpRight0', 'ITIQRQL', 'ITIQRQL0']] =\
tmodelled_data.loc[modelled_data['mouse_name'] == mouse,
tttttt  ['QL', 'QR', 'QRQL', 'QLL', 'QRL', 'QLR', 'QRR','pRight',
tttttttttt 'QL0', 'QR0', 'QLL0', 'QRL0', 'QLR0', 'QRR0', 'pRight0', 'QRQL0',
tttttttttt 'ITIQL', 'ITIQR', 'ITIQLL', 'ITIQRL', 'ITIQLR', 'ITIQRR'
tttttttttt 'ITIQL0', 'ITIQR0', 'ITIQRQL', 'ITIpRight',
tttttttttt 'ITIQLL0', 'ITIQRL0', 'ITIQLR0', 'ITIQRR0'
tttttttttt 'ITIpRight0', 'ITIQRQL', 'ITIQRQL0']].to_numpy()

psy['argmax_choice'] = (psy['pRight']>0.5)*1


for mouse in mice:
tmodel_choice_prob(psy, mouse, save = False)
tmodel_choice_raw_prob(psy, mouse, save = False)


boxplot_model_parameters_per_mouse(model_parameters,
ttttttttt   model_type= 'w_stay',
ttttttttt   save = True)
plot_q_trial_whole_dataset(psy)
plot_q_trial_whole_dataset_per_mouse(psy)
model_performance(model_parameters, modelled_data, model_type=
tttt  'w_stay', save = True)

plot_choice_prob_opto_block(psy, 1, 'dop_7', save =True)
plot_choice_prob_opto_block(psy, 4, 'dop_8', save =True)
plot_choice_prob_opto_block(psy, 15, 'dop_9', save =True)
plot_choice_prob_opto_block(psy, 10, 'dop_11', save =True)
plot_choice_prob_opto_block(psy, 19, 'dop_4', save =True)

plot_qmotivation_trial_whole_dataset(psy, save= True)

sns.lineplot(data = sim_data, x =1 , y=  2, hue = 'laser_side', ci = 0)
sns.lineplot(data = sim_data, x =1 , y= 'real_choice', hue = 'laser_side')



plot_choice_40_trials(psy, 1, 'dop_7', save =True)
plot_choice_40_trials(psy, 4, 'dop_8', save =True)
plot_choice_40_trials(psy, 15, 'dop_9', save =True)
plot_choice_40_trials(psy, 10, 'dop_11', save =True)
plot_choice_40_trials(psy, 19, 'dop_4', save =True)

plot_choice_trial_from_model(psy, save= True)

plot_qr_trial_whole_dataset(psy, save= True)
plot_ql_trial_whole_dataset(psy, save= True)


save_Q(psy, root = '/Volumes/witten/Alex/recordings_march_2020_dop')



all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.t,  0.0625, 0.125 , 0.25  ])
