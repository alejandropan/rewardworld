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
    '''
    Params:
    root: root folder for ephys sessions
    psy: dataframe with all the data
    Description:
    Save QL QR QL0 and QR0 files, QL0 and QR0 files are Q values in the absence
    of laser
    '''
    viruses = psy['virus'].unique()
    for virus in viruses:
        byvirus = psy.loc[psy['virus'] == virus]
        mice = byvirus['mouse_name'].unique()
        for mouse in mice:
            bymouse = byvirus.loc[byvirus['mouse_name'] == mouse]
            mouse_path = os.path.join(root, virus, mouse )
            # First check if mouse has ephys data
            if os.path.exists(mouse_path):
                dates = glob.glob(path.join(mouse_path,  '*'))
                # Get all ephys dates
                for d in dates:
                    sessions = glob.glob(path.join(d, '*'))
                    if len(sessions)>1:
                        print('No current support for 2 sessions in a day'
                              + mouse + d)
                    for ses in sessions:
                        alf  =  os.path.join(ses, 'alf')
                        date = d[-10:]
                        Q_files = ['QL', 'QR', 'QL0', 'QR0', 'ITIQL', 'ITIQR',
                                   'ITIQL0', 'ITIQR0']
                        for Q in Q_files:
                            qfile = os.path.join(alf,Q+'stay_ephysfit.npy')
                            file = bymouse.loc[bymouse['ses'] == date,
                                                     Q].to_numpy()
                            if len(np.load(os.path.join(alf,'_ibl_trials.choice.npy'))) != len(file):
                                print('uniqual number of Q and t'
                                      + mouse + ses)
                            else:
                                np.save(qfile, file)

def model_choice_raw_prob(psy, mouse, save = True):
    '''
    Notes: Plots psychometrics
    '''
    sns.set(style='white', font_scale = 2, rc={"lines.linewidth": 2.5})
    fig, ax =  plt.subplots( figsize = (10,10))
    plt.sca(ax)
    psy_select = psy.loc[psy['mouse_name'] == mouse].copy()
    psy_select['choice'] = psy_select['choice'] * -1
    psy_select.loc[psy_select['choice'] == -1, 'choice'] = 0
    sns.lineplot(data = psy_select, x = 'signed_contrasts', y = 'choice',
                     hue = 'opto_block', hue_order = ['non_opto','L','R'],
                     palette = ['k','b','g'], ci =68, legend=None)
    # Plot model data with dash line
    sns.lineplot(data = psy_select, x = 'signed_contrasts', y = psy_select['pRight'],
                     hue = 'opto_block', hue_order = ['non_opto','L','R'],
                     palette = ['k','b','g'], ci =0, legend=None)
    ax.lines[3].set_linestyle("--")
    ax.lines[4].set_linestyle("--")
    ax.lines[5].set_linestyle("--")
    ax.set_ylabel('Fraction of Choices', fontsize =20)
    ax.set_xlabel('Signed Contrast', fontsize =20)
    ax.set_xlim(-0.3,0.3)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylabel('pRight', color='k', fontsize =20)  # we alread
    ax.set_title(mouse, fontsize =20)
    lines = [Line2D([0], [0], color='black', linewidth=1.5, linestyle='-'),
    Line2D([0], [0], color='black', linewidth=1.5, linestyle='--')]
    labels = ['Real Data', 'Model']
    ax.axvline(0.3, ls='--', color = 'k')
    plt.legend(lines, labels)
    sns.despine()
    if save == True:
         plt.savefig('model_choice_prob_raw'+mouse+'.png', dpi =300)



def plot_qr_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y = psy_chr2['QR'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('QR')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y =  psy_chr2['QR'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('qr_across_trials.svg', dpi =300)
        plt.savefig('qr_across_trials.jpeg',  dpi =300)


def plot_ql_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y = psy_chr2['QR'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('QL')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y =  psy_chr2['QL'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('ql_across_trials.svg', dpi =300)
        plt.savefig('ql_across_trials.jpeg',  dpi =300)


def plot_choice_trial_from_model(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    #Get index for trials of block start
    index = psy_select.loc[psy['trial_within_block'] == 0, 'index']
    index = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
    for i in range(10):
        for idx in index-(i+1):
            psy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
                -(i+1)
            psy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
                psy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]

    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'pRight',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(0,1)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('pRight')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'pRight',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(0,1)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('plot_choice_trial_from_model.svg', dpi =300)
        plt.savefig('plot_choice_trial_from_model.jpeg',  dpi =300)


def plot_qr_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y = psy_chr2['QR'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('QR')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y =  psy_chr2['QR'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('qr_across_trials.svg', dpi =300)
        plt.savefig('qr_across_trials.jpeg',  dpi =300)


def plot_ql_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y = psy_chr2['QL'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('QL')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y =  psy_chr2['QL'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('ql_across_trials.svg', dpi =300)
        plt.savefig('ql_across_trials.jpeg',  dpi =300)



def model_performance(model_parameters, modelled_data, model_type= 'w_stay', save = True):
    '''
    Notes: Plots model accuracy and performance
    '''

    sns.set(style='white', font_scale = 2)
    num_mice  = len(modelled_data['mouse_name'].unique())
    mice = modelled_data['mouse_name'].unique()
    mod_param  = model_parameters.loc[model_parameters['model_name'] == model_type]
    ideal = modelled_data.copy()
    ideal = ideal.groupby('mouse_name').mean()
    for mouse in mice:
        ideal.loc[ideal.index == mouse, 'virus'] = \
            model_parameters.loc[model_parameters['mouse'] == mouse, 'virus'][0]

    fig, ax =  plt.subplots(1,3, figsize=(20,10))
    plt.sca(ax[0])
    sns.barplot(data=mod_param, x = 'virus', y = 'accu', palette = ['dodgerblue', 'orange'],
                 order=['chr2','nphr'])
    sns.swarmplot(data=mod_param, x = 'virus', y = 'accu', color='k', order=['chr2','nphr'])
    ax[0].set_ylim(0,1)
    ax[0].set_ylabel('Model Accuracy (%)')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Virus')
    plt.sca(ax[1])
    sns.barplot(data=ideal, x = 'virus', y = 'rewards', palette = ['dodgerblue', 'orange'],
                 order=['chr2','nphr'])
    sns.swarmplot(data=ideal, x = 'virus', y = 'rewards', color='k', order=['chr2','nphr'])
    ax[1].set_ylim(0,1)
    ax[1].set_title('Model Performance')
    ax[1].set_ylabel('Task Performance (%)')
    ax[1].set_xlabel('Virus')
    plt.sca(ax[2])
    sns.barplot(data=ideal, x = 'virus', y = 'real_rewards', palette = ['dodgerblue', 'orange'],
                 order=['chr2','nphr'])
    sns.swarmplot(data=ideal, x = 'virus', y = 'real_rewards', color='k', order=['chr2','nphr'])
    ax[2].set_ylim(0,1)
    ax[2].set_ylabel('Task Performance (%)')
    ax[2].set_title('Mouse Performance')
    ax[2].set_xlabel('Virus')
    sns.despine()
    plt.tight_layout()
    if save == True:
         plt.savefig('performance_model_real.pdf', dpi =300)



def simulate_and_plot(modelled_data, model_parameters,
                    model_type= 'w_stay'):
    mice = modelled_data['mouse_name'].unique()
    for mouse in mice:
        data_pd = modelled_data.loc[modelled_data['mouse_name'] == mouse,
                    ['real_rewards', 'signed_contrast', 'real_choice', 'laser',
                     'laser_side']]

        # -1 to 0 for laser
        data_pd.loc[data_pd['laser'] == -1, 'laser'] = 0
        # Make data into the right format
        data_np = data_pd.to_numpy()
        array_of_tuples = map(tuple, data_np.T)
        data = tuple(array_of_tuples)
        data2 = tuple(tuple(map(int, tup)) for tup in data[2:])
        data0 = tuple(tuple(map(int, data[0])))
        data1  = data[1]
        data = [data0, data1, data2[0], data2[1], data2[2]]
        params = model_parameters.loc[(model_parameters['mouse'] == mouse)
        & (model_parameters['model_name'] == model_type), 'x'].tolist()[0]

        # Multiply data by 1000
        data_m = []
        for i  in range(len(data)):
            data_m.append(data[i]*1)

        # Calculate Q values
        if model_type == 'standard':
           sim_data = generate_data(data_m, all_contrasts, learning_rate=params[0],
                                               beliefSTD=params[1], extraVal=params[2],
                                               beta=params[3], stay=params[4])

        if model_type == 'w_bias':
            sim_data = generate_data_bias(data_m, all_contrasts, learning_rate=params[0],
                                               beliefSTD=params[1], extraVal=params[2],
                                               beta=params[3], stay=params[4])

        if model_type == 'w_stay':
            sim_data = generate_data_stay(data_m, all_contrasts, learning_rate=params[0],
                                               beliefSTD=params[1], extraVal=params[2],
                                               beta=params[3], stay=params[4])

        if model_type == 'w_bias_n_stay':
           sim_data = generate_data_stay_and_bias(data_m, all_contrasts, learning_rate=params[0],
                                               beliefSTD=params[1], extraVal=params[2],
                                               beta=params[3], stay=params[4])
        # Plots
        sim_data = pd.DataFrame(sim_data)
        sim_data = np.array(sim_data)
        sim_data = pd.DataFrame(sim_data).T
        sim_data['laser'] = data_m[3]
        sim_data['laser_side'] = data_m[4]
        sim_data['real_choice'] = data_m[2]
        sim_data['mouse_name']  = mouse
        sim_data['real_rewards']  = data_m[0]
        sim_data = sim_data.rename(columns={0: "rewards", 1: "signed_contrast", 2: "simulated_choices", 3: "model_laser"})

        model_psychometrics(sim_data, data_pd, mouse, save = True)

    return modelled_data

def calculate_QL_QR(modelled_data, model_parameters,
                    model_type= 'w_stay', zero = False, retrieve_ITIQ = False):
    # Also calculates pRight

    ACC = []
    mice = modelled_data['mouse_name'].unique()
    for mouse in mice:
        data_pd = modelled_data.loc[modelled_data['mouse_name'] == mouse,
                    ['real_rewards', 'signed_contrast', 'real_choice', 'laser',
                     'session_switches']]

        # -1 to 0 for laser
        data_pd.loc[data_pd['laser'] == -1, 'laser'] = 0
        # Make data into the right format
        data_np = data_pd.to_numpy()
        array_of_tuples = map(tuple, data_np.T)
        data = tuple(array_of_tuples)
        data2 = tuple(tuple(map(int, tup)) for tup in data[2:])
        data0 = tuple(tuple(map(int, data[0])))
        data1  = data[1]
        data = [data0, data1, data2[0], data2[1], data2[2]]
        params0 = model_parameters.loc[(model_parameters['mouse'] == mouse)
        & (model_parameters['model_name'] == model_type), 'x'].copy()
        params1 = params0.tolist()[0].copy()

        if zero ==  True:
            params1[2] = 0
        # Calculate Q values
        if model_type == 'w_stay':
            _, acc, Q_L, Q_R, Q_LL, Q_RL, Q_LR, Q_RR, pRight = session_neg_log_likelihood_stay(params1,
                *data, pregen_all_posteriors=True, accu=True, retrieve_Q=True,
                retrieve_ITIQ=retrieve_ITIQ)


        # Return Q values to matrix
        ACC.append(acc)
        if (zero == True) & (retrieve_ITIQ==False):
            print('zero no ITI')
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QL0'] = Q_L
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QR0'] = Q_R
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'pRight0'] = pRight
            modelled_data['QRQL0'] = modelled_data['QR0'].to_numpy() - \
                modelled_data['QL0'].to_numpy()

            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QLL0'] = Q_LL
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QRL0'] = Q_RL
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QLR0'] = Q_LR
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QRR0'] = Q_RR

        elif (zero == False) & (retrieve_ITIQ==False):
            print('standard')
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QL'] = Q_L
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QR'] = Q_R
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'pRight'] = pRight
            # Calculate QR-QL
            modelled_data['QRQL'] = modelled_data['QR'].to_numpy() - \
                modelled_data['QL'].to_numpy()

            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QLL'] = Q_LL
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QRL'] = Q_RL
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QLR'] = Q_LR
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'QRR']  = Q_RR

        elif (zero == True) & (retrieve_ITIQ==True):
            print('zero with ITI')
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQL0'] = Q_L
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQR0'] = Q_R
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIpRight0'] = pRight
            modelled_data['ITIQRQL0'] = \
                modelled_data['ITIQR0'].to_numpy() - \
                modelled_data['ITIQL0'].to_numpy()

            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQLL0'] = Q_LL
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQRL0'] = Q_RL
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQLR0'] = Q_LR
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQRR0'] = Q_RR

        elif (zero == False) & (retrieve_ITIQ==True):
            print('standard ITI')
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQL'] = Q_L
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQR'] = Q_R
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIpRight'] = pRight
            # Calculate QR-QL
            modelled_data['ITIQRQL'] = \
                modelled_data['ITIQR'].to_numpy() - \
                modelled_data['ITIQL'].to_numpy()

            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQLL'] = Q_LL
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQRL'] = Q_RL
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQLR'] = Q_LR
            modelled_data.loc[modelled_data['mouse_name'] == mouse,
                        'ITIQRR'] = Q_RR

    return modelled_data


def boxplot_model_parameters_per_mouse(model_parameters,
                                       model_type= 'w_stay',
                                       save = True):
    '''
    Notes: Plots learned parameters across virus
    '''
    fig, ax =  plt.subplots()
    sns.set(style='white')
    model = model_parameters.loc[model_parameters['model_name'] == model_type,
                                 ['x', 'virus']]
    params = [r'$\alpha$', r'$\theta$', r'$\psi$',
              r'$\tau$', r'$\gamma$', r'$\phi$']
    mod = pd.DataFrame(columns = ['params', 'virus'])
    for i in range(len(model)):
        temp_mod = pd.DataFrame(model['x'].iloc[i])
        temp_mod['params'] = params[:len(temp_mod)]
        temp_mod['virus'] = model.iloc[i,1]
        mod = mod.append(temp_mod)
    sns.swarmplot(data = mod,  x = 'params', y = 0,  hue = 'virus',
                  palette = ['dodgerblue', 'orange'], split = False)
    ax.axhline(0, ls='--', color = 'k')
    ax.set_xlabel('Model Parameter')
    ax.set_ylabel('Fitted Coef')
    sns.despine()
    if save == True:
         plt.savefig('learned_parameters.pdf', dpi =300)

def model_psychometrics(sim_data, data_pd, mouse, save = True):
    '''
    Notes: Plots psychometrics
    '''

    sns.set(style='white', font_scale = 2, rc={"lines.linewidth": 2.5})
    fig, ax =  plt.subplots( figsize = (10,10))
    plt.sca(ax)
    sns.lineplot(data = data_pd, x = 'signed_contrast', y = 'real_choice',
                     hue = 'laser_side', palette = ['k','b','g'], ci =68, legend=None)
    # Plot model data with dash line
    sns.lineplot(data = sim_data, x = 'signed_contrast', y = 'simulated_choices',
                     hue = 'laser_side', palette = ['k','b','g'], ci =0, legend=None)
    ax.lines[3].set_linestyle("--")
    ax.lines[4].set_linestyle("--")
    ax.lines[5].set_linestyle("--")
    ax.set_ylabel('Fraction of Choices', fontsize =20)
    ax.set_xlabel('Signed Contrast', fontsize =20)
    ax.set_title(mouse, fontsize =20)
    lines = [Line2D([0], [0], color='black', linewidth=1.5, linestyle='-'),
    Line2D([0], [0], color='black', linewidth=1.5, linestyle='--')]
    labels = ['Real Data', 'Model']
    plt.legend(lines, labels)
    sns.despine()
    if save == True:
         plt.savefig('model_psychometric_'+mouse+'.png', dpi =300)

def model_choice_prob(psy, mouse, save = True):
    '''
    Notes: Plots psychometrics
    '''
    sns.set(style='white', font_scale = 2, rc={"lines.linewidth": 2.5})
    fig, ax =  plt.subplots( figsize = (10,10))
    plt.sca(ax)
    psy_select = psy.loc[psy['mouse_name'] == mouse].copy()
    psy_select['choice'] = psy_select['choice'] * -1
    psy_select.loc[psy_select['choice'] == -1, 'choice'] = 0
    sns.lineplot(data = psy_select, x = 'signed_contrasts', y = 'choice',
                     hue = 'opto_block', hue_order = ['non_opto','L','R'],
                     palette = ['k','b','g'], ci =68, legend=None)
    # Plot model data with dash line
    sns.lineplot(data = psy_select, x = 'signed_contrasts', y = 1*(psy_select['pRight']>0.5),
                     hue = 'opto_block', hue_order = ['non_opto','L','R'],
                     palette = ['k','b','g'], ci =0, legend=None)
    ax.lines[3].set_linestyle("--")
    ax.lines[4].set_linestyle("--")
    ax.lines[5].set_linestyle("--")
    ax.set_ylabel('Fraction of Choices', fontsize =20)
    ax.set_xlabel('Signed Contrast', fontsize =20)
    ax.set_xlim(-0.3,0.3)
    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylabel('pRight', color='k', fontsize =20)  # we alread
    ax.set_title(mouse, fontsize =20)
    lines = [Line2D([0], [0], color='black', linewidth=1.5, linestyle='-'),
    Line2D([0], [0], color='black', linewidth=1.5, linestyle='--')]
    labels = ['Real Data', 'Model']
    ax.axvline(0.3, ls='--', color = 'k')
    plt.legend(lines, labels)
    sns.despine()
    if save == True:
         plt.savefig('model_choice_prob_'+mouse+'.png', dpi =300)




def plot_q_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    #Get index for trials of block start
    index = psy_select.loc[psy['trial_within_block'] == 0, 'index']
    index = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
    for i in range(10):
        for idx in index-(i+1):
            psy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
                -(i+1)
            psy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
                psy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]

    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QRQL',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(-1,1)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('QR-QL')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'QRQL',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(-1,1)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('q_across_trials.svg', dpi =300)
        plt.savefig('q_across_trials.jpeg',  dpi =300)
        
        
        

def plot_pRight_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    #Get index for trials of block start
    index = psy_select.loc[psy['trial_within_block'] == 0, 'index']
    index = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
    for i in range(10):
        for idx in index-(i+1):
            psy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
                -(i+1)
            psy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
                psy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]

    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'pRight',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(0,1)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('pRight')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'pRight',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(0,1)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('q_across_trials.svg', dpi =300)
        plt.savefig('q_across_trials.jpeg',  dpi =300)


def plot_choice_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    psy_select['choice'] = psy_select['choice'] * - 1
    psy_select.loc[psy_select['choice'] == -1, 'choice']  = 0
    sns.set(style = 'white')
    #Get index for trials of block start
    index = psy_select.loc[psy['trial_within_block'] == 0, 'index']
    index = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
    for i in range(10):
        for idx in index-(i+1):
            psy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
                -(i+1)
            psy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
                psy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]

    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'choice',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(0,1)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('Choice')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block', y = 'choice',
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(-10,50)
    plt.ylim(0,1)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('choice_across_trials.svg', dpi =300)
        plt.savefig('choice_across_trials.jpeg',  dpi =300)





def plot_qmotivation_trial_whole_dataset(psy_df, save= True):
    psy_select = psy_df.copy()
    sns.set(style = 'white')
    fig, ax = plt.subplots(1,2, figsize = [10,5], sharey =True)
    palette ={'R':'g','L':'b','non_opto':'k'}
    plt.sca(ax[0])
    psy_chr2 = psy_select.loc[psy_select['virus']=='chr2']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y = psy_chr2['QR'] +  psy_chr2['QL'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-ChR2')
    ax[0].set_xlabel('Trial in block')
    ax[0].set_ylabel('QR+QL')

    plt.sca(ax[1])
    psy_chr2 = psy_select.loc[psy_select['virus']=='nphr']
    sns.lineplot(data = psy_chr2, x = 'trial_within_block',
                 y =  psy_chr2['QR'] +  psy_chr2['QL'],
                     hue = 'opto_block', palette = palette, ci=68)
    plt.xlim(0,50)
    plt.title('VTA-NpHR')
    ax[1].set_xlabel('Trial in block')
    plt.tight_layout()
    sns.despine()

    if save ==True:
        plt.savefig('qmotiv_across_trials.svg', dpi =300)
        plt.savefig('qmotiv_across_trials.jpeg',  dpi =300)



def plot_q_trial_whole_dataset_per_mouse(psy_df, save=True):
    psy_select = psy_df.copy()
    sns.set(style = 'white', font_scale=3)
    num_mice = len(psy_select['mouse_name'].unique())
    index = psy_select.loc[psy['trial_within_block'] == 0, 'index']
    index = index[1:].to_numpy() # Exclude first (trial index-10 does not exits)
    for i in range(10):
        for idx in index-(i+1):
            psy_select.loc[psy_select['index'] == idx, 'trial_within_block'] = \
                -(i+1)
            psy_select.loc[psy_select['index'] == idx, 'opto_block'] = \
                psy_select.loc[psy_select['index'] == idx+(i+1), 'opto_block'].to_numpy()[0]
    fig, ax = plt.subplots(1,num_mice, figsize = [60,20], sharey =True)
    for i, mouse in enumerate(psy_select['mouse_name'].unique()):
        palette ={'R':'g','L':'b','non_opto':'k'}
        plt.sca(ax[i])
        psy_mouse = psy_select.loc[psy_select['mouse_name']==mouse]
        sns.lineplot(data = psy_mouse, x = 'trial_within_block', y = 'QRQL',
                         hue = 'opto_block', palette = palette, ci=68)
        plt.xlim(-10,50)
        plt.ylim(-2,2)
        plt.title('VTA-'+str(psy_select.loc[psy_select['mouse_name']==mouse,
                                            'virus'].unique()) + '-' +
                                                                  str(mouse))
        ax[i].set_xlabel('Trial in block')
        ax[i].set_ylabel('QR-QL')
    sns.despine()
    if save ==True:
        plt.savefig('q_across_trials_p_mouse.svg', dpi =300)
        plt.savefig('q_across_trials_p_mouse.jpeg',  dpi =300)

def plot_choice_prob_opto_block(psy_df, ses_number, mouse_name, save =False):
    '''
    plot p choice right over trials
    Parameters
    ----------
    psy_df : dataframe with real data
    ses_number :number of  session (int)
    mouse_name : mouse name
    save : whether to save the figure
    Returns
    -------
    Figure with p choice over time, excludes firs 100 trials

    '''
    #
    # neutral block

    sns.set(style='white', font_scale=4)
    psy_df['right_block'] = np.nan
    psy_df['left_block'] = np.nan
    psy_df['right_block'] = (psy_df['opto_block'] == 'R')*1
    psy_df['left_block'] = (psy_df['opto_block'] == 'L')*1
    psy_df['non_opto_block'] = (psy_df['opto_block'] == 'non_opto')*1

    fig, ax1 = plt.subplots( figsize= (30,10))
    plt.sca(ax1)
    psy_subset =  psy_df.loc[psy_df['mouse_name'] == mouse_name].copy()
    psy_subset = psy_subset.loc[psy_subset['ses'] == psy_subset['ses'].unique()[ses_number]].reset_index()
    psy_subset['choice'] = psy_subset['choice']*-1
    psy_subset.loc[psy_subset['choice']==-1,'choice'] = 0
    p_choice = ((psy_subset['choice'].rolling(5).mean() +
                psy_subset['choice'].rolling(5).mean().shift(-5))/2)

    sns.lineplot(data = psy_subset, x = psy_subset.index,
                      y = p_choice,
                      color = 'k')

    plt.fill_between((np.arange(psy_subset['choice'].count())+1),
                     psy_subset['left_block'], color = 'blue', alpha =0.35)

    plt.fill_between((np.arange(psy_subset['choice'].count())+1),
                     psy_subset['right_block'], color = 'green', alpha =0.35)
    plt.fill_between((np.arange(psy_subset['choice'].count())+1),
                     psy_subset['non_opto_block'],
                     color ='black', alpha =0.35)
    sns.scatterplot(data = psy_subset, x = psy_subset.index,
                      y = 'choice',
                      color = 'k', s=100)
    # Probability of rightward choice

    plt.xlim(25,psy_subset['choice'].count())
    ax1.set_ylim(-0.1,1.1)


    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_ylabel('% Right Choices', color='k')
    ax1.set_xlabel('Trial', color='black')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    sns.lineplot(data = psy_subset, x = psy_subset.index,
                      y = ((psy_subset['QRQL'].rolling(5).mean() +
                           psy_subset['QRQL'].rolling(5).mean().shift(-5))/2),
                      color = 'red', ax = ax2)

    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylabel('QR - QL', color='red')  # we already handled the x-label wi
    ax2.set_ylim(-1,1)
    plt.tight_layout()

    plt.savefig('choice_and_q.svg')
    plt.savefig('choice_and_q.jpeg')


def plot_choice_40_trials(psy_df, ses_number, mouse_name, save =False):
    '''
    plot p choice right over trials
    Parameters
    ----------
    psy_df : dataframe with real data
    ses_number :number of  session (int)
    mouse_name : mouse name
    save : whether to save the figure
    Returns
    -------
    Figure with p choice over time, excludes firs 100 trials

    '''
    #
    # neutral block

    sns.set(style='white', font_scale=4)
    psy_df['right_block'] = np.nan
    psy_df['left_block'] = np.nan
    psy_df['right_block'] = (psy_df['opto_block'] == 'R')*1
    psy_df['left_block'] = (psy_df['opto_block'] == 'L')*1
    psy_df['non_opto_block'] = (psy_df['opto_block'] == 'non_opto')*1

    fig, ax1 = plt.subplots( figsize= (30,10))
    plt.sca(ax1)
    psy_subset =  psy_df.loc[psy_df['mouse_name'] == mouse_name].copy()
    psy_subset = psy_subset.loc[psy_subset['ses'] == psy_subset['ses'].unique()[ses_number]].reset_index()
    psy_subset['choice'] = psy_subset['choice']*-1
    psy_subset.loc[psy_subset['choice']==-1,'choice'] = 0

    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['left_block'], color = 'blue', alpha =0.35)

    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['right_block'], color = 'green', alpha =0.35)
    plt.fill_between((np.arange(psy_subset['choice'].count())+1), psy_subset['non_opto_block'],
                     color ='black', alpha =0.35)
    sns.scatterplot(data = psy_subset, x = psy_subset.index,
                      y = 1,
                      color = 'k',
                      hue = (psy_subset['signed_contrasts'].abs() * (1*(psy_subset['signed_contrasts']>=0))),
                      palette = 'Greys',
                      s = 1000,
                      legend= None, edgecolor='black')
    sns.scatterplot(data = psy_subset, x = psy_subset.index,
                      y = 0,
                      color = 'k',
                      hue = (psy_subset['signed_contrasts'].abs() * (1*(psy_subset['signed_contrasts']<=0))),
                             palette = 'Greys',
                      s = 1000,
                      legend= None, edgecolor='black')
    # Add feedback
    sns.scatterplot(data = psy_subset, x = psy_subset.index,
                      y = psy_subset['choice'],
                      color = 'k',
                      hue = psy_subset['feedbackType'], palette = ['r', 'g'],
                      s = 150,
                      legend= None, edgecolor='black')

    # Probability of rightward choice

    plt.xlim(200,140)
    plt.ylim(-0.1,1.1)


    ax1.tick_params(axis='y', labelcolor='k')
    ax1.set_ylabel('Choice', color='k')
    ax1.set_yticks([0,1])
    ax1.set_yticklabels(['Left', 'Right'])
    ax1.set_xlabel('Trial', color='black')
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    sns.lineplot(data = psy_subset, x = psy_subset.index,
                      y = psy_subset['QRQL'],
                      color = 'red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.axhline(0, ls='--', color = 'k', lw='4')
    ax2.set_ylabel('QR - QL', color='red')  # we already handled the x-label wi
    plt.tight_layout()

    plt.savefig('choice_and_40.svg')
    plt.savefig('choice_and_40.jpeg')

def true_stim_posterior(true_contrast, beliefSTD):

    def st_sp_0(percieve_contrast, beliefSTD=beliefSTD):
        all_contrasts = np.array([-0.25, -0.125, -0.0625, 0.0625, 0.125, 0.25])
        a = 0
        b = 0
        for i in all_contrasts[all_contrasts>0]:
            a += norm.pdf((percieve_contrast - i)/ beliefSTD)
        for i in all_contrasts:
            b += norm.pdf((percieve_contrast - i)/ beliefSTD)

        return a/b * norm.pdf(percieve_contrast,true_contrast,beliefSTD)

    bs_right = quad(st_sp_0,-1, 1)

    return [1-bs_right[0],bs_right[0]]


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

def softmax_stay(Q_L, Q_R, beta):
    p = [np.exp(Q_L / beta),
      np.exp(Q_R / beta)]
    p /= np.sum(p)

    return p

def trial_log_likelihood_stay(params, trial_data, Q, all_contrasts, all_posteriors,
                              previous_trial, trial_num, retrieve_Q = False, retrieve_ITIQ = False):
    # Get relevant parameters
    trial_contrast, trial_choice, reward, laser = trial_data
    learning_rate, beliefSTD, extraVal, beta = params
    Q =  Q.copy()

    # Compute the log-likelihood of the actual mouse choice
    if all_posteriors is None:
        contrast_posterior = true_stim_posterior(trial_contrast, beliefSTD)
    else:
        posterior_idx = np.argmin(np.abs(all_contrasts - trial_contrast))
        contrast_posterior = all_posteriors[posterior_idx, :]

    Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)
    # Weighted Q values for encoding/decoding
    posterior = np.array([[contrast_posterior[0]],[contrast_posterior[1]]])
    Q_w = Q * posterior
    Q_LL = Q_w[0,0]
    Q_RL = Q_w[1,0]
    Q_LR = Q_w[0,1]
    Q_RR = Q_w[1,1]

    if trial_num == 0:
        (l_stay, r_stay) = [0,0]
    else:
        previous_choice= [0,0]
        previous_choice[previous_trial] = 1
        (l_stay, r_stay) = previous_choice

    choice_dist = softmax_stay(Q_L, Q_R, beta)
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
        Q_LL = Q[0,0]
        Q_RL = Q[1,0]
        Q_LR = Q[0,1]
        Q_RR = Q[1,1]

    if retrieve_Q==True:
        return LL, Q, Q_L, Q_R, Q_LL, Q_RL, Q_LR, Q_RR, choice_dist[1] #  choice_dist[1] = pChoice_right

    else:
        return LL, Q



def session_neg_log_likelihood_stay(params, *data, pregen_all_posteriors=True,
                                    accu=False, retrieve_Q =  False, retrieve_ITIQ = False):
    # Unpack the arguments
    learning_rate, beliefSTD, extraVal, beta = params
    rewards, true_contrasts, choices, lasers, ses_switch = data
    num_trials = len(rewards)

    if retrieve_Q==True:
        Q_L = []
        Q_R = []
        Q_LL = []
        Q_RL = []
        Q_LR = []
        Q_RR = []
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
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]],
                                            Q, all_contrasts, all_posteriors,
                                            np.nan, i, retrieve_Q=retrieve_Q, retrieve_ITIQ=retrieve_ITIQ)
            else:
                if ses_switch[i] == 1:
                    Q = np.zeros([2,2])

                trial_LL, newQ, Q_Lt, Q_Rt,Q_LLt, Q_RLt, Q_LRt, Q_RRt, pright = trial_log_likelihood_stay(params,
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]],
                                            Q, all_contrasts, all_posteriors,
                                            choices[i-1], i, retrieve_Q=retrieve_Q, retrieve_ITIQ=retrieve_ITIQ)
            if (i != 0) & (np.sum(Q, axis=0)[0] != np.sum(newQ, axis=0)[0]) & (np.sum(Q, axis=0)[1] != np.sum(newQ, axis=0)[1]):
                print('Warning, double update error in trial %d'%i)
            LL += trial_LL
            Q = newQ

            if accu == True:
                acc += (np.exp(trial_LL)>0.5)*1

            Q_L.append(Q_Lt)
            Q_R.append(Q_Rt)
            Q_LL.append(Q_LLt)
            Q_RL.append(Q_RLt)
            Q_LR.append(Q_LRt)
            Q_RR.append(Q_RRt)
            pRight.append(pright)

    else:
        for i in range(num_trials):
            if i == 0:
                trial_LL, newQ = trial_log_likelihood_stay(params,
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]],
                                            Q, all_contrasts, all_posteriors, np.nan, i)
            else:
                if ses_switch[i] == 1:
                    Q = np.zeros([2,2])

                trial_LL, newQ = trial_log_likelihood_stay(params,
                                            [true_contrasts[i], choices[i], rewards[i], lasers[i]],
                                            Q, all_contrasts, all_posteriors, choices[i-1], i)
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

def optimizer_stay(data, num_fits = 3, initial_guess=[0.1, 1, 0, 1, 1]):
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
            lr_guess = np.random.uniform(0, 1)
            beliefSTD_guess = np.random.uniform(0.03, 1)
            extraVal_guess = np.random.uniform(-2,2)
            beta_guess = np.random.uniform(0.01, 1)
            initial_guess = [lr_guess, beliefSTD_guess, extraVal_guess, beta_guess]

        # Run the fit
        res = so.minimize(session_neg_log_likelihood_stay, initial_guess, args=data,
                    method='L-BFGS-B', bounds=[(0, 2), (0.03, 1), (-2, 2), (0.01, 1)])

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
                       beliefSTD=0.1, extraVal=1, beta=0.2,
                       stay = 1, is_verbose=False, propagate_errors = True):

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
        # Add noise

        contrast_posterior = true_stim_posterior(trial_contrast, beliefSTD)

        Q_L, Q_R = compute_QL_QR(Q, trial_contrast, contrast_posterior)

        if t == 0:
            (l_stay, r_stay) = [0,0]
        else:
            previous_choice= [0,0]
            previous_choice[choices[t-1]] = 1
            (l_stay, r_stay) = previous_choice
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
                reward += extraVal
                lasers.append(1)
            else:
                lasers.append(-1)
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
            Q[i, choice] += contrast_posterior[i] * learning_rate * (reward - Q[i, choice])

    return rewards, true_contrasts, choices, lasers



def transform_model_struct_2_POMDP(model_data, simulate_data):
        simulate_data.loc[simulate_data['extraRewardTrials'] == 'right', 'extraRewardTrials' ] = 1
        simulate_data.loc[simulate_data['extraRewardTrials'] == 'left', 'extraRewardTrials' ] = 0
        simulate_data.loc[simulate_data['extraRewardTrials'] == 'none', 'extraRewardTrials' ] = -1
        obj = model_data
        obj['choice'] = obj['choice'] * -1
        obj.loc[obj['choice'] == -1, 'choice'] = 0
        obj['laser_side'] = simulate_data['extraRewardTrials']
        return obj

def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))

def aic(LL,n_param):
    # Calculates Akaike Information Criterion
    aic =  2*n_param - 2*LL
    return aic

def chi2_LLR(L1,L2):
    LR = likelihood_ratio(L1,L2)
    p = chi2.sf(LR, 1) # L2 has 1 DoF more than L1
    return p

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
#              ((psy['ses']>'2020-03-13') & (psy['ses']<'2020-03-19')
#              ((psy['ses']>'2020-03-13') & (psy['mouse_name'] != 'dop_9')) |
#               & (psy['mouse_name'] == 'dop_9'))]

train_set_size = 1
cross_validate = False

all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.    ,  0.0625, 0.125 , 0.25  ])
best_nphr_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),4])
best_nphr_NLL_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])
model_parameters = pd.DataFrame()
modelled_data = pd.DataFrame()
for i, mouse in enumerate(mice):
        model_data_nphr, simulate_data_nphr  = \
            psy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse],
                                              virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0])
        # Obtain sessions switches
        session_switches = np.zeros(len(model_data_nphr))
        for session in model_data_nphr['ses'].unique():
             session_switches[model_data_nphr.ses.ge(session).idxmax()]=1

        obj = transform_model_struct_2_POMDP(model_data_nphr, simulate_data_nphr)


        virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0]

        opto = obj['extraRewardTrials'].to_numpy()
        lasers = []
        for i in range(len(opto)):
            try:
                lasers.append(int(opto[i][0]))
            except:
                lasers.append(int(opto[i]))

        choices = list(obj['choice'].to_numpy())
        contrasts = list(obj['stimTrials'].to_numpy())
        rewards = list(obj['reward'].to_numpy())
        laser_side = list(obj['laser_side'].to_numpy())


        data = (rewards[:int(len(rewards)*train_set_size)],
                contrasts[:int(len(rewards)*train_set_size)],
                choices[:int(len(rewards)*train_set_size)],
                lasers[:int(len(rewards)*train_set_size)],
                session_switches[:int(len(rewards)*train_set_size)])
        simulate_data = (rewards[:int(len(rewards)*train_set_size)],
                         contrasts[:int(len(rewards)*train_set_size)],
                         choices[:int(len(rewards)*train_set_size)],
                      lasers[:int(len(rewards)*train_set_size)],
                      laser_side[:int(len(rewards)*train_set_size)],
                      session_switches[:int(len(rewards)*train_set_size)])


        data_test = data
        simulate_data_test = simulate_data


        (best_x_stay, train_NLL_stay, buffer_NLL_stay,
         buffer_x_stay) = optimizer_stay(data, initial_guess=[0.3, 0.05, -1, 0.2])

        cv_aic_stay = aic((session_neg_log_likelihood_stay(best_x_stay,
                  *data_test, pregen_all_posteriors=True))*-1,5)


        cv_LL_stay = (session_neg_log_likelihood_stay(best_x_stay, *data_test,
                                                      pregen_all_posteriors=True))*-1



        _, cv_acc_stay = session_neg_log_likelihood_stay(best_x_stay, *data_test,
                       pregen_all_posteriors=True, accu=True)


        model_parameters_mouse = pd.DataFrame()
        model_parameters_mouse['x'] = [best_x_stay]
        model_parameters_mouse['LL'] = (cv_LL_stay/len(data_test[0]))
        model_parameters_mouse['aic'] = cv_aic_stay
        model_parameters_mouse['accu'] = cv_acc_stay
        model_parameters_mouse['model_name'] = 'w_stay'
        
        # Concatenate with general dataframes
        model_parameters_mouse['mouse'] = mouse
        model_parameters_mouse['virus'] = virus

        # Concatenate with general dataframes
        model_parameters = pd.concat([model_parameters, model_parameters_mouse])