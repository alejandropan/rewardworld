#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:07:29 2020

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

psy = pd.read_pickle('all_behav.pkl')
psy = psy.loc[((psy['ses']>'2020-01-13') & (psy['mouse_name'] == 'dop_4')) | 
              (psy['ses']>'2020-03-13')]

psy['choice'] = psy['choice'] *-1
psy['choice'] = (psy['choice']>0)*1
pal ={-1:"k",0:"g",1:"b"}
for m, mouse in enumerate(psy['mouse_name'].unique()):
    fig, ax  = plt.subplots(3,5, figsize = (25,20))
    psy_local = psy.loc[(psy['mouse_name'] == mouse)]
    for s, ses in enumerate(psy_local['ses'].unique()):
        ses_local  = psy_local.loc[(psy_local['ses']==ses)]
        plt.sca(ax[0,s])
        sns.pointplot(data = ses_local.iloc[:300,:], x = 'signed_contrasts',
                         y =  'choice',  hue = 'opto_probability_left', ci  = 66,
                         legend = None, palette = pal)
        ax[0,s].set_ylim(0,1)
        ax[0,s].set_title(str(mouse) +str(ses) + 'early')
        ax[0,s].set_xlabel('Fraction of rightward choices')
        plt.sca(ax[1,s])
        sns.pointplot(data = ses_local.iloc[-300:,:], x = 'signed_contrasts',
                         y =  'choice',  hue = 'opto_probability_left', ci  = 66,
                         legend = None, palette = pal,)
        ax[1,s].set_title(str(mouse) +str(ses) + 'late')
        ax[1,s].set_ylim(0,1)
        ax[1,s].set_xlabel('Fraction of rightward choices')
        plt.sca(ax[2,s])
        sns.pointplot(data = ses_local, x = 'signed_contrasts',
                      y =  'choice',  hue = 'opto_probability_left', ci  = 66,
                         legend = None, palette = pal,)
        ax[2,s].set_title(str(mouse) +str(ses) + 'total')
        ax[2,s].set_ylim(0,1)
        ax[2,s].set_xlabel('Fraction of rightward choices')
    plt.tight_layout()
    plt.savefig('early_late'+str(mouse) +'.png')

        
