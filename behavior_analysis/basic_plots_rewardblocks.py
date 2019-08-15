#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 21:06:23 2019
Need to reinstall module, at the moment I need to be in npy2pd module, same for basic plots
@author: ibladmin
"""
from npy2pd import *
from basic_plots import *
from glm import *

#Input folder with raw npy files
psy_raw = load_data('/mnt/s0/Data/Subjects_personal_project/rewblocks10070/')
psy_df  = unpack(psy_raw)

#Plot psychometric data
blocks =  np.array([1, 0.7])
plot_psych_block (psy_df , 'rewprobabilityLeft', blocks)
bias_per_session(psy_df, 'rewprobabilityLeft', blocks)

#Plot glm
#include bias blocks only
psy_df =  psy_df.loc[(psy_df['rewprobabilityLeft'] == 1) | (psy_df['rewprobabilityLeft'] == 0.7)]

#flip psy_df choice so that right is 1 (aesthetic change)
psy_df['choice']  = psy_df['choice']*-1


result, r2 =  glm_logit(psy_df, sex_diff =  False)
plot_glm(psy_df, result, r2)
