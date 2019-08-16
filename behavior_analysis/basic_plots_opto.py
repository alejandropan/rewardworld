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
psy_raw = load_data('/mnt/s0/Data/Subjects_personal_project/standard_task_opto/')
psy_df  = unpack(psy_raw)

#Shift opto to slice through next trial
for mouse in psy_df['mouse_name'].unique():
    for day in psy_df.loc[psy_df['mouse_name']== mouse,'ses'].unique():
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_opto'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'opto.npy'].shift(periods=1)

for mouse in psy_df['mouse_name'].unique():
    for day in psy_df.loc[psy_df['mouse_name']== mouse,'ses'].unique():
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'previous_choice'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'choice'].shift(periods=1)

for mouse in psy_df['mouse_name'].unique():
    for day in psy_df.loc[psy_df['mouse_name']== mouse,'ses'].unique():
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day),'after_win'] = \
        psy_df.loc[(psy_df['mouse_name']== mouse) & (psy_df['ses']== day), 'feedbackType'].shift(periods=1)
        
#1st Plot After laser trials across groups
#Grouping variables
block_variable = 'after_opto'
blocks = [1, 0]
block2_variable ='s.probabilityLeft'
blocks2 = [0.8, 0.5, 0.2]
        

#Plot across different trial history groups
general = psychometric_summary(psy_df , block_variable, blocks, block2_variable, blocks2)
winner = psychometric_summary(psy_df.loc[psy_df['after_win']==1] , block_variable, blocks, block2_variable, blocks2)
loser = psychometric_summary(psy_df.loc[psy_df['after_win']==-1] , block_variable, blocks, block2_variable, blocks2)

#Save figs
general.savefig('general.pdf')
winner.savefig('winner.pdf')
loser.savefig('loser.pdf')


#plot_psych_var_block(nphr_r,'after_opto', blocks, 's.probabilityLeft', blocks2 )


#2nd plot psychometrics after wins


#3rd plot psychometrics after loses 



#4th Same as above for chronometric (different figure)
 


#5th Chronometric measures 


#Probability to stay
stay_bi_on = len(chr2_bi.loc[(chr2_bi['after_opto']==1) & (chr2_bi['choice'] == chr2_bi['previous_choice'] ) & (chr2_bi['contrastLeft']==1) & (chr2_bi['s.probabilityLeft'] == 0.2) ])/len(chr2_bi.loc[(chr2_bi['after_opto']==1) &(chr2_bi['contrastLeft']==1) & (chr2_bi['s.probabilityLeft'] == 0.2)])
stay_bi_off  = len(chr2_bi.loc[(chr2_bi['after_opto']==0) & (chr2_bi['choice'] == chr2_bi['previous_choice']) & (chr2_bi['contrastLeft']==1) & (chr2_bi['s.probabilityLeft'] == 0.2)])/len(chr2_bi.loc[(chr2_bi['after_opto']==0) & (chr2_bi['contrastLeft']==1)& (chr2_bi['s.probabilityLeft'] == 0.2) ])

& (chr2_bi['s.probabilityLeft'] == 0.5)

nphr_bi
nphr_l
nphr_r

#Plot psychometric data
blocks =  np.array([1, 0])
plot_psych_block (chr2_l.loc[chr2_l['s.probabilityLeft']==0.2] , 'after_opto', blocks)



#Plot glm
#include bias blocks only
psy_df =  psy_df.loc[(psy_df['rewprobabilityLeft'] == 1) | (psy_df['rewprobabilityLeft'] == 0.7)]

#flip psy_df choice so that right is 1 (aesthetic change)
psy_df['choice']  = psy_df['choice']*-1


result, r2 =  glm_logit(psy_df, sex_diff =  False)
plot_glm(psy_df, result, r2)
