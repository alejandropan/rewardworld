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
        

#Plot across different trial history groups (after win and loses)
general = psychometric_summary(psy_df , block_variable, blocks, block2_variable, blocks2)
winner = psychometric_summary(psy_df.loc[psy_df['after_win']==1] , block_variable, blocks, block2_variable, blocks2)
loser = psychometric_summary(psy_df.loc[psy_df['after_win']==-1] , block_variable, blocks, block2_variable, blocks2)

#Save figs
general.savefig('general.pdf')
winner.savefig('winner.pdf')
loser.savefig('loser.pdf')


#Plot glm with opto as a regressor for the different stimulation types
chr2_result, chr2_r2 = glm_logit_opto(psy_df.loc[(psy_df['hem_stim'] == 'B') & (psy_df['virus']== 'chr2') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) ])
nphr_result, nphr_r2 = glm_logit_opto(psy_df.loc[(psy_df['hem_stim'] == 'B') & (psy_df['virus']== 'nphr') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) ])
chr2_l_result, chr2_l_r2 = glm_logit_opto(psy_df.loc[(psy_df['hem_stim'] == 'L') & (psy_df['virus']== 'chr2') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) ])
nphr_l_result, nphr_l_r2 = glm_logit_opto(psy_df.loc[(psy_df['hem_stim'] == 'L') & (psy_df['virus']== 'nphr') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) ])
chr2_r_result, chr2_r_r2 = glm_logit_opto(psy_df.loc[(psy_df['hem_stim'] == 'R') & (psy_df['virus']== 'chr2') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) ])
nphr_r_result, nphr_r_r2 = glm_logit_opto(psy_df.loc[(psy_df['hem_stim'] == 'R') & (psy_df['virus']== 'nphr') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) ])
    

#after opto vs non after opto
chr2_bi_after_opto  = psy_df.loc[(psy_df['hem_stim'] == 'B') & (psy_df['virus']== 'chr2') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) & (psy_df['after_opto'] ==1 )]

nphr_bi_after_opto  = psy_df.loc[(psy_df['hem_stim'] == 'B') & (psy_df['virus']== 'chr2') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) & (psy_df['after_opto'] ==0 )]


chr2_l_after_opto = psy_df.loc[(psy_df['hem_stim'] == 'L') & (psy_df['virus']== 'chr2') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) & (psy_df['after_opto'] ==1 )]

nphr_l_after_opto = psy_df.loc[(psy_df['hem_stim'] == 'L') & (psy_df['virus']== 'chr2') &\
                                                 (psy_df['s.probabilityLeft'] == 0.2) |(psy_df['s.probabilityLeft'] == 0.8) & (psy_df['after_opto'] ==0 )]




laser_on_result, laser_on_r2 = glm_logit(chr2_l_after_opto)

laser_off_result, laser_on_r2 = glm_logit(nphr_l_after_opto)


