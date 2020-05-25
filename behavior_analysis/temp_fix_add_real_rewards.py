#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 18:00:47 2020

@author: alex
"""

modelled_data['real_rewards'] = np.nan

for i, mouse in enumerate(psy['mouse_name'].unique()): 
        model_data_nphr, simulate_data_nphr  = \
            psy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse], 
                                              virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0])
        
        
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
                lasers[:int(len(rewards)*train_set_size)])
        simulate_data = (rewards[:int(len(rewards)*train_set_size)], 
                         contrasts[:int(len(rewards)*train_set_size)], 
                         choices[:int(len(rewards)*train_set_size)], 
                      lasers[:int(len(rewards)*train_set_size)], 
                      laser_side[:int(len(rewards)*train_set_size)])
        
        if cross_validate == True:
            
            data_test = (rewards[int(len(rewards)*train_set_size):], 
                         contrasts[int(len(rewards)*train_set_size):], 
                         choices[int(len(rewards)*train_set_size):], 
                         lasers[int(len(rewards)*train_set_size):])
            simulate_data_test = (rewards[int(len(rewards)*train_set_size):], 
                                  contrasts[int(len(rewards)*train_set_size):], 
                                  choices[int(len(rewards)*train_set_size):], 
                          lasers[int(len(rewards)*train_set_size):], 
                          laser_side[int(len(rewards)*train_set_size):])
        else:
            data_test = data
            simulate_data_test = simulate_data
        
        modelled_data.loc[modelled_data['mouse_name'] == mouse, 'laser_trials']  = simulate_data[3]
        modelled_data.loc[modelled_data['mouse_name'] == mouse, 'real_rewards']  = simulate_data[0]
       