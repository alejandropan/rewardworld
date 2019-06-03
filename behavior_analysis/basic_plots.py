#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:09:19 2019
Behavior statistics
Some based on @anneurai and IBL datajoint pipeline
Also functions copied from alex_psy
@author: ibladmin
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 01:11:57 2019

@author: Alejandro
"""
import numpy as np
import pandas as pd
import datajoint as dj
from ibl_pipeline.analyses import behavior as behavior_analysis
from ibl_pipeline import reference, subject, action, acquisition, data, behavior
import seaborn as sns

def ibl_psychometric (psy_df, block_variable):
    """Calculates and plot psychometic curve from dataframe
    datajoint independent
    assumes that there is not signed contrast in dataframe
    INPUTS: Dataframe where index is the trial number, block variable is optional 
    if present it will use bloc_variable as hue
    OUTPUTS:  psychometic fit using IBL function from Miles"""
        
    #1st calculate some useful data...
    psy_df['contrastRight'] = psy_df['contrastRight'].fillna(0)
    psy_df['contrastLeft']  = psy_df['contrastLeft'].fillna(0)
    psy_df['signed_contrasts'] =  psy_df['contrastRight'] - psy_df['contrastLeft']
    unique_signed_contrasts  = sorted(psy_df['signed_contrasts'].unique())
    
    right_choices = psy_df['choice']== -1
            total_trials = []
            right_trials = []
            
            for cont in unique_signed_contrasts:
                matching = (signed_contrasts == cont)
                total_trials.append(np.sum(matching))
                right_trials.append(np.sum(right_choices[matching]))
            
            prop_right_trials = np.divide(right_trials, total_trials)
            psy_df.loc[row,:] = prop_right_trials
        
      return psy_df
  
    
    for cont in unique_signed_contrasts:
        matching = (signed_contrasts == cont)
        total_trials.append(np.sum(matching))
        right_trials.append(np.sum(right_choices[matching]))
    
    prop_right_trials = np.divide(right_trials, total_trials)
    
    pars, L = psy.mle_fit_psycho(
            np.vstack([unique_signed_contrasts * 100, total_trials, prop_right_trials]),
            P_model='erf_psycho_2gammas',
            parstart=np.array([np.mean(unique_signed_contrasts), 20., 0.05, 0.05]),
            parmin=np.array([np.min(unique_signed_contrasts), 0., 0., 0.]),
            parmax=np.array([np.max(unique_signed_contrasts), 100., 1, 1]))
    
    x = np.linspace(-100, 100)
    y = psy.erf_psycho_2gammas(pars, x)
    
    psy_df_local = psy_by_mouse_local (unique_signed_contrasts,'wittenlab')
    y_local = psy_df_local.mean()
    x_local = unique_signed_contrasts
    local_error =  psy_df_local.std()