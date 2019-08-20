#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 20:04:01 2019

@author: ibladmin
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd

def psychometric_summary(psy_df , block_variable, blocks, block2_variable, blocks2):
    """
    INPUTS:
    block_variable: string with name of block variable 1 in this case "after_opto"  -  
    Boolean array determining whether the trial was preceeded by laser on
    block2_variable: string with name of the variable 2 in this case "s.probabilityLeft"
    conditions:  List of potential hemisphere
    viruses: List of viruses
    blocks2: List of blocks of stimuli priors (0.8, 0.5, 0.2)
    blocks: Hast to be [1,0] in this order, it defines the laser on and off
    OUTPUTS:
    Figure with psychometrics
    
    
    pars  : bias    = pars[0]
   threshold    = pars[1]
   gamma1    = pars[2]
   gamma2    = pars[3]
    """
    #Clarify variables:
    conditions  = psy_df['hem_stim'].unique()
    mice = psy_df['mouse_name'].unique()
    viruses =  psy_df['virus'].unique()
    figure,ax = plt.subplots(12,4, figsize=(24,100))
    #psy_measures_rows = np.nan(len(blocks)*len(blocks2)*len(conditions)*len(mice))
    psy_measures =  pd.DataFrame(columns = ['mouse_name','virus','laser_on','probabilityLeft','conditions','bias','threshold', 'gamma1', 'gamma2'  ])
    
    #Plot summaries divided by hem stimulated and virus
    for v, virus in enumerate(viruses):
        for c , hem in enumerate(conditions):
            plt.sca(ax[v,c])
            sns.set()
            lines = ['dashed','solid']
            colors =['green','black','blue']
            for j,i in enumerate(blocks):
                for p, bl2 in enumerate(blocks2):
                    psy_df_block  = psy_df.loc[(psy_df['hem_stim'] == hem) & (psy_df['virus'] == virus) & (psy_df[block2_variable] == bl2) & (psy_df[block_variable] == i)] 
                    pars,  L  =  ibl_psychometric (psy_df_block)
                    plt.plot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), linewidth=2,\
                             color = colors[p], linestyle = lines[j])
                    sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", \
                                 linewidth=0, linestyle='None', mew=0.5,marker='.',
                                 color = colors[p],  ci=68, data= psy_df_block)
                    
                    
            ax[v,c].set_xlim([-25,25])
            ax[v,c].set_title(virus +' '+ hem)
            ax[v,c].set_ylabel('Fraction of CW choices')
            ax[v,c].set_xlabel('Signed contrast')
            green_patch = mpatches.Patch(color='green', label='P(L) = 0.8')
            black_patch  =  mpatches.Patch(color='black', label='P(L) = 0.5')
            blue_patch  =  mpatches.Patch(color='blue', label='P(L) = 0.2')
            dashed  =  plt.Line2D([0], [0], color='black', lw=2, label='Laser on', ls = '--')
            solid  = plt.Line2D([0], [0], color='black', lw=2, label='Laser off', ls = '-')
            plt.legend(handles=[dashed, solid,green_patch, black_patch, blue_patch], loc = 'lower right')
        
    #Plor summaries as above but per mouse
    
    for m, mouse in enumerate(mice):
        for c , hem in enumerate(conditions):
            plt.sca(ax[m+2,c])
            sns.set()
            lines = ['dashed','solid']
            colors =['green','black','blue']
            virus =  psy_df.loc[(psy_df['mouse_name'] == mouse), 'virus'].unique()[0]
            for j,i in enumerate(blocks):
                for p, bl2 in enumerate(blocks2):
                    psy_df_block  = psy_df.loc[(psy_df['hem_stim'] == hem) & (psy_df['mouse_name'] == mouse) & (psy_df[block2_variable] == bl2) & (psy_df[block_variable] == i)] 
                    pars,  L  =  ibl_psychometric (psy_df_block)
                    plt.plot(np.arange(-100,100), psy.erf_psycho_2gammas( pars, np.arange(-100,100)), linewidth=2,\
                             color = colors[p], linestyle = lines[j])
                    sns.lineplot(x='signed_contrasts', y='right_choices', err_style="bars", \
                                 linewidth=0, linestyle='None', mew=0.5,marker='.',
                                 color = colors[p],  ci=68, data= psy_df_block)
                    psy_measures = psy_measures.append({'mouse_name': mouse, 'virus':virus, 'laser_on': i,\
                                                       'probabilityLeft': bl2 , 'conditions':hem,'threshold': pars[1],'bias':pars[0],\
                                                       'gamma1':pars[2],'gamma2':pars[3]}, ignore_index=True)
            ax[m+2,c].set_xlim([-25,25])
            ax[m+2,c].set_title(mouse +' ' + virus +' '+ hem)
            ax[m+2,c].set_ylabel('Fraction of CW choices')
            ax[m+2,c].set_xlabel('Signed contrast')
            green_patch = mpatches.Patch(color='green', label='P(L) = 0.8')
            black_patch  =  mpatches.Patch(color='black', label='P(L) = 0.5')
            blue_patch  =  mpatches.Patch(color='blue', label='P(L) = 0.2')
            dashed  =  plt.Line2D([0], [0], color='black', lw=2, label='Laser on', ls = '--')
            solid  = plt.Line2D([0], [0], color='black', lw=2, label='Laser off', ls = '-')
            plt.legend(handles=[dashed, solid,green_patch, black_patch, blue_patch], loc = 'lower right')
    
    parameters = ['bias','threshold', 'gamma1', 'gamma2' ]
    
    row_counter = 5 #indexing to plot automatically
    for i,c in enumerate(conditions):
        for j,v in enumerate(viruses):
            row_counter = row_counter + 1
            for p, pars in enumerate(parameters):
                
                sns.catplot(data = psy_measures.loc[(psy_measures['conditions']== c) & (psy_measures['virus']==v)],\
                                                    x = 'probabilityLeft', y = pars,  hue = 'laser_on', ax = ax[row_counter,p], kind="bar" )
                ax[row_counter,p].set_title(c +' ' + v +' '+ pars)
                ax[row_counter,p].set_ylabel(pars)
                ax[row_counter,p].set_xlabel('Block (Probability Left)')
    return figure 
    
        #Plot differences in psychometric measures  bar plots (bias, lapse right, lapse left, threshold)
        #1st row comparison across pooled conditions chr2
        
        #2nd row comparison acroos pooled conditions nphr3
    
    
def opto_glm(psy_df_global):
    viruses =  psy_df_global['virus'].unique()
    figure,ax = plt.subplots(2,3, figsize=(24,24))
    for v, virus in enumerate(viruses):
        conditions  = psy_df_global.loc[(psy_df_global['virus']== virus),'hem_stim'].unique()
        for c , hem in enumerate(['B','L']): #(conditions)
            psy_df = psy_df_global.loc[(psy_df_global['virus']== virus) & (psy_df_global['hem_stim']== hem)]
            pool_rew_predictors  = pd.DataFrame()
            pool_urew_predictors  = pd.DataFrame()
            pool_rew_opto_predictors = pd.DataFrame()
            pool_urew_opto_predictors = pd.DataFrame()
            
            plt.sca(ax[v,c])
            
            #Have to cahnge glm function so that it has the optopn to run normal glm only on opto.npy trials
            for i,mouse in enumerate(psy_df['mouse_name'].unique()):
                mouse_result, mouse_r2  = glm_logit(psy_df.loc[(psy_df['mouse_name']==mouse)], sex_diff = False)
                mouse_result_opto, mouse_r2_opto  = glm_logit(psy_df.loc[(psy_df['mouse_name']==mouse)], sex_diff = False, after_opto = True)
                mouse_results  =  pd.DataFrame({"Predictors": mouse_result.model.exog_names , "Coef" : mouse_result.params.values,\
                                      "SEM": mouse_result.bse.values}) #, "Sex": "M"})
                mouse_results_opto  =  pd.DataFrame({"Predictors": mouse_result_opto.model.exog_names , "Coef" : mouse_result_opto.params.values,\
                                      "SEM": mouse_result_opto.bse.values})
                mouse_results.set_index('Predictors', inplace=True)
                mouse_results_opto.set_index('Predictors', inplace=True)
                mouse_rew_predictors =  mouse_results.loc[['rchoice1_zscore','rchoice2_zscore', 'rchoice3_zscore','rchoice4_zscore','rchoice5_zscore'],['Coef']].reset_index()
                mouse_rew_predictors['mouse_name'] = mouse
                mouse_urew_predictors =  mouse_results.loc[['uchoice1_zscore','uchoice2_zscore', 'uchoice3_zscore','uchoice4_zscore','uchoice5_zscore'],['Coef']].reset_index()
                mouse_urew_predictors['mouse_name'] = mouse
                mouse_rew_opto_predictors =  mouse_results_opto.loc[['rchoice1_zscore','rchoice2_zscore', 'rchoice3_zscore','rchoice4_zscore','rchoice5_zscore'],['Coef']].reset_index()
                mouse_rew_opto_predictors['mouse_name'] = mouse
                mouse_urew_opto_predictors =  mouse_results_opto.loc[['uchoice1_zscore','uchoice2_zscore', 'uchoice3_zscore','uchoice4_zscore','uchoice5_zscore'],['Coef']].reset_index()
                mouse_urew_opto_predictors['mouse_name'] = mouse
                
                pool_rew_predictors = pool_rew_predictors.append(mouse_rew_predictors)
                pool_urew_predictors = pool_urew_predictors.append(mouse_urew_predictors)
                
                pool_rew_opto_predictors = pool_rew_opto_predictors.append(mouse_rew_opto_predictors)
                pool_urew_opto_predictors = pool_urew_opto_predictors.append(mouse_urew_opto_predictors)
                
            
            pool_rew_predictors = pool_rew_predictors.reset_index()
            pool_rew_predictors.loc[:,'index']  = pool_rew_predictors.loc[:,'index'] + 1
            pool_urew_predictors = pool_urew_predictors.reset_index()
            pool_urew_predictors.loc[:,'index']  = pool_urew_predictors.loc[:,'index'] + 1
            
            pool_rew_opto_predictors = pool_rew_opto_predictors.reset_index()
            pool_rew_opto_predictors.loc[:,'index']  = pool_rew_opto_predictors.loc[:,'index'] + 1
            pool_urew_opto_predictors = pool_urew_opto_predictors.reset_index()
            pool_urew_opto_predictors.loc[:,'index']  = pool_urew_opto_predictors.loc[:,'index'] + 1
            
            
            #Start plotting
            
            sns.lineplot(x = 'index', y = 'Coef', data=pool_rew_predictors, hue = 'mouse_name', linewidth = 1.5)    
            sns.lineplot(x = 'index', y = 'Coef', data=pool_rew_opto_predictors,hue = 'mouse_name', linewidth = 1.5, alpha =0.3) 
            sns.lineplot(x = 'index', y = 'Coef', data=pool_rew_predictors, color = 'black', ci =0, linewidth = 3)    
            sns.lineplot(x = 'index', y = 'Coef', data=pool_rew_opto_predictors, color = 'black', ci=0, linewidth = 3, alpha =0.3) 
    
            
            #sns.lineplot(x = 'index', y = 'Coef', data=pool_urew_predictors, hue = 'mouse_name')
            #sns.lineplot(x = 'index', y = 'Coef', data=pool_urew_opto_predictors,hue = 'mouse_name', alpha=0.3) 
            #sns.lineplot(x = 'index', y = 'Coef', data=pool_urew_predictors, color = 'black', ci =0)    
            #sns.lineplot(x = 'index', y = 'Coef', data=pool_urew_opto_predictors, color = 'black', alpha=0.3, ci=0) 
            
            

            black_patch  =  mpatches.Patch(color='black', label='Laser off', alpha =1 )
            black_light  =  mpatches.Patch(color='black', label='Laser on', alpha =0.3 )
            thick  =  plt.Line2D([0], [0], color='black', lw=4, label='average', ls = '-')
            light  = plt.Line2D([0], [0], color='black', lw=1.5, label='individual', ls = '-')
            plt.legend(handles=[black_patch, black_light,thick, light], loc = 'upper right')

            
            ax[v,c].set_title(virus +' '+ hem)
            ax[v,c].set_ylabel('Trials back')
            ax[v,c].set_xlabel('Regressor Coefficient')
            ax[v,c].set_xticks([1,2,3,4,5])
            ax[v,c].set_xlabel('Trials back')
            
    return figure

def opto_laser_glm(psy_df_global):
    viruses =  psy_df_global['virus'].unique()
    figure,ax = plt.subplots(5,3, figsize=(24,80))
    for v, virus in enumerate(viruses):
        conditions  = psy_df_global.loc[(psy_df_global['virus']== virus),'hem_stim'].unique()
        for c , hem in enumerate(['B','L']): #(conditions)
            psy_df = psy_df_global.loc[(psy_df_global['virus']== virus) & (psy_df_global['hem_stim']== hem)]
            plt.sca(ax[v,c])
            
            mouse_result, mouse_r2  = glm_logit_opto(psy_df, sex_diff = False)
                
            mouse_result  =  pd.DataFrame({"Predictors": mouse_result.model.exog_names , "Coef" : mouse_result.params.values,\
                              "SEM": mouse_result.bse.values, "Significant": mouse_result.pvalues < 0.05/len(mouse_result.model.exog_names)})
    
            #Drop current evidence
            mouse_result = mouse_result.iloc[2:]
                
            #Plotting
                
            ax[v,c]  = sns.barplot(x = 'Predictors', y = 'Coef', data=mouse_result, yerr= mouse_result['SEM'])    
            ax[v,c].set_xticklabels(mouse_result['Predictors'], rotation=-90)
            ax[v,c].set_ylabel('coef')
            ax[v,c].axhline(y=0, linestyle='--', color='black', linewidth=2)
            ax[v,c].set_title(virus +' '+ hem)
            
            
            #Have to cahnge glm function so that it has the optopn to run normal glm only on opto.npy trials
            for i,mouse in enumerate(psy_df['mouse_name'].unique()):
                mouse_result, mouse_r2  = glm_logit_opto(psy_df.loc[(psy_df['mouse_name']==mouse)], sex_diff = False)
                
                mouse_result  =  pd.DataFrame({"Predictors": mouse_result.model.exog_names , "Coef" : mouse_result.params.values,\
                              "SEM": mouse_result.bse.values, "Significant": mouse_result.pvalues < 0.05/len(mouse_result.model.exog_names)})
    
                #Drop current evidence
                mouse_result = mouse_result.iloc[2:]    
                
                #Plotting
                plt.sca(ax[v+i+2,c])
                ax[v+i+2,c]  = sns.barplot(x = 'Predictors', y = 'Coef', data=mouse_result, yerr= mouse_result['SEM'])    
                ax[v+i+2,c].set_xticklabels(mouse_result['Predictors'], rotation=-90)
                ax[v+i+2,c].set_ylabel('coef')
                ax[v+i+2,c].axhline(y=0, linestyle='--', color='black', linewidth=2)
                ax[v+i+2,c].set_title(virus +' '+mouse +' '+ hem)
                
                
    return figure

def plot_glm(psy_df, result, r2):
    """
    INPUT:  psy_df, result of regression, r2 of regressions
    OUTPUT: Dataframe with data for plotting  + significance
    """
    
    results  =  pd.DataFrame({"Predictors": result.model.exog_names , "Coef" : result.params.values,\
                              "SEM": result.bse.values, "Significant": result.pvalues < 0.05/len(result.model.exog_names)})
    
    
    #Drop current evidence
    results = results.iloc[2:]    
    
    #Plotting
    fig, ax = plt.subplots(figsize=(12, 9))
    ax  = sns.barplot(x = 'Predictors', y = 'Coef', data=results, yerr= results['SEM'])    
    ax.set_xticklabels(results['Predictors'], rotation=-90)
    ax.set_ylabel('coef')
    ax.axhline(y=0, linestyle='--', color='black', linewidth=2)
    fig.suptitle ('GLM Biased Blocks')
    
    return results

def trials_after_opto (psy_df):
   
    #Make right choices 1
    
    psy_df['choice'] = psy_df['choice'] * -1
    psy_df['previous_choice'] = psy_df['previous_choice'] * -1
    
    
    #psy_df.loc[(psy_df['s.probabilityLeft']==0.2)]

    #Start slicing
    psy_df['right_choice'] = 0
    psy_df.loc[(psy_df['choice'] == 1), 'right_choice'] = 1
    
    #Chr2
    trials_right_chr2 = psy_df.loc[(psy_df['previous_choice']==1) & (psy_df['virus']=='chr2')]
    trials_left_chr2 = psy_df.loc[(psy_df['previous_choice']==-1) & (psy_df['virus']=='chr2')]
    #NphR
    trials_right_nphr = psy_df.loc[(psy_df['previous_choice']==1) & (psy_df['virus']=='nphr')]
    trials_left_nphr = psy_df.loc[(psy_df['previous_choice']==-1) & (psy_df['virus']=='nphr')]
    
    figure, ax  = plt.subplots(1,2, figsize=(20,10))
    plt.sca(ax[0])
    sns.lineplot(x='signed_contrasts', y='right_choice', err_style="bars", hue = 'after_opto', \
                                 linewidth=2, linestyle='None', mew=0.5,marker='.',
                                 ci=68, data= trials_right_chr2, palette = "GnBu_d")

    sns.lineplot(x='signed_contrasts', y='right_choice', err_style="bars", hue = 'after_opto', \
                                 linewidth=2, linestyle='None', mew=0.5,marker='.',
                                 ci=68, data= trials_left_chr2, palette = "BuGn_r")
    ax[0].set_xlim(-0.25,0.25)
    green_patch = mpatches.Patch(color='green', label='Previous Left')
    blue_patch  =  mpatches.Patch(color='blue', label='Previous Right')
    green_patch_stim = mpatches.Patch(color='green',alpha=0.2, label='Previous Left + stim')
    blue_patch_stim  =  mpatches.Patch(color='blue',alpha=0.2, label='Previous Right + stim')
    plt.legend(handles=[green_patch, blue_patch, green_patch_stim, blue_patch_stim], loc = 'lower right')
    plt.sca(ax[1])    
    sns.lineplot(x='signed_contrasts', y='right_choice', err_style="bars", hue = 'after_opto', \
                                 linewidth=2, linestyle='None', mew=0.5,marker='.',
                                 ci=68, data= trials_right_nphr, palette = "GnBu_d")

    sns.lineplot(x='signed_contrasts', y='right_choice', err_style="bars", hue = 'after_opto', \
                                 linewidth=2, linestyle='None', mew=0.5,marker='.',
                                 ci=68, data= trials_left_nphr, palette = "BuGn_r")
    
    
    ax[1].set_xlim([-0.25,0.25])
    green_patch = mpatches.Patch(color='green', label='Previous Left')
    blue_patch  =  mpatches.Patch(color='blue', label='Previous Right')
    green_patch_stim = mpatches.Patch(color='green',alpha=0.2, label='Previous Left + stim')
    blue_patch_stim  =  mpatches.Patch(color='blue',alpha=0.2, label='Previous Right + stim')
    plt.legend(handles=[green_patch, blue_patch, green_patch_stim, blue_patch_stim], loc = 'lower right')