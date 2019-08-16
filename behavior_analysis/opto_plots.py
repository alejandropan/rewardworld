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
    figure,ax = plt.subplots(10,4, figsize=(24,80))
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
    
    
