#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 17:13:14 2020

@author: alex
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:15:23 2020

@author: alex


Plots demonstrating RL behavior

"""

# Figure 1 #

'''
First we show the psychometric curves post correct, post error right and left
4 psychometrics in total. No fitting
'''

psy.loc[psy['opto.npy']==1, 'opto.npy'] = 1
psy.loc[psy['opto.npy']==0, 'opto.npy'] = 0
# Make Q deltas
psy['DeltaQL'] = psy['QL'].diff()
psy['DeltaQR'] = psy['QR'].diff()
psy['Qchosen'] =  psy['QL']
psy.loc[psy['choice']==-1,'Qchosen'] = psy.loc[psy['choice']==-1,'QR']
psy['RPE'] = (((psy['feedbackType'] > 0)*1) + psy['opto.npy']) - psy['Qchosen']
psy['RPE'] = psy['RPE'].to_numpy().astype(float)
psy['RPE+1'] = psy['RPE'].shift(1) 
# Set difficulty of trials
hard = [0, 0.0625, -0.0625]
easy = [0.125, 0.25, -0.125, -0.25]
psy['difficulty'] = np.nan
psy.loc[np.isin(psy['signed_contrasts'], hard), 'difficulty'] = 'hard'
psy.loc[np.isin(psy['signed_contrasts'], easy), 'difficulty'] = 'easy'

# Set side
psy['side'] = np.nan
psy.loc[psy['signed_contrasts'] > 0, 'side'] = 'right'
psy.loc[psy['signed_contrasts'] < 0, 'side'] = 'left'
psy.loc[(psy['signed_contrasts'] == 0) & (psy['choice'] == -1) & \
        (psy['feedbackType'] == 1)  , 'side'] = 'right'
psy.loc[(psy['signed_contrasts'] == 0) & (psy['choice'] == 1) & \
        (psy['feedbackType'] == 1)  , 'side'] = 'left'


# Change choice to right==1
psy['choice'] = psy['choice']*-1

# Drop no-go
psy = psy.loc[psy['choice']!=0]

# First calculate previous choice, opto and previous outcome, previous diffficulty
blocks = ['left', 'neutral', 'right']
psy['block'] = np.nan
psy['prev_choice'] = np.nan
psy['prev_outcome'] = np.nan
psy['prev_opto'] = np.nan
psy['prev_difficulty'] = np.nan
psy['prev_stim'] = np.nan
psy['prev_side'] = np.nan
psy['next_side'] = np.nan
psy['next_outcome'] = np.nan
psy['next_difficulty'] = np.nan
psy['next_choice'] = np.nan


for name in psy['mouse_name'].unique():
    psy.loc[psy['mouse_name']==name, 'prev_choice'] = \
       psy.loc[psy['mouse_name']==name, 'choice'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_outcome'] = \
       psy.loc[psy['mouse_name']==name, 'feedbackType'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_opto'] = \
       psy.loc[psy['mouse_name']==name, 'opto.npy'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_difficulty'] = \
       psy.loc[psy['mouse_name']==name, 'difficulty'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_stim'] = \
       psy.loc[psy['mouse_name']==name, 'signed_contrasts'].shift(1)
    psy.loc[psy['mouse_name']==name, 'prev_side'] = \
       psy.loc[psy['mouse_name']==name, 'side'].shift(1)
    # Calculate future as well
    psy.loc[psy['mouse_name']==name, 'next_side'] = \
       psy.loc[psy['mouse_name']==name, 'side'].shift(-1)
    psy.loc[psy['mouse_name']==name, 'next_difficulty'] = \
       psy.loc[psy['mouse_name']==name, 'difficulty'].shift(-1)
    psy.loc[psy['mouse_name']==name, 'next_outcome'] = \
       psy.loc[psy['mouse_name']==name, 'feedbackType'].shift(-1)
    psy.loc[psy['mouse_name']==name, 'next_choice'] = \
       psy.loc[psy['mouse_name']==name, 'choice'].shift(-1)
       
# Make prev_opto into boolean
psy.loc[psy['prev_opto']==1, 'prev_opto'] = True
psy.loc[psy['prev_opto']==0, 'prev_opto'] = False
      
# Cahnge prev choice and outcome to string
psy.loc[psy['prev_outcome'] == -1, 'prev_outcome'] = 'Error'
psy.loc[psy['prev_outcome'] == 1, 'prev_outcome'] = 'Reward'
psy.loc[psy['prev_choice'] == 1, 'prev_choice'] = 'Right'
psy.loc[psy['prev_choice'] == -1, 'prev_choice'] = 'Left'
psy.loc[psy['next_choice'] == 1, 'next_choice'] = 'Right'
psy.loc[psy['next_choice'] == -1, 'next_choice'] = 'Left'
psy.loc[psy['next_outcome'] == -1, 'next_outcome'] = 'Error'
psy.loc[psy['next_outcome'] == 1, 'next_outcome'] = 'Reward'

# Change choice to 0 to 1 range
psy['choice'] = (psy['choice']>0)*1

#Assign blocks
blocks = ['left', 'neutral', 'right']
psy.loc[psy['opto_probability_left']==-1, 'block']  = 'neutral'
psy.loc[psy['opto_probability_left']== 1, 'block']  = 'left'
psy.loc[psy['opto_probability_left']==0, 'block']  = 'right'
       

# Plotting
psy['ITIQRR+1'] = psy['ITIQRR'].shift(1)
psy['ITIQLL+1'] = psy['ITIQLL'].shift(1)
psy['ITIQLR+1'] = psy['ITIQLR'].shift(1)
psy['ITIQRL+1'] = psy['ITIQRL'].shift(1)
psy['ITI+1QRQL'] = psy['ITIQRL'].shift(1)

# Plot by block based on previous reward

pal ={"Right":"r","Left":"b"}


fig, ax  = plt.subplots(9,4, figsize=(15, 25))
for i, var in enumerate(['RPE+1', 'ITIQRR+1', 'ITIQLL+1',
                         'ITIQLR+1', 'ITIQRL+1', 'QRR', 'QLL', 'QRL', 'QLR']):
    # Based on previous outcome current 
    for j, mouse in enumerate(mice):
        # Plot by previous reward
        plt.sca(ax[i,j])
        mouse_d = psy.loc[psy['mouse_name']==mouse]
        if (i==0 & j==0):
            sns.lineplot(data = mouse_d, \
                             x='signed_contrasts', 
                             y= var, hue ='prev_choice', style='prev_outcome', 
                             style_order=['Reward', 'Error'], ci=68,
                             legend='brief', palette=pal)
        else:
            sns.lineplot(data = mouse_d, \
                             x='signed_contrasts', 
                             y= var, hue ='prev_choice', style='prev_outcome', 
                             style_order=['Reward', 'Error'], ci=68,
                             palette=pal, legend=False)
        plt.title(mouse)
plt.tight_layout()
        

# Divided by difficulty
easy_t = psy.loc[(psy['prev_difficulty']=='easy') ]
hard_t = psy.loc[(psy['prev_difficulty']=='hard') ]
    
fig, ax  = plt.subplots(9,4, figsize=(15, 25))
for i, var in enumerate(['RPE+1', 'ITIQRR+1', 'ITIQLL+1',
                         'ITIQLR+1', 'ITIQRL+1', 'QRR', 'QLL', 'QRL', 'QLR']):
    # Based on previous outcome current 
    for j, mouse in enumerate(mice):
        # Plot by previous reward
        plt.sca(ax[i,j])
        mouse_d = easy_t.loc[easy_t['mouse_name']==mouse]
        if (i==0 & j==0):
            sns.lineplot(data = mouse_d, \
                             x='signed_contrasts', 
                             y= var, hue ='prev_choice', style='prev_outcome', 
                             style_order=['Reward', 'Error'], ci=68,
                             legend='brief', palette=pal)
        else:
            sns.lineplot(data = mouse_d, \
                             x='signed_contrasts', 
                             y= var, hue ='prev_choice', style='prev_outcome', 
                             style_order=['Reward', 'Error'], ci=68,
                             legend=False, palette=pal)
        plt.title(mouse)
plt.tight_layout()        
        
        
fig, ax  = plt.subplots(9,4, figsize=(15, 25))
for i, var in enumerate(['RPE+1', 'ITIQRR+1', 'ITIQLL+1',
                         'ITIQLR+1', 'ITIQRL+1', 'QRR', 'QLL', 'QRL', 'QLR']):
    # Based on previous outcome current 
    for j, mouse in enumerate(mice):
        # Plot by previous reward
        plt.sca(ax[i,j])
        mouse_d = hard_t.loc[hard_t['mouse_name']==mouse]
        if (i==0 & j==0):
            sns.lineplot(data = mouse_d, \
                             x='signed_contrasts', 
                             y= var, hue ='prev_choice', style='prev_outcome', 
                             style_order=['Reward', 'Error'], ci=68,
                             legend='brief', palette=pal)
        else:
            sns.lineplot(data = mouse_d, \
                             x='signed_contrasts', 
                             y= var, hue ='prev_choice', style='prev_outcome', 
                             style_order=['Reward', 'Error'], ci=68,
                             legend=False, palette=pal)
        plt.title(mouse)
plt.tight_layout()  
    
    
    

#  Divide by choice difficulty #
   

fig, ax  = plt.subplots(9,4, figsize=(15, 25))
for i, var in enumerate(['RPE+1', 'ITIQRR+1', 'ITIQLL+1',
                         'ITIQLR+1', 'ITIQRL+1', 'QRR', 'QLL', 'QRL', 'QLR']):
    # Based on previous outcome current 
    for j, mouse in enumerate(mice):
        # Plot by previous reward
        plt.sca(ax[i,j])
        mouse_d = psy.loc[(psy['mouse_name']==mouse) & (psy['prev_outcome']== 'Reward')]
        if (i==0 & j==0):
            sns.lineplot(data = mouse_d, \
                             x='signed_contrasts', 
                             y= var, hue ='prev_choice', style='prev_difficulty', 
                             style_order=['easy', 'hard'], ci=68,
                             legend='brief', palette=pal)
        else:
            sns.lineplot(data = mouse_d, \
                             x='signed_contrasts', 
                             y= var, hue ='prev_choice', style='prev_difficulty', 
                             style_order=['easy', 'hard'], ci=68,
                             legend=False, palette=pal)
        plt.title(mouse)
plt.tight_layout()
            
# Divided by opto, block and previous outcome 

for var in ['RPE+1', 'ITIQRR+1', 'ITIQLL+1',
                         'ITIQLR+1', 'ITIQRL+1', 'QRR', 'QLL', 'QRL', 'QLR']:
    
    fig, ax  = plt.subplots(3,2, figsize=(5, 6))
    fig.suptitle(var, fontsize=16)
    for i, virus in enumerate(psy['virus'].unique()):
        # Based on previous outcome current 
        psy_v = psy.loc[psy['virus']==virus]
        for j, block in enumerate(blocks):
            
            # Plot by previous reward
            plt.sca(ax[j,i])
            mouse_d = psy_v.loc[(psy_v['block']==block)]
            if (i==0 and j==1):
                sns.lineplot(data = mouse_d, \
                                 x='signed_contrasts', 
                                 y= var, hue ='prev_opto', style='prev_outcome', 
                                 style_order=['Reward', 'Error'],hue_order=[True, False], ci=68,
                                 legend='brief')
            else:
                sns.lineplot(data = mouse_d, \
                                 x='signed_contrasts', 
                                 y= var, hue ='prev_opto', style='prev_outcome', 
                                 style_order=['Reward', 'Error'],hue_order=[True, False], ci=68,
                                 legend=False)
            plt.title(block + ' ' + virus)
            plt.tight_layout()
            

# Is there something special about easy trials?
sns.barplot(data = psy, x='virus', y = 'ITIQRR+1', hue='difficulty')
sns.barplot(data = psy, x='mouse_name', y = (psy['prev_difficulty']=='easy')*1, hue='difficulty')
plt.ylabel('Fraction of trials preceeded by easy trials')
sns.barplot(data = psy, x='virus', y = (psy['prev_difficulty']=='easy')*1, hue='difficulty')
plt.ylabel('Fraction of trials preceeded by easy trials')
sns.barplot(data = psy, x='block', y = (psy['prev_difficulty']=='easy')*1, hue='difficulty')
plt.ylabel('Fraction of trials preceeded by easy trials')
sns.barplot(data = psy, x='prev_outcome', y = (psy['difficulty']=='easy')*1, hue='prev_difficulty')
plt.ylabel('Fraction of easy trials')

#What is special about easy errors?

psy_error = psy.loc[psy['feedbackType']==-1]

sns.barplot(data = psy_error, x='virus', y = 'ITIQRR+1', hue='difficulty')


sns.barplot(data = psy_error, x='block', y = (psy_error['prev_difficulty']=='easy')*1, hue='difficulty')
plt.ylabel('Fraction of trials preceeded by easy trials')

sns.barplot(data = psy_error, x='difficulty', \
            y = 1*(psy_error['prev_outcome']=='Error'))
plt.ylabel('Fraction of errors preceeded by Error')

sns.barplot(data = psy_error, x='difficulty', \
            y = 1*(psy_error['prev_opto']==True))
plt.ylabel('Fraction of error  preceeded by opto')


sns.barplot(data = psy_error, x='difficulty', \
            y = 1*(psy_error['prev_opto']==True))
plt.ylabel('Fraction of error  preceeded by opto')


fig, ax = plt.subplots(2, figsize=(2, 5))
for i, virus in enumerate(psy['virus'].unique()):
        # Based on previous outcome current 
    psy_error_virus = psy_error.loc[psy_error['virus']==virus] 
    plt.sca(ax[i])
    sns.barplot(data = psy_error_virus, x='difficulty', \
                y = 1*(psy_error_virus['prev_opto']==True))
    plt.ylabel(' Err preceeded by Las')
    plt.title(virus)
plt.tight_layout()


fig, ax = plt.subplots(2, figsize=(2, 5))
for i, virus in enumerate(psy['virus'].unique()):
        # Based on previous outcome current 
    psy_error_virus = psy_error.loc[psy_error['virus']==virus] 
    plt.sca(ax[i])
    sns.barplot(data = psy_error_virus, x='difficulty', \
                y = 1*(psy_error_virus['opto.npy']==True))
    plt.ylabel(' Err with laser Las')
    plt.title(virus)
plt.tight_layout()



psy_prev_error = psy
fig, ax  = plt.subplots(3,2, figsize=(5, 6))
for i, virus in enumerate(psy['virus'].unique()):
        # Based on previous outcome current 
        psy_v = psy_prev_error.loc[(psy_prev_error['virus']==virus) & (psy_prev_error['prev_choice']=='Right')]
        for j, block in enumerate(blocks):
            # Pot by previous reward
            plt.sca(ax[j,i])
            mouse_d = psy_v.loc[(psy_v['block']==block)]
            sns.barplot(data = mouse_d, x='prev_outcome', y = 'ITI+1QRQL')
            plt.title(block + ' ' + virus)
plt.tight_layout()
