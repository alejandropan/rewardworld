#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 16:33:10 2020

@author: alex
"""


psy_1 = psy.copy()
psy_1['choice'] = modelled_data['choices_standard'].to_numpy()
psy_1['feedbackType'] = modelled_data['rewards'].to_numpy()
psy_1['opto.npy']  = modelled_data['laser'].to_numpy()




# Set difficulty of trials
hard = [0, 0.0625, -0.0625]
easy = [0.125, 0.25, -0.125, -0.25]
psy['difficulty'] = np.nan
psy.loc[np.isin(psy['signed_contrasts'], hard), 'difficulty'] = 'hard'
psy.loc[np.isin(psy['signed_contrasts'], easy), 'difficulty'] = 'easy'
modelled_data.loc[np.isin(modelled_data['signed_contrast'], hard), 'difficulty'] = 'hard'
modelled_data.loc[np.isin(modelled_data['signed_contrast'], easy), 'difficulty'] = 'easy'

# Set side
psy['side'] = np.nan
psy.loc[psy['signed_contrasts'] > 0, 'side'] = 'right'
psy.loc[psy['signed_contrasts'] < 0, 'side'] = 'left'
psy.loc[(psy['signed_contrasts'] == 0) & (psy['choice'] == 1) & \
        (psy['feedbackType'] == 1)  , 'side'] = 'right'
psy.loc[(psy['signed_contrasts'] == 0) & (psy['choice'] == 0) & \
        (psy['feedbackType'] == 1)  , 'side'] = 'left'
    
modelled_data['side'] = np.nan
modelled_data.loc[modelled_data['signed_contrast'] > 0, 'side'] = 'right'
modelled_data.loc[modelled_data['signed_contrast'] < 0, 'side'] = 'left'
modelled_data.loc[(modelled_data['signed_contrast'] == 0) & (modelled_data['choices_standard'] == 1) & \
        (modelled_data['rewards'] == 1)  , 'side'] = 'right'
modelled_data.loc[(modelled_data['signed_contrast'] == 0) & (modelled_data['choices_standard'] == 0) & \
        (modelled_data['rewards'] == 1)  , 'side'] = 'left'

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
    # Now for modelled data
    modelled_data.loc[modelled_data['mouse_name']==name, 'prev_choice'] = \
       modelled_data.loc[modelled_data['mouse_name']==name, 'choices_standard'].shift(1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'prev_outcome'] = \
       modelled_data.loc[modelled_data['mouse_name']==name, 'rewards'].shift(1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'prev_opto'] = \
       modelled_data.loc[modelled_data['mouse_name']==name, 'model_laser'].shift(1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'prev_difficulty'] = \
       modelled_data.loc[modelled_data['mouse_name']==name, 'difficulty'].shift(1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'prev_stim'] = \
       modelled_data.loc[modelled_data['mouse_name']==name, 'signed_contrast'].shift(1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'prev_side'] = \
        modelled_data.loc[modelled_data['mouse_name']==name, 'side'].shift(1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'next_side'] = \
        modelled_data.loc[modelled_data['mouse_name']==name, 'side'].shift(-1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'next_difficulty'] = \
       modelled_data.loc[modelled_data['mouse_name']==name, 'difficulty'].shift(-1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'next_outcome'] = \
       modelled_data.loc[modelled_data['mouse_name']==name, 'rewards'].shift(-1)
    modelled_data.loc[modelled_data['mouse_name']==name, 'next_choice'] = \
       modelled_data.loc[modelled_data['mouse_name']==name, 'choices_standard'].shift(-1)
       
# Make prev_opto into boolean
psy.loc[psy['prev_opto']==1, 'prev_opto'] = True
psy.loc[psy['prev_opto']==0, 'prev_opto'] = False
      
# Cahnge prev choice and outcome to string
psy.loc[psy['prev_outcome'] == 0, 'prev_outcome'] = 'Error'
psy.loc[psy['prev_outcome'] == 1, 'prev_outcome'] = 'Reward'
psy.loc[psy['prev_choice'] == 1, 'prev_choice'] = 'Right'
psy.loc[psy['prev_choice'] == 0, 'prev_choice'] = 'Left'
psy.loc[psy['next_choice'] == 1, 'next_choice'] = 'Right'
psy.loc[psy['next_choice'] == 0, 'next_choice'] = 'Left'
psy.loc[psy['next_outcome'] == 0, 'next_outcome'] = 'Error'
psy.loc[psy['next_outcome'] == 1, 'next_outcome'] = 'Reward'

modelled_data.loc[modelled_data['prev_outcome'] == 0, 'prev_outcome'] = 'Error'
modelled_data.loc[modelled_data['prev_outcome'] == 1, 'prev_outcome'] = 'Reward'
modelled_data.loc[modelled_data['prev_choice'] == 1, 'prev_choice'] = 'Right'
modelled_data.loc[modelled_data['prev_choice'] == 0, 'prev_choice'] = 'Left'
modelled_data.loc[modelled_data['next_choice'] == 1, 'next_choice'] = 'Right'
modelled_data.loc[modelled_data['next_choice'] == 0, 'next_choice'] = 'Left'
modelled_data.loc[modelled_data['next_outcome'] == 0, 'next_outcome'] = 'Error'
modelled_data.loc[modelled_data['next_outcome'] == 1, 'next_outcome'] = 'Reward'



#Assign blocks
blocks = ['left', 'neutral', 'right']
psy.loc[psy['opto_probability_left']==-1, 'block']  = 'neutral'
psy.loc[psy['opto_probability_left']== 1, 'block']  = 'left'
psy.loc[psy['opto_probability_left']==0, 'block']  = 'right'
modelled_data.loc[modelled_data['laser_side']==-1, 'block']  = 'neutral'
modelled_data.loc[modelled_data['laser_side']== 1, 'block']  = 'left'
modelled_data.loc[modelled_data['laser_side']==0, 'block']  = 'right'
       

# Plot by block based on previous reward

pal ={"Right":"r","Left":"b"}


# Plot by previous reward
fig, ax =  plt.subplots(2, figsize=(5,10))
for i, v in enumerate (psy['virus'].unique()):
    plt.sca(ax[i])
    sns.lineplot(data = psy.loc[(psy['virus']==v)], \
                 x='signed_contrasts', 
                 y='choice', hue ='prev_choice', style='prev_outcome', ci=68,
                 legend='brief', palette=pal)
    plt.title(v)
plt.tight_layout()

# Plot by opto

fig, ax =  plt.subplots(2, figsize=(5,10))
for i, v in enumerate(psy['virus'].unique()):
    plt.sca(ax[i])
    sns.lineplot(data = psy.loc[(psy['virus']== v)], \
                     x='signed_contrasts', 
                     y='choice', hue ='prev_choice', style = 'prev_opto', ci = 68,
                     legend='brief', palette=pal)
    plt.title(v)
plt.tight_layout()



pal ={"neutral":'k',"right":"r","left":"b"}
pal1 ={0:'b',1:"r",-1:"k"}
pal2 ={"Right":"r","Left":"b"}
pal3 ={"hard":"r","easy":"b"}





psy.loc[np.isin(psy['mouse_name'], mice)]
#Plot simulation vs data
for mouse in mice:
    sns.set()
    sns.set_style('white')
    fig, ax = plt.subplots(2,2, figsize= [5,5])
    plt.sca(ax[0,0])
    sns.lineplot(data = psy.loc[(psy['virus']=='chr2') & (psy['mouse_name']==mouse)], x='signed_contrasts', hue =
                 'block', y='choice', palette = pal)
    plt.title('Real Data - ChR2')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,0])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='chr2') & (modelled_data['mouse_name']==mouse)], x='signed_contrast', y='choices_standard',
                 hue='laser_side', palette = pal1, legend=False)
    plt.title('Model Data - ChR2')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    plt.sca(ax[0,1])
    sns.lineplot(data = psy.loc[(psy['virus']=='nphr') & (psy['mouse_name']==mouse)], x='signed_contrasts', hue =
                 'block', y='choice', palette = pal, legend=False)
    plt.title('Real Data - NpHR')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,1])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='nphr') & (modelled_data['mouse_name']==mouse)], x='signed_contrast', y='choices_standard',
                 hue='laser_side', palette = pal1, legend=False)
    plt.title('Model Data - NpHR')
    plt.ylabel('% Right choice')
    plt.tight_layout()




for mouse in mice:
    sns.set()
    sns.set_style('white')
    
    fig, ax = plt.subplots(2,2, figsize= [5,5])
    plt.sca(ax[0,0])
    sns.lineplot(data = psy.loc[(psy['virus']=='chr2') & (psy['mouse_name']==mouse)], 
                 x='signed_contrasts',y='choice', hue ='prev_choice', style='prev_outcome', palette=pal2, 
                  style_order=['Reward', 'Error'])
    plt.title('Real Data - ChR2')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,0])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='chr2') & (modelled_data['mouse_name']==mouse)],
                 x='signed_contrast', y='choices_standard', hue ='prev_choice', style='prev_outcome', palette=pal2,legend=False, style_order=['Reward', 'Error'])
    plt.title('Real Data - ChR2')
    plt.title('Model Data - ChR2')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    plt.sca(ax[0,1])
    sns.lineplot(data = psy.loc[(psy['virus']=='nphr') & (psy['mouse_name']==mouse)], 
                 x='signed_contrasts',y='choice', hue ='prev_choice', style='prev_outcome',legend=False, palette=pal2,style_order=['Reward', 'Error'])
    plt.title('Real Data - NpHR')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,1])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='nphr') & (modelled_data['mouse_name']==mouse)],
                 x='signed_contrast', y='choices_standard', hue ='prev_choice', style='prev_outcome', palette=pal2,legend=False, style_order=['Reward', 'Error'])
    plt.title('Model Data - NpHR')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    

for mouse in mice:
    sns.set()
    sns.set_style('white')
    
    fig, ax = plt.subplots(2,2, figsize= [5,5])
    plt.sca(ax[0,0])
    sns.lineplot(data = psy.loc[(psy['virus']=='chr2') & (psy['mouse_name']==mouse) & (psy['prev_difficulty']=='easy')], 
                 x='signed_contrasts',y='choice', hue ='prev_choice', style='prev_outcome', palette=pal2, 
                  style_order=['Reward', 'Error'],legend=False)
    plt.title('Real Data - ChR2')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,0])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='chr2') & (modelled_data['mouse_name']==mouse) & (modelled_data['prev_difficulty']=='easy')],
                 x='signed_contrast', y='choices_standard', hue ='prev_choice', style='prev_outcome', palette=pal2,legend=False, style_order=['Reward', 'Error'])
    plt.title('Real Data - ChR2')
    plt.title('Model Data - ChR2')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    plt.sca(ax[0,1])
    sns.lineplot(data = psy.loc[(psy['virus']=='nphr') & (psy['mouse_name']==mouse)& (psy['prev_difficulty']=='easy')], 
                 x='signed_contrasts',y='choice', hue ='prev_choice', style='prev_outcome',legend=False, palette=pal2,style_order=['Reward', 'Error'])
    plt.title('Real Data - NpHR')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,1])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='nphr') & (modelled_data['mouse_name']==mouse) & (modelled_data['prev_difficulty']=='easy')],
                 x='signed_contrast', y='choices_standard', hue ='prev_choice', style='prev_outcome', palette=pal2,legend=False, style_order=['Reward', 'Error'])
    plt.title('Model Data - NpHR')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    
for mouse in mice:
    sns.set()
    sns.set_style('white')
    
    fig, ax = plt.subplots(2,2, figsize= [5,5])
    plt.sca(ax[0,0])
    sns.lineplot(data = psy.loc[(psy['virus']=='chr2') & (psy['mouse_name']==mouse) & (psy['prev_difficulty']=='hard')], 
                 x='signed_contrasts',y='choice', hue ='prev_choice', style='prev_outcome', palette=pal2, 
                  style_order=['Reward', 'Error'], legend=False)
    plt.title('Real Data - ChR2')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,0])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='chr2') & (modelled_data['mouse_name']==mouse) & (modelled_data['prev_difficulty']=='hard')],
                 x='signed_contrast', y='choices_standard', hue ='prev_choice', style='prev_outcome', palette=pal2,legend=False, style_order=['Reward', 'Error'])
    plt.title('Real Data - ChR2')
    plt.title('Model Data - ChR2')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    plt.sca(ax[0,1])
    sns.lineplot(data = psy.loc[(psy['virus']=='nphr') & (psy['mouse_name']==mouse)& (psy['prev_difficulty']=='hard')], 
                 x='signed_contrasts',y='choice', hue ='prev_choice', style='prev_outcome',legend=False, palette=pal2,style_order=['Reward', 'Error'])
    plt.title('Real Data - NpHR')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,1])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='nphr') & (modelled_data['mouse_name']==mouse) & (modelled_data['prev_difficulty']=='hard')],
                 x='signed_contrast', y='choices_standard', hue ='prev_choice', style='prev_outcome', palette=pal2,legend=False, style_order=['Reward', 'Error'])
    plt.title('Model Data - NpHR')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    



for mouse in mice:
    sns.set()
    sns.set_style('white')
    
    fig, ax = plt.subplots(2,2, figsize= [5,5])
    plt.sca(ax[0,0])
    sns.lineplot(data = psy.loc[(psy['virus']=='chr2') & (psy['mouse_name']==mouse) & (psy['prev_outcome']=='Reward')], 
                 x='signed_contrasts',y='choice', hue ='prev_choice', style='prev_difficulty', style_order=['easy', 'hard'], palette=pal2,
                 )
    plt.title('Real Data - ChR2')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,0])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='chr2') & 
                                          (modelled_data['mouse_name']==mouse) & 
                                          (modelled_data['prev_outcome']=='Reward')],
                 x='signed_contrast', y='choices_standard', hue ='prev_choice', style='prev_difficulty', legend=False,palette=pal2,  style_order=['easy', 'hard'])
    plt.title('Real Data - ChR2')
    plt.title('Model Data - ChR2')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    plt.sca(ax[0,1])
    sns.lineplot(data = psy.loc[(psy['virus']=='nphr') & (psy['mouse_name']==mouse) & (psy['prev_outcome']=='Reward')], 
                 x='signed_contrasts',y='choice', hue ='prev_choice', style='prev_difficulty', style_order=['easy', 'hard'],palette=pal2,
                legend=False)
    plt.title('Real Data - NpHR')
    plt.ylabel('% Right choice')
    plt.sca(ax[1,1])
    sns.lineplot(data = modelled_data.loc[(modelled_data['virus']=='nphr') & 
                                          (modelled_data['mouse_name']==mouse) & 
                                          (modelled_data['prev_outcome']=='Reward')],
                 x='signed_contrast', y='choices_standard', hue ='prev_choice', style='prev_difficulty', legend=False,  style_order=['easy', 'hard'], palette=pal2)
    plt.title('Model Data - NpHR')
    plt.ylabel('% Right choice')
    plt.tight_layout()
    
    
    
    
fig, ax =  plt.subplots(2,2, figsize=(10,10))
for i, v in enumerate(modelled_data['virus'].unique()):
    hp = pd.pivot_table( modelled_data.loc[modelled_data['virus']== v], values=['choices_standard'], columns= ['prev_side'], index=['signed_contrast', 
                                'prev_difficulty', 'prev_outcome',
                                'mouse_name']).reset_index()
    hp['bias_shift'] = (hp.choices_standard.right - hp.choices_standard.left) *100
    
    hp_future = pd.pivot_table( modelled_data.loc[modelled_data['virus']== v], values=['choices_standard'], columns= ['next_side'], index=['signed_contrast', 
                                'next_difficulty', 'next_outcome', 'mouse_name']).reset_index()
    hp_future['bias_shift'] = (hp_future.choices_standard.right - hp_future.choices_standard.left)*100
    
    for j, c in enumerate(hp['prev_outcome'].unique()):
       
        hp1= hp.loc[hp['prev_outcome']==c]
        hp2= hp_future.loc[hp_future['next_outcome']==c]
        hp3= hp1.copy()
        hp3['bias_shift_corrected'] = hp1['bias_shift'] - hp2['bias_shift']
        plt.sca(ax[i,j])
        sns.lineplot(data = hp3, x = 'signed_contrast', y = 'bias_shift_corrected',
                     hue = 'prev_difficulty', ci=68)
        plt.title(v + ' after ' + c)
        plt.ylabel('% $\Delta Right Choices')
        plt.xlabel('Trial contrast')   
plt.tight_layout()  
    
    
#Figure 2 replicatein lake figure 1f with correction and prev choice

fig, ax =  plt.subplots(2,2, figsize=(10,10))
for i, v in enumerate(modelled_data['virus'].unique()):
    hp = pd.pivot_table( modelled_data.loc[modelled_data['virus']== v], values=['choices_standard'], columns= ['prev_choice'], index=['signed_contrast', 
                                'prev_difficulty', 'prev_outcome',
                                'mouse_name']).reset_index()
    hp['bias_shift'] = (hp.choices_standard.Right - hp.choices_standard.Left) *100
    
    hp_future = pd.pivot_table( modelled_data.loc[modelled_data['virus']== v], values=['choices_standard'], columns= ['next_choice'], index=['signed_contrast', 
                                'next_difficulty', 'next_outcome', 'mouse_name']).reset_index()
    hp_future['bias_shift'] = (hp_future.choices_standard.Right - hp_future.choices_standard.Left)*100
    
    for j, c in enumerate(hp['prev_outcome'].unique()):
       
        hp1= hp.loc[hp['prev_outcome']==c]
        hp2= hp_future.loc[hp_future['next_outcome']==c]
        hp3= hp1.copy()
        hp3['bias_shift_corrected'] = hp1['bias_shift'] - hp2['bias_shift']
        plt.sca(ax[i,j])
        sns.lineplot(data = hp3, x = 'signed_contrast', y = 'bias_shift_corrected',
                     hue = 'prev_difficulty', ci=68)
        plt.title(v + ' after ' + c)
        plt.ylabel('% $\Delta Right Choices')
        plt.xlabel('Trial contrast')   
plt.tight_layout()
   

# Corrected by next reward
fig, ax =  plt.subplots(2, figsize=(5,10))
for i, v in enumerate (modelled_data['virus'].unique()):
    plt.sca(ax[i])
    data = modelled_data.loc[(modelled_data['virus']==v)]
    next_c_1 = data.loc[data['next_choice']=='Right'].groupby(['signed_contrast','next_outcome']).mean()['choices_standard'].reset_index()
    next_c_0 = data.loc[data['next_choice']=='Left'].groupby(['signed_contrast', 'next_outcome']).mean()['choices_standard'].reset_index()
    prev_c_1 = data.loc[data['prev_choice']=='Right'].groupby(['signed_contrast','prev_outcome']).mean()['choices_standard'].reset_index()
    prev_c_0 = data.loc[data['prev_choice']=='Left'].groupby(['signed_contrast','prev_outcome']).mean()['choices_standard'].reset_index()
    
    diff_p = prev_c_1.loc[prev_c_1['prev_outcome']=='Reward','choices_standard'] - prev_c_0.loc[prev_c_0['prev_outcome']=='Reward','choices_standard']    
    diff_n = next_c_1.loc[next_c_1['next_outcome']=='Reward','choices_standard'] - next_c_0.loc[next_c_0['next_outcome']=='Reward','choices_standard']    
    
    diff_p_e = prev_c_1.loc[prev_c_1['prev_outcome']=='Error','choices_standard'] - prev_c_0.loc[prev_c_0['prev_outcome']=='Error','choices_standard']    
    diff_n_e = next_c_1.loc[next_c_1['next_outcome']=='Error','choices_standard'] - next_c_0.loc[next_c_0['next_outcome']=='Error','choices_standard']    
    
    
    
    plt.plot( next_c_1['signed_contrast'].unique(), diff_p-diff_n)
    plt.plot( next_c_1['signed_contrast'].unique(), diff_p_e-diff_n_e)
    plt.xlabel('Signed Contrasts')
    plt.ylabel('Delta Righwards - i.e How much more likely am I to go right?')
    sns.despine()
    plt.plot(next_c_1['signed_contrast'].unique(), diff_p)
    plt.plot(next_c_1['signed_contrast'].unique(), diff_p_e)
    plt.xlabel('Signed Contrasts')
    plt.ylabel('Delta Righwards - i.e How much more likely am I to go right?')
    sns.despine()
    
    
sns.despine()