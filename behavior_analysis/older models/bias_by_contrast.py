#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 15:12:22 2020

@author: alex
"""
psy = pd.read_pickle('all_behav.pkl')
psy = psy.loc[((psy['ses']>'2020-01-13') & (psy['mouse_name'] == 'dop_4')) | 
             (psy['ses']>'2020-03-13')]

psy['choice'] = psy['choice'] *-1
psy['choice'] = (psy['choice']>0)*1

pal ={'non_opto':"k",'R':"g",'L':"b"}


all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.    ,  0.0625, 0.125 , 0.25  ])

average_by_contrast = psy.groupby(['signed_contrasts','opto_block','virus']).mean()['choice'].reset_index()
sem_by_contrast = psy.groupby(['signed_contrasts','opto_block','virus']).sem()['choice'].reset_index()

# Remove non_opto

average_by_contrast = average_by_contrast.loc[average_by_contrast['opto_block']
                                              != 'non_opto']

average_by_contrast_chr2 = average_by_contrast.loc[average_by_contrast['virus']
                                                   =='chr2']

average_by_contrast_nphr = average_by_contrast.loc[average_by_contrast['virus']
                                                   =='nphr']

nphr =  average_by_contrast_nphr.loc[average_by_contrast_nphr['opto_block'] == 'R','choice'].to_numpy() - \
average_by_contrast_nphr.loc[average_by_contrast_nphr['opto_block'] == 'L','choice'].to_numpy()


chr2 =  average_by_contrast_chr2.loc[average_by_contrast_chr2['opto_block'] == 'R','choice'].to_numpy() - \
average_by_contrast_chr2.loc[average_by_contrast_chr2['opto_block'] == 'L','choice'].to_numpy()


fig, ax = plt.subplots(2)
plt.sca(ax[0])
plt.plot(all_contrasts, chr2)
plt.xlabel('Signed Contrast')
plt.ylabel('% Choice BlockR - BlockL')
plt.title('Chr2')
plt.ylim(0,0.3)

plt.sca(ax[1])
plt.plot(all_contrasts, nphr)
plt.xlabel('Signed Contrast')
plt.ylim(-0.3, 0)
plt.ylabel('% Choice BlockR - BlockL')
plt.title('NpHR')
plt.tight_layout()



psy['repeat'] = 0
for mouse in psy['mouse_name'].unique():
    psy.loc[psy['mouse_name'] == mouse,'repeat'] = \
        (psy.loc[psy['mouse_name'] == mouse,'choice'] == \
            psy.loc[psy['mouse_name'] == mouse,'choice'].shift(periods=1))*1

repeated_byblock_mouse = psy.groupby(['opto_block','mouse_name']).mean()['repeat'].reset_index()
sns.barplot(data = repeated_byblock_mouse, x = 'mouse_name', hue = 'opto_block', y= 'repeat', palette = pal)
plt.ylabel('% Choices (choicet ==ch ')
