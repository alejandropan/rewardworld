from os import F_TEST
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

ses = '/Users/alexpan/Documents/2022-03-30/001'
g_t=np.load(ses+'/alf/_ibl_trials.goCue_times.npy')
f_t=np.load(ses+'/alf/_ibl_trials.feedback_times.npy')
pos=np.load(ses+'/alf/_ibl_wheel.position.npy')
pos_t=np.load(ses+'/alf/_ibl_wheel.timestamps.npy')
screen = pd.read_csv(ses+'/alf/_ibl_trials.screen_info.csv')
stims = np.nan_to_num(np.load(ses+'/alf/_ibl_trials.contrastRight.npy')) - np.nan_to_num(np.load(ses+'/alf/_ibl_trials.contrastLeft.npy'))
f = np.load(ses+'/alf/_ibl_trials.feedbackType.npy')

screen['bpod_time_drift'] =  np.nan
screen.loc[screen['stim']==1, 'bpod_time_drift'] = g_t
t_0 = screen.loc[screen['stim']==1, 'unix'].iloc[0]
t_0_bpod = screen.loc[screen['stim']==1, 'bpod_time_drift'].iloc[0]
screen['d_unix'] = screen['unix'] - t_0
screen['bpod_time_drift'] = screen['d_unix'] + t_0_bpod

def flatten_list(_2d_list):
    flat_list = []
    # Iterate through the outer list
    for element in _2d_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list

def find_nearest(array, values):
    nearests= []
    array = np.asarray(array)
    try:
        for i in values:
            nearests.append(np.abs(array - i).argmin())
    except:
        nearests=np.abs(array - values).argmin()
    return nearests

maximum_displacement = []
for t in np.arange(len(g_t)):
    try:
        timestamps = np.where((pos_t>g_t[t])&(pos_t<f_t[t]))
        pos_0 = pos[timestamps][0]
        max_d = np.max(abs(pos[timestamps]-pos_0))
        maximum_displacement.append(max_d)
    except:
        print(t)

start= 154
stop = start+4
timestamps = np.where((pos_t>g_t[start])&(pos_t<f_t[stop+1]))

plt.plot(pos_t[np.where((pos_t>g_t[start])&(pos_t<f_t[stop+1]))], pos[timestamps], color='k')
plt.scatter(g_t[start:stop], pos[find_nearest(pos_t,g_t[start:stop])], color='g')
plt.scatter(f_t[start:stop], pos[find_nearest(pos_t,f_t[start:stop])], color='r')
plt.xlabel('Seconds')
plt.ylabel('Position(rad)')

start_dots = []
finish_dots = []

for i,trial in enumerate(np.arange(start,stop)):
    p0 = pos[find_nearest(pos_t,g_t[trial])]
    if stims[trial]>0:
        dpos = pos-p0 - 0.35
    else:
        dpos = pos-p0  + 0.35
    if i==0:
        start_dots.append(dpos[find_nearest(pos_t,g_t[trial])])
        finish_dots.append(dpos[find_nearest(pos_t,f_t[trial])])
        data = dpos[find_nearest(pos_t,g_t[trial]):find_nearest(pos_t,f_t[trial])]
        data_t = pos_t[find_nearest(pos_t,g_t[trial]):find_nearest(pos_t,f_t[trial])]
    else:
        start_dots.append(dpos[find_nearest(pos_t,g_t[trial])])
        finish_dots.append(dpos[find_nearest(pos_t,f_t[trial])])
        datat = dpos[find_nearest(pos_t,g_t[trial]):find_nearest(pos_t,f_t[trial])]
        datat_t = pos_t[find_nearest(pos_t,g_t[trial]):find_nearest(pos_t,f_t[trial])]
        data = np.concatenate([data,datat])
        data_t= np.concatenate([data_t,datat_t])

screen_subset = screen.loc[(screen['bpod_time']>=g_t[start]) & (screen['bpod_time']<=f_t[stop])]



#clock drift 
plt.scatter(screen['bpod_time'], screen['bpod_time_drift'])
plt.xlabel('Interp time')
plt.ylabel('Delta time')


plt.plot(screen['bpod_time'], screen['position'], color='dodgerblue')
plt.vlines(g_t,-70,70, color='g')
plt.vlines(f_t,-70,70, color='r')
plt.ylabel('GABOR position')



plt.plot(data_t,data)
plt.plot(screen_subset['bpod_time'], screen_subset['position']*-1/100, color='blue')
plt.scatter(g_t[start:stop], start_dots, color='g')
plt.scatter(f_t[start:stop], finish_dots, color='r')
plt.xlabel('Seconds')
plt.ylabel('Position(rad)')
