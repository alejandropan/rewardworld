from ephys_alf_summary import *
sessions = ephys_behavior_dataset(LIST_OF_SESSIONS_ALEX)
sessions.plot_stay()
sessions.plot_transition()
from model_comparison_accu import *
from scipy.stats import zscore
from scipy.signal import butter, filtfilt


##
qlearning_values =  load_qdata_from_file(ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects', prefix='QLearning_alphalaserdecay_')
reinforce_values =  load_qdata_from_file(ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects', prefix='REINFORCE_mixedstay_alphalaserdecay_')

model_labels = ['QLearning', 'REINFORCE']
all_values_corr_stay = np.corrcoef([qlearning_values['DQstay'].dropna().to_numpy(),reinforce_values['DQstay'].dropna().to_numpy()])
all_values_corr_water = np.corrcoef([qlearning_values['DQwater'].dropna().to_numpy(),reinforce_values['DQwater'].dropna().to_numpy()])
all_values_corr_laser = np.corrcoef([qlearning_values['DQlaser'].dropna().to_numpy(),reinforce_values['DQlaser'].dropna().to_numpy()])
all_values_corr_delta = np.corrcoef([qlearning_values['choice_prediction'].dropna().to_numpy(),reinforce_values['choice_prediction'].dropna().to_numpy()])


fig, ax = plt.subplots(1,4, sharey=True)
plt.sca(ax[0])
plot_corr_matrix(all_values_corr_delta, model_labels,vmin=0.5,vmax=1)
plt.title('Delta')
plt.sca(ax[1])
plot_corr_matrix(all_values_corr_water, model_labels,vmin=0.5,vmax=1)
plt.title('DeltaWater')
plt.sca(ax[2])
plot_corr_matrix(all_values_corr_laser, model_labels,vmin=0.5,vmax=1)
plt.title('DeltaLaser')
plt.sca(ax[3])
plot_corr_matrix(all_values_corr_stay, model_labels,vmin=0.5,vmax=1)
plt.title('DeltaStay')
plt.tight_layout()




# Winners plots
standata = make_stan_data(load_data())
model_standard = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_alphalaserdecay/output/summary.csv')
reinforce_params = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywin_mixedstay/output/summary.csv')
_, _, _, sim_standard = simulate_q_learning_model_alphalaserdecay(standata_recovery,saved_params=model_standard)
_, _, _, sim_REINFORCE = simulate_reinforce_alphalaserdecay_win_stay(standata_recovery,saved_params=reinforce_params)
original1 = stan_data_to_df(standata_recovery,standata)
original1['tb'] = original1['tb']-1
sim_standard.loc[sim_standard['tb']==0,'tb']=1 # 0 tb trials are just the first trials of every session
sim_REINFORCE.loc[sim_REINFORCE['tb']==0,'tb']=1 # 0 tb trials are just the first trials of every session
sim_standard['tb'] = sim_standard['tb']-1
sim_REINFORCE['tb'] = sim_REINFORCE['tb']-1

fig, ax = plt.subplots(1,3)
plt.sca(ax[0])
check_model_behavior(original1)
plt.title('Real data')
plt.sca(ax[1]) 
check_model_behavior(sim_standard)
plt.title('Q-learning with laser and stay')
plt.sca(ax[2])      
check_model_behavior(sim_REINFORCE)
plt.title('REINFORCE with laser and stay')

# Plot parameters
plot_params(model_standard,standata)
plot_params(reinforce_params,standata) # REMEMBER that alpha_laser = Beta laser, betalaser_base = alphalaser_base, betalaser=alphalaser

## Plot session and Q values

ses = '/Volumes/witten/Alex/Data/Subjects/dop_36/2021-07-17/002'
qdata = alf(ses)
qdata.plot_session()
qdata.plot_session_REINFORCE()

## Figure 3

import numpy as np
import matplotlib.pyplot as plt

def find_nearest(array, values):
    nearests= []
    array = np.asarray(array)
    try:
        for i in values:
            nearests.append(np.nanargmin(np.abs(array - i)))
    except:
        nearests=np.nanargmin(np.abs(array - values))
    return nearests


def smooth(timeseries, kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    data_convolved = np.convolve(timeseries, kernel, mode='same')
    return data_convolved

ses = '/Volumes/witten/Alex/Data/Subjects/DChR2_2/2022-02-09/001'
opto_block = np.where(np.load(ses+ '/alf/_ibl_trials.opto_block.npy')==1)
water_block = np.where(np.load(ses+ '/alf/_ibl_trials.opto_block.npy')==0)
DLS =  np.load(ses+ '/alf/_ibl_trials.DLS.npy')
DMS =  np.load(ses + '/alf/_ibl_trials.DMS.npy')
NAcc =  np.load(ses + '/alf/_ibl_trials.NAcc.npy')
feedback_times=  np.load(ses + '/alf/_ibl_trials.feedback_times.npy')
timestamps =  np.load(ses + '/alf/_ibl_fluo.times.npy')
correct =  np.where(np.load(ses + '/alf/_ibl_trials.feedbackType.npy')==1)

opto_trials = np.intersect1d(opto_block,correct)
water_trials= np.intersect1d(water_block,correct)
example = 421 #np.random.choice(opto_trials)
t_0 = feedback_times[example]-0.5
t_1 = feedback_times[example+11]

start=find_nearest(timestamps, t_0)
end=find_nearest(timestamps, t_1)

rate = np.nanmedian(np.diff(timestamps))
trial_range_plotted = np.arange(example,example+11)
opto_trials_example = np.intersect1d(trial_range_plotted,opto_trials)
water_trials_example = np.intersect1d(trial_range_plotted,water_trials)
b,a = butter(4, 5/(50/2),'lowpass') # butter filter: order 4, 5hz objective frequency, lowpass

_, ax = plt.subplots(3,1,sharey=True)
regions = ['NAcc','DMS', 'DLS']
for i, reg in enumerate([NAcc,DMS, DLS]):
    plt.sca(ax[i])
    plt.plot(timestamps[start:end],filtfilt(b,a,zscore(reg[start:end],nan_policy='omit')))
    plt.vlines(feedback_times[opto_trials_example], 0,np.nanmax(zscore(reg,nan_policy='omit')), color='orange')
    plt.vlines(feedback_times[water_trials_example], 0,np.nanmax(zscore(reg,nan_policy='omit')), color='dodgerblue')
    plt.vlines(feedback_times[opto_trials_example]+1, 0,np.nanmax(zscore(reg,nan_policy='omit')), color='black') #end of stim
    plt.title(regions[i])


# Summary of fiber photometry
subtract_baseline = False
sessions = ['/Volumes/witten/Alex/Data/Subjects/DChR2_2/2022-02-09/001',
'/Volumes/witten/Alex/Data/Subjects/DChR2_2/2022-02-16/001',
'/Volumes/witten/Alex/Data/Subjects/DChR2_2/2022-02-22/001',
'/Volumes/witten/Alex/Data/Subjects/DChR2_2/2022-02-26/001',
'/Volumes/witten/Alex/Data/Subjects/DChR2_2/2022-02-28/001'
]

correct_psths_water_DMS=[]
correct_psths_laser_DMS=[]
incorrect_psths_water_DMS=[]
incorrect_psths_laser_DMS=[]

correct_psths_water_DLS=[]
correct_psths_laser_DLS=[]
incorrect_psths_water_DLS=[]
incorrect_psths_laser_DLS=[]

correct_psths_water_NAcc=[]
correct_psths_laser_NAcc=[]
incorrect_psths_water_NAcc=[]
incorrect_psths_laser_NAcc=[]

for ses in sessions:
    fluo_times = np.load(ses +'/alf/_ibl_fluo.times.npy')
    NAcc = np.load(ses +'/alf/_ibl_trials.NAcc.npy')
    DLS = np.load(ses +'/alf/_ibl_trials.DLS.npy')
    DMS = np.load(ses +'/alf/_ibl_trials.DMS.npy')
    cue_times = np.load (ses +'/alf/_ibl_trials.goCueTrigger_times.npy')
    opto_block = np.load (ses +'/alf/_ibl_trials.opto_block.npy')
    response_times = np.load(ses +'/alf/_ibl_trials.response_times.npy')
    feedback_times = np.load(ses +'/alf/_ibl_trials.feedback_times.npy')
    left_trials = np.load(ses +'/alf/_ibl_trials.contrastLeft.npy')
    right_trials = np.load(ses +'/alf/_ibl_trials.contrastRight.npy')
    l_trials = np.nan_to_num(left_trials)
    r_trials = np.nan_to_num(right_trials)
    signed_contrast = r_trials - l_trials
    feedback = np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
    choice = np.load(ses + '/alf/_ibl_trials.choice.npy')[:-1]
    opto_block = opto_block[:-1]
    response_times = response_times[:-1]
    feedback_times = feedback_times[:-1]
    l_trials = l_trials[:-1]
    r_trials = r_trials[:-1]
    signed_contrast = signed_contrast[:-1]
    feedback = feedback[:-1]
    stimOn_times  = np.load(ses + '/alf/_ibl_trials.stimOn_times.npy')
    reg_names = ['DLS','DMS','NAcc']

    correct_psths_water=[]
    correct_psths_laser=[]
    incorrect_psths_water=[]
    incorrect_psths_laser=[]

    for reg in [DLS, DMS, NAcc]:
        fluo_in_trials, fluo_time_in_trials = divide_in_trials(fluo_times, cue_times, zscore(reg,nan_policy='omit'), t_before_epoch = 0.2)
        opto = np.where(opto_block == 1)[0]
        reward = np.where(opto_block == 0)[0]
        correct = np.where(feedback == 1)[0]
        opto = np.intersect1d(opto, correct)
        reward = np.intersect1d(reward, correct)
        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            response_times, trial_list = opto, t_range = [-0.2, 3.0])
        condition2 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            response_times, trial_list = reward, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
            condition2 = (condition2[0]-F2, condition2[1])
        correct_psths_water.append(condition2) 
        correct_psths_laser.append(condition1) 

        opto = np.where(opto_block == 1)[0]
        reward = np.where(opto_block == 0)[0]
        incorrect = np.where(feedback == -1)[0]
        opto = np.intersect1d(opto, incorrect)
        reward = np.intersect1d(reward, incorrect)
        condition1 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            response_times, trial_list = opto, t_range = [-0.2, 3.0])
        condition2 = fp_psth(fluo_time_in_trials,
                            fluo_in_trials,
                            response_times, trial_list = reward, t_range = [-0.2, 3.0])
        if subtract_baseline ==  True:
            F1 = np.reshape(np.nanmean(condition1[0][:,0:3],1), (len(condition1[0]), 1))
            F2 = np.reshape(np.nanmean(condition2[0][:,0:3],1), (len(condition2[0]), 1))
            condition1 = (condition1[0]-F1, condition1[1])
            condition2 = (condition2[0]-F2, condition2[1])
        incorrect_psths_water.append(condition2) 
        incorrect_psths_laser.append(condition1)
    
    #Store psths
    correct_psths_water_DLS.append(correct_psths_water[0]) 
    correct_psths_laser_DLS.append(correct_psths_laser[0]) 
    incorrect_psths_water_DLS.append(incorrect_psths_water[0]) 
    incorrect_psths_laser_DLS.append(incorrect_psths_laser[0]) 

    correct_psths_water_DMS.append(correct_psths_water[1]) 
    correct_psths_laser_DMS.append(correct_psths_laser[1]) 
    incorrect_psths_water_DMS.append(incorrect_psths_water[1]) 
    incorrect_psths_laser_DMS.append(incorrect_psths_laser[1]) 

    correct_psths_water_NAcc.append(correct_psths_water[2]) 
    correct_psths_laser_NAcc.append(correct_psths_laser[2]) 
    incorrect_psths_water_NAcc.append(incorrect_psths_water[2]) 
    incorrect_psths_laser_NAcc.append(incorrect_psths_laser[2]) 


def psth_average(lists_of_psths, start=None, end=None):
    psths = []
    for c in lists_of_psths:
        psths.append(np.nanmean(c[0][start:end],axis=0))
    psths_time = lists_of_psths[0][1] #Always the same so just take in the first one 
    psths_avg = [psths,psths_time]
    return psths_avg

DMS = [psth_average(correct_psths_water_DMS),psth_average(correct_psths_laser_DMS)]
DLS = [psth_average(correct_psths_water_DLS),psth_average(correct_psths_laser_DLS)]
NAcc = [psth_average(correct_psths_water_NAcc),psth_average(correct_psths_laser_NAcc)]

_, ax = plt.subplots(2,3, sharey=True)
for i, reg in enumerate([DLS,DMS, NAcc]):
    plt.sca(ax[0,i])
    plot_psth(reg[1], color='orange', plot_error=True)
    plot_psth(reg[0], color='dodgerblue', plot_error=True)
    plt.title(reg_names[i])

DMS = [psth_average(incorrect_psths_water_DMS),psth_average(incorrect_psths_laser_DMS)]
DLS = [psth_average(incorrect_psths_water_DLS),psth_average(incorrect_psths_laser_DLS)]
NAcc = [psth_average(incorrect_psths_water_NAcc),psth_average(incorrect_psths_laser_NAcc)]

for i, reg in enumerate([DLS,DMS, NAcc]):
    plt.sca(ax[1,i])
    plot_psth(reg[1], color='orange', plot_error=True)
    plot_psth(reg[0], color='dodgerblue', plot_error=True)
    plt.title(reg_names[i])

## Divided by trial bin

DMS_1 = [psth_average(correct_psths_water_DMS, start=0, end=200),psth_average(correct_psths_laser_DMS, start=0, end=200)]
DLS_1 = [psth_average(correct_psths_water_DLS, start=0, end=200),psth_average(correct_psths_laser_DLS, start=0, end=200)]
NAcc_1 = [psth_average(correct_psths_water_NAcc, start=0, end=200),psth_average(correct_psths_laser_NAcc, start=0, end=200)]

DMS_2 = [psth_average(correct_psths_water_DMS, start=201, end=400),psth_average(correct_psths_laser_DMS, start=201, end=400)]
DLS_2 = [psth_average(correct_psths_water_DLS, start=201, end=400),psth_average(correct_psths_laser_DLS, start=201, end=400)]
NAcc_2 = [psth_average(correct_psths_water_NAcc, start=201, end=400),psth_average(correct_psths_laser_NAcc, start=201, end=400)]

DMS_3 = [psth_average(correct_psths_water_DMS, start=401, end=600),psth_average(correct_psths_laser_DMS, start=401, end=600)]
DLS_3 = [psth_average(correct_psths_water_DLS, start=401, end=600),psth_average(correct_psths_laser_DLS, start=401, end=600)]
NAcc_3 = [psth_average(correct_psths_water_NAcc, start=401, end=600),psth_average(correct_psths_laser_NAcc, start=401, end=600)]


_, ax = plt.subplots(2,3, sharey=True)
for i, reg in enumerate([DLS_1,DMS_1, NAcc_1]):
    plt.sca(ax[0,i])
    plot_psth(reg[1], color='orange', plot_error=True)
    plot_psth(reg[0], color='dodgerblue', plot_error=True)
    plt.title(reg_names[i])
for i, reg in enumerate([DLS_2,DMS_2, NAcc_2]):
    plt.sca(ax[0,i])
    plot_psth(reg[1], color='orange', plot_error=True, alpha=0.66)
    plot_psth(reg[0], color='dodgerblue', plot_error=True,alpha=0.66)
    plt.title(reg_names[i])
for i, reg in enumerate([DLS_3,DMS_3, NAcc_3]):
    plt.sca(ax[0,i])
    plot_psth(reg[1], color='orange', plot_error=True,alpha=0.33)
    plot_psth(reg[0], color='dodgerblue', plot_error=True,alpha=0.33)
    plt.title(reg_names[i])

DMS_1 = [psth_average(incorrect_psths_water_DMS, start=0, end=200),psth_average(incorrect_psths_laser_DMS, start=0, end=200)]
DLS_1 = [psth_average(incorrect_psths_water_DLS, start=0, end=200),psth_average(incorrect_psths_laser_DLS, start=0, end=200)]
NAcc_1 = [psth_average(incorrect_psths_water_NAcc, start=0, end=200),psth_average(incorrect_psths_laser_NAcc, start=0, end=200)]

DMS_2 = [psth_average(incorrect_psths_water_DMS, start=201, end=400),psth_average(incorrect_psths_laser_DMS, start=201, end=400)]
DLS_2 = [psth_average(incorrect_psths_water_DLS, start=201, end=400),psth_average(incorrect_psths_laser_DLS, start=201, end=400)]
NAcc_2 = [psth_average(incorrect_psths_water_NAcc,  start=201, end=400),psth_average(incorrect_psths_laser_NAcc, start=201, end=400)]

DMS_3 = [psth_average(incorrect_psths_water_DMS, start=401, end=600),psth_average(incorrect_psths_laser_DMS, start=401, end=600)]
DLS_3 = [psth_average(incorrect_psths_water_DLS, start=401, end=600),psth_average(incorrect_psths_laser_DLS, start=401, end=600)]
NAcc_3 = [psth_average(incorrect_psths_water_NAcc,  start=401, end=600),psth_average(incorrect_psths_laser_NAcc,  start=401, end=600)]

for i, reg in enumerate([DLS_1,DMS_1, NAcc_1]):
    plt.sca(ax[1,i])
    plot_psth(reg[1], color='orange', plot_error=True)
    plot_psth(reg[0], color='dodgerblue', plot_error=True)
    plt.title(reg_names[i])

for i, reg in enumerate([DLS_2,DMS_2, NAcc_2]):
    plt.sca(ax[1,i])
    plot_psth(reg[1], color='orange', plot_error=True,alpha=0.66)
    plot_psth(reg[0], color='dodgerblue', plot_error=True,alpha=0.66)
    plt.title(reg_names[i])

for i, reg in enumerate([DLS_3,DMS_3, NAcc_3]):
    plt.sca(ax[1,i])
    plot_psth(reg[1], color='orange', plot_error=True,alpha=0.33)
    plot_psth(reg[0], color='dodgerblue', plot_error=True,alpha=0.2)
    plt.title(reg_names[i])
