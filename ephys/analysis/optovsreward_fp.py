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
example = np.random.choice(opto_trials)
t_0 = feedback_times[example]-0.5
t_1 = feedback_times[example+11]

start=find_nearest(timestamps, t_0)
end=find_nearest(timestamps, t_1)

rate = np.nanmedian(np.diff(timestamps))
trial_range_plotted = np.arange(example,example+11)
opto_trials_example = np.intersect1d(trial_range_plotted,opto_trials)
water_trials_example = np.intersect1d(trial_range_plotted,water_trials)

_, ax = plt.subplots(3,1, sharey=True)
regions = ['NAcc','DMS', 'DLS']
for i, reg in enumerate([NAcc,DMS, DLS]):
    plt.sca(ax[i])
    plt.plot(timestamps[start:end],smooth(reg[start:end],1))
    plt.vlines(feedback_times[opto_trials_example], 0,np.nanmax(reg), color='orange')
    plt.vlines(feedback_times[water_trials_example], 0,np.nanmax(reg), color='dodgerblue')
    plt.vlines(feedback_times[opto_trials_example]+1, 0,np.nanmax(reg), color='black') #end of stim
    plt.title(regions[i])
