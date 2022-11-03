import numpy as np
import sys 

def find_nearest_past(value,array):
    d = array - value
    idx = np.where(d==d[np.where(d<=0)].max())
    return array[idx]

def find_nearest(value, array):
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def extract_licks(ses, laser_channel = 17, licking_channel = 18, save=True): 
    feedback = np.load(ses + '/alf/_ibl_trials.feedback_times.npy')
    feedback_outcome= np.load(ses + '/alf/_ibl_trials.feedbackType.npy')
    ch = np.load(ses + '/raw_ephys_data/_spikeglx_sync.channels.npy')
    p = np.load(ses + '/raw_ephys_data/_spikeglx_sync.polarities.npy')
    t = np.load(ses + '/raw_ephys_data/_spikeglx_sync.times.npy')
    laser = t[np.intersect1d(np.where(ch==laser_channel), np.where(p==1))]
    licks = t[np.intersect1d(np.where(ch==licking_channel), np.where(p==1))]
    # find firt laser for every train
    first_laser = np.concatenate([np.array([laser[0]]),laser[np.where(np.diff(laser)>1)[0]+1]])
    first_lick = np.array([find_nearest_past(i, licks)[0] for i in first_laser])
    assert len(feedback_outcome) == len(feedback)
    correct_feedback_times = feedback[feedback_outcome==1]
    idx_correct_feedback_times = np.where(feedback_outcome==1)[0]
    closest_laser_reward = np.array([find_nearest(i, first_laser) for i in correct_feedback_times]) - correct_feedback_times
    assert len(idx_correct_feedback_times) == len(closest_laser_reward)
    not_consumed = np.where(abs(closest_laser_reward)>1.5)[0]
    not_consumed_trials = idx_correct_feedback_times[not_consumed]
    feedback_outcome[not_consumed_trials] = -1
    if save==True:
        np.save(ses + '/alf/_ibl_trials.first_laser_times.npy', first_laser)
        np.save(ses + '/alf/_ibl_trials.first_lick_times.npy', first_lick)
        np.save(ses + '/alf/_ibl_trials.lick_times.npy', licks)
        np.save(ses + '/alf/_ibl_trials.feedbackTypeConsumed.npy', feedback_outcome)
    return len(not_consumed_trials)

if __name__ == "__main__":
    ses = sys.argv[1]
    extract_licks(ses)