#@alejandro from ibllb original extrators
#Last edit: 2019-05-29
#Currently laseron is commented, until implemented
from pathlib import Path
import numpy as np
import os
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.training_trials import *


#reward block identity hardcoded 1  and 0.7 in get_rew_probaLR
from ibllib.io.extractors.training_trials import *
#Need to put laser_on in tph

def get_laser(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    laser_on = np.array([t['laser_on'] for t in data])
    if raw.save_bool(save, '_ibl_trials.laseron.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.laseron.npy')
        np.save(lpath, laser_on)
    return laser_on


def get_rew_probaLR(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    p_rew_Left = np.array([t['rew_probability_left'] for t in data])
    p_rew_Right = np.array( [1 if x == 0.7 else 0.5 if x==0.5 else 0.7 for x in p_rew_Left])
    if raw.save_bool(save, '_ibl_trials.rewprobabilityLeft.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.rewprobabilityLeft.npy')
        np.save(lpath, p_rew_Left)
    return p_rew_Left, p_rew_Right

def get_feedback_times(session_path, save=False, data=False):
    """
    Get the times the water or error tone was delivered to the animal.
    **Optional:** saves _ibl_trials.feedback_times.npy

    Gets reward  and error state init times vectors,
    checks if theintersection of nans is empty, then
    merges the 2 vectors.

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    c_rw_times = [tr['behavior_data']['States timestamps']['correct_rewarded'][0][0]
                for tr in data]
    c_urw_times = [tr['behavior_data']['States timestamps']['correct_unrewarded'][0][0]
                for tr in data]
    err_times = [tr['behavior_data']['States timestamps']['error'][0][0]
                 for tr in data]
    nogo_times = [tr['behavior_data']['States timestamps']['no_go'][0][0]
                  for tr in data]
    assert sum(np.isnan(c_rw_times) & np.isnan(c_urw_times) &
               np.isnan(err_times) & np.isnan(nogo_times)) == 0
    merge = np.array([np.array(times)[~np.isnan(times)] for times in
                      zip(c_rw_times, c_urw_times, err_times, nogo_times)]).squeeze()
    if raw.save_bool(save, '_ibl_trials.feedback_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.feedback_times.npy')
        np.save(fpath, merge)
    return np.array(merge)


def get_feedbackType(session_path, save=False, data=False):
    """
    Get the feedback that was delivered to subject.
    **Optional:** saves _ibl_trials.feedbackType.npy

    Checks in raw datafile for error and reward state.
    Will raise an error if more than one of the mutually exclusive states have
    been triggered.

    Sets feedbackType to -1 if error state was trigered
    Sets feedbackType to -2 if correct_unreward state was triggered
    Sets feedbackType to +1 if correct_reward state was triggered
    Sets feedbackType to 0 if no_go state was triggered

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('int64')
    """
    if not data:
        data = raw.load_data(session_path)
    feedbackType = np.empty(len(data))
    feedbackType.fill(np.nan)
    correct_rewarded = []
    correct_unrewarded = []
    error = []
    no_go = []
    for t in data:
        correct_rewarded.append(~np.isnan(t['behavior_data']
                                ['States timestamps']['correct_rewarded'][0][0]))
        correct_unrewarded.append(~np.isnan(t['behavior_data']
                                ['States timestamps']['correct_unrewarded'][0][0]))
        error.append(~np.isnan(t['behavior_data']
                               ['States timestamps']['error'][0][0]))
        no_go.append(~np.isnan(t['behavior_data']
                               ['States timestamps']['no_go'][0][0]))

    if not all(np.sum([correct_rewarded, correct_unrewarded, error, no_go], axis=0) == np.ones(len(data))):
        raise ValueError

    feedbackType[correct_rewarded] = 1
    feedbackType[correct_unrewarded] = -2
    feedbackType[error] = -1
    feedbackType[no_go] = 0
    feedbackType = feedbackType.astype('int64')
    if raw.save_bool(save, '_ibl_trials.feedbackType.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.feedbackType.npy')
        np.save(fpath, feedbackType)
    return feedbackType



def get_contrastLR(session_path, save=False, data=False):
    """
    Get left and right contrasts from raw datafile
    **Optional:** save _ibl_trials.contrastLeft.npy and
        _ibl_trials.contrastRight.npy to alf folder.

    Uses signed_contrast to create left and right contrast vectors.

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)

    contrastLeft = np.array([t['contrast'] if np.sign(
        t['position']) < 0 else np.nan for t in data])
    contrastRight = np.array([t['contrast'] if np.sign(
        t['position']) > 0 else np.nan for t in data])
    # save if needed
    check_alf_folder(session_path)
    if raw.save_bool(save, '_ibl_trials.contrastLeft.npy'):
        lpath = os.path.join(session_path, 'alf', '_ibl_trials.contrastLeft.npy')
        np.save(lpath, contrastLeft)

    if raw.save_bool(save, '_ibl_trials.contrastRight.npy'):
        rpath = os.path.join(session_path, 'alf', '_ibl_trials.contrastRight.npy')
        np.save(rpath, contrastRight)

    return (contrastLeft, contrastRight)

def extract_all(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    feedbackType = get_feedbackType(session_path, save=save, data=data)
    contrastLeft, contrastRight = get_contrastLR(
        session_path, save=save, data=data)
    probabilityLeft = get_probabilityLeft(session_path, save=save, data=data) #Need to double check this
    choice = get_choice(session_path, save=save, data=data)
    rewardVolume = get_rewardVolume(session_path, save=save, data=data)
    feedback_times = get_feedback_times(session_path, save=save, data=data)
    stimOn_times = get_stimOn_times(session_path, save=save, data=data)
    intervals = get_intervals(session_path, save=save, data=data)
    response_times = get_response_times(session_path, save=save, data=data)
    iti_dur = get_iti_duration(session_path, save=save, data=data)
    go_cue_trig_times = get_goCueTrigger_times(session_path, save=save, data=data)
    go_cue_times = get_goCueOnset_times(session_path, save=save, data=data)
    rew_probaLR = get_rew_probaLR(session_path, save=save, data=data)
    #laser_on = get_laser(session_path, save=save, data=data)

    # Missing datasettypes
    # _ibl_trials.deadTime
    out = {'feedbackType': feedbackType,
           'contrastLeft': contrastLeft,
           'contrastRight': contrastRight,
           'probabilityLeft': probabilityLeft,
           'session_path': session_path,
           'choice': choice,
           'rewardVolume': rewardVolume,
           'feedback_times': feedback_times,
           'stimOn_times': stimOn_times,
           'intervals': intervals,
           'response_times': response_times,
           'iti_dur': iti_dur,
           'goCue_times': go_cue_times,
           'goCueTrigger_times': go_cue_trig_times,
           'rew_probaLR': rew_probaLR,
           #'laser_on': go_cue_times,
           }
    return out


