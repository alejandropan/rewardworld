#@alejandro from ibllb original extrators
#Last edit: 2019-05-27

import numpy as np
import os
import ibllib.io.raw_data_loaders as raw
import sys
sys.path.append("~\Documents\python\ibllib") #insert ibllib path
from ibllib.alf.extractors.training_trials import (
    check_alf_folder, get_feedbackType, get_probaLR,
    get_choice, get_rewardVolume, get_feedback_times, get_stimOn_times,
    get_intervals, get_response_times, get_iti_duration,
    get_goCueTrigger_times, get_goCueOnset_times)

#reward block identity hardcoded 0.8  and 0.4 in get_rew_probaLR
#Need to put laser_on in tph

def get_laser(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    laser_on = np.array([t['laser_on'] for t in data])
    if raw.save_bool(save, '_ibl_trials.laseron.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.laseron.npy')
        np.save(lpath, pLeft)
    return laser_on


def get_rew_probaLR(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    p_rew_Left = np.array([t['rew_probability_left'] for t in data])
    p_rew_Right = 0.8 if p_rew_Left == 0.4 else 0.4
    if raw.save_bool(save, '_ibl_trials.rewprobabilityLeft.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.rewprobabilityLeft.npy')
        np.save(lpath, pLeft)
    return rew_pLeft, rew_pRight


def extract_all(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    feedbackType = get_feedbackType(session_path, save=save, data=data)
    contrastLeft, contrastRight = get_contrastLR(
        session_path, save=save, data=data)
    probabilityLeft, _ = get_probaLR(session_path, save=save, data=data)
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
    laser_on = get_laser(session_path, save=save, data=data)

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
           'rew_probaLR': go_cue_times,
           'laser_on': go_cue_times,}
    return out