#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Alejandro Pan
# @Date: Tuesday, February 12th 2019, 11:49:54 am
import numpy as np
import ibllib.io.raw_data_loaders as raw
import argparse
import ibllib.io.flags as flags
import logging
from pathlib import Path

from ibllib.io.extractors import (ephys_trials, ephys_fpga,
                                  biased_wheel, biased_trials,
                                  training_trials, training_wheel)

# this is a decorator to add a logfile to each extraction and registration on top of the logging

def get_hem(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
    if settings['HEM_STIM'] is None:
        hem_stim = settings['STIM_HEM']
    else:
        hem_stim = settings['HEM_STIM']
    hem = np.array([hem_stim for t in data])
    
    
    if raw.save_bool(save, '_ibl_trials.hem_stim.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.hem_stim.npy')
        np.save(lpath, hem)
        
    return hem
    
def get_opto(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    opto = np.array([t['opto'] for t in data])
    if raw.save_bool(save, '_ibl_trials.opto.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.opto.npy')
        np.save(lpath, opto)
        
    dummy_opto  = np.array([t['opto'] for t in data]) #Same as opto but not buffer saved
    if raw.save_bool(save, '_ibl_trials.dummy_opto.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.opto_dummy.npy')
        np.save(lpath, opto)
    return opto, dummy_opto

def extract_opto(session_path, save=False):
        data = data = raw.load_data(session_path)
        opto, dummy_opto = get_opto(session_path, save=save, data=False, settings=False)
        hemisphere  = get_hem(session_path, save=save, data=False, settings=False)
        out = {'laser_on': opto}
        out = {'hem_stim': hemisphere}
        
        return out

def extract(subjects_folder, dry=False):
    ses_path = Path(subjects_folder).glob('**/opto.flag')
    for p in ses_path:
        # @alejandro no need for flags until personal project data starts going to server
        # the flag file may contains specific file names for a targeted extraction
        save = flags.read_flag_file(p)
        if dry:
            print(p)
            continue
        try:
            extract_opto(p.parent, save=save)
        except:
            pass

        p.unlink()
        flags.write_flag_file(p.parent.joinpath('opto_extracted.flag'), file_list=save)


if __name__ == "__main__":
    ALLOWED_ACTIONS = ['extract']
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('action', help='Action: ' + ','.join(ALLOWED_ACTIONS))
    parser.add_argument('folder', help='A Folder containing a session')
    parser.add_argument('--dry', help='Dry Run', required=False, default=False, type=str)
    parser.add_argument('--count', help='Max number of sessions to run this on',
                        required=False, default=False, type=int)
    args = parser.parse_args()  # returns data from the options specified (echo)
    if args.dry and args.dry.lower() == 'false':
        args.dry = False
    assert(Path(args.folder).exists())
    if args.action == 'extract':
        extract(args.folder, dry=args.dry)