""" @alejandro 2019  - Alex extractor functions for Reward choice world, 
fork from ibllib will be complicated for this, requires  the rewardworld.extractors module
a modified alf """
import logging
from pathlib import Path
import traceback
from rew_alf.extractors import (biased_Reward_trials, biased_Reward_wheel)
from ibllib.io import raw_data_loaders as raw
import ibllib.io.flags as flags

logger_= logging.getLogger('ibllib')


def extractors_exist(session_path):
    settings = raw.load_settings(session_path)
    if settings is None:
        logger_.error(f'ABORT: No data found in "raw_behavior_data" folder {session_path}')
        return False
    task_name = settings['PYBPOD_PROTOCOL']
    task_name = task_name.split('_')[-1]
    extractor_type = task_name[:task_name.find('ChoiceWorld')]
    if any([extractor_type in x for x in globals()]):
        return extractor_type
    else:
        logger_.warning(str(session_path) +
                        f" No extractors were found for {extractor_type} ChoiceWorld")
        return False


def is_extracted(session_path):
    sp = Path(session_path)
    if (sp / 'alf').exists():
        return True
    else:
        return False


def from_path(session_path, force=False, save=True):
    """
    Extract a session from full ALF path (ex: '/scratch/witten/ibl_witten_01/2018-12-18/001')
    force: (False) overwrite existing files
    save: (True) boolean or list of ALF file names to extract
    """
    logger_.info('Extracting ' + str(session_path))
    extractor_type = extractors_exist(session_path)
    if is_extracted(session_path) and not force:
        logger_.info(f"Session {session_path} already extracted.")
        return
    if extractor_type == 'training':
        data = raw.load_data(session_path)
        training_trials.extract_all(session_path, data=data, save=save)
        training_wheel.extract_all(session_path, bp_data=data, save=save)
        logger_.info('session extracted \n')  # timing info in log
    if extractor_type == 'biased':
        data = raw.load_data(session_path)
        biased_trials.extract_all(session_path, data=data, save=save)
        biased_wheel.extract_all(session_path, bp_data=data, save=save)
        logger_.info('session extracted \n')  # timing info in log
    if extractor_type == 'Reward':
        data = raw.load_data(session_path)
        biased_Reward_trials.extract_all(session_path, data=data, save=save)
        biased_Reward_wheel.extract_all(session_path, bp_data=data, save=save)
        logger_.info('session extracted \n')  # timing info in log


def bulk(subjects_folder, dry=False):
    ses_path = Path(subjects_folder).glob('**/extract_me.flag')
    for p in ses_path:
        # @alejandro no need for flags until personal project data starts going to server
        # the flag file may contains specific file names for a targeted extraction
        save = flags.read_flag_file(p)
        if dry:
            print(p)
            continue
        try:
            from_path(p.parent, force=True, save=save)
        except Exception as e:
            error_message = str(p.parent) + ' failed extraction' + '\n    ' + str(e)
            error_message += traceback.format_exc()
            err_file = p.parent.joinpath('extract_me.error')
            p.replace(err_file)
            with open(err_file, 'w+') as f:
                f.write(error_message)
            logger_.error(error_message)

            continue
        p.unlink()
        flags.write_flag_file(p.parent.joinpath('register_me.flag'), file_list=save)