import re
from pathlib import Path
import logging
from collections import OrderedDict
import subprocess
import shutil
from pathlib import Path
import mtscomp
import sys
from ibllib.io import ffmpeg, spikeglx
from ibllib.io.extractors import ephys_fpga
from ibllib.pipes import tasks
from ibllib.ephys import ephysqc, sync_probes, spikes
from ibllib.pipes.training_preprocessing import TrainingRegisterRaw as EphysRegisterRaw
from ibllib.qc.task_metrics import TaskQC
from ibllib.qc.task_extractors import TaskQCExtractor

_logger = logging.getLogger('ibllib')

def extract_pulses(session_path, overwrite=False):
    # outputs numpy
    syncs, out_files = ephys_fpga.extract_sync(session_path, overwrite=overwrite)
    for out_file in out_files:
        _logger.info(f"extracted pulses for {out_file}")

    status, sync_files = sync_probes.sync(session_path)
    return out_files + sync_files

if __name__=='__main__':
    extract_pulses(Path(*sys.argv[1]))
