'''
Re-run fullbandit fix, check that all files are presented and that passive was not run twice
'''

from ibllib.io.extractors.ephys_fpga import _assign_events_bpod
from pipe_8_full_bandit_fix import full_bandit_fix
import numpy as np
import sys
def check_sync(ses):
    bpod_ch=16
    t = np.load(ses + '/raw_ephys_data/_spikeglx_sync.times.npy')
    c = np.load(ses + '/raw_ephys_data/_spikeglx_sync.channels.npy')
    p = np.load(ses + '/raw_ephys_data/_spikeglx_sync.polarities.npy')
    bpod_t = t[c==bpod_ch]
    bpod_p = p[c==bpod_ch]
    events = _assign_events_bpod(bpod_t, bpod_p)
    fpga_t = len(events[0])
    fpga_r = len(events[1])

    bpod_t_n = len(np.load(ses + '/alf/_ibl_trials.choice.npy'))
    bpod_r_n = np.sum(np.load(ses + '/alf/_ibl_trials.feedbackType.npy')==1)

    print(str(fpga_t - bpod_t_n) + ' trial dif fpgavsbpod') 
    print(str(fpga_r - bpod_r_n) + ' trial dif fpgavsbpod')

if __name__=='__main__':
        ses = sys.argv[1]
        check_sync(ses)