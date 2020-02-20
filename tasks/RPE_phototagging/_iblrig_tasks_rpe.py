
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:47:48 2019
@author: alex
"""

import logging

import matplotlib.pyplot as plt
from pybpod_rotaryencoder_module.module import RotaryEncoder
from pybpodapi.protocol import Bpod, StateMachine
import numpy as np
import online_plots as op


import task_settings
import user_settings
from session_params import SessionParamHandler



log = logging.getLogger('iblrig')
log.setLevel(logging.INFO)


global sph
sph = SessionParamHandler(task_settings, user_settings)


def bpod_loop_handler():
    f.canvas.flush_events()  # 100�s

# =============================================================================
# CONNECT TO BPOD
# =============================================================================
bpod = Bpod()

# Loop handler function is used to flush events for the online plotting

bpod.loop_handler = bpod_loop_handler
f, axes = op.make_fig(sph)
# Rotary Encoder State Machine handler

# =============================================================================
#     Start state machine definition
# =============================================================================
for i in range(100):  # Looping for no reason
        #tph.next_trial()
        log.info(f'Starting trial: {i + 1}')
        iti_s = np.random.poisson(5)
        log.info(f'iti: {iti_s}')
        sma = StateMachine(bpod)

        sma.add_state(
                state_name='block_start',
                state_timer= 0,  # ~100�s hardware irreducible delay
                state_change_conditions={'Tup': 'iti'}, #Not sure about Port1In
                output_actions= [])  # To FPGA
        
        sma.add_state(
                state_name='iti',
                state_timer= iti_s ,  # ~100�s hardware irreducible delay
                state_change_conditions={'Tup': 'reward'}, #Not sure about Port1In
                output_actions= [('BNC1', 255)])  # To FPGA

        sma.add_state(
                state_name='reward',
                state_timer= 0.5,
                state_change_conditions={'Tup': 'exit'},
                output_actions=[('Valve1', 255),
                        ('BNC1', 255)])  # To FPGA   
	

        # Send state machine description to Bpod device
        bpod.send_state_machine(sma)

        # Run state machine
        bpod.run_state_machine(sma)  # Locks until state machine 'exit' is reached
        #sph = sph.session_completed(bpod.session.current_trial.export())

bpod.close()

if __name__ == '__main__':

    print('main')
