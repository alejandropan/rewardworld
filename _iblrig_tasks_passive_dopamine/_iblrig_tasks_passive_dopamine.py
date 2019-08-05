#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Mon Jul 22 20:26:00 2019



@author: Alejandro

"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date:   2018-02-02 12:31:13

import logging

import matplotlib.pyplot as plt
from pybpod_rotaryencoder_module.module import RotaryEncoder
from pybpodapi.protocol import Bpod, StateMachine

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

# Rotary Encoder State Machine handler
rotary_encoder = [x for x in bpod.modules if x.name == 'RotaryEncoder1'][0]

# ROTARY ENCODER SEVENTS
# Set RE position to zero 'Z' + eneable all RE thresholds 'E'
# re_reset = rotary_encoder.create_resetpositions_trigger()
re_reset = 1
bpod.load_serial_message(rotary_encoder, re_reset,
                         [RotaryEncoder.COM_SETZEROPOS,  # ord('Z')
                          RotaryEncoder.COM_ENABLE_ALLTHRESHOLDS])  # ord('E')

# Close loop
re_close_loop = re_reset + 3
bpod.load_serial_message(rotary_encoder, re_close_loop, [ord('#'), 3])

# =============================================================================

#     Start state machine definition

# =============================================================================
for i in range(1):  # Looping for no reason
        tph.next_trial()
        log.info(f'Starting trial: {i + 1}')


        sma = StateMachine(bpod)

        sma.add_state(
                state_name='block_start',
                state_timer=1200,  # ~100�s hardware irreducible delay
                state_change_conditions={'Tup': 'reset_rotary_encoder'}, #Not sure about Port1In
                output_actions=[('BNC1', 255)])  # To FPGA

        sma.add_state(
                state_name='reset_rotary_encoder',
                state_timer=0,
                state_change_conditions={'Tup': 'baseline'},
                output_actions=[('Serial1', re_reset)])

        sma.add_state(
                state_name='baseline',
                state_timer=1200,  # ~100�s hardware irreducible delay
                state_change_conditions={'Tup': 'laser_on'},
                output_actions=[('BNC1', 255)])  # To FPGA

        sma.add_state(
                state_name='laser_on',
                state_timer=1200,  # ~100�s hardware irreducible delay
                state_change_conditions={'Tup': 'laser_off'},
                output_actions=[('BNC1', 255),('BNC1', 3)])  # To FPGA    

        sma.add_state(
                state_name='laser_off',
                state_timer=1200,  # ~100�s hardware irreducible delay
                state_change_conditions={'Tup': 'exit'},
                output_actions=[('BNC1', 255)])  # To FPGA  

        sma.add_state(
                state_name='exit',
                state_timer=0,
                state_change_conditions={'Tup': 'exit'},
                output_actions=[('BNC1', 255)])

        # Send state machine description to Bpod device
        bpod.send_state_machine(sma)

        # Run state machine
        bpod.run_state_machine(sma)  # Locks until state machine 'exit' is reached
        sph = sph.session_completed(bpod.session.current_trial.export())

bpod.close()





if __name__ == '__main__':

    print('main')

