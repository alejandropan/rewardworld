from pybpodapi.protocol import Bpod, StateMachine

#Remember to introduce pre command: python C:\iblrig\scripts\bpod_lights.py 0
#Remember to introduce post command: python C:\iblrig\scripts\bpod_lights.py 1

# =============================================================================
# CONNECT TO BPOD
# =============================================================================

bpod = Bpod()

for i in range(1):  # Looping for no reason
    sma = StateMachine(bpod)
    sma.add_state(
            state_name='block_start',
            state_timer=0,  # ~100?s hardware irreducible delay
            state_change_conditions={'Tup': 'baseline'}, #Not sure about Port1In
            output_actions=[('BNC1', 255)])  # To FPGA

    sma.add_state(
        state_name='baseline',
        state_timer=10,  # ~100?s hardware irreducible delay
        state_change_conditions={'Tup': 'laser_on'},
        output_actions=[('BNC1', 255)])  # To FPGA

    sma.add_state(state_name='laser_on',
        state_timer=30,  # ~100?s hardware irreducible delay
        state_change_conditions={'Tup': 'laser_off'},
        output_actions=[('BNC1', 255),('BNC2', 3)])  # To FPGA    BNC3 to Pulse Pal

    sma.add_state(state_name='laser_off',
        state_timer=10,  # ~100?s hardware irreducible delay
        state_change_conditions={'Tup': 'exit'},
        output_actions=[('BNC1', 255)])  # To FPGA  

    sma.add_state(state_name='exit',
        state_timer=0,
        state_change_conditions={'Tup': 'exit'},
        output_actions=[('BNC1', 255)])

    # Send state machine description to Bpod device
    bpod.send_state_machine(sma)

    # Run state machine
    bpod.run_state_machine(sma)  # Locks until state machine 'exit' is reached
  
    # Save data
    #data = trial_completed(bpod.session.current_trial.export())
    #params = data.__dict__.copy()

    #data.sav

bpod.close()
