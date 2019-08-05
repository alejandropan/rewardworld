#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, June 6th 2019, 11:42:40 am
import logging

from pythonosc import udp_client

import iblrig.iotasks as iotasks
import iblrig.user_input as user_input
from iblrig.path_helper import SessionPathCreator

import iblrig.adaptive as adaptive
import iblrig.ambient_sensor as ambient_sensor
import iblrig.bonsai as bonsai
import iblrig.frame2TTL as frame2TTL
import iblrig.iotasks as iotasks
import iblrig.misc as misc
import iblrig.sound as sound
import iblrig.user_input as user_input
from iblrig.path_helper import SessionPathCreator
from iblrig.rotary_encoder import MyRotaryEncoder


log = logging.getLogger('iblrig')


class SessionParamHandler(object):
    """Session object imports user_settings and task_settings
    will and calculates other secondary session parameters,
    runs Bonsai and saves all params in a settings file.json"""

    def __init__(self, task_settings, user_settings):
        # =====================================================================
        # IMPORT task_settings, user_settings, and SessionPathCreator params
        # =====================================================================
        ts = {i: task_settings.__dict__[i]
              for i in [x for x in dir(task_settings) if '__' not in x]}
        self.__dict__.update(ts)
        us = {i: user_settings.__dict__[i]
              for i in [x for x in dir(user_settings) if '__' not in x]}
        self.__dict__.update(us)
        self = iotasks.deserialize_pybpod_user_settings(self)
        spc = SessionPathCreator(self.IBLRIG_FOLDER, self.IBLRIG_DATA_FOLDER,
                                 self.PYBPOD_SUBJECTS[0],
                                 protocol=self.PYBPOD_PROTOCOL,
                                 board=self.PYBPOD_BOARD, make=True)
        self.__dict__.update(spc.__dict__)

        # =====================================================================
        # OSC CLIENT
        # =====================================================================
        self.OSC_CLIENT_PORT = 7110
        self.OSC_CLIENT_IP = '127.0.0.1'
        self.OSC_CLIENT = udp_client.SimpleUDPClient(self.OSC_CLIENT_IP,
                                                     self.OSC_CLIENT_PORT)
        # =====================================================================
        # PROBES + WEIGHT
        # =====================================================================
        self.FORM_DATA = user_input.session_form(mouse_name=self.SUBJECT_NAME)
        self = user_input.parse_form_data(self)
        # =====================================================================
        # SAVE SETTINGS FILE AND TASK CODE
        # =====================================================================
        iotasks.save_session_settings(self)

        self.behavior_data = []
        self.elapsed_time = 0
        self.init_datetime = parser.parse(sph.PYBPOD_SESSION)
    # =========================================================================
    # METHODS
    # =========================================================================
    def patch_settings_file(self, patch):
        self.__dict__.update(patch)
        misc.patch_settings_file(self.SETTINGS_FILE_PATH, patch)

    def warn_ephys(self):
        title = 'START EPHYS RECODING'
        msg = ("Please start recording in spikeglx then press OK\n" +
               "Behavior task will run after you start the bonsai workflow")
        # from ibllib.graphic import popup
        # popup(title, msg)
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(title, msg)
        root.quit()

    def get_recording_site_data(self, probe='LEFT'):
        title = f'PROBE {probe} - recording site'
        fields = ['X (float)', 'Y (float)', 'Z (float)', 'D (float)',
                  'Angle (10 or 20)', 'Origin (bregma or lambda)']
        defaults = [0.0, 0.0, 0.0, 0.0, '10', 'bregma']
        types = [float, float, float, float, int, str]
        userdata = multi_input(
            title=title, add_fields=fields, defaults=defaults)
        try:
            data = [t(x) for x, t in zip(userdata, types)]
            data_dict = {'xyzd': data[:4], 'angle': data[4], 'origin': data[5]}
            return data_dict
        except Exception:
            log.warning(
                f"One or more inputs are of the wrong type. Expected {types}")
            return self.get_recording_site_data()

    def save_ambient_sensor_reading(self, bpod_instance):
        return ambient_sensor.get_reading(bpod_instance,
                                          save_to=self.SESSION_RAW_DATA_FOLDER)

    def get_subject_weight(self):
        return numinput(
            "Subject weighing (gr)", f"{self.PYBPOD_SUBJECTS[0]} weight (gr):",
            nullable=False)

    def bpod_lights(self, command: int):
        fpath = Path(self.IBLRIG_FOLDER) / 'scripts' / 'bpod_lights.py'
        os.system(f"python {fpath} {command}")

    def get_port_events(self, events, name=''):
        return misc.get_port_events(events, name=name)

    # =========================================================================
    # JSON ENCODER PATCHES
    # =========================================================================
    def reprJSON(self):
        d = self.__dict__.copy()
        d['OSC_CLIENT'] = str(d['OSC_CLIENT'])
        return d

    # =========================================================================
    # SAVE TRIAL DATA
    # =========================================================================

     def session_completed(self, behavior_data):
        """Update outcome variables using bpod.session.current_trial
        Check trial for state entries, first value of first tuple"""
        # Update elapsed_time
        self.elapsed_time = datetime.datetime.now() - self.init_datetime
        self.behavior_data = behavior_data
         # SAVE TRIAL DATA
        params = self.__dict__.copy()
        params.update({'behavior_data': behavior_data})
        # Convert to str all non serializable params
        params['osc_client'] = 'osc_client_pointer'
        params['init_datetime'] = params['init_datetime'].isoformat()
        params['elapsed_time'] = str(params['elapsed_time'])
'
        # Dump and save
        out = json.dumps(params, cls=ComplexEncoder)
        self.data_file.write(out)
        self.data_file.write('\n')
        self.data_file.close()
        # If more than 42 trials save transfer_me.flag
        if self.trial_num == 42:
            misc.create_flags(self.data_file_path, self.poop_count)
        





if __name__ == '__main__':
    print("Done!")
