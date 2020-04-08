#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:23:30 2020

@author: alex

README extractors

For behavior:

1. Make sure that the flags for extraction are correctly postioned in
each session. These are: extrac_me.flag and opto.flag

2. Run extract.py (in rewardworld) on the root folder (not the folder of an
individual session). 

3. Run opto extractor (run extact from opto extractor on roor folder). It now runs
just with the extract_me flag.

For ephys:
    
Logic of extractor, extract.py (uproot) runs extract_session.py. Extract_session
detects the type of session. For ephys trials it 1st) Calls laser_ephys_trals (bpod_data)
it then calls ephys_fpga(fpga data). This process extracts the sync not the KS2 conversion

KS2 conversion:

In matlab: 



from experimental data import sync_merge_ephys

need sync_merge_ephys.flag in probe folder

"""


