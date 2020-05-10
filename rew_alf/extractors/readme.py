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


For ephys:
    
Logic of extractor, extract.py (uproot) runs extract_session.py. Extract_session
detects the type of session. For ephys trials it 1st) Calls laser_ephys_trals (bpod_data)
it then calls ephys_fpga(fpga data). This process extracts the sync not the KS2 conversion
finallu sync merge converts to alf. Opto extractor, extracts opto signals from bpod



Notes:

From extract_session you can see all the  different steps
    
ephys_fpga_opto.extract_all has to be run in princeton if not is too slow 
opto_extractor can be run from laptop is fast

KS2 conversion:

need sync_merge_ephys.flag in probe folder
then run sync merge ephys in extract_session

"""


