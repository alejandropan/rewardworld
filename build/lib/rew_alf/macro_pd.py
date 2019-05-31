#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:47:35 2019

@author: ibladmin
"""
import os
import pandas as pd
import numpy as np
import seaborn as sns
import re 
from rew_alf.npy2pd import *


subject_folder =  '/mnt/s0/Data/Subjects_personal_project/rewblocks8040/'
mice = sorted(os.listdir (subject_folder))
variables  = pybpod_vars()

df = pd.DataFrame()

for mouse in mice:
    dates =  sorted(os.listdir (subject_folder + mouse))
    for day in dates:
        #merge sessions from the same day
        path = subject_folder + mouse + '/' + day
        data  = session_loader(path, variables)
            
        
        

            
            