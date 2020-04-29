#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:49:25 2020
Is on dev branch I need to import from the actual folder
@author: alex
"""
from ibllib import atlas
import json
from ibllib.pipes import histology
import pandas as pd



with open('/Users/alex/Documents/Postdoc/histology_ibl/temp.json') as f:
  trj = json.load(f)



ins = atlas.Insertion.from_dict(trj)
regions  = histology.get_brain_regions(ins.xyz)[0]


np.save('hist.regions_dop04_15.npy', regions['name'])
np.save('20200115_dop04_regions.npy', regions)
np.save('20200115_dop04_trj.npy', trj)


# To load
trj = np.load('20200314_dop11_trj.npy', allow_pickle=True)
region = np.load('20200314_dop11_regions.npy', allow_pickle=True)
