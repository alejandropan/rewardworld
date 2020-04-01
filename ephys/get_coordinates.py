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


with open('/Users/alex/Documents/Postdoc/histology_ibl/temp.json') as f:
  trj = json.load(f)

ins = atlas.Insertion.from_dict(trj)
regions, insertion_histology = histology.get_brain_regions(ins.xyz)
regions.keys()


