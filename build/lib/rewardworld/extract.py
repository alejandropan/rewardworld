#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 12:03:46 2019

@author: ibladmin
"""
from rewardworld import extract_session
import sys
import logging

logger_= logging.getLogger('ibllib')
logger_.setLevel('INFO')

def extract(root_data_folder, dry=False):
    extract_session.bulk(root_data_folder, dry=dry)


if __name__ == "__main__":
    # Map command line arguments to function arguments.
    extract(*sys.argv[1:])
