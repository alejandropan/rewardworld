#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 13:33:47 2020

@author: alex
"""


model_parameters = pd.read_pickle('armin_ibl.pkl')
psy = pd.read_pickle('armin_psy.pkl')



use_sessions, use_days = query_sessions_around_criterion_CSP(criterion=‘trained’,
                                                         days_from_criterion=[2, 0],
                                                         as_dataframe=False)
