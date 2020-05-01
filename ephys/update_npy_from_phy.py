#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 01:39:50 2020

@author: alex

Fix spike sorted data for matching lengths
Phy only modifies TSV and spike_cluster data therefore the rest need to 
be made to match through this code


NOT FUNCTIONAL YET

"""

import numpy as np

def manual_curation_update(ks2_path, update=True, reset = False):
    '''
    ks2_paht: str with ks2 folder
    update updates npy with phy info
    reset discards phy changes
    Deletes or creates extra clusters
    '''
    
    #import
    spike_clusters = np.load(ks2_path + '/spike_clusters.npy' )
    ## Import unmodifed and spot differences
    similar_templates = np.load(ks2_path + "/similar_templates.npy")
    spike_templates = np.load(ks2_path + "/spike_templates.npy")
    spike_times = np.load(ks2_path + "/spike_times.npy")
    template_feature_ind = np.load(ks2_path + "/template_feature_ind.npy")
    template_features = np.load(ks2_path + "/template_features.npy")
    templates_ind = np.load(ks2_path + "/templates_ind.npy")
    templates = np.load(ks2_path + "/templates.npy")
    whitening_mat_inv = np.load(ks2_path + "/whitening_mat_inv.npy")
    whitening_mat  = np.load(ks2_path + "/whitening_mat.npy")
    # Extra/deleted clusters, this methods seems differences on first compare to second
    # array
    extra_clusters = set(np.unique(spike_clusters)) - set(np.unique(spike_templates))
    deleted_clusters = set(np.unique(spike_templates)) - set(np.unique(spike_clusters))
    
    if update == True:
       # First delete