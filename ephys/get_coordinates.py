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
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import ibllib.atlas as atlas
from path import Path
import inspect



def get_coordinates_from_micro():
    '''
    Example code,you can change manully the valuen for micromanipulator locations
    Will make into gui eventually
    Returns the names of every location in Allen  (region['name']) and the
    xyz location in IBL space for every channel in that insertion
    '''
    with open('/Users/alex/Documents/Postdoc/histology_ibl/temp.json') as f:
      trj = json.load(f)
    
    
    
    ins = atlas.Insertion.from_dict(trj)
    
    
    
    np.save('hist.regions_dop04_15.npy', regions['name'])
    np.save('20200115_dop04_regions.npy', regions)
    np.save('20200115_dop04_trj.npy', trj)
    
    
    # To load
    trj = np.load('20200314_dop11_trj.npy', allow_pickle=True)
    region, _, xyz_channels = np.load('20200314_dop11_regions.npy', allow_pickle=True)
    return region['name'], xyz_channels
    
    
 def channel_id_xyz(channel_id,xyz_channels):
     '''
     

     Parameters
     ----------
     channel_id : channel id for every cluster
     xyz_channels : xyz coordinate for every channel

     Returns
     -------
     xyz for every cluster

     '''
    
    xyz = xyz_channels[channel_id]
    return xyz
    


def plot_neurons_on_allen (location, xyz):
    '''
    Plots the location of neurons in most common AP slice
    size of marker marks the number of neurons in that channel
    color of marker marks the different brain area in allen space
    Parameters
    ----------
    location : location of neuron (array str)
    xyz : xyz location of each neuron

    Returns
    -------
    None.
    
    
    Unit test dataset:
    --------
    from ibllib import atlas
    import json
    from ibllib.pipes import histology
    import pandas as pd
    import random


    with open('/Users/alex/Documents/Postdoc/histology_ibl/temp.json') as f:
      rj = json.load(f)
    
    ins = atlas.Insertion.from_dict(trj)
    regions, _, xyz  = histology.get_brain_regions(ins.xyz)
    ids =  random.sample(list(np.arange(len(xyz))),231)
    location = regions['name'][ids]
    xyz = xyz[ids]
    
    '''
    brat = atlas.AllenAtlas(res_um=25, par=atlas_params)
    
    atlas_params = {
        'PATH_ATLAS': str('/Users/alex/Documents/brain_atlases/Allen/'),
        'FILE_REGIONS':
            str(Path(inspect.getfile(atlas.AllenAtlas)).parent.joinpath('allen_structure_tree.csv')),
        'INDICES_BREGMA': list(np.array([1140 - (570 + 3.9), 540, 0 + 33.2]))
    }
    # origin Allen left, front, up
    
    
    xyz =  xyz*1000 #Pass to mm
    ap_coordinate = np.median(xyz[:,1])
    _, sizes = np.unique(xyz[:,2], return_counts=True)
    marker_size = np.empty(len(xyz))
    for i, coord in enumerate(np.unique(xyz[:,2])):
        indx = np.sum(xyz[:,2] == coord)
        marker_size[np.where(xyz[:,2]==coord)] = indx
             
    fig, ax  =  plt.subplots(figsize=[15,10])
    brat.plot_cslice(ap_coordinate)
    scat = sns.scatterplot(x = xyz[:,0],y = xyz[:,2], size = marker_size, hue = location, linewidth=0.3, alpha =0.5)
    # Put a legend to the right side
    scat.legend(loc='center right', bbox_to_anchor=(1.5, 0.5), ncol=1)
    ax.set_xlabel('Medio-Lateral Distance from bregma (mm)')
    ax.set_ylabel('Dorso-Ventral Distance from bregma (mm)')
    plt.tight_layout()
    plt.savefig('location.png')
    plt.savefig('location.svg')