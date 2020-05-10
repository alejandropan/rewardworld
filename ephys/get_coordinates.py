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
from ibllib.pipes import histology
import alf.io

def get_picked_tracks(histology_path, glob_pattern="*_pts_transformed.csv"):
    """
    This outputs reads in the Lasagna output and converts the picked tracks in the IBL coordinates
    :param histology_path:
    :return:
    """
    xyzs = []
    files_track = list(histology_path.glob(glob_pattern))
    for file_track in files_track:
        print(file_track)
        # apmldv in the histology file is flipped along x and y directions
        ixiyiz = np.loadtxt(file_track, delimiter=',')[:, [1, 0, 2]]
        ixiyiz[:, 1] = 527 - ixiyiz[:, 1]
        ixiyiz = ixiyiz[np.argsort(ixiyiz[:, 2]), :]
        xyz = brat.bc.i2xyz(ixiyiz)
        xyz[:, 0] = - xyz[:, 0]
        xyzs.append(xyz)

    return {'files': files_track, 'xyz': xyzs}

def histology_to_alf(session_path, fit_location, probe= 'probe00'):
    '''
    Parameters
    ----------
    session_path : session path(str)
    fit_location : path to fit from lasagna (str)
    probe : probe folder to save the tracks and neuron locations
    The default is 'probe00'.

    Returns
    -------
    brain_regions : array with brain region for every unit in the reocoding
    currently only returns last TODO
    track_fit : channels along the tracks from histolgy
    '''
    
    atlas_params = {
        'PATH_ATLAS': str('/Users/alex/Documents/brain_atlases/Allen/'),
        'FILE_REGIONS':
            str(Path(inspect.getfile(atlas.AllenAtlas)).parent.joinpath('allen_structure_tree.csv')),
        'INDICES_BREGMA': list(np.array([1140 - (570 + 3.9), 540, 0 + 33.2]))
    }
        
    brat = atlas.AllenAtlas(res_um=25, par=atlas_params)

    ses_path =  Path(session_path)
    alf_path = ses_path.joinpath('alf',probe)
    channels = alf.io.load_object(alf_path, 'channels')['localCoordinates']
    file_tracks  = get_picked_tracks(Path(fit_location), glob_pattern="*_fit.csv")
    for i in range(len(file_tracks['files'])):
        brain_regions, _ ,track_fit = histology.get_brain_regions(file_tracks['xyz'][i], channels)
        np.save(str(alf_path) + '/' + str(i) + 'cluster_location.npy', np.array(brain_regions['name']))
        np.save(str(alf_path) + '/' + str(i)  + 'channels_xyz.npy', track_fit)
    return brain_regions, track_fit

def load_trj(path):
    '''
    Parameters
    ---------
    path : path for npy file with trj
    
    Returns
    --------
    locations: transforms into normal npy array
    '''
    x=np.load(path, allow_pickle=True)
    locations = x.item()
    return locations

def xyz_channels_from_trj(path_trj):
    '''

    Parameters
    ----------
    path_trj : path to the trj file micro/histology

    Returns
    -------
    xyz_channel and regions

    '''
    trj = load_trj(path_trj)
    ins = atlas.Insertion.from_dict(trj)
    regions, _, xyz  = histology.get_brain_regions(ins.xyz)
    np.save(Path(path_trj).parent + '/channels_xyz.npy', xyz)
    np.save(Path(path_trj).parent + '/cluster_location.npy', regions['name'])
    
def genarate_coordinates_from_micro():
    '''
    Example code,you can change manully the valuen for micromanipulator locations
    Will make into gui eventually
    Returns the names of every location in Allen  (region['name']) and the
    xyz location in IBL space for every channel in that insertion
    '''
    with open('/Users/alex/Documents/Postdoc/histology_ibl/temp.json') as f:
      trj = json.load(f)
    
    
    
    ins = atlas.Insertion.from_dict(trj)
    regions, _, xyz  = histology.get_brain_regions(ins.xyz)
    
    
    np.save('cluster_location.npy', regions['name'])
    np.save('xyz_channels.npy', xyz)
    np.save('20200115_dop04_regions.npy', regions)
    np.save('20200115_dop04_trj.npy', trj)
    
    
    # To load

    return region['name'], xyz_channels
    
    
def channel_id_xyz(channel_id,xyz_channels):
     
    '''
     Parameters
     ----------
     channel_id : channel id for every neuron
     xyz_channels : xyz coordinate for every channel

     Returns
     -------
     xyz for every neuron
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
    
    
    atlas_params = {
        'PATH_ATLAS': str('/Users/alex/Documents/brain_atlases/Allen/'),
        'FILE_REGIONS':
            str(Path(inspect.getfile(atlas.AllenAtlas)).parent.joinpath('allen_structure_tree.csv')),
        'INDICES_BREGMA': list(np.array([1140 - (570 + 3.9), 540, 0 + 33.2]))
    }
        
    brat = atlas.AllenAtlas(res_um=25, par=atlas_params)
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