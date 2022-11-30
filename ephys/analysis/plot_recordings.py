from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
import numpy as np
from ephys_alf_summary import alf
from ibllib.atlas.plots import  compute_volume_from_points, plot_volume_on_slice
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from encoding_model_summary_to_df import load_encoding_model
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic
import matplotlib

def compute_volume_from_points_diff(xyz, values=None, aggr='mean', fwhm=500, ba=None):
    ba = ba or AllenAtlas()
    idx = ba._lookup(xyz)
    ba_shape = ba.image.shape[0] * ba.image.shape[1] * ba.image.shape[2]
    if values is not None:
        volume = binned_statistic(idx, values, range=[0, ba_shape], statistic=aggr, bins=ba_shape).statistic
        volume[np.isnan(volume)] = -1
    else:
        volume = np.bincount(idx, minlength=ba_shape, weights=values)

    volume = volume.reshape(ba.image.shape[0], ba.image.shape[1], ba.image.shape[2]).astype(np.float32)

    if fwhm > 0:
        # Compute sigma used for gaussian kernel
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))
        sigma = fwhm / (fwhm_over_sigma_ratio * ba.res_um)
        # TODO to speed up only apply gaussian filter on slices within distance of chosen coordinate
        volume = gaussian_filter(volume, sigma=sigma)

    # Mask so that outside of the brain is set to nan
    volume[ba.label == 0] = np.nan
    return volume
def convolve_w_filter(volume,fwhm=500):
        fwhm_over_sigma_ratio = np.sqrt(8 * np.log(2))
        sigma = fwhm / (fwhm_over_sigma_ratio * ba.res_um)
        volume = gaussian_filter(volume, sigma=sigma)
        #volume[ba.label == 0] = np.nan
        return volume
def get_map(SESSIONS, variable = 'value_laser', thresh = 0.01, model=None):
    all_clusters = []
    all_selections= []
    all_sig=[]
    all_loc=[]
    for ses in SESSIONS:
        ses_data = alf(ses,ephys=True)
        ses_data.mouse = Path(ses).parent.parent.name
        ses_data.date = Path(ses).parent.name
        ses_data.ses = Path(ses).name
        # Make x data always on left hemisphere
        for i in np.arange(4):
            try:
                ses_data.probe[i].cluster_xyz[:,0] =  -abs(ses_data.probe[i].cluster_xyz[:,0])
                try:
                    selection_base  = model.loc[(model['mouse']==ses_data.mouse) & (model['date']==ses_data.date) 
                            & (model['ses']==ses_data.ses) & (model['probe']==i), 'cluster_id'].to_numpy()
                    all_clusters.append(ses_data.probe[i].cluster_xyz[selection_base,:])
                    all_loc.append(ses_data.probe[i].cluster_group_locations[selection_base].to_numpy())
                except:
                    all_clusters.append(ses_data.probe[i].cluster_xyz)
                if variable!=None:
                    selection  = model.loc[(model['mouse']==ses_data.mouse) & (model['date']==ses_data.date) 
                            & (model['ses']==ses_data.ses) & (model['probe']==i) & (model[variable]<thresh), 'cluster_id'].to_numpy()
                    binary_sig =  1*(model.loc[(model['mouse']==ses_data.mouse) & (model['date']==ses_data.date) 
                            & (model['ses']==ses_data.ses) & (model['probe']==i), variable].to_numpy()<thresh)
                    all_selections.append(ses_data.probe[i].cluster_xyz[selection,:])
                    all_sig.append(binary_sig)
            except:
                continue

    plotting_array=np.concatenate(all_clusters)
    plotting_array_selection=np.concatenate(all_selections)
    values = np.concatenate(all_sig)
    locations = np.concatenate(all_loc)
    assert len(values)==len(plotting_array)
    return plotting_array, plotting_array_selection, values, locations

def get_gains(SESSIONS, variable = 'value_laser', model=None):
    all_var=[]
    for ses in SESSIONS:
        mouse = Path(ses).parent.parent.name
        date = Path(ses).parent.name
        ses = Path(ses).name
        # Make x data always on left hemisphere
        for i in np.arange(4):
            try:
                var =  model.loc[(model['mouse']==mouse) & (model['date']==date) 
                            & (model['ses']==ses) & (model['probe']==i), variable].to_numpy()
                all_var.append(var)
            except:
                continue
    values = np.concatenate(all_var)
    return values    

# End of functions
SESSIONS = [
 '/volumes/witten/Alex/Data/Subjects/dop_47/2022-06-05/001',
 '/volumes/witten/Alex/Data/Subjects/dop_47/2022-06-06/001',
 '/volumes/witten/Alex/Data/Subjects/dop_47/2022-06-07/001',
 #'/volumes/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
 '/volumes/witten/Alex/Data/Subjects/dop_47/2022-06-10/002',
 '/volumes/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
 '/volumes/witten/Alex/Data/Subjects/dop_48/2022-06-20/001',
 '/volumes/witten/Alex/Data/Subjects/dop_48/2022-06-27/002',
 '/volumes/witten/Alex/Data/Subjects/dop_48/2022-06-28/001',
 '/volumes/witten/Alex/Data/Subjects/dop_49/2022-06-14/001',
 '/volumes/witten/Alex/Data/Subjects/dop_49/2022-06-15/001',
 '/volumes/witten/Alex/Data/Subjects/dop_49/2022-06-16/001',
 '/volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
 '/volumes/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',
 '/volumes/witten/Alex/Data/Subjects/dop_49/2022-06-19/001',
 '/volumes/witten/Alex/Data/Subjects/dop_49/2022-06-20/001',
 '/volumes/witten/Alex/Data/Subjects/dop_49/2022-06-27/003',
 '/volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
 '/volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
 '/volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
 '/volumes/witten/Alex/Data/Subjects/dop_53/2022-10-02/001',
 '/volumes/witten/Alex/Data/Subjects/dop_53/2022-10-03/001',
 '/volumes/witten/Alex/Data/Subjects/dop_53/2022-10-04/001',
 '/volumes/witten/Alex/Data/Subjects/dop_53/2022-10-05/001',
 '/volumes/witten/Alex/Data/Subjects/dop_53/2022-10-07/001']

output_path = '/Users/alexpan/Documents/neurons'
model = load_encoding_model() 

ba = AllenAtlas()
# Extract xyz coords from clusters dict
# Here we will set all ap values to a chosen value for visualisation purposes

#bin plotting array
plotting_array, plotting_array_selection, sig, locations = get_map(SESSIONS, variable = 'value_laser', thresh = 0.01, model=model)
fig,ax = plt.subplots(2)
plt.sca(ax[0])
sns.histplot(plotting_array[locations=='VPS'][:,1]*1e6)
plt.xlabel('AP coord VPS neurons')
sel = np.where((sig==1)&(locations=='VPS'))
plt.sca(ax[1])
sns.histplot(plotting_array[sel,1][0]*1e6)
plt.xlabel('AP coord VPS value neurons')

### Current work, map where significant are colored by gain
plotting_array, plotting_array_selection, sig, locations = get_map(SESSIONS, variable = 'value_laser', thresh = 0.01, model=model)
o_gains = get_gains(SESSIONS, variable = 'outcome_laser_gain',  model=model)
c_gains = get_gains(SESSIONS, variable = 'laser_contra_gain',  model=model)
cu_gains = get_gains(SESSIONS, variable = 'laser_cue_gain',  model=model)

noise_level=100
bins= np.array([-1250,-250,750,1750,2750])
binned_plotting_array = np.copy(plotting_array*1e6)
slices = bins-500
val = np.digitize(binned_plotting_array[:,1],bins)
val[val==len(bins)]=len(bins)-1
binned_plotting_array[:,1]=bins[val]
cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
cmap_bound.set_under([1, 1, 1], 0)

fig,ax = plt.subplots(3,len(bins))
value=cu_gains*sig
for i in  np.arange(len(bins)):
    plt.sca(ax[0,i])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  value[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points = np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.5, c=sig_point_value, cmap='seismic', edgecolors=None, vmin=-1,vmax=1)
    ax[0,i].set_xlim(-5000,0)
    ax[0,i].set_axis_off()
    sns.despine()

value=c_gains*sig
for i in  np.arange(len(bins)):
    plt.sca(ax[1,i])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  value[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points =  np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.5, c=sig_point_value, cmap='seismic',edgecolors=None, vmin=-1,vmax=1)
    ax[1,i].set_xlim(-5000,0)
    ax[1,i].set_axis_off()
    sns.despine()

value=o_gains*sig
for i in  np.arange(len(bins)):
    plt.sca(ax[2,i])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  value[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points =  np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.5, c=sig_point_value, cmap='seismic', edgecolors=None, vmin=-1,vmax=1)
    ax[2,i].set_xlim(-5000,0)
    ax[2,i].set_axis_off()
    sns.despine()

# Plot cue gains with further resolution

plotting_array, plotting_array_selection, sig, locations = get_map(SESSIONS, variable = 'value_laser', thresh = 0.01, model=model)
o_gains = get_gains(SESSIONS, variable = 'outcome_laser_gain',  model=model)
c_gains = get_gains(SESSIONS, variable = 'laser_contra_gain',  model=model)
cu_gains = get_gains(SESSIONS, variable = 'laser_cue_gain',  model=model)

noise_level=100
bins= np.array([-1750,-1500,-1250,-1000,-750,-500,-250, 0,250, 500,750, 1000,1250, 1500,1750, 2000, 2250, 2500])
binned_plotting_array = np.copy(plotting_array*1e6)
slices = bins-125
val = np.digitize(binned_plotting_array[:,1],bins)
val[val==len(bins)]=len(bins)-1
binned_plotting_array[:,1]=bins[val]
cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
cmap_bound.set_under([1, 1, 1], 0)

fig,ax = plt.subplots(3,6)
value=cu_gains*sig
for i in  np.arange(len(bins)):
    plt.sca(ax[int(i/6),int(i%6)])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  value[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points = np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.5, c=sig_point_value, cmap='seismic', edgecolors=None, vmin=-1,vmax=1)
    ax[int(i/6),int(i%6)].set_xlim(-5000,0)
    ax[int(i/6),int(i%6)].set_axis_off()
    sns.despine()




# deltaq version
plotting_array, plotting_array_selection, sig, locations = get_map(SESSIONS, variable = 'policy_laser', thresh = 0.01, model=model)
o_gains = get_gains(SESSIONS, variable = 'doutcome_laser_gain',  model=model)
c_gains = get_gains(SESSIONS, variable = 'dlaser_contra_gain',  model=model)
cu_gains = get_gains(SESSIONS, variable = 'dlaser_cue_gain',  model=model)

noise_level=100
bins= np.array([-1250,-250,750,1750,2750])
binned_plotting_array = np.copy(plotting_array*1e6)
slices = bins-500
val = np.digitize(binned_plotting_array[:,1],bins)
val[val==len(bins)]=len(bins)-1
binned_plotting_array[:,1]=bins[val]
cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
cmap_bound.set_under([1, 1, 1], 0)

fig,ax = plt.subplots(3,len(bins))
value=cu_gains*sig
for i in  np.arange(len(bins)):
    plt.sca(ax[0,i])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  value[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points = np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.5, c=sig_point_value, cmap='seismic', edgecolors=None, vmin=-1,vmax=1)
    ax[0,i].set_xlim(-5000,0)
    ax[0,i].set_axis_off()
    sns.despine()
value=c_gains*sig
for i in  np.arange(len(bins)):
    plt.sca(ax[1,i])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  value[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points =  np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.5, c=sig_point_value, cmap='seismic',edgecolors=None, vmin=-1,vmax=1)
    ax[1,i].set_xlim(-5000,0)
    ax[1,i].set_axis_off()
    sns.despine()
value=o_gains*sig
for i in  np.arange(len(bins)):
    plt.sca(ax[2,i])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  value[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points =  np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.5, c=sig_point_value, cmap='seismic', edgecolors=None, vmin=-1,vmax=1)
    ax[2,i].set_xlim(-5000,0)
    ax[2,i].set_axis_off()
    sns.despine()



# choice
plotting_array, plotting_array_selection, sig = get_map(SESSIONS, variable = 'choice', thresh = 0.01, model=model)

noise_level=50
bins= np.arange(-1000,3000,500)
binned_plotting_array = np.copy(plotting_array*1e6)
slices = bins-250
val = np.digitize(binned_plotting_array[:,1],bins)
val[val==len(bins)]=len(bins)-1
binned_plotting_array[:,1]=bins[val]
cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
cmap_bound.set_under([1, 1, 1], 0)

fig,ax = plt.subplots(1,len(bins))
for i in  np.arange(len(bins)):
    plt.sca(ax[i])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  sig[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points = np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.01, c='black', edgecolors=None)
    ax[i].set_xlim(-5000,0)
    ax[i].set_axis_off()
    sns.despine()



# choice sagittal
plotting_array, plotting_array_selection, sig = get_map(SESSIONS, variable = 'choice', thresh = 0.01, model=model)

noise_level=75
bins= np.arange(-3000,0,500)
binned_plotting_array = np.copy(plotting_array*1e6)
slices = bins-250
val = np.digitize(binned_plotting_array[:,0],bins)
val[val==len(bins)]=len(bins)-1
binned_plotting_array[:,0]=bins[val]
cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
cmap_bound.set_under([1, 1, 1], 0)

fig,ax = plt.subplots(1,len(bins))
for i in  np.arange(len(bins)):
    plt.sca(ax[i])
    ba.plot_sslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,0]==bins[i])[0],:]
    bin_points =  sig[np.where(binned_plotting_array[:,0]==bins[i])]
    sig_points = np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,1] + x_noise,points[:,2] + z_noise, s=0.1, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,1] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.1, c='black', edgecolors=None)
    ax[i].set_axis_off()
    sns.despine()

fig,ax = plt.subplots(4,1)
for j, i in enumerate(np.array([1,2,3,4])):
    plt.sca(ax[j])
    ba.plot_sslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,0]==bins[i])[0],:]
    bin_points =  sig[np.where(binned_plotting_array[:,0]==bins[i])]
    sig_points = np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,1] + x_noise,points[:,2] + z_noise, s=0.1, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,1] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.1, c='black', edgecolors=None)
    ax[j].set_axis_off()
    sns.despine()


# choice
plotting_array, plotting_array_selection, sig = get_map(SESSIONS, variable = 'choice', thresh = 0.01, model=model)
output_path = '/Users/alexpan/Documents/neurons/choice'
noise_level=100
bins= np.array(np.arange(-5000,3000,25))
binned_plotting_array = np.copy(plotting_array*1e6)
slices = bins-12
val = np.digitize(binned_plotting_array[:,1],bins)
val[val==len(bins)]=len(bins)-1
binned_plotting_array[:,1]=bins[val]
cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
cmap_bound.set_under([1, 1, 1], 0)


for i,binn in enumerate(bins):
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  sig[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points = np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))
    fig, ax = plt.subplots()    
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=0.5, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=0.5, c='black', edgecolors=None)
    ax.set_xlim(-5000,0)
    ax.set_axis_off()
    sns.despine()
    plt.savefig(output_path+'/%s.png'%i)

#best for choice
noise_level=100
bins= np.array([-200,800])
binned_plotting_array = np.copy(plotting_array*1e6)
val = np.digitize(binned_plotting_array[:,1],bins)
binned_plotting_array  = binned_plotting_array[val==1]
binned_plotting_array[:,1]=300
cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
cmap_bound.set_under([1, 1, 1], 0)
sig_point_value = sig[val==1]
sig_points = np.where(sig==1)
x_noise = np.random.normal(-noise_level,noise_level,len(binned_plotting_array))
z_noise = np.random.normal(-noise_level,noise_level,len(binned_plotting_array))
fig, ax = plt.subplots()    
ba.plot_cslice(300/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
plt.scatter(binned_plotting_array[:,0] + x_noise,binned_plotting_array[:,2] + z_noise, s=1, c='grey',edgecolors=None)
plt.scatter(binned_plotting_array[sig_points,0] + x_noise[sig_points], binned_plotting_array[sig_points,2] + z_noise[sig_points], 
                s=1, c='black', edgecolors=None)
ax.set_xlim(-5000,0)
ax.set_axis_off()
sns.despine()


# outcome
plotting_array, plotting_array_selection, sig = get_map(SESSIONS, variable = 'outcome', thresh = 0.01, model=model)

noise_level=100
bins= np.array([-1250,-250,750,1750,2750])
binned_plotting_array = np.copy(plotting_array*1e6)
slices = bins-500
val = np.digitize(binned_plotting_array[:,1],bins)
val[val==len(bins)]=len(bins)-1
binned_plotting_array[:,1]=bins[val]
cmap_bound = matplotlib.cm.get_cmap("bone_r").copy()
cmap_bound.set_under([1, 1, 1], 0)

fig,ax = plt.subplots(1,len(bins))
for i in  np.arange(len(bins)):
    plt.sca(ax[i])
    ba.plot_cslice(slices[i]/1e06, volume='boundary', mapping='Allen', cmap=cmap_bound)
    points = binned_plotting_array[np.where(binned_plotting_array[:,1]==bins[i])[0],:]
    bin_points =  sig[np.where(binned_plotting_array[:,1]==bins[i])]
    sig_points = np.where(bin_points!=0)[0]
    sig_point_value = bin_points[sig_points]
    x_noise = np.random.normal(-noise_level,noise_level,len(points))
    z_noise = np.random.normal(-noise_level,noise_level,len(points))    
    plt.scatter(points[:,0] + x_noise,points[:,2] + z_noise, s=1, c='grey',edgecolors=None)
    plt.scatter(points[sig_points,0] + x_noise[sig_points], points[sig_points,2] + z_noise[sig_points], 
                s=1, c='black', edgecolors=None)
    ax[i].set_xlim(-5000,0)
    ax[i].set_axis_off()
    sns.despine()

####


volume1 = compute_volume_from_points(binned_plotting_array ,aggr='sum', fwhm=200, ba=ba)
volume2 = compute_volume_from_points(binned_plotting_array, values=sig, aggr='mean', fwhm=200, ba=ba)
volume2_s = compute_volume_from_points(binned_plotting_array, values=sig, aggr='sum', fwhm=200, ba=ba)
volume_diff = np.divide(volume2_s,volume1)
assert volume_diff.shape == volume1.shape == volume2.shape == volume2_s.shape

fig, ax = plt.subplots(3,7, sharey=True, sharex=True)
for i in np.arange(7):
    plot_volume_on_slice(volume1, coord=-3000 + (500*i), slice='sagittal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume1)/3], ax=ax[0,i])
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)
    ax[0,i].set_title('ML %s um' %(-3000 + (500*i)))
    sns.despine()
    #plt.savefig(output_path+'/%sAP.png'%i)

for i in np.arange(7):
    plot_volume_on_slice(volume2_s, coord=-3000 + (500*i), slice='sagittal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume1)/3], ax=ax[1,i])
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)
    sns.despine()
    #plt.savefig(output_path+'/%sAP_value.png'%i)

for i in np.arange(7):
    plot_volume_on_slice(volume_diff, coord=-3000 + (500*i), slice='sagittal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume_diff)], ax=ax[2,i])
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)
    ax[2,i].set_facecolor('black')
    sns.despine()

ax[0,0].get_yaxis().set_visible(True)
ax[1,0].get_yaxis().set_visible(True)
ax[2,0].get_yaxis().set_visible(True)
ax[0,0].set_ylabel('n_neurons')
ax[1,0].set_ylabel('n_qchosen')
ax[2,0].set_ylabel('%_qchosen')
ax[0,0].set_yticklabels([])
ax[1,0].set_yticklabels([])
ax[2,0].set_yticklabels([])
plt.tight_layout()


# Coronal
binned_plotting_array = np.copy(plotting_array)
bins= np.arange(-1000,3000, 1000)/1000000
val = np.digitize(plotting_array[:,1],bins)
val[np.where(val==len(bins))] = len(bins)-1
binned_plotting_array[:,1]=bins[val]

volume1 = compute_volume_from_points(binned_plotting_array ,aggr='sum', fwhm=250, ba=ba)
volume2 = compute_volume_from_points(binned_plotting_array, values=sig, aggr='mean', fwhm=250, ba=ba)
volume2_s = compute_volume_from_points(binned_plotting_array, values=sig, aggr='sum', fwhm=250, ba=ba)
volume_diff = np.divide(volume2_s,volume1)
assert volume_diff.shape == volume1.shape == volume2.shape == volume2_s.shape

fig, ax = plt.subplots(3,7, sharey=True, sharex=True)
for i in np.arange(7):
    plot_volume_on_slice(volume1, coord=-1000 + (1000*i), slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume1)*0.], ax=ax[0,i])
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)
    ax[0,i].set_title('AP %s um' %(-1000 + (1000*i)))
    ax[0,i].set_xlim(-5000,0)
    sns.despine()
    #plt.savefig(output_path+'/%sAP.png'%i)

for i in np.arange(7):
    plot_volume_on_slice(volume2_s, coord=-1000 + (1000*i), slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume2_s)*0.8], ax=ax[1,i])
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)
    ax[1,i].set_xlim(-5000,0)
    sns.despine()
    #plt.savefig(output_path+'/%sAP_value.png'%i)

for i in np.arange(7):
    plot_volume_on_slice(volume_diff, coord=-1000 + (1000*i), slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume_diff)*0.8], ax=ax[2,i])
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)
    ax[2,i].set_facecolor('black')
    ax[2,i].set_xlim(-5000,0)
    sns.despine()

ax[0,0].get_yaxis().set_visible(True)
ax[1,0].get_yaxis().set_visible(True)
ax[2,0].get_yaxis().set_visible(True)
ax[0,0].set_ylabel('n_neurons')
ax[1,0].set_ylabel('n_qchosen')
ax[2,0].set_ylabel('%_qchosen')
ax[0,0].set_yticklabels([])
ax[1,0].set_yticklabels([])
ax[2,0].set_yticklabels([])
plt.tight_layout()

# Coronal policy 
plotting_array, plotting_array_selection, sig = get_map(SESSIONS, variable = 'policy_laser', thresh = 0.01, model=model)
binned_plotting_array = np.copy(plotting_array)
bins= np.arange(-500,3000, 500)/1000000
val = np.digitize(plotting_array[:,1],bins)
val[np.where(val==len(bins))] = len(bins)-1
binned_plotting_array[:,1]=bins[val]

volume1 = compute_volume_from_points(binned_plotting_array ,aggr='sum', fwhm=250, ba=ba)
volume2 = compute_volume_from_points(binned_plotting_array, values=sig, aggr='mean', fwhm=250, ba=ba)
volume2_s = compute_volume_from_points(binned_plotting_array, values=sig, aggr='sum', fwhm=250, ba=ba)
volume_diff = np.divide(volume2_s,volume1)
assert volume_diff.shape == volume1.shape == volume2.shape == volume2_s.shape

fig, ax = plt.subplots(3,7, sharey=True, sharex=True)
for i in np.arange(7):
    plot_volume_on_slice(volume1, coord=-500 + (500*i), slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume1)*0.8], ax=ax[0,i])
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)
    ax[0,i].set_title('AP %s um' %(-500 + (500*i)))
    ax[0,i].set_xlim(-5000,0)
    sns.despine()
    #plt.savefig(output_path+'/%sAP.png'%i)

for i in np.arange(7):
    plot_volume_on_slice(volume2_s, coord=-500 + (500*i), slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume2_s)*0.8], ax=ax[1,i])
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)
    ax[1,i].set_xlim(-5000,0)
    sns.despine()
    #plt.savefig(output_path+'/%sAP_value.png'%i)

for i in np.arange(7):
    plot_volume_on_slice(volume_diff, coord=-500 + (500*i), slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume_diff)*0.8], ax=ax[2,i])
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)
    ax[2,i].set_facecolor('black')
    ax[2,i].set_xlim(-5000,0)
    sns.despine()

ax[0,0].get_yaxis().set_visible(True)
ax[1,0].get_yaxis().set_visible(True)
ax[2,0].get_yaxis().set_visible(True)
ax[0,0].set_ylabel('n_neurons')
ax[1,0].set_ylabel('n_PLaser')
ax[2,0].set_ylabel('%_PLaser')
ax[0,0].set_yticklabels([])
ax[1,0].set_yticklabels([])
ax[2,0].set_yticklabels([])
plt.tight_layout()

# coronal 1.5mm projections qchosen

binned_plotting_array = np.copy(plotting_array)
bins= np.array([-500,1000,2500])/1000000
bins_edges = np.array([-100, 1800])/1000000
val = np.digitize(plotting_array[:,1],bins_edges)
binned_plotting_array[:,1]=bins[val]
volume1 = compute_volume_from_points(binned_plotting_array ,aggr='sum', fwhm=100, ba=ba)
volume2 = compute_volume_from_points(binned_plotting_array, values=sig, aggr='mean', fwhm=100, ba=ba)
volume2_s = compute_volume_from_points(binned_plotting_array, values=sig, aggr='sum', fwhm=100, ba=ba)
volume_diff = np.divide(volume2_s,volume1)
assert volume_diff.shape == volume1.shape == volume2.shape == volume2_s.shape

fig, ax = plt.subplots(3,3, sharey=True, sharex=True)
for i in np.arange(3):
    plot_volume_on_slice(volume1, coord=bins[i]*1000000, slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume1)*0.2], ax=ax[0,i])
    ax[0,i].get_xaxis().set_visible(False)
    ax[0,i].get_yaxis().set_visible(False)
    ax[0,i].set_title('AP %s um' %(bins[i]*1000000))
    ax[0,i].set_xlim(-5000,0)
    sns.despine()
    #plt.savefig(output_path+'/%sAP.png'%i)

for i in np.arange(3):
    plot_volume_on_slice(volume2_s, coord=bins[i]*1000000, slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0, np.nanmax(volume2_s)*0.5], ax=ax[1,i])
    ax[1,i].get_xaxis().set_visible(False)
    ax[1,i].get_yaxis().set_visible(False)
    ax[1,i].set_xlim(-5000,0)
    sns.despine()
    #plt.savefig(output_path+'/%sAP_value.png'%i)

for i in np.arange(3):
    plot_volume_on_slice(volume_diff, coord=bins[i]*1000000, slice='coronal', mapping='Allen', background='boundary',
                               cmap='Reds', brain_atlas=ba, clevels=[0.01, np.nanmax(volume_diff)*0.3], ax=ax[2,i])
    ax[2,i].get_xaxis().set_visible(False)
    ax[2,i].get_yaxis().set_visible(False)
    ax[2,i].set_facecolor('black')
    ax[2,i].set_xlim(-5000,0)
    sns.despine()

ax[0,0].get_yaxis().set_visible(True)
ax[1,0].get_yaxis().set_visible(True)
ax[2,0].get_yaxis().set_visible(True)
ax[0,0].set_ylabel('n_neurons')
ax[1,0].set_ylabel('n_qchosen')
ax[2,0].set_ylabel('%_qchosen')
ax[0,0].set_yticklabels([])
ax[1,0].set_yticklabels([])
ax[2,0].set_yticklabels([])
plt.tight_layout()