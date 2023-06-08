
import numpy as np
import zetapy as zp
from brainbox.singlecell import bin_spikes

spikesclust = np.load('/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001/alf/probe00/pykilosort/spikes.clusters.npy')
spikestimes = np.load('/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001/alf/probe00/pykilosort/spikes.times.npy')
first_laser_times = np.load('/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001/alf/_ibl_trials.first_laser_times.npy')

laser_times = []
for l in first_laser_times:
    train = []
    for i in np.arange(20):
        train.append(l+(50*i))
    laser_times.append(train)
laser_times = np.concatenate(laser_times)

sig = []
latencies = []
for cluster in np.unique(spikesclust):
    cluster_spikes = spikestimes[spikesclust==cluster]
    if len(cluster_spikes)<100:
        sig.append(np.nan)
        latencies.append(np.nan)
    else:
        sig.append(zp.getZeta(cluster_spikes, first_laser_times, dblUseMaxDur=0.025)[0])
        try:
            latencies.append(zp.getZeta(cluster_spikes, first_laser_times, dblUseMaxDur=0.025,  boolReturnRate=True)[2]['dblPeakTime'])
        except:
            latencies.append(np.nan)


sig_clusters = np.unique(spikesclust)[np.where(np.array(sig)<0.01)]
sig_latencies = np.array(latencies)[np.where(np.array(sig)<0.01)]

sig_clusters[np.where(sig_latencies<0.001)]


for s in sig_clusters:
    a = bin_spikes(spikestimes[spikesclust==s], first_laser_times, pre_time=1, post_time=2, bin_size=0.01)
    sns.heatmap(a[0])
    plt.axvline(x=99, ymin=0, ymax=1)
    plt.axvline(x=199, ymin=0, ymax=1)
    plt.xticks(np.arange(len(a[1]))[::10], a[1][::10])
    plt.title(str(s))
    plt.show(block=True)




