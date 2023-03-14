from ephys_alf_summary import *
from scipy.stats import zscore

def zscore_set_params(array, params): # array myst be neuron X time_bins
    norm_array = (array - params[:,0][:,None])/(params[:,1][:,None])
    return norm_array


def get_norm_neurons(spike_times,spike_clusters,selection, bin_size=0.025):
    norm = np.zeros([len(selection),2])
    for i,n in enumerate(selection):
        spt = spike_times[spike_clusters==n]
        bins = np.arange(spt[0],spt[-1],bin_size)
        binned = np.bincount(np.digitize(spt,bins))/bin_size #divide by bin_size for firing rate
        mu = np.mean(binned)
        sigma = np.std(binned)
        norm[i,0] = mu
        norm[i,1] = sigma
    return norm

def psths_per_regions(sessions, roi='DLS'):
    region_summary = pd.DataFrame()
    counter = 0
    counter_real = 0
    for i in np.arange(len(sessions[:])):
        ses = sessions[i]
        for j in np.arange(len(ses.probe[:])):
            counter+=1
            prob = pd.DataFrame()
            selection_quality = ses.probe[j].cluster_selection
            locations = pd.Series(ses.probe[j].cluster_locations).map(group_dict)
            selection_location = np.where(locations==roi)[0]
            selection = np.intersect1d(selection_quality,selection_location)
            if len(selection)==0:
                continue
            else:
                counter_real+=1
                prob['binned_spikes_gocue'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.goCue_trigger_times[~np.isnan(ses.goCue_trigger_times)])] #np.array (n_align_times, n_clusters, n_bins)
                #prob['binned_spikes_iti_start'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.firstlaser_times[~np.isnan(ses.firstlaser_times)]+1)] #np.array (n_align_times, n_clusters, n_bins)
                if  np.nanmean(ses.probe[j].channel_hem)>0.5:
                    prob['binned_spikes_choice_ipsi'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.first_move[ses.choice==1][~np.isnan(ses.first_move[ses.choice==1])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  ses.first_move[ses.choice==-1][~np.isnan(ses.first_move[ses.choice==-1])])] #np.array (n_align_times, n_clusters, n_bins)
                else:
                    prob['binned_spikes_choice_ipsi'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.first_move[ses.choice==-1][~np.isnan(ses.first_move[ses.choice==-1])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  ses.first_move[ses.choice==1][~np.isnan(ses.first_move[ses.choice==1])])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_reward'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.response_times[ses.outcome==1][~np.isnan(ses.response_times[ses.outcome==1])], post_time=0.2)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.response_times[ses.outcome==0][~np.isnan(ses.response_times[ses.outcome==0])], post_time=4.0)] #np.array (n_align_times, n_clusters, n_bins)                
                prob['binned_spikes_laser'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.firstlaser_times[~np.isnan(ses.firstlaser_times)], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['neurons_zscore_params'] = [get_norm_neurons(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection)]
                prob['mouse'] = ses.mouse           
                prob['date'] = ses.date
                prob['ses'] = ses.ses
                prob['probe'] = j 
                prob['id'] =  ses.mouse + '_' + ses.date + '_' + ses.ses + '_' + str(j)
                region_summary = pd.concat([region_summary,prob])
    region_summary['region'] = roi
    return region_summary.reset_index()


def xval_mean(x, even=True):
    if even==True:
        return np.nanmean(x[::2,:,:], axis=0)
    else:
        return np.nanmean(x[1::2,:,:], axis=0)

def plot_region_psth(region_df, plot=False):
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True))
    contra_choice_psth = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=True))
    ipsi_choice_psth = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True))
    laser_psth = np.concatenate(region_df.binned_spikes_laser.apply(xval_mean,even=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,contra_choice_psth,ipsi_choice_psth,reward_psth, laser_psth, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    psths_fr_z = zscore(psths_fr,axis=1) # z-score
    #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=False))
    contra_choice_psth_plot = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=False))
    ipsi_choice_psth_plot = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=False))
    reward_psth_plot = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=False))
    error_psth_plot = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=False))
    laser_psth_plot = np.concatenate(region_df.binned_spikes_laser.apply(xval_mean,even=False))
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot,contra_choice_psth_plot,ipsi_choice_psth_plot,reward_psth_plot, laser_psth_plot, error_psth_plot], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],contra_choice_psth.shape[1],ipsi_choice_psth.shape[1],reward_psth.shape[1],laser_psth.shape[1], error_psth.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-10,vmax=10, cmap="seismic", cbar=False)    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 8, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[0], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 8,['GoCue','Choice Contra','Choice Ipsi','CS+', 'DA stim', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary.pdf')
        plt.close()


sessions = ephys_ephys_dataset(len(LASER_ONLY))
for i, ses in enumerate(LASER_ONLY):
        print(ses)
        ses_data = alf(ses, ephys=True)
        ses_data.mouse = Path(ses).parent.parent.name
        ses_data.date = Path(ses).parent.name
        ses_data.ses = Path(ses).name
        sessions[i] = ses_data

# Load at unique regions dictionary
loc = [] 
for i in np.arange(len(LASER_ONLY)):
    ses = sessions[i]
    for j in np.arange(4):
        try:
            loc.append(np.unique(ses.probe[j].cluster_locations.astype(str)))
        except:
            continue
unique_regions = np.unique(np.concatenate(loc))
unique_regions = unique_regions[np.where(unique_regions!='nan')]
# Look for any unreferenced regions
groups = pd.read_csv('/Volumes/witten/Alex/PYTHON/rewardworld/ephys/histology_files/simplified_regions.csv')
groups = groups.iloc[:,1:3]
groups = groups.set_index('original')
group_dict = groups.to_dict()['group']
current_regions = groups.reset_index().original.unique()
[group_dict[r] for r in current_regions] # This will error if dictionary is not complete

# Plot PSTHs for every regions  
rois = np.array(['SS', 'OFC', 'NAc', 'PFC', 'DMS', 'DLS', 'TS', 'VP', 'Olfactory', 'GPe', 'MO'])
locations = np.array([[0,3], [0,1], [1,0], [0,2], [1,2],  [1,3], [1,4], [2,0], [0,0], [2,2], [0,4]])

fig, ax = plt.subplots(3,5, sharex=True)
for i, roi in enumerate(rois):
    region_df = psths_per_regions(sessions, roi=roi)
    plt.sca(ax[locations[i][0], locations[i][1]])
    plot_region_psth(region_df)



def plot_stacked_histogram(rois):
    dist_df = pd.DataFrame()
    for roi in rois:
        region_df = psths_per_regions(sessions, roi=roi)
        dist_roi = pd.DataFrame()
        go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(np.nanmean, axis=0)) 
        contra_choice_psth = np.concatenate(region_df.binned_spikes_choice_contra.apply(np.nanmean, axis=0))
        ipsi_choice_psth = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(np.nanmean, axis=0))
        reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(np.nanmean, axis=0))
        error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(np.nanmean, axis=0))
        laser_psth = np.concatenate(region_df.binned_spikes_laser.apply(np.nanmean, axis=0))
        #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(np.nanmean, axis=0))
        psths = np.concatenate([go_cue_psth,contra_choice_psth,ipsi_choice_psth,reward_psth, laser_psth, error_psth], axis=1)
        psths_fr = psths/0.025 # Transform to firing rates
        psths_fr_z = zscore(psths_fr,axis=1) # z-score
        #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
        # crossvalidated sorting
        order = np.argmax(psths_fr_z, 1)
        xs_sorted = psths_fr_z[order.argsort(),:]
        breakpoints = np.cumsum([go_cue_psth.shape[1],contra_choice_psth.shape[1],ipsi_choice_psth.shape[1],reward_psth.shape[1],laser_psth.shape[1], error_psth.shape[1]])
        epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Co
        dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
        dist = np.round(dist,2)
        dist_roi['dist'] = dist
        dist_roi['region'] = roi
        dist_roi['epoch'] = ['GoCue','Choice Contra','Choice Ipsi','CS+',  'DA stim', 'CS-']
        dist_df = pd.concat([dist_df,dist_roi])
    dist_df.pivot('region', 'epoch').loc[['Olfactory','PFC', 'OFC', 'SS', 'MO', 'NAc', 'DMS', 'DLS', 'TS', 'VP', 'GPe'],:].plot(kind='bar', stacked=True, cmap='tab20c')
    plt.ylabel('% Neurons')





(['SS', 'OFC', 'NAc', 'PFC', 'DMS', 'DLS', 'TS', 'VP', 'Olfactory', 'GPe', 'MO']
