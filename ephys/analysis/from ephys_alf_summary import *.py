from ephys_alf_summary import *


sessions = ephys_ephys_dataset(len(ALL_NEW_SESSIONS))
for i, ses in enumerate(ALL_NEW_SESSIONS):
        print(ses)
        ses_data = alf(ses, ephys=True)
        ses_data.mouse = Path(ses).parent.parent.name
        ses_data.date = Path(ses).parent.name
        ses_data.ses = Path(ses).name
        ses_data.trial_within_block = \
                    trial_within_block(ses_data.to_df())['trial_within_block']
        ses_data.trial_within_block_real = \
                    trial_within_block(ses_data.to_df())['trial_within_block_real']
        ses_data.block_number = \
                    trial_within_block(ses_data.to_df())['block_number']
        ses_data.block_number_real = \
                    trial_within_block(ses_data.to_df())['block_number_real']
        ses_data.probabilityLeft_next = \
                    trial_within_block(ses_data.to_df())['probabilityLeft_next']
        ses_data.probabilityLeft_past = \
                    trial_within_block(ses_data.to_df())['probabilityLeft_past']
        ses_data.transition_type = \
                    add_transition_info(ses_data.to_df())['transition_type']
        ses_data.transition_analysis = \
                    add_transition_info(ses_data.to_df())['transition_analysis']
        sessions[i] = ses_data

# Load at unique regions dictionary
loc = [] 
for i in np.arange(len(ALL_NEW_SESSIONS)):
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


def psths_per_regions(sessions, roi='DLS'):
    region_summary = pd.DataFrame()
    counter = 0
    counter_real = 0
    for i in np.arange(len(sessions[:])):
        ses = sessions[i]
        for j in np.arange(len(ses.probe[:])):
            counter+=1
            prob = pd.DataFrame()
            good_units = ses.probe[j].cluster_selection
            locations = pd.Series(ses.probe[j].cluster_locations[good_units]).map(group_dict)
            selection = np.where(locations==roi)[0]
            if len(selection)==0:
                print('Why wont this ever print!')
                counter+=1
                continue
            else:
                counter_real+=1
                prob['binned_spikes_gocue'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.goCue_trigger_times)] #np.array (n_align_times, n_clusters, n_bins)
                if  np.nanmean(ses.probe[j].channel_hem)>0.5:
                    prob['binned_spikes_choice_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.first_move[ses.choice==1])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_ipsi'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.first_move[ses.choice==-1])] #np.array (n_align_times, n_clusters, n_bins)
                else:
                    prob['binned_spikes_choice_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.first_move[ses.choice==-1])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_ipsi'] =[get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.first_move[ses.choice==1])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_reward'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.response_times[ses.outcome==1])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.response_times[ses.outcome==0])] #np.array (n_align_times, n_clusters, n_bins)
                prob['mouse'] = ses.mouse
                prob['date'] = ses.date
                prob['ses'] = ses.ses
                prob['probe'] = j 
                prob['id'] =  ses.mouse + '_' + ses.date + '_' + ses.ses + '_' + str(j)
                region_summary = pd.concat([region_summary,prob])
    return region_summary.reset_index()

def plot_epoch_psth(binned_spikes)
    go_cue_psth = binned_spikes_gocue.apply(np.nanmean,axis=0).drop_na()


def plot_region_psth(region_df):
    go_cue_psth = region_df.binned_spikes_gocue.apply(np.nanmean,axis=0)

a = time.time()
dls = psths_per_regions(sessions, roi='DLS')
rt = time.time()-a


neural_data = neural_data.loc[neural_data['location']==area]
binned_spikes = np.zeros([len(neural_data['residuals_goCue']), 15])
for i in np.arange(len(neural_data['residuals_goCue'])):
    binned_spikes[i,:] = neural_data['residuals_goCue'].iloc[i].mean(axis=0)
order = np.argmax(binned_spikes, 1)
xs_sorted = binned_spikes[order.argsort(),:]


fig, ax = plt.subplots(1,3, sharey=True,  sharex=True)
plt.sca(ax[0])
sns.heatmap(xs_sorted, vmin=-0.5, vmax=0.5, center=0, cmap="seismic", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.ylabel('Cluster')
plt.title('Raw Data')
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from goCue')
plt.sca(ax[1])
sns.heatmap(xs_sorted, vmin=-0.5, vmax=0.5, center=0, cmap="seismic", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.ylabel('Cluster')
plt.title('Raw Data')
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from goCue')
plt.sca(ax[2])
sns.heatmap(xs_sorted, vmin=-0.5, vmax=0.5, center=0, cmap="seismic", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.ylabel('Cluster')
plt.title('Raw Data')
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from goCue')

