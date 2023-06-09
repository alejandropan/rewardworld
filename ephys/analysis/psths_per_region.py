from ephys_alf_summary import *
from scipy.stats import zscore
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D

# Not needed once i update ephys_alf_summary
def find_nearest_past(value,array):
    d = array - value
    idx = np.where(d==d[np.where(d<=0)].max())
    return array[idx]

def find_nearest(value, array):
    idx = (np.abs(array - value)).argmin()
    return idx

def zscore_set_params(array, params): # array myst be neuron X time_bins
    norm_array = (array - params[:,0][:,None])/(params[:,1][:,None])
    return norm_array

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

def psths_per_regions(sessions, roi='DLS', group_dict=None):
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
            #### This section should move to object (still not working i get a by 1indexing error sometimes) - ignore last trial
            ses.laserconsumed = np.copy(ses.outcome)
            closest_laser_reward = np.array([find_nearest(i, ses.response_times) for i in ses.firstlaser_times])
            if len(np.where(ses.outcome[closest_laser_reward]==0)[0])>0:
                print(len(np.where(ses.outcome[closest_laser_reward]==0)[0])) # ensure all assigned response_time are reward times
            #####
            ses.firstlaserarray = np.zeros(len(ses.laserconsumed)) #same as firstlaser but with nans where there was no laser 
            ses.firstlaserarray[:] = np.nan
            ses.firstlaserarray[closest_laser_reward] = ses.firstlaser_times
            # Determine value quantiles
            ses.fQRreward_cue = np.copy(np.roll(ses.fQRreward,1))
            ses.fQLreward_cue = np.copy(np.roll(ses.fQLreward,1))
            ses.fQRreward_cue[0] = 0
            ses.fQLreward_cue[0] = 0
            qchosen = np.copy(ses.fQRreward_cue)
            qchosen[np.where(ses.choice==-1)] = ses.fQLreward_cue[np.where(ses.choice==-1)]
            deltaq_rl = ses.fQRreward_cue - ses.fQLreward_cue
            deltaq_lr = ses.fQLreward_cue - ses.fQRreward_cue
            deltaq_rl = deltaq_rl
            deltaq_lr = deltaq_lr
            deltaq = [deltaq_lr, deltaq_rl]
            value = (ses.fQRreward * ses.fchoice_prediction) + \
                    (ses.fQLreward * (1- ses.fchoice_prediction))
            if np.nanmean(ses.probe[j].channel_hem)>0.5:
                deltaq = deltaq[0]
            else:
                deltaq = deltaq[1]
            qchosen_upper_lim  = np.quantile(qchosen,0.66)
            qchosen_lower_lim  = np.quantile(qchosen,0.33)
            qchosen_upper = np.where(qchosen>=qchosen_upper_lim)
            qchosen_lower = np.where(qchosen<=qchosen_lower_lim)
            delta_upper_lim  = np.quantile(deltaq,0.66)
            delta_lower_lim  = np.quantile(deltaq,0.33)
            delta_upper = np.where(deltaq>=delta_upper_lim)
            delta_lower = np.where(deltaq<=delta_lower_lim)

            value_upper_lim = np.quantile(value,0.66)
            value_lower_lim = np.quantile(value,0.33)
            value_upper = np.where(deltaq>=value_upper_lim)
            value_lower = np.where(deltaq<=value_lower_lim)
            ## Reaction time
            decision_times =ses.response_times - ses.goCue_trigger_times
            decision_times_upper_lim  = np.nanquantile(decision_times,0.66)
            decision_times_lower_lim  = np.nanquantile(decision_times,0.33)
            decision_times_upper = np.where(decision_times>=decision_times_upper_lim)
            decision_times_lower = np.where(decision_times<=decision_times_lower_lim)

            if len(selection)==0:
                continue
            else:
                print(ses.mouse + ' ' + ses.date + ' enough neurons %s' %len(selection))
                counter_real+=1
                ## General
                prob['binned_spikes_gocue'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                              selection, ses.goCue_trigger_times[~np.isnan(ses.goCue_trigger_times)])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                              selection, ses.response_times[~np.isnan(ses.response_times)],post_time=4.0)] #np.array (n_align_times, n_clusters, n_bins)
                #prob['binned_spikes_iti_start'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.firstlaser_times[~np.isnan(ses.firstlaser_times)]+1)] #np.array (n_align_times, n_clusters, n_bins)
                if  np.nanmean(ses.probe[j].channel_hem)>0.5:
                    lasered_contra_choices = np.intersect1d(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)))
                    lasered_ipsi_choices = np.intersect1d(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)))
                    prob['binned_spikes_choice_ipsi'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                        ses.first_move[ses.choice==1][~np.isnan(ses.first_move[ses.choice==1])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                         ses.first_move[ses.choice==-1][~np.isnan(ses.first_move[ses.choice==-1])])] #np.array (n_align_times, n_clusters, n_bins)
                    error_contra_choices = np.intersect1d(np.where(ses.outcome==0), (np.where(ses.choice==-1)))
                    error_ipsi_choices = np.intersect1d(np.where(ses.outcome==0), (np.where(ses.choice==1)))              
                    correct_contra_choices = np.intersect1d(np.where(ses.outcome==1), (np.where(ses.choice==-1)))
                    correct_ipsi_choices = np.intersect1d(np.where(ses.outcome==1), (np.where(ses.choice==1)))

                else:
                    lasered_contra_choices = np.intersect1d(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)))
                    lasered_ipsi_choices = np.intersect1d(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)))
                    error_contra_choices = np.intersect1d(np.where(ses.outcome==0), (np.where(ses.choice==1)))
                    error_ipsi_choices = np.intersect1d(np.where(ses.outcome==0), (np.where(ses.choice==-1)))
                    correct_contra_choices = np.intersect1d(np.where(ses.outcome==1), (np.where(ses.choice==1)))
                    correct_ipsi_choices = np.intersect1d(np.where(ses.outcome==1), (np.where(ses.choice==-1)))
                    prob['binned_spikes_choice_ipsi'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                                                        ses.first_move[ses.choice==-1][~np.isnan(ses.first_move[ses.choice==-1])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                        ses.first_move[ses.choice==1][~np.isnan(ses.first_move[ses.choice==1])])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_laser'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                         ses.firstlaser_times[~np.isnan(ses.firstlaser_times)], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                     
                prob['binned_spikes_laser_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                         ses.firstlaserarray[lasered_contra_choices][~np.isnan(ses.firstlaserarray[lasered_contra_choices])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_laser_ipsi'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                        ses.firstlaserarray[lasered_ipsi_choices][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                prob['binned_spikes_reward_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                         ses.response_times[correct_contra_choices][~np.isnan(ses.response_times[lasered_contra_choices])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_reward_ipsi'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                        ses.response_times[correct_ipsi_choices][~np.isnan(ses.response_times[lasered_ipsi_choices])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                      
                prob['binned_spikes_error_contra'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                         ses.response_times[error_contra_choices][~np.isnan(ses.response_times[error_contra_choices])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_error_ipsi'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                        ses.response_times[error_ipsi_choices][~np.isnan(ses.response_times[error_ipsi_choices])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                     
                prob['binned_spikes_outcome_reward'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.response_times[ses.outcome==1][~np.isnan(ses.response_times[ses.outcome==1])], post_time=4.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, ses.response_times[ses.outcome==0][~np.isnan(ses.response_times[ses.outcome==0])], post_time=4.0)] #np.array (n_align_times, n_clusters, n_bins)                
                
                ## QChosen
                lasered_qchosen_upper = np.intersect1d((np.where(ses.laserconsumed==1)), qchosen_upper)
                lasered_qchosen_lower = np.intersect1d((np.where(ses.laserconsumed==1)), qchosen_lower)
                rewarded_qchosen_upper = np.intersect1d((np.where(ses.outcome==1)), qchosen_upper)
                rewarded_qchosen_lower = np.intersect1d((np.where(ses.outcome==1)), qchosen_lower)
                errored_qchosen_upper = np.intersect1d((np.where(ses.outcome==0)), qchosen_upper)
                errored_qchosen_lower = np.intersect1d((np.where(ses.outcome==0)), qchosen_lower)

                if  np.nanmean(ses.probe[j].channel_hem)>0.5:
                    lasered_contra_choices_qchosen_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), qchosen_upper))
                    lasered_ipsi_choices_qchosen_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), qchosen_upper))
                    ipsi_qchosen_upper = np.intersect1d((np.where(ses.choice==1)), qchosen_upper)
                    contra_qchosen_upper = np.intersect1d((np.where(ses.choice==-1)), qchosen_upper)
                    lasered_contra_choices_qchosen_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), qchosen_lower))
                    lasered_ipsi_choices_qchosen_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), qchosen_lower))
                    ipsi_qchosen_lower = np.intersect1d((np.where(ses.choice==1)), qchosen_lower)
                    contra_qchosen_lower = np.intersect1d((np.where(ses.choice==-1)), qchosen_lower)                    
                else:
                    lasered_contra_choices_qchosen_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), qchosen_upper))
                    lasered_ipsi_choices_qchosen_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), qchosen_upper))
                    ipsi_qchosen_upper = np.intersect1d((np.where(ses.choice==-1)), qchosen_upper)
                    contra_qchosen_upper = np.intersect1d((np.where(ses.choice==1)), qchosen_upper)
                    lasered_contra_choices_qchosen_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), qchosen_lower))
                    lasered_ipsi_choices_qchosen_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), qchosen_lower))
                    ipsi_qchosen_lower = np.intersect1d((np.where(ses.choice==-1)), qchosen_lower)
                    contra_qchosen_lower = np.intersect1d((np.where(ses.choice==1)), qchosen_lower)   

                empty_qchosen_selections = any([len(sel)<=5 for sel in [
                lasered_contra_choices_qchosen_upper, lasered_ipsi_choices_qchosen_upper, 
                ipsi_qchosen_upper, contra_qchosen_upper, lasered_contra_choices_qchosen_lower, 
                lasered_ipsi_choices_qchosen_lower, ipsi_qchosen_lower, contra_qchosen_lower]])

                prob['binned_spikes_gocue_qchosen_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                            selection, ses.goCue_trigger_times[qchosen_upper][~np.isnan(ses.goCue_trigger_times[qchosen_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_gocue_qchosen_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                            selection, ses.goCue_trigger_times[qchosen_lower][~np.isnan(ses.goCue_trigger_times[qchosen_lower])])] 
                prob['binned_spikes_laser_qchosen_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_qchosen_upper][~np.isnan(ses.firstlaserarray[lasered_qchosen_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                prob['binned_spikes_laser_qchosen_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_qchosen_lower][~np.isnan(ses.firstlaserarray[lasered_qchosen_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                prob['binned_spikes_outcome_reward_qchosen_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[rewarded_qchosen_upper][~np.isnan(ses.response_times[rewarded_qchosen_upper])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_reward_qchosen_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[rewarded_qchosen_lower][~np.isnan(ses.response_times[rewarded_qchosen_lower])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error_qchosen_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[errored_qchosen_upper][~np.isnan(ses.response_times[errored_qchosen_upper])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error_qchosen_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                    selection, ses.response_times[errored_qchosen_lower][~np.isnan(ses.response_times[errored_qchosen_lower])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)

                if empty_qchosen_selections==False:
                    prob['binned_spikes_choice_ipsi_qchosen_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[ipsi_qchosen_upper][~np.isnan(ses.first_move[ipsi_qchosen_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra_qchosen_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[contra_qchosen_upper][~np.isnan(ses.first_move[contra_qchosen_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_ipsi_qchosen_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[ipsi_qchosen_lower][~np.isnan(ses.first_move[ipsi_qchosen_lower])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra_qchosen_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[contra_qchosen_lower][~np.isnan(ses.first_move[contra_qchosen_lower])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_contra_qchosen_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_contra_choices_qchosen_upper][~np.isnan(ses.firstlaserarray[lasered_contra_choices_qchosen_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_contra_qchosen_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_contra_choices_qchosen_lower][~np.isnan(ses.firstlaserarray[lasered_contra_choices_qchosen_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_ipsi_qchosen_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_ipsi_choices_qchosen_upper][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices_qchosen_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                    prob['binned_spikes_laser_ipsi_qchosen_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_ipsi_choices_qchosen_lower][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices_qchosen_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                


                ## DeltaQ                 
                lasered_delta_upper = np.intersect1d((np.where(ses.laserconsumed==1)), delta_upper)
                lasered_delta_lower = np.intersect1d((np.where(ses.laserconsumed==1)), delta_lower)
                rewarded_delta_upper = np.intersect1d((np.where(ses.outcome==1)), delta_upper)
                rewarded_delta_lower = np.intersect1d((np.where(ses.outcome==1)), delta_lower)
                errored_delta_upper = np.intersect1d((np.where(ses.outcome==0)), delta_upper)
                errored_delta_lower = np.intersect1d((np.where(ses.outcome==0)), delta_lower)
                if  np.nanmean(ses.probe[j].channel_hem)>0.5:
                    lasered_contra_choices_delta_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), delta_upper))
                    lasered_ipsi_choices_delta_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), delta_upper))
                    ipsi_delta_upper = np.intersect1d((np.where(ses.choice==1)), delta_upper)
                    contra_delta_upper = np.intersect1d((np.where(ses.choice==-1)), delta_upper)
                    lasered_contra_choices_delta_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), delta_lower))
                    lasered_ipsi_choices_delta_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), delta_lower))
                    ipsi_delta_lower = np.intersect1d((np.where(ses.choice==1)), delta_lower)
                    contra_delta_lower = np.intersect1d((np.where(ses.choice==-1)), delta_lower)                    
                else:
                    lasered_contra_choices_delta_upper = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), delta_upper))
                    lasered_ipsi_choices_delta_upper = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), delta_upper))
                    ipsi_delta_upper = np.intersect1d((np.where(ses.choice==-1)), delta_upper)
                    contra_delta_upper = np.intersect1d((np.where(ses.choice==1)), delta_upper)
                    lasered_contra_choices_delta_lower = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), delta_lower))
                    lasered_ipsi_choices_delta_lower = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), delta_lower))
                    ipsi_delta_lower = np.intersect1d((np.where(ses.choice==-1)), delta_lower)
                    contra_delta_lower = np.intersect1d((np.where(ses.choice==1)), delta_lower)  

                empty_delta_selections = any([len(sel)<=5 for sel in [
                lasered_contra_choices_delta_upper, lasered_ipsi_choices_delta_upper, 
                ipsi_delta_upper, contra_delta_upper, lasered_contra_choices_delta_lower, 
                lasered_ipsi_choices_delta_lower, ipsi_delta_lower, contra_delta_lower]])

                prob['binned_spikes_gocue_delta_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                            selection, ses.goCue_trigger_times[delta_upper][~np.isnan(ses.goCue_trigger_times[delta_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_gocue_delta_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                            selection, ses.goCue_trigger_times[delta_lower][~np.isnan(ses.goCue_trigger_times[delta_lower])])]
                prob['binned_spikes_laser_delta_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_delta_upper][~np.isnan(ses.firstlaserarray[lasered_delta_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                prob['binned_spikes_laser_delta_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_delta_lower][~np.isnan(ses.firstlaserarray[lasered_delta_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)            
                prob['binned_spikes_outcome_reward_delta_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[rewarded_delta_upper][~np.isnan(ses.response_times[rewarded_delta_upper])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_reward_delta_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[rewarded_delta_lower][~np.isnan(ses.response_times[rewarded_delta_lower])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error_delta_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[errored_delta_upper][~np.isnan(ses.response_times[errored_delta_upper])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error_delta_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                    selection, ses.response_times[errored_delta_lower][~np.isnan(ses.response_times[errored_delta_lower])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                if empty_delta_selections==False:              
                    prob['binned_spikes_choice_ipsi_delta_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[ipsi_delta_upper][~np.isnan(ses.first_move[ipsi_delta_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra_delta_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[contra_delta_upper][~np.isnan(ses.first_move[contra_delta_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_ipsi_delta_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[ipsi_delta_lower][~np.isnan(ses.first_move[ipsi_delta_lower])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra_delta_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[contra_delta_lower][~np.isnan(ses.first_move[contra_delta_lower])])] #np.array (n_align_times, n_clusters, n_bins)                 
                    prob['binned_spikes_laser_contra_delta_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_contra_choices_delta_upper][~np.isnan(ses.firstlaserarray[lasered_contra_choices_delta_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_contra_delta_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_contra_choices_delta_lower][~np.isnan(ses.firstlaserarray[lasered_contra_choices_delta_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_ipsi_delta_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_ipsi_choices_delta_upper][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices_delta_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                    prob['binned_spikes_laser_ipsi_delta_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_ipsi_choices_delta_lower][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices_delta_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                ## Value                 
                lasered_value_upper = np.intersect1d((np.where(ses.laserconsumed==1)), value_upper)
                lasered_value_lower = np.intersect1d((np.where(ses.laserconsumed==1)), value_lower)
                rewarded_value_upper = np.intersect1d((np.where(ses.outcome==1)), value_upper)
                rewarded_value_lower = np.intersect1d((np.where(ses.outcome==1)), value_lower)
                errored_value_upper = np.intersect1d((np.where(ses.outcome==0)), value_upper)
                errored_value_lower = np.intersect1d((np.where(ses.outcome==0)), value_lower)
                if  np.nanmean(ses.probe[j].channel_hem)>0.5:
                    lasered_contra_choices_value_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), value_upper))
                    lasered_ipsi_choices_value_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), value_upper))
                    ipsi_value_upper = np.intersect1d((np.where(ses.choice==1)), value_upper)
                    contra_value_upper = np.intersect1d((np.where(ses.choice==-1)), value_upper)
                    lasered_contra_choices_value_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), value_lower))
                    lasered_ipsi_choices_value_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), value_lower))
                    ipsi_value_lower = np.intersect1d((np.where(ses.choice==1)), value_lower)
                    contra_value_lower = np.intersect1d((np.where(ses.choice==-1)), value_lower)                    
                else:
                    lasered_contra_choices_value_upper = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), value_upper))
                    lasered_ipsi_choices_value_upper = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), value_upper))
                    ipsi_value_upper = np.intersect1d((np.where(ses.choice==-1)), value_upper)
                    contra_value_upper = np.intersect1d((np.where(ses.choice==1)), value_upper)
                    lasered_contra_choices_value_lower = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), value_lower))
                    lasered_ipsi_choices_value_lower = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), value_lower))
                    ipsi_value_lower = np.intersect1d((np.where(ses.choice==-1)), value_lower)
                    contra_value_lower = np.intersect1d((np.where(ses.choice==1)), value_lower)  

                empty_value_selections = any([len(sel)<=5 for sel in [
                lasered_contra_choices_value_upper, lasered_ipsi_choices_value_upper, 
                ipsi_value_upper, contra_value_upper, lasered_contra_choices_value_lower, 
                lasered_ipsi_choices_value_lower, ipsi_value_lower, contra_value_lower]])

                prob['binned_spikes_gocue_value_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                            selection, ses.goCue_trigger_times[value_upper][~np.isnan(ses.goCue_trigger_times[value_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_gocue_value_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                            selection, ses.goCue_trigger_times[value_lower][~np.isnan(ses.goCue_trigger_times[value_lower])])]
                prob['binned_spikes_laser_value_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_value_upper][~np.isnan(ses.firstlaserarray[lasered_value_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                prob['binned_spikes_laser_value_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_value_lower][~np.isnan(ses.firstlaserarray[lasered_value_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)            
                prob['binned_spikes_outcome_reward_value_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[rewarded_value_upper][~np.isnan(ses.response_times[rewarded_value_upper])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_reward_value_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[rewarded_value_lower][~np.isnan(ses.response_times[rewarded_value_lower])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error_value_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[errored_value_upper][~np.isnan(ses.response_times[errored_value_upper])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error_value_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                    selection, ses.response_times[errored_value_lower][~np.isnan(ses.response_times[errored_value_lower])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                
                if empty_value_selections==False:              
                    prob['binned_spikes_choice_ipsi_value_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[ipsi_value_upper][~np.isnan(ses.first_move[ipsi_value_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra_value_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[contra_value_upper][~np.isnan(ses.first_move[contra_value_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_ipsi_value_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[ipsi_value_lower][~np.isnan(ses.first_move[ipsi_value_lower])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra_value_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[contra_value_lower][~np.isnan(ses.first_move[contra_value_lower])])] #np.array (n_align_times, n_clusters, n_bins)                 
                    prob['binned_spikes_laser_contra_value_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_contra_choices_value_upper][~np.isnan(ses.firstlaserarray[lasered_contra_choices_value_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_contra_value_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_contra_choices_value_lower][~np.isnan(ses.firstlaserarray[lasered_contra_choices_value_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_ipsi_value_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_ipsi_choices_value_upper][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices_value_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                    prob['binned_spikes_laser_ipsi_value_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_ipsi_choices_value_lower][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices_value_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)    

                ## Decision times                 
                lasered_decision_times_upper = np.intersect1d((np.where(ses.laserconsumed==1)), decision_times_upper)
                lasered_decision_times_lower = np.intersect1d((np.where(ses.laserconsumed==1)), decision_times_lower)
                rewarded_decision_times_upper = np.intersect1d((np.where(ses.outcome==1)), decision_times_upper)
                rewarded_decision_times_lower = np.intersect1d((np.where(ses.outcome==1)), decision_times_lower)
                errored_decision_times_upper = np.intersect1d((np.where(ses.outcome==0)), decision_times_upper)
                errored_decision_times_lower = np.intersect1d((np.where(ses.outcome==0)), decision_times_lower)
                if  np.nanmean(ses.probe[j].channel_hem)>0.5:
                    lasered_contra_choices_decision_times_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), decision_times_upper))
                    lasered_ipsi_choices_decision_times_upper = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), decision_times_upper))
                    ipsi_decision_times_upper = np.intersect1d((np.where(ses.choice==1)), decision_times_upper)
                    contra_decision_times_upper = np.intersect1d((np.where(ses.choice==-1)), decision_times_upper)
                    lasered_contra_choices_decision_times_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), decision_times_lower))
                    lasered_ipsi_choices_decision_times_lower = reduce(np.intersect1d,(np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), decision_times_lower))
                    ipsi_decision_times_lower = np.intersect1d((np.where(ses.choice==1)), decision_times_lower)
                    contra_decision_times_lower = np.intersect1d((np.where(ses.choice==-1)), decision_times_lower)                    
                else:
                    lasered_contra_choices_decision_times_upper = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), decision_times_upper))
                    lasered_ipsi_choices_decision_times_upper = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), decision_times_upper))
                    ipsi_decision_times_upper = np.intersect1d((np.where(ses.choice==-1)), decision_times_upper)
                    contra_decision_times_upper = np.intersect1d((np.where(ses.choice==1)), decision_times_upper)
                    lasered_contra_choices_decision_times_lower = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==1)), decision_times_lower))
                    lasered_ipsi_choices_decision_times_lower = reduce(np.intersect1d, (np.where(ses.laserconsumed==1), (np.where(ses.choice==-1)), decision_times_lower))
                    ipsi_decision_times_lower = np.intersect1d((np.where(ses.choice==-1)), decision_times_lower)
                    contra_decision_times_lower = np.intersect1d((np.where(ses.choice==1)), decision_times_lower)  

                empty_decision_times_selections = any([len(sel)<=5 for sel in [
                lasered_contra_choices_decision_times_upper, lasered_ipsi_choices_decision_times_upper, 
                ipsi_decision_times_upper, contra_decision_times_upper, lasered_contra_choices_decision_times_lower, 
                lasered_ipsi_choices_decision_times_lower, ipsi_decision_times_lower, contra_decision_times_lower]])

                prob['binned_spikes_gocue_decision_times_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                            selection, ses.goCue_trigger_times[decision_times_upper][~np.isnan(ses.goCue_trigger_times[decision_times_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_gocue_decision_times_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                                            selection, ses.goCue_trigger_times[decision_times_lower][~np.isnan(ses.goCue_trigger_times[decision_times_lower])])]
                prob['binned_spikes_laser_decision_times_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_decision_times_upper][~np.isnan(ses.firstlaserarray[lasered_decision_times_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                prob['binned_spikes_laser_decision_times_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_decision_times_lower][~np.isnan(ses.firstlaserarray[lasered_decision_times_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)            
                prob['binned_spikes_outcome_reward_decision_times_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[rewarded_decision_times_upper][~np.isnan(ses.response_times[rewarded_decision_times_upper])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_reward_decision_times_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[rewarded_decision_times_lower][~np.isnan(ses.response_times[rewarded_decision_times_lower])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error_decision_times_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection, 
                    ses.response_times[errored_decision_times_upper][~np.isnan(ses.response_times[errored_decision_times_upper])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                prob['binned_spikes_outcome_error_decision_times_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, 
                    selection, ses.response_times[errored_decision_times_lower][~np.isnan(ses.response_times[errored_decision_times_lower])], post_time=2.0)] #np.array (n_align_times, n_clusters, n_bins)
                
                if empty_decision_times_selections==False:              
                    prob['binned_spikes_choice_ipsi_decision_times_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[ipsi_decision_times_upper][~np.isnan(ses.first_move[ipsi_decision_times_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra_decision_times_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[contra_decision_times_upper][~np.isnan(ses.first_move[contra_decision_times_upper])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_ipsi_decision_times_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[ipsi_decision_times_lower][~np.isnan(ses.first_move[ipsi_decision_times_lower])])] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_choice_contra_decision_times_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,
                                                            ses.first_move[contra_decision_times_lower][~np.isnan(ses.first_move[contra_decision_times_lower])])] #np.array (n_align_times, n_clusters, n_bins)                 
                    prob['binned_spikes_laser_contra_decision_times_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_contra_choices_decision_times_upper][~np.isnan(ses.firstlaserarray[lasered_contra_choices_decision_times_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_contra_decision_times_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_contra_choices_decision_times_lower][~np.isnan(ses.firstlaserarray[lasered_contra_choices_decision_times_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)
                    prob['binned_spikes_laser_ipsi_decision_times_upper'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_ipsi_choices_decision_times_upper][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices_decision_times_upper])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)                
                    prob['binned_spikes_laser_ipsi_decision_times_lower'] = [get_binned_spikes(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection,  
                                                            ses.firstlaserarray[lasered_ipsi_choices_decision_times_lower][~np.isnan(ses.firstlaserarray[lasered_ipsi_choices_decision_times_lower])], post_time=3.0)] #np.array (n_align_times, n_clusters, n_bins)    

                ## 
                prob['cluster_selection'] = [selection]
                prob['value_upper_trials']= value_upper
                prob['value_lower_trials']= value_lower
                prob['qchosen_upper_trials']= qchosen_upper
                prob['qchosen_lower_trials']= qchosen_lower
                prob['deltaq_upper_trials']= delta_upper
                prob['delta_lower_trials']= delta_lower
                prob['choice']= [ses.choice]
                prob['outcome']= [ses.outcome]
                prob['qchosen']= [qchosen]
                prob['delta']= [deltaq]
                prob['state_value']= [value]
                prob['neurons_zscore_params'] = [get_norm_neurons(ses.probe[j].spike_times, ses.probe[j].spike_clusters, selection)]
                prob['date'] = ses.date
                prob['ses'] = ses.ses
                prob['hem'] = np.nanmean(ses.probe[j].channel_hem)>0.5
                prob['probe'] = j 
                prob['mouse'] = ses.mouse
                prob['id'] =  ses.mouse + '_' + ses.date + '_' + ses.ses + '_' + str(j)
                region_summary = pd.concat([region_summary,prob])
    region_summary['region'] = roi
    return region_summary.reset_index()

def xval_mean(x, even=True):

    if even==True:
        return np.nanmean(x[::2,:,:], axis=0)
    else:
        return np.nanmean(x[1::2,:,:], axis=0)

def pca_mean(x):
    return np.nanmean(x, axis=0)

def plot_region_psth(region_df, plot=False):
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True))
    contra_choice_psth = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=True))
    ipsi_choice_psth = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True))
    laser_psth = np.concatenate(region_df.binned_spikes_laser.apply(xval_mean,even=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    psths = np.concatenate([go_cue_psth,contra_choice_psth,ipsi_choice_psth,reward_psth, laser_psth, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    #psths_fr_z = zscore(psths_fr,axis=1) # z-score
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
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
    psths_plot = np.concatenate([go_cue_psth_plot,contra_choice_psth_plot,ipsi_choice_psth_plot,
                                reward_psth_plot, laser_psth_plot, error_psth_plot], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    #psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
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
    sns.heatmap(xs_sorted,center=0,vmin=-2,vmax=2, cmap="seismic", cbar=False)    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','Choice Contra','Choice Ipsi','CS+', 'DA stim', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary.pdf')
        plt.close()

def plot_region_psth_dachoice_interaction(region_df, plot=False):
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True))
    contra_choice_psth = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=True))
    ipsi_choice_psth = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True))
    error_psth_contra = np.concatenate(region_df.binned_spikes_error_contra.apply(xval_mean,even=True))
    error_psth_ipsi = np.concatenate(region_df.binned_spikes_error_ipsi.apply(xval_mean,even=True))
    laser_psth_contra = np.concatenate(region_df.binned_spikes_laser_contra.apply(xval_mean,even=True))
    laser_psth_ipsi = np.concatenate(region_df.binned_spikes_laser_ipsi.apply(xval_mean,even=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,contra_choice_psth,ipsi_choice_psth,reward_psth, 
                            laser_psth_contra, laser_psth_ipsi, error_psth_contra, error_psth_ipsi], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    #psths_fr_z = zscore(psths_fr,axis=1) # z-score
    psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=False))
    contra_choice_psth_plot = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=False))
    ipsi_choice_psth_plot = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=False))
    reward_psth_plot = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=False))
    error_psth_contra_plot = np.concatenate(region_df.binned_spikes_error_contra.apply(xval_mean,even=False))
    error_psth_ipsi_plot = np.concatenate(region_df.binned_spikes_error_ipsi.apply(xval_mean,even=False))    
    laser_psth_contra_plot = np.concatenate(region_df.binned_spikes_laser_contra.apply(xval_mean,even=False))
    laser_psth_ipsi_plot = np.concatenate(region_df.binned_spikes_laser_ipsi.apply(xval_mean,even=False))
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot,contra_choice_psth_plot,ipsi_choice_psth_plot,reward_psth_plot, 
                laser_psth_contra_plot,laser_psth_ipsi_plot,
                error_psth_contra_plot, error_psth_ipsi_plot], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    #psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],contra_choice_psth.shape[1],ipsi_choice_psth.shape[1],
                            reward_psth.shape[1],laser_psth_contra.shape[1], laser_psth_ipsi.shape[1], 
                             error_psth_contra.shape[1], error_psth_ipsi.shape[1], laser_psth_ipsi.shape[1],
                            error_psth_ipsi.shape[1]])
    # add diff plots
    deltalaser = xs_sorted[:,breakpoints[3]:breakpoints[4]] - xs_sorted[:,breakpoints[4]:breakpoints[5]]
    deltaerror = xs_sorted[:,breakpoints[5]:breakpoints[6]] - xs_sorted[:,breakpoints[6]:breakpoints[7]]
    xs_sorted = np.concatenate([xs_sorted,deltalaser,deltaerror], axis=1)
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-2,vmax=2, cmap="seismic", cbar=False)    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','Choice Contra','Choice Ipsi','CS+', 'DAContra',  'DAIpsi',
                                'CS- Contra', 'CS- Ipsi', 'Delta DA', 'Delta CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_da_interaction.pdf')
        plt.close()

def plot_region_psth_dachoice_interaction_diff(region_df, plot=False):
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True))
    contra_choice_psth = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=True))
    ipsi_choice_psth = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True))
    laser_psth_ipsi = np.concatenate(region_df.binned_spikes_laser_ipsi.apply(xval_mean,even=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,contra_choice_psth,ipsi_choice_psth,reward_psth, laser_psth_ipsi, error_psth], axis=1)
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
    laser_psth_contra_plot = np.concatenate(region_df.binned_spikes_laser_contra.apply(xval_mean,even=False))
    laser_psth_ipsi_plot = np.concatenate(region_df.binned_spikes_laser_ipsi.apply(xval_mean,even=False))
    laser_psth_diff_plot = laser_psth_ipsi_plot-laser_psth_contra_plot
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot,contra_choice_psth_plot,ipsi_choice_psth_plot,reward_psth_plot, laser_psth_diff_plot, error_psth_plot], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],contra_choice_psth.shape[1],ipsi_choice_psth.shape[1],reward_psth.shape[1],laser_psth_diff_plot.shape[1], error_psth.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-4,vmax=4, cmap="seismic", cbar=False)    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','Choice Contra','Choice Ipsi','CS+', 'DAIpsi-Contra', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_da_interaction_diff.pdf')
        plt.close()

def plot_region_psth_dachoice_interaction_trial_qchosen(region_df, plot=False):
    no_missing_comb_qchosen_idx = np.where(region_df.binned_spikes_choice_contra_qchosen_lower.isnull()==False)[0]
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    contra_choice_psth = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    ipsi_choice_psth = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    laser_psth_contra = np.concatenate(region_df.binned_spikes_laser_contra.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    laser_psth_ipsi = np.concatenate(region_df.binned_spikes_laser_ipsi.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,contra_choice_psth,ipsi_choice_psth,reward_psth, laser_psth_contra, laser_psth_ipsi, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    psths_fr_z = zscore(psths_fr,axis=1) # z-score
    #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_gocue_qchosen_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    contra_choice_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_choice_contra_qchosen_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    ipsi_choice_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_choice_ipsi_qchosen_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    reward_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_outcome_reward_qchosen_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    error_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_outcome_error_qchosen_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    laser_psth_contra_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_laser_contra_qchosen_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    laser_psth_ipsi_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_laser_ipsi_qchosen_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    go_cue_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_gocue_qchosen_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    contra_choice_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_choice_contra_qchosen_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    ipsi_choice_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_choice_ipsi_qchosen_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    reward_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_outcome_reward_qchosen_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    error_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_outcome_error_qchosen_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    laser_psth_contra_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_laser_contra_qchosen_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    laser_psth_ipsi_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_laser_ipsi_qchosen_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_qchosen_idx].reset_index(drop=True))

    go_cue_psth_plot_qchosen_diff = go_cue_psth_plot_qchosen_upper - go_cue_psth_plot_qchosen_lower
    contra_choice_psth_plot_qchosen_diff = contra_choice_psth_plot_qchosen_upper - contra_choice_psth_plot_qchosen_lower
    ipsi_choice_psth_plot_qchosen_diff = ipsi_choice_psth_plot_qchosen_upper - ipsi_choice_psth_plot_qchosen_lower
    reward_psth_plot_qchosen_diff = reward_psth_plot_qchosen_upper - reward_psth_plot_qchosen_lower
    error_psth_plot_qchosen_diff = error_psth_plot_qchosen_upper -  error_psth_plot_qchosen_lower
    laser_psth_contra_plot_qchosen_diff = laser_psth_contra_plot_qchosen_upper - laser_psth_contra_plot_qchosen_lower
    laser_psth_ipsi_plot_qchosen_diff = laser_psth_ipsi_plot_qchosen_upper - laser_psth_ipsi_plot_qchosen_lower
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot_qchosen_diff,contra_choice_psth_plot_qchosen_diff,
                ipsi_choice_psth_plot_qchosen_diff,reward_psth_plot_qchosen_diff, 
                laser_psth_contra_plot_qchosen_diff,laser_psth_ipsi_plot_qchosen_diff, 
                error_psth_plot_qchosen_diff], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],contra_choice_psth.shape[1],ipsi_choice_psth.shape[1],reward_psth.shape[1],laser_psth_contra.shape[1], laser_psth_ipsi.shape[1], error_psth.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-4,vmax=4, cmap="seismic")    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','Choice Contra','Choice Ipsi','CS+', 'DAContra',  'DAIpsi', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_qchosen_da_interaction.pdf')
        plt.close()

def plot_region_psth_dachoice_interaction_trial_delta(region_df, plot=False):
    
    ### laser_psth_contra_plot_delta_lower / binned_spikes_laser_contra_delta_lower is the problem, it has nan
    
    no_missing_comb_delta_idx = np.where(region_df.binned_spikes_choice_contra_delta_lower.isnull()==False)[0]
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    contra_choice_psth = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    ipsi_choice_psth = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    laser_psth_contra = np.concatenate(region_df.binned_spikes_laser_contra.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    laser_psth_ipsi = np.concatenate(region_df.binned_spikes_laser_ipsi.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,contra_choice_psth,ipsi_choice_psth,reward_psth, laser_psth_contra, laser_psth_ipsi, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    psths_fr_z = zscore(psths_fr,axis=1) # z-score
    #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_gocue_delta_upper[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    contra_choice_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_choice_contra_delta_upper[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    ipsi_choice_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_choice_ipsi_delta_upper[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_outcome_reward_delta_upper[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_outcome_error_delta_upper[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_contra_plot_delta_upper = np.concatenate(region_df.binned_spikes_laser_contra_delta_upper[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_ipsi_plot_delta_upper = np.concatenate(region_df.binned_spikes_laser_ipsi_delta_upper[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    go_cue_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_gocue_delta_lower[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    contra_choice_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_choice_contra_delta_lower[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    ipsi_choice_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_choice_ipsi_delta_lower[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_outcome_reward_delta_lower[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_outcome_error_delta_lower[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_contra_plot_delta_lower = np.concatenate(region_df.binned_spikes_laser_contra_delta_lower[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_ipsi_plot_delta_lower = np.concatenate(region_df.binned_spikes_laser_ipsi_delta_lower[no_missing_comb_delta_idx].apply(xval_mean,even=False).reset_index(drop=True))

    go_cue_psth_plot_delta_diff = go_cue_psth_plot_delta_upper - go_cue_psth_plot_delta_lower
    contra_choice_psth_plot_delta_diff = contra_choice_psth_plot_delta_upper - contra_choice_psth_plot_delta_lower
    ipsi_choice_psth_plot_delta_diff = ipsi_choice_psth_plot_delta_upper - ipsi_choice_psth_plot_delta_lower
    reward_psth_plot_delta_diff = reward_psth_plot_delta_upper - reward_psth_plot_delta_lower
    error_psth_plot_delta_diff = error_psth_plot_delta_upper -  error_psth_plot_delta_lower
    laser_psth_contra_plot_delta_diff = laser_psth_contra_plot_delta_upper - laser_psth_contra_plot_delta_lower
    laser_psth_ipsi_plot_delta_diff = laser_psth_ipsi_plot_delta_upper - laser_psth_ipsi_plot_delta_lower
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot_delta_diff,contra_choice_psth_plot_delta_diff,
                ipsi_choice_psth_plot_delta_diff,reward_psth_plot_delta_diff, 
                laser_psth_contra_plot_delta_diff,laser_psth_ipsi_plot_delta_diff, 
                error_psth_plot_delta_diff], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],contra_choice_psth.shape[1],ipsi_choice_psth.shape[1],reward_psth.shape[1],laser_psth_contra.shape[1], laser_psth_ipsi.shape[1], error_psth.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted, center=0, vmin=-4,vmax=4, cmap="seismic")    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','Choice Contra','Choice Ipsi','CS+', 'DAContra',  'DAIpsi', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_delta_da_interaction.pdf')
        plt.close()

def plot_region_psth_latency(region_df, plot=False):
    no_missing_comb_decision_times_idx = np.where(region_df.binned_spikes_gocue_decision_times_upper.isnull()==False)[0]
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True)[no_missing_comb_decision_times_idx].reset_index(drop=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True)[no_missing_comb_decision_times_idx].reset_index(drop=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True)[no_missing_comb_decision_times_idx].reset_index(drop=True))
    laser_psth = np.concatenate(region_df.binned_spikes_laser.apply(xval_mean,even=True)[no_missing_comb_decision_times_idx].reset_index(drop=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,reward_psth, laser_psth, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    psths_fr_z = zscore(psths_fr,axis=1) # z-score
    #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot_decision_times_upper = np.concatenate(region_df.binned_spikes_gocue_decision_times_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_decision_times_upper = np.concatenate(region_df.binned_spikes_outcome_reward_decision_times_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_decision_times_upper = np.concatenate(region_df.binned_spikes_outcome_error_decision_times_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_plot_decision_times_upper = np.concatenate(region_df.binned_spikes_laser_decision_times_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    go_cue_psth_plot_decision_times_lower = np.concatenate(region_df.binned_spikes_gocue_decision_times_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_decision_times_lower = np.concatenate(region_df.binned_spikes_outcome_reward_decision_times_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_decision_times_lower = np.concatenate(region_df.binned_spikes_outcome_error_decision_times_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_plot_decision_times_lower = np.concatenate(region_df.binned_spikes_laser_decision_times_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))

    go_cue_psth_plot_decision_times_diff = go_cue_psth_plot_decision_times_upper - go_cue_psth_plot_decision_times_lower
    reward_psth_plot_decision_times_diff = reward_psth_plot_decision_times_upper - reward_psth_plot_decision_times_lower
    error_psth_plot_decision_times_diff = error_psth_plot_decision_times_upper -  error_psth_plot_decision_times_lower
    laser_psth_plot_decision_times_diff = laser_psth_plot_decision_times_upper - laser_psth_plot_decision_times_lower
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot_decision_times_diff,
                reward_psth_plot_decision_times_diff, 
                laser_psth_plot_decision_times_diff, 
                error_psth_plot_decision_times_diff], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],reward_psth.shape[1],laser_psth.shape[1], error_psth.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-4,vmax=4, cmap="seismic")    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','CS+', 'DA', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_decision_times.pdf')
        plt.close()


def plot_region_psth_qchosen(region_df, plot=False):
    no_missing_comb_qchosen_idx = np.where(region_df.binned_spikes_gocue_qchosen_upper.isnull()==False)[0]
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    laser_psth = np.concatenate(region_df.binned_spikes_laser.apply(xval_mean,even=True)[no_missing_comb_qchosen_idx].reset_index(drop=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,reward_psth, laser_psth, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    psths_fr_z = zscore(psths_fr,axis=1) # z-score
    #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_gocue_qchosen_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_outcome_reward_qchosen_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_outcome_error_qchosen_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_plot_qchosen_upper = np.concatenate(region_df.binned_spikes_laser_qchosen_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    go_cue_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_gocue_qchosen_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_outcome_reward_qchosen_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_outcome_error_qchosen_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_plot_qchosen_lower = np.concatenate(region_df.binned_spikes_laser_qchosen_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))

    go_cue_psth_plot_qchosen_diff = go_cue_psth_plot_qchosen_upper - go_cue_psth_plot_qchosen_lower
    reward_psth_plot_qchosen_diff = reward_psth_plot_qchosen_upper - reward_psth_plot_qchosen_lower
    error_psth_plot_qchosen_diff = error_psth_plot_qchosen_upper -  error_psth_plot_qchosen_lower
    laser_psth_plot_qchosen_diff = laser_psth_plot_qchosen_upper - laser_psth_plot_qchosen_lower
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot_qchosen_diff,
                reward_psth_plot_qchosen_diff, 
                laser_psth_plot_qchosen_diff, 
                error_psth_plot_qchosen_diff], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],reward_psth.shape[1],laser_psth.shape[1], error_psth.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-4,vmax=4, cmap="seismic")    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','CS+', 'DA', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_qchosen.pdf')
        plt.close()

def plot_region_psth_delta(region_df, plot=False):
    no_missing_comb_delta_idx = np.where(region_df.binned_spikes_gocue_delta_upper.isnull()==False)[0]
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    laser_psth = np.concatenate(region_df.binned_spikes_laser.apply(xval_mean,even=True)[no_missing_comb_delta_idx].reset_index(drop=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,reward_psth, laser_psth, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    psths_fr_z = zscore(psths_fr,axis=1) # z-score
    #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_gocue_delta_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_outcome_reward_delta_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_outcome_error_delta_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_plot_delta_upper = np.concatenate(region_df.binned_spikes_laser_delta_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    go_cue_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_gocue_delta_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_outcome_reward_delta_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_outcome_error_delta_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_plot_delta_lower = np.concatenate(region_df.binned_spikes_laser_delta_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))

    go_cue_psth_plot_delta_diff = go_cue_psth_plot_delta_upper - go_cue_psth_plot_delta_lower
    reward_psth_plot_delta_diff = reward_psth_plot_delta_upper - reward_psth_plot_delta_lower
    error_psth_plot_delta_diff = error_psth_plot_delta_upper -  error_psth_plot_delta_lower
    laser_psth_plot_delta_diff = laser_psth_plot_delta_upper - laser_psth_plot_delta_lower
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot_delta_diff,
                reward_psth_plot_delta_diff, 
                laser_psth_plot_delta_diff, 
                error_psth_plot_delta_diff], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],reward_psth.shape[1],laser_psth.shape[1], error_psth.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-4,vmax=4, cmap="seismic")    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','CS+', 'DA', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_delta.pdf')
        plt.close()

def plot_region_psth_value(region_df, plot=False):
    no_missing_comb_value_idx = np.where(region_df.binned_spikes_gocue_value_upper.isnull()==False)[0]
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    laser_psth = np.concatenate(region_df.binned_spikes_laser.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,reward_psth, laser_psth, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    psths_fr_z = zscore(psths_fr,axis=1) # z-score
    #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_gocue_value_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_outcome_reward_value_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_outcome_error_value_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_laser_value_upper.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    go_cue_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_gocue_value_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    reward_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_outcome_reward_value_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    error_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_outcome_error_value_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))
    laser_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_laser_value_lower.dropna().apply(xval_mean,even=False).reset_index(drop=True))

    go_cue_psth_plot_value_diff = go_cue_psth_plot_value_upper - go_cue_psth_plot_value_lower
    reward_psth_plot_value_diff = reward_psth_plot_value_upper - reward_psth_plot_value_lower
    error_psth_plot_value_diff = error_psth_plot_value_upper -  error_psth_plot_value_lower
    laser_psth_plot_value_diff = laser_psth_plot_value_upper - laser_psth_plot_value_lower
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot_value_diff,
                reward_psth_plot_value_diff, 
                laser_psth_plot_value_diff, 
                error_psth_plot_value_diff], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth_plot_value_diff.shape[1],reward_psth_plot_value_diff.shape[1],laser_psth_plot_value_diff.shape[1], error_psth_plot_value_diff.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-4,vmax=4, cmap="seismic")    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','CS+', 'DA', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_value.pdf')
        plt.close()

def plot_region_psth_dachoice_interaction_trial_value(region_df, plot=False):
    no_missing_comb_value_idx = np.where(region_df.binned_spikes_choice_contra_value_lower.isnull()==False)[0]
    go_cue_psth = np.concatenate(region_df.binned_spikes_gocue.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    contra_choice_psth = np.concatenate(region_df.binned_spikes_choice_contra.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    ipsi_choice_psth = np.concatenate(region_df.binned_spikes_choice_ipsi.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    reward_psth = np.concatenate(region_df.binned_spikes_outcome_reward.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    error_psth = np.concatenate(region_df.binned_spikes_outcome_error.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    laser_psth_contra = np.concatenate(region_df.binned_spikes_laser_contra.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    laser_psth_ipsi = np.concatenate(region_df.binned_spikes_laser_ipsi.apply(xval_mean,even=True)[no_missing_comb_value_idx].reset_index(drop=True))
    #iti_psth = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=True))
    neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
    psths = np.concatenate([go_cue_psth,contra_choice_psth,ipsi_choice_psth,reward_psth, laser_psth_contra, laser_psth_ipsi, error_psth], axis=1)
    psths_fr = psths/0.025 # Transform to firing rates
    psths_fr_z = zscore(psths_fr,axis=1) # z-score
    #psths_fr_z = zscore_set_params(psths_fr,neurons_zscore_params) # z-score
    # crossvalidated sorting
    order = np.argmax(psths_fr_z, 1)
    # plotting means
    go_cue_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_gocue_value_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    contra_choice_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_choice_contra_value_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    ipsi_choice_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_choice_ipsi_value_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    reward_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_outcome_reward_value_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    error_psth_plot_value_upper = np.concatenate(region_df.binned_spikes_outcome_error_value_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    laser_psth_contra_plot_value_upper = np.concatenate(region_df.binned_spikes_laser_contra_value_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    laser_psth_ipsi_plot_value_upper = np.concatenate(region_df.binned_spikes_laser_ipsi_value_upper.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    go_cue_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_gocue_value_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    contra_choice_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_choice_contra_value_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    ipsi_choice_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_choice_ipsi_value_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    reward_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_outcome_reward_value_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    error_psth_plot_value_lower = np.concatenate(region_df.binned_spikes_outcome_error_value_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    laser_psth_contra_plot_value_lower = np.concatenate(region_df.binned_spikes_laser_contra_value_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))
    laser_psth_ipsi_plot_value_lower = np.concatenate(region_df.binned_spikes_laser_ipsi_value_lower.dropna().apply(xval_mean,even=False)[no_missing_comb_value_idx].reset_index(drop=True))

    go_cue_psth_plot_value_diff = go_cue_psth_plot_value_upper - go_cue_psth_plot_value_lower
    contra_choice_psth_plot_value_diff = contra_choice_psth_plot_value_upper - contra_choice_psth_plot_value_lower
    ipsi_choice_psth_plot_value_diff = ipsi_choice_psth_plot_value_upper - ipsi_choice_psth_plot_value_lower
    reward_psth_plot_value_diff = reward_psth_plot_value_upper - reward_psth_plot_value_lower
    error_psth_plot_value_diff = error_psth_plot_value_upper -  error_psth_plot_value_lower
    laser_psth_contra_plot_value_diff = laser_psth_contra_plot_value_upper - laser_psth_contra_plot_value_lower
    laser_psth_ipsi_plot_value_diff = laser_psth_ipsi_plot_value_upper - laser_psth_ipsi_plot_value_lower
    #iti_psth_plot = np.concatenate(region_df.binned_spikes_iti_start.apply(xval_mean,even=False))
    psths_plot = np.concatenate([go_cue_psth_plot_value_diff,contra_choice_psth_plot_value_diff,
                ipsi_choice_psth_plot_value_diff,reward_psth_plot_value_diff, 
                laser_psth_contra_plot_value_diff,laser_psth_ipsi_plot_value_diff, 
                error_psth_plot_value_diff], axis=1)
    psths_fr_plot = psths_plot/0.025 # Transform to firing rates
    psths_fr_z_plot = zscore(psths_fr_plot,axis=1) # z-score
    #psths_fr_z_plot = zscore_set_params(psths_fr_plot,neurons_zscore_params) # z-score
    # Sort based on unused trials
    xs_sorted = psths_fr_z_plot[order.argsort(),:]
    breakpoints = np.cumsum([go_cue_psth.shape[1],contra_choice_psth_plot_value_diff.shape[1],
                ipsi_choice_psth_plot_value_diff.shape[1],reward_psth_plot_value_diff.shape[1],
                laser_psth_contra_plot_value_diff.shape[1], laser_psth_ipsi_plot_value_diff.shape[1], 
                error_psth_plot_value_diff.shape[1]])
    epoch_breakpoints = np.where(np.diff(np.digitize(np.argmax(psths_fr_z[order.argsort(),:], 1), breakpoints))!=0)[0] # Convoluted function for getting rows at which neurons start alignment to new epoch
    dist = 100*(np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/xs_sorted.shape[0])
    dist = np.round(dist,1)
    #dist_str = [str(d) + ' %' for d in dist]
    mid_points = np.insert(epoch_breakpoints, 0, 0)+ (np.diff(epoch_breakpoints, prepend=0, append=xs_sorted.shape[0])/2) #midpoints for putting ticks and labels
    # plotting means
    # Plot
    sns.heatmap(xs_sorted,center=0,vmin=-4,vmax=4, cmap="seismic")    
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]), ymin=0,ymax=len(xs_sorted), linestyles='solid', color='k', linewidth=0.5)
    plt.vlines(np.concatenate([np.zeros(1),breakpoints]) + 20, ymin=0,ymax=len(xs_sorted), linestyles='dashed', color='k', linewidth=0.5)
    plt.hlines(epoch_breakpoints, xmin=0,xmax=xs_sorted.shape[1], linestyles='solid', color='grey')
    plt.xticks(np.concatenate([np.zeros(1),breakpoints[:-1]]) + 20,['GoCue','Choice Contra','Choice Ipsi','CS+', 'DAContra',  'DAIpsi', 'CS-'], rotation=90)
    plt.yticks(mid_points,dist)
    plt.title(region_df['region'].unique()[0] + ' %s neurons' %xs_sorted.shape[0])
    if plot == True:
        plt.tight_layout()
        plt.savefig(region_df['region'].unique()[0]+'_summary_value_da_interaction.pdf')
        plt.close()

def plot_PCA_trajectory(region_df, trial_type_binned_spikes_1 = 'binned_spikes_laser_contra', 
                        trial_type_binned_spikes_2 =  'binned_spikes_laser_ipsi', pre_time=0.5,
                        zscoring=False, baseline_subtraction=False, use_pre_time=True, color='orange'):
    '''
    region_df: region specific dataframe containing psths
    epoch: binned spikes variables ([binned_spikes_gocue,binned_spikes_choice_ipsi,
        binned_spikes_choice_contra,binned_spikes_laser,binned_spikes_laser_contra,binned_spikes_laser_ipsi,
        binned_spikes_outcome_reward,binned_spikes_outcome_error])
    '''
    pre_time_bins = int(pre_time/0.025)
    if zscoring==True:
        neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
        data1 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True)), 
                neurons_zscore_params)
        data2 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True)), 
                neurons_zscore_params)    
    else:
        data1 = np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True))
        data2 = np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True))          

    if baseline_subtraction==True:
        data1_baseline = np.mean(data1[:,:pre_time_bins,], axis=1)
        data2_baseline = np.mean(data2[:,:pre_time_bins,], axis=1)
        data1_norm = data1.T - data1_baseline
        data2_norm = data2.T - data2_baseline
    else:
        data1_norm = data1.T
        data2_norm = data2.T

    z_t_averaged_data_with_baseline = np.concatenate([data1_norm,data2_norm])
    z_t_averaged_data_with_baseline = z_t_averaged_data_with_baseline[:,np.where(~np.isnan(np.mean(z_t_averaged_data_with_baseline,axis=0)))[0]]
    z_t_averaged_data = np.concatenate([data1_norm[pre_time_bins:,:],data2_norm[pre_time_bins:,:]])
    z_t_averaged_data = z_t_averaged_data[:,np.where(~np.isnan(np.mean(z_t_averaged_data,axis=0)))[0]]
    


    pca = PCA()

    if use_pre_time==True:
        x_all = pca.fit_transform(z_t_averaged_data_with_baseline)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        pca2d = PCA(n_components=2)
        pca2d.fit(z_t_averaged_data_with_baseline)
    else:
        x_all = pca.fit_transform(z_t_averaged_data)
        explained_variance = np.cumsum(pca.explained_variance_ratio_)
        pca2d = PCA(n_components=2)
        pca2d.fit(z_t_averaged_data)

    transformed_z_t_averaged_data = pca2d.transform(z_t_averaged_data_with_baseline)
    gradient = np.arange(int(transformed_z_t_averaged_data.shape[0]/2))

    plt.plot(transformed_z_t_averaged_data[:int(transformed_z_t_averaged_data.shape[0]/2),0], 
        transformed_z_t_averaged_data[:int(transformed_z_t_averaged_data.shape[0]/2),1], color=color, zorder=0)
    plt.plot(transformed_z_t_averaged_data[int(transformed_z_t_averaged_data.shape[0]/2):int(transformed_z_t_averaged_data.shape[0]),0], 
        transformed_z_t_averaged_data[int(transformed_z_t_averaged_data.shape[0]/2):int(transformed_z_t_averaged_data.shape[0]),1],
                color=color, zorder=1)
    plt.scatter(transformed_z_t_averaged_data[int(transformed_z_t_averaged_data.shape[0]/2):int(transformed_z_t_averaged_data.shape[0]),0], 
        transformed_z_t_averaged_data[int(transformed_z_t_averaged_data.shape[0]/2):int(transformed_z_t_averaged_data.shape[0]),1],
                c=color, alpha=0.35, marker='o', s=50, vmin=-100, vmax=100, zorder=2)
    plt.scatter(transformed_z_t_averaged_data[:int(transformed_z_t_averaged_data.shape[0]/2),0], 
        transformed_z_t_averaged_data[:int(transformed_z_t_averaged_data.shape[0]/2),1],c=color,  marker='o', s=50, vmin=-100, vmax=100,zorder=3)

    plt.scatter(transformed_z_t_averaged_data[19,0], transformed_z_t_averaged_data[19,1],  color='k',  marker='o', s=50, vmin=-100, vmax=100,zorder=4)
    plt.scatter(transformed_z_t_averaged_data[int(transformed_z_t_averaged_data.shape[0]/2)+ 19,0], transformed_z_t_averaged_data[int(transformed_z_t_averaged_data.shape[0]/2) + 19,1],  color='k',  marker='o', s=50, vmin=-100, vmax=100,zorder=5)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(region_df['region'].unique()[0] + '\n %s Variance Explained' % np.round(explained_variance[1],2) + '\n %s Neurons' % z_t_averaged_data.shape[1])
    sns.despine()
    return explained_variance
    
def plot_explained_variance(explained_variance):
    '''
    pca: object contaning all the variance explained 
    '''
    plt.plot(explained_variance, color='k',  marker='o')
    sns.despine()
    plt.xlabel('PCs')
    plt.ylabel('Variance Explained')

def plot_time_varying_ED(region_df, trial_type_binned_spikes_1 = 'binned_spikes_laser_contra_qchosen_upper', 
                        trial_type_binned_spikes_2 =  'binned_spikes_laser_contra_qchosen_lower', pre_time=0.5,
                        zscoring=True, baseline_subtraction=False):
    '''
    region_df: region specific dataframe containing psths
    epoch: binned spikes variables ([binned_spikes_gocue,binned_spikes_choice_ipsi,
        binned_spikes_choice_contra,binned_spikes_laser,binned_spikes_laser_contra,binned_spikes_laser_ipsi,
        binned_spikes_outcome_reward,binned_spikes_outcome_error])
    '''
    pre_time_bins = int(pre_time/0.025)
    if zscoring==True:
        neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
        data1 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True)), 
                neurons_zscore_params)
        data2 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True)), 
                neurons_zscore_params)    
    else:
        data1 = np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True))
        data2 = np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True))          

    if baseline_subtraction==True:
        data1_baseline = np.mean(data1[:,:pre_time_bins,], axis=1)
        data2_baseline = np.mean(data2[:,:pre_time_bins,], axis=1)
        data1_norm = data1.T - data1_baseline
        data2_norm = data2.T - data2_baseline
    else:
        data1_norm = data1.T
        data2_norm = data2.T

    z_t_averaged_data_with_baseline = np.concatenate([data1_norm,data2_norm])
    z_t_averaged_data_with_baseline = z_t_averaged_data_with_baseline[:,np.where(~np.isnan(np.mean(z_t_averaged_data_with_baseline,axis=0)))[0]]
    # Calculate eucledian distance with dist = numpy.linalg.norm(a-b)
    tED = np.array([np.linalg.norm(z_t_averaged_data_with_baseline[i,:] - \
        z_t_averaged_data_with_baseline[i+int(z_t_averaged_data_with_baseline.shape[0]/2), :]) \
        for i in np.arange(int(z_t_averaged_data_with_baseline.shape[0]/2))])
    tED_norm = tED/np.sqrt(z_t_averaged_data_with_baseline.shape[1])
    plt.plot(tED_norm, color='k')
    plt.ylabel('Normalized Eucledian Distance')
    plt.xlabel('Time from first laser (ms)')
    plt.axvline(x = 20, color = 'r', linestyle='dashed')
    plt.xticks(np.arange(len(tED_norm))[::50],np.arange(-500,3000,25)[::50])
    plt.title(region_df['region'].unique()[0])

def plot_time_varying_ED_nd(region_df, trial_type_binned_spikes_1 = 'binned_spikes_laser_contra_qchosen_upper', 
                        trial_type_binned_spikes_2 =  'binned_spikes_laser_contra_qchosen_lower', pre_time=0.5,
                        nd=2, zscoring=False, baseline_subtraction=False, use_pre_time=True):
    '''
    region_df: region specific dataframe containing psths
    epoch: binned spikes variables ([binned_spikes_gocue,binned_spikes_choice_ipsi,
        binned_spikes_choice_contra,binned_spikes_laser,binned_spikes_laser_contra,binned_spikes_laser_ipsi,
        binned_spikes_outcome_reward,binned_spikes_outcome_error])
    '''
    pre_time_bins = int(pre_time/0.025)
    if zscoring==True:
        neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
        data1 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True)), 
                neurons_zscore_params)
        data2 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True)), 
                neurons_zscore_params)    
    else:
        data1 = np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True))
        data2 = np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True))          

    if baseline_subtraction==True:
        data1_baseline = np.mean(data1[:,:pre_time_bins,], axis=1)
        data2_baseline = np.mean(data2[:,:pre_time_bins,], axis=1)
        data1_norm = data1.T - data1_baseline
        data2_norm = data2.T - data2_baseline
    else:
        data1_norm = data1.T
        data2_norm = data2.T

    z_t_averaged_data_with_baseline = np.concatenate([data1_norm,data2_norm])
    z_t_averaged_data_with_baseline = z_t_averaged_data_with_baseline[:,np.where(~np.isnan(np.mean(z_t_averaged_data_with_baseline,axis=0)))[0]]

    if nd == 'max':
        pca_nd = PCA()
    elif nd == 'n_max': # PCs up to 95% variance explained
        pre_pca = PCA()
        x_all = pre_pca.fit_transform(z_t_averaged_data_with_baseline)
        explained_variance = np.cumsum(pre_pca.explained_variance_ratio_)
        nd = np.where(explained_variance>=0.90)[0][0]+1 # +1 to avoid 0 dimension due to python indexing
        pca_nd = PCA(n_components=nd)
    else:
        pca_nd = PCA(n_components=nd)

    if use_pre_time==True: 
        pca_nd.fit(z_t_averaged_data_with_baseline)
        transformed_z_t_averaged_data = pca_nd.transform(z_t_averaged_data_with_baseline) 

    else:
        z_t_averaged_data = np.concatenate([data1_norm[pre_time_bins:,:],data2_norm[pre_time_bins:,:]])
        z_t_averaged_data = z_t_averaged_data[:,np.where(~np.isnan(np.mean(z_t_averaged_data,axis=0)))[0]]
        pca_nd.fit(z_t_averaged_data)
        transformed_z_t_averaged_data = pca_nd.transform(z_t_averaged_data_with_baseline)
    
    tED = np.array([np.linalg.norm(transformed_z_t_averaged_data[i,:] - \
        transformed_z_t_averaged_data[i+int(transformed_z_t_averaged_data.shape[0]/2), :]) \
        for i in np.arange(int(transformed_z_t_averaged_data.shape[0]/2))])

    tED_norm = tED/np.sqrt(transformed_z_t_averaged_data.shape[1])
    plt.plot(tED_norm, color='k')
    plt.ylabel('Normalized Eucledian Distance')
    plt.xlabel('Time from epoch (ms)')
    plt.axvline(x = pre_time_bins, color = 'r', linestyle='dashed')
    plt.xticks(np.arange(len(tED_norm))[::25],np.arange(-1*(pre_time_bins*25),
                ((len(tED_norm)-pre_time_bins)*25),25)[::25])
    plt.title(region_df['region'].unique()[0] +'\n %s dim' %nd)

def plot_time_varying_ED_global(region_df, variables_1, variables_2, palette, pre_time=0.5,
                        nd=2, zscoring=False, baseline_subtraction=False, use_pre_time=True,
                        xlabels=None):
    '''
    region_df: region specific dataframe containing psths
    epoch: binned spikes variables ([binned_spikes_gocue,binned_spikes_choice_ipsi,
        binned_spikes_choice_contra,binned_spikes_laser,binned_spikes_laser_contra,binned_spikes_laser_ipsi,
        binned_spikes_outcome_reward,binned_spikes_outcome_error])
    '''
    for i in np.arange(len(variables_1)):
        trial_type_binned_spikes_1=variables_1[i]
        trial_type_binned_spikes_2=variables_2[i]
        pre_time_bins = int(pre_time/0.025)
        if zscoring==True:
            neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
            data1 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True)), 
                    neurons_zscore_params)
            data2 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True)), 
                    neurons_zscore_params)    
        else:
            data1 = np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True))
            data2 = np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True))          

        if baseline_subtraction==True:
            data1_baseline = np.mean(data1[:,:pre_time_bins,], axis=1)
            data2_baseline = np.mean(data2[:,:pre_time_bins,], axis=1)
            data1_norm = data1.T - data1_baseline
            data2_norm = data2.T - data2_baseline
        else:
            data1_norm = data1.T
            data2_norm = data2.T

        if nd == 'max':
            pca_nd = PCA()
        else:
            pca_nd = PCA(n_components=nd)

        z_t_averaged_data_with_baseline = np.concatenate([data1_norm,data2_norm])
        z_t_averaged_data_with_baseline = z_t_averaged_data_with_baseline[:,np.where(~np.isnan(np.mean(z_t_averaged_data_with_baseline,axis=0)))[0]]

        if use_pre_time==True: 
            pca_nd.fit(z_t_averaged_data_with_baseline)
            transformed_z_t_averaged_data = pca_nd.transform(z_t_averaged_data_with_baseline) 

        else:
            z_t_averaged_data = np.concatenate([data1_norm[pre_time_bins:,:],data2_norm[pre_time_bins:,:]])
            z_t_averaged_data = z_t_averaged_data[:,np.where(~np.isnan(np.mean(z_t_averaged_data,axis=0)))[0]]
            pca_nd.fit(z_t_averaged_data)
            transformed_z_t_averaged_data = pca_nd.transform(z_t_averaged_data_with_baseline)
        
        tED = np.array([np.linalg.norm(transformed_z_t_averaged_data[i,:] - \
            transformed_z_t_averaged_data[i+int(transformed_z_t_averaged_data.shape[0]/2), :]) \
            for i in np.arange(int(transformed_z_t_averaged_data.shape[0]/2))])

        tED_norm = tED/np.sqrt(transformed_z_t_averaged_data.shape[1])
        plt.plot(tED_norm, color=palette[i])
        plt.ylabel('Normalized Eucledian Distance')
        if xlabels is not None:
            plt.xlabel(xlabels)
        else:
            plt.xlabel('Time from epoch (ms)')
        plt.axvline(x = pre_time_bins, color = 'r', linestyle='dashed')
        plt.xticks(np.arange(len(tED_norm))[::25],np.arange(-1*(pre_time_bins*25),
                    ((len(tED_norm)-pre_time_bins)*25),25)[::25])
        plt.title(region_df['region'].unique()[0])


def barplot_ED(region_df_grouped, rois, trial_type_binned_spikes_1 = 'binned_spikes_laser_contra_qchosen_upper', 
                        trial_type_binned_spikes_2 =  'binned_spikes_laser_contra_qchosen_lower', pre_time=0.5,
                        nd=2, zscoring=False, baseline_subtraction=False, use_pre_time=True):


    ED_per_region = pd.DataFrame()
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]

        pre_time_bins = int(pre_time/0.025)
        if zscoring==True:
            neurons_zscore_params = np.concatenate(region_df.neurons_zscore_params)
            data1 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True)), 
                    neurons_zscore_params)
            data2 = zscore_set_params(np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True)), 
                    neurons_zscore_params)    
        else:
            data1 = np.concatenate(region_df[trial_type_binned_spikes_1].dropna().apply(pca_mean).reset_index(drop=True))
            data2 = np.concatenate(region_df[trial_type_binned_spikes_2].dropna().apply(pca_mean).reset_index(drop=True))          

        if baseline_subtraction==True:
            data1_baseline = np.mean(data1[:,:pre_time_bins,], axis=1)
            data2_baseline = np.mean(data2[:,:pre_time_bins,], axis=1)
            data1_norm = data1.T - data1_baseline
            data2_norm = data2.T - data2_baseline
        else:
            data1_norm = data1.T
            data2_norm = data2.T

        z_t_averaged_data_with_baseline = np.concatenate([data1_norm,data2_norm])
        z_t_averaged_data_with_baseline = z_t_averaged_data_with_baseline[:,np.where(~np.isnan(np.mean(z_t_averaged_data_with_baseline,axis=0)))[0]]

        if nd == 'max':
            pca_nd = PCA()
        elif nd == 'n_max': # PCs up to 95% variance explained
            pre_pca = PCA()
            x_all = pre_pca.fit_transform(z_t_averaged_data_with_baseline)
            explained_variance = np.cumsum(pre_pca.explained_variance_ratio_)
            nd = np.where(explained_variance>=0.90)[0][0]+1 # +1 to avoid 0 dimension due to python indexing
            pca_nd = PCA(n_components=nd)
        else:
            pca_nd = PCA(n_components=nd)

        if use_pre_time==True: 
            pca_nd.fit(z_t_averaged_data_with_baseline)
            transformed_z_t_averaged_data = pca_nd.transform(z_t_averaged_data_with_baseline) 

        else:
            z_t_averaged_data = np.concatenate([data1_norm[pre_time_bins:,:],data2_norm[pre_time_bins:,:]])
            z_t_averaged_data = z_t_averaged_data[:,np.where(~np.isnan(np.mean(z_t_averaged_data,axis=0)))[0]]
            pca_nd.fit(z_t_averaged_data)
            transformed_z_t_averaged_data = pca_nd.transform(z_t_averaged_data_with_baseline)
        
        tED = np.array([np.linalg.norm(transformed_z_t_averaged_data[i,:] - \
            transformed_z_t_averaged_data[i+int(transformed_z_t_averaged_data.shape[0]/2), :]) \
            for i in np.arange(int(transformed_z_t_averaged_data.shape[0]/2))])

        tED_norm = tED/np.sqrt(transformed_z_t_averaged_data.shape[1])

        ED = pd.DataFrame()
        ED['region'] = [roi]
        ED['ED_norm'] = [np.max(tED_norm)]
        ED_per_region = pd.concat([ED_per_region,ED])

    sns.barplot(data= ED_per_region, x= 'region', y='ED_norm')
    plt.ylabel('Normalized Eucledian Distance')
    plt.xlabel('Region')

if __name__=="__main__":


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

    # Bin all data
    region_df_grouped = pd.DataFrame()
    for i, roi in enumerate(rois):
        region_df = psths_per_regions(sessions, roi=roi)
        region_df_grouped = pd.concat([region_df_grouped,region_df])

    # Plot without differentiating choice*dA interaction
    locations = np.array([[0,3], [0,1], [1,0], [0,2], [1,2],  [1,3], [1,4], [2,0], [0,0], [2,2], [0,4]])

    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        try:
            region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
            plt.sca(ax[locations[i][0], locations[i][1]])
            plot_region_psth(region_df)
        except:
            continue

    # Plot  differentiating choice*dA interaction
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_region_psth_dachoice_interaction(region_df)

    # Plot  differentiating choice*dA interaction with diff
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_region_psth_dachoice_interaction_diff(region_df)

    # Plot  differentiating choice*dA interaction and qchosen
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_region_psth_dachoice_interaction_trial_qchosen(region_df)

    # Plot  differentiating choice*dA interaction and deltaq(contra-ipsi)
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_region_psth_dachoice_interaction_trial_delta(region_df)

    # Plot  differentiating qchosen
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_region_psth_qchosen(region_df)

    # Plot  differentiating deltaq(contra-ipsi)
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_region_psth_delta(region_df)

    # Plot differentiating latency
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_region_psth_latency(region_df)

    # PCA analysis at laser
    fig, ax = plt.subplots(2,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_laser_qchosen_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_laser_qchosen_lower')

    # PCA analysis at laser
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_laser_contra_qchosen_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_laser_contra_qchosen_lower')

    # PCA analysis at laser
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_laser_ipsi_qchosen_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_laser_ipsi_qchosen_lower')

    # PCA analysis at laser contra vs ipsi
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_laser_contra', 
                            trial_type_binned_spikes_2 =  'binned_spikes_laser_ipsi')

    # PCA analysis at error contra vs ipsi
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_error_contra', 
                            trial_type_binned_spikes_2 =  'binned_spikes_error_ipsi')


    # PCA analysis at choice * contra
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_choice_contra_qchosen_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_choice_contra_qchosen_lower')

    # PCA analysis at choice * ipsi
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_choice_ipsi_qchosen_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_choice_ipsi_qchosen_lower')


    # PCA analysis at gocue
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_gocue_qchosen_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_gocue_qchosen_lower')


    # PCA analysis at gocue
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_gocue_delta_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_gocue_delta_lower')

    # PCA analysis at choice
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_choice_contra', 
                            trial_type_binned_spikes_2 =  'binned_spikes_choice_ipsi')

    # PCA analysis at choice
    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_outcome_reward', 
                            trial_type_binned_spikes_2 =  'binned_spikes_outcome_error')

    # ED in full dimensional space
    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED(region_df,trial_type_binned_spikes_1 = 'binned_spikes_laser_contra', 
                            trial_type_binned_spikes_2 =  'binned_spikes_laser_ipsi')

    # ED in low dimensional space
    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_laser_contra', 
                            trial_type_binned_spikes_2 =  'binned_spikes_laser_ipsi', nd='max')


    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_choice_contra_qchosen_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_choice_contra_qchosen_lower', nd='max')

    # Global ED plots

    # contra vs ipsi

    variables_1 = ['binned_spikes_choice_contra',
                    'binned_spikes_error_contra', 'binned_spikes_reward_contra','binned_spikes_laser_contra']

    variables_2 = ['binned_spikes_choice_ipsi',
                    'binned_spikes_error_ipsi', 'binned_spikes_reward_ipsi','binned_spikes_laser_ipsi']
    xlabels = 'Time from outcome/first laser'

    palette = ['b', 'k', 'g', 'r']
    custom_lines = [Line2D([0], [0], color=palette[0], lw=4),
                    Line2D([0], [0], color=palette[1], lw=4),
                    Line2D([0], [0], color=palette[2], lw=4),
                    Line2D([0], [0], color=palette[3], lw=4)]

    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_global(region_df, variables_1, variables_2, palette, pre_time=0.5,
                            nd='max', zscoring=False, baseline_subtraction=False, use_pre_time=True, 
                            xlabels = xlabels)
        plt.legend(custom_lines, ['Choice Contra-Ipsi', 'CS- Contra-Ipsi', 'CS+ Contra-Ipsi',
                    'Laser Contra-Ipsi'])

    variables_1 = ['binned_spikes_outcome_error',
                    'binned_spikes_error_contra' , 'binned_spikes_error_ipsi', 
                    'binned_spikes_error_contra', 'binned_spikes_error_ipsi']
    variables_2 = ['binned_spikes_outcome_reward',
                    'binned_spikes_reward_contra','binned_spikes_reward_ipsi' ,
                    'binned_spikes_laser_contra', 'binned_spikes_laser_ipsi']
    xlabels = 'Time from outcome/first laser'

    palette = ['b', 'k', 'g', 'r', 'orange']
    custom_lines = [Line2D([0], [0], color=palette[0], lw=4),
                    Line2D([0], [0], color=palette[1], lw=4),
                    Line2D([0], [0], color=palette[2], lw=4),
                    Line2D([0], [0], color=palette[3], lw=4),
                    Line2D([0], [0], color=palette[4], lw=4)]

    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_global(region_df, variables_1, variables_2, palette, pre_time=0.5,
                            nd='max', zscoring=False, baseline_subtraction=False, use_pre_time=True,
                            xlabels = xlabels)
        plt.legend(custom_lines, ['CS- vs CS+', 'CS- contra vs  CS+ contra', 'CS- ipsi vs  CS+ ipsi',
                    'CS- ipsi vs  Laser contra', 'CS- ipsi vs  Laser ipsi'])

    # Normalized to 95 variance 

    # ED in low dimensional space
    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_laser_contra', 
                            trial_type_binned_spikes_2 =  'binned_spikes_laser_ipsi', nd='n_max')

    # State value plots

    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_region_psth_value(region_df)

    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_gocue_value_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_gocue_value_lower', nd = 'max')


    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_outcome_reward_value_upper', 
        trial_type_binned_spikes_2 =  'binned_spikes_outcome_reward_value_lower', nd = 'max')


    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_outcome_error_value_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_outcome_error_value_lower', nd = 'max')


    fig, ax = plt.subplots(3,5, sharex=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_gocue_value_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_gocue_value_lower')




    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_choice_contra', 
                            trial_type_binned_spikes_2 =  'binned_spikes_choice_ipsi', nd='max')



    #
    ev =  pd.DataFrame()
    for i, roi in enumerate(['NAc', 'DLS']):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        ev_roi  = pd.DataFrame()
        ev_roi['var_explained'] = plot_PCA_trajectory(region_df,trial_type_binned_spikes_1 = 'binned_spikes_choice_contra', 
                            trial_type_binned_spikes_2 =  'binned_spikes_choice_ipsi')
        ev_roi['roi'] = roi
        ev_roi['n_d'] = np.arange(len(ev_roi['var_explained']))
        ev = pd.concat([ev, ev_roi])

    sns.pointplot(data=ev, x='n_d', y='var_explained', hue = 'roi')

    # decision latency
    fig, ax = plt.subplots(3,5, sharex=True,  sharey=True)
    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plt.sca(ax[locations[i][0], locations[i][1]])
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_gocue_decision_times_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_gocue_decision_times_lower', nd='max')


    for i, roi in enumerate(rois):
        region_df = region_df_grouped.loc[region_df_grouped['region'] == roi]
        plot_time_varying_ED_nd(region_df,trial_type_binned_spikes_1 = 'binned_spikes_gocue_decision_times_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_gocue_decision_times_lower', nd='max')

    # summary ED
    barplot_ED(region_df_grouped,rois,trial_type_binned_spikes_1 = 'binned_spikes_gocue_qchosen_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_gocue_qchosen_lower')

    barplot_ED(region_df_grouped,rois,trial_type_binned_spikes_1 = 'binned_spikes_gocue_delta_upper', 
                            trial_type_binned_spikes_2 =  'binned_spikes_gocue_delta_lower')
