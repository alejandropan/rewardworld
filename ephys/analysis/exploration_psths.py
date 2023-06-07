from ephys_alf_summary import *
import brainbox.singlecell as sc
import copy
import os
#Functions  - Add significance to neurons

def plot_choice_summary(encoding_df_cluster, sessions, translator):
    fig, ax = plt.subplots(2,3)
    fig.suptitle('alpha=contra-ipsi' +' '+ 
                'recording='+str(encoding_df_cluster['id'])+ ' ' + 
                'cluster='+str(encoding_df_cluster['cluster'])+' ' + 
                'region='+str(encoding_df_cluster['region'])+
                '\nChoice_p '+str(encoding_df_cluster['choice'])
                )
    plt.sca(ax[0,0])
    # Based on cluster of interest get and divide qdeltas and plot cue for water trials
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster'])  
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    water = water[water>10]
    water = water[water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.choice_ipsi = ses_of_interest_behavior.choice_ipsi[water]
    ses_of_interest_behavior.choice_contra = ses_of_interest_behavior.choice_contra[water]
    ses_of_interest_behavior.goCue_trigger_times = ses_of_interest_behavior.goCue_trigger_times[water]

    ipsi = np.where(ses_of_interest_behavior.choice_ipsi==1)
    contra = np.where(ses_of_interest_behavior.choice_contra==1)

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[ipsi], 
                    pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='dodgerblue', alpha=0.3, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[contra], 
                    pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='dodgerblue', n_sample_size=len(bs))
    plt.title('Water at cue')
    plt.xlabel('Time from cue(s)')

    # Based on cluster of interest get and divide qdeltas and plot feedback for rewarded water trials
    plt.sca(ax[0,1])
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster'])  
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)  
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    outcome_water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    correct_water = np.intersect1d(outcome_water,outcome)
    correct_water = correct_water[correct_water>10]
    correct_water = correct_water[correct_water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.choice_ipsi = ses_of_interest_behavior.choice_ipsi[correct_water]
    ses_of_interest_behavior.choice_contra = ses_of_interest_behavior.choice_contra[correct_water]
    ses_of_interest_behavior.response_times = ses_of_interest_behavior.response_times[correct_water]

    ipsi = np.where(ses_of_interest_behavior.choice_ipsi==1)
    contra = np.where(ses_of_interest_behavior.choice_contra==1)

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[ipsi], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='dodgerblue', alpha=0.3, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[contra], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='dodgerblue', n_sample_size=len(bs))
    plt.title('Water at cue')
    plt.xlabel('Time from cue(s)')

    # Based on cluster of interest get and divide qdeltas and plot cue for laser trials
    plt.sca(ax[1,0])
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster'])  
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    laser = laser[laser>10]
    laser = laser[laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.choice_ipsi = ses_of_interest_behavior.choice_ipsi[laser]
    ses_of_interest_behavior.choice_contra = ses_of_interest_behavior.choice_contra[laser]
    ses_of_interest_behavior.goCue_trigger_times = ses_of_interest_behavior.goCue_trigger_times[laser]

    ipsi = np.where(ses_of_interest_behavior.choice_ipsi==1)
    contra = np.where(ses_of_interest_behavior.choice_contra==1)

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[ipsi], 
                    pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', alpha=0.3, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[contra], 
                    pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', n_sample_size=len(bs))
    plt.title('Laser at cue')
    plt.xlabel('Time from cue(s)')


    # Based on cluster of interest get and divide qdeltas and plot feedback for rewarded laser trials
    plt.sca(ax[1,1])
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster'])  
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    outcome_laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    correct_laser = np.intersect1d(outcome_laser,outcome)
    correct_laser = correct_laser[correct_laser>10]
    correct_laser = correct_laser[correct_laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.choice_ipsi = ses_of_interest_behavior.choice_ipsi[correct_laser]
    ses_of_interest_behavior.choice_contra = ses_of_interest_behavior.choice_contra[correct_laser]
    ses_of_interest_behavior.response_times = ses_of_interest_behavior.response_times[correct_laser]

    ipsi = np.where(ses_of_interest_behavior.choice_ipsi==1)
    contra = np.where(ses_of_interest_behavior.choice_contra==1)

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[ipsi], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', alpha=0.3, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[contra], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', n_sample_size=len(bs))
    plt.title('Correct Laser at outcome')
    plt.xlabel('Time from outcome(s)')
    plt.tight_layout()


    ##### Now the incorrect
    # Based on cluster of interest get and divide qdeltas and plot feedback for rewarded water trials
    plt.sca(ax[0,2])
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster'])  
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    outcome_water = np.where(ses_of_interest_behavior.opto_block==0)[0]
    incorrect_water = np.intersect1d(outcome_water,outcome)
    incorrect_water = incorrect_water[incorrect_water>10]
    incorrect_water = incorrect_water[incorrect_water<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.choice_ipsi = ses_of_interest_behavior.choice_ipsi[incorrect_water]
    ses_of_interest_behavior.choice_contra = ses_of_interest_behavior.choice_contra[incorrect_water]
    ses_of_interest_behavior.response_times = ses_of_interest_behavior.response_times[incorrect_water]

    ipsi = np.where(ses_of_interest_behavior.choice_ipsi==1)
    contra = np.where(ses_of_interest_behavior.choice_contra==1)

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[ipsi], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='dodgerblue', alpha=0.3, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[contra], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='dodgerblue', n_sample_size=len(bs))
    plt.title('Incorrect water at outcome')
    plt.xlabel('Time from outcome(s)')

    # Based on cluster of interest get and divide qdelta and plot feedback for unrewarded laser trials
    plt.sca(ax[1,2])

    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster'])  
    ses_of_interest_behavior = calculate_ipsi_contra_choice(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    outcome_laser = np.where(ses_of_interest_behavior.opto_block==1)[0]
    incorrect_laser = np.intersect1d(outcome_laser,outcome)
    incorrect_laser = incorrect_laser[incorrect_laser>10]
    incorrect_laser = incorrect_laser[incorrect_laser<len(ses_of_interest_behavior.outcome)-150]
    ses_of_interest_behavior.choice_ipsi = ses_of_interest_behavior.choice_ipsi[incorrect_laser]
    ses_of_interest_behavior.choice_contra = ses_of_interest_behavior.choice_contra[incorrect_laser]
    ses_of_interest_behavior.response_times = ses_of_interest_behavior.response_times[incorrect_laser]

    ipsi = np.where(ses_of_interest_behavior.choice_ipsi==1)
    contra = np.where(ses_of_interest_behavior.choice_contra==1)

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[ipsi], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', alpha=0.3, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[contra], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', n_sample_size=len(bs))
    plt.title('Incorrect Laser at outcome')
    plt.xlabel('Time from outcome(s)')


def plot_qchosen_summary(encoding_df_cluster, sessions, translator):
    fig, ax = plt.subplots(1,3)
    fig.suptitle('alpha=Qchosen' +' '+ 
                'recording='+str(encoding_df_cluster['id'])+ ' ' + 
                'cluster='+str(encoding_df_cluster['cluster_id'])+' ' + 
                'region='+str(encoding_df_cluster['region'])+
                '\nAll_p '+str(encoding_df_cluster['value'])+
                '\nwater_p '+str(encoding_df_cluster['value_water'])+
                ' laser_p  '+str(encoding_df_cluster['value_laser'])+
                ' stay_p '+str(encoding_df_cluster['value_stay'])
                )
    plt.sca(ax[0])
    # Based on cluster of interest get and divide qchosens and plot cue for water trials
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster_id'])
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)

    low_value = np.where(ses_of_interest_behavior.qchosen_l<(np.quantile(ses_of_interest_behavior.qchosen_l,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_l>=np.quantile(ses_of_interest_behavior.qchosen_l,0.333))&
        (ses_of_interest_behavior.qchosen_l<=(np.quantile(ses_of_interest_behavior.qchosen_l,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_l>(np.quantile(ses_of_interest_behavior.qchosen_l,0.666)))

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[low_value[0][:-1]+1], 
                    pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', alpha=0.3, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[medium_value[0][:-1]+1], 
                    pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth,color='orange', alpha=0.6, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[high_value[0][:-1]+1], 
                    pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', n_sample_size=len(bs))
    plt.title('Laser at cue')
    plt.xlabel('Time from cue(s)')

    # Based on cluster of interest get and divide qchosens and plot feedback for rewarded laser trials
    plt.sca(ax[1])
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster_id'])
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    ses_of_interest_behavior.qchosen_l = ses_of_interest_behavior.qchosen_l[outcome]
    ses_of_interest_behavior.response_times = ses_of_interest_behavior.response_times[outcome]

    low_value = np.where(ses_of_interest_behavior.qchosen_l<(np.quantile(ses_of_interest_behavior.qchosen_l,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_l>=np.quantile(ses_of_interest_behavior.qchosen_l,0.333))&
        (ses_of_interest_behavior.qchosen_l<=(np.quantile(ses_of_interest_behavior.qchosen_l,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_l>(np.quantile(ses_of_interest_behavior.qchosen_l,0.666)))
    n_sample_size  = len(low_value)

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[low_value], 
                    pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', alpha=0.3, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[medium_value], 
                    pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth,color='orange', alpha=0.6, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[high_value], 
                    pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', n_sample_size=len(bs))
    plt.title('Correct Laser at outcome')
    plt.xlabel('Time from outcome(s)')
    plt.tight_layout()


    ##### Now the incorrect
    # Based on cluster of interest get and divide qchosens and plot feedback for rewarded water trials
    plt.sca(ax[2])
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster_id'])
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    ses_of_interest_behavior.qchosen_l = ses_of_interest_behavior.qchosen_l[outcome]
    ses_of_interest_behavior.response_times = ses_of_interest_behavior.response_times[outcome]

    low_value = np.where(ses_of_interest_behavior.qchosen_l<(np.quantile(ses_of_interest_behavior.qchosen_l,0.333)))
    medium_value = np.where((ses_of_interest_behavior.qchosen_l>=np.quantile(ses_of_interest_behavior.qchosen_l,0.333))&
        (ses_of_interest_behavior.qchosen_l<=(np.quantile(ses_of_interest_behavior.qchosen_l,0.666))))
    high_value = np.where(ses_of_interest_behavior.qchosen_l>(np.quantile(ses_of_interest_behavior.qchosen_l,0.666)))

    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[low_value], 
                    pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, alpha=0.3, color = 'orange', n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[medium_value], 
                    pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth,color='orange', alpha=0.6, n_sample_size=len(bs))
    psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                    np.array([cluster_id]), ses_of_interest_behavior.response_times[high_value], 
                    pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
    plot_psth(psth, color='orange', n_sample_size=len(bs))
    plt.title('Incorrect Laser at outcome')
    plt.xlabel('Time from outcome(s)')


def plot_qchosen_summary_detailed(encoding_df_cluster, sessions, translator):
    fig, ax = plt.subplots()
    fig.suptitle('alpha=qcontra-ipsi' +' '+ 
                'recording='+str(encoding_df_cluster['id'])+ ' ' + 
                'cluster='+str(encoding_df_cluster['cluster_id'])+' ' + 
                'region='+str(encoding_df_cluster['region'])+
                '\nAll_p '+str(encoding_df_cluster['policy'])+
                '\nwater_p '+str(encoding_df_cluster['policy_water'])+
                ' laser_p  '+str(encoding_df_cluster['policy_laser'])+
                ' stay_p '+str(encoding_df_cluster['policy_stay'])
                )
    # Based on cluster of interest get and divide qdeltas and plot cue for water trials
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster_id'])
    ses_of_interest_behavior = calculate_qchosen(ses_of_interest_behavior)
    ses_of_interest_behavior.qchosen = ses_of_interest_behavior.qchosen
    ses_of_interest_behavior.goCue_trigger_times = ses_of_interest_behavior.goCue_trigger_times

    a_value = np.where(ses_of_interest_behavior.qchosen<(np.quantile(ses_of_interest_behavior.qchosen,0.2)))
    b_value = np.where((ses_of_interest_behavior.qchosen>=np.quantile(ses_of_interest_behavior.qchosen,0.2))&
        (ses_of_interest_behavior.qchosen<=(np.quantile(ses_of_interest_behavior.qchosen,0.4))))
    c_value = np.where((ses_of_interest_behavior.qchosen>=np.quantile(ses_of_interest_behavior.qchosen,0.4))&
        (ses_of_interest_behavior.qchosen<=(np.quantile(ses_of_interest_behavior.qchosen,0.6))))
    d_value = np.where((ses_of_interest_behavior.qchosen>=np.quantile(ses_of_interest_behavior.qchosen,0.6))&
        (ses_of_interest_behavior.qchosen<=(np.quantile(ses_of_interest_behavior.qchosen,0.8))))
    e_value = np.where(ses_of_interest_behavior.qchosen>(np.quantile(ses_of_interest_behavior.qchosen,0.8)))

    for i, trial_set in enumerate([a_value, b_value, c_value, d_value, e_value]):
        try:
            psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                            np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[trial_set[0][:-1]+1], 
                            pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
            plot_psth(psth, color='orange', alpha=(1+i)*0.2, n_sample_size=len(bs))
        except:
            print('No high value laser trials')



def plot_qdelta_summary_detailed(encoding_df_cluster, sessions, translator):
    fig, ax = plt.subplots()
    fig.suptitle('alpha=qcontra-ipsi' +' '+ 
                'recording='+str(encoding_df_cluster['id'])+ ' ' + 
                'cluster='+str(encoding_df_cluster['cluster_id'])+' ' + 
                'region='+str(encoding_df_cluster['region'])+
                '\nAll_p '+str(encoding_df_cluster['policy'])+
                '\nwater_p '+str(encoding_df_cluster['policy_water'])+
                ' laser_p  '+str(encoding_df_cluster['policy_laser'])+
                ' stay_p '+str(encoding_df_cluster['policy_stay'])
                )
    # Based on cluster of interest get and divide qdeltas and plot cue for water trials
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster_id'])
    ses_of_interest_behavior = calculate_dq(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    ses_of_interest_behavior.Qdelta = ses_of_interest_behavior.Qdelta
    ses_of_interest_behavior.goCue_trigger_times = ses_of_interest_behavior.goCue_trigger_times

    a_value = np.where(ses_of_interest_behavior.Qdelta<(np.quantile(ses_of_interest_behavior.Qdelta,0.2)))
    b_value = np.where((ses_of_interest_behavior.Qdelta>=np.quantile(ses_of_interest_behavior.Qdelta,0.2))&
        (ses_of_interest_behavior.Qdelta<=(np.quantile(ses_of_interest_behavior.Qdelta,0.4))))
    c_value = np.where((ses_of_interest_behavior.Qdelta>=np.quantile(ses_of_interest_behavior.Qdelta,0.4))&
        (ses_of_interest_behavior.Qdelta<=(np.quantile(ses_of_interest_behavior.Qdelta,0.6))))
    d_value = np.where((ses_of_interest_behavior.Qdelta>=np.quantile(ses_of_interest_behavior.Qdelta,0.6))&
        (ses_of_interest_behavior.Qdelta<=(np.quantile(ses_of_interest_behavior.Qdelta,0.8))))
    e_value = np.where(ses_of_interest_behavior.Qdelta>(np.quantile(ses_of_interest_behavior.Qdelta,0.8)))

    for i, trial_set in enumerate([a_value, b_value, c_value, d_value, e_value]):
        try:
            psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                            np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[trial_set[0][:-1]+1], 
                            pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
            plot_psth(psth, color='orange', alpha=(1+i)*0.2, n_sample_size=len(bs))
        except:
            print('No high value laser trials')


def plot_qdelta_summary(encoding_df_cluster, sessions, translator):
    fig, ax = plt.subplots(1,3)
    fig.suptitle('alpha=qcontra-ipsi' +' '+ 
                'recording='+str(encoding_df_cluster['id'])+ ' ' + 
                'cluster='+str(encoding_df_cluster['cluster_id'])+' ' + 
                'region='+str(encoding_df_cluster['region'])+
                '\nAll_p '+str(encoding_df_cluster['policy'])+
                '\nwater_p '+str(encoding_df_cluster['policy_water'])+
                ' laser_p  '+str(encoding_df_cluster['policy_laser'])+
                ' stay_p '+str(encoding_df_cluster['policy_stay'])
                )
    plt.sca(ax[0])
    # Based on cluster of interest get and divide qdeltas and plot cue for water trials
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster_id'])
    ses_of_interest_behavior = calculate_dq(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    ses_of_interest_behavior.QLaserdelta = ses_of_interest_behavior.QLaserdelta
    ses_of_interest_behavior.goCue_trigger_times = ses_of_interest_behavior.goCue_trigger_times

    low_value = np.where(ses_of_interest_behavior.QLaserdelta<(np.quantile(ses_of_interest_behavior.QLaserdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QLaserdelta>=np.quantile(ses_of_interest_behavior.QLaserdelta,0.333))&
        (ses_of_interest_behavior.QLaserdelta<=(np.quantile(ses_of_interest_behavior.QLaserdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QLaserdelta>(np.quantile(ses_of_interest_behavior.QLaserdelta,0.666)))

    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[low_value[0][:-1]+1], 
                        pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth, color='orange', alpha=0.3, n_sample_size=len(bs))
    except:
        print('No low value laser trials')
    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[medium_value[0][:-1]+1], 
                        pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth,color='orange', alpha=0.6, n_sample_size=len(bs))
    except:
        print('No mid value laser trials')
    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.goCue_trigger_times[high_value[0][:-1]+1], 
                        pre_time=0.2, post_time=0.5, bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth, color='orange', n_sample_size=len(bs))
    except:
        print('No high value laser trials')
    plt.title('Laser at cue')
    plt.xlabel('Time from cue(s)')


    # Based on cluster of interest get and divide qdeltas and plot feedback for rewarded laser trials
    plt.sca(ax[1])
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster_id'])
    ses_of_interest_behavior = calculate_dq(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==1)[0]
    ses_of_interest_behavior.QLaserdelta = ses_of_interest_behavior.QLaserdelta[outcome]
    ses_of_interest_behavior.response_times = ses_of_interest_behavior.response_times[outcome]

    low_value = np.where(ses_of_interest_behavior.QLaserdelta<(np.quantile(ses_of_interest_behavior.QLaserdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QLaserdelta>=np.quantile(ses_of_interest_behavior.QLaserdelta,0.333))&
        (ses_of_interest_behavior.QLaserdelta<=(np.quantile(ses_of_interest_behavior.QLaserdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QLaserdelta>(np.quantile(ses_of_interest_behavior.QLaserdelta,0.666)))


    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.response_times[low_value], 
                        pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth, color='orange', alpha=0.3, n_sample_size=len(bs))
    except:
        print('No low value laser trials')
    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.response_times[medium_value], 
                        pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth,color='orange', alpha=0.6, n_sample_size=len(bs))

    except:
        print('No mid value laser trials')
    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.response_times[high_value], 
                        pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth, color='orange', n_sample_size=len(bs))
    except:
        print('No high value laser trials')
    plt.title('Correct Laser at outcome')
    plt.xlabel('Time from outcome(s)')
    plt.tight_layout()


    ##### Now the incorrect
    # Based on cluster of interest get and divide qdelta and plot feedback for unrewarded laser trials
    plt.sca(ax[2])
    ses_of_interest_ephys, ses_of_interest_behavior  =  \
        get_ses_from_encoding_cluster (encoding_df_cluster, sessions, translator)
    cluster_id = int(encoding_df_cluster['cluster_id'])
    ses_of_interest_behavior = calculate_dq(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster)
    outcome = np.where(ses_of_interest_behavior.outcome==0)[0]
    ses_of_interest_behavior.QLaserdelta = ses_of_interest_behavior.QLaserdelta[outcome]
    ses_of_interest_behavior.response_times = ses_of_interest_behavior.response_times[outcome]

    low_value = np.where(ses_of_interest_behavior.QLaserdelta<(np.quantile(ses_of_interest_behavior.QLaserdelta,0.333)))
    medium_value = np.where((ses_of_interest_behavior.QLaserdelta>=np.quantile(ses_of_interest_behavior.QLaserdelta,0.333))&
        (ses_of_interest_behavior.QLaserdelta<=(np.quantile(ses_of_interest_behavior.QLaserdelta,0.666))))
    high_value = np.where(ses_of_interest_behavior.QLaserdelta>(np.quantile(ses_of_interest_behavior.QLaserdelta,0.666)))

    n_sample_size  = len(low_value)
    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.response_times[low_value], 
                        pre_time=0.2, post_time=1, bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth, alpha=0.3, color = 'orange', n_sample_size=len(bs))
    except:
        print('No low value water trials')
    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.response_times[medium_value], 
                        pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth,color='orange', alpha=0.6, n_sample_size=len(bs))
    except:
        print('No mid value water trials')
    try:
        psth,bs = calculate_peths(ses_of_interest_ephys.spike_times, ses_of_interest_ephys.spike_clusters, 
                        np.array([cluster_id]), ses_of_interest_behavior.response_times[high_value], 
                        pre_time=0.2, post_time=1,  bin_size=0.025, smoothing=0.025, return_fr=True)
        plot_psth(psth, color='orange', n_sample_size=len(bs))
    except:
        print('No high value water trials')
    plt.title('Incorrect Laser at outcome')
    plt.xlabel('Time from outcome(s)')

def calculate_qchosen(ses_of_interest_behavior):
    ses_of_interest_behavior.fQRreward_cue = np.copy(np.roll(ses_of_interest_behavior.fQRreward,1))
    ses_of_interest_behavior.fQLreward_cue = np.copy(np.roll(ses_of_interest_behavior.fQLreward,1))
    ses_of_interest_behavior.fQRreward_cue[0] = 0
    ses_of_interest_behavior.fQLreward_cue[0] = 0
    ses_of_interest_behavior.fQR_cue = np.copy(np.roll(ses_of_interest_behavior.fQR,1))
    ses_of_interest_behavior.fQL_cue = np.copy(np.roll(ses_of_interest_behavior.fQL,1))
    ses_of_interest_behavior.fQR_cue[0] = 0
    ses_of_interest_behavior.fQL_cue[0] = 0
    ses_of_interest_behavior.qchosen = ses_of_interest_behavior.fQR_cue
    ses_of_interest_behavior.qchosen[np.where(ses_of_interest_behavior.choice==-1)] = \
        ses_of_interest_behavior.fQL_cue[np.where(ses_of_interest_behavior.choice==-1)]
    ses_of_interest_behavior.qchosen_l = ses_of_interest_behavior.fQRreward_cue
    ses_of_interest_behavior.qchosen_l[np.where(ses_of_interest_behavior.choice==-1)] = \
        ses_of_interest_behavior.fQLreward_cue[np.where(ses_of_interest_behavior.choice==-1)]
    return ses_of_interest_behavior

def calculate_outcome(ses_of_interest_behavior):
    ses_of_interest_behavior.laser = ses_of_interest_behavior.outcome
    ses_of_interest_behavior.water = ses_of_interest_behavior.outcome
    ses_of_interest_behavior.laser[np.where(ses_of_interest_behavior.opto_block==0)] = 0
    ses_of_interest_behavior.water[np.where(ses_of_interest_behavior.opto_block==1)] = 0
    return ses_of_interest_behavior

def calculate_ipsi_contra_choice(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster):
    idx = np.where(np.unique(ses_of_interest_ephys.spike_clusters)==encoding_df_cluster['cluster'])
    right_is_ipsi = 1*(ses_of_interest_ephys.cluster_hem[idx]==1)
    ses_of_interest_behavior.choice_ipsi = np.zeros(len(ses_of_interest_behavior.choice))
    ses_of_interest_behavior.choice_contra = np.zeros(len(ses_of_interest_behavior.choice))
    if right_is_ipsi==1:
        ses_of_interest_behavior.choice_ipsi = 1*(ses_of_interest_behavior.choice==1)
        ses_of_interest_behavior.choice_contra = 1*(ses_of_interest_behavior.choice==-1)
    if right_is_ipsi==0:
        ses_of_interest_behavior.choice_ipsi = 1*(ses_of_interest_behavior.choice==-1)
        ses_of_interest_behavior.choice_contra = 1*(ses_of_interest_behavior.choice==1)
    return ses_of_interest_behavior

def calculate_dq(ses_of_interest_behavior, ses_of_interest_ephys,encoding_df_cluster):
    ses_of_interest_behavior.fQRreward_cue = np.copy(np.roll(ses_of_interest_behavior.fQRreward,1))
    ses_of_interest_behavior.fQLreward_cue = np.copy(np.roll(ses_of_interest_behavior.fQLreward,1))
    ses_of_interest_behavior.fQRreward_cue[0] = 0
    ses_of_interest_behavior.fQLreward_cue[0] = 0
    ses_of_interest_behavior.fQR_cue = np.copy(np.roll(ses_of_interest_behavior.fQR,1))
    ses_of_interest_behavior.fQL_cue = np.copy(np.roll(ses_of_interest_behavior.fQL,1))
    ses_of_interest_behavior.fQR_cue[0] = 0
    ses_of_interest_behavior.fQL_cue[0] = 0
    ses_of_interest_behavior.fQRstay_cue = np.copy(np.roll(ses_of_interest_behavior.fQRstay,1))
    ses_of_interest_behavior.fQLstay_cue = np.copy(np.roll(ses_of_interest_behavior.fQLstay,1))
    ses_of_interest_behavior.fQRstay_cue[0] = 0
    ses_of_interest_behavior.fQLstay_cue[0] = 0
    idx = np.where(np.unique(ses_of_interest_ephys.spike_clusters)==encoding_df_cluster['cluster_id'])
    right_is_ipsi = 1*(ses_of_interest_ephys.cluster_hem[idx]==1)    
    if right_is_ipsi==1:
        ses_of_interest_behavior.Qipsi = ses_of_interest_behavior.fQR_cue
        ses_of_interest_behavior.QLaseripsi = ses_of_interest_behavior.fQRreward_cue
        ses_of_interest_behavior.QStayipsi = ses_of_interest_behavior.fQRstay_cue
        ses_of_interest_behavior.Qcontra = ses_of_interest_behavior.fQL_cue
        ses_of_interest_behavior.QLasercontra = ses_of_interest_behavior.fQLreward_cue
        ses_of_interest_behavior.QStaycontra = ses_of_interest_behavior.fQLstay_cue      
    if right_is_ipsi==0:
        ses_of_interest_behavior.Qipsi = ses_of_interest_behavior.fQL_cue
        ses_of_interest_behavior.QLaseripsi = ses_of_interest_behavior.fQLreward_cue
        ses_of_interest_behavior.QStayipsi = ses_of_interest_behavior.fQLstay_cue
        ses_of_interest_behavior.Qcontra = ses_of_interest_behavior.fQR_cue
        ses_of_interest_behavior.QLasercontra = ses_of_interest_behavior.fQRreward_cue
        ses_of_interest_behavior.QStaycontra = ses_of_interest_behavior.fQRstay_cue
    ses_of_interest_behavior.Qdelta = (ses_of_interest_behavior.Qcontra - ses_of_interest_behavior.Qipsi)*100 # To avoid numpy precision problem
    ses_of_interest_behavior.QLaserdelta = (ses_of_interest_behavior.QLasercontra - ses_of_interest_behavior.QLaseripsi)*100 # To avoid numpy precision problem
    ses_of_interest_behavior.QStaydelta = (ses_of_interest_behavior.QStaycontra - ses_of_interest_behavior.QStayipsi)*100 # To avoid numpy precision problem
    return ses_of_interest_behavior

def get_object_sessions_dict (sessions,LIST_OF_SESSIONS):
    translator = pd.DataFrame()
    for i in np.arange(len(LIST_OF_SESSIONS)):
        temp_ses =pd.DataFrame()
        temp_ses['date'] = [sessions[i].date]
        temp_ses['ses'] = [sessions[i].ses]
        temp_ses['mouse']  = [sessions[i].mouse]
        temp_ses['index'] = i
        translator = pd.concat([translator,temp_ses])
    return translator

def get_ses_from_encoding_cluster(encoding_df_cluster, sessions, translator):
    idx = str(encoding_df_cluster['mouse']+ encoding_df_cluster['date'])
    encoding_df_cluster['id'] = idx
    ses_i = int(translator.loc[(translator['date']== encoding_df_cluster['date']) &
                    (translator['mouse']== encoding_df_cluster['mouse']), 'index'])
    try:
        ses_of_interest_ephys = sessions[ses_i].probe[int(encoding_df_cluster['probe'])]
    except:
        ses_of_interest_ephys = sessions[ses_i].probe[int(encoding_df_cluster['probe'])-1]
        print('probe00 missing')
    ses_of_interest_behavior = sessions[ses_i]
    return copy.deepcopy(ses_of_interest_ephys), copy.deepcopy(ses_of_interest_behavior)

def plot_psth(psth, color='k', alpha=1, n_sample_size = np.nan):
    plt.plot(psth.tscale, psth.means.T, color=color, alpha=alpha)
    for m in np.arange(psth.means.shape[0]):
        plt.fill_between(psth.tscale,
                        psth.means[m, :].T - (psth.stds[m, :].T / np.sqrt(n_sample_size)),
                        psth.means[m, :].T + (psth.stds[m, :].T / np.sqrt(n_sample_size)),
                        alpha=alpha/2, color=color)
    plt.ylabel('Firing Rate')
    plt.xlabel('Time (s)')
# Load Data
sessions = ephys_ephys_dataset(len(LASER_ONLY))
for i, ses in enumerate(LASER_ONLY):
            print(ses)
            ses_data = alf(ses, ephys=True)
            ses_data.mouse = Path(ses).parent.parent.name
            ses_data.date = Path(ses).parent.name
            ses_data.ses = Path(ses).name
            sessions[i] = ses_data

# Get translation table
translator =  get_object_sessions_dict (sessions,LIST_OF_SESSIONS_ALEX)

# Load significance table & add it to recordings object
encoding_df = pd.read_csv('/Users/alexpan/Downloads/encoding_model_summary1.csv')
encoding_df.loc[encoding_df['region'].str.contains('/', regex=True),'region'] = \
    encoding_df.loc[encoding_df['region'].str.contains('/', regex=True),'region'].replace('/','_',regex=True) # Remove backslashes for saving errors
encoding_df['id'] = encoding_df['mouse'] + encoding_df['date'] +  encoding_df['ses'].astype(str) +'_' + encoding_df['probe'].astype(str)

# For loop and asave by region
SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/qchosen'


for reg in encoding_df.region.unique():
    reg_encoding = encoding_df.loc[encoding_df['region']==reg].copy()
    reg_encoding_significant = reg_encoding.loc[(reg_encoding['value']<=0.00101)|
                    (reg_encoding['value_laser']<=0.00101)|
                    (reg_encoding['value_water']<=0.00101)|
                    (reg_encoding['value_stay']<=0.00101)]
    reg_encoding_nonsignificant = reg_encoding.loc[(reg_encoding['value']>0.00101)&
                    (reg_encoding['value_laser']>0.00101)&
                    (reg_encoding['value_water']>0.00101)&
                    (reg_encoding['value_stay']>0.00101)]
    for i in np.arange(len(reg_encoding_significant)):
        try:
        encoding_df_cluster = reg_encoding_significant.iloc[i,:]
        plot_qchosen_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/significant/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    str(encoding_df_cluster['probe']) +'_'+ 
                    str(encoding_df_cluster['cluster_id']))
        plt.close()

    for i in np.arange(len(reg_encoding_nonsignificant)):
        encoding_df_cluster = reg_encoding_nonsignificant.iloc[i,:]
        plot_qchosen_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/nonsignificant/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'] +'_'+ 
                    str(encoding_df_cluster['cluster_id']))
        plt.close()


SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/deltaq'
errors = []
for reg in encoding_df.region.unique():
    reg_encoding = encoding_df.loc[encoding_df['region']==reg].copy()
    reg_encoding_significant = reg_encoding.loc[(reg_encoding['policy']<=0.00101)|
                    (reg_encoding['policy_laser']<=0.00101)|
                    (reg_encoding['policy_water']<=0.00101)|
                    (reg_encoding['policy_stay']<=0.00101)]
    reg_encoding_nonsignificant = reg_encoding.loc[(reg_encoding['policy']>0.00101)&
                    (reg_encoding['policy_laser']>0.00101)&
                    (reg_encoding['policy_water']>0.00101)&
                    (reg_encoding['policy_stay']>0.00101)]
    for i in np.arange(len(reg_encoding_significant)):
        encoding_df_cluster = reg_encoding_significant.iloc[i,:]
        plot_qdelta_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/significant/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'].astype(str) +'_'+ 
                    str(encoding_df_cluster['cluster_id']))
        plt.close()

    for i in np.arange(len(reg_encoding_nonsignificant)):
        encoding_df_cluster = reg_encoding_nonsignificant.iloc[i,:]
        plot_qdelta_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/nonsignificant/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'] +'_'+ 
                    str(encoding_df_cluster['cluster_id']))
        plt.close()

SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/choice'
errors = []
for reg in encoding_df.region.unique():
    reg_encoding = encoding_df.loc[encoding_df['region']==reg].copy()
    reg_encoding_significant = reg_encoding.loc[(reg_encoding['choice']<=0.001)]
    reg_encoding_nonsignificant = reg_encoding.loc[(reg_encoding['choice']>0.001)]
    for i in np.arange(len(reg_encoding_significant)):
        encoding_df_cluster = reg_encoding_significant.iloc[i,:]
        plot_choice_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/significant/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/significant/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'] +'_'+ 
                    str(encoding_df_cluster['cluster']))
        plt.close()

    for i in np.arange(len(reg_encoding_nonsignificant)):
        encoding_df_cluster = reg_encoding_nonsignificant.iloc[i,:]
        plot_choice_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/nonsignificant/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/nonsignificant/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'] +'_'+ 
                    str(encoding_df_cluster['cluster']))
        plt.close()


SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/laser'
errors = []
for reg in encoding_df.region.unique():
    reg_encoding = encoding_df.loc[encoding_df['region']==reg].copy()
    reg_encoding_significant = reg_encoding.loc[reg_encoding['peak_epoch']==5]
    for i in np.arange(len(reg_encoding_significant)):
        encoding_df_cluster = reg_encoding_significant.iloc[i,:]
        plot_qdelta_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/qdelta/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/qdelta/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/qdelta/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'] +'_'+ 
                    str(encoding_df_cluster['cluster']))
        plt.close()


SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/deltaq'
errors = []
for reg in encoding_df.region.unique():
    reg_encoding = encoding_df.loc[encoding_df['region']==reg].copy()
    reg_encoding_significant = reg_encoding.loc[(reg_encoding['QLEARNING_policy_all']<=0.01) &
                    (reg_encoding['QLEARNING_policy_water']>0.01)&
                    (reg_encoding['QLEARNING_policy_laser']>0.01)&
                    (reg_encoding['QLEARNING_policy_stay']>0.01)]
    for i in np.arange(len(reg_encoding_significant)):
        encoding_df_cluster = reg_encoding_significant.iloc[i,:]
        plot_qdelta_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant_to_all_but_not_any/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/significant_to_all_but_not_any/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/significant_to_all_but_not_any/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'] +'_'+ 
                    str(encoding_df_cluster['cluster']))
        plt.close()

SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/deltaq'
errors = []
for reg in encoding_df.region.unique():
    reg_encoding = encoding_df.loc[encoding_df['region']==reg].copy()
    reg_encoding_significant = reg_encoding.loc[(reg_encoding['QLEARNING_policy_all']>0.01) & (
                    (reg_encoding['QLEARNING_policy_water']<=0.01)|
                    (reg_encoding['QLEARNING_policy_laser']<=0.01)|
                    (reg_encoding['QLEARNING_policy_stay']<=0.01))]
    for i in np.arange(len(reg_encoding_significant)):
        encoding_df_cluster = reg_encoding_significant.iloc[i,:]
        plot_qdelta_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant_to_any_but_not_all/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/significant_to_any_but_not_all/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/significant_to_any_but_not_all/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'] +'_'+ 
                    str(encoding_df_cluster['cluster']))
        plt.close()

SAVE_PATH ='/Users/alexpan/Documents/neuron_summaries/qchosen'
for reg in encoding_df.region.unique():
    reg_encoding = encoding_df.loc[encoding_df['region']==reg].copy()
    reg_encoding_significant = reg_encoding.loc[(reg_encoding['QLEARNING_value_all']<=0.01) &
                    (reg_encoding['QLEARNING_value_water']>0.01)&
                    (reg_encoding['QLEARNING_value_laser']>0.01)&
                    (reg_encoding['QLEARNING_value_stay']>0.01)]
    for i in np.arange(len(reg_encoding_significant)):
        encoding_df_cluster = reg_encoding_significant.iloc[i,:]
        plot_qchosen_summary(encoding_df_cluster, sessions, translator)
        if os.path.isdir(os.path.dirname(SAVE_PATH+'/significant_to_all_but_not_any/'+reg+'/'))==False:
            os.mkdir(os.path.dirname(SAVE_PATH+'/significant_to_all_but_not_any/'+reg+'/'))
        plt.savefig(SAVE_PATH+'/significant_to_all_but_not_any/'+reg+'/'+str(encoding_df_cluster['id'])+'_'+ 
                    encoding_df_cluster['probe'] +'_'+ 
                    str(encoding_df_cluster['cluster']))
        plt.close()