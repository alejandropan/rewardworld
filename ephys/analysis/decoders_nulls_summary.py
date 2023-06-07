from decoders_summary import *

def plot_null_analysis(decoders_restricted , nsummary_restricted, varss = None, epoch = None, pale = None):
    fig,ax  = plt.subplots(6,2, sharey=True, sharex=True)
    for i, reg in enumerate(selected_regions):
        plt.sca(ax[i%6,int(i/6)])
        region_data = decoders_restricted.loc[decoders_restricted['region']==reg]
        region_null = nsummary_restricted.loc[nsummary_restricted['region']==reg]
        sns.lineplot(data=region_data, x='time_bin', y='r', errorbar=None, hue='ses_id', palette=pale)
        sns.lineplot(data=region_null, x='time_bin', y='r', errorbar=('pi', 99), color='k', alpha=0.5)
        plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
        plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[1::5],2), rotation=90)
        plt.xlabel('Time from ' + epoch)
        plt.ylabel('Pearson - r')
        if i!=len(selected_regions):
            plt.legend().remove()
        plt.title(reg + ' '+ varss +  ' real vs null')
        sns.despine()
    plt.tight_layout()



def null_results(decoders_restricted, nsummary_restricted):
    results = pd.DataFrame()
    for m in decoders_restricted.mouse.unique():
        for date in decoders_restricted.loc[decoders_restricted['mouse']== m].date.unique():
            for r in decoders_restricted.loc[(decoders_restricted['mouse']== m) & (decoders_restricted['date']== date)].region.unique():
                select =  decoders_restricted.loc[(decoders_restricted['mouse']== m) & (decoders_restricted['date']== date)
                                                & (decoders_restricted['region']== r)]
                select_null = nsummary_restricted.loc[(nsummary_restricted['mouse']== m) & (nsummary_restricted['date']== date)
                                                & (nsummary_restricted['region']== r)]
                select_r = select['r'].to_numpy()
                null_r = select_null.groupby(['time_bin']).quantile(0.99)['r'].to_numpy()

                try:
                    #print(select_null.groupby(['time_bin']).count().iloc[0][0])
                    if select_null.groupby(['time_bin']).count().iloc[0][0]<90:
                        continue
                    else:
                        results_ses = pd.DataFrame()
                        results_ses['mouse'] = [m]
                        results_ses['date'] = [date]
                        results_ses['region'] = [r]
                        if any((select_r[5:]-null_r[5:])>0):
                            results_ses['sig'] = True
                        else:
                            results_ses['sig'] = False
                        results = pd.concat([results, results_ses])
                except:
                        continue
    return results.groupby(['region']).mean()['sig']

# Stats
def summary_null_stats(decoders_restricted, nsummary_restricted): 
    post_cue_sum_real = decoders_restricted.loc[(decoders_restricted['time_bin']>4) & (decoders_restricted['n_neurons']==20)].groupby(['id','region','mouse','date','hemisphere']).median().reset_index()
    post_cue_sum_null = nsummary_restricted.loc[(nsummary_restricted['time_bin']>4) & (nsummary_restricted['n_neurons']==20)].groupby(['id','region','mouse','date','hemisphere','iter']).median().reset_index()
    pre_cue_sum_real = decoders_restricted.loc[(decoders_restricted['time_bin']<5) & (decoders_restricted['n_neurons']==20)].groupby(['id','region','mouse','date','hemisphere']).median().reset_index()
    pre_cue_sum_null = nsummary_restricted.loc[(nsummary_restricted['time_bin']<5) & (nsummary_restricted['n_neurons']==20)].groupby(['id','region','mouse','date','hemisphere','iter']).median().reset_index()

    post_cue_sum_real['sig'] = np.nan
    post_cue_sum_real['d_score'] = np.nan
    pre_cue_sum_real['sig'] = np.nan
    pre_cue_sum_real['d_score'] = np.nan

    for mouse in post_cue_sum_real.mouse.unique():
        for date in post_cue_sum_real.loc[post_cue_sum_real['mouse']==mouse].date.unique():
            for hem in post_cue_sum_real.loc[(post_cue_sum_real['mouse']==mouse) & (post_cue_sum_real['date']==date)].hemisphere.unique():
                for reg in post_cue_sum_real.loc[(post_cue_sum_real['mouse']==mouse) & 
                            (post_cue_sum_real['date']==date) & (post_cue_sum_real['hemisphere']==hem)].region.unique():
                    if  np.isnan(post_cue_sum_null.loc[(post_cue_sum_null['mouse']==mouse) & (post_cue_sum_null['date']==date) 
                                            & (post_cue_sum_null['hemisphere']==int(hem))  & (post_cue_sum_null['region']==reg), 'r'].quantile(0.99)):
                        print(post_cue_sum_real.loc[(post_cue_sum_real['mouse']==mouse) & (post_cue_sum_real['date']==date) 
                                            & (post_cue_sum_real['hemisphere']==hem)  & (post_cue_sum_real['region']==reg),'id'])
                        continue
                    post_cue_sum_real.loc[(post_cue_sum_real['mouse']==mouse) & (post_cue_sum_real['date']==date) 
                                            & (post_cue_sum_real['hemisphere']==hem) & (post_cue_sum_real['region']==reg), 'sig'] = \
                    1*(post_cue_sum_real.loc[(post_cue_sum_real['mouse']==mouse) & (post_cue_sum_real['date']==date) 
                                            & (post_cue_sum_real['hemisphere']==hem)  & (post_cue_sum_real['region']==reg), 'r'] > \
                    post_cue_sum_null.loc[(post_cue_sum_null['mouse']==mouse) & (post_cue_sum_null['date']==date) 
                                            & (post_cue_sum_null['hemisphere']==int(hem))  & (post_cue_sum_null['region']==reg), 'r'].quantile(0.99))

                    post_cue_sum_real.loc[(post_cue_sum_real['mouse']==mouse) & (post_cue_sum_real['date']==date) 
                                            & (post_cue_sum_real['hemisphere']==hem) & (post_cue_sum_real['region']==reg), 'd_score'] = \
                    post_cue_sum_real.loc[(post_cue_sum_real['mouse']==mouse) & (post_cue_sum_real['date']==date) 
                                            & (post_cue_sum_real['hemisphere']==hem)  & (post_cue_sum_real['region']==reg), 'r'] - \
                    post_cue_sum_null.loc[(post_cue_sum_null['mouse']==mouse) & (post_cue_sum_null['date']==date) 
                                            & (post_cue_sum_null['hemisphere']==int(hem))  & (post_cue_sum_null['region']==reg), 'r'].median()

                    pre_cue_sum_real.loc[(pre_cue_sum_real['mouse']==mouse) & (pre_cue_sum_real['date']==date) 
                                            & (pre_cue_sum_real['hemisphere']==hem) & (pre_cue_sum_real['region']==reg), 'sig'] = \
                    1*(pre_cue_sum_real.loc[(pre_cue_sum_real['mouse']==mouse) & (pre_cue_sum_real['date']==date) 
                                            & (pre_cue_sum_real['hemisphere']==hem)  & (pre_cue_sum_real['region']==reg), 'r'] > \
                    pre_cue_sum_null.loc[(pre_cue_sum_null['mouse']==mouse) & (pre_cue_sum_null['date']==date) 
                                            & (pre_cue_sum_null['hemisphere']==int(hem))  & (pre_cue_sum_null['region']==reg), 'r'].quantile(0.99))

                    pre_cue_sum_real.loc[(pre_cue_sum_real['mouse']==mouse) & (pre_cue_sum_real['date']==date) 
                                            & (pre_cue_sum_real['hemisphere']==hem) & (pre_cue_sum_real['region']==reg), 'd_score'] = \
                    pre_cue_sum_real.loc[(pre_cue_sum_real['mouse']==mouse) & (pre_cue_sum_real['date']==date) 
                                            & (pre_cue_sum_real['hemisphere']==hem)  & (pre_cue_sum_real['region']==reg), 'r'] - \
                    pre_cue_sum_null.loc[(pre_cue_sum_null['mouse']==mouse) & (pre_cue_sum_null['date']==date) 
                                            & (pre_cue_sum_null['hemisphere']==int(hem))  & (pre_cue_sum_null['region']==reg), 'r'].median()
    post_cue_sum_real.groupby(['region']).mean()
    pre_cue_sum_real.groupby(['region']).mean()
    return  post_cue_sum_real.groupby(['region']).mean(), pre_cue_sum_real.groupby(['region']).mean()



def add_normalized_r(decoders_summary, nsummary):
    decoders_summary['r_norm'] = np.nan
    for mouse in decoders_summary.mouse.unique():
        for date in decoders_summary.loc[decoders_summary['mouse']==mouse].date.unique():
            for hem in decoders_summary.loc[(decoders_summary['mouse']==mouse) & (decoders_summary['date']==date)].hemisphere.unique():
                for reg in decoders_summary.loc[(decoders_summary['mouse']==mouse) & 
                            (decoders_summary['date']==date) & (decoders_summary['hemisphere']==hem)].region.unique():

                    try:
                        decoders_summary.loc[(decoders_summary['mouse']==mouse) & (decoders_summary['date']==date) 
                                                                    & (decoders_summary['hemisphere']==hem) & (decoders_summary['region']==reg), 'r_norm'] = \
                            decoders_summary.loc[(decoders_summary['mouse']==mouse) & (decoders_summary['date']==date) 
                                                & (decoders_summary['hemisphere']==hem) & (decoders_summary['region']==reg)].groupby(['time_bin']).median()['r'].to_numpy() \
                                                - nsummary.loc[(nsummary['mouse']==mouse) & (nsummary['date']==date) 
                                                & (nsummary['hemisphere']==int(hem)) & (nsummary['region']==reg)].groupby(['time_bin']).median()['r'].to_numpy()
                    except:
                        continue             
    return decoders_summary


# delta q summary scores
str_pal = dict({
    'DLS':'#c77c2c', 
    'DMS':'#a61678', 
    'TS':'#1a2696', 
    'NAc':'#3ea33e'})


decoders_summary  =  load_decoders('//Volumes/witten/Alex/decoders_raw_results/decoder_output_deltaq_cue_forget', var = 'deltaq', 
                                    epoch = 'cue', x_type = 'raw', null=False)    
nsummary = pd.read_csv('/Volumes/witten/Alex/decoders_raw_results/nsummary_deltaq.csv')
decoders_summary = add_normalized_r(decoders_summary, nsummary)
selected_regions = np.array(['SS', 'MO', 'PFC', 'OFC', 'DLS', 'DMS', 'TS', 'NAc', 'GPe', 'VP', 'Olfactory'])
selected_regions = np.array(['NAc','DMS', 'DLS'])
decoders_restricted = decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
nsummary_restricted = nsummary.loc[np.isin(nsummary['region'], selected_regions)]
nullstats = summary_null_stats(decoders_restricted, nsummary_restricted)[0].reset_index()
sns.barplot(data=nullstats, x='region', y='d_score', palette=str_pal)
plt.xlabel('Region')
plt.ylabel('Decoding Score over null')
plt.ylim(0,0.3)

selected_regions = np.array(['SS','MO','DLS', 'GPe'])
decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
sns.lineplot(data=decoders_restricted, x='time_bin', y='r_norm', hue='region', errorbar='se')
plt.xticks(np.array([0,4,9]),np.array([-0.4,0,0.5]), rotation=90)
plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
plt.xlabel('Time from cue')
plt.ylabel('Decoding peformance over null (r)')

selected_regions = np.array(['SS','MO', 'OFC', 'PFC'])
decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
sns.barplot(data=decoders_restricted.loc[decoders_restricted['time_bin']>4].groupby(['id','region']).median().reset_index()
            , x='region', y='r_norm', errorbar='se')
plt.xlabel('Region')
plt.ylabel('Decoding Score over null')
plt.ylim(0,0.5)


selected_regions = np.array(['GPe','VP'])
decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
sns.barplot(data=decoders_restricted.loc[decoders_restricted['time_bin']>4].groupby(['id','region']).median().reset_index()
            , x='region', y='r_norm', errorbar='se')
plt.xlabel('Region')
plt.ylabel('Decoding Score over null')
plt.ylim(0,0.5)


# qchosen summary scores
decoders_summary  =  load_decoders('//Volumes/witten/Alex/decoders_raw_results/decoder_output_qchosen_cue_forget', var = 'qchosen_pre', 
                                    epoch = 'cue', x_type = 'raw', null=False)    
nsummary = pd.read_csv('/Volumes/witten/Alex/decoders_raw_results/nsummary_qchosen.csv')
decoders_summary = add_normalized_r(decoders_summary, nsummary)
selected_regions = np.array(['SS', 'MO', 'PFC', 'OFC', 'DLS', 'DMS', 'TS', 'NAc', 'GPe', 'VP', 'Olfactory'])
selected_regions = np.array(['NAc','DMS', 'DLS'])
decoders_restricted = decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
nsummary_restricted = nsummary.loc[np.isin(nsummary['region'], selected_regions)]
nullstats = summary_null_stats(decoders_restricted, nsummary_restricted)[0].reset_index()
sns.barplot(data=nullstats, x='region', y='d_score', palette=str_pal)
plt.xlabel('Region')
plt.ylabel('Decoding Score over null')
plt.ylim(0,0.3)



selected_regions = np.array(['NAc','DMS', 'DLS'])
decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
sns.lineplot(data=decoders_restricted, x='time_bin', y='r_norm', hue='region', palette=str_pal, errorbar='se')
plt.xticks(np.array([0,4,9]),np.array([-0.4,0,0.5]), rotation=90)
plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
plt.xlabel('Time from cue')
plt.ylabel('Decoding peformance over null (r)')


selected_regions = np.array(['NAc','DMS', 'DLS'])
decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
sns.barplot(data=decoders_restricted.loc[decoders_restricted['time_bin']>4].groupby(['id','region']).median().reset_index()
            , x='region', y='r_norm', errorbar='se')
plt.xlabel('Region')
plt.ylabel('Decoding Score over null')
plt.ylim(0,0.5)



selected_regions = np.array(['SS','MO', 'OFC', 'PFC'])
decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
sns.barplot(data=decoders_restricted.loc[decoders_restricted['time_bin']>4].groupby(['id','region']).median().reset_index()
            , x='region', y='r_norm', errorbar='se')
plt.xlabel('Region')
plt.ylabel('Decoding Score over null')
plt.ylim(0,0.5)


selected_regions = np.array(['GPe','VP'])
decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
sns.barplot(data=decoders_restricted.loc[decoders_restricted['time_bin']>4].groupby(['id','region']).median().reset_index()
            , x='region', y='r_norm', errorbar='se')
plt.xlabel('Region')
plt.ylabel('Decoding Score over null')
plt.ylim(0,0.5)