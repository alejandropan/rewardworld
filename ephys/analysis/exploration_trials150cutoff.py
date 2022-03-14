    fig, ax= plt.subplots(1,2, sharey=True)
    rates_w=[]
    rates_l=[]
    plt.sca(ax[0])
    for i in np.arange(0.0,0.25,0.01):
        ses_dl_all_reduced = pd.DataFrame()
        for ses in ses_df_all['ses_id'].unique():
            r = ses_df_all.loc[ses_df_all['ses_id']==ses]
            thres = int(i*len(r))
            ses_dl_all_reduced = pd.concat([ses_dl_all_reduced,r[:-thres]])
        ses_dl_all_reduced_m = pd.DataFrame()
        rates_w.append(ses_dl_all_reduced.loc[ses_dl_all_reduced['opto_block']==0].groupby(['ses_id']).mean()['outcome'].median())
        rates_l.append(ses_dl_all_reduced.loc[ses_dl_all_reduced['opto_block']==1].groupby(['ses_id']).mean()['outcome'].median())      
    plt.plot(np.arange(0.0,0.25,0.01),rates_w)
    plt.plot(np.arange(0.0,0.25,0.01),rates_l)
    plt.xlabel('Fraction Excluded')
    plt.ylabel('Median session performance')

    plt.sca(ax[1])
    rates_w=[]
    rates_l=[]
    for i in np.arange(0,200,10):
        ses_dl_all_reduced = pd.DataFrame()
        for ses in ses_df_all['ses_id'].unique():
            r = ses_df_all.loc[ses_df_all['ses_id']==ses]
            ses_dl_all_reduced = pd.concat([ses_dl_all_reduced,r[:-i]])
        ses_dl_all_reduced_m = pd.DataFrame()
        rates_w.append(ses_dl_all_reduced.loc[ses_dl_all_reduced['opto_block']==0].groupby(['ses_id']).mean()['outcome'].median())
        rates_l.append(ses_dl_all_reduced.loc[ses_dl_all_reduced['opto_block']==1].groupby(['ses_id']).mean()['outcome'].median()) 
    plt.plot(np.arange(0,200,10),rates_w)
    plt.plot(np.arange(0,200,10),rates_l)
    plt.xlabel('Number of trials Excluded')
    plt.ylabel('Median session performance')


    # (For this I need to load the data with my alf script )

    fig, ax= plt.subplots(1,2, sharey=True)
    plt.sca(ax[0])
    rts_meta=[]
    for j in np.arange(0.0,0.25,0.01):
        rts = []
        for i, idx in enumerate(data.id.unique()):
            subdata = data.loc[data['id']==idx]
            rts.append(subdata['rt'].mean())

        rts_150 = []
        for i, idx in enumerate(data.id.unique()):
            subdata = data.loc[data['id']==idx]
            thres = int(j*len(subdata))
            rts_150.append(subdata['rt'][:-thres].mean())
        rts_meta.append(np.mean(np.array(rts_150)-np.array(rts)))
    plt.plot(np.arange(0.0,0.25,0.01),rts_meta)
    plt.xlabel('Fraction Excluded')
    plt.ylabel('Mean DeltaRT')
    plt.sca(ax[1])
    rts_meta=[]
    for j in np.arange(0,200,10):
        rts = []
        for i, idx in enumerate(data.id.unique()):
            subdata = data.loc[data['id']==idx]
            rts.append(subdata['rt'].mean())

        rts_150 = []
        for i, idx in enumerate(data.id.unique()):
            subdata = data.loc[data['id']==idx]
            rts_150.append(subdata['rt'][:-j].mean())
        rts_meta.append(np.mean(np.array(rts_150)-np.array(rts)))
    plt.plot(np.arange(0,200,10),rts_meta)
    plt.xlabel('Number of trials Excluded')
    plt.ylabel('Mean DeltaRT')
