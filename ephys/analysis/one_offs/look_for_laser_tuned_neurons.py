# Investigate laser tuned neurons

session = 'dop_47/2022-06-06/001'


region_df = region_df_grouped.loc[(region_df_grouped['mouse']=='dop_47') & (region_df_grouped['date']=='2022-06-06')]
region_df = p_df.copy()




# Units in probe 3 (imec2) are the ones tuned

region_df = region_df_grouped.loc[(region_df_grouped['mouse']=='dop_47') & (region_df_grouped['date']=='2022-06-06') & 
(region_df_grouped['probe']==2) ]


#
sessions_n = ephys_ephys_dataset(1)
sessions_n[0] = sessions[16]

p_df = pd.DataFrame()
for i, roi in enumerate(rois):
    region_df = psths_per_regions(sessions_n, roi=roi)
    p_df = pd.concat([p_df,region_df])

i=3
sns.heatmap(region_df.binned_spikes_laser_contra.apply(pca_mean).iloc[i], cmap='seismic')
plt.yticks(np.arange(len(region_df.cluster_selection.iloc[i])), region_df.cluster_selection.iloc[i])
plt.title(region_df.probe.iloc[i])

region_df.cluster_selection.iloc[0]

#  dop_47/2022-06-06/001 example clusters: 119 probe02, cluster 39 15 from  probe03, and 18 and 23 from probe02(these for the absence of spikes during stim)
