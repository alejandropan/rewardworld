
selected_regions = [['NAcc', 'NAcsh'], ['DMS','VMS'], 'PFC', 'FA', 'VP', 'SPT', 'MOC']
for reg in selected_regions:
    counter=0
    for s in np.arange(len(sessions.sessions)):
        for p in np.arange(len(sessions[s].probe.probes)):
            loc_clusters = \
                np.where(np.isin(sessions[s].probe[p].cluster_group_locations,reg))[0]
            if len(loc_clusters)==0:
                continue
            counter+=1
            good_clusters = \
                np.where(sessions[s].probe[p].cluster_metrics=='good')[0]
            cluster_selection = np.intersect1d(loc_clusters,good_clusters)
            binned_fr = get_binned_spikes(sessions[s].probe[p].spike_times,
                            sessions[s].probe[p].spike_clusters,
                            cluster_selection, getattr(sessions[s],epoch_of_interest),
                            pre_time=pre_time, post_time=post_time,
                            bin_size=bin_size)/bin_size
            #Flatten matrix to n_neuron n_neuron x n_trials
            binned_fr_flat = np.squeeze(binned_fr,axis=2).T
            if transition_based==True:
                left_block = np.where(sessions[s].probabilityLeft==0.7)[0]
                right_block = np.where(sessions[s].probabilityLeft==0.1)[0]

                X = np.zeros([binned_fr_flat.shape[0],20])#Start matrix that will hold data
                X[:]=np.nan
                # This matrix has been triple checked test below
                for j,i in enumerate(np.arange(-5,5)):
                    if i<0:
                        X[:,5+j]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                    left_block)].mean(axis=1) # last 5 t of left
                        X[:,15+j]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                    right_block)].mean(axis=1) # last 5 t of right
                    if i>-1:
                        X[:,i]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                    left_block)].mean(axis=1) # first 5 t of left
                        X[:,10+i]=binned_fr_flat[:,np.intersect1d(np.where(sessions[s].trial_within_block==i)[0],
                                    right_block)].mean(axis=1) # first 5 t of right
                if counter==1:
                    X_reg=X
                else:
                    X_reg = np.concatenate([X_reg, X])

    #Plot PCA 3D
    Z_reg = scale(X_reg.T)
    pca = PCA(n_components=3)
    pca.fit(Z_reg)
    DX = pca.transform(Z_reg)
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(DX[:9, 0], DX[:9, 1], DX[:9, 2], color='blue')
    ax.plot(DX[:10, 0], DX[:10, 1], DX[:10, 2], color='blue')
    ax.scatter(DX[9, 0], DX[9, 1], DX[9, 2], marker='>', color='blue', s=50)
    ax.scatter(DX[10:19, 0], DX[10:19, 1], DX[10:19, 2], color='red') # last 5 t of right laser
    ax.plot(DX[10:20, 0], DX[10:20, 1], DX[10:20, 2], color='red') # last 5 t of right laser
    ax.scatter(DX[19, 0], DX[19, 1], DX[19, 2], marker='>', color='red', s=50) # last 5 t of left water
    ax.text2D(0.05, 0.95,reg, transform=ax.transAxes)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

    #Plot PCA 3D
    Z_reg = scale(X_reg.T)
    pca = PCA(n_components=3)
    pca.fit(Z_reg)
    DX = pca.transform(Z_reg)
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(DX[:, 0], DX[:, 1], DX[:, 2], c=np.arange(20), cmap='inferno')
    ax.plot(DX[:, 0], DX[:, 1], DX[:, 2])
    ax.text2D(0.05, 0.95,reg, transform=ax.transAxes)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

    #Plot PCA 3D
    Z_reg = scale(X_reg.T)
    pca = PCA(n_components=3)
    pca.fit(Z_reg)
    DX = pca.transform(Z_reg)
    fig = plt.figure(1, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    ax.scatter(DX[:9, 0], DX[:9, 1], DX[:9, 2], c=np.arange(9), cmap='Blues')
    ax.scatter(DX[9, 0], DX[9, 1], DX[9, 2], color='blue', s=50)
    ax.scatter(DX[10:19, 0], DX[10:19, 1], DX[10:19, 2], c=np.arange(9), cmap='Wistia') # last 5 t of right laser
    ax.scatter(DX[19, 0], DX[19, 1], DX[19, 2], color='red', s=50)
    ax.plot(DX[:, 0], DX[:, 1], DX[:, 2], color='k')
    ax.text2D(0.05, 0.95,reg, transform=ax.transAxes)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

    del X_reg
