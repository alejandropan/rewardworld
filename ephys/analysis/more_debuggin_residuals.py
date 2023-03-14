neural_data = neural_data.loc[neural_data['location']==area]
binned_spikes = np.zeros([len(neural_data['residuals_goCue']), 15])
for i in np.arange(len(neural_data['residuals_goCue'])):
    binned_spikes[i,:] = neural_data['residuals_goCue'].iloc[i].mean(axis=0)
order = np.argmax(binned_spikes, 1)
xs_sorted = binned_spikes[order.argsort(),:]

p_neural_data = p_neural_data.loc[p_neural_data['location']==area]
p_c_neural_data_res = common_neural_data(p_neural_data, n_trials_minimum=100)
p_binned_spikes = np.zeros([len(p_c_neural_data_res['residuals_goCue']), 15])
for i in np.arange(len(p_c_neural_data_res['residuals_goCue'])):
    p_binned_spikes[i,:] = p_c_neural_data_res['residuals_goCue'].iloc[i].mean(axis=0)
p_xs_sorted = p_binned_spikes[order.argsort(),:]

r_neural_data = r_neural_data.loc[r_neural_data['location']==area]
r_c_neural_data_res = common_neural_data(r_neural_data, n_trials_minimum=100)
r_binned_spikes = np.zeros([len(r_c_neural_data_res['residuals_goCue']), 15])
for i in np.arange(len(r_c_neural_data_res['residuals_goCue'])):
    r_binned_spikes[i,:] = r_c_neural_data_res['residuals_goCue'].iloc[i].mean(axis=0)
r_xs_sorted = r_binned_spikes[order.argsort(),:]

fig, ax = plt.subplots(1,4, sharey=True,  sharex=True)
plt.sca(ax[0])
sns.heatmap(xs_sorted, vmin=-0.5, vmax=0.5, center=0, cmap="seismic", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.ylabel('Cluster')
plt.title('Raw Data')
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from goCue')
plt.sca(ax[1])
sns.heatmap(p_xs_sorted, vmin=-0.5, vmax=0.5, center=0, cmap="seismic", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from goCue')
plt.title('Reduced model prediction')
plt.sca(ax[2])
sns.heatmap(xs_sorted - p_xs_sorted, vmin=-0.5, vmax=0.5, center=0, cmap="seismic", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from goCue')
plt.title('Real-Pred')
plt.sca(ax[3])
sns.heatmap(r_xs_sorted, vmin=-0.5, vmax=0.5, center=0, cmap="seismic", cbar=False)
plt.xticks(np.arange((1-(-0.5))/0.1)[::5], np.round(np.arange(-0.5,1,0.1)[1::5],2), rotation=90)
plt.xlabel('Time from goCue')
plt.title('Model Res')

fig.savefig('nacc_trial.svg')