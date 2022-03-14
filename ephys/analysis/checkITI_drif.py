ses = '/Volumes/witten/Alex/Data/Subjects/ibl_witten_13/2019-12-03/001'

a = pd.DataFrame()
a['start']= np.load(ses+'/alf/' + '_ibl_trials.intervals.npy')[:,0]
a['end_p']= np.load(ses+'/alf/' + '_ibl_trials.intervals.npy')[:,1]
a['end_p'] = a['end_p'].shift(1)
a['dITI']= a['start'] - a['end_p']
#a['s_block']= np.load(ses+'/alf/' + '_ibl_trials.probabilityLeft.npy')
#a['o_block'] =  np.load(ses+'/alf/' + '_ibl_trials.opto_block.npy')


#a['s_block'] = a['s_block'].diff()
#a['o_block'] = a['o_block'] .diff()

#b = a.reset_index()
#b = b.loc[b['s_block']!=0]
plt.plot(a['dITI'])
plt.ylabel('ITI to trial start (s)')
plt.xlabel('Trial no')

plt.scatter(b.index, b['dITI'],color = 'k')


b = a.reset_index()
b = b.loc[b['o_block']!=0]
plt.plot(a['dITI'])
plt.scatter(b.index, b['dITI'],color = 'k')


