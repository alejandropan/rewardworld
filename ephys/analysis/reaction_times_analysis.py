# Reaction times by value quantile

from ephys_alf_summary import *
from scipy.stats import ttest_rel, ttest_ind,ranksums, pearsonr

session_data = pd.DataFrame()
for i, ses in enumerate(LASER_ONLY):
    print(ses)
    alf_ses = alf(ses, ephys=False)
    ses_data = pd.DataFrame()
    ses_data['choice'] = 1*(alf_ses.choice>0)
    ses_data['pLeft'] = alf_ses.probabilityLeft
    ses_data['outcome'] = alf_ses.outcome
    qR = np.roll(alf_ses.fQR,1)
    qL = np.roll(alf_ses.fQL,1)
    qR[0] = 0
    qL[0] = 0 
    qchosen = np.copy(qR)
    qchosen[alf_ses.choice==-1] = np.copy(qL[alf_ses.choice==-1])
    ses_data['qchosen'] = qchosen
    qdelta = np.roll((alf_ses.fQR - alf_ses.fQL),1)
    qdelta[0] = 0 
    ses_data['qdelta'] = qdelta
    qRl = np.roll(alf_ses.fQRreward,1)
    qLl = np.roll(alf_ses.fQLreward,1)
    qRl[0] = 0
    qLl[0] = 0 
    qchosenl = np.copy(qRl)
    qchosenl[alf_ses.choice==-1] = np.copy(qLl[alf_ses.choice==-1])
    ses_data['qchosen_laser'] = qchosenl
    qdelta_l = np.roll((alf_ses.fQRreward - alf_ses.fQLreward),1)
    qdelta_l[0] = 0 
    ses_data['qdelta_laser'] = qdelta_l
    qRs = np.roll(alf_ses.fQRstay,1)
    qLs = np.roll(alf_ses.fQLstay,1)
    qRs[0] = 0
    qLs[0] = 0 
    qchosens = np.copy(qRs)
    qchosens[alf_ses.choice==-1] = np.copy(qLs[alf_ses.choice==-1])
    ses_data['qchosen_stay'] = qchosens
    qdelta_s = np.roll((alf_ses.fQRstay - alf_ses.fQLstay),1)
    qdelta_s[0] = 0
    ses_data['qdelta_stay'] = qdelta_s
    ses_data['first_move_time'] = alf_ses.first_move
    ses_data['goCue_time'] = alf_ses.goCue_trigger_times
    ses_data['response_time'] = alf_ses.response_times
    ses_data['reaction_time'] = ses_data['first_move_time'] - ses_data['goCue_time']
    ses_data['decision_time'] = ses_data['response_time'] - ses_data['goCue_time']
    ses_data['qchosen_terciles'] = np.digitize(ses_data['qchosen'], [ses_data['qchosen'].quantile(0.333), ses_data['qchosen'].quantile(0.75)])
    ses_data['laser_qchosen_terciles'] = np.digitize(ses_data['qchosen_laser'], [ses_data['qchosen_laser'].quantile(0.333), ses_data['qchosen_laser'].quantile(0.75)])
    ses_data['stay_qchosen_terciles'] = np.digitize(ses_data['qchosen_stay'], [ses_data['qchosen_stay'].quantile(0.333), ses_data['qchosen_stay'].quantile(0.75)])
    ses_data['delta_terciles'] = np.digitize(abs(ses_data['qdelta']), [abs(ses_data['qdelta'].quantile(0.333)), abs(ses_data['qdelta'].quantile(0.75))])
    ses_data['laser_delta_terciles'] = np.digitize(abs(ses_data['qdelta_laser']), [abs(ses_data['qdelta_laser'].quantile(0.333)), abs(ses_data['qdelta_laser'].quantile(0.75))])
    ses_data['stay_delta_terciles'] = np.digitize(abs(ses_data['qdelta_stay']), [abs(ses_data['qdelta_stay'].quantile(0.333)), abs(ses_data['qdelta_stay'].quantile(0.75))])
    ses_data['mouse'] = Path(ses).parent.parent.name
    ses_data['date'] = Path(ses).parent.name
    ses_data['ses'] = Path(ses).name
    ses_data['id'] =  ses_data.mouse+'_'+ses_data.date+'_'+ses_data.ses
    session_data = pd.concat([session_data, ses_data])

# Start plotting

# Drop trials without reliable goCue,first 10 trials and trials where no choice was made or choice took way too long
session_data = session_data.reset_index()
session_data = session_data.loc[session_data['index']>10]
session_data = session_data.loc[~(np.isnan(session_data['goCue_time']))]
session_data = session_data.loc[~(session_data['decision_time']==60)]


sns.barplot(data = session_data.groupby(['id','qchosen_terciles']).median().reset_index(), x='qchosen_terciles', y='decision_time', errorbar='se')

fig, ax = plt.subplots()
sns.barplot(data = session_data, x='qchosen_terciles', y='decision_time', errorbar='se', palette = 'Reds')
plt.xlabel('Chosen Value (QChosen)')
plt.ylabel('Time to decision')
plt.xticks(np.arange(3), ['Low', 'Middle', 'High'])
plt.title('Pearson - r %s' % np.round(pearsonr(session_data['qchosen'],session_data['decision_time'])[1], 3))
sns.despine()
