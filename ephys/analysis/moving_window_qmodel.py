import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import model_comparison_accu as m
from investigating_laser_expdecaymodel import *


def load_sim_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', LOAD_DF = False, REEXTRACT=False, trial_start=0,trial_end=-150):
    if LOAD_DF==False:
        root_path = Path(ROOT_FOLDER)
        psy = pd.DataFrame()
        for animal in root_path.iterdir():
            if animal.is_dir():
                for day in animal.iterdir():
                    if day.is_dir():
                        for ses in day.iterdir():
                            if ses.is_dir():
                                if REEXTRACT==True:
                                    full_bandit_fix(ses.as_posix())
                                mouse_psy = pd.DataFrame()
                                mouse_psy['feedback'] = \
                                    1*(np.load(ses.joinpath('alf', '_ibl_trials.feedbackType.npy'))>0)[trial_start:trial_end]
                                mouse_psy['choices'] = \
                                    1*((-1*np.load(ses.joinpath('alf', '_ibl_trials.choice.npy')))>0)[trial_start:trial_end]
                                mouse_psy['laser_block'] = \
                                    np.load(ses.joinpath('alf', '_ibl_trials.opto_block.npy'))[trial_start:trial_end]
                                mouse_psy['probabilityLeft'] = \
                                    np.load(ses.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]
                                mouse_psy['laser'] = mouse_psy['feedback']*mouse_psy['laser_block']
                                mouse_psy['reward'] = mouse_psy['feedback']*(1*(mouse_psy['laser_block']==0))
                                mouse_psy['mouse'] = animal.name
                                mouse_psy['ses'] = ses.name
                                mouse_psy['date'] = day.name
                                psy = pd.concat([psy, mouse_psy])
                            else:
                                continue
                    else:
                        continue
            else:
                continue
    else:
        psy = pd.read_pickle("psy.pkl")
    # Prepara data for Stan​
    # 1) optomization variables
    NS = len(psy['mouse'].unique())
    NSESS = psy.groupby(['mouse','date','ses']).size().reset_index()['mouse'].value_counts().max() # Maximum number of sessions
    NT = psy.groupby(['mouse','date','ses']).size().reset_index()[0].max()
    NSxNSESS = NS*NSESS
    NT_all = np.zeros(NSxNSESS)
    sub_idx = np.zeros(NSxNSESS) #  Mapping parameter to subject
    for i in np.arange(psy['mouse'].unique().shape[0]):
        sub_idx[i*NSESS:(i+1)*NSESS]=i
    sess_idx =  np.zeros(NSxNSESS) # Mapping parameter to session
    folder_dic = pd.DataFrame(columns=['mouse', 'day', 'ses', 'sess_idx', 'NT_all']) # Will hold relation between arrays and folders
    # 2) trial variables
    r = np.zeros([NS,NSESS,NT])
    c = np.zeros([NS,NSESS,NT])
    l = np.zeros([NS,NSESS,NT])
    p = np.zeros([NS,NSESS,NT])
    b = np.zeros([NS,NSESS,NT])
    tb = np.zeros([NS,NSESS,NT])
    for ns, mouse in enumerate(psy['mouse'].unique()):
        animal = psy.loc[psy['mouse']==mouse]
        counter=0
        for d, day in enumerate(animal['date'].unique()):
            day_s = animal.loc[animal['date']==day]
            for nsess, ses in enumerate(day_s['ses'].unique()):
                session = day_s.loc[day_s['ses']==ses]
                r[ns, counter, :len(session)] = session['reward']
                c[ns, counter, :len(session)] = session['choices']
                l[ns, counter, :len(session)] = session['laser']
                b[ns, counter, :len(session)] = session['laser_block']
                p[ns, counter, :len(session)] = session['probabilityLeft']
                bchange = 1*(np.diff(session['probabilityLeft'])!=0)
                for t in np.arange(len(session)):
                    if t==0:
                        twb = 1
                        tb[ns, counter, t] = twb
                        twb+=1
                    elif t==len(session)-1:
                        tb[ns, counter, t] = twb
                    else:
                        if bchange[t]==1:
                            tb[ns, counter, t] = twb
                            twb = 1
                        if bchange[t]==0:
                            tb[ns, counter, t] = twb
                            twb+=1
                sess_idx[(ns*NSESS)+counter] = counter
                NT_all[(ns*NSESS)+counter] = len(session)
                counter+=1
                ses_df = pd.DataFrame()
                ses_df['mouse'] = [mouse]
                ses_df['day'] = [day]
                ses_df['ses'] = [ses]
                ses_df['sess_idx'] = [counter]
                ses_df['NT_all'] = [len(session)]
                folder_dic = pd.concat([folder_dic, ses_df])

    standata_recovery = {'NS': int(NS) ,'NT': int(NT),'NSESS': int(NSESS),
                'b':b.astype(int), 'p':p.astype(float), 'tb':tb.astype(int),
               'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
               'sess_idx':np.array(sess_idx).astype(int)+1,
               'sub_idx':np.array(sub_idx).astype(int)+1,}

    return standata_recovery



def q_learning_model(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        q_init = standata['q_init']
        qlaser_init = standata['qlaser_init']
        qstay_init = standata['qstay_init']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
            q = q_init[sub_idx[ms_i], sess_idx[ms_i],:]
            qstay = qstay_init[sub_idx[ms_i], sess_idx[ms_i],:]
            qlaser = qlaser_init[sub_idx[ms_i], sess_idx[ms_i],:]
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data

def simulate_q_learning_model(std_recovery,saved_params=None, fit=None, csv=True,
            q_inits = None, qstay_inits =None, qlaser_inits =None, t_start=None,t_end=None):
    standata_recovery = copy.copy(std_recovery)      
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =10
    NT_all =standata_recovery['NT_all']

    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()

    q_inits_o=[]
    qlaser_inits_o=[]
    qstay_inits_o=[]
    counter=0
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())
            if q_inits is None:
                q = np.zeros(2)
                qstay = np.zeros(2)
                qlaser = np.zeros(2)
            else:
                q = q_inits[counter]
                qstay = qstay_inits[counter]
                qlaser = qlaser_inits[counter]
            counter+=1
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            QLlaser=[]
            QRlaser=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(t_start,t_end):
                if t_end > NT_all[ms_i]:
                    continue
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables
                predicted_choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1])+(stay_mouse*qstay[1]))
            q_inits_o.append(q)
            qlaser_inits_o.append(qlaser)
            qstay_inits_o.append(qstay)

            if (NT_all[ms_i]!=0) &  (t_end < NT_all[ms_i]):
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],t_start:t_end]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],t_start:t_end]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],t_start:t_end]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]

                sim_data = pd.concat([sim_data,ses_data])

    return q_inits_o, qlaser_inits_o, qstay_inits_o, sim_data

ROOT =  '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/standard_moving_window/output/'
ITERS = 49
SES_NUM = [0,1,6,7,13,14,15,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,36,37,38]
ANIMALS = 7 
# Load data
parameters_summary = pd.DataFrame()
animal_parameters_summary  = pd.DataFrame()
population_parameters_summary = pd.DataFrame()
for i in np.arange(ITERS):
    for j in SES_NUM:
        session_params = pd.DataFrame()
        saved_params = pd.read_csv(ROOT+str(int(i))+'summary.csv')
        session_params['βWater'] = [float(saved_params.loc[saved_params['name']=='beta_ses['+str(j+1)+']', 'Mean'])]
        session_params['βLaser'] = [float(saved_params.loc[saved_params['name']=='laser_ses['+str(j+1)+']', 'Mean'])]
        session_params['βStay'] = [float(saved_params.loc[saved_params['name']=='stay_ses['+str(j+1)+']', 'Mean'])]
        session_params['αWater'] = [float(saved_params.loc[saved_params['name']=='alpha_ses['+str(j+1)+']', 'Mean'])]
        session_params['αLaser'] = [float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(j+1)+']', 'Mean'])]
        session_params['αStay'] = [float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(j+1)+']', 'Mean'])]
        session_params['Bias'] = [float(saved_params.loc[saved_params['name']=='sides['+str(j+1)+']', 'Mean'])]
        session_params['ses'] = [j]
        session_params['Trials'] = [10*i]
        parameters_summary = pd.concat([parameters_summary,session_params])

parameters_summary['laser_ corr'] = np.corrcoef(parameters_summary['αLaser'],
                                                parameters_summary['βLaser'])


summ = parameters_summary.groupby(['Trials','ses']).mean().reset_index()

coffs = []
for i in summ.Trials.unique():
    coffs.append(np.corrcoef(summ.loc[summ['Trials']==i, 'αLaser'], 
                summ.loc[summ['Trials']==i, 'βLaser'])[0,1])

plt.scatter(summ.Trials.unique(),coffs)


# Plot alpha beta laser relation
sum200 = parameters_summary.loc[parameters_summary['Trials']<200]
sum200['Section'] = '>100'
sum200.loc[sum200['Trials']<100, 'Section'] = '<100'
sns.scatterplot(data = sum200, x = 'αLaser', 
        y = 'βLaser', hue='Section', palette=['magenta','red'])
sns.despine()

# Plot results session level
fig, ax = plt.subplots(2,4, sharex=True)
plt.subplots_adjust(wspace=0.4)
plt.sca(ax[0,0])
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50), rotation=90)
sns.pointplot(data=parameters_summary, x='Trials', y='βWater', hue='ses', color='dodgerblue')
plt.ylim(-0.6,2.5)
plt.legend().remove()
plt.sca(ax[0,1])
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50), rotation=90)
sns.pointplot(data=parameters_summary, x='Trials', y='βLaser', hue='ses', color='orange')
plt.legend().remove()
plt.ylim(-0.6,2.5)
plt.sca(ax[0,2])
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50), rotation=90)
sns.pointplot(data=parameters_summary, x='Trials', y='βStay', hue='ses', color='gray')
plt.legend().remove()
plt.ylim(-0.6,2.5)
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50), rotation=90)
plt.sca(ax[1,0])
sns.pointplot(data=parameters_summary, x='Trials', y='αWater', hue='ses', color='dodgerblue')
plt.legend().remove()
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50), rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,1])
sns.pointplot(data=parameters_summary, x='Trials', y='αLaser', hue='ses', color='orange')
plt.legend().remove()
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50), rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,2])
sns.pointplot(data=parameters_summary, x='Trials', y='αStay', hue='ses', color='gray')
plt.legend().remove()
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50), rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,3])
sns.pointplot(data=parameters_summary, x='Trials', y='Bias', hue='ses', color='red')
plt.legend().remove()
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50), rotation=90)
sns.despine()


# Plot accuracy






for i in np.arange(ITERS):
    for j in np.arange(ANIMALS):
        session_params = pd.DataFrame()
        saved_params = pd.read_csv(ROOT+str(int(i))+'summary.csv')
        session_params['βWater'] = [float(saved_params.loc[saved_params['name']=='beta_mouse['+str(j+1)+']', 'Mean'])]
        session_params['βLaser'] = [float(saved_params.loc[saved_params['name']=='laser_mouse['+str(j+1)+']', 'Mean'])]
        session_params['βStay'] = [float(saved_params.loc[saved_params['name']=='stay_mouse['+str(j+1)+']', 'Mean'])]
        session_params['αWater'] = [float(saved_params.loc[saved_params['name']=='alpha_mouse['+str(j+1)+']', 'Mean'])]
        session_params['αLaser'] = [float(saved_params.loc[saved_params['name']=='alphalaser_mouse['+str(j+1)+']', 'Mean'])]
        session_params['αStay'] = [float(saved_params.loc[saved_params['name']=='alphastay_mouse['+str(j+1)+']', 'Mean'])]
        session_params['Bias'] = [float(saved_params.loc[saved_params['name']=='side_mouse['+str(j+1)+']', 'Mean'])]
        session_params['ses'] = [j]
        session_params['Trials'] = [10*i]
        animal_parameters_summary = pd.concat([animal_parameters_summary,session_params])
    

# Plot results
fig, ax = plt.subplots(2,4, sharex=True)
plt.subplots_adjust(wspace=0.4)
plt.sca(ax[0,0])
sns.pointplot(data=animal_parameters_summary, x='Trials', y='βWater', hue='ses', color='dodgerblue')
plt.ylim(-0.6,2.5)
plt.legend().remove()
plt.sca(ax[0,1])
sns.pointplot(data=animal_parameters_summary, x='Trials', y='βLaser', hue='ses', color='orange')
plt.legend().remove()
plt.ylim(-0.6,2.5)
plt.sca(ax[0,2])
sns.pointplot(data=animal_parameters_summary, x='Trials', y='βStay', hue='ses', color='gray')
plt.legend().remove()
plt.ylim(-0.6,2.5)
plt.sca(ax[1,0])
sns.pointplot(data=animal_parameters_summary, x='Trials', y='αWater', hue='ses', color='dodgerblue')
plt.legend().remove()
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,1])
sns.pointplot(data=animal_parameters_summary, x='Trials', y='αLaser', hue='ses', color='orange')
plt.legend().remove()
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,2])
sns.pointplot(data=animal_parameters_summary, x='Trials', y='αStay', hue='ses', color='gray')
plt.legend().remove()
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,3])
sns.pointplot(data=animal_parameters_summary, x='Trials', y='Bias', hue='ses', color='red')
plt.legend().remove()
plt.xticks(rotation=90)
sns.despine()




population_parameters_summary=pd.DataFrame()
for i in np.arange(ITERS):
    saved_params = pd.read_csv(ROOT+str(int(i))+'summary.csv')
    session_params['βWater'] = [float(saved_params.loc[saved_params['name']=='betam', 'Mean'])]
    session_params['βLaser'] = [float(saved_params.loc[saved_params['name']=='laserm', 'Mean'])]
    session_params['βStay'] = [float(saved_params.loc[saved_params['name']=='staym', 'Mean'])]
    session_params['αWater'] = [float(saved_params.loc[saved_params['name']=='alpham', 'Mean'])]
    session_params['αLaser'] = [float(saved_params.loc[saved_params['name']=='alphalaserm', 'Mean'])]
    session_params['αStay'] = [float(saved_params.loc[saved_params['name']=='alphastaym', 'Mean'])]
    session_params['Bias'] = [float(saved_params.loc[saved_params['name']=='sidem', 'Mean'])]
    session_params['ses'] = [j]
    session_params['Trials'] = [10*i]
    population_parameters_summary =  pd.concat([population_parameters_summary,session_params])


# Plot results
fig, ax = plt.subplots(2,4, sharex=True)
plt.subplots_adjust(wspace=0.4)
plt.sca(ax[0,0])
sns.pointplot(data=population_parameters_summary, x='Trials', y='βWater', color='dodgerblue')
plt.ylim(-0.6,2.5)
plt.legend().remove()
plt.sca(ax[0,1])
sns.pointplot(data=population_parameters_summary, x='Trials', y='βLaser', color='orange')
plt.legend().remove()
plt.ylim(-0.6,2.5)
plt.sca(ax[0,2])
sns.pointplot(data=population_parameters_summary, x='Trials', y='βStay', color='gray')
plt.legend().remove()
plt.ylim(-0.6,2.5)
plt.sca(ax[1,0])
sns.pointplot(data=population_parameters_summary, x='Trials', y='αWater', color='dodgerblue')
plt.legend().remove()
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,1])
sns.pointplot(data=population_parameters_summary, x='Trials', y='αLaser', color='orange')
plt.legend().remove()
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,2])
sns.pointplot(data=population_parameters_summary, x='Trials', y='αStay', color='gray')
plt.legend().remove()
plt.xticks(rotation=90)
plt.ylim(0,1)
plt.sca(ax[1,3])
sns.pointplot(data=population_parameters_summary, x='Trials', y='Bias', color='red')
plt.legend().remove()
plt.xticks(rotation=90)
sns.despine()

## Accuracy
def calculate_block_acc(model, mean=True):
    model['b_predicted_choice'] =  1*(model['predicted_choice']>0.5)
    model['match'] = model['b_predicted_choice']==model['choices']
    if mean==True:
        accs = model.groupby(['id','laser_block']).mean()['match'].groupby(['laser_block']).mean().to_numpy()
        return accs
    else:
        acc = model.groupby(['id','laser_block']).mean()['match'].reset_index()
        accs_laser = acc.loc[acc['laser_block']==1]
        accs_water = acc.loc[acc['laser_block']==0]
        return accs_water['match'].to_numpy(), accs_laser['match'].to_numpy()


accu = pd.DataFrame()
for i in np.arange(ITERS):
    print(i)
    ses = pd.DataFrame()
    data = m.load_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', REEXTRACT=False, trial_start=10*i,trial_end=(10*i)+150)
    standata = m.make_stan_data(data)
    saved_params = pd.read_csv(ROOT+str(int(i))+'summary.csv')
    saved_params_std = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output/summary.csv')
    model = m.q_learning_model(standata,saved_params=saved_params)
    model['laser_block'] = data['laser_block'].copy()
    model['id'] = model['mouse']*100+model['ses']
    accs = calculate_block_acc(model)
    ses['accuracy'] = [np.mean(model.groupby(['id']).mean()['acc'])]
    ses['accuracy_laser_blocks'] = [accs[1]]
    ses['accuracy_water_blocks'] = [accs[0]]
    model_standard = m.q_learning_model(standata,saved_params=saved_params_std)
    model_standard['id'] = model_standard['mouse']*100+model_standard['ses']
    model_standard['laser_block'] = data['laser_block'].copy()
    accs_standard = calculate_block_acc(model_standard)
    ses['accuracy_standard'] = [np.mean(model_standard.groupby(['id']).mean()['acc'])]
    ses['accuracy_standard_laser'] = [accs_standard[1]]
    ses['accuracy_standard_water'] = [accs_standard[0]]
    ses['Trials'] = [10*i]
    accu = pd.concat([ses,accu])


fig, ax  =plt.subplots(1,2)
plt.sca(ax[0])
sns.pointplot(data=accu, x='Trials', y='accuracy', color='k')
sns.pointplot(data=accu, x='Trials', y='accuracy_standard', color='gray', alpha=0.5)
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50))
plt.ylabel('Accuracy')
plt.sca(ax[1])
sns.pointplot(data=accu, x='Trials', y='accuracy_laser_blocks', color='orange')
sns.pointplot(data=accu, x='Trials', y='accuracy_standard_laser', color='gold', alpha=0.5)
sns.pointplot(data=accu, x='Trials', y='accuracy_water_blocks', color='dodgerblue')
sns.pointplot(data=accu, x='Trials', y='accuracy_standard_water', color='cyan', alpha=0.5)
plt.xticks(np.arange(0,ITERS,5),np.arange(0,490,50))
plt.ylabel('Accuracy')
sns.despine()


# Now the same but only evaluating the 10 trials that change


accu_10 = pd.DataFrame()
sim_10 = pd.DataFrame()
sim_10_std = pd.DataFrame()
for i in np.arange(ITERS):
    print(i)
    ses = pd.DataFrame()
    data = m.load_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', REEXTRACT=False, trial_start=10*i,trial_end=(10*i)+10)
    standata = m.make_stan_data(data)
    standata_recovery = load_sim_data(trial_start=10*i,trial_end=(10*i)+10)
    saved_params = pd.read_csv(ROOT+str(int(i))+'summary.csv')
    saved_params_std = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output/summary.csv')
    if i==0:
        q_inits, qlaser_inits, qstay_inits, sim_standard = simulate_q_learning_model(standata_recovery,saved_params=saved_params,
                                                            q_inits = None, qstay_inits = None, qlaser_inits = None)      
    else:
        q_inits, qlaser_inits, qstay_inits, sim_standard = simulate_q_learning_model(standata_recovery,saved_params=saved_params,
                                                            q_inits = q_inits, qstay_inits = qstay_inits, qlaser_inits = qlaser_inits)
    model = m.q_learning_model(standata,saved_params=saved_params)
    model['laser_block'] = data['laser_block'].copy()
    model['id'] = model['mouse']*100+model['ses']
    accs = calculate_block_acc(model)
    ses['accuracy'] = [np.mean(model.groupby(['id']).mean()['acc'])]
    ses['accuracy_laser_blocks'] = [accs[1]]
    ses['accuracy_water_blocks'] = [accs[0]]
    model_standard = m.q_learning_model(standata,saved_params=saved_params_std)
    model_standard['id'] = model_standard['mouse']*100+model_standard['ses']
    model_standard['laser_block'] = data['laser_block'].copy()
    accs_standard = calculate_block_acc(model_standard)
    if i==0:
        q_inits_std, qlaser_inits_std, qstay_inits_std, sim_standard_std = simulate_q_learning_model(standata_recovery,saved_params=saved_params_std,
                                                            q_inits = None, qstay_inits = None, qlaser_inits = None)      
    else:
        q_inits_std, qlaser_inits_std, qstay_inits_std, sim_standard_std = simulate_q_learning_model(standata_recovery,saved_params=saved_params_std,
                                                            q_inits = q_inits, qstay_inits = qstay_inits, qlaser_inits = qlaser_inits)
    ses['accuracy_standard'] = [np.mean(model_standard.groupby(['id']).mean()['acc'])]
    ses['accuracy_standard_laser'] = [accs_standard[1]]
    ses['accuracy_standard_water'] = [accs_standard[0]]
    ses['Trials'] = [10*i]
    accu_10 = pd.concat([accu_10,ses])
    sim_10 = pd.concat([sim_10,sim_standard])
    sim_10_std = pd.concat([sim_10_std,sim_standard_std])


accu_10 = pd.DataFrame()
for i in np.arange(ITERS):
    print(i)
    ses = pd.DataFrame()
    data = m.load_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', REEXTRACT=False, trial_start=10*i,trial_end=(10*i)+10)
    standata = m.make_stan_data(data)
    standata_recovery = load_sim_data(trial_start=10*i,trial_end=(10*i)+10)
    saved_params = pd.read_csv(ROOT+str(int(i))+'summary.csv')
    saved_params_std = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output/summary.csv')
    model = m.q_learning_model(standata,saved_params=saved_params)
    model['laser_block'] = data['laser_block'].copy()
    model['id'] = model['mouse']*100+model['ses']
    accs = calculate_block_acc(model)
    ses['accuracy'] = [np.mean(model.groupby(['id']).mean()['acc'])]
    ses['accuracy_laser_blocks'] = [accs[1]]
    ses['accuracy_water_blocks'] = [accs[0]]
    model_standard = m.q_learning_model(standata,saved_params=saved_params_std)
    model_standard['id'] = model_standard['mouse']*100+model_standard['ses']
    model_standard['laser_block'] = data['laser_block'].copy()
    accs_standard = calculate_block_acc(model_standard)
    ses['accuracy_standard'] = [np.mean(model_standard.groupby(['id']).mean()['acc'])]
    ses['accuracy_standard_laser'] = [accs_standard[1]]
    ses['accuracy_standard_water'] = [accs_standard[0]]
    ses['Trials'] = [10*i]
    accu_10 = pd.concat([accu_10,ses])



sim_10 = pd.DataFrame()
sim_10_std = pd.DataFrame()
standata_recovery = load_sim_data()

ROOT = '/Users/alexpan/Desktop/output/'
for j in np.arange(100):
    for i in np.arange(ITERS):
        print(i)
        ses = pd.DataFrame()
        saved_params = pd.read_csv(ROOT+str(int(i))+'summary.csv')
        saved_params_std = pd.read_csv('/Users/alexpan/Desktop/summary.csv')
        if i==0:
            q_inits, qlaser_inits, qstay_inits, sim_standard = simulate_q_learning_model(standata_recovery,saved_params=saved_params,
                                                                q_inits = None, qstay_inits = None, qlaser_inits = None, t_start=10*i,t_end=(10*i)+10)      
        else:
            q_inits, qlaser_inits, qstay_inits, sim_standard = simulate_q_learning_model(standata_recovery,saved_params=saved_params,
                                                                q_inits = q_inits, qstay_inits = qstay_inits, qlaser_inits = qlaser_inits,t_start=10*i,t_end=(10*i)+10)    
        if i==0:
            q_inits_std, qlaser_inits_std, qstay_inits_std, sim_standard_std = simulate_q_learning_model(standata_recovery,saved_params=saved_params_std,
                                                                q_inits = None, qstay_inits = None, qlaser_inits = None, t_start=10*i,t_end=(10*i)+10)      
        else:
            q_inits_std, qlaser_inits_std, qstay_inits_std, sim_standard_std = simulate_q_learning_model(standata_recovery,saved_params=saved_params_std,
                                                                q_inits = q_inits_std, qstay_inits = qstay_inits_std, qlaser_inits = qlaser_inits_std,
                                                                t_start=10*i,t_end=(10*i)+10)
        sim_standard['bin'] = 10*i                                       
        sim_standard_std['bin'] = 10*i
        sim_standard['sim_no']=j
        sim_standard_std['sim_no']=j
        sim_10 = pd.concat([sim_10,sim_standard])
        sim_10_std = pd.concat([sim_10_std,sim_standard_std])

sim_10['trial'] = np.nan
for mouse in sim_10.mouse.unique():
    for ses in sim_10.loc[sim_10['mouse']==mouse].ses.unique():
        for sim in sim_10.loc[(sim_10['mouse']==mouse) & (sim_10['ses']==ses)].sim_no.unique():
            sim_10.loc[(sim_10['mouse']==mouse) & (sim_10['ses']==ses) 
                        & (sim_10['sim_no']==sim), 'trial'] = \
                np.arange(len(sim_10.loc[(sim_10['mouse']==mouse) & (sim_10['ses']==ses) 
                        & (sim_10['sim_no']==sim)]))

sim_10['id'] = sim_10['mouse']*100+sim_10['ses']+sim_10['sim_no']/100
sim_10['dLaser'] = sim_10['QRlaser'] - sim_10['QLlaser']
sim_10['dwater'] = sim_10['QRreward'] - sim_10['QLreward']

sns.lineplot(data = sim_10, x='trial', y='dLaser', color='orange', ci=95)
sns.lineplot(data = sim_10, x='trial', y='dwater', color='dodgerblue', ci=95)
plt.xlabel('Trial')
plt.ylabel('deltaQ')

fig, ax = plt.subplots(2,1,sharey=True)
plt.sca(ax[0])
sns.lineplot(data = sim_10, x='trial', y='dLaser', palette='Oranges', ci=None, hue='id')
plt.xlabel('Trial')
plt.ylabel('DeltaQWater')
plt.sca(ax[1])
sns.lineplot(data = sim_10, x='trial', y='dwater', palette='Blues', ci=None,hue='id')
plt.xlabel('Trial')
plt.ylabel('DeltaQLaser')
ax[0].get_legend().remove()
ax[1].get_legend().remove()




fig, ax = plt.subplots(1,2,sharey=True)
plt.sca(ax[0])
pal = ['dodgerblue', 'orange']
sim_10['high_prob_choice'] = sim_10['choices'] == (sim_10['probabilityLeft']==0.1)
sim_10['section'] = pd.cut(sim_10['bin'],5)
sim_10_end = sim_10.loc[sim_10['tb']>5]
sns.pointplot(data = sim_10_end, x='section', y='high_prob_choice', hue='laser_block', palette=pal)
plt.ylabel('High Probability Choice')
plt.xlabel('Trial')
plt.xticks(np.arange(5),['0:100','100:200','200:300', '300:400', '400:500'])
plt.title('Moving window Q-learning Simulation')
plt.ylim(0.25,0.75)
plt.hlines(0.5,0,4, linestyles='dashed', color='k')
plt.sca(ax[1])
plt.title('Standard Q-learning Simulation')
sim_10_std['high_prob_choice'] = sim_10_std['choices'] == (sim_10_std['probabilityLeft']==0.1)
sim_10_std['section'] = pd.cut(sim_10_std['bin'],5)
sim_10_end = sim_10_std.loc[sim_10_std['tb']>5]
sns.pointplot(data = sim_10_end, x='section', y='high_prob_choice', hue='laser_block', palette=pal)
plt.ylabel('High Probability Choice')
plt.xlabel('Trial')
plt.xticks(np.arange(5),['0:100','100:200','200:300', '300:400', '400:500'])
plt.ylim(0.25,0.75)
plt.hlines(0.5,0,4, linestyles='dashed', color='k')


fig, ax  =plt.subplots(1,2)
plt.sca(ax[0])
sns.pointplot(data=accu_10, x='Trials', y='accuracy', color='k')
sns.pointplot(data=accu_10, x='Trials', y='accuracy_standard', color='gray', alpha=0.5)
plt.sca(ax[1])
sns.pointplot(data=accu_10, x='Trials', y='accuracy_laser_blocks', color='orange')
sns.pointplot(data=accu_10, x='Trials', y='accuracy_standard_laser', color='gold', alpha=0.5)
sns.pointplot(data=accu_10, x='Trials', y='accuracy_water_blocks', color='dodgerblue')
sns.pointplot(data=accu_10, x='Trials', y='accuracy_standard_water', color='cyan', alpha=0.5)
sns.despine()



############# with error bars

accu = pd.DataFrame()
for i in np.arange(ITERS):
    print(i)
    ses = pd.DataFrame()
    data = m.load_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', REEXTRACT=False, trial_start=10*i,trial_end=(10*i)+150)
    standata = m.make_stan_data(data)
    saved_params = pd.read_csv(ROOT+str(int(i))+'summary.csv')
    saved_params_std = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output/summary.csv')
    model = m.q_learning_model(standata,saved_params=saved_params)
    model['laser_block'] = data['laser_block'].copy()
    model['id'] = model['mouse']*100+model['ses']
    accs_water, accs_laser = calculate_block_acc(model,  mean=False)
    ses['accuracy'] = model.groupby(['id']).mean()['acc']
    ses['accuracy_laser_blocks'] = accs_laser
    ses['accuracy_water_blocks'] = accs_water
    model_standard = m.q_learning_model(standata,saved_params=saved_params_std)
    model_standard['id'] = model_standard['mouse']*100+model_standard['ses']
    model_standard['laser_block'] = data['laser_block'].copy()
    accs_water, accs_laser = calculate_block_acc(model_standard, mean=False)
    ses['accuracy_standard'] = model_standard.groupby(['id']).mean()['acc']
    ses['accuracy_standard_laser'] = accs_laser
    ses['accuracy_standard_water'] = accs_water
    ses['Trials'] = 10*i
    accu = pd.concat([accu,ses])


fig, ax  =plt.subplots(1,2,sharey=True)
plt.sca(ax[0])
sns.lineplot(data=accu.reset_index(), x='Trials', y='accuracy', color='k', ci=66)
sns.lineplot(data=accu.reset_index(), x='Trials', y='accuracy_standard', color='gray', alpha=0.5, ci=66)
plt.ylabel('Accuracy')
plt.sca(ax[1])
sns.lineplot(data=accu.reset_index(), x='Trials', y='accuracy_laser_blocks', color='orange', ci=66)
sns.lineplot(data=accu.reset_index(), x='Trials', y='accuracy_standard_laser', color='gold', alpha=0.5, ci=66)
sns.lineplot(data=accu.reset_index(), x='Trials', y='accuracy_water_blocks', color='dodgerblue', ci=66)
sns.lineplot(data=accu.reset_index(), x='Trials', y='accuracy_standard_water', color='cyan', alpha=0.5, ci=66)
plt.ylabel('Accuracy')
sns.despine()

######
