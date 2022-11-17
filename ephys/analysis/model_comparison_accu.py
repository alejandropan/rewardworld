# Import packages
import numpy as np
import seaborn as sns
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import glob
from session_maker import TrialParamHandler   
from scipy.stats import zscore
import scipy.stats as ss
# Common Functions

def inv_logit(arr):
    '''Elementwise inverse logit (logistic) function.'''
    return 1 / (1 + np.exp(-arr))
def phi_approx(arr):
    '''Elementwise fast approximation of the cumulative unit normal.
    For details, see Bowling et al. (2009). "A logistic approximation
    to the cumulative normal distribution."'''
    return inv_logit(0.07056 * arr ** 3 + 1.5976 * arr)

def  num_to_name(psy):
    ses_id=[]
    for ns, mouse in enumerate(psy['mouse'].unique()):
        animal = psy.loc[psy['mouse']==mouse]
        counter=0
        for d, day in enumerate(animal['date'].unique()):
            day_s = animal.loc[animal['date']==day]
            for nsess, ses in enumerate(day_s['ses'].unique()):
                ses_id.append(mouse+'_'+day+'_'+ses)
    return ses_id

def load_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', REEXTRACT=False, trial_start=0,trial_end=-150):
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
                            mouse_psy['laser'] = mouse_psy['feedback']*mouse_psy['laser_block']
                            mouse_psy['reward'] = mouse_psy['feedback']*(1*(mouse_psy['laser_block']==0))
                            mouse_psy['mouse'] = animal.name
                            mouse_psy['ses'] = ses.name
                            mouse_psy['date'] = day.name
                            mouse_psy['probabilityLeft'] = \
                                np.load(ses.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]

                            psy = pd.concat([psy, mouse_psy])
                        else:
                            continue
                else:
                    continue
        else:
            continue
    return psy
def load_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', REEXTRACT=False, trial_start=0,trial_end=-150):
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
                            mouse_psy['reward'] = mouse_psy['feedback']
                            mouse_psy['mouse'] = animal.name
                            mouse_psy['ses'] = ses.name
                            mouse_psy['date'] = day.name
                            mouse_psy['probabilityLeft'] = \
                                np.load(ses.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]

                            psy = pd.concat([psy, mouse_psy])
                        else:
                            continue
                else:
                    continue
        else:
            continue
    return psy


def make_stan_data(psy, n_trials=None):
    # Prepara data for Stan​
    # 0) First check if there is a limit on trials
    if n_trials is not None:
        psy = psy.reset_index()
        psy = psy.loc[psy['index']<=n_trials]
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

    # 2) trial variables
    r = np.zeros([NS,NSESS,NT])
    c = np.zeros([NS,NSESS,NT])
    l = np.zeros([NS,NSESS,NT])
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

    standata = {'NS': int(NS) ,'NT': int(NT),'NSESS': int(NSESS),
               'r':r.astype(int), 'c':c.astype(int), 'l':l.astype(int), 'b':b.astype(int),
               'tb':tb.astype(int),
               'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
               'sess_idx':np.array(sess_idx).astype(int)+1,
               'sub_idx':np.array(sub_idx).astype(int)+1}

    return standata
def make_stan_data_reduced(psy, n_trials=None):
    # Prepara data for Stan​
    # 0) First check if there is a limit on trials
    if n_trials is not None:
        psy = psy.reset_index()
        psy = psy.loc[psy['index']<=n_trials]
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

    # 2) trial variables
    r = np.zeros([NS,NSESS,NT])
    c = np.zeros([NS,NSESS,NT])
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

    standata = {'NS': int(NS) ,'NT': int(NT),'NSESS': int(NSESS),
               'r':r.astype(int), 'c':c.astype(int),
               'tb':tb.astype(int),
               'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
               'sess_idx':np.array(sess_idx).astype(int)+1,
               'sub_idx':np.array(sub_idx).astype(int)+1}

    return standata

def load_sim_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_reduced', LOAD_DF = False, REEXTRACT=False, trial_start=0,trial_end=-150):
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
                                mouse_psy['probabilityLeft'] = \
                                    np.load(ses.joinpath('alf', '_ibl_trials.probabilityLeft.npy'))[trial_start:trial_end]
                                mouse_psy['reward'] = mouse_psy['feedback']
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
    p = np.zeros([NS,NSESS,NT])
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
            'p':p.astype(float), 'tb':tb.astype(int),
               'NT_all':np.array(NT_all).astype(int), 'NSxNSESS':int(NSxNSESS),
               'sess_idx':np.array(sess_idx).astype(int)+1,
               'sub_idx':np.array(sub_idx).astype(int)+1}

    return standata_recovery



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
def load_qdata_from_file(ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects', prefix='QLearning_laserdecay_'):
    root_path = Path(ROOT_FOLDER)
    psy = pd.DataFrame()
    for animal in root_path.iterdir():
        if animal.is_dir():
            for day in animal.iterdir():
                if day.is_dir():
                    for ses in day.iterdir():
                        if ses.is_dir():
                            if ses.joinpath('alf', prefix+'water.npy').is_file():
                                mouse_psy = pd.DataFrame()
                                mouse_psy['DQwater'] = np.load(ses.joinpath('alf', prefix+'water.npy'))
                                mouse_psy['DQlaser'] = np.load(ses.joinpath('alf', prefix+'laser.npy'))
                                mouse_psy['choice_prediction'] = np.load(ses.joinpath('alf', prefix+'choice_prediction.npy'))
                                mouse_psy['QLstay'] = np.load(ses.joinpath('alf', prefix+'QLstay.npy'))
                                mouse_psy['QRstay'] = np.load(ses.joinpath('alf', prefix+'QRstay.npy'))
                                mouse_psy['DQstay'] =  mouse_psy['QRstay']- mouse_psy['QLstay']
                                mouse_psy['ses'] = str(ses)[-14:]
                                mouse_psy['mouse']= str(animal)[-6:]
                                psy = pd.concat([psy, mouse_psy])
                                try:
                                    mouse_psy['QL'] = np.load(ses.joinpath('alf', prefix+'QL.npy'))
                                    mouse_psy['QR'] = np.load(ses.joinpath('alf', prefix+'QR.npy'))
                                    mouse_psy['QLlaser'] = np.load(ses.joinpath('alf', prefix+'QLlaser.npy'))
                                    mouse_psy['QRlaser'] = np.load(ses.joinpath('alf', prefix+'QRlaser.npy'))
                                    mouse_psy['QLreward'] = np.load(ses.joinpath('alf', prefix+'QLreward.npy'))
                                    mouse_psy['QRreward'] = np.load(ses.joinpath('alf', prefix+'QRreward.npy'))
                                except:
                                    continue
                            else:
                                continue
                        else:
                            continue
                else:
                    continue
        else:
            continue
    return psy
def stan_data_to_df_reduced(standata_recovery,standata):
    r =  standata['r']
    c =  standata['c']
    tb = standata_recovery['tb']
    p = standata_recovery['p']
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    NT_all = standata_recovery['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
                ses_data = pd.DataFrame()
                ses_data['choices'] = c[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['water'] = r[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                data = pd.concat([data,ses_data])
    return data


def stan_data_to_df(standata_recovery,standata):
    r =  standata['r']
    c =  standata['c']
    l =  standata['l']
    b =  standata_recovery['b']
    tb = standata_recovery['tb']
    p = standata_recovery['p']
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    NT_all = standata_recovery['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
                ses_data = pd.DataFrame()
                ses_data['choices'] = c[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['water'] = r[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser'] = l[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                data = pd.concat([data,ses_data])
    return data
def load_cmd_stan_output(folder):
    chains = glob.glob(folder + '/*.csv')
    output = pd.DataFrame()
    for c,i in enumerate(chains):
        chain = pd.read_csv(i
            , skiprows= \
            np.concatenate([np.arange(42), np.array([44,45,46,43]),
            np.arange(1547,1552)]))
        chain['chain'] = c+1
        output = pd.concat([output,chain])
    return output



def chains_2_summary(chain_folder, mode=True):
    output = load_cmd_stan_output(chain_folder)
    if mode==True:
        output = pd.DataFrame(output.apply(ss.mode).iloc[0,:]).reset_index()
    else:
        output = pd.DataFrame(output.mean()).reset_index()
    output.rename(columns={'index':'name', 0:'Mean'}, inplace=True)
    output.loc[(output['name'].str.contains('\d')),'name']=\
        output.loc[(output['name'].str.contains('\d')),'name']+']'
    output['name']=output['name'].str.replace('.','[')
    return output
def make_deltas(model_data):
    model_data['deltaQ'] = model_data['QR']-model_data['QL']
    model_data['Qwater'] = model_data['QRreward']-model_data['QLreward']
    try:
        model_data['Qstay'] = model_data['QRstay']-model_data['QLstay']
    except:
        print('No stay')
    model_data['Qlaser'] = model_data['QRlaser']-model_data['QLlaser']
    return model_data
def plot_params(saved_params, standata, save=False, output_path=None, phi_a=False):
    var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                'alphalaserforgetting_ses':'αLaserForget',
                'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                'balphastay_ses':'αBStay', 'alphalaserloss_ses':'αLaserLoss', 'alphaloss_ses':'αWaterLoss',
                'laserdecay_ses':'αLaserDecay', 'laserdecayloss_ses':'αLaserDecayLoss',
                'betalaserlossbase_ses':'βLaserLossBase', 'betalaserbase_ses':'βLaserBase', 'ep_ses':'βStay'}
    var_order = ['βWater','βLaser','βStay', 'βWaterLoss', 'βLaserLoss','βLaserStay','βLaserLossBase','βLaserBase',
                'αWater','αLaser','αStay','αWaterForget','αLaserForget','αWaterLoss','αLaserLoss','αStayLaser',
                'αBWater','αBLaser','αBStay','αLaserDecay','αLaserDecayLoss','ε','εStay','Bias']
    pal = ['dodgerblue','orange','gray', 'dodgerblue', 'orange','gray','orange','orange',
                'dodgerblue','orange','gray','dodgerblue','orange','dodgerblue','orange','gray',
                'dodgerblue','orange','gray','orange','orange','violet','gray','salmon']

    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    saved_params_new = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
    saved_params = saved_params_new
    params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
        (saved_params['name'].str.contains('s\[')),['name','Mean']]
    params['parameter']=params['name'].str.rstrip('[123456789102030]')
    params['parameter'] = params['parameter'].map(var_names)
    if phi_a==True:
        params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
    sns.swarmplot(data=params, x='parameter', y='Mean',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                color='k')
    sns.barplot(data=params, x='parameter', y='Mean', color='gray',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                palette=np.array(pal)[np.isin(var_order, params['parameter'].unique())])
    plt.ylabel('Coefficient')
    plt.xticks(rotation=90)
    sns.despine()
    if save==True:
        plt.savefig(filepath+'/params.pdf')

def plot_params_reduced(saved_params, standata, save=False, filepath=None, phi_a=False):
    var_names = {'sides':'Bias', 'alphalaser_ses':'αReward', 'alpha_ses':'αReward',
                'alphastay_ses':'αStay', 'laser_ses':'βReward',
                'stay_ses':'βStay', 'beta_ses': 'βReward', 'alphaforgetting_ses':'αRewardForget',
                'alphalaserforgetting_ses':'αRewardForget',
                'dmlaser_ses':'βRewardLoss', 'dmwater_ses':'βRewardLoss',
                'laserstay_ses':'βRewardStay', 'alphalaserstay_ses': 'αStayReward',
                'lapse_ses':'ε', 'balphalaser_ses':'αBReward', 'balpha_ses':'αBReward',
                'balphastay_ses':'αBStay', 'alphalaserloss_ses':'αRewardLoss', 'alphaloss_ses':'αRewardLoss',
                'laserdecay_ses':'αRewardDecay', 'laserdecayloss_ses':'αRewardDecayLoss',
                'betalaserlossbase_ses':'βLaserLossBase', 'betalaserbase_ses':'βLaserBase', 'ep_ses':'βStay'}
    var_order = ['βReward','βLaser','βStay', 'βRewardLoss', 'βLaserLoss','βLaserStay','βLaserLossBase','βLaserBase',
                'αReward','αLaser','αStay','αRewardForget','αLaserForget','αRewardLoss','αLaserLoss','αStayLaser',
                'αBReward','αBLaser','αBStay','αLaserDecay','αLaserDecayLoss','ε','εStay','Bias']
    pal = ['dodgerblue','orange','gray', 'dodgerblue', 'orange','gray','orange','orange',
                'dodgerblue','orange','gray','dodgerblue','orange','dodgerblue','orange','gray',
                'dodgerblue','orange','gray','orange','orange','violet','gray','salmon']
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    saved_params_new = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
    saved_params = saved_params_new
    params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
        (saved_params['name'].str.contains('s\[')),['name','Mean']]
    params['parameter']=params['name'].str.rstrip('[123456789102030]')
    params['parameter'] = params['parameter'].map(var_names)
    if phi_a==True:
        params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
    sns.swarmplot(data=params, x='parameter', y='Mean',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                color='k')
    sns.barplot(data=params, x='parameter', y='Mean', color='gray',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                palette=np.array(pal)[np.isin(var_order, params['parameter'].unique())])
    plt.ylabel('Coefficient')
    plt.xticks(rotation=90)
    sns.despine()
    if save==True:
        plt.savefig(filepath+'/params.pdf')

def compare_two_models(saved_params, stan_data, save=False, output_path=None, phi_a=True):
        var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                    'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                    'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                    'alphalaserforgetting_ses':'αLaserForget',
                    'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                    'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                    'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                    'balphastay_ses':'αBStay'}
        var_order = ['βWater','βLaser','βStay', 'βWaterLoss', 'βLaserLoss','βLaserStay',
                    'αWater','αLaser','αStay','αWaterForget','αLaserForget','αStayLaser',
                    'αBWater','αBLaser','αBStay','ε','Bias']
        pal = ['dodgerblue','orange','gray', 'dodgerblue', 'orange','gray',
                    'dodgerblue','orange','gray','dodgerblue','orange','gray',
                    'dodgerblue','orange','gray','violet','salmon']
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        saved_params_new = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
        saved_params = saved_params_new
        params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
            (saved_params['name'].str.contains('s\[')),['name','Mean','model_name']]
        params['parameter']=params['name'].str.rstrip('[123456789102030]')
        params['parameter'] = params['parameter'].map(var_names)
        if phi_a==True:
            params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                    phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
        sns.catplot(data=params, x='model_name', col='parameter' ,y='Mean', hue='name',kind='point',legend=False, color='k')
        plt.legend().remove()
        plt.ylabel('Coefficients')
        sns.despine()
        plt.tight_layout()
def plot_params_diff_learning(saved_params, stan_data, save=False, output_path=None, phi_a=True):
    var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                'alphalaserforgetting_ses':'αLaserForget',
                'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                'balphastay_ses':'αBStay'}
    var_order = ['βWater','βLaser','βStay', 'βWaterLoss', 'βLaserLoss','βLaserStay',
                'αWater','αLaser','αStay','αWaterForget','αLaserForget','αStayLaser',
                'αBWater','αBLaser','αBStay','ε','Bias']
    pal = ['dodgerblue','orange','gray', 'dodgerblue', 'orange','gray',
                'dodgerblue','orange','gray','dodgerblue','orange','gray',
                'dodgerblue','orange','gray','violet','salmon']
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    saved_params_new = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
    saved_params = saved_params_new
    params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
        (saved_params['name'].str.contains('s\[')),['name','Mean']]
    params['parameter']=params['name'].str.rstrip('[123456789102030]')
    params['parameter'] = params['parameter'].map(var_names)
    params = params.loc[np.isin(params['parameter'], ['αWater','αLaser','αStay','αBWater','αBLaser','αBStay'])]
    params['type'] = 'End'
    params.loc[np.isin(params['parameter'], ['αBWater','αBLaser','αBStay']), 'type'] = 'Beggining'
    params.loc[params['parameter']=='αBWater','parameter'] = 'αWater'
    params.loc[params['parameter']=='αBLaser','parameter'] ='αLaser'
    params.loc[params['parameter']=='αBStay','parameter'] = 'αStay'

    if phi_a==True:
        params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
    sns.barplot(data=params, x='parameter', y='Mean', hue='type',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())],
                palette='inferno')
    plt.ylabel('Coefficient')
    sns.despine()
    plt.ylim(0.3,0.6)
    if save==True:
        plt.savefig(filepath+'/params.pdf')
def plot_model_comparison(models_summary, save=False, output_path=None):
    sns.pointplot(data=models_summary, x='Model', y='Accuracy', color='r', hue=models_summary.index,
                scale=0.7, order=models_summary.groupby(['Model']).mean().sort_values('Accuracy').index)
    sns.barplot(data=models_summary, x='Model', y='Accuracy', color='gray',
                order=models_summary.groupby(['Model']).mean().sort_values('Accuracy').index)
    plt.xticks(rotation=45)
    plt.ylim(0.47,0.85)
    plt.legend().remove()
    sns.despine()
    plt.tight_layout()
    if save==True:
        plt.savefig(filepath+'/model_comparison.pdf')
def compare_accuracy_reinforce(reinforce_summary):
    sns.pointplot(data=reinforce_summary, x='Model', y='Accuracy', color='r', hue=reinforce_summary.index,
                scale=0.7, order=reinforce_summary.groupby(['Model']).mean().sort_values('Accuracy').index,
                palette='gray')
    plt.legend().remove()
    sns.barplot(data=reinforce_summary, x='Model', y='Accuracy', color='gray',
                order=reinforce_summary.groupby(['Model']).mean().sort_values('Accuracy').index)
    plt.ylim(0.5,0.85)
    sns.despine()
def check_model_behavior(sim_data):
    sim_data_with_rep = sim_data.reset_index().copy()
    sim_data_plotting = pd.DataFrame()
    for mouse in sim_data_with_rep['mouse'].unique():
        select_1 = sim_data_with_rep.loc[sim_data_with_rep['mouse']==mouse]
        for ses in select_1['ses'].unique():
            select2 = select_1.loc[select_1['ses']==ses]
            for i in np.where(select2['probabilityLeft'].diff()!=0)[0]:
                reps = select2.iloc[i-5:i].copy()
                reps['tb'] = np.arange(-5,0)
                if reps['probabilityLeft'].iloc[0]==0.1:
                    reps['probabilityLeft'] = 0.7
                else:
                    reps['probabilityLeft'] = 0.1
                sim_data_plotting = pd.concat([sim_data_plotting, reps])
    sim_data_plotting = pd.concat([sim_data_plotting, sim_data_with_rep])
    sns.lineplot(data=sim_data_plotting.reset_index(), x='tb', y= 'choices',
                  ci=68, hue='laser_block', err_style='bars', 
                  style='probabilityLeft', palette = ['dodgerblue','orange'])
    plt.legend().remove()
    plt.xlim(-5,15)
    plt.xlabel('Trials back')
    plt.vlines(0,0,1,linestyles='dashed', color='k')
    plt.ylabel('Fraction of Right Choices')
    sns.despine()
# Models
def reduced_uchida_model(standata,saved_params):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qloss = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLloss=[]
            QRloss=[]           
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * ((q[1] - qloss[1])-(q[0] - qloss[0]))
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                qloss[choice] = (1-alphaloss) * qloss[choice] + alphaloss * (1 - r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLloss.append(qloss[0])
                QRloss.append(qloss[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*(q[0]-qloss[0]))+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*(q[1]-qloss[1]))+(stay_mouse*qstay[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse 
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLloss'] = np.array(QLloss) * beta_mouse 
                ses_data['QRloss'] =  np.array(QRloss) * beta_mouse
                ses_data['QLstay'] =  np.array(QLstay) * stay_mouse
                ses_data['QRstay'] =  np.array(QRstay) * stay_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data

def double_update_model(standata,saved_params):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                q[unchoice] = q[unchoice] - alphaloss * (r[sub_idx[ms_i], sess_idx[ms_i],t]-q[choice]) 
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t])
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
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
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])

def simulate_double_update_model(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                unchoice = 1*(choice==0)
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[unchoice] = q[unchoice] - alphaloss * (f-q[choice]) 
                q[choice] = (1-alpha) * q[choice] + alpha * f
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
            
                # Store variables
                rewards.append(f)
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, r_sim, sim_data



def simulate_reduced_uchida_model(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qloss = np.zeros(2)
            predicted_choices=[]
            rewards = []
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * ((q[1] - qloss[1])-(q[0] - qloss[0]))
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * f
                qloss[choice] = (1-alphaloss) * qloss[choice] + alphaloss * (1 - f) # Join r+l for reduced model               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
            
                # Store variables
                rewards.append(f)
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, r_sim, sim_data

def q_learning_model_reduced(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    l =  standata['l']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]+
                                                             l[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLgeneral.append((beta_mouse*q[0]))
                QRgeneral.append((beta_mouse*q[1]))
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
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data

def pearce_hall_model(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    eta = float(saved_params.loc[saved_params['name']=='eta_ses['+str(ms_i+1)+']', 'Mean'])
                    kappa_laser_ses = float(saved_params.loc[saved_params['name']=='kappa_laser_ses['+str(ms_i+1)+']', 'Mean'])
                    kappa_ses = float(saved_params.loc[saved_params['name']=='kappa_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
            alpha_laser=1
            alpha_water=1
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]

                delta_water =  r[sub_idx[ms_i], sess_idx[ms_i],t] - q[choice]
                delta_laser =  l[sub_idx[ms_i], sess_idx[ms_i],t] - qlaser[choice]
                q[choice] = q[choice] + kappa_ses*alpha_water*delta_water
                qlaser[choice] = qlaser[choice] + kappa_laser_ses*alpha_laser*delta_laser
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                alpha_water = (eta * abs(delta_water)) + ((1-eta)*alpha_water)
                alpha_laser = (eta * abs(delta_laser)) + ((1-eta)*alpha_laser)

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
            print(np.mean(acc))

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
    return make_deltas(data)
def q_learning_model_asymmetric(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserloss = float(saved_params.loc[saved_params['name']=='alphalaserloss_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
                rt = r[sub_idx[ms_i], sess_idx[ms_i],t]
                lt = l[sub_idx[ms_i], sess_idx[ms_i],t]
                if rt==1:
                    q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if rt==0:
                    q[choice] = (1-alphaloss) * q[choice] + alphaloss * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if lt==1:
                    qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if lt==0:
                    qlaser[choice] = (1-alphalaserloss) * qlaser[choice] + alphalaserloss * l[sub_idx[ms_i], sess_idx[ms_i],t]
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
            print(np.mean(acc))

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
    return make_deltas(data)
def q_learning_model_asymmetric_nostay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaloss = float(saved_params.loc[saved_params['name']=='alphaloss_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserloss = float(saved_params.loc[saved_params['name']=='alphalaserloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))    
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]    
                rt = r[sub_idx[ms_i], sess_idx[ms_i],t]
                lt = l[sub_idx[ms_i], sess_idx[ms_i],t]
                if rt==1:
                    q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if rt==0:
                    q[choice] = (1-alphaloss) * q[choice] + alphaloss * r[sub_idx[ms_i], sess_idx[ms_i],t]
                if lt==1:
                    qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if lt==0:
                    qlaser[choice] = (1-alphalaserloss) * qlaser[choice] + alphalaserloss * l[sub_idx[ms_i], sess_idx[ms_i],t]
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1]))
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['QL'] = QLgeneral
                ses_data['QR'] = QRgeneral
                ses_data['QLreward'] = np.array(QL) * beta_mouse # Note this QL is the reward  - TODO change name
                ses_data['QRreward'] =  np.array(QR) * beta_mouse
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)

def q_learning_model_reduced_stay_forgetting(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            forget = float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                unchoice = 1*(choice==0)
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                q[unchoice] =  q[unchoice] - forget* q[unchoice]
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
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
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data


def q_learning_model_reduced_stay(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])
                QLgeneral.append((beta_mouse*q[0])+(stay_mouse*qstay[0]))
                QRgeneral.append((beta_mouse*q[1])+(stay_mouse*qstay[1]))
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
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data
def q_learning_model_reduced(standata,saved_params=None):
    r =  standata['r']
    c =  standata['c']
    l =  standata['l']
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
            side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
            alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
            q = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * (r[sub_idx[ms_i], sess_idx[ms_i],t]+
                                                             l[sub_idx[ms_i], sess_idx[ms_i],t]) # Join r+l for reduced model
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLgeneral.append((beta_mouse*q[0]))
                QRgeneral.append((beta_mouse*q[1]))
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
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def q_learning_lasernostay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            QL=[]
            QR=[]
            QLlaser=[]
            QRlaser=[]
            choices=[]
            QLgeneral=[]
            QRgeneral=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                # Store variables
                predicted_choices.append(predicted_choice)
                choices.append(choice)
                QL.append(q[0])
                QR.append(q[1])
                QLlaser.append(qlaser[0])
                QRlaser.append(qlaser[1])
                QLgeneral.append((beta_mouse*q[0])+(laser_mouse*qlaser[0]))
                QRgeneral.append((beta_mouse*q[1])+(laser_mouse*qlaser[1]))
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
                ses_data['QLlaser'] =  np.array(QLlaser) * laser_mouse
                ses_data['QRlaser'] =  np.array(QRlaser) * laser_mouse
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def q_learning_model(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
    return make_deltas(data)
def simulate_q_learning_model_alphalaserdecay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
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
                    alphalaser_init = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            alphalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]

            alphalaser = alphalaser_init

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse  * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-(alphalaser+alphalaserbase)) * qlaser[choice] + ((alphalaser+alphalaserbase) * f * l)
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    alphalaser = (1-laserdecay)*alphalaser
                # Store variables
                alphalaser_buffer.append(alphalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['a_init_laser'] = alphalaser_init
                ses_data['alpha_laser'] = alphalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data

def simulate_q_learning_model_alphalaserdecay_everytrial(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
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
                    alphalaser_init = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            alphalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]

            alphalaser = alphalaser_init

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse  * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-(alphalaser+alphalaserbase)) * qlaser[choice] + (alphalaser+alphalaserbase) * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                alphalaser = (1-laserdecay)*alphalaser
                # Store variables
                alphalaser_buffer.append(alphalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['a_init_laser'] = alphalaser_init
                ses_data['alpha_laser'] = alphalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def q_learning_model_alphalaserdecay_everytrial(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
            alphalaser_buffer=[]
            alphalaser = alphalaser
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-(alphalaser+alphalaserbase)) * qlaser[choice] + (alphalaser+alphalaserbase) * l[sub_idx[ms_i], sess_idx[ms_i],t]
                alphalaser = (1-laserdecay)*alphalaser
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
                alphalaser_buffer.append(alphalaser+alphalaserbase)

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
                ses_data['alphalaser'] = np.array(alphalaser_buffer)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_alphalaserdecay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    alphalaser_init = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
            alphalaser_buffer=[]
            alphalaser = alphalaser_init
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-(alphalaser+alphalaserbase)) * qlaser[choice] + (alphalaser+alphalaserbase) * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    alphalaser = (1-laserdecay)*alphalaser
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
                alphalaser_buffer.append(alphalaser+alphalaserbase)

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
                ses_data['alphalaser'] = np.array(alphalaser_buffer)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_laserdecay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
            betalaser_buffer=[]
            betalaser = laser_mouse
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + (betalaser+betalaserbase) * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
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
                betalaser_buffer.append(betalaser+betalaserbase)
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
                ses_data['betalaser'] = np.array(betalaser_buffer)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def simulate_q_learning_model_new(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()

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

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data

def simulate_q_learning_model_reduced_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*f)
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append(1*f)
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, r_sim, sim_data

def simulate_q_learning_model_noqlaser_new(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()

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

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)

                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_noqwater_new(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()

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

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)

                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_noqwaterorlaser_new(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()

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

            tph = TrialParamHandler()
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                
                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)

                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_laserdecay_winloss_stay_same_trials(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay = float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    print('Error: Only csv version available at the moment!')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            print(betalaser)
            print(betalaserloss)
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + working_choice * (betalaser+betalaserbase)
                    betalaser = (1-laserdecay)*betalaser
                if (f*l) == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * (betalaserloss+betalaserlossbase)               
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tb[sub_idx[ms_i], sess_idx[ms_i],t])
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_laserdecay_winloss_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay = float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + working_choice * (betalaser+betalaserbase)
                if (f*l) == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * (betalaserloss+betalaserlossbase)               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data


def simulate_reinforce_reduced_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + q) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if f == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if f == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                rewards.append(f)
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, r_sim, sim_data

def simulate_reinforce_reduced(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            predicted_choices=[]
            rewards = []
            outcome=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = inv_logit(side_ses + q)
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if f == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if f == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses

                # Store variables
                rewards.append(f)
                outcome.append(f)
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, r_sim, sim_data





def simulate_reinforce_alphalaserdecay_win_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay = float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - (betalaser+betalaserbase)) * qlaser + working_choice * alphalaser
                if (f*l) == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * (betalaserloss)               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_laserdecay_winloss_stay_nolaser(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + q) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses           
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_winloss_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + working_choice * laser_ses
                if (f*l) == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * dmlaser_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_reinforce_laserdecay_winloss_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay = float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            betalaser = laser_ses
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)    
                p_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                working_choice = 2 * choice -1
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                if (f*(l==0)) == 1:
                    q = (1 - alpha) * q + working_choice * beta_ses
                if (f*(l==0)) == 0:
                    q = (1 - alphaforgetting) * q + working_choice * dmwater_ses
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + working_choice * (betalaser+betalaserbase)
                if (f*l) == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + working_choice * (betalaserloss+betalaserlossbase)               
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay

                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser)
                buffer_qwater.append(q)
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_ses
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
                
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_laserdecay_nolaser(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
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
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_mouse

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_mouse
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_alphalaserdecay_same_trials(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
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
                    alphalaser_init = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserbase= float(saved_params.loc[saved_params['name']=='alphalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            alphalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            alphalaser = alphalaser_init

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + laser_mouse * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7                
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-(alphalaser+alphalaserbase) ) * qlaser[choice] + (alphalaser+alphalaserbase)  * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    alphalaser = (1-laserdecay)*alphalaser
                # Store variables
                alphalaser_buffer.append(alphalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tb[sub_idx[ms_i], sess_idx[ms_i],t])
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['a_init_laser'] = alphalaser_init
                ses_data['beta_laser'] = alphalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_laserdecay_same_trials(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
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
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_mouse

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + (betalaser+betalaserbase)  * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7                
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                if choice==0:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tb[sub_idx[ms_i], sess_idx[ms_i],t])
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_mouse
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def simulate_q_learning_model_laserdecay(standata_recovery,saved_params=None, fit=None, csv=True):
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
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
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Error: Only csv version available at the moment!')

            tph = TrialParamHandler()
            q = np.zeros(2)
            qlaser = np.zeros(2)
            qstay = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]
            opto_blocks=[]
            side_blocks=[]
            trial_within_block=[]
            right_reward=[]
            left_reward=[]
            betalaser_buffer=[]
            buffer_qlaser=[]
            buffer_qwater=[]
            betalaser = laser_mouse

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + (betalaser+betalaserbase)  * (qlaser[1] - qlaser[0])
                    + stay_mouse * (qstay[1] - qstay[0]))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                tph.response_side_buffer.append(choice)  # Add choice to response buffer
                p_left= tph.stim_probability_left
                l = tph.opto_block
                if choice==1:
                    f = tph.right_reward
                if choice==0:
                    f = tph.left_reward
                q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                if (f*l) == 1:
                    betalaser = (1-laserdecay)*betalaser
                # Store variables
                betalaser_buffer.append(betalaser)
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                opto_blocks.append(l) 
                side_blocks.append(p_left)
                predicted_choices.append(choice)
                trial_within_block.append(tph.block_trial_num)
                right_reward.append(tph.right_reward)
                left_reward.append(tph.left_reward)
                buffer_qlaser.append(qlaser[choice])
                buffer_qwater.append(q[choice])
                # Init next trial
                tph =  tph.next_trial()

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['b_init_laser'] = laser_mouse
                ses_data['beta_laser'] = betalaser_buffer
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = opto_blocks
                ses_data['tb'] = trial_within_block
                ses_data['probabilityLeft'] = side_blocks
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['right_reward'] = right_reward
                ses_data['left_reward'] = left_reward
                ses_data['qwater'] = buffer_qwater
                ses_data['qlaser'] = buffer_qlaser
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
def q_learning_model_w_forgetting(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2))
                else:
                    print('Error only CSV version available')

            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
                q[choice^1] = q[choice^1] - (q[choice^1] * alphaforgetting)
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice^1] = qlaser[choice^1] - (alphalaserforgetting * qlaser[choice^1])
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
            print(np.mean(acc))

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
    return make_deltas(data)


def simulate_q_learning_w_forgetting(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb =  standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
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
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphaforgetting= float(phi_approx(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaserforgetting= float(phi_approx(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    print('Error only CSV version available')

            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
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
                q[choice^1] = q[choice^1] - (q[choice^1] * alphaforgetting)
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                qlaser[choice^1] = qlaser[choice^1] - (alphalaserforgetting * qlaser[choice^1])
                qstay = qstay * (1 - alphastay)
                qstay[choice] = qstay[choice] + alphastay
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables
                predicted_choices.append(choice)

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])
    return c_sim, l_sim, r_sim, sim_data
    
def reinforce_model_mixed_perseveration(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_mixed_perseveration_noloss(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_nolaser_mixed_perseveration(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    o=r+l
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + q) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if o[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if o[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_reduced(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Non csv version not avaialble atm')
            q = 0
            predicted_choices=[]
            Q=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + q)
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                o = r[sub_idx[ms_i], sess_idx[ms_i], t]
                if o == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if o == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['deltaQ'] = ses_data['Qwater']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data

def reinforce_model_reduced_stay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
            #    if ms_i==0:
            #        print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    print('Non csv version not avaialble atm')
            q = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + q) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                ses_data['id'] = ses_data['mouse']*100 + ses_data['ses']
                data = pd.concat([data,ses_data])
    return data

def reinforce_model_laserdecay_linear(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betalaser = laser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * betalaser
                    betalaser = betalaser - laserdecay
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            print(laser_ses-betalaser)
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_alldecay_exp(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    waterdecay= float(saved_params.loc[saved_params['name']=='waterdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else: 
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betalaser = laser_ses
            betawater = beta_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betawater = (1-waterdecay)*betawater
                    q = (1 - alpha) * q + choice * betawater
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * betalaser
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            print(laser_ses-betalaser)

            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_laserdecay_mixed_perseveration(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            betalaser_buffer=[]
            betalaserloss_buffer=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * (betalaser+betalaserbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * (betalaserloss+betalaserlossbase)               
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                betalaser_buffer.append(betalaser+betalaserbase)
                betalaserloss_buffer.append(betalaserloss+betalaserlossbase)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['betalaser'] = betalaser_buffer
                ses_data['betalaserloss'] = betalaserloss_buffer
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_alphalaserdecay_mixed_perseveration(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            betalaser_buffer=[]
            betalaserloss_buffer=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - (betalaser+betalaserbase)) * qlaser + choice * alphalaser
                    betalaser = (1-laserdecay)*betalaser
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * betalaserloss               
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                betalaser_buffer.append(betalaser+betalaserbase)
                betalaserloss_buffer.append(betalaserloss)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['betalaser'] = betalaser_buffer
                ses_data['betalaserloss'] = betalaserloss_buffer
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_laserdecay_mixed_perseveration_noloss(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    epses = float(saved_params.loc[saved_params['name']=='ep_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = np.zeros(2)
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            QLstay=[]
            QRstay=[]
            choices=[]
            betalaser = laser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = (1-epses) * inv_logit(side_ses + (q + qlaser)) + epses*(inv_logit(qstay[1]-qstay[0]))
                choice_10 = c[sub_idx[ms_i], sess_idx[ms_i], t]
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * (betalaser+betalaserbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser
                qstay = qstay * (1 - alphastay)
                qstay[choice_10] = qstay[choice_10] + alphastay
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                QLstay.append(qstay[0])
                QRstay.append(qstay[1])

            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['QLstay'] =  np.array(QLstay)
                ses_data['QRstay'] =  np.array(QRstay)
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_laserdecaywinloss_exp(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * (betalaser+betalaserbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * (betalaserloss+betalaserlossbase)
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            #print(laser_ses-betalaser)

            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_waterlaserdecaywinloss_exp(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])
                    waterdecay= float(saved_params.loc[saved_params['name']=='waterdecay_ses['+str(ms_i+1)+']', 'Mean'])
                    betawaterlossbase= float(saved_params.loc[saved_params['name']=='betawaterlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                    betawaterbase= float(saved_params.loc[saved_params['name']=='betawaterbase_ses['+str(ms_i+1)+']', 'Mean'])
                    waterdecayloss= float(saved_params.loc[saved_params['name']=='waterdecayloss_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betawater = beta_ses
            betalaser = laser_ses
            betalaserloss = dmlaser_ses
            betawaterloss = dmwater_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betawater = (1-waterdecay)*betawater
                    q = (1 - alpha) * q + choice * (betawater+betawaterbase)
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    betawaterloss = (1-waterdecayloss)*betawaterloss
                    q = (1 - alphaforgetting) * q + choice * (betawaterloss+betawaterlossbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    betalaser = (1-laserdecay)*betalaser
                    qlaser = (1 - alphalaser) * qlaser + choice * (betalaser+betalaserbase)
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    betalaserloss = (1-laserdecayloss)*betalaserloss
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * (betalaserloss+betalaserlossbase)
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            #print(laser_ses-betalaser)

            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def reinforce_model_laserdecay_exp(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            choices=[]
            betalaser = laser_ses
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * betalaser
                    betalaser = (1-laserdecay)*betalaser
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)

            acc = []
            print(laser_ses-betalaser)

            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['Qlaser'] = Qlaser
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def simulate_reinforce_model(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                    dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')

            q = 0
            qlaser = 0
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_ses + (q + qlaser))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                predicted_choices.append(choice)
                choice = 2 * choice - 1
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                else:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])

                if (1*(f*(l==0))) == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if (1*(f*(l==0))) == 0:
                    q = (1 - alphaforgetting) * q + choice * dmwater_ses
                if (f * l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if (f * l)== 0:
                    qlaser = (1 - alphalaserforgetting) * qlaser + choice * dmlaser_ses

                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data

def reinforce_model_w_stay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    stay_ses = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay= float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            Qstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser + qstay))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                qstay = (1 - alphastay) * qstay + choice * stay_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                Qstay.append(qstay)
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            #print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['bias'] = side_ses
                ses_data['Qlaser'] = Qlaser
                ses_data['Qstay'] = Qstay
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser'] + ses_data['Qstay']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def simulate_reinforce_model_w_stay(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    stay_ses = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay= float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = 0
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_ses + (q + qlaser + qstay))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                predicted_choices.append(choice)
                choice = 2 * choice - 1
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                else:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])

                if (1*(f*(l==0))) == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if (f * l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                qstay = (1 - alphastay) * qstay + choice * stay_ses

                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data
def reinforce_model_w_2stay(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        b =  standata['b']
        tb = standata['tb']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    stay_ses = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay= float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserstay= float(saved_params.loc[saved_params['name']=='alphalaserstay_ses['+str(ms_i+1)+']', 'Mean'])
                    laserstay_ses = float(saved_params.loc[saved_params['name']=='laserstay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = 0
            predicted_choices=[]
            Q=[]
            Qlaser=[]
            Qstay=[]
            choices=[]
            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                choices.append(c[sub_idx[ms_i], sess_idx[ms_i], t])
                predicted_choice = inv_logit(side_ses + (q + qlaser + qstay))
                choice = 2 * c[sub_idx[ms_i], sess_idx[ms_i], t] -1
                if r[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if l[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if b[sub_idx[ms_i], sess_idx[ms_i], t] == 0:
                    qstay = (1 - alphastay) * qstay + choice * stay_ses
                if b[sub_idx[ms_i], sess_idx[ms_i], t] == 1:
                    qstay = (1 - alphalaserstay) * qstay + choice * laserstay_ses
                # Store variables
                predicted_choices.append(predicted_choice)
                Q.append(q)
                Qlaser.append(qlaser)
                Qstay.append(qstay)
            acc = []
            for i in np.arange(len(choices)):
                acc.append(1*(choices[i]==(1*(predicted_choices[i]>0.5))))
            print(np.mean(acc))

            if NT_all[ms_i]!=0:
                ses_data = pd.DataFrame()
                ses_data['predicted_choice'] = predicted_choices
                ses_data['Qwater'] = Q
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['trial_block'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['Qlaser'] = Qlaser
                ses_data['Qstay'] = Qstay
                ses_data['deltaQ'] = ses_data['Qwater'] + ses_data['Qlaser'] + ses_data['Qstay']
                ses_data['choices'] = choices
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return data
def simulate_reinforce_model_w_2stay(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    stay_ses = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    laserstay_ses = float(saved_params.loc[saved_params['name']=='laserstay_ses['+str(ms_i+1)+']', 'Mean'])
                    beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    alphastay= float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
                    alphalaserstay= float(saved_params.loc[saved_params['name']=='alphalaserstay_ses['+str(ms_i+1)+']', 'Mean'])

                else:
                    Print('Non csv version not avaialble atm')
            q = 0
            qlaser = 0
            qstay = 0
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
                t = int(t)
                p_choice = inv_logit(side_ses + (q + qlaser + qstay))
                choice =  np.random.choice(2,p= [1-p_choice,p_choice])
                predicted_choices.append(choice)
                choice = 2 * choice - 1
                assert np.isin(p[sub_idx[ms_i], sess_idx[ms_i],t], [0.7,0.1])
                p_left= p[sub_idx[ms_i], sess_idx[ms_i],t]
                l = b[sub_idx[ms_i], sess_idx[ms_i],t]
                if p_left==0.7:
                    p_right=0.1
                else:
                    p_right=0.7
                if choice==1:
                    f = np.random.choice([0,1], p=[1-p_right,p_right])
                else:
                    f = np.random.choice([0,1], p=[1-p_left,p_left])

                if (1*(f*(l==0))) == 1:
                    q = (1 - alpha) * q + choice * beta_ses
                if (f * l) == 1:
                    qlaser = (1 - alphalaser) * qlaser + choice * laser_ses
                if l == 1:
                    qstay = (1 - alphalaserstay) * qstay + choice * laserstay_ses
                if l == 0:
                    qstay = (1 - alphastay) * qstay + choice * stay_ses

                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data
def q_learning_model_w_2learning(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        tb =  standata['tb']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    balpha = float(saved_params.loc[saved_params['name']=='balpha_ses['+str(ms_i+1)+']', 'Mean'])
                    balphalaser = float(saved_params.loc[saved_params['name']=='balphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    balphastay = float(saved_params.loc[saved_params['name']=='balphastay_ses['+str(ms_i+1)+']', 'Mean'])
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
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
                if tb[sub_idx[ms_i], sess_idx[ms_i],t]>10:
                    q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                    qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                    qstay = qstay * (1 - alphastay)
                    qstay[choice] = qstay[choice] + alphastay
                if tb[sub_idx[ms_i], sess_idx[ms_i],t]<11:
                    q[choice] = (1-balpha) * q[choice] + balpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                    qlaser[choice] = (1-balphalaser) * qlaser[choice] + balphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                    qstay = qstay * (1 - balphastay)
                    qstay[choice] = qstay[choice] + balphastay
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
            print(np.mean(acc))
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
    return make_deltas(data)
def simulate_q_learning_w_2learning(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb =  standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()
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
                    balpha = float(saved_params.loc[saved_params['name']=='balpha_ses['+str(ms_i+1)+']', 'Mean'])
                    balphalaser = float(saved_params.loc[saved_params['name']=='balphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                    balphastay = float(saved_params.loc[saved_params['name']=='balphastay_ses['+str(ms_i+1)+']', 'Mean'])
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
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
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
                if tb[sub_idx[ms_i], sess_idx[ms_i],t]>10:
                    q[choice] = (1-alpha) * q[choice] + alpha * (1*(f*(l==0)))
                    qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * f * l
                    qstay = qstay * (1 - alphastay)
                    qstay[choice] = qstay[choice] + alphastay
                if tb[sub_idx[ms_i], sess_idx[ms_i],t]<11:
                    q[choice] = (1-balpha) * q[choice] + balpha * (1*(f*(l==0)))
                    qlaser[choice] = (1-balphalaser) * qlaser[choice] + balphalaser * f * l
                    qstay = qstay * (1 - balphastay)
                    qstay[choice] = qstay[choice] + balphastay
                rewards.append((1*(f*(l==0))))
                lasers.append(f*l)
                outcome.append(f)
                # Store variables
                predicted_choices.append(choice)

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data
def q_learning_model_Lrpe(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
    data = pd.DataFrame()
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
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    beta_mouse =  saved_params['beta_ses['+str(ms_i+1)+']'].mean()
                    stay_mouse = saved_params['stay_ses['+str(ms_i+1)+']'].mean()
                    side_mouse = saved_params['sides['+str(ms_i+1)+']'].mean()
                    laser_mouse = saved_params['laser_ses['+str(ms_i+1)+']'].mean()
                    alpha = phi_approx(saved_params['alpha_ses['+str(ms_i+1)+']'].mean())
                    alphalaser =phi_approx(saved_params['alphalaser_ses['+str(ms_i+1)+']'].mean())
                    alphastay = phi_approx(saved_params['alphastay_ses['+str(ms_i+1)+']'].mean())
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
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
                qlaser[choice] = qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
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
            print(np.mean(acc))

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
    return make_deltas(data)
def q_learning_model_Lrpeconditional(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
    data = pd.DataFrame()
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
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    beta_mouse =  saved_params['beta_ses['+str(ms_i+1)+']'].mean()
                    stay_mouse = saved_params['stay_ses['+str(ms_i+1)+']'].mean()
                    side_mouse = saved_params['sides['+str(ms_i+1)+']'].mean()
                    laser_mouse = saved_params['laser_ses['+str(ms_i+1)+']'].mean()
                    alpha = phi_approx(saved_params['alpha_ses['+str(ms_i+1)+']'].mean())
                    alphalaser =phi_approx(saved_params['alphalaser_ses['+str(ms_i+1)+']'].mean())
                    alphastay = phi_approx(saved_params['alphastay_ses['+str(ms_i+1)+']'].mean())
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
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
                if l[sub_idx[ms_i], sess_idx[ms_i],t]==1:
                    qlaser[choice] = qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                else:
                    qlaser[choice] = qlaser[choice] + alphalaser * (l[sub_idx[ms_i], sess_idx[ms_i],t] - qlaser[choice])
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
            print(np.mean(acc))

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
    return make_deltas(data)
def q_learning_model_2alphas(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        b =  standata['b']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
    else:
        r =  fit.data['r']
        c =  fit.data['c']
        l =  fit.data['l']
        b =  fit.data['b']
        sub_idx =  fit.data['sub_idx']-1
        sess_idx = fit.data['sess_idx']-1
        NSxNSESS = fit.data['NSxNSESS']
    data = pd.DataFrame()

    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            if saved_params is not None:
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    laserstay_mouse = float(saved_params.loc[saved_params['name']=='laserstay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaserstay = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaserstay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    beta_mouse =  saved_params['beta_ses['+str(ms_i+1)+']'].mean()
                    stay_mouse = saved_params['stay_ses['+str(ms_i+1)+']'].mean()
                    laserstay_mouse = saved_params['laserstay_ses['+str(ms_i+1)+']'].mean()
                    side_mouse = saved_params['sides['+str(ms_i+1)+']'].mean()
                    laser_mouse = saved_params['laser_ses['+str(ms_i+1)+']'].mean()
                    alpha = phi_approx(saved_params['alpha_ses['+str(ms_i+1)+']'].mean())
                    alphalaser =phi_approx(saved_params['alphalaser_ses['+str(ms_i+1)+']'].mean())
                    alphastay = phi_approx(saved_params['alphastay_ses['+str(ms_i+1)+']'].mean())
                    alphalaserstay = phi_approx(saved_params['alphalaserstay_ses['+str(ms_i+1)+']'].mean())
            else:
                if ms_i==0:
                    print('No saved parameters')
                beta_mouse = fit.extract()['beta_ses'][:,int(ms_i)].mean()
                stay_mouse = fit.extract()['stay_ses'][:,int(ms_i)].mean()
                laserstay_mouse = fit.extract()['laserstay_ses'][:,int(ms_i)].mean()
                side_mouse = fit.extract()['sides'][:,int(ms_i)].mean()
                laser_mouse = fit.extract()['laser_ses'][:,int(ms_i)].mean()
                alpha = phi_approx(fit.extract()['alpha_ses'][:,int(ms_i)].mean())
                alphalaser =phi_approx(fit.extract()['alphalaser_ses'][:,int(ms_i)].mean())
                alphastay = phi_approx(fit.extract()['alphastay_ses'][:,int(ms_i)].mean())
                alphalaserstay = phi_approx(fit.extract()['alphalaserstay_ses'][:,int(ms_i)].mean())
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
                if b[sub_idx[ms_i], sess_idx[ms_i], t]==1:
                    predicted_choice = inv_logit(side_mouse
                        + beta_mouse  * (q[1] - q[0])
                        + laserstay_mouse * (qstay[1] - qstay[0])
                        + laser_mouse * (qlaser[1] - qlaser[0]))
                else:
                    predicted_choice = inv_logit(side_mouse
                        + beta_mouse  * (q[1] - q[0])
                        + laserstay_mouse * (qstay[1] - qstay[0])
                        + laser_mouse * (qlaser[1] - qlaser[0]))
                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if b[sub_idx[ms_i], sess_idx[ms_i], t]==1:
                    qstay = qstay * (1 - alphalaserstay)
                else:
                    qstay = qstay * (1 - alphastay)
                if b[sub_idx[ms_i], sess_idx[ms_i], t]==1:
                    qstay[choice] = qstay[choice] + alphalaserstay
                else:
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
            print(np.mean(acc))

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
                ses_data['choices'] = choices
                ses_data['deltaQ'] = ses_data['QR'] - ses_data['QL']
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                ses_data['acc'] = np.mean(acc)
                data = pd.concat([data,ses_data])
    return make_deltas(data)
def q_learning_model_Lrpeconditional_w_forgetting(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                if ms_i==0:
                    print('Using saved parameters')
                if csv==True:
                    beta_mouse = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                    stay_mouse = float(saved_params.loc[saved_params['name']=='stay_ses['+str(ms_i+1)+']', 'Mean'])
                    side_mouse = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                    laser_mouse = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                    alpha = float(phi_approx(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaser = float(phi_approx(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphastay = float(phi_approx(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphaforgetting= float(phi_approx(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                    alphalaserforgetting= float(phi_approx(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean']/np.sqrt(2)))
                else:
                    beta_mouse =  saved_params['beta_ses['+str(ms_i+1)+']'].mean()
                    stay_mouse = saved_params['stay_ses['+str(ms_i+1)+']'].mean()
                    side_mouse = saved_params['sides['+str(ms_i+1)+']'].mean()
                    laser_mouse = saved_params['laser_ses['+str(ms_i+1)+']'].mean()
                    alpha = phi_approx(saved_params['alpha_ses['+str(ms_i+1)+']'].mean())
                    alphalaser =phi_approx(saved_params['alphalaser_ses['+str(ms_i+1)+']'].mean())
                    alphastay = phi_approx(saved_params['alphastay_ses['+str(ms_i+1)+']'].mean())
                    alphaforgetting= phi_approx(saved_params['alphaforgetting_ses['+str(ms_i+1)+']'].mean())
                    alphalaserforgetting= phi_approx(saved_params['alphalaserforgetting_ses['+str(ms_i+1)+']'].mean())
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
                alphaforgetting = phi_approx(fit.extract()['alphaforgetting_ses'][:,int(ms_i)].mean())
                alphalaserforgetting = phi_approx(fit.extract()['alphalaserforgetting_ses'][:,int(ms_i)].mean())
            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
                q[choice^1] = q[choice^1] - (q[choice^1] * alphaforgetting)
                if l[sub_idx[ms_i], sess_idx[ms_i],t]==1:
                    qlaser[choice] = qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                    qlaser[choice^1] = qlaser[choice^1] - (alphalaserforgetting * qlaser[choice^1])
                else:
                    qlaser[choice] = qlaser[choice] + alphalaser * (l[sub_idx[ms_i], sess_idx[ms_i],t] - qlaser[choice])
                    qlaser[choice^1] = qlaser[choice^1] - (alphalaserforgetting * qlaser[choice^1])
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
            print(np.mean(acc))

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
    return make_deltas(data)
def q_learning_model_lapse(standata,saved_params=None, fit=None, csv=True):
    if saved_params is not None:
        r =  standata['r']
        c =  standata['c']
        l =  standata['l']
        sub_idx =  standata['sub_idx']-1
        sess_idx = standata['sess_idx']-1
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
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
                    lapse = float(saved_params.loc[saved_params['name']=='lapse_ses['+str(ms_i+1)+']', 'Mean'])
                else:
                    Print('Non csv version not avaialble atm')

            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
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
                predicted_choice = (1-lapse)* inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0])) + (lapse/2)

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
    return make_deltas(data)   
def simulate_q_learning_model(standata_recovery,saved_params=None, fit=None, csv=True):
    b =  standata_recovery['b']
    p =  standata_recovery['p']
    tb = standata_recovery['tb']
    NS =standata_recovery['NS']
    NSESS=standata_recovery['NSESS']
    NT =standata_recovery['NT']
    NT_all =standata_recovery['NT_all']
    r_sim = np.zeros([NS,NSESS,NT])
    c_sim = np.zeros([NS,NSESS,NT])
    l_sim = np.zeros([NS,NSESS,NT])
    sub_idx =  standata_recovery['sub_idx']-1
    sess_idx = standata_recovery['sess_idx']-1
    NSxNSESS = standata_recovery['NSxNSESS']
    sim_data = pd.DataFrame()

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

            q = np.zeros(2)
            qstay = np.zeros(2)
            qlaser = np.zeros(2)
            predicted_choices=[]
            rewards = []
            lasers = []
            outcome=[]

            for t in np.arange(NT_all[ms_i]):
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

            if NT_all[ms_i]!=0:
                c_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])]= predicted_choices
                l_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = lasers
                r_sim[sub_idx[ms_i], sess_idx[ms_i], :int(NT_all[ms_i])] = rewards
                ses_data = pd.DataFrame()
                ses_data['choices'] = predicted_choices
                ses_data['water'] = rewards
                ses_data['laser'] = lasers
                ses_data['laser_block'] = b[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['tb'] = tb[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['probabilityLeft'] = p[sub_idx[ms_i], sess_idx[ms_i],:int(NT_all[ms_i])]
                ses_data['outcome'] = outcome
                ses_data['mouse'] = sub_idx[ms_i]
                ses_data['ses'] = sess_idx[ms_i]
                sim_data = pd.concat([sim_data,ses_data])

    return c_sim, l_sim, r_sim, sim_data
# Correlation matrix
def correlation_matrix(standata, corr_var, model_standard,model_forgetting,reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    '''
    if corr_var != 'Qstay':
        model_labels = ['Standard','F-Q', 'Laser=RPE/Reward','Laser=RPE/Reward w/ F',
                        '2 QStay', 'Lapse','REINFORCE','REINFORCE_stay','REINFORCE_2stay']
    else:
        model_labels = ['Standard','F-Q', 'Laser=RPE/Reward','Laser=RPE/Reward w/ F',
                        '2 QStay', 'Lapse','REINFORCE_stay','REINFORCE_2stay']

    standard = q_learning_model(standata,saved_params=model_standard)[corr_var].to_numpy()
    fq = q_learning_model_w_forgetting(standata,saved_params=model_forgetting)[corr_var].to_numpy()
    if corr_var != 'Qstay':
        rnfrc = reinforce_model(standata,saved_params=reinforce)[corr_var].to_numpy()
    l = q_learning_model_Lrpe(standata,saved_params=LRPE)[corr_var].to_numpy()
    lr = q_learning_model_Lrpeconditional(standata,saved_params=LRPEc)[corr_var].to_numpy()
    twoq = q_learning_model_2alphas(standata,saved_params=a2stay)[corr_var].to_numpy()
    lrf = q_learning_model_Lrpeconditional_w_forgetting(standata,saved_params=LRPEcf)[corr_var].to_numpy()
    lapse = q_learning_model_lapse (standata, saved_params=stdlapse)[corr_var].to_numpy()
    rnfrc_stay = reinforce_model_w_stay(standata,saved_params=reinforce_w_stay)[corr_var].to_numpy()
    rnfrc_2stay = reinforce_model_w_2stay(standata,saved_params=reinforce_2stay)[corr_var].to_numpy()

    if corr_var != 'Qstay':
        corr_matrix = np.corrcoef([standard,fq,lr,lrf,twoq,lapse,rnfrc,rnfrc_stay,rnfrc_2stay])
    else:
        corr_matrix = np.corrcoef([standard,fq,lr,lrf,twoq,lapse,rnfrc_stay,rnfrc_2stay])
    return corr_matrix, model_labels
def correlation_matrix_reduced(standata, corr_var, model_standard,reinforce_w_stay):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    '''

    model_labels = ['Standard', 'REINFORCE_stay']

    standard = q_learning_model(standata,saved_params=model_standard)[corr_var].to_numpy()
    rnfrc_stay = reinforce_model_w_stay(standata,saved_params=reinforce_w_stay)[corr_var].to_numpy()

    corr_matrix = np.corrcoef([standard,rnfrc_stay])

    return corr_matrix, model_labels
def plot_corr_matrix(corr_matrix, model_labels, vmin=None, vmax=None):
    sns.heatmap(
    corr_matrix,
    cmap='viridis',
    square=True, vmin=vmin, vmax=vmax, annot=True)
    plt.xticks(np.arange(len(model_labels))+0.5, model_labels,rotation=45)
    plt.yticks(np.arange(len(model_labels))+0.5, model_labels, rotation=0)
def plot_corr_analysis(standata,model_standard,model_forgetting,reinforce,LRPE,LRPEc,a2stay,LRPEcf, stdlapse):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    Return: Plot correlation heatmaps for DeltaQ and predicted choice
    '''
    corr_delta_q, model_labels = correlation_matrix(standata, 'deltaQ', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay)
    corr_water, model_labels_water = correlation_matrix(standata, 'Qwater', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay)
    corr_stay, model_labels_stay = correlation_matrix(standata, 'Qstay', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay)
    corr_laser, model_labels_laser = correlation_matrix(standata, 'Qlaser', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse, reinforce_w_stay, reinforce_2stay)

    fig, ax = plt.subplots(2,2)
    plt.sca(ax[0,0])
    plot_corr_matrix(corr_delta_q, model_labels)
    plt.title('ΔQ')
    sns.despine()
    plt.tight_layout()
    plt.sca(ax[0,1])
    plot_corr_matrix(corr_water, model_labels_water)
    plt.title('ΔWater')
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1,0])
    plot_corr_matrix(corr_laser, model_labels_laser)
    plt.title('ΔLaser')
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1,1])
    plot_corr_matrix(corr_stay, model_labels_stay)
    plt.title('ΔStay')
    plt.tight_layout()
    sns.despine()
    plt.tight_layout()
def confusion_accu_matrix(standata_recovery):
    standard_recovery = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/output/summary.csv')
    reinforce_w_stay_recovery = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/outputsummary.csv')
    confusion_recover_REINFORCEstay_w_standard = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/confusion_recover_REINFORCEstay_w_standard/output/summary.csv')
    confusion_recover_standard_w_REINFORCEstay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/confusion_recover_standard_w_REINFORCEstay/output/summary.csv')
    standard_sim_r = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/r_sim.npy')
    standard_sim_l = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/l_sim.npy')
    standard_sim_c = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/c_sim.npy')
    reinforce_sim_r = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/r_sim.npy')
    reinforce_sim_l = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/l_sim.npy')
    reinforce_sim_c = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/c_sim.npy')

    standata_standard = replace_with_sim_data(standata_recovery,standard_sim_r,standard_sim_l,standard_sim_c)
    standata_reinforce_w_stay = replace_with_sim_data(standata_recovery,reinforce_sim_r,reinforce_sim_l,reinforce_sim_c)

    reinforce_og_accu = reinforce_model_w_stay(standata_reinforce_w_stay,saved_params=reinforce_w_stay_recovery)['acc'].unique().mean()
    reinforce_confussion_accu = reinforce_model_w_stay(standata_reinforce_w_stay,saved_params=confusion_recover_standard_w_REINFORCEstay)['acc'].unique().mean()
    standard_og_accu = q_learning_model(standata_standard,saved_params=standard_recovery)['acc'].unique().mean()
    standard_confussion_accu = q_learning_model(standata_standard,saved_params=confusion_recover_REINFORCEstay_w_standard)['acc'].unique().mean()

    accu_matrix = np.array([[standard_og_accu, standard_confussion_accu],[reinforce_confussion_accu,reinforce_og_accu]])
    model_labels = ['Q-Learning','REINFORCE w/ stay']

    plot_corr_matrix(accu_matrix, model_labels, vmin=0.5, vmax=0.9)
    plt.ylabel('Sim w/ Q-Learning')
    plt.xlabel('Sim w/ REINFORCE w/ stay')
    plt.tight_layout()
def replace_with_sim_data(standata_recovery,r,l,c):
    standata_recovery['r'] = r.astype(int)
    standata_recovery['l'] = l.astype(int)
    standata_recovery['c'] = c.astype(int)
    return standata_recovery
def params_reinforce_model_laserdecay_exp(standata, saved_params):
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
                dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                ses_params = pd.DataFrame()
                ses_params['Parameter'] = ['βWaterLoss','βLaserLoss','βWater','Bias', 
                                           'βLaser_init', 'αWater', 'αLaser','αWaterForget',
                                           'αLaserForget','αLaserDecay']
                ses_params['values'] = [dmwater_ses, dmlaser_ses, beta_ses, side_ses, laser_ses, alpha, alphalaser, alphaforgetting, alphalaserforgetting, laserdecay]
                ses_params['mouse'] = sub_idx[ms_i]
                ses_params['ses'] = sess_idx[ms_i]
                data = pd.concat([data,ses_params])
    return data
def params_reinforce_model_laserdecay_winloss_exp(standata, saved_params):
    sub_idx =  standata['sub_idx']-1
    sess_idx = standata['sess_idx']-1
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    data = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
                dmwater_ses = float(saved_params.loc[saved_params['name']=='dmwater_ses['+str(ms_i+1)+']', 'Mean'])
                dmlaser_ses = float(saved_params.loc[saved_params['name']=='dmlaser_ses['+str(ms_i+1)+']', 'Mean'])
                beta_ses = float(saved_params.loc[saved_params['name']=='beta_ses['+str(ms_i+1)+']', 'Mean'])
                side_ses = float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                laser_ses = float(saved_params.loc[saved_params['name']=='laser_ses['+str(ms_i+1)+']', 'Mean'])
                alpha = float(saved_params.loc[saved_params['name']=='alpha_ses['+str(ms_i+1)+']', 'Mean'])
                alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])
                alphaforgetting= float(saved_params.loc[saved_params['name']=='alphaforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                alphalaserforgetting= float(saved_params.loc[saved_params['name']=='alphalaserforgetting_ses['+str(ms_i+1)+']', 'Mean'])
                laserdecay= float(saved_params.loc[saved_params['name']=='laserdecay_ses['+str(ms_i+1)+']', 'Mean'])
                laserdecayloss= float(saved_params.loc[saved_params['name']=='laserdecayloss_ses['+str(ms_i+1)+']', 'Mean'])                
                betalaserbase= float(saved_params.loc[saved_params['name']=='betalaserbase_ses['+str(ms_i+1)+']', 'Mean'])
                betalaserlossbase= float(saved_params.loc[saved_params['name']=='betalaserlossbase_ses['+str(ms_i+1)+']', 'Mean'])
                ses_params = pd.DataFrame()
                ses_params['Parameter'] = ['βWaterLoss','βLaserLoss_init','βWater','Bias', 
                                           'βLaser_init', 'αWater', 'αLaser','αWaterLoss',
                                           'αLaserLoss','αLaserWinDecay','αLaserLossDecay','LaserWinBaseline', 
                                           'LaserLossBaseline']
                ses_params['values'] = [dmwater_ses, dmlaser_ses, beta_ses, side_ses, laser_ses, 
                                    alpha, alphalaser, alphaforgetting, alphalaserforgetting, laserdecay,
                                    laserdecayloss,betalaserbase,betalaserlossbase]
                ses_params['mouse'] = sub_idx[ms_i]
                ses_params['ses'] = sess_idx[ms_i]
                data = pd.concat([data,ses_params])
    return data
def recovery_params(params_real, recovered_params, standata, phi_a=False, laserdecay=True, z_score=True):
        params_real['fitting_type'] = 'Normal'
        recovered_params['fitting_type'] = 'Recovery'
        saved_params = pd.concat([params_real,recovered_params])
        if laserdecay==True:
            var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                        'alphastay_ses':'αStay', 'laser_ses':'βLaser_init', 'betalaserbase_ses':'βLaser','betalaserlossbase_ses':'βLaserLoss',
                        'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                        'alphalaserforgetting_ses':'αLaserForget',
                        'dmlaser_ses':'βLaserLoss_init', 'dmwater_ses':'βWaterLoss',
                        'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                        'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                        'laserdecay_ses':'αlaserdecay',
                        'balphastay_ses':'αBStay'}
        else:
            var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                        'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                        'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                        'alphalaserforgetting_ses':'αLaserForget',
                        'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                        'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                        'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                        'balphastay_ses':'αBStay'}           
 
        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        saved_params_new = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
        saved_params = saved_params_new
        params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
            (saved_params['name'].str.contains('s\[')),['name','Mean','fitting_type']]
        params['parameter']=params['name'].str.rstrip('[123456789102030]')
        params['parameter'] = params['parameter'].map(var_names)
        if phi_a==True:
            params.loc[(params['parameter'].str.contains('α')), 'Mean'] = \
                    phi_approx(params.loc[(params['parameter'].str.contains('α')), 'Mean']/np.sqrt(2))
        if z_score==True:
            for par in params.parameter.unique():
                params.loc[params['parameter']==par,'Mean'] = zscore(params.loc[params['parameter']==par,'Mean'], nan_policy='omit')

        sns.catplot(data=params, x='fitting_type', col='parameter' ,y='Mean', hue='name',kind='point',
                    legend=False, color= 'k')
        plt.ylabel('Coefficients')
        plt.legend().remove()
        sns.despine()
        plt.tight_layout()     
def plot_corr_analysis_reduced_confusion (standata,model_standard, reinforce_w_stay):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    Return: Plot correlation heatmaps for DeltaQ and predicted choice
    '''
    corr_delta_q, model_labels = correlation_matrix_reduced_confusion(standata, 'deltaQ', model_standard, reinforce_w_stay)
    corr_water, model_labels_water = correlation_matrix_reduced_confusion(standata, 'Qwater', model_standard, reinforce_w_stay)
    corr_stay, model_labels_stay = correlation_matrix_reduced_confusion(standata, 'Qstay', model_standard, reinforce_w_stay)
    corr_laser, model_labels_laser = correlation_matrix_reduced_confusion(standata, 'Qlaser', model_standard, reinforce_w_stay)

    fig, ax = plt.subplots(2,2)
    plt.sca(ax[0,0])
    plot_corr_matrix(corr_delta_q, model_labels,vmin=0,vmax=1)
    plt.title('ΔQ')
    sns.despine()
    plt.tight_layout()
    plt.sca(ax[0,1])
    plot_corr_matrix(corr_water, model_labels_water,vmin=0,vmax=1)
    plt.title('ΔWater')
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1,0])
    plot_corr_matrix(corr_laser, model_labels_laser,vmin=0,vmax=1)
    plt.title('ΔLaser')
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1,1])
    plot_corr_matrix(corr_stay, model_labels_stay,vmin=0,vmax=1)
    plt.title('ΔStay')
    plt.tight_layout()
    sns.despine()
    plt.tight_layout()
def correlation_matrix_reduced_confusion(standata, corr_var, model_standard,reinforce_w_stay, mode='standard'):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    '''
    standard = q_learning_model(standata,saved_params=model_standard)[corr_var].to_numpy()
    rnfrc_stay = reinforce_model_w_stay(standata,saved_params=reinforce_w_stay)[corr_var].to_numpy()
    rnfrc_stay_m = reinforce_model_w_stay(standata,saved_params=model_standard)[corr_var].to_numpy()
    standard_m = q_learning_model(standata,saved_params=reinforce_w_stay)[corr_var].to_numpy()

    standard_waters =model_standard.loc[(model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('beta_ses\['))]
    standard_lasers=model_standard.loc[(model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('laser_ses\['))]
    standard_stays=model_standard.loc[(model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('stay_ses\['))]
    standard_alphas=model_standard.loc[(model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('s\[')) & (model_standard['name'].str.contains('alpha'))]
    reinforce_w_stay_alphas=reinforce_w_stay.loc[(reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('s\[')) & (reinforce_w_stay['name'].str.contains('alpha'))]
    reinforce_w_stay_waters=reinforce_w_stay.loc[(reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('beta_ses\['))]
    reinforce_w_stay_lasers=reinforce_w_stay.loc[(reinforce_w_stay['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('laser_ses\['))]
    reinforce_w_stay_stays=reinforce_w_stay.loc[(reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('stay_ses\['))]
    standard_wo_waters =model_standard.loc[(~((model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('beta_ses\['))))]
    standard_wo_lasers=model_standard.loc[~((model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('laser_ses\[')))]
    standard_wo_stays=model_standard.loc[~((model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('stay_ses\[')))]
    standard_wo_alphas=model_standard.loc[~((model_standard['name'].str.endswith(']')) &
        (model_standard['name'].str.contains('alpha')))]
    reinforce_w_stay_wo_waters=reinforce_w_stay.loc[~(
        reinforce_w_stay['name'].str.contains('beta_ses\['))]
    reinforce_w_stay_wo_lasers=reinforce_w_stay.loc[~((reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('laser_ses\[')))]
    reinforce_w_stay_wo_stays=reinforce_w_stay.loc[~((reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('stay_ses\[')))]
    reinforce_w_stay_wo_alphas=reinforce_w_stay.loc[~((reinforce_w_stay['name'].str.endswith(']')) &
        (reinforce_w_stay['name'].str.contains('alpha')))]

    reinforce_alphas_m = reinforce_model_w_stay(standata,saved_params=pd.concat([standard_alphas,reinforce_w_stay_wo_alphas]))[corr_var].to_numpy()
    standard_alphas_m = q_learning_model(standata,saved_params=pd.concat([reinforce_w_stay_alphas,standard_wo_alphas]))[corr_var].to_numpy()
    reinforce_water_m = reinforce_model_w_stay(standata,saved_params=pd.concat([standard_waters,reinforce_w_stay_wo_waters]))[corr_var].to_numpy()
    standard_water_m = q_learning_model(standata,saved_params=pd.concat([reinforce_w_stay_waters,standard_wo_waters]))[corr_var].to_numpy()
    reinforce_laser_m = reinforce_model_w_stay(standata,saved_params=pd.concat([standard_lasers,reinforce_w_stay_wo_lasers]))[corr_var].to_numpy()
    standard_laser_m = q_learning_model(standata,saved_params=pd.concat([reinforce_w_stay_lasers,standard_wo_lasers]))[corr_var].to_numpy()
    reinforce_stay_m = reinforce_model_w_stay(standata,saved_params=pd.concat([standard_stays,reinforce_w_stay_wo_stays]))[corr_var].to_numpy()
    standard_stay_m = q_learning_model(standata,saved_params=pd.concat([reinforce_w_stay_stays,standard_wo_stays]))[corr_var].to_numpy()

    corr_matrix = np.corrcoef([standard,standard_m,standard_alphas_m,standard_water_m,standard_laser_m,standard_stay_m,
                    rnfrc_stay,rnfrc_stay_m,reinforce_alphas_m,reinforce_water_m,reinforce_laser_m,reinforce_stay_m])

    model_labels = ['standard','standard_m','standard_alphas_m','standard_water_m','standard_laser_m','standard_stay_m',
                    'rnfrc_stay','rnfrc_stay_m','reinforce_alphas_m','reinforce_water_m','reinforce_laser_m','reinforce_stay_m']

    return corr_matrix, model_labels
def accuracy_by_section(model_data, n_sections=8):
    model_data = model_data.reset_index()
    model_data['section'] = pd.cut(model_data['index'],bins=n_sections)
    model_data['bin_prediction'] = 1*(model_data['predicted_choice']>0.5)
    model_data['correct_guess'] = model_data['bin_prediction']==model_data['choices']
    breakdown = model_data.groupby(['section']).mean()['correct_guess']
    return breakdown
def time_to_laserbaseline(model,standata,params, variable='betalaser'):
        data = model(standata,saved_params=params)
        data = data.reset_index()
        data['betaqlaser'] = data['Qlaser']*data['betalaser']
        data['id'] = data['mouse']*100+data['ses']
        sns.lineplot(data=data, x='index', y=variable, hue='id', palette='mako')
def plot_params_reinforce_laserdecay_stay(saved_params, stan_data, save=False, output_path=None):
        var_names = {'beta_ses': 'βWater','sides':'Bias', 'laser_ses':'βLaser_init', 'ep_ses':'εStay', 'alpha_ses':'αWater',
                    'alphalaser_ses':'αLaser', 'alphastay_ses':'αStay', 'alphaforgetting_ses':'αWaterForget', 'alphalaserforgetting_ses':'αLaserForget',
                    'laserdecay_ses':'αLaserDecay','betalaserbase_ses':'βLaserBase','dmwater_ses':'βWaterLoss','dmlaser_ses':'βLaserLoss_init',
                    'betalaserlossbase_ses':'βLaserLossBase', 'laserdecayloss_ses':'αLaserDecayLoss'}

        category = {'beta_ses': 'beta_bases','sides':'bias', 'laser_ses':'beta_inits', 'ep_ses':'beta_bases', 'alpha_ses':'alphas',
                    'alphalaser_ses':'alphas', 'alphastay_ses':'alphas', 'alphaforgetting_ses':'alphas', 'alphalaserforgetting_ses':'alphas',
                    'laserdecay_ses':'decays','betalaserbase_ses':'beta_bases','dmwater_ses':'beta_bases','dmlaser_ses':'beta_inits',
                    'betalaserlossbase_ses':'beta_bases', 'laserdecayloss_ses':'decays'}

        pal = {'beta_ses': 'dodgerblue','sides':'gray', 'laser_ses':'orange', 'ep_ses':'gray', 'alpha_ses':'dodgerblue',
                    'alphalaser_ses':'orange', 'alphastay_ses':'gray', 'alphaforgetting_ses':'dodgerblue', 'alphalaserforgetting_ses':'orange',
                    'laserdecay_ses':'orange','betalaserbase_ses':'orange','dmwater_ses':'dodgerblue','dmlaser_ses':'orange',
                    'betalaserlossbase_ses':'orange', 'laserdecayloss_ses':'orange'}    

        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        saved_params_new = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
        saved_params = saved_params_new
        params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
            (saved_params['name'].str.contains('s\[')),['name','Mean']]
        params['parameter']=params['name'].str.rstrip('[123456789102030]')
        params['category'] = params['parameter'].map(category)
        params['pal']= params['parameter'].map(pal)
        params['parameter'] = params['parameter'].map(var_names)
        fig, ax = plt.subplots(2,3, figsize=(10,12))
        plt.sca(ax[0,0])
        sns.swarmplot(data=params.loc[params['category']=='beta_bases'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='beta_bases'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='beta_bases','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Betas')
        plt.sca(ax[0,1])
        sns.swarmplot(data=params.loc[params['category']=='alphas'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='alphas'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='alphas','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Alphas')
        plt.sca(ax[0,2])
        sns.swarmplot(data=params.loc[params['category']=='bias'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='bias'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='bias','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Bias')
        plt.sca(ax[1,0])
        sns.swarmplot(data=params.loc[params['category']=='beta_inits'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='beta_inits'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='beta_inits','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Initializations')
        plt.sca(ax[1,1])
        sns.swarmplot(data=params.loc[params['category']=='decays'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='decays'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='decays','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.title('Decays')
        plt.xticks(rotation=45)
        sns.despine()
        plt.tight_layout()
def plot_params_standard_laserdecay(saved_params, stan_data, save=False, output_path=None):
        var_names = {'beta_ses': 'βWater','sides':'Bias', 'laser_ses':'βLaser_init', 'stay_ses':'βStay', 'alpha_ses':'αWater',
                    'alphalaser_ses':'αLaser', 'alphastay_ses':'αStay', 
                    'laserdecay_ses':'αLaserDecay', 'betalaserbase_ses':'βLaserBaser'}

        category = {'beta_ses': 'beta_bases','sides':'bias', 'laser_ses':'beta_inits', 'stay_ses':'beta_bases', 'alpha_ses':'alphas',
                    'alphalaser_ses':'alphas', 'alphastay_ses':'alphas', 
                    'laserdecay_ses':'decays',  'betalaserbase_ses':'beta_bases'}

        pal = {'beta_ses': 'dodgerblue','sides':'gray', 'laser_ses':'orange', 'stay_ses':'gray', 'alpha_ses':'dodgerblue',
                    'alphalaser_ses':'orange', 'alphastay_ses':'gray',
                    'laserdecay_ses':'orange',  'betalaserbase_ses':'orange'}    

        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']
        saved_params_new = pd.DataFrame()
        for ms_i in np.arange(NSxNSESS):
            if NT_all[ms_i]>0:
                saved_params_new = pd.concat([saved_params_new, saved_params.loc[(saved_params['name'].str.endswith('['+str(ms_i+1)+']'))]])
        saved_params = saved_params_new
        params=saved_params.loc[(saved_params['name'].str.endswith(']')) &
            (saved_params['name'].str.contains('s\[')),['name','Mean']]
        params['parameter']=params['name'].str.rstrip('[123456789102030]')
        params['category'] = params['parameter'].map(category)
        params['pal']= params['parameter'].map(pal)
        params['parameter'] = params['parameter'].map(var_names)
        fig, ax = plt.subplots(2,3, figsize=(10,12))
        plt.sca(ax[0,0])
        sns.swarmplot(data=params.loc[params['category']=='beta_bases'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='beta_bases'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='beta_bases','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Betas')
        plt.sca(ax[0,1])
        sns.swarmplot(data=params.loc[params['category']=='alphas'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='alphas'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='alphas','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Alphas')
        plt.sca(ax[0,2])
        sns.swarmplot(data=params.loc[params['category']=='bias'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='bias'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='bias','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Bias')
        plt.sca(ax[1,0])
        sns.swarmplot(data=params.loc[params['category']=='beta_inits'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='beta_inits'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='beta_inits','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.xticks(rotation=45)
        plt.title('Initializations')
        plt.sca(ax[1,1])
        sns.swarmplot(data=params.loc[params['category']=='decays'], 
                        x='parameter', y='Mean', color='k')
        sns.barplot(data=params.loc[params['category']=='decays'], x='parameter', y='Mean',
                    palette=params.loc[params['category']=='decays','pal'])
        plt.ylabel('Coefficient')
        plt.xlabel(' ')
        plt.title('Decays')
        plt.xticks(rotation=45)
        sns.despine()
        plt.tight_layout()

if __name__=='__main__':
    ## 0. Load data
    data=load_data()
    standata = make_stan_data(data)
    standata_recovery = load_sim_data() # Raw data for simulations
    ## 1. Standard
    model_standard = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output/summary.csv')
    accu_standard = pd.DataFrame()
    accu_standard['Accuracy'] = q_learning_model(standata,saved_params=model_standard)['acc'].unique()
    accu_standard['Model'] = 'Standard'
    _, _, _, sim_standard = simulate_q_learning_model(standata_recovery,saved_params=model_standard)
    _, _, _, sim_standard_new = simulate_q_learning_model_new(standata_recovery,saved_params=model_standard)
    _, _, _, sim_standard_noqlaser_new = simulate_q_learning_model_noqlaser_new(standata_recovery,saved_params=model_standard)
    _, _, _, sim_standard_noqwater_new = simulate_q_learning_model_noqwater_new(standata_recovery,saved_params=model_standard)
    _, _, _, sim_standard_noqwaterorlaser_new = simulate_q_learning_model_noqwaterorlaser_new(standata_recovery,saved_params=model_standard)

    # 0.1.1 REINFORCE alpa laser decay w stay
    REINFORCEalphalaserdecaystay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywin_mixedstay/output/summary.csv')
    rdata = reinforce_model_alphalaserdecay_mixed_perseveration(standata,saved_params=REINFORCEalphalaserdecaystay)
    _, _, _, sim_REINFORCEwinalhalaserdecaystay = simulate_reinforce_alphalaserdecay_win_stay(standata_recovery,saved_params=REINFORCEalphalaserdecaystay)
    rdata['laser_block'] = data['laser_block']
    rdata['pchoice_correct'] = 1*((rdata['predicted_choice']>=0.5)==rdata['choices'])
    opto_rdata_acc = rdata.loc[rdata['laser_block']==1,'pchoice_correct'].mean()
    water_rdata_acc = rdata.loc[rdata['laser_block']==0,'pchoice_correct'].mean()
    ## 0.2 Q learning laser decay w alphalaserdecay 
    standard_laserdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_alphalaserdecay/output/summary.csv')
    qdata = q_learning_model_alphalaserdecay(standata,saved_params=standard_laserdecay)
    _, _, _, sim_standard_laserdecay = simulate_q_learning_model_alphalaserdecay(standata_recovery,saved_params=standard_laserdecay)
    qdata['laser_block'] = data['laser_block']
    qdata['pchoice_correct'] = 1*((qdata['predicted_choice']>=0.5)==qdata['choices'])
    opto_qdata_acc = qdata.loc[qdata['laser_block']==1,'pchoice_correct'].mean()
    water_qdata_acc = qdata.loc[qdata['laser_block']==0,'pchoice_correct'].mean() 

    ## 0.1 REINFORCE laser decay w stay
    REINFORCEwinlosslaserdecaystay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywinloss_mixedstay/output/summary.csv')
    _, _, _, sim_REINFORCEwinlosslaserdecaystay = simulate_reinforce_laserdecay_winloss_stay(standata_recovery,saved_params=REINFORCEwinlosslaserdecaystay)
    _, _, _, sim_REINFORCEwinlosslaserdecaystay_nolaser = simulate_reinforce_laserdecay_winloss_stay_nolaser(standata_recovery,saved_params=REINFORCEwinlosslaserdecaystay)
    ## 0.2 Q learning laser decay w stay
    standard_laserdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_laserdecay/output/summary.csv')
    _, _, _, sim_standard_laserdecay = simulate_q_learning_model_laserdecay(standata_recovery,saved_params=standard_laserdecay)
    _, _, _, sim_standard_laserdecay_nolaser = simulate_q_learning_model_laserdecay_nolaser(standata_recovery,saved_params=standard_laserdecay)
    
    ## 1. Standard_reduced
    model_standard_reduced = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_reduced/output/summary.csv')
    accu_standard_reduced = pd.DataFrame()
    accu_standard_reduced['Accuracy'] = q_learning_model_reduced(standata,saved_params=model_standard_reduced)['acc'].unique()
    accu_standard_reduced['Model'] = 'Standard reduced'

    ## 1. Standard_reduced_with_stay
    model_standard_reduced_stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_reduced_stay/output/summary.csv')
    accu_standard_reduced_stay = pd.DataFrame()
    accu_standard_reduced_stay['Accuracy'] = q_learning_model_reduced_stay(standata,saved_params=model_standard_reduced_stay)['acc'].unique()
    accu_standard_reduced_stay['Model'] = 'Standard reduced stay'
    ## 2. Standard with forgetting
    model_forgetting = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandisimulate_q_learning_model_alphalaserdecayt/stan_fits/standard_w_forgetting/output/summary.csv')
    accu_standard_w_forgetting = pd.DataFrame()
    accu_standard_w_forgetting['Accuracy'] = q_learning_model_w_forgetting(standata,saved_params=model_forgetting)['acc'].unique()
    accu_standard_w_forgetting['Model'] = 'F-Q'
    _, _, _, sim_standard_w_forgetting = simulate_q_learning_w_forgetting(standata_recovery,saved_params=model_forgetting)
    ## 3. REINFORCE
    reinforce = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE/output/summary.csv')
    accu_reinforce = pd.DataFrame()
    accu_reinforce['Accuracy'] = reinforce_model(standata,saved_params=reinforce)['acc'].unique()
    accu_reinforce['Model'] = 'REINFORCE'
    _, _, _, sim_reinforce = simulate_reinforce_model(standata_recovery,saved_params=reinforce)
    ## 3. REINFORCE reduced
    reinforce_reduced = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_reduced/output/summary.csv')
    accu_reinforce_reduced = pd.DataFrame()
    accu_reinforce_reduced['Accuracy'] = reinforce_model_reduced(standata,saved_params=reinforce_reduced)['acc'].unique()
    accu_reinforce_reduced['Model'] = 'REINFORCE reduced'
    ## 4. REINFORCE w 2 stays
    reinforce_2stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_w_2stay/output/summary.csv')
    accu_reinforce_2stay = pd.DataFrame()
    accu_reinforce_2stay['Accuracy'] = reinforce_model_w_2stay(standata,saved_params=reinforce_2stay)['acc'].unique()
    accu_reinforce_2stay['Model'] = 'REINFORCE w 2 stays'
    _, _, _, sim_reinforce_w_2stays = simulate_reinforce_model_w_2stay(standata_recovery,saved_params=reinforce_2stay)
    ## 5. Laser==RPE (Pending)
    LRPE = chains_2_summary('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/LRPE/old_output/output')
    accu_LRPE = pd.DataFrame()
    accu_LRPE['Accuracy'] = q_learning_model_Lrpe(standata,saved_params=LRPE)['acc'].unique()
    accu_LRPE['Model'] = 'Laser=RPE'
    # 6. Laser==RPE conditional
    ## RPEc = pd.read_csv('')
    LRPEc = chains_2_summary('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/LRPEconditional/output')
    accu_LRPEc = pd.DataFrame()
    accu_LRPEc['Accuracy'] = q_learning_model_Lrpeconditional(standata,saved_params=LRPEc)['acc'].unique()
    accu_LRPEc['Model'] = 'Laser=RPE/Reward'
    ## 7. 2 stays
    a2stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_w_2stays/output/summary.csv')
    accu_2stay = pd.DataFrame()
    accu_2stay['Accuracy'] = q_learning_model_2alphas(standata,saved_params=a2stay)['acc'].unique()
    accu_2stay['Model'] = '2 QStay'
    ## 8. Laser==RPE conditional with forgetting
    LRPEcf = chains_2_summary('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/LRPE_condtiional_w_forgetting')
    accu_LRPEcf = pd.DataFrame()
    accu_LRPEcf['Accuracy'] = q_learning_model_Lrpeconditional_w_forgetting(standata,saved_params=LRPEcf)['acc'].unique()
    accu_LRPEcf['Model'] = 'Laser=RPE/Reward w/ F'
    # 9. Lapse standard
    stdlapse = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_lapse/output/summary.csv')
    accu_stdlapse = pd.DataFrame()
    accu_stdlapse['Accuracy'] = q_learning_model_lapse(standata, saved_params=stdlapse)['acc'].unique()
    accu_stdlapse['Model'] = 'Lapse'
    # 10. REINFORCE w Stay
    reinforce_w_stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_w_stay/output/summary.csv')
    accu_reinforce_w_stay = pd.DataFrame()
    accu_reinforce_w_stay['Accuracy'] = reinforce_model_w_stay(standata,saved_params=reinforce_w_stay)['acc'].unique()
    accu_reinforce_w_stay['Model'] = 'REINFORCE w Stay'
    _, _, _, sim_reinforce_w_stays = simulate_reinforce_model_w_stay(standata_recovery,saved_params=reinforce_w_stay)
    # 11. Standard w 2learning
    q_learning_2learning = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_w_2alphas/output/summary.csv')
    accu_2learning = pd.DataFrame()
    accu_2learning['Accuracy'] = q_learning_model_w_2learning(standata,saved_params=q_learning_2learning)['acc'].unique()
    accu_2learning['Model'] = 'Standard w 2 learning rates'
    _, _, _, sim_q_learning_w_2learning = simulate_q_learning_w_2learning(standata_recovery,saved_params=q_learning_2learning)
    # 12. pearce_hall_model
    ph_model = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/pearce_hall_2k/output/summary.csv')
    accu_pearce_hall = pd.DataFrame()
    accu_pearce_hall['Accuracy'] = pearce_hall_model(standata,saved_params=ph_model)['acc'].unique()
    accu_pearce_hall['Model'] = 'Pearce-Hall'
    # 13. Standard w asymmetric learning
    asymmetric_alpha = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_asymmetric_alpha/output/summary.csv')
    accu_asym_learning = pd.DataFrame()
    accu_asym_learning['Accuracy'] = q_learning_model_asymmetric(standata,saved_params=asymmetric_alpha)['acc'].unique()
    accu_asym_learning['Model'] = 'Asymmetric Learning rates'
    # 14. Standard w asymmetric learning no stay
    asymmetric_alpha_nostay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_asymmetric_alpha_nostay/output/summary.csv')
    accu_asym_learning_nostay = pd.DataFrame()
    accu_asym_learning_nostay['Accuracy'] = q_learning_model_asymmetric_nostay(standata,saved_params=asymmetric_alpha_nostay)['acc'].unique()
    accu_asym_learning_nostay['Model'] = 'Asymmetric Learning rates No Stay'
    # 15. 'REINFORCE w linear decay
    reinforce_lineardecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecay_linear/output/summary.csv')
    accu_reinforce_lineardecay = pd.DataFrame()    
    accu_reinforce_lineardecay['Accuracy'] = reinforce_model_laserdecay_linear(standata,saved_params=reinforce_lineardecay)['acc'].unique()
    accu_reinforce_lineardecay['Model'] = 'REINFORCE w linear decay'
    # 16. 'REINFORCE w exp decay
    reinforce_expdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecay/output/summary.csv')
    accu_reinforce_expdecay = pd.DataFrame()    
    accu_reinforce_expdecay['Accuracy'] = reinforce_model_laserdecay_exp(standata,saved_params=reinforce_expdecay)['acc'].unique()
    accu_reinforce_expdecay['Model'] = 'REINFORCE w exp decay'
    _, _, _, sim_reinforce_decay_new = simulate_reinforce_laserdecay_exp(standata_recovery,saved_params=reinforce_expdecay)
    # 16. REINFORCE w exp decay 200 trials
    standata_200 = make_stan_data(load_data(),n_trials=200)
    reinforce_expdecay_200 = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecay_200/output/summary.csv')
    accu_reinforce_expdecay_200 = pd.DataFrame()    
    accu_reinforce_expdecay_200['Accuracy'] = reinforce_model_laserdecay_exp(standata_200,saved_params=reinforce_expdecay_200)['acc'].unique()
    accu_reinforce_expdecay_200['Model'] = 'REINFORCE w exp decay 200'
    # 16. REINFORCE w exp decay 600 trials
    standata_600 = make_stan_data(load_data(),n_trials=600)
    reinforce_expdecay_600 = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecay_600/output/summary.csv')
    accu_reinforce_expdecay_600 = pd.DataFrame()    
    accu_reinforce_expdecay_600['Accuracy'] = reinforce_model_laserdecay_exp(standata_600,saved_params=reinforce_expdecay_600)['acc'].unique()
    accu_reinforce_expdecay_600['Model'] = 'REINFORCE w exp decay 600'
    # 17. REINFORCE all decay (doesn't fit great)
    reinforce_allexpdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_alldecay/output/summary.csv')
    accu_reinforce_allexpdecay = pd.DataFrame()    
    accu_reinforce_allexpdecay['Accuracy'] = reinforce_model_alldecay_exp(standata,saved_params=reinforce_allexpdecay)['acc'].unique()
    accu_reinforce_allexpdecay['Model'] = 'REINFORCE w exp decay'
    # 17. REINFORCE laser exp decay, win and loss decay
    laserdecaywinloss = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywinloss/output/summary.csv')
    accu_reinforce_allexpdecay = pd.DataFrame()
    accu_reinforce_allexpdecay['Accuracy'] = reinforce_model_laserdecaywinloss_exp(standata,saved_params=laserdecaywinloss)['acc'].unique() 
    accu_reinforce_allexpdecay['Model'] = 'REINFORCE w exp decay for win and loss'

    # 19. REINFORCE mixed perseveration
    mixed_perseveration = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_mixedperseveration/output/summary.csv')
    accu_reinforce_allexpdecay = pd.DataFrame()
    accu_reinforce_allexpdecay['Accuracy'] = reinforce_model_mixed_perseveration(standata,saved_params=mixed_perseveration)['acc'].unique() 
    accu_reinforce_allexpdecay['Model'] = 'REINFORCE w stay mixed model'

    # 17. Recovery parameters
    standard_recovery = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/output/summary.csv')
    reinforce_w_stay_recovery = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/outputsummary.csv')
    confusion_recover_REINFORCEstay_w_standard = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/confusion_recover_REINFORCEstay_w_standard/output/summary.csv')
    confusion_recover_standard_w_REINFORCEstay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/confusion_recover_standard_w_REINFORCEstay/output/summary.csv')
    standard_sim_r = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/r_sim.npy')
    standard_sim_l = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/l_sim.npy')
    standard_sim_c = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/standard_recovery/c_sim.npy')
    reinforce_sim_r = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/r_sim.npy')
    reinforce_sim_l = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/l_sim.npy')
    reinforce_sim_c = np.load('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models/REINFORCE_recovery/c_sim.npy')

    # 17. Recovery parameters Laser Decay
    standard_recovery = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_laserdecay/output/summary.csv')
    standard_laserdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models_laser_decay/standard_recovery/output/summary.csv')



    #plot_params
    plot_params(model_standard,standata,phi_a=False)
    plot_params(model_forgetting,standata)
    plot_params(reinforce,standata)
    plot_params(LRPE,standata)
    plot_params(LRPEc,standata)
    plot_params(a2stay,standata)
    plot_params(LRPEcf,standata)
    plot_params(stdlapse,standata,phi_a=False)
    plot_params(reinforce_2stay,standata,phi_a=False)
    plot_params(reinforce_w_stay,standata,phi_a=False)
    plot_params(q_learning_2learning,standata,phi_a=False)
    plot_params(asymmetric_alpha_nostay, standata,phi_a=False)
    plot_params(reinforce_expdecay, standata,phi_a=False)
    plot_params(laserdecaywinloss, standata,phi_a=False)
    plot_params(reinforce_reduced, standata,phi_a=False)
    plot_params(model_standard_reduced, standata,phi_a=False)
    plot_params(model_standard_reduced_stay, standata,phi_a=False)
    plot_params(mixed_perseveration, standata,phi_a=False)

    # plot differences in parameters
    plot_params_diff_learning(q_learning_2learning,standata,phi_a=False)

    # Concatenate and plot
    top_summary = pd.concat([accu_standard, accu_reinforce, accu_reinforce_lineardecay, accu_reinforce_expdecay])
    models_summary = pd.concat([accu_standard, accu_standard_w_forgetting, accu_reinforce,
                                accu_LRPE, accu_LRPEc, accu_2stay, accu_LRPEcf, accu_stdlapse,
                                accu_2learning, accu_reinforce_w_stay, accu_reinforce_2stay])

    reinforce_summary = pd.concat([accu_reinforce, accu_reinforce_w_stay, accu_reinforce_2stay])
    plot_model_comparison(models_summary)
    compare_accuracy_reinforce(reinforce_summary)
    # Plot correlation_matrix
    plot_corr_analysis(standata, 'deltaQ', model_standard,model_forgetting,
                            reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse)
    # Check where accuracy drops in standard model
    data = q_learning_model(standata,saved_params=model_standard)
    data['correct_prediction']= 1*(1*(data['predicted_choice']>0.5)==data['choices'])

    # Plot model behavior on simulations
    fig, ax  =  plt.subplots(3,2)
    plt.sca(ax[0,0])
    #original_choices = q_learning_model_w_2learning(standata,saved_params=q_learning_2learning)['choices']
    #original = sim_standard.copy()
    #original['choices'] = original_choices
    original = stan_data_to_df(standata_recovery,standata)
    original['tb'] = original['tb']-1
    check_model_behavior(original)
    plt.title('Real data')
    plt.sca(ax[0,1])
    check_model_behavior(sim_standard)
    plt.title('Q-Learning simulation wo debiasing')
    plt.sca(ax[1,0])
    sim_standard_new.loc[sim_standard_new['tb']==0,'tb']=1
    check_model_behavior(sim_standard_new)
    plt.title('Q-Learning simulation w debiasing')
    plt.sca(ax[1,1])
    #sim_standard_noqwater_new.loc[sim_standard_noqwater_new['tb']==0,'tb']=1
    #check_model_behavior(sim_standard_noqwater_new)
    sim_standard_noqwaterorlaser_new.loc[sim_standard_noqwaterorlaser_new['tb']==0,'tb']=1
    check_model_behavior(sim_standard_noqwaterorlaser_new)
    plt.title('Q-Learning simulation w debiasing wo water or laser')
    plt.sca(ax[2,0])
    sim_standard_noqwater_new.loc[sim_standard_noqwater_new['tb']==0,'tb']=1
    check_model_behavior(sim_standard_noqwater_new)
    plt.title('Q-Learning simulation w debiasing no qwater')
    plt.sca(ax[2,1])
    sim_standard_noqlaser_new.loc[sim_standard_noqlaser_new['tb']==0,'tb']=1
    check_model_behavior(sim_standard_noqlaser_new)
    plt.title('Q-Learning simulation w debiasing no qlaser')


    # Plot model comparison of REINFORCE_w_stay vs Q_learning
    reinforce_w_stay['model_name'] = 'reinforce_w_stay'
    model_standard['model_name'] = 'q_learning'
    pd_compare = pd.concat([reinforce_w_stay, model_standard])
    compare_two_models(pd_compare, stan_data)

    # Plot recovery
    recovery_params(reinforce_w_stay, reinforce_w_stay_recovery)
    recovery_params(reinforce_w_stay, confusion_recover_REINFORCEstay_w_standard)
    recovery_params(model_standard, confusion_recover_REINFORCEstay_w_standard)
    recovery_params(reinforce_w_stay, confusion_recover_standard_w_REINFORCEstay)
    recovery_params(model_standard, confusion_recover_standard_w_REINFORCEstay)
    recovery_params(confusion_recover_REINFORCEstay_w_standard, confusion_recover_standard_w_REINFORCEstay)


    # mix parameters and look at plot_correlations
    plot_corr_analysis_reduced_confusion (standata,model_standard, reinforce_w_stay)

    # Confussion matrix of the accuracies
    confusion_accu_matrix(standata_recovery)

    # Reinforce decay behavior
    sim_reinforce_decay_new.loc[sim_reinforce_decay_new['tb']==0,'tb']=1
    check_model_behavior(sim_reinforce_decay_new)
    plt.title('REINFORCE w/ exp laser decay')

    # Differences in parameters based on trials fitted
    params=params_reinforce_model_laserdecay_exp(standata, reinforce_expdecay)
    params['n_trials']='All'
    params_200=params_reinforce_model_laserdecay_exp(standata, reinforce_expdecay_200)
    params_200['n_trials']='200'
    params_600=params_reinforce_model_laserdecay_exp(standata, reinforce_expdecay_600)
    params_600['n_trials']='600'
    params_all = pd.concat([params,params_200])

    def plot_params_exp(params):
        ord= ['βWater_init', 'βLaser_init', 'βWaterLoss', 'βLaserLoss', 'Bias',  'αWater', 'αLaser', 'αWaterForget', 'αLaserForget', 'αLaserDecay']
        pal = ['dodgerblue', 'orange', 'dodgerblue', 'gray', 'orange',
        'dodgerblue', 'orange', 'dodgerblue', 'orange', 'orange']
        params['id'] = params['mouse']*100+ params['ses']
        sns.swarmplot(data=params.loc[np.isin(params['Parameter'], ['βWaterLoss', 'βLaserLoss', 'βWater_init', 'Bias', 'βLaser_init'])], 
                    x='Parameter', y='values',hue='id', order=ord)
        sns.barplot(data=params.loc[np.isin(params['Parameter'], ['βWaterLoss', 'βLaserLoss', 'βWater_init', 'Bias', 'βLaser_init'])], 
                    x='Parameter', y='values',color='k', order=ord)
        plt.legend().remove()
        plt.ylabel('β abd Bias Coefficients')
        ax2 = plt.twinx()
        sns.swarmplot(data=params.loc[np.isin(params['Parameter'], ['αWater', 'αLaser', 'αWaterForget', 'αLaserForget', 'αLaserDecay'])], 
                    x='Parameter', y='values',hue='id', ax=ax2,order=ord)
        ax2.set_ylim(-1,1)
        sns.barplot(data=params.loc[np.isin(params['Parameter'], ['αWater', 'αLaser', 'αWaterForget', 'αLaserForget', 'αLaserDecay'])], 
                    x='Parameter', y='values',color='k', ax=ax2,order=ord)
        ax2.set_ylim(-1,1)
        plt.legend().remove()
        ax2.set_ylabel('α Coefficients')    
    def diff_params_exp(params_all, params_200, params_600):
        fig, ax  = plt.subplots(2,5)
        for i, par in enumerate(params_all['Parameter'].unique()):
            plt.sca(ax[int(i/5),i%5])
            sns.pointplot(data=params_all.loc[params_all['Parameter']==par],x='n_trials',y='values',
                          ci=68)
            plt.title(par)
            plt.ylabel('Coefficient')
            plt.tight_layout()
    def plot_params_exp_winloss(params):
        ord= ['βWater_init','βWaterLoss_init', 'Bias', 
              'αWater', 'αLaser', 'αWaterForget', 'αLaserForget', 'αLaserWinDecay', 
              'αLaserLossDecay','LaserWinBaseline', 'LaserLossBaseline']

        params['id'] = params['mouse']*100+ params['ses']
        params['color'] = 'Red'
        params_select = params.loc[(params['Parameter']=='βLaser_init')]
        ids_negative  = params_select.loc[params_select['values']<0, 'id']
        params.loc[np.isin(params['id'],ids_negative), 'color']='blue'
        fig,ax = plt.subplots(1,3)
        plt.sca(ax[0])
        sns.swarmplot(data=params.loc[np.isin(params['Parameter'], ['βLaserLoss_init', 'βLaser_init'])], 
                    x='Parameter', y='values',hue='id')
        sns.barplot(data=params.loc[np.isin(params['Parameter'], ['βLaserLoss_init', 'βLaser_init'])], 
                    x='Parameter', y='values',color='k')
        plt.legend().remove()
        plt.ylabel('β Laser Coefficients')
        plt.sca(ax[1])
        sns.swarmplot(data=params.loc[np.isin(params['Parameter'], ['βWaterLoss', 'βWater', 'Bias'])], 
                    x='Parameter', y='values',hue='id')
        sns.barplot(data=params.loc[np.isin(params['Parameter'], ['βWaterLoss', 'βWater', 'Bias'])], 
                    x='Parameter', y='values',color='k')
        plt.legend().remove()
        plt.ylabel('β abd Bias Coefficients')
        plt.sca(ax[2])
        sns.swarmplot(data=params.loc[np.isin(params['Parameter'], ['αWater', 'αLaser', 'αWaterForget', 'αLaserForget', 'αLaserWinDecay', 
              'αLaserLossDecay','LaserWinBaseline', 'LaserLossBaseline'])], 
                    x='Parameter', y='values',hue='id')
        sns.barplot(data=params.loc[np.isin(params['Parameter'], ['αWater', 'αLaser', 'αWaterForget', 'αLaserForget', 'αLaserWinDecay', 
              'αLaserLossDecay','LaserWinBaseline', 'LaserLossBaseline'])], 
                    x='Parameter', y='values',color='k')
        plt.legend().remove()
        plt.xticks(rotation=45)
        plt.ylabel('α and baseline Coefficients')
        plt.tight_layout()


    params = params_reinforce_model_laserdecay_winloss_exp(standata, laserdecaywinloss)

    
    s_model = q_learning_model(standata,saved_params=model_standard)
    s_reduced_stay_model = q_learning_model_reduced_stay(standata,saved_params=model_standard_reduced_stay)
    reinforce_reduced = reinforce_model_reduced(standata,saved_params=reinforce_reduced)
    r_model = reinforce_model(standata,saved_params=reinforce)
    r_win_loss = reinforce_model_laserdecaywinloss_exp(standata,saved_params=laserdecaywinloss)
    s_reduced_stay_model_ll =np.sum(np.log(abs(s_reduced_stay_model['choices'].replace({0:1, 1:0})-s_reduced_stay_model['predicted_choice']))) #trick to get the log_likelihood for the choice taken by the mouse e.g. c_pred = 0.75, ll_left=0.25 (same as abs(1-0.75)), ll_right=0.75(same as abs(0-0.75)
    r_reduced = np.sum(np.log(abs(reinforce_reduced['choices'].replace({0:1, 1:0})-reinforce_reduced['predicted_choice'])))
    r_model_ll =np.sum(np.log(abs(r_model['choices'].replace({0:1, 1:0})-r_model['predicted_choice'])))
    r_win_loss_ll = np.sum(np.log(abs(r_win_loss['choices'].replace({0:1, 1:0})-r_win_loss['predicted_choice'])))
    s_model_ll =np.sum(np.log(abs(s_model['choices'].replace({0:1, 1:0})-s_model['predicted_choice'])))
    lls = pd.DataFrame()
    lls['model'] = ['Qlearning_reduced','REINFORCE_reduced','Qlearning','REINFORCE','REINFORCE_L_decay']
    lls['ll'] = [s_reduced_stay_model_ll,r_reduced,s_model_ll,r_model_ll,r_win_loss_ll]
    sns.barplot(data=lls, x='model',y='ll', )
    plt.ylim(-7200,-7600)


    print('qlearning_reduced  ' + str(accu_standard_reduced_stay['Accuracy'].mean()))
    print('qlearning  ' + str(accu_standard['Accuracy'].mean()))
    print('REINFORCE_reduced  ' + str(accu_reinforce_reduced['Accuracy'].mean()))
    print('REINFORCE  ' + str(accu_reinforce['Accuracy'].mean()))
    print('REINFORCE_laser_decay  ' + str(accu_reinforce_allexpdecay['Accuracy'].mean()))




    # Winners plots
    fig, ax = plt.subplots(2,3)
    plt.sca(ax[0,0])
    original = stan_data_to_df(standata_recovery,standata)
    check_model_behavior(original)
    plt.title('Original data')
    plt.sca(ax[0,1]) 
    sim_standard_laserdecay.loc[sim_standard_laserdecay['tb']==0,'tb']=1
    check_model_behavior(sim_standard_laserdecay)
    plt.title('Q-learning with laser, laserdecay and stay')
    plt.sca(ax[0,2])      
    sim_REINFORCEwinlosslaserdecaystay.loc[sim_REINFORCEwinlosslaserdecaystay['tb']==0,'tb']=1
    check_model_behavior(sim_REINFORCEwinlosslaserdecaystay)
    plt.title('REINFORCE with laser, laserdecay and stay')
    plt.sca(ax[1,1])      
    sim_standard_laserdecay_nolaser.loc[sim_standard_laserdecay_nolaser['tb']==0,'tb']=1
    check_model_behavior(sim_standard_laserdecay_nolaser)
    plt.title('Q-learning with laser, laserdecay and stay\n'\
                'No laser\n')
    plt.sca(ax[1,2])      
    sim_REINFORCEwinlosslaserdecaystay_nolaser.loc[sim_REINFORCEwinlosslaserdecaystay_nolaser['tb']==0,'tb']=1
    check_model_behavior(sim_REINFORCEwinlosslaserdecaystay_nolaser)
    plt.title('REINFORCE with laser, laserdecay and stay\n' \
                'No laser\n')
    plt.tight_layout

    fig, ax = plt.subplots(1,2)
    plt.sca(ax[0])
    plot_params(standard_laserdecay,standata)
    plt.sca(ax[1])
    plot_params(REINFORCEwinlosslaserdecaystay,standata)





    params_standardlaserdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_laserdecay/output/summary.csv')
    params_reinforcelaserdecaystay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywinloss_mixedstay/output/summary.csv')
    fig,ax=plt.subplots(1,3)
    plt.sca(ax[0])
    time_to_laserbaseline(q_learning_model_laserdecay,standata,params_standardlaserdecay)
    plt.xlabel('Trial')
    plt.ylabel('βLaser')
    plt.xlim(0,125)
    plt.title('Q-learning with laser, laserdecay and stay')
    plt.legend().remove()
    plt.sca(ax[1])
    time_to_laserbaseline(reinforce_model_laserdecay_mixed_perseveration,standata,params_reinforcelaserdecaystay)
    plt.ylabel('βLaser')
    plt.xlabel('Trial')
    plt.xlim(0,125)
    plt.legend().remove()
    plt.title('REINFORCE with laser, laserdecay and stay')
    plt.sca(ax[2])
    time_to_laserbaseline(reinforce_model_laserdecay_mixed_perseveration,standata,params_reinforcelaserdecaystay, variable='betalaserloss')
    plt.ylabel('βLaserLoss')
    plt.xlabel('Trial')
    plt.xlim(0,125)
    plt.title('REINFORCE with laser, laserdecay and stay')
    plt.legend().remove()




    reinforce_w_laserdecay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_laserdecay/output/summary.csv')
    reinforce_w_laserdecay_recovery = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/recovery_models_laser_decay/standard_recovery/output/summary.csv')
    recovery_params(reinforce_w_laserdecay, reinforce_w_laserdecay_recovery)


    fig,ax=plt.subplots(2,2)
    plt.sca(ax[0,0])
    time_to_laserbaseline(q_learning_model_laserdecay,standata,params_standardlaserdecay, variable='betaqlaser')
    plt.legend().remove()
    plt.title('Q-learning with laser, laserdecay and stay')
    plt.sca(ax[0,1])
    time_to_laserbaseline(q_learning_model_laserdecay,standata,params_standardlaserdecay, variable='betaqlaser')
    plt.legend().remove()
    plt.xlim(0,100)
    plt.title('Q-learning with laser, laserdecay and stay  (1st 100)')
    plt.sca(ax[1,0])
    time_to_laserbaseline(reinforce_model_laserdecay_mixed_perseveration,standata,params_reinforcelaserdecaystay, variable='Qlaser')
    plt.legend().remove()
    plt.title('REINFORCE with laser, laserdecay and stay')
    plt.sca(ax[1,1])
    time_to_laserbaseline(reinforce_model_laserdecay_mixed_perseveration,standata,params_reinforcelaserdecaystay, variable='Qlaser')
    plt.legend().remove()
    plt.title('REINFORCE with laser, laserdecay and stay (1st 100)')
    plt.xlim(0,100)


    # Winners without decay
    reinforce_params = chains_2_summary('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_mixedperseveration/output')
    reinforce_modelled_data = reinforce_model_mixed_perseveration(standata, reinforce_params)
    model_standard = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output/summary.csv')
    standard_modelled_data = q_learning_model(standata,saved_params=model_standard)['acc'].unique()
    fig, ax = plt.subplots(1,2, sharey=True)
    plt.sca(ax[0])
    plot_params(model_standard,standata,phi_a=False)
    plt.sca(ax[1])
    plot_params(reinforce_params,standata)
    plt.tight_layout()
    # Compare q values from different models
    qlearning_values =  load_qdata_from_file(ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects', prefix='standard_')
    reinforce_values =  load_qdata_from_file(ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects', prefix='REINFORCE_mixedstay_')

    model_labels = ['QLearning', 'REINFORCE']
    all_values_corr_stay = np.corrcoef([qlearning_values['DQstay'].to_numpy(),reinforce_values['DQstay'].to_numpy()])
    all_values_corr_water = np.corrcoef([qlearning_values['DQwater'].to_numpy(),reinforce_values['DQwater'].to_numpy()])
    all_values_corr_laser = np.corrcoef([qlearning_values['DQlaser'].to_numpy(),reinforce_values['DQlaser'].to_numpy()])
    all_values_corr_delta = np.corrcoef([qlearning_values['choice_prediction'].to_numpy(),reinforce_values['choice_prediction'].to_numpy()])
 

    fig, ax = plt.subplots(1,4, sharey=True)
    plt.sca(ax[0])
    plot_corr_matrix(all_values_corr_delta, model_labels,vmin=0,vmax=1)
    plt.title('Delta')
    plt.sca(ax[1])
    plot_corr_matrix(all_values_corr_water, model_labels,vmin=0,vmax=1)
    plt.title('DeltaWater')
    plt.sca(ax[2])
    plot_corr_matrix(all_values_corr_laser, model_labels,vmin=0,vmax=1)
    plt.title('DeltaLaser')
    plt.sca(ax[3])
    plot_corr_matrix(all_values_corr_stay, model_labels,vmin=0,vmax=1)
    plt.title('DeltaStay')
    plt.tight_layout()


    # Winners plots
    _, _, _, sim_standard = simulate_q_learning_model_new(standata_recovery,saved_params=model_standard)
    _, _, _, sim_REINFORCE = simulate_reinforce_winloss_stay(standata_recovery,saved_params=reinforce_params)
    original1 = stan_data_to_df(standata_recovery,standata)
    original1['tb'] = original1['tb']-1
    sim_standard.loc[sim_standard['tb']==0,'tb']=1 # 0 tb trials are just the first trials of every session
    sim_REINFORCE.loc[sim_REINFORCE['tb']==0,'tb']=1 # 0 tb trials are just the first trials of every session
    sim_standard['tb'] = sim_standard['tb']-1
    sim_REINFORCE['tb'] = sim_REINFORCE['tb']-1

    fig, ax = plt.subplots(1,3)
    plt.sca(ax[0])
    check_model_behavior(original1)
    plt.title('Real data')
    plt.sca(ax[1]) 
    check_model_behavior(sim_standard)
    plt.title('Q-learning with laser and stay')
    plt.sca(ax[2])      
    check_model_behavior(sim_REINFORCE)
    plt.title('REINFORCE with laser and stay')
