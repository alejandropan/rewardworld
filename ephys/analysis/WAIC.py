import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import glob
import model_comparison_accu as mc
import multiprocess as mp
import time

# Functions
def load_cmd_stan_output(chains):
    output=pd.DataFrame()
    for c,i in enumerate(chains):
        chain = pd.read_csv(i
            , skiprows= \
            np.concatenate([np.arange(42), np.array([44,45,46,43]),
            np.arange(1547,1552)]))
        chain['chain'] = c+1
        output = pd.concat([output,chain])
    return output

def lppd_from_chain(path, model, standata, save=False, save_name=None):
    p_chain1 =  list(glob.glob(path+'/output/*1.csv'))[-1] #Latest 1 chain
    p_chain2 =  list(glob.glob(path+'/output/*2.csv'))[-1] #Latest 1 chain
    p_chain3 =  list(glob.glob(path+'/output/*3.csv'))[-1] #Latest 1 chain
    p_chain4 =  list(glob.glob(path+'/output/*4.csv'))[-1] #Latest 1 chain
    chains = [p_chain1,p_chain2,p_chain3,p_chain4]
    output = load_cmd_stan_output(chains)
    model = model  #initialize model so that multoprocess sims works
    def multiprocess_sims(i, output=output, model=model, chain_len=1500):
        params = output.iloc[i,:].copy().reset_index()
        params.rename(columns={'index':'name', i%chain_len:'Mean'}, inplace=True)
        params.loc[(params['name'].str.contains('\d')),'name'] = params.loc[(params['name'].str.contains('\d')),'name']+']'
        params['name']=params['name'].str.replace('.','[')
        sim = model(standata, saved_params=params)
        sim.loc[sim['choices']==0,'predicted_choice'] = 1 - sim.loc[sim['choices']==0,'predicted_choice']
        return sim['predicted_choice'].to_numpy()
    posterior = np.zeros([len(output),standata['NT_all'].sum()])
    pool = mp.Pool()
    start = time.time()
    for ind, res in enumerate(pool.imap(multiprocess_sims, range(len(output)),chunksize=100)):
        posterior[ind,:] = res
    print(time.time()-start)
    LPPD = np.sum(np.log(posterior.mean(axis=0)))
    P = np.sum(np.var(np.log(posterior), axis=0))
    WAIC = -2*(LPPD-P)
    if save==True:
        np.save(save_name+'.npy', posterior)
    return WAIC


if __name__=='__main__':
    try:
        reducedstay=np.load('qreducedstay.npy')
        reduced=np.load('qreduced.npy')
        full=np.load('qfull.npy')
        rreduced=np.load('rreduced.npy')
        rfull=np.load('rfull.npy')
        rwstay=np.load('rwstay.npy')
        laserdecay=np.load('laserdecay.npy')
        waterlaserdecay=np.load('waterlaserdecay.npy')
        mixedpersever=np.load('rmixed_perseveration.npy')
        rmixed_perseveration_noloss = np.load('rmixed_perseveration_noloss.npy')
        standardlaser_nostay = np.load('standardlaser_nostay.npy')
        REINFORCE_nolaser_mixedperseveration = np.load('REINFORCE_nolaser_mixedperseveration.npy')
        REINFORCE_laserdecaywinloss_mixedstay = np.load('REINFORCE_laserdecaywinloss_mixedstay.npy')
        REINFORCE_laserdecaywinloss_mixedstay_noloss = np.load('REINFORCE_laserdecaywinloss_mixedstay_noloss.npy')
        standard_laserdecay =  np.load('standard_laserdecay.npy')
        WAIC_reducedstay = -2*(np.sum(np.log(reducedstay.mean(axis=0))) - np.sum(np.var(np.log(reducedstay), axis=0)))
        WAIC_reduced = -2*(np.sum(np.log(reduced.mean(axis=0))) - np.sum(np.var(np.log(reduced), axis=0)))
        WAIC_full = -2*(np.sum(np.log(full.mean(axis=0))) - np.sum(np.var(np.log(full), axis=0)))
        WAIC_rreduced = -2*(np.sum(np.log(rreduced.mean(axis=0))) - np.sum(np.var(np.log(rreduced), axis=0)))
        WAIC_rfull = -2*(np.sum(np.log(rfull.mean(axis=0))) - np.sum(np.var(np.log(rfull), axis=0)))
        WAIC_mixedperseveration =-2*(np.sum(np.log(mixedpersever.mean(axis=0))) - np.sum(np.var(np.log(mixedpersever), axis=0)))
        WAIC_mixedperseveration_noloss = -2*(np.sum(np.log(rmixed_perseveration_noloss.mean(axis=0))) - np.sum(np.var(np.log(rmixed_perseveration_noloss), axis=0)))
        WAIC_full_nostay = -2*(np.sum(np.log(standardlaser_nostay.mean(axis=0))) - np.sum(np.var(np.log(standardlaser_nostay), axis=0)))
        WAIC_nolaser_mixedperseveration =  -2*(np.sum(np.log(REINFORCE_nolaser_mixedperseveration.mean(axis=0))) - np.sum(np.var(np.log(REINFORCE_nolaser_mixedperseveration), axis=0)))
        WAIC_laserdecay_mixedperseveration=  -2*(np.sum(np.log(REINFORCE_laserdecaywinloss_mixedstay.mean(axis=0))) - np.sum(np.var(np.log(REINFORCE_laserdecaywinloss_mixedstay), axis=0)))
        WAIC_laserdecay_mixedperseveration_noloss=  -2*(np.sum(np.log(REINFORCE_laserdecaywinloss_mixedstay_noloss.mean(axis=0))) - np.sum(np.var(np.log(REINFORCE_laserdecaywinloss_mixedstay_noloss), axis=0)))
        WAIC_standard_laserdecay=  -2*(np.sum(np.log(standard_laserdecay.mean(axis=0))) - np.sum(np.var(np.log(standard_laserdecay), axis=0)))
    except:
        # 1
        standata = mc.make_stan_data(mc.load_data(trial_end=-1))
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_reduced_stay'
        model = mc.q_learning_model_reduced_stay
        WAIC_reducedstay = lppd_from_chain(path, model, standata, save=True, save_name='qreducedstay')
        # 2
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_reduced'
        model = mc.q_learning_model_reduced
        WAIC_reduced = lppd_from_chain(path, model, standata, save=True, save_name='qreduced')
        # 3
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi'
        model = mc.q_learning_model
        WAIC_full = lppd_from_chain(path, model, standata, save=True, save_name='qfull')
        # 4
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_reduced'
        model = mc.reinforce_model_reduced
        WAIC_rreduced = lppd_from_chain(path, model, standata, save=True, save_name='rreduced')
        # 5
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE'
        model = mc.reinforce_model
        WAIC_rfull = lppd_from_chain(path, model, standata, save=True, save_name='rfull')
        # 6
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywin_mixedstay'
        model = mc.reinforce_model_alphalaserdecay_mixed_perseveration
        WAIC_laserdecay = lppd_from_chain(path, model, standata, save=True, save_name='laserdecay')
        # 7
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_mixedperseveration'
        model = mc.reinforce_model_mixed_perseveration
        WAIC_mixedperseveration = lppd_from_chain(path, model, standata, save=True, save_name='rmixed_perseveration')
        # 8 
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_mixedperseveration_noloss'
        model = mc.reinforce_model_mixed_perseveration_noloss
        WAIC_mixedperseveration_noloss = lppd_from_chain(path, model, standata, save=True, save_name='rmixed_perseveration_noloss')
        # 9 
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_nostay'
        model = mc.q_learning_lasernostay
        WAIC_full_nostay = lppd_from_chain(path, model, standata, save=True, save_name='standardlaser_nostay')
        # 10
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_nolaser_mixedperseveration'
        model = mc.reinforce_model_nolaser_mixed_perseveration
        WAIC_nolaser_mixedperseveration = lppd_from_chain(path, model, standata, save=True, save_name='REINFORCE_nolaser_mixedperseveration')
        # 11
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywin_mixedstay'
        model = mc.reinforce_model_alphalaserdecay_mixed_perseveration
        WAIC_laserdecay_mixedperseveration=lppd_from_chain(path, model, standata, save=True, save_name='REINFORCE_laserdecaywinloss_mixedstay')
        # 12    
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_laserdecaywinloss_mixedstay_noloss'
        model = mc.reinforce_model_laserdecay_mixed_perseveration_noloss
        WAIC_laserdecay_mixedperseveration_noloss=lppd_from_chain(path, model, standata, save=True, save_name='REINFORCE_laserdecaywinloss_mixedstay_noloss')
        # 13
        path = '/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_alphalaserdecay'
        model = mc.q_learning_model_alphalaserdecay
        WAIC_standard_laserdecay=lppd_from_chain(path, model, standata, save=True, save_name='standard_laserdecay')

    reducedstay=np.mean(np.log(np.load('qreducedstay.npy')))
    reduced=np.mean(np.log(np.load('qreduced.npy')))
    full=np.mean(np.log(np.load('qfull.npy')))
    rreduced=np.mean(np.log(np.load('rreduced.npy')))
    rfull=np.mean(np.log(np.load('rfull.npy')))
    rwstay=np.mean(np.log(np.load('rwstay.npy')))
    mixedpersever=np.mean(np.log(np.load('rmixed_perseveration.npy')))
    qlasernostay=np.mean(np.log(np.load('standardlaser_nostay.npy')))
    nolasermixedpersever=np.mean(np.log(np.load('REINFORCE_nolaser_mixedperseveration.npy')))
    mixedpersevernoloss=np.mean(np.log(np.load('rmixed_perseveration_noloss.npy')))
    laserdecay=np.mean(np.log(np.load('laserdecay.npy')))
    rwaterlaserdecay=np.mean(np.log(np.load('waterlaserdecay.npy')))
    laserdecaymixedpersever=np.mean(np.log(np.load('REINFORCE_laserdecaywinloss_mixedstay.npy')))
    laserdecaymixedpersevernoloss=np.mean(np.log(np.load('REINFORCE_laserdecaywinloss_mixedstay_noloss.npy')))
    q_laserdecay=np.mean(np.log(np.load('standard_laserdecay.npy')))

    # no_loss models were worse so not including them in the model
    waics=pd.DataFrame()
    waics['WAIC'] = [WAIC_reducedstay,
                    WAIC_reduced,
                    WAIC_full,
                    WAIC_rreduced,
                    WAIC_rfull, 
                    WAIC_mixedperseveration,
                    WAIC_full_nostay,
                    WAIC_nolaser_mixedperseveration,
                    WAIC_laserdecay_mixedperseveration,
                    WAIC_standard_laserdecay]
    waics['LL'] = [reducedstay,
                   reduced,
                   full,
                   rreduced,
                   rfull, 
                   mixedpersever,
                   qlasernostay,
                   nolasermixedpersever, 
                   laserdecaymixedpersever,
                   q_laserdecay]

    waics['name'] = ['Common + Stay',
                     'Common',
                     'Water + Laser + Stay',
                     'Common',
                     'Water + Laser',
                     'Water + Laser + Stay',
                     'Water + Laser',
                     'Common + Stay',
                     'Water + Stay +  Laser + Laser Decay',
                     'Water + Stay + Laser + Laser Decay'] 


    waics['type'] = ['qlearning',
                     'qlearning',
                     'qlearning', 
                     'reinforce',
                     'reinforce',
                     'reinforce',
                     'qlearning',
                     'reinforce',
                     'reinforce',
                     'qlearning'] 



    fig, ax = plt.subplots(1,2)
    plt.sca(ax[0])
    sns.pointplot(data=waics.loc[waics['type']=='qlearning'].sort_values(by=['WAIC']), x='name',y='WAIC', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1])
    sns.pointplot(data=waics.loc[waics['type']=='qlearning'].sort_values(by=['LL'],ascending=[False]), x='name',y='LL', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.tight_layout()
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('WAIC', color='k')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('LL', color='k')


    fig, ax = plt.subplots(1,2, sharey=True)
    plt.sca(ax[0])
    sns.pointplot(data=waics.loc[waics['type']=='reinforce'].sort_values(by=['WAIC']), x='name',y='WAIC', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1])
    sns.pointplot(data=waics.loc[waics['type']=='reinforce'].sort_values(by=['LL'], ascending=[False]), x='name',y='LL', color='k')
    plt.xticks(rotation=90)
    sns.despine()
    plt.tight_layout()
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('WAIC', color='k')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('LL', color='k')



    fig, ax = plt.subplots(1,2, sharey=True)
    plt.sca(ax[0])
    sns.pointplot(data=waics.loc[waics['type']=='reinforce'].sort_values(by=['WAIC']), x='name',y='WAIC', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.title('REINFORCE')
    plt.sca(ax[1])
    sns.pointplot(data=waics.loc[waics['type']=='qlearning'].sort_values(by=['WAIC']), x='name',y='WAIC', color='k')
    plt.xticks(rotation=90)
    sns.despine()
    plt.title('Q-Learning')
    plt.tight_layout()
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('WAIC', color='k')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('WAIC', color='k')

    waics2 = waics.loc[~np.isin(waics['name'],['Water + Laser + Laser Decay',
                     'Water + Laser + Laser Decay + Water Decay'])]
    fig, ax = plt.subplots(1,2, sharey=True)                   
    plt.sca(ax[0])
    sns.pointplot(data=waics2.loc[waics2['type']=='qlearning'].sort_values(by=['WAIC']), x='name',y='WAIC', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1])
    sns.pointplot(data=waics2.loc[waics2['type']=='reinforce'].sort_values(by=['WAIC']), x='name',y='WAIC', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.tight_layout()
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('WAIC', color='k')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel(' ')

    waics3 = waics2.loc[~np.isin(waics2['name'],['Water + Stay +  Laser + Laser Decay',
                     'Water + Stay + Laser + Laser Decay'])]


    order = ['Common',  'Water + Laser', 'Common + Stay', 'Water + Laser + Stay']
    
    sns.pointplot(data=waics3, order=order, x='name',y='WAIC',hue='type', palette=['k','r'])
    plt.xticks(rotation=90)
    sns.despine()
    plt.tight_layout()
    plt.ylabel('WAIC', color='k')