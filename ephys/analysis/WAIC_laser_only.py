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
        qlaseronly=np.mean(np.log(np.load('qlaseronly.npy')))
        qlaserforgetting=np.mean(np.log(np.load('qlaserforgetting.npy')))
        reinforcelaseronly=np.mean(np.log(np.load('reinforcelaseronly.npy')))
        reinforcelaseronly_w_stay=np.mean(np.log(np.load('reinforcelaseronly_w_stay.npy')))
        uchidalaseronly=np.mean(np.log(np.load('uchidalaseronly.npy')))
    except:
        # 1
        standata = mc.make_stan_data_reduced(mc.load_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_laser_only', trial_start=0, trial_end=None))
        path = '/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/standard_reduced_stay'
        model = mc.q_learning_model_reduced_stay
        WAIC_qlaseronly = lppd_from_chain(path, model, standata, save=True, save_name='qlaseronly')
        # 2
        path = '/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/REINFORCE_reduced'
        model = mc.reinforce_model_reduced
        WAIC_reinforcelaseronly = lppd_from_chain(path, model, standata, save=True, save_name='reinforcelaseronly')
        # 3
        path = '/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/REINFORCE_mixedperseveration'
        model = mc.reinforce_model_reduced_stay
        WAIC_reinforcelaseronly_w_stay = lppd_from_chain(path, model, standata, save=True, save_name='reinforcelaseronly_w_stay')
        # 4
        path = '/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/standard_uchida'
        model = mc.reduced_uchida_model
        WAIC_uchidalaseronly = lppd_from_chain(path, model, standata, save=True, save_name='uchidalaseronly')
        # 5
        path = '/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/standard_w_forgetting'
        model = mc.q_learning_model_reduced_stay_forgetting
        WAIC_qlaserforgetting = lppd_from_chain(path, model, standata, save=True, save_name='qlaserforgetting')

        qlaseronly=np.mean(np.log(np.load('qlaseronly.npy')))
        reinforcelaseronly=np.mean(np.log(np.load('reinforcelaseronly.npy')))
        reinforcelaseronly_w_stay=np.mean(np.log(np.load('reinforcelaseronly_w_stay.npy')))
        uchidalaseronly=np.mean(np.log(np.load('uchidalaseronly.npy')))
        qlaserforgetting=np.mean(np.log(np.load('qlaserforgetting.npy')))

    # no_loss models were worse so not including them in the model
    waics=pd.DataFrame()
    waics['WAIC'] = [WAIC_qlaseronly,
                    WAIC_qlaserforgetting,
                    WAIC_reinforcelaseronly,
                    WAIC_reinforcelaseronly_w_stay,
                    WAIC_uchidalaseronly]
    waics['LL'] = [qlaseronly,
                    qlaserforgetting,
                   reinforcelaseronly,
                   reinforcelaseronly_w_stay,
                   uchidalaseronly]

    waics['name'] = ['Q-Learning',
                    'Q-Learning with forgetting',
                     'REINFORCE',
                     'REINFORCE-w-stay',
                     'Q-Two-Accumulators'] 

    fig, ax = plt.subplots(1,2)
    plt.sca(ax[0])
    sns.pointplot(data=waics.sort_values(by=['WAIC']), x='name',y='WAIC', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.sca(ax[1])
    sns.pointplot(data=waics.sort_values(by=['LL'],ascending=[False]), x='name',y='LL', color='k')
    plt.xticks(rotation=90)
    plt.tight_layout()
    sns.despine()
    plt.tight_layout()
    ax[0].set_xlabel('Model')
    ax[0].set_ylabel('WAIC', color='k')
    ax[1].set_xlabel('Model')
    ax[1].set_ylabel('LL', color='k')
