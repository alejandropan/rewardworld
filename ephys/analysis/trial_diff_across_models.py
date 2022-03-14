import numpy as np
import pandas as pd
import seaborn as sns
from model_comparison_accu import *
# Load the Q values for each model

Index(['predicted_choice', 'QL', 'QR', 'QLreward', 'QRreward', 'QLstay',
       'QRstay', 'QLlaser', 'QRlaser', 'deltaQ', 'choices', 'mouse', 'ses',
       'acc', 'binary_pred', 'choices1', 'binary_pred_argmax', 'Qlaser'],
      dtype='object')

def make_deltas(model_data):
    model_data['deltaQ'] = model_data['QR']-model_data['QL']
    model_data['Qwater'] = model_data['QRreward']-model_data['QLreward']
    model_data['Qstay'] = model_data['QRstay']-model_data['QLstay']
    model_data['Qlaser'] = model_data['QRlaser']-model_data['QLlaser']
    return model_data

def trial_diff(standata, corr_var, model_standard,model_forgetting,
                        reinforce,LRPE,LRPEc,a2stay,LRPEcf,stdlapse):
    '''
    Standata: is data used to fit the models in stan format.
    corr_var: is the variable to correlate(e.g. deltaQ) (str)
    the rest are the parameters for every model used
    '''

    standard = q_learning_model(standata,saved_params=model_standard)
    fq = q_learning_model_w_forgetting(standata,saved_params=model_forgetting)
    rnfrc = reinforce_model(standata,saved_params=reinforce)
    l = q_learning_model_Lrpe(standata,saved_params=LRPE)
    lr = q_learning_model_Lrpeconditional(standata,saved_params=LRPEc)
    twoq = q_learning_model_2alphas(standata,saved_params=a2stay)
    lrf = q_learning_model_Lrpeconditional_w_forgetting(standata,saved_params=LRPEcf)
    lapse = q_learning_model_lapse (standata, saved_params=stdlapse)
    rnfrc_2stay = reinforce_model_w_2stay(standata, saved_params=reinforce_2stay)
    rnfrc_stay = reinforce_model_w_stay(standata, saved_params=reinforce_w_stay)

    standard['binary_pred'] = 1*(standard['predicted_choice']>0.5)
    fq['binary_pred'] = 1*(fq['predicted_choice']>0.5)
    rnfrc['binary_pred'] = 1*(rnfrc['predicted_choice']>0.5)
    rnfrc_2stay['binary_pred'] = 1*(rnfrc_2stay['predicted_choice']>0.5)
    rnfrc_stay['binary_pred'] = 1*(rnfrc_stay['predicted_choice']>0.5)
    l['binary_pred'] = 1*(l['predicted_choice']>0.5)
    lr['binary_pred'] = 1*(lr['predicted_choice']>0.5)
    twoq['binary_pred'] = 1*(twoq['predicted_choice']>0.5)
    lrf['binary_pred'] = 1*(lrf['predicted_choice']>0.5)
    lapse['binary_pred'] = 1*(lapse['predicted_choice']>0.5)


    rnfrc['binary_pred_argmax'] = 1*(rnfrc['deltaQ']>0)
    standard['binary_pred_argmax'] = 1*(standard['deltaQ']>0)
    rnfrc_stay['binary_pred_argmax'] = 1*(rnfrc_stay['deltaQ']>0)
    rnfrc_2stay['binary_pred_argmax'] = 1*(rnfrc_2stay['deltaQ']>0)


    comp = np.zeros([10, len(standard)])
    comp[0,:] = abs(standard['choices'].to_numpy() - standard['binary_pred'].to_numpy())
    comp[1,:] = abs(standard['choices'].to_numpy() - rnfrc['binary_pred'].to_numpy())
    comp[2,:] = abs(standard['choices'].to_numpy() - rnfrc_stay['binary_pred'].to_numpy())
    comp[3,:] = abs(standard['choices'].to_numpy() - rnfrc_2stay['binary_pred'].to_numpy())
    comp[4,:] = abs(standard['choices'].to_numpy() - fq['binary_pred'].to_numpy())
    comp[5,:] = abs(standard['choices'].to_numpy() - l['binary_pred'].to_numpy())
    comp[6,:] = abs(standard['choices'].to_numpy() - lr['binary_pred'].to_numpy())
    comp[7,:] = abs(standard['choices'].to_numpy() - twoq['binary_pred'].to_numpy())
    comp[8,:] = abs(standard['choices'].to_numpy() - lrf['binary_pred'].to_numpy())
    comp[9,:] = abs(standard['choices'].to_numpy() - lapse['binary_pred'].to_numpy())


    models = ['standard','REIN','REIN+stay','REIN+2stay','FQ','L=RPE',
             'L=/RPE','2a', 'L=/RPE+FQ', 'Lapse']

    #fig,ax = plt.subplots(6,6, sharey=True)
    rnfrc_2stay = rnfrc_2stay.reset_index().reset_index()
    for i in rnfrc_2stay.mouse.unique():
        sel1 = rnfrc_2stay.loc[rnfrc_2stay['mouse']==i]
        for j in sel1.ses.unique():
            sel2 = sel1.loc[sel1['ses']==j]
            idx = sel2['level_0'].to_numpy()
            laserb = np.where(sel2['laser_block']==1)
            waterb = np.where(sel2['laser_block']==0)
            #plt.sca(ax[i,j])
            sns.heatmap(comp[:,idx], cmap= 'Set1', cbar=False)
            plt.title('Mouse: '+str(i)+' Session: '+str(j))
            plt.yticks(np.arange(10)+0.5,models, rotation=0)
            plt.vlines(laserb, 0,.25, color='orange', linewidth=3)
            plt.vlines(waterb, 0,.25, color='dodgerblue', linewidth=3)
            plt.vlines(np.where(sel2['trial_block'].diff()<0),0,.25,linewidth=3, color='k')
            plt.tight_layout()
            plt.xlabel('Trial number')
            plt.show()

    models1 = [' ','REIN','REIN+stay','REIN+2stay']
    comp1r = np.zeros([4, len(standard)])
    comp1r[0,:] = abs(standard['binary_pred'].to_numpy() - standard['binary_pred'].to_numpy())
    comp1r[1,:] = abs(standard['binary_pred'].to_numpy() - rnfrc['binary_pred'].to_numpy())
    comp1r[2,:] = abs(standard['binary_pred'].to_numpy() - rnfrc_stay['binary_pred'].to_numpy())
    comp1r[3,:] = abs(standard['binary_pred'].to_numpy() - rnfrc_2stay['binary_pred'].to_numpy())

    for i in rnfrc_2stay.mouse.unique():
        sel1 = rnfrc_2stay.loc[rnfrc_2stay['mouse']==i]
        for j in sel1.ses.unique():
            sel2 = sel1.loc[sel1['ses']==j]
            idx = sel2['level_0'].to_numpy()
            laserb = np.where(sel2['laser_block']==1)
            waterb = np.where(sel2['laser_block']==0)
            #plt.sca(ax[i,j])
            sns.heatmap(comp1r[:,idx], cmap= 'Dark2', cbar=False)
            plt.title('Mouse: '+str(i)+' Session: '+str(j))
            plt.yticks(np.arange(4)+0.5,models1, rotation=0)
            plt.vlines(laserb, 0,1, color='orange', linewidth=3)
            plt.vlines(waterb, 0,1, color='dodgerblue', linewidth=3)
            plt.vlines(np.where(sel2['trial_block'].diff()<0),0,1,linewidth=3, color='k')
            plt.tight_layout()
            plt.xlabel('Trial number')
            plt.show()

#plt.plot(rnfrc_stay.loc[(rnfrc_stay['mouse']==i)&(rnfrc_stay['ses']==j), 'choices'].to_numpy(), color='k')
standard['choices1'] = standard['choices'].copy()
standard.loc[standard['choices']==0,'choices1']=-1
sel2 = rnfrc_2stay.loc[(rnfrc_2stay['mouse']==i)&(rnfrc_2stay['ses']==j)]
laserb = np.where(sel2['laser_block'].to_numpy()==1)
waterb = np.where(sel2['laser_block'].to_numpy()==0)
plt.vlines(laserb, 3.25,3.5, color='orange', linewidth=3)
plt.vlines(waterb, 3.25,3.5, color='dodgerblue', linewidth=3)
plt.vlines(np.where(sel2['trial_block'].diff()<0),3.25,3.5,linewidth=3, color='k')
plt.plot(standard.loc[(standard['mouse']==i)&(standard['ses']==j), 'choices1'].rolling(5, center=True).mean().to_numpy(), color='gray')
plt.plot(standard.loc[(standard['mouse']==i)&(standard['ses']==j), 'QStay'].shift(1).to_numpy(), color='k')
plt.plot(rnfrc_2stay.loc[(rnfrc_2stay['mouse']==i)&(rnfrc_2stay['ses']==j), 'QStay'].shift(1).to_numpy(), color='dodgerblue')
plt.plot(rnfrc.loc[(rnfrc['mouse']==i)&(rnfrc['ses']==j), 'QStay'].shift(1).to_numpy(), color='red')
plt.plot(rnfrc_stay.loc[(rnfrc_stay['mouse']==i)&(rnfrc_stay['ses']==j), 'QStay'].shift(1).to_numpy(), color='blue')
plt.title('Mouse: '+str(i)+' Session: '+str(j))
plt.xlabel('Trial number')
plt.ylabel('DeltaQ/Choice')
sns.despine()


#plt.plot(rnfrc_stay.loc[(rnfrc_stay['mouse']==i)&(rnfrc_stay['ses']==j), 'choices'].to_numpy(), color='k')
rnfrc_stay['softmax_input'] = rnfrc_stay['deltaQ'] + rnfrc_stay['bias']
fig, ax1 = plt.subplots()
ax1.plot(rnfrc_stay.loc[(rnfrc_stay['mouse']==i)&(rnfrc_stay['ses']==j), 'deltaQ'].to_numpy(), color='blue')
ax1.plot(rnfrc_stay.loc[(rnfrc_stay['mouse']==i)&(rnfrc_stay['ses']==j), 'softmax_input'].to_numpy(), color='green')
ax2 = ax1.twinx()
ax2.plot(rnfrc_stay.loc[(rnfrc_stay['mouse']==i)&(rnfrc_stay['ses']==j), 'predicted_choice'].to_numpy(), color='k')
plt.title('Mouse: '+str(i)+' Session: '+str(j))
plt.xlabel('Trial number')
ax1.set_ylabel('softmax_input/deltaQ')
ax2.set_ylabel('softmax_output')
sns.despine()





if __name__ == '__main__':
    ## 0. Load data
    standata = make_stan_data(load_data())
    ## 1. Standard
    model_standard = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_n_phi/output/summary.csv')
    ## 2. Standard with forgetting
    model_forgetting = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_w_forgetting/output/summary.csv')
    ## 3. REINFORCE
    reinforce = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE/output/summary.csv')
    ## 4. REINFORCE w 2 stays
    reinforce_2stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_w_2stay/output/summary.csv')
    ## 5. Laser==RPE (Pending)
    LRPE = chains_2_summary('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/LRPE/old_output/output')
    # 6. Laser==RPE conditional
    ## RPEc = pd.read_csv('')
    LRPEc = chains_2_summary('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/LRPEconditional/output')
    ## 7. 2 stays
    a2stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_w_2stays/output/summary.csv')
    ## 8. Laser==RPE conditional with forgetting
    LRPEcf = chains_2_summary('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/LRPE_condtiional_w_forgetting')
    # 9. Lapse standard
    stdlapse = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_lapse/output/summary.csv')
    # 10. REINFORCE w Stay
    reinforce_w_stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/REINFORCE_w_stay/output/summary.csv')
    # 11. Standard w 2learning
    q_learning_2learning = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_w_2alphas/output/summary.csv')


    # Make make_deltas
    standard = make_deltas(standard)
    fq =make_deltas(fq)
    l = make_deltas(l)
    lr = make_deltas(lr)
    twoq = make_deltas(twoq)
    lrf = make_deltas(lrf)
    lapse = make_deltas(lapse)
