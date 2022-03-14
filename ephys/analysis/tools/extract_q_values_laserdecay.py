import pandas as pd
import numpy as np
import model_comparison_accu as mc

def save_q_values(ROOT_FOLDER,psy,data,prefix):
    for ns, mouse in enumerate(psy['mouse'].unique()):
        animal = psy.loc[psy['mouse']==mouse]
        counter=0
        for day in animal['date'].unique():
            day_s = animal.loc[animal['date']==day]
            for ses in day_s['ses'].unique():
                session = day_s.loc[day_s['ses']==ses]
                alf = ROOT_FOLDER+'/'+mouse+'/'+day+'/'+ses+'/alf'
                choice = np.load(alf+'/_ibl_trials.choice.npy')
                choice=1*(choice==-1)
                ses_data = data.loc[(data['mouse']==ns) &
                                    (data['ses']==counter)]
                counter+=1
                assert len(ses_data) == len(session)
                assert len(choice[:-150]) == len(ses_data)
                assert np.array_equal(choice[:-150],ses_data['choices'].to_numpy(),equal_nan=True)==True

                ses_data.loc[ses_data['choices']]
                filler=np.empty(150)
                filler[:] = np.nan
                assert (np.isnan(np.nanmean(ses_data['QL'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['QR'].to_numpy()))==False) 
                assert (np.isnan(np.nanmean(ses_data['Qwater'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['Qlaser'].to_numpy()))==False) 
                assert (np.isnan(np.nanmean(ses_data['QRstay'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['QLstay'].to_numpy()))==False) 
                assert (np.isnan(np.nanmean(ses_data['QRreward'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['QLreward'].to_numpy()))==False) 
                assert (np.isnan(np.nanmean(ses_data['QRlaser'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['QLlaser'].to_numpy()))==False) 
                np.save(alf+'/'+prefix+'_choice_prediction.npy', np.concatenate([ses_data['predicted_choice'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_water.npy', np.concatenate([ses_data['Qwater'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_laser.npy', np.concatenate([ses_data['Qlaser'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_QLstay.npy', np.concatenate([ses_data['QLstay'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_QRstay.npy', np.concatenate([ses_data['QRstay'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_QLreward.npy', np.concatenate([ses_data['QLreward'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_QRreward.npy', np.concatenate([ses_data['QRreward'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_QLlaser.npy', np.concatenate([ses_data['QLlaser'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_QRlaser.npy', np.concatenate([ses_data['QRlaser'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_QL.npy', np.concatenate([ses_data['QL'].to_numpy(),filler]))
                np.save(alf+'/'+prefix+'_QR.npy', np.concatenate([ses_data['QR'].to_numpy(),filler]))
                print(alf)

ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects'
psy=mc.load_data()
standata = mc.make_stan_data(psy)
qlearning_params = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/standard_alphalaserdecay/output/summary.csv')
qlearning_data = pd.DataFrame()
qlearning_data = mc.q_learning_model_alphalaserdecay(standata,saved_params=qlearning_params)
assert len(psy) == len(qlearning_data)
prefix='QLearning_alphalaserdecay'
save_q_values(ROOT_FOLDER,psy,qlearning_data,prefix)