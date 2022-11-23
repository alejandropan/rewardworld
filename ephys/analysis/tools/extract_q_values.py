import pandas as pd
import numpy as np
import model_comparison_accu as mc

def save_q_values(ROOT_FOLDER,psy,data,prefix,trial_start=0,trial_end=-150, no_reward_block=True):
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
                assert len(choice[trial_start:trial_end]) == len(ses_data)
                assert np.array_equal(choice[trial_start:trial_end],ses_data['choices'].to_numpy(),equal_nan=True)==True

                ses_data.loc[ses_data['choices']]
                if trial_end!=None:
                    if trial_end<0:
                        filling_length_end = abs(trial_end)
                    else:
                        filling_length_end = len(choice)-trial_end
                else:
                    filling_length_end = None

                assert (np.isnan(np.nanmean(ses_data['QL'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['QR'].to_numpy()))==False) 
                assert (np.isnan(np.nanmean(ses_data['QRstay'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['QLstay'].to_numpy()))==False) 
                assert (np.isnan(np.nanmean(ses_data['QRreward'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['QLreward'].to_numpy()))==False)
                if no_reward_block==False:
                    assert (np.isnan(np.nanmean(ses_data['QRlaser'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['QLlaser'].to_numpy()))==False) 
                    assert (np.isnan(np.nanmean(ses_data['Qwater'].to_numpy()))==False) & (np.isnan(np.nanmean(ses_data['Qlaser'].to_numpy()))==False) 


                if filling_length_end!=None:
                    filling_length_start = trial_start
                    end_filler=np.empty(filling_length_end)
                    end_filler[:] = np.nan
                    start_filler=np.empty(filling_length_start)
                    start_filler[:] = np.nan
                    np.save(alf+'/'+prefix+'_choice_prediction.npy', np.concatenate([start_filler,ses_data['predicted_choice'].to_numpy(),end_filler]))
                    if no_reward_block==False:
                        np.save(alf+'/'+prefix+'_water.npy', np.concatenate([start_filler,ses_data['Qwater'].to_numpy(),end_filler]))
                        np.save(alf+'/'+prefix+'_laser.npy', np.concatenate([start_filler,ses_data['Qlaser'].to_numpy(),end_filler]))
                        np.save(alf+'/'+prefix+'_QLlaser.npy', np.concatenate([start_filler,ses_data['QLlaser'].to_numpy(),end_filler]))
                        np.save(alf+'/'+prefix+'_QRlaser.npy', np.concatenate([start_filler,ses_data['QRlaser'].to_numpy(),end_filler]))
                    np.save(alf+'/'+prefix+'_QLstay.npy', np.concatenate([start_filler,ses_data['QLstay'].to_numpy(),end_filler]))
                    np.save(alf+'/'+prefix+'_QRstay.npy', np.concatenate([start_filler,ses_data['QRstay'].to_numpy(),end_filler]))
                    np.save(alf+'/'+prefix+'_QLreward.npy', np.concatenate([start_filler,ses_data['QLreward'].to_numpy(),end_filler]))
                    np.save(alf+'/'+prefix+'_QRreward.npy', np.concatenate([start_filler,ses_data['QRreward'].to_numpy(),end_filler]))
                    np.save(alf+'/'+prefix+'_QL.npy', np.concatenate([start_filler,ses_data['QL'].to_numpy(),end_filler]))
                    np.save(alf+'/'+prefix+'_QR.npy', np.concatenate([start_filler,ses_data['QR'].to_numpy(),end_filler]))
                else:
                    np.save(alf+'/'+prefix+'_choice_prediction.npy', ses_data['predicted_choice'].to_numpy())
                    if no_reward_block==False:
                        np.save(alf+'/'+prefix+'_water.npy', ses_data['Qwater'].to_numpy())
                        np.save(alf+'/'+prefix+'_laser.npy', ses_data['Qlaser'].to_numpy())
                        np.save(alf+'/'+prefix+'_QLlaser.npy', ses_data['QLlaser'].to_numpy())
                        np.save(alf+'/'+prefix+'_QRlaser.npy', ses_data['QRlaser'].to_numpy())
                    np.save(alf+'/'+prefix+'_QLstay.npy', ses_data['QLstay'].to_numpy())
                    np.save(alf+'/'+prefix+'_QRstay.npy', ses_data['QRstay'].to_numpy())
                    np.save(alf+'/'+prefix+'_QLreward.npy', ses_data['QLreward'].to_numpy())
                    np.save(alf+'/'+prefix+'_QRreward.npy', ses_data['QRreward'].to_numpy())
                    np.save(alf+'/'+prefix+'_QL.npy', ses_data['QL'].to_numpy())
                    np.save(alf+'/'+prefix+'_QR.npy', ses_data['QR'].to_numpy())                    

ROOT_FOLDER = '/Volumes/witten/Alex/Data/Subjects'
psy=mc.load_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_waterlaser', trial_start=0, trial_end=None)
standata = mc.make_stan_data_reduced(psy)
qlearning_params = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/dual_task_fits/standard_reduced_stay/output/summary.csv')
qlearning_data = pd.DataFrame()
qlearning_data = mc.q_learning_model_reduced_stay(standata,saved_params=qlearning_params)
assert len(psy) == len(qlearning_data)
prefix='standard'
save_q_values(ROOT_FOLDER,psy,qlearning_data,prefix,trial_start=0,trial_end=None)