from model_comparison_accu import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# First load the data
data=load_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_waterlaser', trial_start=0, trial_end=-1)
standata = make_stan_data(data)
standata_recovery=load_sim_data(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_waterlaser', trial_start=0, trial_end=-1)
mice = num_to_name(data)

## 1. Standard
qlearning_params = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/dual_task_fits/standard_n_phi/output/summary.csv')
qlearning = pd.DataFrame()
qlearning['Accuracy'] =  q_learning_model(standata,saved_params=qlearning_params)['acc'].unique()
qlearning['Model'] = 'qlearning'

# 2. Forgetting
model_forgetting = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/dual_task_fits/standard_w_forgetting/output/summary.csv')
qlearning_w_forgetting = pd.DataFrame()
qlearning_w_forgetting['Accuracy'] = q_learning_model_w_forgetting(standata,saved_params=model_forgetting)['acc'].unique()
qlearning_w_forgetting['Model'] = 'F-Q'

# 2. REINFORCE
model_REINFORCE = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/dual_task_fits/REINFORCE/output/summary.csv')
reinforce = pd.DataFrame()
reinforce['Accuracy'] = reinforce_model(standata,saved_params=model_REINFORCE)['acc'].unique()
reinforce['Model'] = 'REINFORCE'

# 2. REINFORCE w stay
model_REINFORCE_stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/dual_task_fits/REINFORCE_mixedperseveration/output/summary.csv')
reinforce_stay = pd.DataFrame()
reinforce_stay['Accuracy'] = reinforce_model_mixed_perseveration(standata,saved_params=model_REINFORCE_stay)['acc'].unique()
reinforce_stay['Model'] = 'REINFORCE_w_stay' 