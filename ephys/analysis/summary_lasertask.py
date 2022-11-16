from model_comparison_accu import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



14
# First load the data
data=load_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_laser_only', trial_start=0, trial_end=-1)
standata = make_stan_data_reduced(data)
standata_recovery=load_sim_data_reduced(ROOT_FOLDER = '/Volumes/witten/Alex/Data/ephys_bandit/data_laser_only', trial_start=0, trial_end=-1)
num_to_name(data)

## 1. Standard
qlearning_params = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/standard_reduced_stay/output/summary.csv')
qlearning = pd.DataFrame()
qlearning['Accuracy'] =  q_learning_model_reduced_stay(standata,saved_params=qlearning_params)['acc'].unique()
qlearning['Model'] = 'qlearning'

# 2. Forgetting
model_forgetting = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/standard_w_forgetting/output/summary.csv')
qlearning_w_forgetting = pd.DataFrame()
qlearning_w_forgetting['Accuracy'] = q_learning_model_reduced_stay_forgetting(standata,saved_params=model_forgetting)['acc'].unique()
qlearning_w_forgetting['Model'] = 'F-Q'

# 2. REINFORCE
model_REINFORCE = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/REINFORCE_reduced/output/summary.csv')
reinforce = pd.DataFrame()
reinforce['Accuracy'] = reinforce_model_reduced(standata,saved_params=model_REINFORCE)['acc'].unique()
reinforce['Model'] = 'REINFORCE'

# 2. REINFORCE w stay
model_REINFORCE_stay = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/laser_stan_fits/REINFORCE_mixedperseveration/output/summary.csv')
reinforce_stay = pd.DataFrame()
reinforce_stay['Accuracy'] = reinforce_model_reduced_stay(standata,saved_params=model_REINFORCE_stay)['acc'].unique()
reinforce_stay['Model'] = 'REINFORCE_w_stay' 