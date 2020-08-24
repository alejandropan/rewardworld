# Test the fitting procedure by fitting on arbitrarily generated data
# num_trials = 30000
all_contrasts = np.array([-0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25])

# data = generate_data(num_trials, all_contrasts)
# x, NLL = optimizer(data, initial_guess=[0.3, 0.1, 1, 0.2])

# print(x)
mice = np.array(['dop_8', 'dop_9', 'dop_11', 'dop_4', 'dop_7'])

	# Load Alex's actual data
psy = pd.read_pickle('all_behav.pkl')

train_set_size = 1
cross_validate = False
 
all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.    ,  0.0625, 0.125 , 0.25  ])
best_nphr_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),4])
best_nphr_NLL_individual = np.zeros([len(psy.loc[psy['virus'] == 'nphr', 'mouse_name'].unique()),1])
model_parameters = pd.DataFrame()
modelled_data = pd.DataFrame()
for i, mouse in enumerate(mice): 
    model_data_nphr, simulate_data_nphr  = \
        psy_df_to_Q_learning_model_format(psy.loc[psy['mouse_name'] == mouse], 
                                          virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0])
    
    
    obj = transform_model_struct_2_POMDP(model_data_nphr, simulate_data_nphr)
    
    virus = psy.loc[psy['mouse_name'] == mouse, 'virus'].unique()[0]
    
    opto = obj['extraRewardTrials'].to_numpy()
    lasers = []
    for i in range(len(opto)):
        try:
            lasers.append(int(opto[i][0]))
        except:
            lasers.append(int(opto[i]))

    choices = list(obj['choice'].to_numpy())
    contrasts = list(obj['stimTrials'].to_numpy())
    rewards = list(obj['reward'].to_numpy())
    laser_side = list(obj['laser_side'].to_numpy())

    
    data = (rewards[:int(len(rewards)*train_set_size)],
            contrasts[:int(len(rewards)*train_set_size)], 
            choices[:int(len(rewards)*train_set_size)], 
            lasers[:int(len(rewards)*train_set_size)])
    simulate_data = (rewards[:int(len(rewards)*train_set_size)], 
                     contrasts[:int(len(rewards)*train_set_size)], 
                     choices[:int(len(rewards)*train_set_size)], 
                  lasers[:int(len(rewards)*train_set_size)], 
                  laser_side[:int(len(rewards)*train_set_size)])
    
    if cross_validate == True:
        
        data = (rewards[:int(len(rewards)*train_set_size)],
            contrasts[:int(len(rewards)*train_set_size)], 
            choices[:int(len(rewards)*train_set_size)], 
            lasers[:int(len(rewards)*train_set_size)])
        simulate_data_test = (rewards[int(len(rewards)*train_set_size):], 
                              contrasts[int(len(rewards)*train_set_size):], 
                              choices[int(len(rewards)*train_set_size):], 
                      lasers[int(len(rewards)*train_set_size):], 
                      laser_side[int(len(rewards)*train_set_size):])
    else:
        data_test = data
        simulate_data_test = simulate_data
    

    (best_x_stay, train_NLL_stay, buffer_NLL_stay, 
     buffer_x_stay) = optimizer_stay(data, initial_guess=[0.3, 0.05, -1, 0.2])
    
    cv_aic_stay = aic((session_neg_log_likelihood_stay(best_x_stay,
              *data_test, pregen_all_posteriors=True))*-1,5)
    
   
    cv_LL_stay = (session_neg_log_likelihood_stay(best_x_stay, *data_test,
                                                  pregen_all_posteriors=True))*-1
   
    
    
    _, cv_acc_stay = session_neg_log_likelihood_stay(best_x_stay, *data_test, 
                   pregen_all_posteriors=True, accu=True)
   
    
    model_parameters_mouse = pd.DataFrame()
    model_parameters_mouse['x'] = [best_x_stay]
    model_parameters_mouse['LL'] = (cv_LL_stay/len(data_test[0]))
    model_parameters_mouse['aic'] = cv_aic_stay
    model_parameters_mouse['accu'] = cv_acc_stay
    model_parameters_mouse['model_name'] = 'w_stay'

    
    sim_data = generate_data_stay(simulate_data_test, all_contrasts, learning_rate=best_x_stay[0], 
                                   beliefSTD=best_x_stay[1], extraVal=best_x_stay[2], beta=best_x_stay[3])
    sim_data = pd.DataFrame(sim_data)
    
    sim_data = sim_data.rename(columns={0: "rewards", 1: "signed_contrast", 2: "simulated_choices", 3: "model_laser"})
    sim_data = np.array(sim_data)
    sim_data = pd.DataFrame(sim_data).T
    sim_data['laser'] = lasers[:int(len(rewards)*train_set_size)]
    sim_data['laser_side'] = laser_side[:int(len(rewards)*train_set_size)]
    sim_data['real_choice'] = choices[:int(len(rewards)*train_set_size)]
    sim_data['mouse_name']  = mouse
    sim_data['virus']  = virus
    sim_data['real_rewards']  = simulate_data[0]
   
    # Concatenate with general dataframes
    model_parameters_mouse['mouse'] = mouse
    model_parameters_mouse['virus'] = virus
    
    # Concatenate with general dataframes
    model_parameters = pd.concat([model_parameters, model_parameters_mouse])
    modelled_data = pd.concat([modelled_data, sim_data])

# Analysis

modelled_data = modelled_data.rename(columns={0: "rewards", 
   1: "signed_contrast", 2: "choices_standard", 3: "model_laser"})

modelled_data = calculate_QL_QR(modelled_data, model_parameters, 
                model_type= 'w_stay')
modelled_data = calculate_QL_QR(modelled_data, model_parameters, 
                model_type= 'w_stay', zero=True)
modelled_data = calculate_QL_QR(modelled_data, model_parameters, 
                model_type= 'w_stay', zero=True, retrieve_ITIQ=  True)
modelled_data = calculate_QL_QR(modelled_data, model_parameters, 
                model_type= 'w_stay', zero=False, retrieve_ITIQ=  True)

# Calculate a few things
psy['QL'] = np.nan
psy['QR'] = np.nan
psy['QL0'] = np.nan
psy['QR0'] = np.nan
psy['QRQL'] = np.nan
psy['QRQL0'] = np.nan
psy['pRight'] = np.nan
psy['pRight0'] = np.nan
psy['ITIQL'] = np.nan
psy['ITIQR'] = np.nan
psy['ITIQL0'] = np.nan
psy['ITIQR0'] = np.nan
psy['ITIQRQL'] = np.nan
psy['ITIpRight'] = np.nan
psy['ITIpRight0'] = np.nan
psy['ITIQRQL'] = np.nan
psy['ITIQRQL0'] = np.nan

for i, mouse in enumerate(mice):
psy.loc[psy['mouse_name'] == mouse, ['QL', 'QR', 'QRQL','pRight', 'QL0', 'QR0', 'pRight0', 'QRQL0',
                                     'ITIQL', 'ITIQR', 'ITIQL0', 'ITIQR0', 'ITIQRQL', 'ITIpRight',
                                     'ITIpRight0', 'ITIQRQL', 'ITIQRQL0']] =\
modelled_data.loc[modelled_data['mouse_name'] == mouse,
                      ['QL', 'QR', 'QRQL','pRight', 'QL0', 'QR0', 'pRight0', 'QRQL0',
                                     'ITIQL', 'ITIQR', 'ITIQL0', 'ITIQR0', 'ITIQRQL', 'ITIpRight',
                                     'ITIpRight0', 'ITIQRQL', 'ITIQRQL0']].to_numpy()

psy['argmax_choice'] = (psy['pRight']>0.5)*1


for mouse in mice:
model_choice_prob(psy, mouse, save = False)
model_choice_raw_prob(psy, mouse, save = False)


boxplot_model_parameters_per_mouse(model_parameters, 
                                   model_type= 'w_stay', 
                                   save = True)
plot_q_trial_whole_dataset(psy)
plot_q_trial_whole_dataset_per_mouse(psy)
model_performance(model_parameters, modelled_data, model_type= 
              'w_stay', save = True)

plot_choice_prob_opto_block(psy, 1, 'dop_7', save =True)
plot_choice_prob_opto_block(psy, 4, 'dop_8', save =True)
plot_choice_prob_opto_block(psy, 15, 'dop_9', save =True)
plot_choice_prob_opto_block(psy, 10, 'dop_11', save =True)
plot_choice_prob_opto_block(psy, 19, 'dop_4', save =True)

plot_qmotivation_trial_whole_dataset(psy, save= True)

sns.lineplot(data = sim_data, x =1 , y=  2, hue = 'laser_side', ci = 0)
sns.lineplot(data = sim_data, x =1 , y= 'real_choice', hue = 'laser_side')



plot_choice_40_trials(psy, 1, 'dop_7', save =True)
plot_choice_40_trials(psy, 4, 'dop_8', save =True)
plot_choice_40_trials(psy, 15, 'dop_9', save =True)
plot_choice_40_trials(psy, 10, 'dop_11', save =True)
plot_choice_40_trials(psy, 19, 'dop_4', save =True)

plot_choice_trial_from_model(psy, save= True)

plot_qr_trial_whole_dataset(psy, save= True)
plot_ql_trial_whole_dataset(psy, save= True)


save_Q(psy, root = '/Volumes/witten/Alex/recordings_march_2020_dop')



all_contrasts = np.array([-0.25  , -0.125 , -0.0625,  0.    ,  0.0625, 0.125 , 0.25  ])


norm.cdf(0, loc = -0.25, scale = 0.12*2)

a= 0
for j,i in enumerate(all_contrasts):
a += simulation_contrast_distribution( -0.25, 0.12, all_contrasts)[j] * norm.cdf(0,loc=i, scale=0.12)