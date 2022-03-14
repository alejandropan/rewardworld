import numpy as np
import pandas as pd
from model_comparison_accu import make_deltas, load_data, inv_logit, make_stan_data
import seaborn as sns
from matplotlib import pyplot as plt

def q_learning_model_start_end(standata,saved_params=None, fit=None, csv=True,st=0,end=-150):
    # st and end are the start and end of the trial set to be included
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
            if end < 0: #for negative indexing
                end = NT_all[ms_i]+end
            for t in np.arange(st,end):
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

def compare_params_3(saved_params1,saved_params2,saved_params3,standata):
    var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                'alphalaserforgetting_ses':'αLaserForget',
                'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                'balphastay_ses':'αBStay', 'alphalaserloss_ses':'αLaserLoss', 'alphaloss_ses':'αWaterLoss',
                'laserdecay_ses':'αLaserDecay', 'laserdecayloss_ses':'αLaserDecayLoss',
                'betalaserlossbase_ses':'βLaserLossBase', 'betalaserbase_ses':'βLaserBaser', 'ep_ses':'βStay'}
    var_order = ['βWater','βLaser','βStay', 'βWaterLoss', 'βLaserLoss','βLaserStay','βLaserLossBase','βLaserBase',
                'αWater','αLaser','αStay','αWaterForget','αLaserForget','αWaterLoss','αLaserLoss','αStayLaser',
                'αBWater','αBLaser','αBStay','αLaserDecay','αLaserDecayLoss','ε','εStay','Bias']
    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    saved_params_new1 = pd.DataFrame()
    saved_params_new2 = pd.DataFrame()
    saved_params_new3 = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            saved_params_new1 = pd.concat([saved_params_new1, saved_params1.loc[(saved_params1['name'].str.endswith('['+str(ms_i+1)+']'))]])
            saved_params_new2 = pd.concat([saved_params_new2, saved_params2.loc[(saved_params2['name'].str.endswith('['+str(ms_i+1)+']'))]])
            saved_params_new3 = pd.concat([saved_params_new3, saved_params3.loc[(saved_params3['name'].str.endswith('['+str(ms_i+1)+']'))]])


    saved_params1 = saved_params_new1
    saved_params2 = saved_params_new2
    saved_params3 = saved_params_new3


    params1=saved_params1.loc[(saved_params1['name'].str.endswith(']')) &
        (saved_params1['name'].str.contains('s\[')),['name','Mean']]
    params1['parameter']=params1['name'].str.rstrip('[123456789102030]')
    params1['parameter'] = params1['parameter'].map(var_names)

    params2=saved_params2.loc[(saved_params2['name'].str.endswith(']')) &
        (saved_params2['name'].str.contains('s\[')),['name','Mean']]
    params2['parameter']=params2['name'].str.rstrip('[123456789102030]')
    params2['parameter']=params2['parameter'].map(var_names)

    params3=saved_params3.loc[(saved_params3['name'].str.endswith(']')) &
        (saved_params3['name'].str.contains('s\[')),['name','Mean']]
    params3['parameter']=params3['name'].str.rstrip('[123456789102030]')
    params3['parameter']=params3['parameter'].map(var_names)

    params1['type']='1-75'
    params2['type']='75-150'
    params3['type']='150-'

    params = pd.DataFrame()
    params = pd.concat([params1,params2, params3]).reset_index()

    sns.swarmplot(data=params, x='parameter', y='Mean',
                order=np.array(var_order)[np.isin(var_order, params1['parameter'].unique())],hue='type')

    sns.barplot(data=params, x='parameter', y='Mean', color='gray',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())])
    plt.ylabel('Coefficient')
    plt.xticks(rotation=90)
    sns.despine()

    g = sns.catplot(x='type', y='Mean', col='parameter',
                    palette='viridis',
                    hue='name',
                    kind="point", data=params, ci=None)
    g._legend.remove()

    plt.legend().remove()
    plt.xlabel('Trial set')
    plt.ylabel('Coefficient')
    plt.show()


    laser_1 = params1.loc[params1['parameter']=='βLaser', 'Mean'].to_numpy()
    water_1 = params1.loc[params1['parameter']=='βWater', 'Mean'].to_numpy()
    laser_2 = params2.loc[params1['parameter']=='βLaser', 'Mean'].to_numpy()
    water_2 = params2.loc[params1['parameter']=='βWater', 'Mean'].to_numpy()
    laser_3 = params3.loc[params1['parameter']=='βLaser', 'Mean'].to_numpy()
    water_3 = params3.loc[params1['parameter']=='βWater', 'Mean'].to_numpy()

    fig, ax = plt.subplots(1,3)
    plt.sca(ax[0])
    plt.scatter(laser_1,water_1)
    plt.ylabel('βWater')
    plt.xlabel('βLaser')
    plt.title('1-75')
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.sca(ax[1])
    plt.scatter(laser_2,water_2)
    plt.xlabel('βLaser')
    plt.title('75-150')
    plt.xlim(0,3)
    plt.ylim(0,3)
    plt.sca(ax[2])
    plt.scatter(laser_3,water_3)
    plt.xlabel('βLaser')
    sns.despine()
    plt.title('150-')
    plt.xlim(0,3)
    plt.ylim(0,3)

def compare_params_2(saved_params1,saved_params2,standata):
    var_names = {'sides':'Bias', 'alphalaser_ses':'αLaser', 'alpha_ses':'αWater',
                'alphastay_ses':'αStay', 'laser_ses':'βLaser',
                'stay_ses':'βStay', 'beta_ses': 'βWater', 'alphaforgetting_ses':'αWaterForget',
                'alphalaserforgetting_ses':'αLaserForget',
                'dmlaser_ses':'βLaserLoss', 'dmwater_ses':'βWaterLoss',
                'laserstay_ses':'βLaserStay', 'alphalaserstay_ses': 'αStayLaser',
                'lapse_ses':'ε', 'balphalaser_ses':'αBLaser', 'balpha_ses':'αBWater',
                'balphastay_ses':'αBStay', 'alphalaserloss_ses':'αLaserLoss', 'alphaloss_ses':'αWaterLoss',
                'laserdecay_ses':'αLaserDecay', 'laserdecayloss_ses':'αLaserDecayLoss',
                'betalaserlossbase_ses':'βLaserLossBase', 'betalaserbase_ses':'βLaserBaser', 'ep_ses':'βStay'}
    var_order = ['βWater','βLaser','βStay', 'βWaterLoss', 'βLaserLoss','βLaserStay','βLaserLossBase','βLaserBase',
                'αWater','αLaser','αStay','αWaterForget','αLaserForget','αWaterLoss','αLaserLoss','αStayLaser',
                'αBWater','αBLaser','αBStay','αLaserDecay','αLaserDecayLoss','ε','εStay','Bias']

    NSxNSESS = standata['NSxNSESS']
    NT_all = standata['NT_all']
    saved_params_new1 = pd.DataFrame()
    saved_params_new2 = pd.DataFrame()
    for ms_i in np.arange(NSxNSESS):
        if NT_all[ms_i]>0:
            saved_params_new1 = pd.concat([saved_params_new1, saved_params1.loc[(saved_params1['name'].str.endswith('['+str(ms_i+1)+']'))]])
            saved_params_new2 = pd.concat([saved_params_new2, saved_params2.loc[(saved_params2['name'].str.endswith('['+str(ms_i+1)+']'))]])

    saved_params1 = saved_params_new1
    saved_params2 = saved_params_new2

    params1=saved_params1.loc[(saved_params1['name'].str.endswith(']')) &
        (saved_params1['name'].str.contains('s\[')),['name','Mean']]
    params1['parameter']=params1['name'].str.rstrip('[123456789102030]')
    params1['parameter'] = params1['parameter'].map(var_names)

    params2=saved_params2.loc[(saved_params2['name'].str.endswith(']')) &
        (saved_params2['name'].str.contains('s\[')),['name','Mean']]
    params2['parameter']=params2['name'].str.rstrip('[123456789102030]')
    params2['parameter']=params2['parameter'].map(var_names)

    params1['type']='1-75'
    params2['type']='150-'

    params = pd.DataFrame()
    params = pd.concat([params1,params2]).reset_index()

    sns.swarmplot(data=params, x='parameter', y='Mean',
                order=np.array(var_order)[np.isin(var_order, params1['parameter'].unique())],hue='type')

    sns.barplot(data=params, x='parameter', y='Mean', color='gray',
                order=np.array(var_order)[np.isin(var_order, params['parameter'].unique())])
    plt.ylabel('Coefficient')
    plt.xticks(rotation=90)
    sns.despine()

    g = sns.catplot(x='type', y='Mean', col='parameter',
                    palette='viridis',
                    hue='name',
                    kind="point", data=params, ci=None)
    g._legend.remove()

    plt.legend().remove()
    plt.xlabel('Trial set')
    plt.ylabel('Coefficient')
    plt.show()

    laser_1 = params1.loc[params1['parameter']=='βLaser', 'Mean'].to_numpy()
    water_1 = params1.loc[params1['parameter']=='βWater', 'Mean'].to_numpy()
    laser_2 = params2.loc[params1['parameter']=='βLaser', 'Mean'].to_numpy()
    water_2 = params2.loc[params1['parameter']=='βWater', 'Mean'].to_numpy()
    fig, ax = plt.subplots(1,2)
    plt.sca(ax[0])
    plt.scatter(laser_1,water_1)
    plt.ylabel('βWater')
    plt.xlabel('βLaser')
    plt.title('1-75')
    plt.sca(ax[1])
    plt.scatter(laser_2,water_2)
    plt.xlabel('βLaser')
    plt.title('150-')

def q_learning_model_start_end_alpha_decay(standata,saved_params=None, fit=None, csv=True,st=0,end=-150):
    # st and end are the start and end of the trial set to be included
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
                    alphalaser = float(saved_params.loc[saved_params['name']=='alphalaser_ses['+str(ms_i+1)+']', 'Mean'])*2
                    alphastay = float(saved_params.loc[saved_params['name']=='alphastay_ses['+str(ms_i+1)+']', 'Mean'])
            #    else:
            #        print('Error: Only csv version available at the moment!')
            else:
                if ms_i==0:
                    print('No saved parameters')
            alphalaserdecay = 0.1
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
            if end < 0: #for negative indexing
                end = NT_all[ms_i]+end
            for t in np.arange(st,end):
                t = int(t)
                predicted_choice = inv_logit(side_mouse
                    + beta_mouse  * (q[1] - q[0])
                    + stay_mouse * (qstay[1] - qstay[0])
                    + laser_mouse * (qlaser[1] - qlaser[0]))

                choice = c[sub_idx[ms_i], sess_idx[ms_i], t]
                q[choice] = (1-alpha) * q[choice] + alpha * r[sub_idx[ms_i], sess_idx[ms_i],t]
                qlaser[choice] = (1-alphalaser) * qlaser[choice] + alphalaser * l[sub_idx[ms_i], sess_idx[ms_i],t]
                if  l[sub_idx[ms_i], sess_idx[ms_i],t]==1:
                    alphalaser=(1-alphalaserdecay)*alphalaser
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


        NSxNSESS = standata['NSxNSESS']
        NT_all = standata['NT_all']





def first_block_analysis(standata, psy, saved_params):
    ses_2_number = \
    ['dop_21','dop_21','dop_21','dop_21','dop_21','dop_21',
    'dop_16','dop_16','dop_16','dop_16','dop_16','dop_16',
    'dop_36','dop_36','dop_36','dop_36','dop_36','dop_36',
    'dop_38','dop_38','dop_38','dop_38','dop_38','dop_38',
    'dop_14','dop_14','dop_14','dop_14','dop_14','dop_14',
    'dop_24','dop_24','dop_24','dop_24','dop_24','dop_24',
    'dop_30','dop_30','dop_30','dop_30','dop_30','dop_30']

    psy['id'] = psy['mouse'] + psy['date']
    bias_data = pd.DataFrame()
    for ms_i in np.arange(standata['NSxNSESS']):
            if NT_all[ms_i]>0:
                ses_bias_data = pd.DataFrame()
                ses_bias_data['mouse']=[ses_2_number[ms_i]]
                ses_bias_data['bias']= float(saved_params.loc[saved_params['name']=='sides['+str(ms_i+1)+']', 'Mean'])
                psy_mouse = psy.loc[psy['mouse']==ses_2_number[ms_i]]
                ses_id = psy_mouse.groupby(['id']).count().reset_index().loc[psy_mouse.groupby(['id']).count().reset_index()['choices']==standata['NT_all'][ms_i],'id'].to_numpy()[0]
                p_left_f_block = psy.loc[(psy['id']==ses_id)&(psy['laser_block']==1),'probabilityLeft'].to_numpy()[0]
                ses_bias_data['right_choices'] = psy.loc[psy['id']==ses_id,'choices'].mean()
                if p_left_f_block>0.5:
                    ses_bias_data['first_opto_block']=['L']
                else:
                    ses_bias_data['first_opto_block']=['R']
                bias_data = pd.concat([bias_data,ses_bias_data])
    sns.barplot(data=bias_data,x='first_opto_block', y='bias', ci=68,palette='gray')
    sns.swarmplot(data=bias_data,x='first_opto_block', y='bias', color='k')
    plt.ylabel('Bias Coefficient')
    plt.show()
    sns.barplot(data=bias_data,x='first_opto_block', y='right_choices', ci=68, palette='gray')
    sns.swarmplot(data=bias_data,x='first_opto_block', y='right_choices', color='k')
    plt.ylabel('Fraction of right choices')
    plt.show()
    return bias_data


psy=load_data()
standata = make_stan_data(psy)
model_standard_75 = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/standard_75/output/summary.csv')
model_standard_75150 = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/standard_75150/output/summary.csv')
model_standard_rest = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/standard_rest/output/summary.csv')
model_reinforce_75 = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/reinforce_75/output/summary.csv')
model_reinforce_75150 = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/reinforce_75150/output/summary.csv')
model_reinforce_rest = pd.read_csv('/Volumes/witten/Alex/Data/ephys_bandit/stan_fits/investigation_laserdecay/reinforce_rest/output/summary.csv')

compare_params_3(model_standard_75,model_standard_75150,model_standard_rest,standata)

compare_params_2(model_reinforce_75,model_reinforce_rest,standata)

saved_params1=model_standard_75
saved_params2=model_standard_75150
saved_params3=model_standard_rest

accuracies = np.zeros([2,2])
accuracies[0,0] = q_learning_model_start_end(standata,saved_params=model_standard_first150,st=0,end=150)['acc'].unique().mean()
accuracies[0,1]= q_learning_model_start_end(standata,saved_params=model_standard_rest,st=0,end=150)['acc'].unique().mean()
accuracies[1,1] = q_learning_model_start_end(standata,saved_params=model_standard_rest,st=150,end=-150)['acc'].unique().mean()
accuracies[1,0] = q_learning_model_start_end(standata,saved_params=model_standard_first150,st=150,end=-150)['acc'].unique().mean()
sns.heatmap(accuracies, annot=True)
plt.xticks([0.5,1.5],['First 150 params','Rest Params'])
plt.yticks([0.5,1.5],['First 150 data','Rest Data'])
accuracies = q_learning_model_start_end_alpha_decay(standata,saved_params=model_standard_first150,st=0,end=-150)['acc'].unique().mean()
