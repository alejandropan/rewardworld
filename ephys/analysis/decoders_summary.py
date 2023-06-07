import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob

# Hardcoded parameters
n_neuron_combos_tried = np.array([10,20,30,50])
pre_time = -0.5
post_time = 1
bin_size = 0.1

#Palette
pals = dict({
    'SS':'#80007f',
    'MO':'#a040a0',
    'PFC':'#c181c1',
    'OFC':'#dfbfdf',
    'DLS':'#008000', 
    'DMS':'#40a040', 
    'TS':'#81c181', 
    'NAc':'#bfdfbf', 
    'GPe':'#d94a26',
    'VP':'#eba393',
    'Olfactory':'#404040',
    'Septum':'#666',
    'Thalamus':'#8c8c8c',
    'Hypothal':'#b3b3b3',
    'Other':'#d9d9d9'})

# Decoders

decoder_paths = [
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_outcome_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_rpe_outcome_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_rpe_outcome_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_qchosen_outcome_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_outcome_outcome_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_outcome_outcome_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_qchosen_cue_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_deltaq_cue_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_choice_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_deltaq_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_choice_cue_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_totalq_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_totalq_cue_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_qcontra_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qcontra_cue_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_qipsi_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qipsi_cue_forget'
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_qchosen_outcome_post_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_outcome_post_forget'
]

varss = [
'qchosen_pre',
'rpe',
'rpe',
'qchosen_pre',
'outcome',
'outcome',
'qchosen_pre',
'deltaq',
'choice',
'qchosen_pre',
'deltaq',
'choice',
'totalq',
'totalq',
'qcontra',
'qcontra',
'qipsi',
'qipsi',
'qchosen_post',
'qchosen_post',
]

epochs = [
'outcome',
'outcome',
'outcome',
'outcome',
'outcome',
'outcome',
'cue',
'cue',
'cue',
'cue',
'cue',
'cue',
'cue',
'cue',
'cue',
'cue',
'cue',
'cue',
'outcome',
'outcome'
]

x_types = [
'residuals',
'residuals',
'raw',
'raw',
'residuals',
'raw',
'raw',
'raw',
'raw',
'residuals',
'residuals',
'residuals',
'raw',
'residuals',
'raw',
'residuals',
'raw',
'residuals',
'raw',
'residuals'
]



def load_decoders(decoder_path, var = None, epoch = None, x_type = None, null=False, n_neuron_combos_tried = np.array([20])):
    '''
    This load decoders that were run with decoders that each combination of neurons is free to 
    have a different number of trials, depending on when then neurons were active
    var = the variable being decoded
    epoch = the aligment time
    x_type = residuals or raw spikes
    '''
    # First list all the "real"sessions
    f=[]
    if null == True:
        f= glob.glob(decoder_path+'/*null*p_summary.npy')
    else:
        f= glob.glob(decoder_path+'/*real*p_summary.npy')
    # generate big dataframe
    os.chdir(decoder_path)
    summary = pd.DataFrame()
    for f_path in tqdm(f):
        p_summary = np.load(f_path)
        mse_summary = np.load(f_path[:-13]+'mse_summary.npy')
        # Find optimal lambda 
        # Get summary with optimal lambda
        acc = pd.DataFrame()
        for c in np.arange(np.shape(p_summary)[3]):
            predict = []
            predict_mse = []
            for b in np.arange(np.shape(p_summary)[2]):
                predict_f = []
                predict_f_mse = []
                for fold in np.arange(np.shape(p_summary)[0]):
                    for combo in np.arange(np.shape(p_summary)[4]):
                        predict_f.append(np.nanmean(p_summary[fold,:,b,c,combo])) # this nan mean is the same as selecting the lambda asigned, since only one value will be non nan
                        predict_f_mse.append(np.nanmean(mse_summary[fold,:,b,c,:]))
                predict.append(np.nanmean(predict_f))
                predict_mse.append(np.nanmean(predict_f_mse))           
            acc_combo = pd.DataFrame()
            acc_combo['r'] = predict
            acc_combo['mse'] = predict_mse
            acc_combo['time_bin'] = np.arange(np.shape(p_summary)[2])
            acc_combo['n_neurons'] = n_neuron_combos_tried[c]
            acc = pd.concat([acc,acc_combo])
        if null==True:
            acc['region'] = f_path.split('null_')[1][:-34]
        else:
            acc['region'] = f_path[len(decoder_path)+6:-34]
        acc['mouse'] = f_path[-33:-27]
        acc['hemisphere'] = f_path[-15:-14]
        acc['date'] = f_path[-26:-16]
        if null == True:
            acc['iter'] = f_path.split('/')[-1].split('_null')[0]
            acc['type'] = 'null'
            acc['id'] =   acc['iter']+ acc['region'] + acc['type'] + \
                acc['date'] +acc['mouse'] + acc['hemisphere'] + acc['type']
        else:
            acc['type'] = 'real'
            acc['id'] =   acc['region'] + acc['type'] + \
                acc['date'] +acc['mouse'] + acc['hemisphere'] + acc['type']
        summary = pd.concat([summary,acc])
    summary  = summary.reset_index()
    summary['variable'] = var
    summary['epoch'] = epoch
    summary['x_type']  = x_type
    return summary

def load_all_decoders(decoder_paths, varss, epochs, x_types):
    decoders_summary = pd.DataFrame()
    for i, p in enumerate(decoder_paths):
        print(p)
        decoder_summary = load_decoders(p, var = varss[i], epoch = epochs[i], x_type = x_types[i])
        decoders_summary = pd.concat([decoders_summary, decoder_summary])
    return decoders_summary

#Plotting function

def plot_summary(decoders_restricted, plot_variables=None, plot_epochs=None,  n_neurons=20):
    plot_variables= np.unique(plot_variables)
    fig, ax = plt.subplots(2,len(plot_variables), sharex=True, sharey=False)
    for i, var in enumerate(plot_variables):
        plt.sca(ax[0,i])
        decoders_restricted_var_residuals = decoders_restricted.loc[(decoders_restricted['variable']==var) & 
                                                    (decoders_restricted['x_type']=='residuals') & 
                                                    (decoders_restricted['epoch']==plot_epochs[i]) & 
                                                    (decoders_restricted['n_neurons']==n_neurons)]
        sns.lineplot(data=decoders_restricted_var_residuals, x='time_bin', y='r', hue='region', errorbar='se',
                    palette=pals)
        plt.title('Residuals\n' + var) 
        plt.legend().remove()
        plt.axvline(x = 4, ymin = 0, linestyle='--',  color='k', alpha=0.5)
        if i==0:
            plt.ylabel('pearson-r')
        if np.isin(var,['choice','outcome']):
            plt.ylim(0.4,1)
        else:
            plt.ylim(0,0.7)
        plt.sca(ax[1,i])
        decoders_restricted_var_residuals = decoders_restricted.loc[(decoders_restricted['variable']==var) & 
                                                    (decoders_restricted['x_type']=='raw')& 
                                                    (decoders_restricted['epoch']==plot_epochs[i])& 
                                                    (decoders_restricted['n_neurons']==n_neurons)]
        sns.lineplot(data=decoders_restricted_var_residuals, x='time_bin', y='r', hue='region', errorbar='se',
                    palette=pals)       
        plt.title('Raw\n' + var) 
        plt.xlabel('Time from ' + plot_epochs[i])
        plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
        plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[1::5],2), rotation=90)
        if np.isin(var,['choice','outcome']):
            plt.ylim(0.5,1)
        else:
            plt.ylim(0,0.7)
        if i==0:
            plt.ylabel('pearson-r')
        if i!=len(plot_variables)-1:
            plt.legend().remove()
        else:
            plt.legend(bbox_to_anchor=(2, 1.5))
        sns.despine()

def plot_n_neurons_r(decoders_restricted, plot_variables = None, plot_epochs = None):
    plot_variables= np.unique(plot_variables)
    fig, ax = plt.subplots(2,len(plot_variables), sharex=True, sharey=True)
    for i, var in enumerate(plot_variables):
        plt.sca(ax[0,i])
        decoders_restricted_res = decoders_restricted.loc[(decoders_restricted['variable']==var) & 
                                                        (decoders_restricted['x_type']=='residuals')& 
                                                        (decoders_restricted['epoch']==plot_epochs[i])]
        nsum = decoders_restricted_res.groupby(['region','n_neurons','id']).mean()['r'].reset_index()
        sns.pointplot(data=nsum, x='n_neurons', y='r', hue='region', errorbar='se', palette=pals)
        if i==0:
            plt.ylabel('Mean pearson-r')    
        plt.legend().remove()
        plt.title('Residuals\n' + var) 
        plt.sca(ax[1,i])
        decoders_restricted_res = decoders_restricted.loc[(decoders_restricted['variable']==var) & 
                                                        (decoders_restricted['x_type']=='raw')& 
                                                        (decoders_restricted['epoch']==plot_epochs[i])]
        nsum = decoders_restricted_res.groupby(['region','n_neurons','id']).mean()['r'].reset_index()
        sns.pointplot(data=nsum, x='n_neurons', y='r', hue='region', errorbar='se', palette=pals)
        if i==0:
            plt.ylabel('Mean pearson-r')
        if i!=len(plot_variables)-1:
            plt.legend().remove()
        else:
            plt.legend(bbox_to_anchor=(2, 1.5))
        plt.xlabel('decoding neurons')
        plt.title('Raw\n' + var) 
        sns.despine()

def plot_mean_by_session(decoders_restricted, var='qchosen_pre', epoch='outcome', x_type='residuals'):
    decoders_restricted = decoders_restricted.loc[(decoders_restricted['variable']==var) & (decoders_restricted['epoch']==epoch) &
                            (decoders_restricted['x_type']==x_type)]
    decoders_restricted['ses_id'] = decoders_restricted['mouse']+decoders_restricted['date']
    n_ses = len(decoders_restricted['ses_id'].unique())
    n_col = int(4)
    n_row =  int(np.ceil(n_ses/4))
    fig, ax = plt.subplots(n_col,n_row, sharex=True, sharey=True)
    for i, ses in enumerate(decoders_restricted['ses_id'].unique()):
        plt.sca(ax[int(i/4),int(i%4)])
        data = decoders_restricted.loc[decoders_restricted['ses_id']==ses]
        data = data.groupby(['region','n_neurons']).mean().reset_index()
        sns.scatterplot(data=data, x='n_neurons', y='r', hue='region', palette=pals)
        sns.lineplot(data=data, x='n_neurons', y='r', hue='region', palette=pals)
        sns.despine()
        plt.legend().remove()
        plt.ylabel('pearson-r')
        plt.title(ses)
        plt.xlabel('n_neurons')

def plot_mean_by_session_n_neurons(decoders_restricted, var='qchosen_pre', epoch='outcome', x_type='residuals', n_neurons=10):
    decoders_restricted = decoders_restricted.loc[(decoders_restricted['variable']==var) & (decoders_restricted['epoch']==epoch) &
                            (decoders_restricted['x_type']==x_type)]
    decoders_restricted['ses_id'] = decoders_restricted['mouse']+decoders_restricted['date']
    n_ses = len(decoders_restricted['ses_id'].unique())
    n_col = int(4)
    n_row =  int(np.ceil(n_ses/4))
    fig, ax = plt.subplots(n_col,n_row, sharey=True, sharex=True)
    for i, ses in enumerate(decoders_restricted['ses_id'].unique()):
        plt.sca(ax[int(i/4),int(i%4)])
        data = decoders_restricted.loc[(decoders_restricted['ses_id']==ses) & (decoders_restricted['n_neurons']==n_neurons)]
        data = data.groupby(['region','n_neurons']).mean().reset_index()
        sns.scatterplot(data=data, x='region', y='r', palette=pals)
        sns.lineplot(data=data, x='region', y='r', palette=pals)
        sns.despine()
        plt.legend().remove()
        plt.ylabel('pearson-r')
        plt.xlabel('n_neurons')
        plt.xticks(rotation=90)
        plt.title(ses)

def plot_null_analysis(decoders_restricted , nsummary_restricted, varss = None, epoch = None):
    decoders_restricted['ses_id'] = decoders_restricted['mouse']+decoders_restricted['date']+'_'+str(decoders_restricted['hemisphere'])
    ids = decoders_restricted['ses_id'].unique().tolist()
    colors = [list(np.random.choice(range(256), size=3)/256) for i in range(len(ids))]
    pale = dict(zip(ids,colors))
    fig,ax  = plt.subplots(6,2, sharey=True, sharex=True)
    for i, reg in enumerate(selected_regions):
        plt.sca(ax[i%6,int(i/6)])
        region_data = decoders_restricted.loc[decoders_restricted['region']==reg]
        region_null = nsummary_restricted.loc[nsummary_restricted['region']==reg]
        sns.lineplot(data=region_data, x='time_bin', y='r', errorbar=None, hue='ses_id', palette=pale)
        sns.lineplot(data=region_null, x='time_bin', y='r', errorbar=('pi', 99), color='k', alpha=0.5)
        plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
        plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[1::5],2), rotation=90)
        plt.xlabel('Time from ' + epoch)
        plt.ylabel('Pearson - r')
        if i!=len(selected_regions):
            plt.legend().remove()
        plt.title(reg + ' '+ varss +  ' real vs null')
        sns.despine()
    plt.tight_layout()

def plot_null_r(nsummary_restricted):
    sns.histplot(nsummary_restricted['r'])
    plt.title('Null Rs')
    sns.despine()


if __name__=='__main__':
#### Run summary ######
    try:
        decoders_summary = pd.read_csv('/Volumes/witten/Alex/decoders_raw_results/decoders_summary.csv')
    except:
        decoders_summary = load_all_decoders(decoder_paths, varss, epochs, x_types)
        decoders_summary.to_csv('/Volumes/witten/Alex/decoders_raw_results/decoders_summary.csv')

    # Delete excluded sessions: dop_53\2022-10-03
    decoders_summary = decoders_summary.loc[~(decoders_summary['id'].str.contains('2022-10-03dop_53'))].reset_index()
    # Make 'r' accuracy for categorical variables, choice and outcome
    decoders_summary.loc[(decoders_summary['variable']=='choice')| (decoders_summary['variable']=='outcome'),'r'] = \
        decoders_summary.loc[(decoders_summary['variable']=='choice')| (decoders_summary['variable']=='outcome'),'mse']
    # Figure 1
    selected_regions = np.array(['SS', 'MO', 'PFC', 'OFC', 'DLS', 'DMS', 'TS', 'NAc', 'GPe', 'VP', 'Olfactory'])
    decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
    plotting_varss = [
    'choice',
    'totalq',
    'deltaq',
    'qchosen_pre']

    plotting_epochs = [
    'cue',
    'cue',
    'cue',
    'cue']
    
    plot_summary(decoders_restricted, plot_variables=plotting_varss, plot_epochs=plotting_epochs,  n_neurons=20)    

    plotting_varss = [
    'outcome',
    'rpe',
    'qchosen_pre',
    'qchosen_post']

    plotting_epochs = [
    'outcome',
    'outcome',
    'outcome',
    'outcome']
    
    plot_summary(decoders_restricted, plot_variables=plotting_varss, plot_epochs=plotting_epochs,  n_neurons=20)    
    
    plot_n_neurons_r(decoders_restricted, plot_variables=varss, plot_epochs=epochs)
    #plot_n_neurons_r(decoders_restricted)

    selected_regions = np.array(['SS', 'MO', 'PFC', 'OFC', 'Olfactory'])
    decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
    plot_summary(decoders_restricted, plot_variables=varss, plot_epochs=epochs,  n_neurons=20)    

    selected_regions = np.array(['DLS', 'DMS', 'TS', 'NAc'])
    decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
    plot_summary(decoders_restricted, plot_variables=varss, plot_epochs=epochs,  n_neurons=20)   

    selected_regions = np.array(['GPe', 'VP'])
    decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
    plot_summary(decoders_restricted, plot_variables=varss, plot_epochs=epochs,  n_neurons=20)   



# By region vs null for q chosen
    # 1st load the nulls
    null_path = '/jukebox/witten/Alex/decoders_raw_results/decoder_output_qchosen_cue_forget'
    try:
        nsummary = pd.read_csv('/Volumes/witten/Alex/decoders_residuals_results/nsummary_qchosen_pre_cue.csv')
    except:
        nsummary = load_decoders(null_path, var = 'qchosen_pre', epoch = 'cue', x_type = 'residuals', null=True)
        nsummary.to_csv('/juekbox/witten/Alex/decoders_raw_results/nsummary_qchosen.csv')

    selected_regions = np.array(['SS', 'MO', 'PFC', 'OFC', 'DLS', 'DMS', 'TS', 'NAc', 'GPe', 'VP', 'Olfactory'])
    decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
    decoders_restricted = decoders_restricted.loc[(decoders_restricted['variable']=='qchosen_pre') & 
                                                        (decoders_restricted['x_type']=='residuals') & 
                                                        (decoders_restricted['epoch']=='cue') & 
                                                        (decoders_restricted['n_neurons']==20)]
    nsummary_restricted =  nsummary.loc[np.isin(nsummary['region'], selected_regions) & (nsummary['n_neurons']==20)] # The loaded null is only for one variable so there is no need to filter further
    plot_null_analysis(decoders_restricted , nsummary_restricted, varss = 'qchosen_pre', epoch = 'cue')
    plot_null_r(nsummary_restricted)


######

    # Lambdas summary
    decoder_path = '/Volumes/witten/Alex/decoders_residuals_results/decoder_output_choice_cue_forget'
    f=[]
    f= glob.glob(decoder_path+'/*real*p_summary.npy')    
    os.chdir(decoder_path)
    lams = np.array([0.00001,0.0001,0.001,0.01,0.1,1,10,100])
    lambdas = pd.DataFrame()
    for f_path in tqdm(f):
        p_summary = np.load(f_path)
        # Find optimal lambda 
        l_performance = np.mean(np.nanmean(p_summary, axis=4), axis=2)
        # Get summary with optimal lambda
        for c in np.arange(np.shape(p_summary)[3]):
            temp_lambdas  = pd.DataFrame()
            n_lambdas = []
            for fold in np.arange(np.shape(p_summary)[0]):
                n_lambdas.append(lams[np.nanargmax(l_performance[fold,:,c])])
            temp_lambdas['l'] = n_lambdas
            temp_lambdas['fold'] = np.arange(np.shape(p_summary)[0])
            temp_lambdas['n_neurons'] = n_neuron_combos_tried[c]
            lambdas = pd.concat([lambdas,temp_lambdas])
    sns.histplot(lambdas['l'])

    # Run Q_chosen with new decoders (combinations have different number of trials)
    def plot_pilot_analysis(decoders_restricted, selected_regions, pale=None):
        decoders_restricted['ses_id'] = decoders_restricted['mouse']+decoders_restricted['date']+decoders_restricted['hemisphere'].astype(str)
        fig,ax  = plt.subplots(6,2, sharey=True, sharex=True)
        for i, reg in enumerate(selected_regions):
            plt.sca(ax[i%6,int(i/6)])
            region_data = decoders_restricted.loc[decoders_restricted['region']==reg]
            sns.lineplot(data=region_data, x='time_bin', y='mse', errorbar='se', palette=pale)
            plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
            plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[1::5],2), rotation=90)
            plt.xlabel('Time from cue')
            plt.ylabel('Pearson - r')
            plt.title(reg)
            if i!=len(selected_regions):
                plt.legend().remove()
            sns.despine()
        plt.tight_layout()

        sns.lineplot(data=decoders_restricted, x='time_bin', y='r', errorbar='se', hue='region')
        plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
        plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[1::5],2), rotation=90)  



    decoder_path = '//Volumes/witten/Alex/decoders_raw_results/decoder_output_deltaq_cue_forget'
    d_data = load_decoders(decoder_path, var = 'decision_time', epoch = 'cue', x_type = 'residual', null=False)    
    selected_regions = np.array(['SS', 'OFC', 'NAc', 'PFC', 'DMS', 'DLS', 'TS', 'VP','Olfactory', 'PFC', 'GPe'])
    decoders_restricted =  d_data.loc[np.isin(d_data['region'], selected_regions)]
    region_data = decoders_restricted.loc[(decoders_restricted['n_neurons']==20)]
    plot_pilot_analysis(decoders_restricted, selected_regions, pale=pals)

    ### Plot striatal regions only
    # For Policy first
    str_pal = dict({
        'DLS':'#c77c2c', 
        'DMS':'#a61678', 
        'TS':'#1a2696', 
        'NAc':'#3ea33e'})
    decoder_path = '//Volumes/witten/Alex/decoders_raw_results/decoder_output_deltaq_cue_forget'
    d_data = load_decoders(decoder_path, var = 'deltaq', epoch = 'cue', x_type = 'raw', null=False)    
    selected_regions = np.array(['NAc','DMS', 'DLS', 'TS'])
    decoders_restricted =  d_data.loc[np.isin(d_data['region'], selected_regions)]
    sns.lineplot(data=decoders_restricted, x='time_bin', y='r', hue='region', palette=str_pal, errorbar='se')
    plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[1::5],2), rotation=90)
    plt.xlabel('Time from cue')
    plt.ylabel('r')
    sns.despine()

    decoder_path = '//Volumes/witten/Alex/decoders_raw_results/decoder_output_qchosen_cue_forget'
    d_data = load_decoders(decoder_path, var = 'deltaq', epoch = 'cue', x_type = 'raw', null=False)    
    selected_regions = np.array(['NAc','DMS', 'DLS', 'TS'])
    decoders_restricted =  d_data.loc[np.isin(d_data['region'], selected_regions)]
    sns.lineplot(data=decoders_restricted, x='time_bin', y='r', hue='region', palette=str_pal, errorbar='se')
    plt.xticks(np.array([0,4,9]),np.array([-0.4,0,0.5]), rotation=90)
    plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
    plt.xlabel('Time from cue')
    plt.ylabel('Pearson - r')
