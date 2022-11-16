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
pals = dict({'OFC':'#6495ED',
    'NAcc':'#7FFF00', 
    'PFC':'#7AC5CD',
    'DMS':'#76EEC6',
    'VPS':'#3D9140',
    'VP':'#F08080',
    'SNr':'#8B1A1A',
    'Olfactory':'#838B8B',
    'DLS':'#9BCD9B',
    'GPe':'#FF3030'})


# Decoders
decoder_paths = ['/Volumes/witten/Alex/decoders_raw_results/decoder_output_rpe_outcome_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_choice_cue_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_deltaq_cue_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_outcome_outcome_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_qchosen_cue_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_qchosen_outcome_forget',
'/Volumes/witten/Alex/decoders_raw_results/decoder_output_qchosen_outcome_post_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_rpe_outcome_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_choice_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_deltaq_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_outcome_outcome_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_outcome_forget',
'/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_outcome_post_forget']

varss = ['rpe',
'choice',
'deltaq',
'outcome',
'qchosen_pre',
'qchosen_pre',
'q_chosen_post',
'rpe',
'choice',
'deltaq',
'outcome',
'qchosen_pre',
'qchosen_pre',
'q_chosen_post']

epochs = ['outcome',
'action',
'cue',
'outcome',
'cue',
'outcome',
'outcome',
'outcome',
'action',
'cue',
'outcome',
'cue',
'outcome',
'outcome']

x_types = ['raw',
'raw',
'raw',
'raw',
'raw',
'raw',
'raw',
'residuals',
'residuals',
'residuals',
'residuals',
'residuals',
'residuals',
'residuals']

def load_decoders(decoder_path, var = None, epoch = None, x_type = None, null=False):
    '''
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
        l_performance = np.mean(np.nanmean(p_summary, axis=4), axis=2)
        # Get summary with optimal lambda
        acc = pd.DataFrame()
        for c in np.arange(np.shape(p_summary)[3]):
            predict = []
            predict_mse = []
            for b in np.arange(np.shape(p_summary)[2]):
                predict_f = []
                predict_f_mse = []
                for fold in np.arange(np.shape(p_summary)[0]):
                    l = np.nanargmax(l_performance[fold,:,c])
                    predict_f.append(np.nanmean(p_summary[fold,l,b,c,:]))
                    predict_f_mse.append(np.nanmean(mse_summary[fold,l,b,c,:]))
                predict.append(np.nanmean(predict_f))
                predict_mse.append(np.nanmean(predict_f_mse))           
            acc_combo = pd.DataFrame()
            acc_combo['r'] = predict
            acc_combo['mse'] = predict_mse
            acc_combo['time_bin'] = np.arange(np.shape(p_summary)[2])
            acc_combo['n_neurons'] = n_neuron_combos_tried[c]
            acc = pd.concat([acc,acc_combo])
        if null==True:
            acc['region'] = f_path[len(decoders_path)+9:-34]
        else:
            acc['region'] = f_path[len(decoder_path)+6:-34]
        acc['mouse'] = f_path[-33:-27]
        acc['hemisphere'] = f_path[-15:-14]
        acc['date'] = f_path[-26:-16]
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
if __name__=='__main__':
#### Run summary ######
    try:
        decoders_summary = pd.read_csv('/Volumes/witten/Alex/decoders_raw_results/decoders_summary.csv')
    except:
        decoders_summary = load_all_decoders(decoder_paths, varss, epochs, x_types)
        decoders_summary.to_csv('/Volumes/witten/Alex/decoders_raw_results/decoders_summary.csv')

    # Figure 1
    def plot_summary(decoders_restricted, n_neurons=20):

        plot_variables = [ 'outcome', 'choice', 'deltaq', 'qchosen_pre', 'qchosen_pre', 'q_chosen_post', 'rpe']
        plot_epochs = [ 'outcome', 'action', 'cue', 'cue', 'outcome', 'outcome', 'outcome']
        ys =  [ 'r', 'r', 'r', 'r', 'r', 'r', 'r']
        plot_ys = [ 'Accuracy', 'Accuracy', 'pearson-r', 'pearson-r', 'pearson-r','pearson-r', 'pearson-r']
        n_neurons = 20

        fig, ax = plt.subplots(2,7, sharex=True, sharey=True)
        for i, var in enumerate(plot_variables):
            plt.sca(ax[0,i])
            decoders_restricted_var_residuals = decoders_restricted.loc[(decoders_restricted['variable']==var) & 
                                                        (decoders_restricted['x_type']=='residuals') & 
                                                        (decoders_restricted['epoch']==plot_epochs[i]) & 
                                                        (decoders_restricted['n_neurons']==n_neurons)]
            sns.lineplot(data=decoders_restricted_var_residuals, x='time_bin', y=ys[i], hue='region', errorbar='se',
                        palette=pals)
            plt.title('Residuals\n' + var) 
            plt.legend().remove()
            plt.axvline(x = 4, ymin = 0, linestyle='--',  color='k', alpha=0.5)
            if i==0:
                plt.ylabel('pearson-r')
            plt.sca(ax[1,i])
            decoders_restricted_var_residuals = decoders_restricted.loc[(decoders_restricted['variable']==var) & 
                                                        (decoders_restricted['x_type']=='raw')& 
                                                        (decoders_restricted['epoch']==plot_epochs[i])& 
                                                        (decoders_restricted['n_neurons']==n_neurons)]
            sns.lineplot(data=decoders_restricted_var_residuals, x='time_bin', y=ys[i], hue='region', errorbar='se',
                        palette=pals)       
            plt.title('Raw\n' + var) 
            plt.xlabel('Time from ' + plot_epochs[i])
            plt.axvline(x = 4, ymin = 0, linestyle='--', color='k', alpha=0.5)
            plt.xticks(np.arange((post_time-pre_time)/bin_size)[::5], np.round(np.arange(pre_time,post_time,bin_size)[1::5],2), rotation=90)
            if i==0:
                plt.ylabel('pearson-r')
            if i!=len(plot_variables)-1:
                plt.legend().remove()
            else:
                plt.legend(bbox_to_anchor=(2, 1.5))
            sns.despine()

    def plot_n_neurons_r(decoders_restricted):
        plot_variables = [ 'outcome', 'choice', 'deltaq', 'qchosen_pre', 'qchosen_pre', 'q_chosen_post', 'rpe']
        plot_epochs = [ 'outcome', 'action', 'cue', 'cue', 'outcome', 'outcome', 'outcome']
        fig, ax = plt.subplots(2,7, sharex=True, sharey=True)
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


    selected_regions = np.array(['OFC', 'NAcc', 'PFC', 'DMS', 'VPS', 'VP', 'Olfactory', 'DLS'])
    decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
    plot_summary(decoders_restricted)
    plot_n_neurons_r(decoders_restricted)
    selected_regions = np.array(['OFC', 'NAcc', 'PFC', 'DMS', 'VPS', 'VP', 'SNr','Olfactory', 'DLS', 'GPe'])
    decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
    plot_summary(decoders_restricted)
    plot_n_neurons_r(decoders_restricted)

# By region vs null for q chosen
    # 1st load the nulls
    null_path = '/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget'
    try:
        nsummary = pd.read_csv('/Volumes/witten/Alex/decoders_raw_results/nsummary.csv')
    except:
        nsummary = load_decoders(null_path, var = 'qchosen_pre', epoch = 'cue', x_type = 'residuals', null=True)
        nsummary.to_csv('/Volumes/witten/Alex/decoders_raw_results/nsummary.csv')

    selected_regions = np.array(['OFC', 'NAcc', 'PFC', 'DMS', 'VPS', 'VP', 'SNr','Olfactory', 'DLS', 'GPe'])
    decoders_restricted =  decoders_summary.loc[np.isin(decoders_summary['region'], selected_regions)]
    decoders_restricted = decoders_restricted.loc[(decoders_restricted['variable']=='qchosen_pre') & 
                                                        (decoders_restricted['x_type']=='residuals') & 
                                                        (decoders_restricted['epoch']=='cue') & 
                                                        (decoders_restricted['n_neurons']==20)]
    nsummary_restricted =  nsummary.loc[np.isin(nsummary['region'], selected_regions) & (nsummary['n_neurons']==20)] # The loaded null is only for one variable so there is no need to filter further


    def plot_null_analysis(decoders_restricted , nsummary_restricted, varss = None, epoch = None):
        decoders_restricted['ses_id'] = decoders_restricted['mouse']+decoders_restricted['date']+decoders_restricted['hemisphere']
        ids = decoders_restricted['ses_id'].unique().tolist()
        colors = [list(np.random.choice(range(256), size=3)/256) for i in range(len(ids))]
        pale = dict(zip(ids,colors))
        fig,ax  = plt.subplots(5,2, sharey=True, sharex=True)
        for i, reg in enumerate(selected_regions):
            plt.sca(ax[i%5,int(i/5)])
            region_data = decoders_restricted.loc[decoders_restricted['region']==reg]
            region_null = nsummary_restricted.loc[nsummary_restricted['region']==reg]
            sns.lineplot(data=region_data, x='time_bin', y='r', errorbar=None, hue='ses_id', palette='Reds')
            sns.lineplot(data=region_null, x='time_bin', y='r', errorbar=('pi', 95), color='k', alpha=0.5)
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



    plot_null_analysis(decoders_restricted , nsummary_restricted, varss = 'qchosen_pre', epoch = 'cue')
    plot_null_r(nsummary_restricted)

    # Lambdas summary
    decoder_path = '/Volumes/witten/Alex/decoders_residuals_results/decoder_output_qchosen_cue_forget'
    f=[]
    f= glob.glob(decoder_path+'/*real*p_summary.npy')    
    os.chdir(decoder_path)
    lams = [0.001,0.01,0.1,1,10]
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
    sns.hisplot(lambdas)