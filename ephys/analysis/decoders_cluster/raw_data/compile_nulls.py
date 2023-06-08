import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import glob


n_neuron_combos_tried = np.array([20])
pre_time = -0.5
post_time = 1
bin_size = 0.1


def load_decoders(decoder_path, var = None, epoch = None, x_type = None, null=False):
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


#nsummary = load_decoders('/jukebox/witten/Alex/decoders_raw_results/decoder_output_qchosen_cue_forget', 
#                        var = 'qchosen_pre', epoch = 'cue', x_type = 'raw', null=True)
#nsummary.to_csv('/jukebox/witten/Alex/decoders_raw_results/nsummary_qchosen.csv')


nsummary = load_decoders('/jukebox/witten/Alex/decoders_raw_results/decoder_output_deltaq_cue_forget', 
                        var = 'deltaq', epoch = 'cue', x_type = 'raw', null=True)
nsummary.to_csv('/jukebox/witten/Alex/decoders_raw_results/nsummary_deltaq.csv')
