# Mat to alf for 2022 global encoding model ssumary

from scipy.io import loadmat
import pandas as pd


ENCODING_MODEL_MAT_FILE = '/Volumes/witten/Chris/matlab/cz/ibl_da_neuropixels_2022/archive/20220913/encoding-model/encoding-model-output-2022-08-03-12-31-PM.mat'

def load_encoding_model(model_path = ENCODING_MODEL_MAT_FILE):
    model_dict = loadmat(model_path)
    encoding_model = pd.DataFrame()
    mouse=[]
    date=[]
    ses=[]
    id=[]
    probe=[]
    [mouse.append(i[0][:6]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [date.append(i[0][7:17]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [ses.append(i[0][18:21]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [id.append(i[0]) for i in model_dict['EncodingSummary']['recording_name'][0][0][0]]
    [probe.append(int(i[0][6])) for i in model_dict['EncodingSummary']['probe'][0][0][0]]
    encoding_model['pretest'] = model_dict['EncodingSummary']['pvalues'][0][0]['pretest'][0][0][0]
    encoding_model['choice'] = model_dict['EncodingSummary']['pvalues'][0][0]['choice'][0][0][0]
    encoding_model['outcome'] = model_dict['EncodingSummary']['pvalues'][0][0]['outcome'][0][0][0]
    encoding_model['policy'] = model_dict['EncodingSummary']['pvalues'][0][0]['policy'][0][0][0]
    encoding_model['value'] = model_dict['EncodingSummary']['pvalues'][0][0]['value'][0][0][0]
    encoding_model['value_laser'] = model_dict['EncodingSummary']['pvalues'][0][0]['value_laser'][0][0][0]
    encoding_model['value_water'] = model_dict['EncodingSummary']['pvalues'][0][0]['value_water'][0][0][0]
    encoding_model['value_stay'] = model_dict['EncodingSummary']['pvalues'][0][0]['value_stay'][0][0][0]
    encoding_model['policy_laser'] = model_dict['EncodingSummary']['pvalues'][0][0]['policy_laser'][0][0][0]
    encoding_model['policy_water'] = model_dict['EncodingSummary']['pvalues'][0][0]['policy_water'][0][0][0]
    encoding_model['policy_stay'] = model_dict['EncodingSummary']['pvalues'][0][0]['policy_stay'][0][0][0]
    encoding_model['cluster_id'] = model_dict['EncodingSummary']['cluster'][0][0][0].astype(int)
    encoding_model['laser_cue_gain'] = model_dict['EncodingSummary']['gains'][0][0]['cue'][0][0][:,5]
    encoding_model['laser_contra_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_contra'][0][0][:,5]
    encoding_model['laser_ipsi_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_ipsi'][0][0][:,5]
    encoding_model['outcome_laser_gain'] = model_dict['EncodingSummary']['gains'][0][0]['outcome_laser'][0][0][:,5]
    encoding_model['dlaser_cue_gain'] = model_dict['EncodingSummary']['gains'][0][0]['cue'][0][0][:,2]
    encoding_model['dlaser_contra_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_contra'][0][0][:,2]
    encoding_model['dlaser_ipsi_gain'] = model_dict['EncodingSummary']['gains'][0][0]['movement_ipsi'][0][0][:,2]
    encoding_model['doutcome_laser_gain'] = model_dict['EncodingSummary']['gains'][0][0]['outcome_laser'][0][0][:,2]
    encoding_model['firing_rate'] =  model_dict['EncodingSummary']['firing_rate'][0][0][0]
    encoding_model['region'] =  model_dict['EncodingSummary']['region_key'][0][0][0][model_dict['EncodingSummary']['region_id'][0][0][0]-1]
    encoding_model['mouse'] = mouse
    encoding_model['date'] = date
    encoding_model['ses'] = ses
    encoding_model['probe'] = probe
    encoding_model['id'] = id
    return encoding_model

def summary_firing_rate(encoding_model):
    fig,ax = plt.subplots(5,5, sharex=True, sharey=True)
    for i, reg in enumerate(np.unique(encoding_model['region'])):
        plt.sca(ax[int(i/5),i%5])
        encoding_model_reg = encoding_model.loc[encoding_model['region']==reg[0]]
        value = encoding_model_reg.loc[encoding_model_reg['policy_laser']<=0.01, 'firing_rate']
        non_value = encoding_model_reg.loc[encoding_model_reg['policy_laser']>0.01, 'firing_rate']
        sns.histplot(non_value,stat='percent', bins=np.arange(0,100,5))
        sns.histplot(value,stat='percent',bins=np.arange(0,100,5), color='orange')
        plt.title(reg[0])
        sns.despine()
