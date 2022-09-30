import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pandas as pd

def plot_binned_performance(ses):
    data = pd.DataFrame()
    data['block'] = np.load(ses + '/alf/_ibl_trials.probabilityLeft.npy')
    data['choices'] = np.load(ses + '/alf/_ibl_trials.choice.npy')
    data['high_prob_choices'] = np.nan
    data.loc[data.block>0.5,'high_prob_choices']= pd.Series(1*(data.loc[data['block']>0.5,'choices']==1))
    data.loc[data.block<0.5,'high_prob_choices']= pd.Series(1*(data.loc[data['block']<0.5,'choices']==-1))
    #assign within block number
    data['block_change'] = data.block.diff()
    tb = []
    counter = 0
    for i in np.arange(len(data)):
        if data.iloc[i,-1]!=0:
            counter=0
        tb.append(counter)
        counter+=1
    data['tb'] = tb
    data['high_prob_choices_s'] = data['high_prob_choices'].copy()
    data.loc[data['tb']<=5,'high_prob_choices_s'] = np.nan
    data['trial'] = data.index
    bin_np = int(len(data)/100)
    data = data.iloc[:bin_np*100,:]
    data['bin'] = pd.cut(data['trial'], bin_np)
    #sns.pointplot(data=data, x='bin', y='high_prob_choices', ci=66)
    sns.pointplot(data=data, x='bin', y='high_prob_choices_s', ci=66)
    plt.xlabel('Trial Bin')
    plt.ylabel('Performance')
    sns.despine()
    plt.savefig(ses+'/performance.pdf')

if __name__ == "__main__":
    ses = sys.argv[1]
    plot_binned_performance(ses)