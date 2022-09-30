import pandas as pd
import matplotlib.pyplot as plt
import sys
import numpy as np
import seaborn as sns

if __name__ == "__main__":
    ses = sys.argv[1]
    data = pd.DataFrame()
    data['choices'] = np.load(ses+'/alf/_ibl_trials.choice.npy')*-1
    data['opto_block'] = np.load(ses+'/alf/_ibl_trials.opto_block.npy')
    data['feedback'] = np.load(ses+'/alf/_ibl_trials.feedbackType.npy')
    data['repeat'] = 1*(data['choices'] == data['choices'].shift(1))
    data['feedback'] =  data['feedback'].shift(1)
    data =  data.reset_index()
    fig, ax = plt.subplots(1,3, sharey=True)
    plt.sca(ax[0])
    sns.barplot(data=data.loc[data['index']<200], x = 'opto_block', y='repeat', hue='feedback', palette='viridis', ci=68)
    plt.xlabel('Laser Block')
    plt.ylabel('Fraction of repeated choices')
    plt.title('Trial 1-199')
    plt.sca(ax[1])
    sns.barplot(data=data.loc[(data['index']>=200) & (data['index']<399)], x = 'opto_block', y='repeat', hue='feedback', palette='viridis', ci=68)
    plt.xlabel('Laser Block')
    plt.title('Trial 200-399')
    plt.sca(ax[2])
    sns.barplot(data=data.loc[(data['index']>=400) & (data['index']<600)], x = 'opto_block', y='repeat', hue='feedback', palette='viridis', ci=68)
    plt.xlabel('Laser Block')
    plt.title('Trial 400-599')
    plt.savefig(ses+'/decay.png')