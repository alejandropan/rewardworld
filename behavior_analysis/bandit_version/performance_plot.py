import numpy as np
import pandas as pd
import seaborn as sns
import sys
from matplotlib import pyplot as plt
def plot_performance(ses):
    data = pd.DataFrame()
    data['performance'] = 1*(np.load(ses+'/alf/_ibl_trials.feedbackType.npy')>0)
    sns.lineplot(data=data.performance.rolling(50, min_periods=5).mean())
    plt.xlabel('Trial Number')
    plt.savefig(ses+'/performance.png')

if __name__ == "__main__":
    ses = sys.argv[1]
    plot_performance(ses)