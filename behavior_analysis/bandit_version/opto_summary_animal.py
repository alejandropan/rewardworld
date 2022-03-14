import numpy as np
import seaborn as sns
import pandas as pd
from session_summary \
    import load_session_dataframe, fit_GLM, plot_GLM
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from behavior_analysis.bandit_version.full_bandit_fix \
    import full_bandit_fix

# First obtain paths and load data
root = tk.Tk()
root.withdraw()
paths = []
for i in range(5):
    paths.append(filedialog.askdirectory(initialdir='/Volumes/witten/Alex/Data/Subjects'))

behav = pd.DataFrame()
for i, ses in enumerate(paths):
    if not ses:
        continue
    else:
        full_bandit_fix(ses)
        behav_t = load_session_dataframe(ses)
        if i==0:
            behav = behav_t
        else:
            behav = pd.concat([behav, behav_t])
behav = behav.reset_index()
# Plot results

def stay_opto(behav):
    sns.barplot(x=behav['previous_outcome_1'], y=behav['choice']==behav['previous_choice_1'],
                hue = behav['previous_laser_1'], ci =68)
    plt.xlabel('Outcome t-1')
    plt.ylabel('% Stay')
    plt.show()

def stay_opto_lite(behav):
    sns.barplot(x=behav['previous_laser_1'], y=behav['choice']==behav['previous_choice_1'], ci =68)
    plt.xlabel('Laser t-1')
    plt.ylabel('% Stay')
    plt.show()

def percentage_each_choice(behav):
    sns.barplot(x=behav['previous_choice_1'], y=behav['choice']>0, ci =68)
    plt.xlabel('previous_choice')
    plt.ylabel('Choice == 1')
    plt.show()

params, acc = fit_GLM(behav)
plot_GLM(params, acc)
plt.show()
stay_opto(behav)
stay_opto_lite(behav)