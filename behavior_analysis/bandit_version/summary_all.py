import numpy as np
import seaborn as sns
import pandas as pd
from behavior_analysis.bandit_version.session_summary \
    import load_session_dataframe, fit_GLM, plot_GLM
from matplotlib import pyplot as plt
from behavior_analysis.bandit_version.full_bandit_fix \
    import full_bandit_fix
import os


# Load all data for a given virus
# Ignore hidden files
def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

def load_condition(root):
    '''Load data from all animals in a condition (e.g all chr2)'''
    behav=pd.DataFrame()
    for mouse in listdir_nohidden(root):
        print(mouse)
        behav_m=pd.DataFrame()
        for day in listdir_nohidden(root+'/'+mouse):
            print(day)
            behav_d=pd.DataFrame()
            for ses in listdir_nohidden(root+'/'+mouse+'/'+day):
                print(ses)
                path = root+'/'+mouse+'/'+day+'/'+ses
                full_bandit_fix(path)
                behav_s = load_session_dataframe(path)
                behav_s['mouse']=mouse
                behav_d = pd.concat([behav_d, behav_s])
            behav_m = pd.concat([behav_m,behav_d])
        behav = pd.concat([behav, behav_m])
    return behav

def stay_opto(behav,mouse, root):
    sns.barplot(x=behav['previous_outcome_1'], y=behav['choice']==behav['previous_choice_1'],
                hue = behav['previous_laser_1'], ci =68)
    plt.xlabel('Outcome t-1')
    plt.ylabel('% Stay')
    plt.savefig(root+'/'+mouse+'stay.pdf')

def plot_glm_per_animal(behav, root):
    params_pool = pd.DataFrame()
    pool_acc=[]
    for mouse in behav['mouse'].unique():
        behav_m = behav.loc[behav['mouse']==mouse]
        params, acc = fit_GLM(behav_m.reset_index())
        plot_GLM(params,acc)
        plt.savefig(root+'/'+mouse+'.pdf')
        plt.close()
        stay_opto(behav_m,mouse,root)
        plt.close()
        params_pool = pd.concat([params_pool,params])
        pool_acc+=acc
    pool_acc = pool_acc/len(behav['mouse'].unique())
    params_pool = params_pool.reset_index()
    pool_plot = params_pool.groupby(['index','type']).mean().reset_index()
    pool_plot['ci_95'] = params_pool.groupby('index').sem()['coefficient'].to_numpy()
    plot_GLM(pool_plot,pool_acc)
    plt.savefig(root+'/summary_GLM.pdf')
    plt.close()
    behav['repeat'] = 1*(behav['choice']==behav['previous_choice_1'])
    summary_stay = behav.groupby(['mouse', 'previous_outcome_1', 'previous_laser_1']).mean().reset_index()
    sns.catplot(x='previous_laser_1',
                y='repeat',
                hue='mouse',
                col='previous_outcome_1',
                    capsize=.2, height=6, aspect=.75,
                    kind="point", data = summary_stay, ci=68)
    plt.ylim(0.6,1)
    plt.savefig(root+'/summary_stay.pdf')
    plt.close()
if __name__=='__main__':
    root = '/Volumes/witten/Alex/Data/Subjects/opto_chr2_bandit/chr2'
    root_control = '/Volumes/witten/Alex/Data/Subjects/opto_chr2_bandit/yfp'

    chr2 = load_condition(root)
    yfp = load_condition(root_control)

    plot_glm_per_animal(chr2, root)
    plot_glm_per_animal(yfp, root_control)
    params, acc = fit_GLM(chr2.reset_index())
    plot_GLM(params,acc)
    plt.savefig(root+'/GLM_all.pdf')
    plt.close()
    params, acc = fit_GLM(yfp.reset_index())
    plot_GLM(params,acc)
    plt.savefig(root_control+'/GLM_all.pdf')
    plt.close()
