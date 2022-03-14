from ibllib.io.extractors.biased_trials import extract_all
from ibllib.io.extractors.training_wheel import extract_all as extract_all_wheel
from full_bandit_fix import full_bandit_fix
from session_summary_10 import *
######################################################################################################
if __name__ == "__main__":
    ses = sys.argv[1]
    extract_all(ses, save=True)
    full_bandit_fix(ses)
    ses_df=pd.DataFrame()
    ses_df= load_session_dataframe(ses)
    # Fit and plot
    params, acc = fit_GLM(ses_df)
    plot_GLM(params,acc)
    plt.savefig(ses+'/glm_summary.png')
    plot_session_wo_laser(ses_df)
    plt.savefig(ses+'/example.png')
    plt.close()

