
import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
import subprocess

SESSIONS = [
 'dop_47/2022-06-05/001',
 'dop_47/2022-06-06/001',
 'dop_47/2022-06-07/001',
 'dop_47/2022-06-09/003',
 'dop_47/2022-06-10/002',
 'dop_47/2022-06-11/001',
 #'dop_48/2022-06-20/001', Some issue with the video synching
 'dop_48/2022-06-27/002',
 'dop_48/2022-06-28/001',
 'dop_49/2022-06-14/001',
 'dop_49/2022-06-15/001',
 'dop_49/2022-06-16/001',
 'dop_49/2022-06-17/001',
 'dop_49/2022-06-18/002',
 'dop_49/2022-06-19/001',
 'dop_49/2022-06-20/001',
 'dop_49/2022-06-27/003',
 'dop_50/2022-09-12/001',
 'dop_50/2022-09-13/001',
 'dop_50/2022-09-14/003',
 'dop_53/2022-10-02/001',
 'dop_53/2022-10-03/001',
 'dop_53/2022-10-04/001',
 'dop_53/2022-10-05/001',
 'dop_53/2022-10-07/001']

#print(subprocess.run(['sbatch --array=0-'+str(len(SESSIONS)-1)+' /jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/video/decoder_qchosen_cue_forget/cue_qchosen_video.cmd'], shell=True))
print(subprocess.run(['sbatch --array=6-7 /jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis/decoders_cluster/video/decoder_qchosen_cue_forget/cue_qchosen_video.cmd'], shell=True))