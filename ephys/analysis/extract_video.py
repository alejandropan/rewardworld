
import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/analysis')
from ibllib.io import ffmpeg
from pathlib import Path

LASER_ONLY = [
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-11/001'
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-10/002',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-06/001']


command = ('ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 17 '
            '-loglevel 0 -codec:a copy {file_out}')
output_files = ffmpeg.iblrig_video_compression(Path(LASER_ONLY[int(sys.argv[1])]), command)