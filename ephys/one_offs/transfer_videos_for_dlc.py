import shutil

LASER_ONLY = [


'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 

 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-16/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-17/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-18/002/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-19/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-27/003/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-20/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-11/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-10/002/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-09/003/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-12/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-13/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-14/003/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-16/003/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-17/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-18/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-07/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-05/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-04/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-03/001/raw_video_data/_iblrig_rightCamera.raw.mp4',
 '/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-02/001/raw_video_data/_iblrig_rightCamera.raw.mp4']







for src in LASER_ONLY:
    dst=dest+ '/' + src[-64:-58]+ '_' + src[-57:-47] + '_' + src[-26:]
    shutil.copy(src,dst)