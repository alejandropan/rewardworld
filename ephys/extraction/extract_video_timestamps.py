from pathlib import Path
from ibllib.io.extractors import camera

LASER_ONLY = [
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-16/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',  
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-19/001', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-27/003', 
'/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-20/001', 
#'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
#'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-10/002', 
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001',
'/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-06/001',
'/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
'/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
'/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
'/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
'/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
'/jukebox/witten/Alex/Data/Subjects/dop_50/2022-09-18/001',
'/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-07/001',
'/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-05/001',
'/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-04/001',
'/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-03s/001',
'/jukebox/witten/Alex/Data/Subjects/dop_53/2022-10-02/001']

for session in LASER_ONLY:
    camera.extract_all(Path(session))
