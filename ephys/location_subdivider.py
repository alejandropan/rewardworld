from skimage import io
import pandas as pd
import numpy as np
#json_file = '/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001/alf/probe00/channel_locations.json'
img_path = '/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/allen_subdivisions.tif'
subdivisions = pd.read_csv('/Users/alexpan/Documents/PYTHON/rewardworld/ephys/histology_files/41467_2019_13057_MOESM4_ESM.csv')
BREGMA = np.array([43,33,570]) # AP, DV, ML

def get_location(json_file,save=False, img_path=img_path, subdivisions=subdivisions,
                BREGMA=BREGMA):
  im = io.imread(img_path)
  imarray = np.array(im)
  json_df = pd.read_json(json_file)
  subareas = []
  for i in np.arange(json_df.shape[1]-1):
    coord = [json_df.iloc[1,i], json_df.iloc[2,i], json_df.iloc[0,i]] # AP, DV, ML
    # Relation to allen_subdivisions bregma
    coord[0] = -1*int(np.round(coord[0]/100)) # AP every 100um, positive is more posterior
    coord[1] = -1*int(np.round(coord[1]/10)) # DV every 10um, down is positive
    coord[2] = int(np.round(coord[2]/10)) # ML every 10um
    label_idx = BREGMA + coord
    label = imarray[label_idx[0], label_idx[1], label_idx[2]]
    if label==0:
        subarea='void'
    else:
      subarea = subdivisions.loc[subdivisions['Structural ID']==label, 'Franklin-Paxinos Full name'].to_list()[0]
    assert len(subarea)!=0
    subareas.append(subarea)
  subareas.append('nan')
  subareas = np.array(subareas)
  if save == True:
      np.save(json_file.parent.as_posix() + '/channels.sublocations.npy', subareas)
  return subareas

if __name__=='__main__':
  json_file = sys.argv[1] + '/channel_locations.json'
  get_location(json_file)
