import pandas as pd
from pathlib import Path
import sys
import numpy as np
import location_subdivider as sub

def get_location(json_file, save=True):
    json_df = pd.read_json(json_file)
    locations = json_df.iloc[np.where(json_df.index=='brain_region')[0],:].to_numpy()[0]
    sublocations = sub.get_location(json_file.as_posix())
    assert len(locations)== len(sublocations)
    if len(np.where(np.isin(locations,['ACB','CP','STR']))[0])>0:
        locations[np.where(np.isin(locations,['ACB','CP','STR']))] = \
            sublocations[np.where(np.isin(locations,['ACB','CP','STR']))]
    # Save hemisphere
    x = 1*(json_df.iloc[np.where(json_df.index=='x')[0],:].to_numpy()[0]>0)
    x = x.astype('float')
    x[np.where(np.isnan(
        json_df.iloc[np.where(json_df.index=='x')[0],:].to_numpy()[0].astype('float')))]=np.nan

    if save == True:
        np.save(json_file.parent.as_posix() + '/channels.locations.npy', locations)
        np.save(json_file.parent.as_posix() +'/channels.hemisphere.npy', x)

if __name__=='__main__':
    folder_path = sys.argv[1] + '/channel_locations.json'
    json_file = Path(folder_path)
    get_location(json_file)
