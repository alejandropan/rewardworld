import numpy as np
import pandas as pd

# list of ready recordings

recordings = [
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-27/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-25/004/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-23/002/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_24/2021-03-19/002/alf/probe00_glbdmx',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-16/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-14/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-12/002/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001/alf/Probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-10/001/alf/Probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-08/001/alf/probe01',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001/alf/probe00',
'/Volumes/witten/Alex/Data/Subjects/dop_14/2021-04-06/001/alf/probe01',
]

index = np.array(['id', 'Amplitude', 'ContamPct', 'KSLabel', 'amp', 'ch',
          'depth', 'fr', 'group', 'n_spikes', 'sh'])


for r in recordings:
      prev = pd.read_csv(r+'/clusters.metrics.csv')
      metrics = pd.DataFrame()
      for var in index:
          try:
              metrics[var]=prev[var]
          except:
              metrics[var]=np.nan
      assert prev.shape[0] == metrics.shape[0]
      metrics.to_csv(r+'/clusters.metrics.csv')

for r in recordings:
    json = pd.read_json(r+'/channel_locations.json')
    x = 1*(json.iloc[np.where(json.index=='x')[0],:].to_numpy()[0]>0)
    x = x.astype('float')
    x[np.where(np.isnan(
        json.iloc[np.where(json.index=='x')[0],:].to_numpy()[0].astype('float')))]=np.nan
    np.save(r+'/channels.hemisphere.npy', x)
