import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BREGMA = np.array([312,13,228]) #ap, dv, ml
CORONAL_SLICE = 198
HORIZONTAL_SLICE = 181


def transform_to_paxinos(ap_dv_ml, res = 25):
    loc = ap_dv_ml - BREGMA
    x,y,z= loc
    X = x * np.cos(0.0873) - y * np.sin(0.0873)
    Y = x * np.sin(0.0873) + y * np.cos(0.0873)
    Y = Y * 0.9434
    pax_coord = np.array([X,Y,z])*res
    return pax_coord

fiber_coord = pd.DataFrame()
fiber_coord['mouse'] = ['dop_47','dop_48','dop_49','dop_50','dop_53','dop_47','dop_48','dop_49','dop_50','dop_53']
fiber_coord['fibers'] = [np.array([199,181,247]), np.array([206,182,245]),np.array([197,200,245]),np.array([196,186,253]), np.array([183,188,248]),
                            np.array([199,181,218]), np.array([206,182,219]),np.array([197,200,210]), np.array([196,186,220]),np.array([183,186,210])]   #ap, dv, ml
fiber_coord['pax']  = fiber_coord['fibers'].apply(transform_to_paxinos)
fiber_coord['AP(um)'] = [c[0] for c in fiber_coord['pax']]
fiber_coord['DV(um)'] = [c[1] for c in fiber_coord['pax']]
fiber_coord['ML(um)'] = [c[2] for c in fiber_coord['pax']]


fig, ax = plt.subplots(1,2)
plt.sca(ax[0])
sns.scatterplot(data=fiber_coord,x='ML(um)',y='AP(um)', hue='mouse')
plt.axvline(x = 0, color = 'k', linestyle='dashed')
plt.ylim(-7800,5400)
plt.xlim(-2850,2850)
plt.sca(ax[1])
sns.scatterplot(data=fiber_coord,x='ML(um)',y='DV(um)', hue='mouse')
plt.axvline(x = 0, color = 'k', linestyle='dashed')
plt.ylim(7675,-325)
plt.xlim(-5700,5700)


