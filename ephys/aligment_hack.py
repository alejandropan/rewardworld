import numpy as np
import sys

if __name__ == "__main__":
    probe = sys.argv[1]
    cpt = np.load(probe+'/clusters.peakToTrough.npy')
    cc = np.load(probe+'/clusters.channels.npy')
    np.save(probe+'/clusters.peakToTrough_original.npy', cpt)
    np.save(probe+'/clusters.peakToTrough.npy', np.zeros(10000))
