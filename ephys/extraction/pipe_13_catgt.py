import spikeglx as sp
import sys
import numpy as np
import glob
import os
import subprocess

ses = sys.argv[1]
for i in np.arange(4):
    probe = ses+'/raw_ephys_data/probe0%s' %i
    try:
        os.chdir(probe)
        file = glob.glob('*.ap.cbin')[0]
        #sp.Reader(file).decompress_file(save=True)
        file = glob.glob('*.ap.bin')[0]
        print(subprocess.run(['/home/ibladmin/Documents/PYTHON/CatGT-linux/runit1.sh'+ ' ' + probe + ' ' + file[:-19] + ' ' + file[-17:-16] + ' ' +file[-8]], shell=True))
    except:
        continue