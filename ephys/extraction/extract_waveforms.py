from phylib.io.model import load_model
from pathlib import Path
import numpy as np
import sys

def extract_waveforms(p_path):
    clusters = np.unique(np.load(p_path+'/pykilosort/spikes.clusters.npy'))
    m  = load_model(p_path+'/pykilosort/params.py')
    new_waveforms = np.zeros([np.max(clusters)+1,82, 32])
    new_waveforms[:] = np.nan

    for cluster_id in clusters:
        w_m = m.get_cluster_mean_waveforms(cluster_id, unwhiten=False)
        new_waveforms[cluster_id,:,:] = np.array(w_m['mean_waveforms'])
    np.save(p_path +'/pykilosort/clusters.waveforms.npy', new_waveforms)

def update_params_file(ses):
    for p_path in Path(ses).glob('alf/probe0*'):
        # First make sure that params file points to the right file
        p = p_path.as_posix()[-7:]
        # Get bin file path, try to find a CAR version, if not, use original, if not present, use compressed version
        try:
            bin_path = [b.as_posix() for b in Path(ses+'/raw_ephys_data/'+p).rglob('*tcat*.bin')][0]
        except:
            try:
                bin_path = [b.as_posix() for b in Path(ses+'/raw_ephys_data/'+p).rglob('*.bin')][0]
            except:
                bin_path = [b.as_posix() for b in Path(ses+'/raw_ephys_data/'+p).rglob('*.cbin')][0]
        # Now edit params files
        with open(p_path.as_posix()+'/pykilosort/params.py', 'r') as file:
            lines = file.readlines()
        lines[0] = 'dat_path = "%s"\n' %bin_path
        with open(p_path.as_posix()+'/pykilosort/params.py', 'w') as file:
            file.writelines(lines)

if __name__ == "__main__":
    ses = sys.argv[1]
    update_params_file(ses)
    for p_path in Path(ses).glob('alf/probe0*'):
        try:
            extract_waveforms(p_path.as_posix())
        except:
            print('error extracting waveforms from ' + p_path.as_posix())

