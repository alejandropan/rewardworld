# environment installation guide https://github.com/int-brain-lab/iblenv
# run "%qui qt" magic command from Ipython prompt for interactive mode
#%gui qt
import numpy as np
from mayavi import mlab
from one.api import ONE
from iblutil.util import Bunch

import ibllib.plots
from atlaselectrophysiology import rendering
import ibllib.atlas as atlas

one = ONE(base_url='https://alyx.internationalbrainlab.org')
subjects = ['dop_24','dop_14', 'dop_13', 'dop_16', 'dop_21', 'dop_22', 'dop_36']

fig = rendering.figure()

for subject in subjects:
    ba = atlas.AllenAtlas(25)
    channels_rest = one.alyx.rest('channels', 'list', subject=subject)
    channels = Bunch({
            'atlas_id': np.array([ch['brain_region'] for ch in channels_rest]),
            'xyz': np.c_[np.array([ch['x'] for ch in channels_rest]),
                         np.array([ch['y'] for ch in channels_rest]),
                         np.array([ch['z'] for ch in channels_rest])] / 1e6,
            'axial_um': np.array([ch['axial'] for ch in channels_rest]),
            'lateral_um': np.array([ch['lateral'] for ch in channels_rest]),
            'trajectory_id': np.array([ch['trajectory_estimate'] for ch in channels_rest])
        })

    for m, probe_id in enumerate(np.unique(channels['trajectory_id'])):
        traj_dict = one.alyx.rest('trajectories', 'read', id=probe_id)
        ses = traj_dict['session']
        label = (f"{ses['subject']}/{ses['start_time'][:10]}/"
                 f"{str(ses['number']).zfill(3)}/{traj_dict['probe_name']}")
        print(label)

        color = ibllib.plots.color_cycle(m)
        it = np.where(channels['trajectory_id'] == probe_id)[0]
        xyz = channels['xyz'][it]
        ins = atlas.Insertion.from_track(xyz, brain_atlas=ba)

        mlapdv = ba.xyz2ccf(ins.xyz)
        # display the interpolated tracks
        mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                    line_width=3, color=color, tube_radius=20)
    # Plot fibers and dopamine neurons [ytop,ybottom],[ztop,zbottom],[xtop,xbottom]
    mlab.plot3d(np.array([8500., 8500.]),
                    np.array([0, 5032.]),
                    np.array([4639., 5239.]),
                line_width=3, color=(0., 0.72, 1.), tube_radius=300)
    mlab.plot3d(np.array([8500., 8500.]),
                    np.array([0, 5032.]),
                    np.array([6839., 6339.]),
                line_width=3, color=(0, 0.72, 1.), tube_radius=300)

    mlab.savefig('penetrations.tiff', size=(1024, 1024))
