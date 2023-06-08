import sys
sys.path.append('/jukebox/witten/Alex/PYTHON/rewardworld/ephys/extraction')
from decompress import decomp_file
from pathlib import Path
import subprocess

probe_files = [
'/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-20/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-20/001/raw_ephys_data/probe02/_spikeglx_ephysData_g0_t0.imec2.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-20/001/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-19/002/raw_ephys_data/probe00/_spikeglx_ephysData_g1_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-19/002/raw_ephys_data/probe02/_spikeglx_ephysData_g1_t0.imec2.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-19/002/raw_ephys_data/probe01/_spikeglx_ephysData_g1_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-28/001/raw_ephys_data/probe00/_spikeglx_ephysData48_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-27/002/raw_ephys_data/probe00/_spikeglx_ephysData48_g1_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_48/2022-06-27/002/raw_ephys_data/probe01/_spikeglx_ephysData48_g1_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-14/001/raw_ephys_data/probe00/_spikeglx_ephysData49_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-14/001/raw_ephys_data/probe01/_spikeglx_ephysData49_g0_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-15/001/raw_ephys_data/probe00/_spikeglx_ephysData49_r_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-15/001/raw_ephys_data/probe01/_spikeglx_ephysData49_r_g0_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-16/001/raw_ephys_data/probe00/_spikeglx_ephysData49_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-16/001/raw_ephys_data/probe01/_spikeglx_ephysData49_g0_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-17/001/raw_ephys_data/probe00/_spikeglx_ephysData49_g1_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-17/001/raw_ephys_data/probe01/_spikeglx_ephysData49_g1_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-18/002/raw_ephys_data/probe00/_spikeglx_ephysData49_r1_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-18/002/raw_ephys_data/probe02/_spikeglx_ephysData49_r1_g0_t0.imec2.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-18/002/raw_ephys_data/probe01/_spikeglx_ephysData49_r1_g0_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-19/001/raw_ephys_data/probe00/_spikeglx_ephysDatar_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-19/001/raw_ephys_data/probe02/_spikeglx_ephysDatar_g0_t0.imec2.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-19/001/raw_ephys_data/probe01/_spikeglx_ephysDatar_g0_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-27/003/raw_ephys_data/probe00/_spikeglx_ephysData49_g2_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-27/003/raw_ephys_data/probe01/_spikeglx_ephysData49_g2_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-20/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_49/2022-06-20/001/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-11/001/raw_ephys_data/probe00/_spikeglx_ephysData47_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-10/002/raw_ephys_data/probe00/_spikeglx_ephysData47_g1_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-10/002/raw_ephys_data/probe01/_spikeglx_ephysData47_g1_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-09/003/raw_ephys_data/probe00/_spikeglx_ephysData_g1_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-09/003/raw_ephys_data/probe01/_spikeglx_ephysData_g1_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001/raw_ephys_data/probe03/_spikeglx_ephysData_g0_t0.imec3.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001/raw_ephys_data/probe02/_spikeglx_ephysData_g0_t0.imec2.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-05/001/raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-06/001/raw_ephys_data/probe03/_spikeglx_ephysData_g1_t0.imec3.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-06/001/raw_ephys_data/probe00/_spikeglx_ephysData_g1_t0.imec0.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-06/001/raw_ephys_data/probe02/_spikeglx_ephysData_g1_t0.imec2.ap.cbin',
 '/jukebox/witten/Alex/Data/Subjects/dop_47/2022-06-06/001/raw_ephys_data/probe01/_spikeglx_ephysData_g1_t0.imec1.ap.cbin'
]

fpath = probe_files[int(sys.argv[1])]
try:
    decomp_file(fpath)
except:
    print('failed to decompress file: ' + fpath)

file_path = Path(fpath)
data_dir = file_path.parent.as_posix()
run_name = fpath[fpath.find('_spikeglx_'):fpath.find('_g')]
ga = fpath[-18:-17]
ta = fpath[-15:-14]
prb_no = fpath[-9:-8]
dst_fld = file_path.parent.as_posix()

print(subprocess.run([
'/jukebox/witten/Alex/CatGT-linux_3/runit.sh -dir='+data_dir+ ' -run=' + run_name + \
' -g=' + ga + ' -t=' +ta + ' -prb=' + prb_no + \
' -ap -no_run_fld -gbldmx -dest=' +dst_fld +\
' -out_prb_fld'], shell=True))