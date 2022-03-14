from ibllib.pipes import ephys_preprocessing as ep    
import sys
from pathlib import Path

if __name__=='__main__':
        path = Path(*sys.argv[1])
        ep.EphysPulses(path).run()
        ep.RawEphysQC(path).run()