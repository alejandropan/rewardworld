from atlaselectrophysiology.load_histology import tif2nrrd
from pathlib import Path
import sys

"""
Creates nrrd from image path string
"""

if __name__=='__main__':
    tif2nrrd(Path(*sys.argv[1:]))
