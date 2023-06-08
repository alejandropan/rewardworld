# Script to decompress raw data from spikeglx
from spikeglx import Reader
from pathlib import Path

def decomp_file(file_path):
    # Get the file path
    file_path = Path(file_path)
    # Decompress the file
    reader = Reader(file_path)
    reader.decompress_file(output_path=file_path.parent, save=True)
    