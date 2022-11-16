import sys
sys.path.append('/Users/alexpan/Documents/PYTHON/int-brain-lab/iblapps/atlaselectrophysiology')
from ephys_atlas_gui import *
import os
from PyQt5 import QtTest
from pathlib import Path

def plot_alignment(probe, output_folder = '/Users/alexpan/Downloads/alignments'):
    app = QtWidgets.QApplication([])
    mainapp = MainWindow(offline=True)
    mainapp.data_status = False
    mainapp.folder_line.setText(str(probe))
    mainapp.prev_alignments, shank_options = mainapp.loaddata.get_info(probe)
    mainapp.populate_lists(shank_options, mainapp.shank_list, mainapp.shank_combobox)
    mainapp.populate_lists(mainapp.prev_alignments, mainapp.align_list, mainapp.align_combobox)
    mainapp.on_shank_selected(0)
    mainapp.data_button_pressed()
    mainapp.filter_unit_pressed('IBL good')
    mainapp.plot_slice(mainapp.slice_data, 'label')
    mainapp.show()
    QtTest.QTest.qWait(5000)
    app.primaryScreen().grabWindow(0).save(output_folder + probe[-19:-11] + '.png')
    sys.exit()
    #QtCore.QTimer.singleShot(3000, lambda:app.primaryScreen().grabWindow(0).save(output_folder + probe[-19:-11] + '.png'))



SESSIONS = \
['/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-19/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-28/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_48/2022-06-27/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-14/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-15/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-16/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-18/002',  
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-19/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-27/003', 
'/Volumes/witten/Alex/Data/Subjects/dop_49/2022-06-20/001', 
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-11/001',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-10/002', 
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-09/003',
'/Volumes/witten/Alex/Data/Subjects/dop_47/2022-06-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-12/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-13/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-14/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-16/003',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-17/001',
'/Volumes/witten/Alex/Data/Subjects/dop_50/2022-09-18/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-21/002',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-22/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-23/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-26/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-27/001',
'/Volumes/witten/Alex/Data/Subjects/dop_51/2022-09-28/001',
'/Volumes/witten/Alex/Data/Subjects/dop_52/2022-10-02/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-07/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-05/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-04/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-03/001',
'/Volumes/witten/Alex/Data/Subjects/dop_53/2022-10-02/001']

def run_screenshots():
    output = '/Users/alexpan/Downloads/alignments'
    errors=[]
    for ses in SESSIONS:
        output_folder = output+ses[-22:]
        os.makedirs(output_folder)
        for i in np.arange(4):
            probe = ses + '/alf/probe0' + str(i) + '/pykilosort'
            if Path(probe).exists():
                print(probe)
                try:
                    plot_alignment(probe, output_folder)
                except:    
                    errors.append(probe)
                


if __name__ == "__main__":
    probe = sys.argv[1]    
    output = '/Users/alexpan/Downloads/alignments'
    output_folder = output + probe[34:-23]
    if Path(output_folder).exists()==False:
        os.makedirs(output_folder)
    plot_alignment(probe, output_folder = output_folder)