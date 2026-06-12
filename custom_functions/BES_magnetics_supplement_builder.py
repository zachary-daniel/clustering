'''
This is a file for building a an hdf5 file that will contain the magnetics data for a list of shots that will be interpolated onto the 
BES timebase
'''
import numpy as np
import h5py
import time
import data_processing_functions as dpf


def magnetics_interp(magnetics_path, bes_times):
    '''
    This function is for interpolating the magnetics data onto the BES time base
    '''
    interp_data_dict = {} #this is where the upsampled magnetics data will be stored
    with h5py.File(magnetics_path, 'r') as mag_file:
        ids = list(mag_file.keys())
        for id in ids: 
            interp_data_dict[id] = {}
            bes_time = bes_times[id] #bes time signal
            for diagnostic in mag_file[id].keys():
                current_data = mag_file[id][diagnostic]['zdata'][()] #data for the current diagnostic
                interp_data_dict[id][diagnostic] = {}
                #check if there was any significant data for the event. Write 'NA'
                #if there is no data
                if len(current_data) < 100:
                    interp_data_dict[id][diagnostic]['zdata'] = b'NA'
                    continue
                current_time = mag_file[id][diagnostic]['xdata'][()] #time base for the magnetics
                #interpolate the magnetics data onto the bes time base
                interp_data = np.interp(bes_time, current_time, current_data)
                interp_data_dict[id][diagnostic]['zdata'] = interp_data #only store the zdata
    
    return interp_data_dict

if __name__ == '__main__':
    magnetics_path = '../elm_free_magnetics_data/type_I_elm_magnetics.h5'
    bes_path = '../../CETOP/BES_DIIID/labeled_elm_events.hdf5'
    save_path = '../elm_free_magnetics_data/type_I_elm_magnetics_bes_timebase.h5'
    with h5py.File(bes_path, 'r') as bes_file:
        # bes_signals = dpf.get_datas(bes_file, 'signals') #signals
        bes_times = dpf.get_datas(bes_file,'time') #times
    #since this file is already in the correct structure for the keys and where the data is, all we have to do is interpolate the data onto 
    #the bes time base
    interpolated_magnetics_dict = magnetics_interp(magnetics_path, bes_times)
    #Write magnetics to h5 file
    with h5py.File(save_path, 'w') as mag_file:
        for id in interpolated_magnetics_dict.keys():
            id_group = mag_file.create_group(id)
            for diagnostic in interpolated_magnetics_dict[id].keys():
                diagnostic_group = id_group.create_group(diagnostic)
                diagnostic_group.create_dataset('zdata', data = interpolated_magnetics_dict[id][diagnostic]['zdata'])