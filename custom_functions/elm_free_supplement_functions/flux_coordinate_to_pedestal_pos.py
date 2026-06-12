'''
This file is for converting the position of the BES sensor, which is the normalized flux, to position
w.r.t the pedestal
'''
import numpy as np
import matplotlib.pyplot as plt

def flux_coordinate_to_pedestal(bes_flux_coordinate_dict, pedestal_dict, pedestal_loc):
    '''
    This function will convert the bes sensors in flux coordinates to be w.r.t the pedestal position
    bes_flux_coordinate_dict: dict. each key is an event id and nested in that is a 64x2 array of psi, Z positions where Z is w.r.t the 
    magnetic axis, and psi is normalized flux

    pedestal_dict: dict. each key is an event id and each key nested in the top dict is 'top', or 'foot'. Corresponding to that is the location of the 
    pedestal top or foot in normalized flux

    pedestal_loc: str. 'top' or 'foot' depending on what the user would like the sensor position to be w.r.t
    returns:
    shifted_dict: dict. same structure as 'bes_flux_coordinate_dict'. Just the positions of the BES sensors are shifted to be w.r.t the pedestal
    '''

    shifted_dict = {}
    for id in bes_flux_coordinate_dict.keys():
        try:
            pedestal_dict[id]['top'] #check if there is pedestal data for the given id
        except:
            print('No data for this id')
            continue
        pedestal_pos = pedestal_dict[id][pedestal_loc]
        
        shifted_pos = np.zeros(np.shape(bes_flux_coordinate_dict[id]))
        shifted_rho = bes_flux_coordinate_dict[id][:,0] - pedestal_pos
        shifted_pos[:,0] = shifted_rho
        shifted_pos[:,1] = bes_flux_coordinate_dict[id][:,1]
        shifted_dict[id] = shifted_pos

    return shifted_dict
        

if __name__ == '__main__':
    pedestal_locs = ['top', 'foot']
    pedestal_dict = np.load('../../elm_free_bes_data/pedestal_dict_elm_free.npy', allow_pickle=True).item()
    bes_flux_coordinate_dict = np.load('../../elm_free_bes_data/bes_sensors_flux_coordinates.npy', allow_pickle=True).item()
    for pedestal_loc in pedestal_locs:
        shifted_coords = flux_coordinate_to_pedestal(bes_flux_coordinate_dict, pedestal_dict, pedestal_loc)
        save_path = f'../../ELM_free_data_analysis/shifted_coords_{pedestal_loc}.npy' 
        np.save(save_path, shifted_coords, allow_pickle=True)
        print(f'Shifted coordinate dict successfully saved to {save_path}!') 