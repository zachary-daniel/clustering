'''
This is a function for computing the BES ids that fall within a user prescribed range in normalized flux and Z from the magnetic axis.
User passes in two ranges of R and Z values to look between for data. first entry is lower bound, second is the upper bound
'''

import numpy as np
import sys
import scipy

#function for selecting ids that lie within a certain range of R and Z. This will be done with respect to the fifth channel in the top row of the BES
def flux_coordinate_sort_elm_free(bes_flux_coordinate_dict, bes_dict, R_range, Z_range, include_H_mode):
    acceptable_ids = [] #all ids that fall within the acceptable R and Z
    num_L = 0
    num_QH = 0
    num_WPQH = 0
    for id in bes_flux_coordinate_dict.keys():
        
        if include_H_mode == False and bes_dict[id]['label'] == 1: #if the id is in H mode don't include it
            continue
        current_R = bes_flux_coordinate_dict[id][4,0] #R and Z position for current id
        current_Z = bes_flux_coordinate_dict[id][4,1]
        in_R = current_R > R_range[0] and current_R < R_range[1] #check if the R and Z position lie in the correct range
        in_Z = current_Z > Z_range[0] and current_Z < Z_range[1]

        if in_R and in_Z: #append if they do
            acceptable_ids.append(id)
            if bes_dict[id]['label'] == 0:
                num_L+=1
            if bes_dict[id]['label'] == 2:
                num_QH+=1
            if bes_dict[id]['label'] == 3:
                num_WPQH+=1
    print(f'There are {num_L} L mode discharges')
    print(f'There are {num_QH} QH mode discharges')
    print(f'There are {num_WPQH} WPQH mode discharges')
    return acceptable_ids


def flux_coordinate_sort(bes_flux_coordinate_dict, R_range, Z_range):
    acceptable_ids = [] #all ids that fall within the acceptable R and Z
    for id in bes_flux_coordinate_dict.keys():
        current_R = bes_flux_coordinate_dict[id][4,0] #R and Z position for current id
        current_Z = bes_flux_coordinate_dict[id][4,1]
        in_R = current_R > R_range[0] and current_R < R_range[1] #check if the R and Z position lie in the correct range
        in_Z = current_Z > Z_range[0] and current_Z < Z_range[1] 
        if in_R and in_Z: #append the id if it is in the R and Z range
            acceptable_ids.append(id) 

    return acceptable_ids




if __name__ == '__main__':
    R_low = float(sys.argv[1]) #read in the range of acceptable R and Z values
    R_high = float(sys.argv[2])
    Z_low = float(sys.argv[3])
    Z_high = float(sys.argv[4])
    include_H_mode_arg = str(sys.argv[5])
    elm_bool = sys.argv[6] #bool for if using elm free or elmy data. Determines which dict will be loaded in
    if include_H_mode_arg.lower() == 'false':
        include_H_mode = False
    else:
        include_H_mode = True
    R_range = [R_low, R_high]
    Z_range = [Z_low, Z_high]
    if elm_bool.lower() == "false":
    #dictionary of the positions of the bes sensors in flux coordinates
        bes_flux_coordinate_dict = np.load('../../elm_free_bes_data/bes_sensors_flux_coordinates.npy', allow_pickle=True).item() #flux coordinates for elm free
        bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered.npy', allow_pickle=True).item()
        print(bes_dict['00001'].keys())
        acceptable_ids = flux_coordinate_sort_elm_free(bes_flux_coordinate_dict, bes_dict, R_range, Z_range, include_H_mode)
        save_path = f'../../computed_matrices/elm_free_matrices/ids_in_{R_range[0]}' #save based on the ids that are in a given range of R
    else:
        bes_flux_coordinate_dict = np.load('../type_I_elm_bes_data/type_I_elm_bes_flux_coordinates.npy', allow_pickle=True).item() #flux coordinates for type I elm
        acceptable_ids = flux_coordinate_sort(bes_flux_coordinate_dict, R_range, Z_range)
        save_path = f'../computed_matrices/type_I_elm_ids_in_{R_range[0]}.npy'
    
    print(f'There are {len(acceptable_ids)} ids in Rho from {R_range} and Z from {Z_range}') 
    np.save(save_path, acceptable_ids, allow_pickle=True) 
    print(f'Acceptable ids were saved to {save_path}!')