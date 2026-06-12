'''
This is a function for computing the BES ids that fall within a user prescribed range in normalized flux from the density pedestal. 
'''

import numpy as np
import sys
import random
import copy

#function for selecting ids that lie within a certain range of R and Z. This will be done with respect to the fifth channel in the top row of the BES
def flux_coordinate_sort_elm_free(bes_flux_coordinate_dict, bes_dict, pedestal_loc, pedestal_dict, include_H_mode, data_trim = False, shot_dict = None):
    '''
    This function is for finding events that occur within a certain distance in psi of the pedestal top or foot (depends on user input)

    bes_flux_coordiante_dict: dict. contains sensor position for each BES sensor
    bes_dict: dict. contains the label for each event
    pedestal_loc: str. 'top' or 'foot' depending on where the user would like to compare w.r.t
    pedestal_dict. dict. contains the position of the density, pressure and temperature pedestal in psi. Calculated from Thomson data
    include_H_mode: bool. If the user would like to include H mode discharges. 
    data_trim: bool. if the user would like a more balenced database, this variable will be used to limit the number of a given discharge type
    Hierarchical clustering performs worse when there is a small minority class. To combat this, we will implement a data balencing
    scheme where will will keep fewer events, but make sure they are from more shots to increase the database variance
    shot_dict: defaults to none. Dict where each key is an event id, and each value is the shot the id comes from
    returns:
    acceptable_ids_dict: dict. This will contain all of the sensors that lie within a desired range of positions w.r.t the pedestal.
    Each key will be an id, and each value will be an array of size (64,2). If a sensor is not in the desired range, the value of
    the entry in the array for psi at that point will be -1. This is an easy check since psi is p.s.d.
    '''
    
    acceptable_ids_dict = {} #all ids that fall within the acceptable R and Z. This will also include the the sensors and their positions that lie within the 
                        #proper range. 
    acceptable_ids = []
    num_L = 0
    num_QH = 0
    num_WPQH = 0
    num_missing = 0
    regime_to_ids = {'L': [], 'QH': [], 'WPQH': []} #will only be used if data_trim == True
    ids = list(bes_dict.keys())[:-1] #last key is a weight vector for computation not an id
    for id in ids:
        num_in_range = 0
        acceptable_ids_dict[id] = np.zeros(np.shape(bes_flux_coordinate_dict[id]))
        acceptable_ids_dict[id][:,0] = -1 #default value for sensors that are not in the proper range in psi is -1.
        if include_H_mode == False and bes_dict[id]['label'] == 1: #if the id is in H mode don't include it
            continue
        # current_psi = bes_flux_coordinate_dict[id][4,0] #R and Z position for current id
        #let's start with distance from pedestal top. This can be easily changed to foot or sym point
        try:
            pedestal_dict[id]['pe']['top'] #check if there is pedestal data
        except:
            # print(f'No pedestal data for {id}')
            num_missing+=1
            continue

        pedestal_foot = pedestal_dict[id]['pe']['foot']
        pedestal_top = pedestal_dict[id]['pe']['top']
        pedestal_width = (pedestal_foot - pedestal_top)/2 #width of the pedestal
        #FIXME need to change hardcoded L mode pedestal values
        if bes_dict[id]['label'] == 0:
            pedestal_width = .07
        pedestal_bound = 1.25*pedestal_width #arbitrary for right now. Can be changed as one wants
        num_sensors = len(bes_flux_coordinate_dict[id][:,0])
        # in_range = False
        #acceptable_num_sensors = 1 #number of sensors in range required to consider an id
        for sensor_pos in range(num_sensors):
            current_psi = bes_flux_coordinate_dict[id][sensor_pos, 0]
            #If we are looking for shots in the pedestal top, we only want to consider shots that are 
            #at the top or further. We don't want to look at shots that are further inboard
            #I want to add the option to have sensors that are in both the pedestal top and pedestal foot. My hope is that maybe with more sensor information my method
            #will make better predictions
            if bes_dict[id]['label'] == 0 and pedestal_loc == 'top':
                pedestal_dist = current_psi - .85 #fixed value of psi for the pedestal top if this is an L mode. 
            elif bes_dict[id]['label'] == 0 and pedestal_loc == 'foot':
                pedestal_dist = current_psi - .95 #fixed value of psi for the pedestal foot if this is an L mode. 
            else:
                pedestal_dist = current_psi - pedestal_dict[id]['pe'][pedestal_loc]
            
            #this will insure that we only take shots that are in the pedestal top or further towards the foot
            in_psi = pedestal_dist < 1*pedestal_bound and pedestal_dist > 0
            
            if in_psi: #if any of the sensors are within the acceptable range of the pedestal position given by the user, we'll keep the id  
                num_in_range += 1
                acceptable_ids_dict[id][sensor_pos,:] = bes_flux_coordinate_dict[id][sensor_pos,:]



        #we will determine if the id is in the top or 
        
        if num_in_range > 0: #append if they do
            acceptable_ids.append(id)
            if bes_dict[id]['label'] == 0:
                num_L+=1
                regime_to_ids['L'].append(id)
            if bes_dict[id]['label'] == 2:
                num_QH+=1
                regime_to_ids['QH'].append(id)
            if bes_dict[id]['label'] == 3:
                num_WPQH+=1
                regime_to_ids['WPQH'].append(id)
    print(f'number of ids with no data: {num_missing}')
    print(f'There are {num_L} L mode discharges')
    print(f'There are {num_QH} QH mode discharges')
    print(f'There are {num_WPQH} WPQH mode discharges')
    if data_trim and shot_dict is not None:
        limit_regime = int(2*min(num_L, num_QH, num_WPQH))
        final_acceptable_ids = []
        #we want to pop ids from the shots that have the most ids for each regime
        for regime,regime_ids in regime_to_ids.items():
            if len(regime_ids) > limit_regime:
                #group the ids by shot number. This would be a dict where each key is a shot, and each value is a list of all
                #of the ids that are from that shot
                shot_group = {}
                for regime_id in regime_ids:
                    shot = shot_dict[regime_id]
                    if shot not in shot_group.keys():
                        shot_group[shot] = []
                    shot_group[shot].append(regime_id)
                total_ids = len(regime_ids)   
                while total_ids > limit_regime:
                    largest_shot = max(shot_group, key= lambda x: len(shot_group[x]))
                    shot_group[largest_shot].pop() #remove an id from shot group
                    if len(shot_group[largest_shot]) == 0: #if there are no ids left, remove the largest shot
                        del shot_group[largest_shot]
                    total_ids -= 1
                final_regime_ids = [id for shot in shot_group.keys() for id in shot_group[shot]]
                #after we delete all of the ids that we don't need, append them to the final ids
                final_acceptable_ids.extend(final_regime_ids)
            else:
                #keep all the ids if the regime is already below the limit
                final_acceptable_ids.extend(regime_ids)

                
        acceptable_ids_dict = {id: acceptable_ids_dict[id] for id in final_acceptable_ids}

        return acceptable_ids_dict, final_acceptable_ids
    else:
        unique_ids = list(set(acceptable_ids))
        return acceptable_ids_dict, unique_ids


    

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

def weight_vector_update(acceptable_ids, Eros_dict):
    '''
    This function is for computing a new weight vector for the ids in the database
    '''
    num_sensors = len(Eros_dict[acceptable_ids[0]]['S'])
    S = np.zeros((num_sensors,len(acceptable_ids)))
    w = np.zeros(num_sensors)
    for k,id in enumerate(acceptable_ids):
        s = Eros_dict[id]['S']
        S[:,k] = s
    for i in range(num_sensors):
        w[i] = np.mean(S[i,:]) #average over all cols of S
    w = w/np.sum(w) #w will be normalized later in the process
    return w
def bad_shot_remover(shot_dict, bad_shots):
    '''
    For removing shots that have some kind of issue (noisey data or what have you)
    shot_dict: dict where each key is an id and each value is the shot corresponding to that id
    bad_shots: list of bad shots to remove

    returns:
    bad_ids: list of ids to remove from bes_dict
    '''
    bad_ids = []
    for id in shot_dict.keys():
        current_shot = shot_dict[id]
        if current_shot in bad_shots:
            bad_ids.append(id)

    return bad_ids

def RMP_sort(threeD_coil_dict, starting_ids):
    '''
    This function is for determining and removing any ids that have significant RMP coil current. This may distort the turbulence that is seen by BES and certainly magnetics
    so we want to make sure that isn't present in the data

    threeD_coil_dict: dict. Contains the current amplitudes in the I coils and C coils
    ids: list of ids to be filtered. 
    '''
    ids = copy.copy(starting_ids) #make a copy and change those instead of the original object
    threshold = 2000 #if the I or C coil is above 2 kA we will remove the id
    coils = list(threeD_coil_dict[ids[0]].keys())
    to_remove = []
    for id in ids:
        coil_maxes = np.asarray([np.max(threeD_coil_dict[id][coil]) for coil in coils]) #max value in each coil for each event
        num_above_threshold = np.where(coil_maxes > threshold)[0] #where that coil goes above the set threshold value
        if len(num_above_threshold) > 0: #remove the ids that are above the threshold
            
            to_remove.append(id)
    

    print(f'{len(to_remove)} ids were removed')
    return to_remove

if __name__ == '__main__':

    pedestal_loc = (str(sys.argv[1])).lower() # 'top', 'foot', 'both

    include_H_mode_arg = str(sys.argv[2])
    elm_bool = sys.argv[3] #bool for if using elm free or elmy data. Determines which dict will be loaded in
    data_trim = sys.argv[4] #bool for checking if the user would like to trim the data to have a balenced dataset
    if include_H_mode_arg.lower() == 'false':
        include_H_mode = False
    else:
        include_H_mode = True
    if data_trim.lower() == 'false':
        data_trim = False
    else:
        data_trim = True


    if elm_bool.lower() == "false":
    #dictionary of the positions of the bes sensors in flux coordinates for 6 x 8 config
        bes_flux_coordinate_dict = np.load('../../elm_free_bes_data/bes_sensors_flux_coordinates.npy', allow_pickle=True).item()
        #FIXME commented out. Use dict with new BES start times
        #bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered.npy', allow_pickle=True).item()
        bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered_new_start.npy', allow_pickle=True).item()
        #FIXME This is the eros dict for the 6x8 configuration
        # bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_6_8.npy', allow_pickle=True).item()
        shot_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_shot_dict.npy', allow_pickle=True).item()
        pedestal_dict = np.load('../../elm_free_bes_data/all_pedestal_dict_elm_free.npy', allow_pickle=True).item()
        threeD_coil_dict = np.load('../../elm_free_bes_data/elm_free_3D_coil_dict.npy', allow_pickle=True).item()
        bad_shots = ['159531']
        all_ids = list(bes_dict.keys())[:-1]
        bad_ids_rmp = RMP_sort(threeD_coil_dict, all_ids) 
        bad_data_ids = bad_shot_remover(shot_dict, bad_shots)

        #remove ids with high rmp
        for bad_id in bad_ids_rmp:
            bes_dict.pop(bad_id)
        #remove ids with bad data
        for bad_data in bad_data_ids:
            bes_dict.pop(bad_data)

        acceptable_ids_dict, acceptable_ids = flux_coordinate_sort_elm_free(bes_flux_coordinate_dict, bes_dict, pedestal_loc, pedestal_dict, include_H_mode, data_trim=data_trim, shot_dict = shot_dict)
        Eros_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered_new_start.npy', allow_pickle=True).item()
        new_weight_vector = weight_vector_update(acceptable_ids, Eros_dict)

        
        
        save_path_ids = f'../../computed_matrices/elm_free_matrices/ids_in_{pedestal_loc}.npy' #save based on the ids that are in a given range of R
        save_path_dict = f'../../elm_free_bes_data/bes_sensors_in_pedestal_{pedestal_loc}.npy'
        save_path_weight_vector = f'../../computed_matrices/elm_free_matrices/weight_vector_{pedestal_loc}.npy'
        num_L = 0
        num_QH = 0
        num_WPQH = 0
        removed_ids = 0
        num_keys_old = len(acceptable_ids)
        for id in acceptable_ids:
            if bes_dict[id]['label'] == 0:
                num_L+=1
            if bes_dict[id]['label'] == 2:
                num_QH+=1
            if bes_dict[id]['label'] == 3:
                num_WPQH+=1
        print(f'L modes: {num_L}')
        print(f'QH modes: {num_QH}')
        print(f'WPQH modes: {num_WPQH}')
        num_keys_new = len(acceptable_ids)
        print(f'removed_ids = {removed_ids}')
        print(f'previous number of keys: {num_keys_old}')
        print(f'new number of keys: {num_keys_new}')
    else:
        pass
        # #FIXME. Elmy version
        # bes_flux_coordinate_dict = np.load('../type_I_elm_bes_data/type_I_elm_bes_flux_coordinates.npy', allow_pickle=True).item() #flux coordinates for type I elm
        # acceptable_ids = flux_coordinate_sort(bes_flux_coordinate_dict, R_range, Z_range)
        # save_path = f'../computed_matrices/type_I_elm_ids_in_{R_range[0]}_pedestal.npy'
    
    print(f'There are {len(acceptable_ids)} ids in {pedestal_loc}') 
    np.save(save_path_ids, acceptable_ids, allow_pickle=True) 
    print(f'Acceptable ids were saved to {save_path_ids}!')
    np.save(save_path_dict, acceptable_ids_dict , allow_pickle=True)
    print(f'Dict of acceptable sensor positions saved successfully!')
    np.save(save_path_weight_vector, new_weight_vector, allow_pickle=True)
    print(f'New weight vector for pedestal {pedestal_loc} saved!')
    