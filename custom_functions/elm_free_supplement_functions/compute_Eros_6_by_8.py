import numpy as np

def compute_Eros_6_8(bes_dict, shifted_dict, bes_coordinate_dict):
    '''
    This function is for computing the Eros dict for the BES sensor in the 6x8 configuration. It will also reconfigure the 
    sensor dict and return that. 
    bes_dict: dict containing labels, signals, and shot numbers
    shifted_dict: dict contianing which rows have been shifted 
    bes_coordinate_dict: dict of flux coordinates for each sensor
    '''
    Eros_dict = {}
    six_by_eight_coordinate_dict = {}
    ids = list(bes_dict.keys())
    num_sensors = len(bes_dict[ids[0]]['signals'][:,0]) - 16 #subtract off two rows of BES
    S = np.zeros((num_sensors, len(ids)))
    for idx,id in enumerate(ids):
        Eros_dict[id] = {}
        signals = bes_dict[id]['signals']
        coordinates = bes_coordinate_dict[id]
        if len(shifted_dict[id]) == 0:
            signals = signals[0:48,:]
            six_by_eight_coordinate_dict[id] = coordinates[0:48,:]
        else:
            start_index = shifted_dict[id][0] * 8
            stop_index = start_index+8
            print(start_index)
            if len(shifted_dict[id]) == 1:
                signals = np.vstack((signals[0:start_index,:], signals[stop_index::,:]))
                #Even if there is only one shifted row, we still need to remove the last row of the signals/coordinate dict for
                #6x8 compatability
                signals = signals[0:-8, :]
                six_by_eight_coordinate_dict[id] = np.vstack((coordinates[0:start_index,:], coordinates[stop_index::,:]))
                six_by_eight_coordinate_dict[id] = six_by_eight_coordinate_dict[id][0:-8,:]
            else:
                start_index_2 = shifted_dict[id][1]*8
                stop_index_2 = start_index_2 + 8
                signals = np.vstack((signals[0:start_index,:], signals[stop_index:start_index_2,:], signals[stop_index_2::,:]))
                six_by_eight_coordinate_dict[id] = np.vstack((coordinates[0:start_index,:], coordinates[stop_index:start_index_2,:],
                                                            coordinates[stop_index_2::,:]))
        cov_mat = np.cov(signals)
        U,s,Vt = np.linalg.svd(cov_mat)
        S[:,idx] = s
        Eros_dict[id]['V'] = Vt.T
        Eros_dict[id]['S'] = s
        Eros_dict[id]['label'] = bes_dict[id]['label']
    weight_vector = np.zeros(len(S[0,:]))
    weight_vector = np.mean(S, axis = 1)
    weight_vector = weight_vector/np.sum(weight_vector)
    Eros_dict['weight_vector'] = weight_vector
    return Eros_dict, six_by_eight_coordinate_dict

if __name__ == '__main__':
    bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_bes_dict_filtered_new_start_no_spike.npy', allow_pickle=True).item()
    
    shifted_dict = np.load('../../elm_free_bes_data/shifted_rows_elm_free.npy', allow_pickle=True).item()
    bes_coordinate_dict = np.load('../../elm_free_bes_data/bes_sensors_flux_coordinates.npy', allow_pickle=True).item()
    Eros_dict_6_8, bes_coordinate_dict_6_8 = compute_Eros_6_8(bes_dict, shifted_dict, bes_coordinate_dict)
    Eros_dict_6_8_save_path = '../../computed_matrices/elm_free_matrices/elm_free_eros_dict_6_8.npy'
    bes_coordinate_dict_6_8_save_path = '../../elm_free_bes_data/bes_sensors_fluc_coordinates_6_8.npy'
    np.save(Eros_dict_6_8_save_path, Eros_dict_6_8, allow_pickle=True)
    print(f'Eros dict saved to {Eros_dict_6_8_save_path}')
    np.save(bes_coordinate_dict_6_8_save_path, bes_coordinate_dict_6_8, allow_pickle=True)
    print(f'BES dict saved to {bes_coordinate_dict_6_8_save_path}')