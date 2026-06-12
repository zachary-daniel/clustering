'''
This script is for computing an Eros dict from the data that is determined to be in the edge region of a discharge. 

'''
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler

def compute_Eros_dict(acceptable_ids_dict, acceptable_ids, bes_dict, num_sensors):
    '''
    This function is for computing the Eros dict for the sensors that are in an edge region of the device
    acceptable_ids_dict: dict. each key is an id, and each value is a 64x2 array corresponding to the sensor positions for the BES sensors. If a sensor is not in the correct range then 
    it's radial coordinate is set to -1
    acceptable_ids: list. all of the acceptable ids
    bes_dict: dict containing filtered BES data 
    '''
    Eros_dict = {}
    S = np.zeros((num_sensors, len(acceptable_ids)))
    for idx,id in enumerate(acceptable_ids):
        Eros_dict[id] = {}
        bes_data = bes_dict[id]['signals']
        acceptable_sensor_pos = np.where(acceptable_ids_dict[id][:,0] != -1)
        edge_bes_data = bes_data[acceptable_sensor_pos]
        if num_sensors == 64: #if we use all sensors select all sensors from the BES signals
            edge_bes_data = bes_data
        scaler = StandardScaler()
        edge_bes_data_norm = scaler.fit_transform(edge_bes_data.T).T
        cov_mat = np.cov(edge_bes_data_norm)
        U,s,Vt = np.linalg.svd(cov_mat)
        S[:,idx] = s
        Eros_dict[id]['V'] = Vt.T
        Eros_dict[id]['S'] = s
    weight_vector = np.mean(S, axis = 1)
    weight_vector = weight_vector/np.sum(weight_vector)
    Eros_dict['weight_vector'] = weight_vector
    return Eros_dict


if __name__ == '__main__':
    num_sensors = int(sys.argv[1])
    use_all = sys.argv[2]
    if use_all.lower() == 'true':
        use_all = True
        Eros_dict_save_path = '../../computed_matrices/elm_free_matrices/Eros_dict_edge_all_sensors.npy'
        num_sensors = 64
    else:
        use_all = False
        Eros_dict_save_path = '../../computed_matrices/elm_free_matrices/Eros_dict_edge.npy'
    acceptable_ids_dict = np.load('../../computed_matrices/elm_free_matrices/bes_sensors_in_edge.npy', allow_pickle=True).item()
    acceptable_ids = np.load('../../computed_matrices/elm_free_matrices/ids_in_edge.npy', allow_pickle=True)
    bes_path = '../../computed_matrices/elm_free_matrices/elm_free_bes_dict_filtered_new_start_no_spike.npy'
    bes_dict = np.load(bes_path, allow_pickle=True).item()
    Eros_dict = compute_Eros_dict(acceptable_ids_dict, acceptable_ids, bes_dict, num_sensors)

    np.save(Eros_dict_save_path, Eros_dict, allow_pickle=True)
    print('Eros dict for the edge saved')