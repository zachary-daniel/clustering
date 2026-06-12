'''
This file is for computing pca on the bes supplement data in the elm_free_bes_data folder. This data corresponds to wide pedestal QH, QH, H and L mode discharges. The data is also structured
in a slightly different way than in the bes file for type I ELMs. The sutructure is as follows

- Shot
    -Event 1
        -time
        -signals (64,t)
        -labels
            - 0: L mode, 1: H mode, 2: QH mode, 3: wide pedestal QH
    event 2...
etc.

This file will also create a dictionary that is suitable for the Eros clustering metric that is implemented in correlation_functions.py. Two birds one python file
'''


import numpy as np
from sklearn.decomposition import PCA
import h5py
from tqdm import tqdm
import time
import copy
import scipy
import multiprocessing
from sklearn.preprocessing import StandardScaler

'''
Function to reorganize data into a dictionary. each key will be an event id, and the values will be the bes data, label (QH, L mode etc.), time.
Wow this is annoying. Okay so the event ids in the supplement files actually aren't unique. Go figure. So we need to rewrite any event ids that are not unique to be unique
'''


def init_worker(data_dict): #for use with multiprocessing
    global _shared_bes_dict_
    _shared_bes_dict_ = data_dict


def bes_reorganize(file, new_event_ids):
    bes_dict = {} #need a global dict for multiprocessing
    total_ids = 0 #total number of ids in the shot
    skipped_ids = 0 #number of ids we skip
    for shot in tqdm(file.keys()):
        for event_id in file[shot]:
            total_ids+=1
            try:
                bes_dict[new_event_ids[total_ids]] = {'time': file[shot][event_id]['time'][()], 'signals':
                 np.array(file[shot][event_id]['signals'][()], dtype=np.float32), 'label': file[shot][event_id]['labels'][()][0], 'shot': shot}
            except:
                skipped_ids +=1
                
    print(f'total ids: {total_ids}, skipped ids: {skipped_ids}, len bes dict: {len(bes_dict.keys())}. If this is false, we messed up: {len(bes_dict.keys()) == (total_ids - skipped_ids)}')
    return bes_dict

'''
Function to compute pca on dictionary of BES data
bes_dict: dictionary organized as per the above function
'''

def pca_on_dict(bes_dict,n_components = 3):
    pca_dict = {}
    pca_dict_ds = {}
     #number by which we will reduce the size of the signal. Should speed up calculations
    pca = PCA(n_components)
    pc_keys = [f'PC{i}' for i in range(n_components)] #keys for each principal component
    for event_id in tqdm(bes_dict.keys()):
        time_series = bes_dict[event_id]['signals'].T #need to transpose for the way sklearn expects data
        time = bes_dict[event_id]['time'] #time vector
        principal_components = pca.fit_transform(time_series) #compute pca on the time series
        variance_captured = pca.explained_variance_ratio_ #total variance captured by the pca
        pca_dict[event_id] = {'time': time, 'variance': variance_captured}
        pca_dict_ds[event_id] = {'time': time, 'variance': variance_captured}
        for k in range(n_components):
            pca_time_series = principal_components[:,k]
            skip = 20
            if len(pca_time_series) > 100000:
                skip = 60
            down_sampled_signal = pca_time_series[::skip]#downsample dict by a factor of
            pca_dict_ds[event_id][pc_keys[k]] = down_sampled_signal
            pca_dict[event_id][pc_keys[k]] = principal_components[:,k]
    return pca_dict, pca_dict_ds
    
def highpass_filter(id):
    ''''
    function for highpass filtering the bes signals
    id: id being processed from the BES dict
    '''
    order = 10
    lowcut = 2500 #low cut for the bandpass filter
    highcut = 150000 #high cut for the bandpass filter
    try:
        fs = 1/((_shared_bes_dict_[id]['time'][1] - _shared_bes_dict_[id]['time'][0])/1000)
    except:
        raise Exception(f'id {id} correpsonds to a time series with length {len(_shared_bes_dict_[id]['time'])}')
    sos = scipy.signal.butter(order, [lowcut, highcut], fs = fs, btype = 'band', output = 'sos')

    #demean and standardize the data
    scalar = StandardScaler()
    current_ts = _shared_bes_dict_[id]['signals']
    scaled_data = scalar.fit_transform(current_ts.T).T
    #filter over the time axis
    #nested dict for compatability with exisiting code
    
    filtered_signal = scipy.signal.sosfiltfilt(sos, current_ts, axis=1)
    return filtered_signal

def remove_spikes(bes_dict, threshold = 10):
    '''
    This function is for identifying and removing spikes/density filaments from the BES data
    bes_dict: dict. contains BES signals
    threshold: number of standard deviations above which we consider a datapoint to be a spike
    '''
    cut_thresh = 20
    left_cut_dict = {} #keep track of the portions of the sensor signals that are removed. We will need to cut everything down
    right_cut_dict = {}             #to the length of the shortest sensor
    cut_bes_dict = copy.deepcopy(bes_dict)
    num_ids = len(list(bes_dict.keys()))
    for count,id in enumerate(bes_dict.keys()):
        print(f'id {count} of {num_ids}')
        left_cut_dict[id] = []
        right_cut_dict[id] = []
        num_sensors = len(bes_dict[id]['signals'][:,0])
        for k in range(num_sensors):
            bes_signal = bes_dict[id]['signals'][k,:]
            spike_points = np.where(np.abs(bes_signal)>threshold*np.std(bes_signal))[0] #get the indices of the points where there are spikes
            if len(spike_points) !=0:
                left_cut = spike_points[0]
                right_cut = 0
                for i in range(len(spike_points)-1):
                    if spike_points[i+1] - spike_points[i] < 10: #check if we're iterating through points in the same spike
                        continue
                    else: #if we're not in the same spike and we have found a place to cut
                        right_cut = spike_points[i]
                        right_cut_index = min(right_cut+cut_thresh, len(bes_signal))
                        left_cut_index = max(left_cut - cut_thresh, 0)
                        left_cut_dict[id].append(left_cut_index)
                        right_cut_dict[id].append(right_cut_index)  
                        left_cut = spike_points[i+1]
                        right_cut = 0
                if right_cut == 0:
                    right_cut = spike_points[-1]
                    right_cut_index = min(right_cut+cut_thresh, len(bes_signal))
                    left_cut_index = max(left_cut - cut_thresh, 0)
                    left_cut_dict[id].append(left_cut_index)
                    right_cut_dict[id].append(right_cut_index)
            else:
                continue
        #now we need to recut and restack all of the data
        left_cut_dict[id] = sorted(left_cut_dict[id])
        right_cut_dict[id] = sorted(right_cut_dict[id])
        if len(left_cut_dict[id]) == 0: #check to see if there were no cuts made. skip these events if so
            continue
        new_signal_list = []
        new_signal_list.append(bes_dict[id]['signals'][:,0:left_cut_dict[id][0]])
        for k in range(len(left_cut_dict[id])-1):
            left_cut = left_cut_dict[id][k+1]
            right_cut = right_cut_dict[id][k]
            new_signal_list.append(bes_dict[id]['signals'][:,right_cut:left_cut])
        right_cut = right_cut_dict[id][-1]
        new_signal_list.append( bes_dict[id]['signals'][:,right_cut::])
        new_signal = np.hstack(new_signal_list)
        cut_bes_dict[id]['signals'] = new_signal
    return cut_bes_dict

#This function is preprocessing the data from bes so it can be used for the Eros metric
def Eros_preprocess(bes_dict):
    eros_dict = {} #this is where the matrices from the svd will be stored
    bes_keys = list(bes_dict.keys())
    num_sensors = len(bes_dict[bes_keys[0]]['signals'][:,0]) #number of bes channels

    S = np.zeros((num_sensors, len(bes_keys) )) #S matrix that will store all of the singular values for each event
    for i,key in enumerate(bes_keys):
        time_series = bes_dict[key]['signals'] #grab current time series 
        label = bes_dict[key]['label']
        cov_mat = np.cov(time_series) #covaraince matrix
        s, VT = np.linalg.svd(cov_mat)[1::]
        S[:,i] = s #assign current col of S with singular values
        eros_dict[key] = {'S': s, 'V': VT.T, 'label': label} #right singular vectors
    w = np.zeros(num_sensors) #empty weight vector
    for i in range(num_sensors):
        w[i] = np.mean(S[i,:]) #average over all cols of S
    w = w/np.sum(w) #normalize w
    eros_dict['weight_vector'] = w
    return eros_dict
'''
This is a method to compute DMD on the bes data 
'''
def DMD_preprocess(bes_dict):
    dmd_dict = {} #this is where the matrices from the svd will be stored
    bes_keys = list(bes_dict.keys())
    num_sensors = len(bes_dict[bes_keys[0]]['signals'][:,0]) #number of bes channels
    S = np.zeros((num_sensors, len(bes_dict.keys()))) #S matrix that will store all of the singular values for each event
    for i,key in enumerate(bes_dict.keys()):
        time_series = bes_dict[key]['signals'] #grab current time series 
        X = time_series[:,0:-1] # time series from 0 through end - 1
        X_prime = time_series[:,1::] #time series from 1 through end
        dmd_mat =  X_prime@np.linalg.pinv(X) #dmd matrix
        s, VT = np.linalg.svd(dmd_mat)[1::] #Take SVD of dmd matrix
        S[:,i] = s #assign current col of S with singular values
        dmd_dict[key] = {'S': s, 'V': VT.T, 'A': dmd_mat} #right singular vectors, singular values, and the A matrix from DMD
    w = np.zeros(num_sensors) #empty weight vector
    for i in range(num_sensors):
        w[i] = np.mean(S[i,:]) #average over all cols of S
    w = w/np.sum(w) #normalize w
    dmd_dict['weight_vector'] = w
    return dmd_dict

if __name__ == '__main__':
    filter_spikes = True #bool to determine if a function will be run to remove density spikes from the data
    BES_supplement_path = '../../elm_free_bes_data/confinement_data.20240112.hdf5'
    new_start_times_dict = np.load('../../elm_free_bes_data/new_start_times_elm_free.npy', allow_pickle=True).item()
    # with h5py.File(BES_supplement_path, 'r') as file:
    #     #this will create new event ids for each of the events in the supplemental data file. The events in that file are not unique so we need unique ids. 
    #     new_event_ids = [] #list of new event ids
    #     current_id = 0 #base for the current id
    #     for shot in file.keys(): #cycle over all shots
    #         for id in file[shot]: #each event in the shot
    #             num_zeros = 5 - len(str(current_id)) #number of zeros to append to the front of the event id. 
    #             str_prefix = ''
    #             for k in range(num_zeros): #add zeros to front of string identifier
    #                 str_prefix += '0'
    #             new_event_id = str_prefix + str(current_id) #add zeros to the beginning of the current id
    #             new_event_ids.append(new_event_id) #add to list
    #             current_id+=1 #increment

        # bes_dict = bes_reorganize(file, new_event_ids)
    bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_bes_dict.npy', allow_pickle=True).item()
    bes_dict_new_start = copy.deepcopy(bes_dict)
    # bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_bes_dict.npy', allow_pickle=True).item()
    #since there are some times when the beam turns on and causes artifacts in the BES, we will remove these time slices
    yikes_ids = []
    ids = list(bes_dict.keys())
    print('beginning NBI slicing')
    for id in ids[0::]:
        new_start_index = np.argmin(np.abs(bes_dict[id]['time'] - new_start_times_dict[id]['new_start']))
        new_stop_index = np.argmin(np.abs(bes_dict[id]['time'] - new_start_times_dict[id]['new_stop']))
        # print(f'new start: {new_start_index}, new stop {new_stop_index}. Event of length {len(bes_dict[id]['time'][new_start_index:new_stop_index])}')
        bes_dict_new_start[id]['time'] = bes_dict[id]['time'][new_start_index:new_stop_index]
        bes_dict_new_start[id]['signals'] = bes_dict[id]['signals'][:, new_start_index:new_stop_index]
        if len(bes_dict_new_start[id]['time']) < 65:
            print(f'Event is too short: {id}')
            print(f'total length: {len(bes_dict[id]['time'])}. new start_index: {new_start_index}. new end index: {new_stop_index}')
            
            yikes_ids.append(id)
    print('done with NBI slicing')
    #filtered signals from BES
    print('Done with new start and end indices')
    #save paths
    filtered_bes_save_path = '../../computed_matrices/elm_free_matrices/elm_free_bes_dict_filtered.npy'
    bes_new_start_save_path = '../../computed_matrices/elm_free_matrices/elm_free_bes_dict_new_start.npy'
    filtered_new_start_save_path = '../../computed_matrices/elm_free_matrices/elm_free_bes_dict_filtered_new_start.npy'
    eros_save_path_filtered_new_start = '../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered_new_start.npy'
    print('Starting cutting')
    if filter_spikes:
        cut_bes_dict_new_start = remove_spikes(bes_dict_new_start, threshold = 10) #bes dict where spikes are removed
        del bes_dict_new_start
        bes_dict_new_start = copy.deepcopy(cut_bes_dict_new_start) #reassign variable name for future use
        #reassign save paths
        filtered_new_start_save_path = '../../computed_matrices/elm_free_matrices/elm_free_bes_dict_filtered_new_start_no_spike.npy'
        bes_new_start_save_path = '../../computed_matrices/elm_free_matrices/elm_free_bes_dict_new_start_no_spike.npy'
        eros_save_path_filtered_new_start = '../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered_new_start_no_spike.npy'
    print(f'ending cutting')
    #FIXME test dict is currently being loaded
    # bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_test_dict.npy', allow_pickle=True).item()
    #^^ Can uncomment for further testing
    processes = 75
    ids = list(bes_dict.keys())
    start = time.monotonic() #start time
    # with multiprocessing.Pool(processes=processes, initializer=init_worker, initargs=(bes_dict,)) as pool:

    #     filtered_signals = pool.map(highpass_filter, ids)
    print('starting filtering')
    with multiprocessing.Pool(processes=processes, initializer=init_worker, initargs=(bes_dict_new_start,)) as pool:

        filtered_signals_new_start = pool.map(highpass_filter, ids)
    print('filtering finished')
    stop = time.monotonic() - start
    filtered_dict = {}
    #create the filtered signals dictionary
    #need to uncomment to run the bes filtered signals
    # for idx,id in enumerate(ids):
    #     filtered_dict[id] = {}
    #     filtered_dict[id] = bes_dict[id]
    #     filtered_dict[id]['signals'] = np.array(filtered_signals[idx], dtype = np.float32)
    filtered_dict_new_start = {}
    for idx,id in enumerate(ids):
        filtered_dict_new_start[id] = {}
        filtered_dict_new_start[id] = bes_dict_new_start[id]
        filtered_dict_new_start[id]['signals'] = np.array(filtered_signals_new_start[idx], dtype = np.float32)
    print(f'This took {stop} seconds')

    
    # np.save(filtered_bes_save_path, filtered_dict, allow_pickle=True) #save the filtered dict
    # print(f'Bes dict was successfully saved to {filtered_bes_save_path}')

    # #save the bes dictionary 
    # bes_save_path = '../../computed_matrices/elm_free_matrices/elm_free_bes_dict.npy'
    # np.save(bes_save_path, bes_dict, allow_pickle=True)
    # print(f'Bes dict was successfully saved to {bes_save_path}')


    np.save(bes_new_start_save_path, bes_dict_new_start, allow_pickle=True)
    print('new start bes dict saved')


    np.save(filtered_new_start_save_path, filtered_dict_new_start, allow_pickle=True)
    print('new start bes dict filtered saved')
    # pca_dict, pca_dict_ds = pca_on_dict(bes_dict) # compute pca on the dict
    # pca_save_path = '../../computed_matrices/elm_free_matrices/elm_free_pca_dict.npy'
    # pca_ds_save_path = '../../computed_matrices/elm_free_matrices/elm_free_pca_dict_ds.npy'
    # np.save(pca_save_path, pca_dict, allow_pickle=True) #save the pca dict
    # np.save(pca_ds_save_path, pca_dict_ds, allow_pickle=True) #save the down sampled pca dict
    # print(f'pca dict was saved successfully to {pca_save_path}')
    
    # save the Eros dictionary
    filtered_Eros_dict_new_start = Eros_preprocess(filtered_dict_new_start) #eros preprocess. Save the filtered dict

    np.save(eros_save_path_filtered_new_start, filtered_Eros_dict_new_start, allow_pickle=True)
    print(f'eros dict was saved successfully to {eros_save_path_filtered_new_start}') 