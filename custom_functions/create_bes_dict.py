'''
This file computes the covaraince matrix, and the svd of the covariance matrix for each signal in the BES dataset. This file will save a new dictionary that can be used 
for future calculations
'''
import numpy as np
import h5py
import data_processing_functions as dpf
import scipy
import matplotlib.pyplot as plt

BES_path = '../../CETOP/BES_DIIID/labeled_elm_events.hdf5' #path to BES data
with h5py.File(BES_path, 'r') as bes_file:
    bes_signals = dpf.get_datas(bes_file, 'signals') #load in the BES signals 
    bes_labels = dpf.get_datas(bes_file, 'labels') # 1 if ELM event, 0 otherwise
    bes_times = dpf.get_datas(bes_file, 'time')

#We're going to loop through each of the signals in the array, compute the covaraince matrix, and take the SVD of the covariance matrix


def highpass_filter(bes_dict, fs, highcut, order = 10):
    sos = scipy.signal.butter(order, highcut, fs = fs, btype = 'highpass', output = 'sos')
    filtered_dict = {}
    for i,id in enumerate(bes_dict.keys()): #FIXME remove enumerate and if statement when done testing
        if i == 10:
            break
        current_ts = bes_dict[id]
        demeaned_ts = current_ts - np.mean(current_ts, axis = 1)[:,np.newaxis]
        filtered_dict[id] = {}
        #filter over the time axis
        #nested dict for compatability with exisiting code
    
        filtered_signal = scipy.signal.sosfiltfilt(sos, demeaned_ts, axis=1)
        filtered_dict[id]['signals'] = filtered_signal
    return filtered_dict



#sampling frequency in hertz
fs = 1/((bes_times['00000'][1] - bes_times['00000'][0])/1000)
highcut = 5 #Hertz
filtered_dict = highpass_filter(bes_signals, fs, highcut) #filter the bes dict
plt.figure()
plt.plot(filtered_dict['00000']['signals'][0,:])
plt.plot(bes_signals['00000'][0,:])
plt.show()


# bes_dict = {}
# num_sensors = len(bes_signals['00000'][:,0])
# S = np.zeros(( num_sensors, len(bes_signals.keys()) )) #matrix of all singular values for each sensor. num_sensors x num events
# for i,key in enumerate(bes_signals.keys()):

#     current_ts = filtered_dict[key]['signals'] #get time series of BES data
#     mask = np.where(bes_labels[key] == 1)[0] #only select where the signal is elming
#     filtered_ts = current_ts[:,mask]
#     cov_mat = np.cov(filtered_ts) #covariance matrix of current time series
#     s, VT = np.linalg.svd(cov_mat)[1::] #get just the singular values and right singular vectors
#     S[:,i] = s #update matrix of singular values
#     bes_dict[key] = {'S': s, 'V': VT.T} #I actually don't need the original time series in here. Just the singular values and singular vectors

# #compute the weights for each of the signals

# w = np.zeros(num_sensors) # initialize empty weight vector. should be as many weights as there are sensors
# for i in range(num_sensors):
#     w[i] = np.mean(S[i,:]) #average over all cols of S

# w = w/np.sum(w) #normalize the weight vector

# bes_dict['weight_vector'] = w

# print(f'w is of length {len(w)}. Sum of all entries of w is {np.sum(w)}')
# np.save('../computed_matrices/bes_dict.npy', bes_dict, allow_pickle= True)
# print('BES dictionary successfully saved!')
# filtered_bes_dict_save_path = '../computed_matrices/filtered_bes_type_I_elm.npy'
# np.save(filtered_bes_dict_save_path, filtered_dict, allow_pickle=True)
# print(f'filtered bes dict saved to {filtered_bes_dict_save_path}')