'''
This file computes the covaraince matrix, and the svd of the covariance matrix for each signal in the BES dataset. This file will save a new dictionary that can be used 
for future calculations
'''
import numpy as np
import h5py
import data_processing_functions as dpf
import scipy
from sklearn.preprocessing import StandardScaler

BES_path = '../../CETOP/BES_DIIID/labeled_elm_events.hdf5' #path to BES data
with h5py.File(BES_path, 'r') as bes_file:
    bes_signals = dpf.get_datas(bes_file, 'signals') #load in the BES signals 
    bes_labels = dpf.get_datas(bes_file, 'labels') # 1 if ELM event, 0 otherwise
    bes_times = dpf.get_datas(bes_file, 'time')

#We're going to loop through each of the signals in the array, compute the covaraince matrix, and take the SVD of the covariance matrix
bes_dict = {}

def bandpass_filter(bes_signals, lowcut, highcut, fs, order = 10):
    ''''
    function for highpass filtering the bes signals
    bes_signals: dict of signals. each key is an id. Values are 64xT np arrays
    highcut: start of pass band in Hz
    fs: sampling rate in Hz
    order: filter order
    '''
    filtered_bes_dict = {}
    sos = scipy.signal.butter(order, [lowcut,highcut], fs=fs, btype = 'bandpass', output = 'sos')
    scalar = StandardScaler
    for key in bes_signals.keys():
        current_ts = bes_signals[key]
        scaled_data = scalar.fit_transform(current_ts.T).T #expects columns as features. BES data has rows as features
        #filter over the time axis
        filtered_bes_dict[key] = scipy.signal.sosfiltfilt(sos, scaled_data, axis=1)
    return filtered_bes_dict
#sampling frequency in hertz
fs = 1/((bes_times['00000'][1] - bes_times['00000'][0])/1000)
lowcut = 2500 #Hertz
highcut = 150000
filtered_signals = bandpass_filter(bes_signals, lowcut, highcut, fs)

num_sensors = len(filtered_signals['00000'][:,0])
S = np.zeros(( num_sensors, len(filtered_signals.keys()) )) #matrix of all singular values for each sensor. num_sensors x num events
for i,key in enumerate(filtered_signals.keys()):

    current_ts = filtered_signals[key] #get time series of BES data
    mask = np.where(bes_labels[key] == 1)[0] #only select where the signal is elming
    filtered_ts = current_ts[:,mask]
    cov_mat = np.cov(filtered_ts) #covariance matrix of current time series
    s, VT = np.linalg.svd(cov_mat)[1::] #get just the singular values and right singular vectors
    S[:,i] = s #update matrix of singular values
    bes_dict[key] = {'S': s, 'V': VT.T} #I actually don't need the original time series in here. Just the singular values and singular vectors

#compute the weights for each of the signals

w = np.zeros(num_sensors) # initialize empty weight vector. should be as many weights as there are sensors
for i in range(num_sensors):
    w[i] = np.mean(S[i,:]) #average over all cols of S

w = w/np.sum(w) #normalize the weight vector

bes_dict['weight_vector'] = w

print(f'w is of length {len(w)}. Sum of all entries of w is {np.sum(w)}')
np.save('../computed_matrices/filtered_bes_dict.npy', bes_dict, allow_pickle= True)
print('BES dictionary successfully saved!')