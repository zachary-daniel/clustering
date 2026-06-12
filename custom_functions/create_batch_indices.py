import numpy as np
import math
import os
import copy
import sys
import h5py
import argparse
import shutil
import data_processing_functions as dpf


def shot_filter(pca_dict): # function for filtering out ids from shots that are in a reversed configuration
    supplemental_data_path = '../../CETOP/supplemented_data.hdf5'


    with h5py.File(supplemental_data_path, 'r') as data_file:
        plasma_current = dpf.get_datas(data_file, 'ip') #plasma current
        B_toroidal = dpf.get_datas(data_file, 'bt') #toroidal B field
        bes_labels = dpf.get_datas(data_file, 'labels') #labels for where an elm is occuring for each time series

    filtered_ids = [] #if a shot is in the standard configuration, keep it
    for id in B_toroidal.keys():
        if np.mean(B_toroidal[id]) < 0 and np.mean(plasma_current[id]) > 0: #if the average B field is negative and average current is positive, the shot is in standard configuration and                                                                 #should be kept
            filtered_ids.append(id)
    print(f'After filtering, there are now {len(filtered_ids)} ids instead of {len(B_toroidal.keys())} ids')
    filtered_dict = elm_filter(pca_dict, bes_labels) #filter the time series to only when an elm is occuring
    return filtered_ids, filtered_dict

def elm_filter(pca_dict, bes_labels):
    '''
    subselect the portion of the ELM signals where the elm is occuring. 
    pca_dict: dicitonary of signals to be filtered
    bes_labels: dictionary of labels for each time series. 1 corresponds to the signal ELMing and 0 means the signal is not ELMing
    '''
    for key in pca_dict.keys():
        mask = np.where(bes_labels[key] == 1) #all indices where an ELM is occuring
        current_ts = pca_dict[key]['PC1'] #current time series
        filtered_ts = current_ts[mask] #filter time series with mask
        pca_dict[key]['PC1'] = filtered_ts #reassign the key in the diciontary
    return pca_dict


#Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('num_batches', type=int)
parser.add_argument('--Rho_low', type=float, default=None)

args = parser.parse_args()
Rho_low = args.Rho_low
num_batches = args.num_batches

pca_dict = dpf.load_results('../computed_matrices/pca_results')
if Rho_low == None:

    event_ids = list(pca_dict.keys())
    filtered_ids_save_path = '../computed_matrices/filtered_ids.npy'
else:
    event_ids = np.load(f'../computed_matrices/type_I_elm_ids_in_{Rho_low}.npy', allow_pickle=True)
    filtered_ids_save_path = f'../computed_matrices/type_I_elm_flux_coordinates/{Rho_low}_filtered_ids.npy'




filtered_dict = copy.deepcopy(pca_dict) #make a copy of the pca dict. this object will be filtered for the portion of the shot where elms are occuring

shot_filtered_ids, filtered_dict = shot_filter(filtered_dict) #filter the event ids
#need to delete the keys that are in the filtered dict that are not in the full dict
for id in pca_dict.keys():
    if id not in shot_filtered_ids or id not in event_ids: del filtered_dict[id] #if the id is not in the filtered id list, delete it

print(f'This many keys in new dict  {len(filtered_dict.keys())}')
np.save('../computed_matrices/filtered_pca_dict.npy', filtered_dict, allow_pickle=True) #save the pca dict filtered for when the elm event is occuring. 
print('filtered dictionary saved')
n_events = len(filtered_dict.keys())
filtered_ids = list(filtered_dict.keys()) #number of keys in the filtered dict
pairs = [(filtered_ids[i], filtered_ids[j]) for i in range(n_events) for j in range(i + 1, n_events)]

np.save(filtered_ids_save_path, filtered_ids, allow_pickle=True)

print('filtered ids are saved!')
#total number of batches that will be created


batch_size = math.ceil(len(pairs)/num_batches) #round the number of batches to be an int. need to round up to get batch sizing correct
print(f'Number of pairs in a batch: {batch_size}')
#This will overwrite the current batches with new batches

#check if batch arrays folder exists, and remove it's contents if it does
if os.path.exists('batch_arrays'):
    shutil.rmtree('batch_arrays')

#create a batch_arrays folder to put the batch indices in
os.mkdir('batch_arrays')

for k in range((num_batches)):
    start = k*batch_size
    end = min( (k+1)*batch_size, len(pairs))
    batch = pairs[start:end]
    np.save(f'batch_arrays/batch_{k+1}', batch)
    print(f"Saved batch {k+1}/{num_batches}, shape: {np.array(batch).shape}")
