'''
This is a file for computing the Eros dissimilarity metric for multivariate time series.  
'''
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
import correlation_functions


if __name__ == "__main__":
    
    batch_num = sys.argv[1] #batch number from the bash file. 
    pedestal_loc = sys.argv[2] #pedestal top or foot from bash file
    print(pedestal_loc)
    print(batch_num) # Returns the current batch being ocleperated on
    
    if os.path.exists(f'../../computed_matrices/elm_free_matrices/DTW_dissimilarity_vals_{batch_num}.txt'): #If there is already a dissimilarity vals file for the batch, we remove it and write a new one
        os.remove(f'../../computed_matrices/elm_free_matrices/DTW_dissimilarity_vals_{batch_num}.txt')


    #eros_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered_new_start.npy', allow_pickle=True).item() #load in BES dict. the .item() call accessses the actual dict object. Use filtered object
    # bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_bes_dict_filtered_new_start_no_spike.npy', allow_pickle=True).item() 
    bes_path = '../../computed_matrices/elm_free_matrices/bes_signals_new_start_no_spike.h5'
    pedestal_dict = np.load(f'../../elm_free_bes_data/bes_sensors_in_pedestal_{pedestal_loc}.npy', allow_pickle = True).item()

    #pass in a pass to an h5 file instead of a dictionary
    TLCC_obj = correlation_functions.matrix_correlation_builder(bes_path, batch_num, bes_flux_coordinate_dict=pedestal_dict, diag = 'BES') #create an Eros object

    test_results = TLCC_obj.compute_pedestal_TLCC() #compute Eros on the batch

    print(np.shape(test_results))

    with open(f'../../computed_matrices/elm_free_matrices/TLCC_dissimilarity_vals_{batch_num}.txt', 'a') as output: #write output to text file
        for result in test_results:
            output.write(f'{result}' + '\n')