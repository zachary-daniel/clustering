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
    print(batch_num) # Returns the current batch being operated on
    
    if os.path.exists(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt'): #If there is already a dissimilarity vals file for the batch, we remove it and write a new one
        os.remove(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt')

    
    eros_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered.npy', allow_pickle=True).item() #load in BES dict. the .item() call accessses the actual dict object. Use filtered object

    bes_flux_coordinate_dict = np.load('../../ELM_free_data_analysis/bes_sensors_flux_coordinates.npy', allow_pickle=True).item()
    
    Eros_obj = correlation_functions.matrix_correlation_builder(eros_dict, batch_num, bes_flux_coordinate_dict=bes_flux_coordinate_dict) #create an Eros object

    test_results = Eros_obj.compute_Eros_flux_coordinate() #compute Eros on the batch

    print(np.shape(test_results))

    with open(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt', 'a') as output: #write output to text file
        for result in test_results:
            output.write(f'{result}' + '\n')