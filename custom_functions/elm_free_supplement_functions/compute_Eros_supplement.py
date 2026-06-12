'''
This is a file for computing the Eros dissimilarity metric for multivariate time series.  
'''
import numpy as np
import os
import sys
sys.path.append('..')
import correlation_functions


if __name__ == "__main__":
    
    batch_num = sys.argv[1] #batch number from the bash file. 
    print(batch_num) # Returns the current batch being operated on
    
    if os.path.exists(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt'): #If there is already a dissimilarity vals file for the batch, we remove it and write a new one
        os.remove(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt')

    Eros_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered_new_start_no_spike.npy', allow_pickle=True).item() #load in BES dict. the .item() call accessses the actual dict object
    
    
    Eros_obj = correlation_functions.matrix_correlation_builder(Eros_dict, batch_num) #create an Eros opbject

    test_results = Eros_obj.compute_Eros() #compute Eros on the batch
    print(np.shape(test_results))
    with open(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt', 'a') as output: #write output to text file
        for result in test_results:
            output.write(f'{result}' + '\n')