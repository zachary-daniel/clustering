'''
This is a file for computing the Eros dissimilarity metric on the DMD matrices for each mode in the ELM free dataset
'''
import numpy as np
import os
import sys
sys.path.append('..')
import correlation_functions



if __name__ == "__main__":
    
    batch_num = sys.argv[1] #batch number from the bash file. 
    print(batch_num) # Returns the current batch being operated on
    
    if os.path.exists(f'../../computed_matrices/elm_free_matrices/DMD_dissimilarity_vals_{batch_num}.txt'): #If there is already a dissimilarity vals file for the batch, we remove it and write a new one
        os.remove(f'../../computed_matrices/elm_free_matrices/DMD_dissimilarity_vals_{batch_num}.txt')

    dmd_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_dmd_dict.npy', allow_pickle=True).item() #load in DMD dict. the .item() call accessses the actual dict object
    
    
    DMD_obj = correlation_functions.matrix_correlation_builder(dmd_dict, batch_num) #create a DMD object

    test_results = DMD_obj.compute_Eros_dmd() #compute DMD similarirty on dmd matrices
    print(np.shape(test_results))
    with open(f'../../computed_matrices/elm_free_matrices/DMD_dissimilarity_vals_{batch_num}.txt', 'a') as output: #write output to text file
        for result in test_results:
            output.write(f'{result}' + '\n')