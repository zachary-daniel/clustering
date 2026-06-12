'''
This file is for computing the dynamic time warping between all of the signals in the elm free pca dict
'''
import numpy as np
import os
import sys
sys.path.append('..')
import correlation_functions 




if __name__ == "__main__":
    
    batch_num = sys.argv[1] #batch number from the bash file. 
    min_fraction = sys.argv[2] #overlap for the signals
    print(batch_num) # Returns the current batch being operated on
    
    if os.path.exists(f'../../computed_matrices/elm_free_matrices/DTW_dissimilarity_vals_{batch_num}.txt'): #If there is already a dissimilarity vals file for the batch, we remove it and write a new one
        os.remove(f'../../computed_matrices/elm_free_matrices/DTW_dissimilarity_vals_{batch_num}.txt')

    pca_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_pca_dict_ds.npy', allow_pickle=True).item() # load in the pca signals dict. filtered for only portion when elming 
    
    DTW_obj = correlation_functions.matrix_correlation_builder(pca_dict, batch_num, min_fraction = min_fraction) #create a TLCC object

    test_results = DTW_obj.compute_DTW() #compute DTW on the batch
    print(np.shape(test_results))
    with open(f'../../computed_matrices/elm_free_matrices/DTW_dissimilarity_vals_{batch_num}.txt', 'a') as output: #write output to text file

        for result in test_results:
            output.write(f'{result}' + '\n')

