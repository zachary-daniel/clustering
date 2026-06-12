'''
This file is for computing the time lag cross correlation between all of the signals in a PCA dictionary. This is run from the bash file 'run_compute_TLCC.sh'
'''


import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import correlation_functions
import data_processing_functions as dpf



if __name__ == "__main__":
    batch_num = sys.argv[1] #batch number from the bash file. 
    print(batch_num) # Returns the current batch being operated on
    
    if os.path.exists(f'../../computed_matrices/elm_free_matrices/TLCC_dissimilarity_vals_{batch_num}.txt'): #If there is already a dissimilarity vals file for the batch, we remove it and write a new one
        os.remove(f'../../computed_matrices/elm_free_matrices/TLCC_dissimilarity_vals_{batch_num}.txt')

    pca_dict = np.load('../../computed_matrices/elm_free_matrices/filtered_pca_dict.npy', allow_pickle=True).item() # load in the pca signals dict. filtered for only portion when elming

    TLCC_obj = correlation_functions.matrix_correlation_builder(pca_dict, batch_num) #create a TLCC opbject

    test_results = TLCC_obj.compute_TLCC() #compute TLCC on the batch

    print(np.shape(test_results))
    with open(f'../../computed_matrices/elm_free_matrices/TLCC_dissimilarity_vals_{batch_num}.txt', 'a') as output: #write output to text file

        for result in test_results:
            output.write(f'{result}' + '\n')






