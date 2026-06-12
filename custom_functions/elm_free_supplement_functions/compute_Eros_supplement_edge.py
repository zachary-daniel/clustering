'''
This is a file for computing the Eros dissimilarity metric for multivariate time series.  
'''
import numpy as np
import os
import sys
import copy
import matplotlib.pyplot as plt
sys.path.append('..')
import correlation_functions


if __name__ == "__main__":
    
    batch_num = sys.argv[1] #batch number from the bash file. 
    use_all = sys.argv[2]
    if use_all.lower() == 'true':
        use_all = True
    else:
        use_all = False
    print(batch_num) # Returns the current batch being ocleperated on
    
    if os.path.exists(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt'): #If there is already a dissimilarity vals file for the batch, we remove it and write a new one
        os.remove(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt')


    #eros_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered_new_start.npy', allow_pickle=True).item() #load in BES dict. the .item() call accessses the actual dict object. Use filtered object
    if use_all == True:
        eros_dict = np.load('../../computed_matrices/elm_free_matrices/Eros_dict_edge_all_sensors.npy', allow_pickle=True).item() 
    else:
        eros_dict = np.load('../../computed_matrices/elm_free_matrices/Eros_dict_edge.npy', allow_pickle=True).item() 

    Eros_obj = correlation_functions.matrix_correlation_builder(eros_dict, batch_num, diag = 'BES') #create an Eros object

    test_results = Eros_obj.compute_Eros() #compute Eros on the batch

    # print(np.shape(test_results))


    with open(f'../../computed_matrices/elm_free_matrices/Eros_dissimilarity_vals_{batch_num}.txt', 'a') as output: #write output to text file
        for result in test_results:
            output.write(f'{result}' + '\n')