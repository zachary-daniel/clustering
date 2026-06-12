'''
This file will recombine results from either DTW, TLED, or TLCC. All results should be saved in the 'computed_matrices' folder, and will be saved
with the following naming scheme: '{correlation metric}_dissimilarity_vals_{batch number}.txt'
correlation metrics: TLCC, TLED, DTW
'''

from correlation_functions import combine_batches_elm
import data_processing_functions as dpf
import numpy as np
from tqdm import tqdm
import argparse
from dtaidistance.similarity import distance_to_similarity
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", type=str)

    parser.add_argument('num_batches', type = int)

    parser.add_argument('--Rho_low', type = float, default=None)

    parser.add_argument('--min_fraction', type = float, default = None)



    args = parser.parse_args()
    metric = args.metric
    num_batches = args.num_batches
    Rho_low = args.Rho_low
    min_fraction = args.min_fraction


    combine_batches_elm(metric, num_batches, folder_prefix='type_I_elm_flux_coordinates') #This script combines all of the batches into a single text file

    if Rho_low == None:
        pca_dict = np.load('../computed_matrices/pca_dict.npy', allow_pickle=True).item() # load in the pca signals dict
        event_ids = list(pca_dict.keys())
    else:
        event_ids = list(np.load(f'../computed_matrices/type_I_elm_flux_coordinates/{Rho_low}_filtered_ids.npy'))
        
    n_events = len(event_ids)
    pairs = [(event_ids[i], event_ids[j]) for i in range(n_events) for j in range(i+1, n_events)] #create the list list of tuples of all possible pairs
    print(f'There are {len(pairs)} pairs')
    dissimilarity_matrix = np.zeros((n_events, n_events)) # create a matrix of zeros of the dim n_events x n_events
    dissimilarity_vals = []

    if Rho_low == None:
        total_output_path = f'../computed_matrices/{metric}_dissimilarity_vals_total_output.txt'
    else:
        total_output_path = f'../computed_matrices/type_I_elm_flux_coordinates/{metric}_dissimilarity_vals_total_output.txt'
    with open(total_output_path, 'r') as results: #open the output file in read mode
        
        for line in results: #save all of the results to a list
            dissimilarity_vals.append(line)

    for index,pair in enumerate(tqdm(pairs)):
        event_id_1 = str(pair[0])
        event_id_2 = str(pair[1])

        #find the index in event ids for the current pair of events
        i = event_ids.index(event_id_1) #need these double zero indices to pull out the integer value of the position
        j = event_ids.index(event_id_2)


        dissimilarity_matrix[i,j] = dissimilarity_vals[index]
        dissimilarity_matrix[j,i] = dissimilarity_vals[index]

    if min_fraction == None:
        if Rho_low == None:
            matrix_file_name = metric + '_dissimilarity_matrix.npy'
        else:
            matrix_file_name = metric + "_R_" + str(Rho_low) + '_dissimilarity_matrix.npy'
    else:
        matrix_file_name = metric + '_dissimilarity_matrix_' + str(min_fraction) + '_overlap.npy' #save name for the computed matric

    if metric == 'DTW': #DTW is the only metric that returns a distance matrix which must be converted to dissimilarity. This will be done using the distance_to_similarity function
                        #from DTAI
        similarity_matrix = distance_to_similarity(dissimilarity_matrix) #convert to similarity
        dissimilarity_matrix = 1 - similarity_matrix #subtract to get dissimilarity


    np.save(matrix_file_name, dissimilarity_matrix)
    print(f'{matrix_file_name} was successfully saved!')