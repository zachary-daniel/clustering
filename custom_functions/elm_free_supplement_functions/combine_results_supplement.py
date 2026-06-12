'''
This file will recombine results from either DTW, TLED, or TLCC. All results should be saved in the 'computed_matrices' folder, and will be saved
with the following naming scheme: '{correlation metric}_dissimilarity_vals_{batch number}.txt'
correlation metrics: TLCC, TLED, DTW
'''
import sys
import argparse
sys.path.append('..')
from correlation_functions import combine_batches_elm_free
import data_processing_functions as dpf
import numpy as np
from tqdm import tqdm
from dtaidistance.similarity import distance_to_similarity
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("metric", type=str)
    # metric = sys.argv[1] #metric used (DTW, TLCC, Eros etc.)
    parser.add_argument('num_batches', type = int)
    #num_batches = int(sys.argv[2]) #total number of batches
    parser.add_argument('--Rho_low', default=None)
    parser.add_argument('--magnetics_bool', type = str, default='False')
    #Rho_low = float(sys.argv[3]) 

    args = parser.parse_args()
    metric = args.metric
    num_batches = args.num_batches
    Rho_low = args.Rho_low
    magnetics_bool = args.magnetics_bool

    if magnetics_bool.lower() == 'false':
        magnetics_bool = False
    else:
        magnetics_bool = True

    combine_batches_elm_free(metric, num_batches, folder_prefix='elm_free_matrices') #This script combines all of the batches into a single text file. Add the elm free flag to correct the save directory
    if Rho_low == None:
        pca_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_pca_dict.npy', allow_pickle=True).item() # load in the pca signals dict
        event_ids = list(pca_dict.keys())
    else:
        event_ids = list(np.load(f'../../computed_matrices/elm_free_matrices/ids_in_{Rho_low}.npy'))
        

    n_events = len(event_ids)
    pairs = [(event_ids[i], event_ids[j]) for i in range(n_events) for j in range(i+1, n_events)] #create the list list of tuples of all possible pairs

    dissimilarity_matrix = np.zeros((n_events, n_events)) # create a matrix of zeros of the dim n_events x n_events
    dissimilarity_vals = []
    
    with open(f'../../computed_matrices/elm_free_matrices/{metric}_dissimilarity_vals_total_output.txt', 'r') as results: #open the output file in read mode
        
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
    if metric != 'Eros' and metric != 'DMD':
        matrix_file_name = metric + '_dissimilarity_matrix_.npy' #save name for the computed matric
    else:
        if Rho_low == None:
            matrix_file_name = metric + '_dissimilarity_matrix.npy' #save name for the computed matric. Eros doesn't consider the overlap of 2 signals so we don't include it in the name
        else:
            matrix_file_name = metric + "_R_" + str(Rho_low) + '_dissimilarity_matrix.npy' #add in the R lower bound to the save name
    if metric == 'DTW': #DTW is the only metric that returns a distance matrix which must be converted to dissimilarity. This will be done using the distance_to_similarity function
                        #from DTAI
        similarity_matrix = distance_to_similarity(dissimilarity_matrix) #convert to similarity
        dissimilarity_matrix = 1 - similarity_matrix #subtract to get dissimilarity
    if magnetics_bool:
        matrix_file_name = matrix_file_name[0:-4] + '_magnetics.npy'
    np.save(matrix_file_name, dissimilarity_matrix)
    print(f'{matrix_file_name} was successfully saved!')