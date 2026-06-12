'''
This is a file for computing the Eros dissimilarity metric for multivariate time series.  
'''
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('..')
import correlation_functions_test
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

if __name__ == "__main__":
    
    batch_num = 1 #batch number from the bash file. 
    print(batch_num) # Returns the current batch being operated on
    

    eros_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_eros_dict_filtered.npy', allow_pickle=True).item() #load in BES dict. the .item() call accessses the actual dict object. Use filtered object


    bes_flux_coordinate_dict = np.load('../../ELM_free_data_analysis/bes_sensors_flux_coordinates.npy', allow_pickle=True).item()
    
    Eros_obj = correlation_functions_test.matrix_correlation_builder(eros_dict, batch_num, bes_flux_coordinate_dict=bes_flux_coordinate_dict) #create an Eros object

    test_results, pairs = Eros_obj.compute_Eros_flux_coordinate() #compute Eros on the batch
    
    dissimimlarity = test_results[0][0]
    channels_to_keep = test_results[0][1]
    channels_to_remove = test_results[0][2]

    id_1 = pairs[0][0]
    id_2 = pairs[0][1]
    plt.figure()
    plt.rcParams.update({'font.size': 16})
    plt.title('BES sensor layouts')
    plt.scatter(bes_flux_coordinate_dict[id_1][:,0][channels_to_keep], bes_flux_coordinate_dict[id_1][:,1][channels_to_keep], label = 'kept', s = 100, facecolors = 'none', edgecolors='r', zorder = 3)
    plt.scatter(bes_flux_coordinate_dict[id_1][:,0], bes_flux_coordinate_dict[id_1][:,1], label = 'id 1', s= 50, zorder = 2)
    plt.scatter(bes_flux_coordinate_dict[id_2][:,0], bes_flux_coordinate_dict[id_2][:,1], label = 'id 2', s = 50)
    plt.legend(fontsize = 12)
    plt.xlabel('Psi')
    plt.ylabel('Z dist. from mag. axis (m)')
    plt.tight_layout()
    plt.savefig('../../APS_figures/BES_layout.png', transparent = True)

    plt.show()
    # print(np.shape(test_results))
