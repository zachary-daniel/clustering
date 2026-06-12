import numpy as np
import os

bes_path = '../../computed_matrices/elm_free_matrices/elm_free_bes_dict.npy'
bes_dict = np.load(bes_path, allow_pickle=True).item()

shots_total = np.zeros(len(bes_dict.keys())) #non_unique dictionary of shot numbers for each event in the bes_supplement dict

for i,id in enumerate(bes_dict.keys()): #cycle over all ids
    shot_num = bes_dict[id]['shot']
    shots_total[i] = shot_num

shot_nums_unique = np.unique(shots_total) #unique array of shot numbers 

if os.path.exists('shots_nums_supplement.txt'):
    os.remove('shots_nums_supplement.txt')

with open('shot_nums_supplement.txt', 'w') as file: #write the shot nums to an output file
    for shot in shot_nums_unique:
        file.write(str(shot)[:-2] + '\n')
