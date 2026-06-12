'''
This file is for creating a small version of the full elm free bes dict that can be used for testing functions
'''
import numpy as np

full_bes_dict = np.load('../../computed_matrices/elm_free_matrices/elm_free_bes_dict.npy', allow_pickle=True).item()

test_dict = {} #dictioary to be created

#ten random ids from the full bes dict
random_indices = np.random.randint(0, len(full_bes_dict.keys()), size = 10)

ids = list(full_bes_dict.keys())
test_ids = []
#add the random ids to a list
for k in random_indices:
    test_ids.append(ids[k])

#assign the test id to the random ids
for test_id in test_ids:
    test_dict[test_id] = full_bes_dict[test_id]
#save
save_path = '../../computed_matrices/elm_free_matrices/elm_free_test_dict.npy'
np.save(save_path, test_dict, allow_pickle=True)
print(f'ELM free test dict saved to {save_path}')