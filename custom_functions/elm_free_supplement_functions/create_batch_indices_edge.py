import numpy as np
import argparse
import math
import os
import sys
sys.path.append('..')
import shutil

'''
This file is for creating batches for the elm free supplement. Pass in an argument if you would like to use the dict of indices that are within a prespecified range of R and Z
'''
parser = argparse.ArgumentParser()
parser.add_argument('num_batches', type=int)

args = parser.parse_args()
num_batches = args.num_batches



event_ids = np.load(f'../../computed_matrices/elm_free_matrices/ids_in_edge.npy')


print(f'This many keys in new dict  {len(event_ids)}')

n_events = len(event_ids)
pairs = [(event_ids[i], event_ids[j]) for i in range(n_events) for j in range(i + 1, n_events)]

np.save('../../computed_matrices/elm_free_matrices/elm_free_ids_edge.npy', event_ids, allow_pickle=True)

print('elm free ids are saved!')
#total number of batches that will be created
#num_batch = int(sys.argv[1])

batch_size = math.ceil(len(pairs)/num_batches) #round the number of batches to be an int. need to round up to get batch sizing correct
print(f'Number of pairs in a batch: {batch_size}')
#This will overwrite the current batches with new batches

#check if batch arrays folder exists, and remove it's contents if it does
if os.path.exists('batch_arrays'):
    shutil.rmtree('batch_arrays')

#create a batch_arrays folder to put the batch indices in
os.mkdir('batch_arrays')

for k in range((num_batches)):
    start = k*batch_size
    end = min( (k+1)*batch_size, len(pairs))
    batch = pairs[start:end]
    np.save(f'batch_arrays/batch_{k+1}', batch)
    print(f"Saved batch {k+1}/{num_batches}, shape: {np.array(batch).shape}")
