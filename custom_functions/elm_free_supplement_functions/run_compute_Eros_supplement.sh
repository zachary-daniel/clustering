#!/bin/bash
echo "How many batches are there to run?"
read batches
python create_batch_indices_supplement.py $batches 

for i in $(seq 1 $batches);
do
	python compute_Eros_supplement.py $i 
done
python combine_results_supplement.py "Eros" $batches 
