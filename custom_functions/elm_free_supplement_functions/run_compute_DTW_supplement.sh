#!/bin/bash
echo "How many batches are there to run?"
read batches
echo "What min fraction would you like?"
read min_fraction
python create_batch_indices_supplement.py $batches $min_fraction

for i in $(seq 1 $batches);
do
	python compute_DTW_supplement.py $i $min_fraction
done
python combine_results_supplement.py "DTW" $batches $min_fraction
