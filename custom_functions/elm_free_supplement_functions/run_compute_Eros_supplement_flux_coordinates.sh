#!/bin/bash
echo "Rho Lower bound?" 
read Rho_low
echo "Rho upper bound?"
read Rho_high
echo "Z lower bound?"
read Z_low
echo "Z upper bound?"
read Z_high
echo "Include H-mode?"
read H_mode
python bes_flux_coordinate_sort.py $Rho_low $Rho_high $Z_low $Z_high $H_mode "False"

echo "How many batches are there to run?"
read batches
python create_batch_indices_supplement.py $batches --Rho_low $Rho_low

for i in $(seq 1 $batches);
do
	python compute_Eros_supplement_normalized_flux.py $i 
done
python combine_results_supplement.py "Eros" $batches --Rho_low $Rho_low
