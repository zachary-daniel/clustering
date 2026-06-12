#!/bin/bash
echo 'Pedestal top or pedestal foot?'
read pedestal_pos
echo "Include H-mode?"
read H_mode
echo "Balence dataset?"
read data_trim
python bes_pedestal_sort.py $pedestal_pos $H_mode "False" $data_trim
echo "How many batches are there to run?"
read batches
python create_batch_indices_supplement.py $batches --Rho_low $pedestal_pos

for i in $(seq 1 $batches);
do
	python compute_Eros_supplement_pedestal_TLCC.py $i $pedestal_pos
done
python combine_results_supplement.py "TLCC" $batches --Rho_low $pedestal_pos
