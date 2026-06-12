#!/bin/bash
echo 'Pedestal top, pedestal foot, both or edge?'
read pedestal_pos
if [[ "$pedestal_pos" == "edge" ]]; then
	echo 'num sensors?'
	read num_sensors
fi

echo "Include H-mode?"
read H_mode
echo "Balence dataset?"
read data_trim

if [[ "$pedestal_pos" == "top" || "$pedestal_pos" == "foot" ]]; then
	python bes_pedestal_sort.py $pedestal_pos $H_mode "False" $data_trim
	echo "How many batches are there to run?"
	read batches
	python create_batch_indices_supplement.py $batches --Rho_low $pedestal_pos
fi

if [[ "$pedestal_pos" == "edge" ]]; then
	python bes_edge_region_sort.py $H_mode "False" $data_trim $num_sensors "False"
	echo "How many batches are there to run?"
	read batches
	python create_batch_indices_edge.py $batches 
fi




for i in $(seq 1 $batches);
do
	python compute_Eros_supplement_magnetics.py $i $pedestal_pos
done
python combine_results_supplement.py "Eros" $batches --Rho_low $pedestal_pos --magnetics_bool "True"
