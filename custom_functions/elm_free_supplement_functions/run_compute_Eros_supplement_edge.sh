#!/bin/bash
echo "Include H-mode?"
read H_mode
echo "Balence dataset?"
read data_trim
echo "number of sensors?"
read num_sensors
echo "use all 64 sensors?"
read use_all
python bes_edge_region_sort.py $H_mode "False" $data_trim $num_sensors $use_all
echo "How many batches are there to run?"
read batches
python create_batch_indices_edge.py $batches 
echo "computing Eros dict"
python compute_Eros_dict_edge_region.py $num_sensors $use_all
for i in $(seq 1 $batches);
do
	python compute_Eros_supplement_edge.py $i $use_all
done
if [[ "${use_all,,}" == 'true' ]]; then
	python combine_results_supplement.py "Eros" $batches --Rho_low "edge_all"
else 
	python combine_results_supplement.py "Eros" $batches --Rho_low "edge"
fi