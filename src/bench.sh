#!/bin/sh


# for algo in "gpu" "cpu"; do
# 	echo "$algo"
# 	for size in "1000" "10000" "100000" "1000000"; do
# 		res=$(./naive_serial linear $algo --size $size | tail -1 | cut -d ' ' -f 2)
# 		echo "$res"
# 	done
# done

# for dataset in "iris" "linear"; do
# 	for threads in "256" "128" "64"; do
# 		out=$(./naive_serial "$dataset" gpu --threads "$threads")
# 		res=$(echo "$out" | tail -1 | cut -d ' ' -f 2)
# 		blocks=$(echo "$out" | tail -2 | head -1 | cut -d ' ' -f 2)
# 		echo "$method $dataset $threads $blocks $res"
# 	done
# done

for method in "cpu" "gpu"; do
	for dataset in "iris" "linear"; do
		for _ in $(seq 10); do
			out=$(./naive_serial "$dataset" "$method" --threads 128 --size 1000 2> /dev/null)
			res=$(echo "$out" | tail -1 | cut -d ' ' -f 2)
			echo "$method $dataset $res"
		done
	done
done
