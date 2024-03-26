#!/bin/sh

system="wsl"

# for algo in "gpu" "cpu"; do
# 	echo "$algo"
# 	for size in "1000" "10000" "100000" "1000000"; do
# 		res=$(./naive_serial linear $algo --size $size | tail -1 | cut -d ' ' -f 2)
# 		echo "$res"
# 	done
# done


# echo '"threads","blocks","time","system"'
# for threads in "512" "256"; do
# for threads in "256" "128" "64"; do
# 	for _ in $(seq 10); do
# 		out=$(./naive_serial linear gpu --threads "$threads" --size 1000000 2>/dev/null)
# 		res=$(echo "$out" | tail -1 | cut -d ' ' -f 2)
# 		blocks=$(echo "$out" | tail -2 | head -1 | cut -d ' ' -f 2)
# 		echo "$threads $blocks $res $system"
# 	done
# done

for method in "cpu" "gpu"; do
	for dataset in "iris" "linear"; do
		for _ in $(seq 10); do
			out=$(./naive_serial "$dataset" "$method" --threads 256 --size 1000 2> /dev/null)
			res=$(echo "$out" | tail -1 | cut -d ' ' -f 2)
			echo "$method $dataset $res $system"
		done
	done
done
