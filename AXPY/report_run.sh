#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <compiler> <version> <optimization_level> <num_runs>"
    exit 1
fi

compiler=$1
version=$2
optimization=$3
num_runs=$4

# Initialize total elapsed time
total_elapsed_time=0

# Run the program multiple times
for ((i = 1; i <= num_runs; i++)); do
    echo "Run $i of $num_runs..."
    make clean
    ./run.sh "$compiler" "$version" "$optimization" 0
    # Capture the output, which is assumed to be a single number
    elapsed_time=$(./run.sh "$compiler" "$version" "$optimization" 0)

    # Accumulate the elapsed time
    total_elapsed_time=$(echo "$total_elapsed_time + $elapsed_time" | awk '{printf "%.6f", $1}')
done

# Calculate the average elapsed time
average_elapsed_time=$(echo "$total_elapsed_time / $num_runs" | awk '{printf "%.6f", $1}')

echo "Average elapsed time over $num_runs runs: $average_elapsed_time seconds"
