#!/bin/bash

#INPUT_SIZE=102400000  # Default value is 102400000 if not provided
#NUM_THREADS=-4  # Default value is 4 if not provided

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <exename> <num_runs>"
    exit 1
fi

# Extract user inputs
EXE_NAME=$1
NUM_RUNS=$2

# Check if the executable exists
if [ ! -x "$EXE_NAME" ]; then
    echo "Error: Executable '$EXE_NAME' not found or not executable."
    exit 1
fi

# Run the executable multiple times
total_time=0
for ((i=1; i<=$NUM_RUNS; i++)); do
    echo "Run $i of $NUM_RUNS..."
    elapsed_time=$(./$EXE_NAME 0 $INPUT_SIZE $NUM_THREADS )
    total_time=$(echo "$total_time + $elapsed_time" | bc)
    echo "Iteration $i: $elapsed_time"
done

# Calculate average execution time
average_time=$(echo "scale=4; $total_time / $NUM_RUNS" | bc)

# Output the result (only the average execution time)
echo "$average_time"
