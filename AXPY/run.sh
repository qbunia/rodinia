#!/bin/bash
INPUT_SIZE=${2:-102400000}  # Default value is 102400000 if not provided
NUM_THREADS=${3:-4}   # Default value is 4 if not provided

# Check if the user provided an argument
if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
    echo "Usage: $0 <executable_name> [input_size] [num_threads]"
    exit 1
fi

# Extract user input
USER_INPUT=$1

# Run the specified executable
./$USER_INPUT 1 $INPUT_SIZE $NUM_THREADS 
