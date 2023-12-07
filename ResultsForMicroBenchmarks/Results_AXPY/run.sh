#!/bin/bash

# List of file names
file_names=("AXPY_ompCPU_P0_result.txt" "AXPY_ompCPU_P1_result.txt" "AXPY_ompCPU_P2_result.txt" "AXPY_ompCPU_P3_result.txt")

# Output file name
output_file="output_runtimes.csv"

# Function to extract runtime value from a file
extract_runtime() {
    file_name=$1
    if [ -e "$file_name" ]; then
        runtime_value=$(awk 'NR==6 {print $2}' "$file_name")
        echo -n "$runtime_value"
    else
        echo "File not found: $file_name"
    fi
}

# Iterate over each file and extract runtime values
for file_name in "${file_names[@]}"; do
    extract_runtime "$file_name"
    echo -n ","
done | sed 's/,$//' > "$output_file"

echo "Runtimes extracted and saved to $output_file"
