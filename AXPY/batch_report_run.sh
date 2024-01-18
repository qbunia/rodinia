#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <num_runs> <exe1> <exe2> ... <exeN>"
    exit 1
fi

# Extract user inputs
NUM_RUNS=$1
shift  # Remove the first argument (num_runs)
EXECS=("$@")

# Create a CSV file for results
CSV_FILE="execution_times.csv"
echo "Executable,AvgExecutionTime" > "$CSV_FILE"

# Run report_run.sh for each executable and append results to CSV
for exe in "${EXECS[@]}"; do
    echo "Running report_run.sh for $exe..."
    result=$(./bash_run.sh "$exe" "$NUM_RUNS" | tail -n 1)  # Capture the last line of output
    echo "$exe,$result" >> "$CSV_FILE"
done

echo "CSV file generated: $CSV_FILE"
