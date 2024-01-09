#!/bin/bash

# Function to run AXPY with specified parameters
run_axpy() {
    bash run.sh "$@"
}

# Function to run other benchmarks
run_benchmark() {
    # Add logic to run other benchmarks based on user input
    echo "Running benchmark: $1"
    # Add relevant commands here
}

# Main menu
echo "Welcome to NeoRodinia!"
echo "Choose Mode:"
echo "1: Fast Mode"
echo "2: Interactive Mode"
read -p "Enter your choice (1 or 2): " mode_choice

case $mode_choice in
    1)
        # Fast Mode
        read -p "Please input your instruction: " instruction

        # Parse the instruction and execute the corresponding benchmark
        # Add logic to parse the instruction and call the appropriate benchmark
        ;;

    2)
        # Interactive Mode
        echo "1: Full report for all benchmarks"
        echo "2: One benchmark"
        read -p "Enter your choice (1 or 2): " interactive_choice

        case $interactive_choice in
            1)
                # Full report for all benchmarks
                echo "1: Full report"
                echo "2: CSV report"
                read -p "Enter your choice (1 or 2): " report_choice

                case $report_choice in
                    1)
                        # Full report - All versions
                        echo "1: One time"
                        echo "2: Average for 10 times (Some benchmarks are not allowed.)"
                        read -p "Enter your choice (1 or 2): " run_choice

                        case $run_choice in
                            1)
                                # Execute full report for all versions (one time)
                                echo "Executing full report for all versions (one time)..."
                                # Add relevant commands here
                                ;;
                            2)
                                # Execute full report for all versions (average for 10 times)
                                echo "Executing full report for all versions (average for 10 times)..."
                                # Add relevant commands here
                                ;;
                            *)
                                echo "Invalid choice. Exiting."
                                ;;
                        esac
                        ;;

                    2)
                        # CSV report
                        # Add logic for CSV report
                        echo "CSV report not implemented yet."
                        ;;
                    *)
                        echo "Invalid choice. Exiting."
                        ;;
                esac
                ;;

            2)
                # One benchmark
                echo "Choose a benchmark:"
                # List of benchmarks
                benchmarks=("AXPY" "hotspot" "lavaMD" "MatMul" "nw" "rex" "Sum" "backprop" "hotspot3D" "leukocyte" "MatVec" "bfs" "dwt2d" "huffman" "mummergpu" "particlefilter" "srad" "b+tree" "gaussian" "hybridsort" "lud" "myocyte" "pathfinder" "Stencil" "cfd" "heartwall" "kmeans" "nn" "streamcluster")

                for ((i = 0; i < ${#benchmarks[@]}; i++)); do
                    echo "$((i + 1)): ${benchmarks[i]}"
                done

                read -p "Enter the benchmark number (1-${#benchmarks[@]}): " benchmark_number

                if ((benchmark_number >= 1 && benchmark_number <= ${#benchmarks[@]})); then
                    chosen_benchmark="${benchmarks[benchmark_number - 1]}"
                    echo "Chosen benchmark: $chosen_benchmark"
                    # Enter the folder with the same name as the chosen benchmark
                    cd "$chosen_benchmark" || { echo "Failed to enter the folder. Exiting."; exit 1; }

                    # Make the run.sh script executable
                    chmod +x run.sh

                    # Execute the run.sh script within the folder
                    bash run.sh
                    run_benchmark "$chosen_benchmark"
                else
                    echo "Invalid benchmark number. Exiting."
                fi
                ;;
            *)
                echo "Invalid choice. Exiting."
                ;;
        esac
        ;;
    *)
        echo "Invalid choice. Exiting."
        ;;
esac

echo "Script completed."
