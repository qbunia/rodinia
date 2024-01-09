#!/bin/bash

# Function to execute the program
execute_program() {
    # Run the executable with user-specified parameters
    ./"$executable" "$data_size" "$num_threads" "$full_report"
}

# Function to compile the program
compile_program() {
    # Compile the source code with the specified options
    make "$executable" COMPILER="$compiler" OPT_LEVEL="$opt_level"

    # Check if compilation was successful
    if [ $? -eq 0 ]; then
        echo "Compilation successful."
        execute_program
    else
        echo "Compilation failed."
    fi
}

# Check if user provided command line arguments
if [ "$1" == "clean" ]; then
    # Clean option
    make clean
    echo "Cleaned the project."
elif [ $# -eq 7 ] && [ "$1" == "y" ]; then
    version="$2"
    compiler="$3"
    opt_level="$4"
    data_size="$5"
    num_threads="$6"
    full_report="$7"
    
    case $version in
        0) executable="axpy_P0_exec";;
        1) executable="axpy_omp_CPU_P1_exec";;
        2) executable="axpy_omp_CPU_P2_exec";;
        3) executable="axpy_omp_CPU_P3_exec";;
        4) executable="axpy_omp_Offloading_exec";;
        5)
            executable="axpy_cuda_exec"
            num_threads=""  # No need for threads in CUDA version
            ;;
        *) echo "Invalid version choice. Exiting."; exit 1;;
    esac

    # Compile and execute the program
    compile_program
else
    # Interactive mode
    # Get user input for version
    echo "Choose a version to compile and execute:"
    echo "0. axpy_P0"
    echo "1. axpy_omp_CPU_P1"
    echo "2. axpy_omp_CPU_P2"
    echo "3. axpy_omp_CPU_P3"
    echo "4. axpy_omp_Offloading"
    echo "5. axpy_cuda"

    read -p "Enter the version number or name: " version

    if [ "$version" == "clean" ]; then
        # Clean option
        make clean
        echo "Cleaned the project."
        exit 0
    fi

    # Set default values for compiler and opt_level if version is axpy_cuda_exec
    case $version in
        5)
            compiler="nvcc"
            opt_level="-O2"
            num_threads="4"   # No need in CUDA version
            ;;
        *)
            read -p "Enter the compiler (clang/gcc): " compiler
            read -p "Enter the optimization level (-O1/-O2/-O3): " opt_level
            read -p "Enter number of threads: " num_threads
            ;;
    esac

    # Get user input for data_size and full_report
    read -p "Enter data size: " data_size
    read -p "Enter full report (0 for false, 1 for true): " full_report

    # Compile and execute the program
    case $version in
        0) executable="axpy_P0_exec";;
        1) executable="axpy_omp_CPU_P1_exec";;
        2) executable="axpy_omp_CPU_P2_exec";;
        3) executable="axpy_omp_CPU_P3_exec";;
        4) executable="axpy_omp_Offloading_exec";;
        5) executable="axpy_cuda_exec";;
        *) echo "Invalid version choice. Exiting."; exit 1;;
    esac

    compile_program
fi
