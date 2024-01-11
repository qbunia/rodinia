#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <compiler> <version> <optimization_level>"
    exit 1
fi

# Extract compiler, version, and optimization level from the arguments
compiler=$1
version=$2
optimization=$3

# Set the compiler and optimization level based on the input
case "$compiler" in
    clang | gcc)
        CC=$compiler
        ;;
    *)
        echo "Unsupported compiler: $compiler"
        exit 1
        ;;
esac

case "$optimization" in
    -O1 | -O2 | -O3)
        OPT_LEVEL="${optimization}"
        ;;
    *)
        echo "Unsupported optimization level: $optimization"
        exit 1
        ;;
esac

# Run the Makefile with the specified target
make CC="$CC" OPT_LEVEL="$OPT_LEVEL" "${version}"

# Check if the compilation was successful
if [ $? -eq 0 ]; then
    # Run the executable
    ./"${version}"
else
    echo "Compilation failed."
fi
