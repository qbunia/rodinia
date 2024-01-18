#!/bin/bash

# Check if the user provided an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <executable_name>"
    exit 1
fi

# Extract user input
USER_INPUT=$1

# Run the specified executable
./$USER_INPUT
