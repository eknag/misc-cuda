#!/bin/bash

# This script formats CUDA code using clang-format.
# It applies the formatting rules defined in the .clang-format file.

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null
then
    echo "clang-format could not be found. Please install it to use this script."
    exit 1
fi

# Format all CUDA files in the current directory and subdirectories
find . -name '*.cu' -o -name '*.cuh' | xargs clang-format -i

echo "CUDA code formatting completed."
