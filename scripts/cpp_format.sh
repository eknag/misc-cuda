#!/bin/bash

# This script formats C++ code using clang-format based on the rules defined in .clang-format

# Check if clang-format is installed
if ! command -v clang-format &> /dev/null
then
    echo "clang-format could not be found. Please install it to use this script."
    exit 1
fi

# Format all C++ files in the repository
find . -name '*.cpp' -o -name '*.h' | xargs clang-format -i

echo "C++ code formatting complete."
