#!/bin/bash

# This script formats Python code using Black.
# It assumes that Black is installed in the environment.

# Exit immediately if a command exits with a non-zero status.
set -e

# Format all Python files in the repository.
black . --line-length 88

# Optionally, you can add additional formatting tools or options here.
