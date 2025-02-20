# Makefile for building a CUDA program

# Set the CUDA compiler (nvcc)
NVCC ?= nvcc

FOLDER = bin
# Name of the output executable
TARGET = $(FOLDER)/copy

# Source files
SRCS = copy.cu

# Auto-detect GPU compute capability and form the ARCH flag
# This assumes nvidia-smi is available and a GPU is detected
GPU_ARCH := $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
NVCC_ARCH_FLAG := -arch=sm_$(subst .,,$(GPU_ARCH))

# nvcc flags
NVCC_FLAGS = -O2 $(NVCC_ARCH_FLAG)

# Default target
all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

clean:
	rm -rf $FOLDER/*
