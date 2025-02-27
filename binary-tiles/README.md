mkdir build && cd build
cmake .. -DCMAKE_CUDA_COMPILER=nvcc
make &&  ctest --verbose