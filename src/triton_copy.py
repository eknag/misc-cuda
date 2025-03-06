import torch
import triton
import triton.language as tl
import triton.testing as tt
import time

# Simple non-autotuned kernel as a baseline
@triton.jit
def copy_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate start offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Calculate indices for each thread
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask = offsets < n_elements
    
    # Load data from source
    x = tl.load(src_ptr + offsets, mask=mask)
    
    # Store data to destination
    tl.store(dst_ptr + offsets, x, mask=mask)

# Autotuned kernel that tries different block sizes
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def copy_kernel_tuned(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate start offset for this block
    block_start = pid * BLOCK_SIZE
    
    # Calculate indices for each thread
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices
    mask = offsets < n_elements
    
    # Load data from source
    x = tl.load(src_ptr + offsets, mask=mask)
    
    # Store data to destination
    tl.store(dst_ptr + offsets, x, mask=mask)

# Vectorized copy kernel - each thread processes multiple elements
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32, 'ELEMENTS_PER_THREAD': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32, 'ELEMENTS_PER_THREAD': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32, 'ELEMENTS_PER_THREAD': 4}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 4}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 4}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 32, 'ELEMENTS_PER_THREAD': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32, 'ELEMENTS_PER_THREAD': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32, 'ELEMENTS_PER_THREAD': 8}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 8}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 8}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 8}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def copy_kernel_vec(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ELEMENTS_PER_THREAD: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate base offset for this block
    block_start = pid * BLOCK_SIZE * ELEMENTS_PER_THREAD
    
    # Each thread processes ELEMENTS_PER_THREAD elements in sequence
    for i in range(ELEMENTS_PER_THREAD):
        # Calculate element offset for this iteration
        offsets = block_start + i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        
        # Create mask for valid indices
        mask = offsets < n_elements
        
        # Load data from source
        x = tl.load(src_ptr + offsets, mask=mask)
        
        # Store data to destination
        tl.store(dst_ptr + offsets, x, mask=mask)

def simple_copy(src, dst):
    """
    Basic copy function using Triton
    """
    assert src.is_cuda and dst.is_cuda, "Inputs must be CUDA tensors"
    assert src.shape == dst.shape, "Source and destination must have the same shape"
    
    n_elements = src.numel()
    block_size = 32
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, block_size),)
    
    # Launch kernel
    copy_kernel[grid](
        src_ptr=src,
        dst_ptr=dst,
        n_elements=n_elements,
        BLOCK_SIZE=block_size,
    )

def autotuned_copy(src, dst):
    """
    Copy function using auto-tuning
    """
    assert src.is_cuda and dst.is_cuda, "Inputs must be CUDA tensors"
    assert src.shape == dst.shape, "Source and destination must have the same shape"
    
    n_elements = src.numel()
    
    # Use lambda META to calculate grid based on autotuned parameters
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    # Launch kernel
    copy_kernel_tuned[grid](
        src_ptr=src,
        dst_ptr=dst,
        n_elements=n_elements,
    )

def vectorized_copy(src, dst):
    """
    Vectorized copy function using auto-tuning
    """
    assert src.is_cuda and dst.is_cuda, "Inputs must be CUDA tensors"
    assert src.shape == dst.shape, "Source and destination must have the same shape"
    
    n_elements = src.numel()
    
    # Use lambda META to calculate grid based on autotuned parameters
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE'] * META['ELEMENTS_PER_THREAD']),)
    
    # Launch kernel
    copy_kernel_vec[grid](
        src_ptr=src,
        dst_ptr=dst,
        n_elements=n_elements,
    )

def benchmark_copy(size):
    """
    Benchmark different copy implementations
    """
    # Create source and destination tensors
    src = torch.arange(size, dtype=torch.int32, device='cuda')
    dst = torch.empty_like(src)
    
    # Define benchmark functions
    def basic_copy():
        simple_copy(src, dst)
    
    def tuned_copy():
        autotuned_copy(src, dst)
    
    def vector_copy():
        vectorized_copy(src, dst)
    
    def torch_copy():
        dst.copy_(src)
    
    # Performance constants
    A100_SM_FREQ = 1.41e9  # A100 SM frequency in Hz
    
    # Benchmark simple copy
    print("\nBenchmarking simple copy...")
    ms_basic = tt.do_bench(basic_copy)
    cycles_basic = ms_basic * A100_SM_FREQ / 1e3
    elem_per_cycle_basic = size / cycles_basic
    
    # Benchmark autotuned copy
    print("\nBenchmarking autotuned copy...")
    ms_tuned = tt.do_bench(tuned_copy)
    cycles_tuned = ms_tuned * A100_SM_FREQ / 1e3
    elem_per_cycle_tuned = size / cycles_tuned
    
    # Benchmark vectorized copy
    print("\nBenchmarking vectorized copy...")
    ms_vec = tt.do_bench(vector_copy)
    cycles_vec = ms_vec * A100_SM_FREQ / 1e3
    elem_per_cycle_vec = size / cycles_vec
    
    # Benchmark PyTorch copy
    print("\nBenchmarking PyTorch copy...")
    ms_torch = tt.do_bench(torch_copy)
    cycles_torch = ms_torch * A100_SM_FREQ / 1e3
    elem_per_cycle_torch = size / cycles_torch
    
    # Print results
    print(f"\nResults for {size} elements:")
    print(f"Simple copy:    {elem_per_cycle_basic:.0f} elem/clk, {ms_basic:.3f} ms")
    print(f"Autotuned copy: {elem_per_cycle_tuned:.0f} elem/clk, {ms_tuned:.3f} ms")
    print(f"Vectorized copy: {elem_per_cycle_vec:.0f} elem/clk, {ms_vec:.3f} ms")
    print(f"PyTorch copy:   {elem_per_cycle_torch:.0f} elem/clk, {ms_torch:.3f} ms")
    
    # Verify correctness
    simple_copy(src, dst)  # Reset dst to src values
    assert torch.all(src == dst), "Copy verification failed"
    
    return max(elem_per_cycle_basic, elem_per_cycle_tuned, elem_per_cycle_vec)

def main():
    # Define problem size (same as CUDA implementation)
    N = 16384 * 1024 * 108
    
    # Run benchmarks
    print("Benchmarking Triton copy implementations...")
    print("This will take a moment as autotuning finds optimal parameters...")
    benchmark_copy(N)

if __name__ == "__main__":
    main() 