/* Copyright (c) 2017-2018, 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda/atomic>

#ifdef HAVE_CUB
#include <cub/block/block_reduce.cuh>
#endif  // HAVE_CUB

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus) {                                                    \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
            exit(cudaStatus);                                                               \
        }                                                                                   \
    }

#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#endif

struct real_int_pair {
    real value;
    unsigned int arrival_counter;
};

__global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                      const real pi, const int offset, const int nx,
                                      const int my_ny, const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < my_ny; iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * (offset + iy) / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

void launch_initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                  const real pi, const int offset, const int nx, const int my_ny,
                                  const int ny) {
    initialize_boundaries<<<my_ny / 128 + 1, 128>>>(a_new, a, pi, offset, nx, my_ny, ny);
    CUDA_RT_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              real* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, const bool calculate_norm) {
#ifdef HAVE_CUB
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#endif  // HAVE_CUB
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    real local_l2_norm = 0.0;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;
        if (calculate_norm) {
            real residue = new_val - a[iy * nx + ix];
            local_l2_norm += residue * residue;
        }
    }
    if (calculate_norm) {
#ifdef HAVE_CUB
        real block_l2_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
        if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
#else
        atomicAdd(l2_norm, local_l2_norm);
#endif  // HAVE_CUB
    }
}

void launch_jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                          real* __restrict__ const l2_norm, const int iy_start, const int iy_end,
                          const int nx, const bool calculate_norm, cudaStream_t stream) {
    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                  ((iy_end - iy_start) + dim_block_y - 1) / dim_block_y, 1);
    jacobi_kernel<dim_block_x, dim_block_y><<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, stream>>>(
        a_new, a, l2_norm, iy_start, iy_end, nx, calculate_norm);
    CUDA_RT_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_p2p_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                                  real* __restrict__ const l2_norm, const int iy_start,
                                  const int iy_end, const int nx,
                                  real* __restrict__ const a_new_top, const int top_iy,
                                  real* __restrict__ const a_new_bottom, const int bottom_iy,
                                  const bool calculate_norm) {
#ifdef HAVE_CUB
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#endif  // HAVE_CUB
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x + 1;
    real local_l2_norm = 0.0;

    if (iy < iy_end && ix < (nx - 1)) {
        const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                     a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
        a_new[iy * nx + ix] = new_val;

        if (iy_start == iy) {
            a_new_top[top_iy * nx + ix] = new_val;
        }

        if ((iy_end - 1) == iy) {
            a_new_bottom[bottom_iy * nx + ix] = new_val;
        }

        if (calculate_norm) {
            real residue = new_val - a[iy * nx + ix];
            local_l2_norm += residue * residue;
        }
    }
    if (calculate_norm) {
#ifdef HAVE_CUB
        real block_l2_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
        if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
#else
        atomicAdd(l2_norm, local_l2_norm);
#endif  // HAVE_CUB
    }
}

void launch_jacobi_p2p_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              real* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, real* __restrict__ const a_new_top,
                              const int top_iy, real* __restrict__ const a_new_bottom,
                              const int bottom_iy, const bool calculate_norm, cudaStream_t stream) {
    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                  ((iy_end - iy_start) + dim_block_y - 1) / dim_block_y, 1);
    jacobi_p2p_kernel<dim_block_x, dim_block_y>
        <<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, stream>>>(
            a_new, a, l2_norm, iy_start, iy_end, nx, a_new_top, top_iy, a_new_bottom, bottom_iy,calculate_norm);
    CUDA_RT_CALL(cudaGetLastError());
}

__global__ void all_reduce_norm_barrier_kernel(real* const l2_norm,
                                               real_int_pair* partial_l2_norm_uc,
                                               real_int_pair* partial_l2_norm_mc,
                                               const unsigned int expected_count) {
    assert(1 == blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z);
    real l2_norm_sum = 0.0;
#if __CUDA_ARCH__ >= 900
    // atomic reduction to all replicas
    // this can be conceptually thought of as __threadfence_system(); atomicAdd_system(arrival_counter_mc, 1);
    // See https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red
    // for multimem PTX doc
    asm volatile ("multimem.red.release.sys.global.add.u32 [%0], %1;" ::"l"(&(partial_l2_norm_mc->arrival_counter)), "n"(1) : "memory");

    // Need a fence between MC and UC access to the same memory:
    // - fence.proxy instructions establish an ordering between memory accesses that may happen through different proxies
    // - Value .alias of the .proxykind qualifier refers to memory accesses performed using virtually aliased addresses to the same memory location.
    // from https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-membar
    asm volatile ("fence.proxy.alias;" ::: "memory");

    // spin wait with acquire ordering on UC mapping till all peers have arrived in this iteration
    // Note: all ranks reach an MPI_Barrier after this kernel, such that it is not possible for the barrier to be unblocked by an
    // arrival of a rank for the next iteration if some other rank is slow.
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ac(partial_l2_norm_uc->arrival_counter);
    while (expected_count > ac.load(cuda::memory_order_acquire));

    // Atomic load reduction from all replicas. It does not provide ordering so it can be relaxed.
#ifdef USE_DOUBLE
    asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f64 %0, [%1];" : "=d"(l2_norm_sum) : "l"(&(partial_l2_norm_mc->value)) : "memory");
#else
    asm volatile ("multimem.ld_reduce.relaxed.sys.global.add.f32 %0, [%1];" : "=f"(l2_norm_sum) : "l"(&(partial_l2_norm_mc->value)) : "memory");
#endif
#endif
    *l2_norm = std::sqrt(l2_norm_sum);
}

void launch_all_reduce_norm_barrier_kernel(real* __restrict__ const l2_norm,
                                           real_int_pair* __restrict__ partial_l2_norm_uc,
                                           real_int_pair* __restrict__ partial_l2_norm_mc,
                                           const int num_gpus, const int iter,
                                           cudaStream_t stream) {
    // calculating expected count as unsigned for well defined overflow to correctly handle large
    // iteration counts with many GPUs
    unsigned int expected_count = num_gpus;
    // iter starts at 0 so need to scale with iter+1
    expected_count *= (iter + 1);
    all_reduce_norm_barrier_kernel<<<1, 1, 0, stream>>>(l2_norm, partial_l2_norm_uc,
                                                        partial_l2_norm_mc, expected_count);
    CUDA_RT_CALL(cudaGetLastError());
}
