/* Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#include <cstdio>
#include <cstdlib>

#include <nccl.h>
#include <nccl_device.h>

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
            exit( cudaStatus );                                                             \
        }                                                                                   \
    }

#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#endif

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

// Returns true if peer (a world rank) is reached with GIN and false if it is reached with plain
// loads and stores through the LSA domain (e.g. NVLink).
__device__ bool use_gin_for_peer(const ncclDevComm& devComm, const int peer,
                                          const bool force_gin) {
    return force_gin || !ncclTeamRankIsMember(ncclTeamLsa(devComm), ncclTeamWorld(devComm), peer);
}

// Copies count elements from src_offset to dst_offset of the window win on peer (a world rank).
// Like nvshmemx_float_put_nbi_block it must be called by all threads of a 1D block with
// block-uniform arguments. src points to src_offset of win.
template <bool USE_GIN>
__device__ void put_to_peer(const ncclDevComm& devComm, ncclWindow_t win, const int peer,
                            const size_t dst_offset, const size_t src_offset,
                            const real* __restrict__ const src, const int count,
                            const bool force_gin) {
    if (USE_GIN && use_gin_for_peer(devComm, peer, force_gin)) {
        // With ncclCoopCta the put syncs the block, so src is complete, and is issued by one thread.
        ncclGin gin(devComm, 0);
        gin.put(ncclTeamWorld(devComm), peer, win, dst_offset, win, src_offset,
                count * sizeof(real), ncclGin_None{}, ncclGin_None{}, ncclCoopCta());
    } else {
        // No block sync needed: in a 1D block src[threadIdx.x] was written by this thread.
        const ncclTeam lsa = ncclTeamLsa(devComm);
        real* const dst = (real*)ncclGetLsaPointer(
            win, dst_offset, ncclTeamRankToTeam(lsa, ncclTeamWorld(devComm), peer));
        if (threadIdx.x < count) dst[threadIdx.x] = src[threadIdx.x];
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, bool USE_GIN>
__global__ void jacobi_block_comm_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              real* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, const bool calculate_norm,
                              const __grid_constant__ ncclDevComm devComm,
                              ncclWindow_t a_new_win, const int top_rank, const int top_iy,
                              const int bottom_rank, const int bottom_iy, const bool force_gin) {
    static_assert(1 == BLOCK_DIM_Y, "put_to_peer requires 1D blocks");
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

    /* starting (x, y) coordinate of the block */
    int block_iy = iy - threadIdx.y;
    int block_ix = ix - threadIdx.x;

    /* Communicate the boundaries */
    const int count = min(blockDim.x, nx - 1 - block_ix);
    if ((block_iy <= iy_start) && (iy_start < block_iy + blockDim.y)) {
        put_to_peer<USE_GIN>(devComm, a_new_win, top_rank, (top_iy * nx + block_ix) * sizeof(real),
                             (iy_start * nx + block_ix) * sizeof(real),
                             a_new + iy_start * nx + block_ix, count, force_gin);
    }
    if ((block_iy < iy_end) && (iy_end <= block_iy + blockDim.y)) {
        put_to_peer<USE_GIN>(devComm, a_new_win, bottom_rank,
                             (bottom_iy * nx + block_ix) * sizeof(real),
                             ((iy_end - 1) * nx + block_ix) * sizeof(real),
                             a_new + (iy_end - 1) * nx + block_ix, count, force_gin);
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

// Completes the halo exchange and protects the input rows reused by the next iteration.
template <bool USE_GIN>
__global__ void syncneighborhood_kernel(const __grid_constant__ ncclDevComm devComm) {
    if constexpr (USE_GIN) {
        ncclBarrierSession<ncclCoopCta> bar(ncclCoopCta(), ncclTeamTagWorld(),
                                          ncclGin(devComm, 0), 0);
        bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel, ncclGinFenceLevel::Put);
    } else {
        ncclLsaBarrierSession<ncclCoopCta> bar(ncclCoopCta(), devComm, ncclTeamTagLsa(), 0);
        bar.sync(ncclCoopCta(), cuda::memory_order_acq_rel);
    }
}

bool use_gin_for_peer(const ncclComm_t comm, const int peer,
                                             const bool force_gin) {
    return force_gin || !ncclTeamRankIsMember(ncclTeamLsa(comm), ncclTeamWorld(comm), peer);
}

// The GIN code needs more registers, which lowers the occupancy of the whole kernel, so it is
// only compiled into the kernels if a neighbor is reached with GIN.
bool use_gin(const ncclComm_t comm, const int top_rank, const int bottom_rank,
             const bool force_gin) {
    return use_gin_for_peer(comm, top_rank, force_gin) ||
           use_gin_for_peer(comm, bottom_rank, force_gin);
}

void launch_jacobi_block_comm_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                          real* __restrict__ const l2_norm, const int iy_start, const int iy_end,
                          const int nx, const bool calculate_norm, const ncclDevComm& devComm, const ncclComm_t comm,
                          ncclWindow_t a_new_win, const int top_rank, const int top_iy,
                          const int bottom_rank, const int bottom_iy, const bool force_gin,
                          cudaStream_t stream) {
    constexpr int dim_block_x = 1024;
    constexpr int dim_block_y = 1;
    dim3 dim_grid((nx - 2 + dim_block_x - 1) / dim_block_x,
                  ((iy_end - iy_start) + dim_block_y - 1) / dim_block_y, 1);
    auto kernel = use_gin(comm, top_rank, bottom_rank, force_gin)
                      ? jacobi_block_comm_kernel<dim_block_x, dim_block_y, true>
                      : jacobi_block_comm_kernel<dim_block_x, dim_block_y, false>;
    kernel<<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, stream>>>(
        a_new, a, l2_norm, iy_start, iy_end, nx, calculate_norm, devComm, a_new_win, top_rank,
        top_iy, bottom_rank, bottom_iy, force_gin);
    CUDA_RT_CALL(cudaGetLastError());
}

void launch_syncneighborhood_kernel(const ncclDevComm& devComm, const ncclComm_t comm,
                                    const bool force_gin, cudaStream_t stream) {
    const bool gin = force_gin || ncclTeamLsa(comm).nRanks != ncclTeamWorld(comm).nRanks;
    auto kernel = gin ? syncneighborhood_kernel<true> : syncneighborhood_kernel<false>;
    kernel<<<1, 32, 0, stream>>>(devComm);
    CUDA_RT_CALL(cudaGetLastError());
}
