/* Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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
#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <sstream>

#include <omp.h>

#ifdef HAVE_CUB
#include <cub/block/block_reduce.cuh>
#endif  // HAVE_CUB

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color_id = cid;                                \
        color_id = color_id % num_colors;                  \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }

typedef float real;
constexpr real tol = 1.0e-8;

const real PI = 2.0 * std::asin(1.0);

__global__ void initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                      const real pi, const int nx, const int ny) {
    for (int iy = blockIdx.x * blockDim.x + threadIdx.x; iy < ny; iy += blockDim.x * gridDim.x) {
        const real y0 = sin(2.0 * pi * iy / (ny - 1));
        a[iy * nx + 0] = y0;
        a[iy * nx + (nx - 1)] = y0;
        a_new[iy * nx + 0] = y0;
        a_new[iy * nx + (nx - 1)] = y0;
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              real* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx) {
#ifdef HAVE_CUB
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#endif  // HAVE_CUB
    const int iy = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    real local_l2_norm = 0.0;

    if (iy < iy_end) {
        if (ix >= 1 && ix < (nx - 1)) {
            const real new_val = 0.25 * (a[iy * nx + ix + 1] + a[iy * nx + ix - 1] +
                                         a[(iy + 1) * nx + ix] + a[(iy - 1) * nx + ix]);
            a_new[iy * nx + ix] = new_val;

            // apply boundary conditions
            if (iy_start == iy) {
                a_new[iy_end * nx + ix] = new_val;
            }

            if ((iy_end - 1) == iy) {
                a_new[(iy_start - 1) * nx + ix] = new_val;
            }

            real residue = new_val - a[iy * nx + ix];
            local_l2_norm = residue * residue;
        }
    }
#ifdef HAVE_CUB
    real block_l2_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
    if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
#else
    atomicAdd(l2_norm, local_l2_norm);
#endif  // HAVE_CUB
}

double noopt(const int nx, const int ny, const int iter_max, real* const a_ref_h, const int nccheck,
             const bool print);

template <typename T>
T get_argval(char** begin, char** end, const std::string& arg, const T default_val) {
    T argval = default_val;
    char** itr = std::find(begin, end, arg);
    if (itr != end && ++itr != end) {
        std::istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const std::string& arg) {
    char** itr = std::find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

struct l2_norm_buf {
    cudaEvent_t copy_done;
    real* d;
    real* h;
};

int main(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");

    if (nccheck != 1) {
        fprintf(stderr, "Only nccheck = 1 is supported\n");
        return -1;
    }

    real* a;
    real* a_new;

    cudaStream_t compute_stream;
    cudaStream_t copy_l2_norm_stream;
    cudaStream_t reset_l2_norm_stream;

    cudaEvent_t compute_done;
    cudaEvent_t reset_l2_norm_done[2];

    real l2_norms[2];
    l2_norm_buf l2_norm_bufs[2];

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaSetDevice(0));
    CUDA_RT_CALL(cudaFree(0));

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, nx, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    CUDA_RT_CALL(cudaStreamCreate(&copy_l2_norm_stream));
    CUDA_RT_CALL(cudaStreamCreate(&reset_l2_norm_stream));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&reset_l2_norm_done[0], cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&reset_l2_norm_done[1], cudaEventDisableTiming));

    for (int i = 0; i < 2; ++i) {
        CUDA_RT_CALL(cudaEventCreateWithFlags(&l2_norm_bufs[i].copy_done, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaMalloc(&l2_norm_bufs[i].d, sizeof(real)));
        CUDA_RT_CALL(cudaMemset(l2_norm_bufs[i].d, 0, sizeof(real)));
        CUDA_RT_CALL(cudaMallocHost(&l2_norm_bufs[i].h, sizeof(real)));
        (*l2_norm_bufs[i].h) = 1.0;
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (!csv)
        printf(
            "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
            "every %d iterations\n",
            iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    int iter = 0;
    for (int i = 0; i < 2; ++i) {
        l2_norms[i] = 0.0;
    }

    double start = omp_get_wtime();

    PUSH_RANGE("Jacobi solve", 0)

    bool l2_norm_greater_than_tol = true;
    while (l2_norm_greater_than_tol && iter < iter_max) {
        // on new iteration: old current vars are now previous vars, old
        // previous vars are no longer needed
        int prev = iter % 2;
        int curr = (iter + 1) % 2;

        // wait for memset from old previous iteration to complete
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, reset_l2_norm_done[curr], 0));

        jacobi_kernel<dim_block_x, dim_block_y>
            <<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
                a_new, a, l2_norm_bufs[curr].d, iy_start, iy_end, nx);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        // perform L2 norm calculation
        if ((iter % nccheck) == 0 || (!csv && (iter % 100) == 0)) {
            CUDA_RT_CALL(cudaStreamWaitEvent(copy_l2_norm_stream, compute_done, 0));
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_bufs[curr].h, l2_norm_bufs[curr].d, sizeof(real),
                                         cudaMemcpyDeviceToHost, copy_l2_norm_stream));
            CUDA_RT_CALL(cudaEventRecord(l2_norm_bufs[curr].copy_done, copy_l2_norm_stream));

            // make sure D2H copy is complete before using the data for
            // calculation
            CUDA_RT_CALL(cudaEventSynchronize(l2_norm_bufs[prev].copy_done));

            l2_norms[prev] = *(l2_norm_bufs[prev].h);
            l2_norms[prev] = std::sqrt(l2_norms[prev]);
            l2_norm_greater_than_tol = (l2_norms[prev] > tol);

            if (!csv && (iter % 100) == 0) {
                printf("%5d, %0.6f\n", iter, l2_norms[prev]);
            }

            // reset everything for next iteration
            l2_norms[prev] = 0.0;
            *(l2_norm_bufs[prev].h) = 0.0;
            CUDA_RT_CALL(
                cudaMemsetAsync(l2_norm_bufs[prev].d, 0, sizeof(real), reset_l2_norm_stream));
            CUDA_RT_CALL(cudaEventRecord(reset_l2_norm_done[prev], reset_l2_norm_stream));
        }

        std::swap(a_new, a);
        iter++;
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    POP_RANGE
    double stop = omp_get_wtime();

    if (csv) {
        printf("single_gpu, %d, %d, %d, %d, %f\n", nx, ny, iter_max, nccheck, (stop - start));
    } else {
        printf("%dx%d: 1 GPU: %8.4f s\n", ny, nx, (stop - start));
    }

    for (int i = 0; i < 2; ++i) {
        CUDA_RT_CALL(cudaFreeHost(l2_norm_bufs[i].h));
        CUDA_RT_CALL(cudaFree(l2_norm_bufs[i].d));
        CUDA_RT_CALL(cudaEventDestroy(l2_norm_bufs[i].copy_done));
    }

    CUDA_RT_CALL(cudaEventDestroy(reset_l2_norm_done[1]));
    CUDA_RT_CALL(cudaEventDestroy(reset_l2_norm_done[0]));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));

    CUDA_RT_CALL(cudaStreamDestroy(reset_l2_norm_stream));
    CUDA_RT_CALL(cudaStreamDestroy(copy_l2_norm_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));

    return 0;
}
