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
#include <cmath>
#include <cstdio>
#include <iostream>
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

constexpr int MAX_NUM_DEVICES = 32;

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
                              const int iy_end, const int nx, const int ny,
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
        if (1 == iy) {
            a_new[(ny - 1) * nx + ix] = new_val;
        }
        if ((ny - 2) == iy) {
            a_new[0 * nx + ix] = new_val;
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

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref,
                  const int nccheck, const bool print);

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

int main(int argc, char* argv[]) {
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");

    real* a;
    real* a_new;

    real* a_ref;
    double runtime_serial = 0.0;

    CUDA_RT_CALL(cudaSetDevice(0));
    CUDA_RT_CALL(cudaFree(0));

    CUDA_RT_CALL(cudaMallocManaged(&a_ref, nx * ny * sizeof(real)));
    runtime_serial = single_gpu(nx, ny, iter_max, a_ref, nccheck, !csv);

    CUDA_RT_CALL(cudaMallocManaged(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMallocManaged(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, nx, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (!csv)
        printf(
            "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
            "every %d iterations\n",
            iter_max, ny, nx, nccheck);

    real l2_norm = 1.0;

    cudaEvent_t compute_done[2][MAX_NUM_DEVICES];

    bool result_correct = true;
    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));
#pragma omp parallel num_threads(num_devices) firstprivate(a, a_new)
    {
        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaFree(0));

        // ny - 2 rows are distributed amongst `size` ranks in such a way
        // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
        // This optimizes load balancing when (ny - 2) % size != 0
        int chunk_size;
        int chunk_size_low = ny / num_devices;
        int chunk_size_high = chunk_size_low + 1;
        // To calculate the number of ranks that need to compute an extra row,
        // the following formula is derived from this equation:
        // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
        int num_ranks_low = num_devices * chunk_size_low + num_devices -
                            ny;  // Number of ranks with chunk_size = chunk_size_low
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        // Calculate local domain boundaries
        int iy_start;
        if (dev_id < num_ranks_low) {
            iy_start = dev_id * chunk_size_low;
        } else {
            iy_start = num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high;
        }
        int iy_end = iy_start + chunk_size;
        // Do not process boundaries
        iy_start = std::max(iy_start, 1);
        iy_end = std::min(iy_end, ny - 1);

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        int canAccessPeer = 0;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
        if (canAccessPeer) {
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
        }
        const int bottom = (dev_id + 1) % num_devices;
        canAccessPeer = 0;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
        if (top != bottom && canAccessPeer) {
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
        }

#ifdef UM_HINTS
        CUDA_RT_CALL(cudaMemAdvise(a + iy_start * nx, (iy_end - iy_start) * nx * sizeof(real),
                                   cudaMemAdviseSetPreferredLocation, dev_id));
        CUDA_RT_CALL(cudaMemAdvise(a + (iy_start - 1) * nx, nx * sizeof(real),
                                   cudaMemAdviseSetAccessedBy, dev_id));
        CUDA_RT_CALL(
            cudaMemAdvise(a + iy_end * nx, nx * sizeof(real), cudaMemAdviseSetAccessedBy, dev_id));
        CUDA_RT_CALL(cudaMemAdvise(a_new + iy_start * nx, (iy_end - iy_start) * nx * sizeof(real),
                                   cudaMemAdviseSetPreferredLocation, dev_id));
        CUDA_RT_CALL(cudaMemAdvise(a_new + (iy_start - 1) * nx, nx * sizeof(real),
                                   cudaMemAdviseSetAccessedBy, dev_id));
        CUDA_RT_CALL(cudaMemAdvise(a_new + iy_end * nx, nx * sizeof(real),
                                   cudaMemAdviseSetAccessedBy, dev_id));
#endif  // UM_HINTS

        constexpr int dim_block_x = 32;
        constexpr int dim_block_y = 32;
        dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                      (ny + (num_devices * dim_block_y) - 1) / (num_devices * dim_block_y), 1);

        real* l2_norm_d;
        real* l2_norm_h;

        CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
        CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

        CUDA_RT_CALL(cudaEventCreateWithFlags(compute_done[0] + dev_id, cudaEventDisableTiming));
        CUDA_RT_CALL(cudaEventCreateWithFlags(compute_done[1] + dev_id, cudaEventDisableTiming));

        CUDA_RT_CALL(cudaDeviceSynchronize());

        int iter = 0;
        bool calculate_norm;
#pragma omp master
        { l2_norm = 1.0; }

#pragma omp barrier
        double start = omp_get_wtime();
        PUSH_RANGE("Jacobi solve", 0)
        while (l2_norm > tol && iter < iter_max) {
            CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), 0));

// need to wait for other threads due to sharing of a, a_new and compute_done
// between threads
#pragma omp barrier
            CUDA_RT_CALL(cudaStreamWaitEvent(0, compute_done[iter % 2][top], 0));
            CUDA_RT_CALL(cudaStreamWaitEvent(0, compute_done[iter % 2][bottom], 0));

            calculate_norm = (iter % nccheck) == 0 || (!csv && (iter % 100) == 0);
            jacobi_kernel<dim_block_x, dim_block_y><<<dim_grid, {dim_block_x, dim_block_y, 1}>>>(
                a_new, a, l2_norm_d, iy_start, iy_end, nx, ny, calculate_norm);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaEventRecord(compute_done[(iter + 1) % 2][dev_id], 0));
#pragma omp barrier

            if (calculate_norm) {
                CUDA_RT_CALL(
                    cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost, 0));
#pragma omp barrier
#pragma omp single
                { l2_norm = 0.0; }
#pragma omp barrier
                CUDA_RT_CALL(cudaStreamSynchronize(0));
#pragma omp atomic
                l2_norm += *(l2_norm_h);
#pragma omp barrier
#pragma omp single
                { l2_norm = std::sqrt(l2_norm); }
#pragma omp barrier
                if (!csv && (iter % 100) == 0) {
#pragma omp master
                    printf("%5d, %0.6f\n", iter, l2_norm);
                }
            }

            std::swap(a_new, a);
            iter++;
        }
        CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier
        POP_RANGE
        double stop = omp_get_wtime();

#pragma omp barrier

#pragma omp master
        {
            result_correct = true;
            for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
                for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                    if (std::fabs(a_ref[iy * nx + ix] - a[iy * nx + ix]) > tol) {
                        fprintf(stderr,
                                "ERROR: a[%d * %d + %d] = %f does not match %f "
                                "(reference)\n",
                                iy, nx, ix, a[iy * nx + ix], a_ref[iy * nx + ix]);
                        result_correct = false;
                    }
                }
            }
            if (result_correct) {
                if (csv) {
                    printf("multi_threaded_um, %d, %d, %d, %d, %d, 1, %f, %f\n", nx, ny, iter_max,
                           nccheck, num_devices, (stop - start), runtime_serial);
                } else {
                    printf("Num GPUs: %d.\n", num_devices);
                    printf(
                        "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: "
                        "%8.2f, "
                        "efficiency: %8.2f \n",
                        ny, nx, runtime_serial, num_devices, (stop - start),
                        runtime_serial / (stop - start),
                        runtime_serial / (num_devices * (stop - start)) * 100);
                }
            }
        }

        CUDA_RT_CALL(cudaEventDestroy(compute_done[1][dev_id]));
        CUDA_RT_CALL(cudaEventDestroy(compute_done[0][dev_id]));

        CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
        CUDA_RT_CALL(cudaFree(l2_norm_d));
        CUDA_RT_CALL(cudaDeviceSynchronize());
    }

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));

    CUDA_RT_CALL(cudaFree(a_ref));

    return result_correct ? 0 : 1;
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

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref,
                  const int nccheck, const bool print) {
    real* a;
    real* a_new;

    cudaStream_t compute_stream;
    cudaStream_t push_top_stream;
    cudaStream_t push_bottom_stream;
    cudaEvent_t compute_done;
    cudaEvent_t push_top_done;
    cudaEvent_t push_bottom_done;

    real* l2_norm_d;
    real* l2_norm_h;

    int iy_start = 1;
    int iy_end = (ny - 1);

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, nx, ny);
    CUDA_RT_CALL(cudaGetLastError());
    CUDA_RT_CALL(cudaDeviceSynchronize());

    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_top_stream));
    CUDA_RT_CALL(cudaStreamCreate(&push_bottom_stream));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));

    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf(
            "Single GPU jacobi relaxation: %d iterations on %d x %d mesh with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    int iter = 0;
    bool calculate_norm;
    real l2_norm = 1.0;

    double start = omp_get_wtime();
    PUSH_RANGE("Jacobi solve", 0)
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));

        calculate_norm = (iter % nccheck) == 0 || (print && ((iter % 100) == 0));

        jacobi_kernel<dim_block_x, dim_block_y>
            <<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
                a_new, a, l2_norm_d, iy_start, iy_end, nx, calculate_norm);
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost,
                                         compute_stream));
        }

        // Apply periodic boundary conditions

        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new, a_new + (iy_end - 1) * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, push_top_stream));
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        CUDA_RT_CALL(cudaMemcpyAsync(a_new + iy_end * nx, a_new + iy_start * nx, nx * sizeof(real),
                                     cudaMemcpyDeviceToDevice, compute_stream));
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            l2_norm = *l2_norm_h;
            l2_norm = std::sqrt(l2_norm);
            if (print && (iter % 100) == 0) printf("%5d, %0.6f\n", iter, l2_norm);
        }

        std::swap(a_new, a);
        iter++;
    }
    POP_RANGE
    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));
    return (stop - start);
}
