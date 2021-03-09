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

template <int BLOCK_DIM_X, int BLOCK_DIM_Y>
__global__ void jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              real* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, real* __restrict__ const a_new_top,
                              const int top_iy, real* __restrict__ const a_new_bottom,
                              const int bottom_iy) {
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
        real residue = new_val - a[iy * nx + ix];
        local_l2_norm += residue * residue;
    }
#ifdef HAVE_CUB
    real block_l2_norm = BlockReduce(temp_storage).Sum(local_l2_norm);
    if (0 == threadIdx.y && 0 == threadIdx.x) atomicAdd(l2_norm, block_l2_norm);
#else
    atomicAdd(l2_norm, local_l2_norm);
#endif  // HAVE_CUB
}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h,
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

    real* a_new[MAX_NUM_DEVICES];

    real* a_ref_h;
    real* a_h;
    double runtime_serial = 0.0;

    int iy_end[MAX_NUM_DEVICES];

    cudaEvent_t compute_done[2][MAX_NUM_DEVICES];
    cudaEvent_t reset_l2_norm_done[2][MAX_NUM_DEVICES];

    bool result_correct = true;
    bool p2p_works = true;
    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    real l2_norms[2];

#pragma omp parallel num_threads(num_devices) shared(l2_norms)
    {
        real* a;

        cudaStream_t compute_stream;
        cudaStream_t reset_l2_norm_stream;

        l2_norm_buf l2_norm_bufs[2];

        int dev_id = omp_get_thread_num();

        CUDA_RT_CALL(cudaSetDevice(dev_id));
        CUDA_RT_CALL(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        CUDA_RT_CALL(cudaFree(0));

        if (0 == dev_id) {
            CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
            CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));
            runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, nccheck, !csv);
        }
#pragma omp barrier
        // ny - 2 rows are distributed amongst `size` ranks in such a way
        // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
        // This optimizes load balancing when (ny - 2) % size != 0
        int chunk_size;
        int chunk_size_low = (ny - 2) / num_devices;
        int chunk_size_high = chunk_size_low + 1;
        // To calculate the number of ranks that need to compute an extra row,
        // the following formula is derived from this equation:
        // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
        int num_ranks_low = num_devices * chunk_size_low + num_devices -
                            (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
        if (dev_id < num_ranks_low)
            chunk_size = chunk_size_low;
        else
            chunk_size = chunk_size_high;

        const int top = dev_id > 0 ? dev_id - 1 : (num_devices - 1);
        const int bottom = (dev_id + 1) % num_devices;
        if (top != dev_id) {
            int canAccessPeer = 0;
            CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, top));
            if (canAccessPeer) {
                CUDA_RT_CALL(cudaDeviceEnablePeerAccess(top, 0));
            } else {
                std::cerr << "P2P access required from " << dev_id << " to " << top << std::endl;
#pragma omp critical
                {
                    if (p2p_works) p2p_works = false;
                }
            }
            if (top != bottom) {
                canAccessPeer = 0;
                CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, dev_id, bottom));
                if (canAccessPeer) {
                    CUDA_RT_CALL(cudaDeviceEnablePeerAccess(bottom, 0));
                } else {
                    std::cerr << "P2P access required from " << dev_id << " to " << bottom
                              << std::endl;
#pragma omp critical
                    {
                        if (p2p_works) p2p_works = false;
                    }
                }
            }
        }

#pragma omp barrier
        if (p2p_works) {
            CUDA_RT_CALL(cudaMalloc(&a, nx * (chunk_size + 2) * sizeof(real)));
            CUDA_RT_CALL(cudaMalloc(a_new + dev_id, nx * (chunk_size + 2) * sizeof(real)));

            CUDA_RT_CALL(cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(real)));
            CUDA_RT_CALL(cudaMemset(a_new[dev_id], 0, nx * (chunk_size + 2) * sizeof(real)));

            // Calculate local domain boundaries
            int iy_start_global;  // My start index in the global array
            if (dev_id < num_ranks_low) {
                iy_start_global = dev_id * chunk_size_low + 1;
            } else {
                iy_start_global =
                    num_ranks_low * chunk_size_low + (dev_id - num_ranks_low) * chunk_size_high + 1;
            }
            int iy_end_global =
                iy_start_global + chunk_size - 1;  // My last index in the global array

            int iy_start = 1;
            iy_end[dev_id] = (iy_end_global - iy_start_global + 1) + iy_start;

            // Set diriclet boundary conditions on left and right boarder
            initialize_boundaries<<<(ny / num_devices) / 128 + 1, 128>>>(
                a, a_new[dev_id], PI, iy_start_global - 1, nx, (chunk_size + 2), ny);
            CUDA_RT_CALL(cudaGetLastError());
            CUDA_RT_CALL(cudaDeviceSynchronize());

            CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
            CUDA_RT_CALL(cudaStreamCreate(&reset_l2_norm_stream));

            CUDA_RT_CALL(
                cudaEventCreateWithFlags(compute_done[0] + dev_id, cudaEventDisableTiming));
            CUDA_RT_CALL(
                cudaEventCreateWithFlags(compute_done[1] + dev_id, cudaEventDisableTiming));
            CUDA_RT_CALL(
                cudaEventCreateWithFlags(reset_l2_norm_done[0] + dev_id, cudaEventDisableTiming));
            CUDA_RT_CALL(
                cudaEventCreateWithFlags(reset_l2_norm_done[1] + dev_id, cudaEventDisableTiming));

            for (int i = 0; i < 2; ++i) {
                CUDA_RT_CALL(
                    cudaEventCreateWithFlags(&l2_norm_bufs[i].copy_done, cudaEventDisableTiming));
                CUDA_RT_CALL(cudaMalloc(&l2_norm_bufs[i].d, sizeof(real)));
                CUDA_RT_CALL(cudaMemset(l2_norm_bufs[i].d, 0, sizeof(real)));
                CUDA_RT_CALL(cudaMallocHost(&l2_norm_bufs[i].h, sizeof(real)));
                *(l2_norm_bufs[i].h) = 1.0;
            }

            CUDA_RT_CALL(cudaDeviceSynchronize());

#pragma omp master
            {
                if (!csv)
                    printf(
                        "Jacobi relaxation: %d iterations on %d x %d mesh with "
                        "norm "
                        "check every %d iterations\n",
                        iter_max, ny, nx, nccheck);
            }

            constexpr int dim_block_x = 32;
            constexpr int dim_block_y = 32;
            dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x,
                          (ny + (num_devices * dim_block_y) - 1) / (num_devices * dim_block_y), 1);

            int iter = 0;
#pragma omp master
            {
                for (int i = 0; i < 2; ++i) {
                    l2_norms[i] = 1.0;
                }
            }

            CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier
            double start = omp_get_wtime();

            PUSH_RANGE("Jacobi solve", 0)

            bool l2_norm_greater_than_tol = true;
            while (l2_norm_greater_than_tol && iter < iter_max) {
                // on new iteration: old current vars are now previous vars, old
                // previous vars are no longer needed
                int prev = iter % 2;
                int curr = (iter + 1) % 2;

// need to wait for other threads due to sharing of a_new and compute_done
// between threads
#pragma omp barrier
                CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, compute_done[prev][top], 0));
                CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, compute_done[prev][bottom], 0));
                CUDA_RT_CALL(
                    cudaStreamWaitEvent(compute_stream, reset_l2_norm_done[curr][dev_id], 0));
                jacobi_kernel<dim_block_x, dim_block_y>
                    <<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
                        a_new[dev_id], a, l2_norm_bufs[curr].d, iy_start, iy_end[dev_id], nx,
                        a_new[top], iy_end[top], a_new[bottom], 0);
                CUDA_RT_CALL(cudaGetLastError());
                CUDA_RT_CALL(cudaEventRecord(compute_done[curr][dev_id], compute_stream));

                // perform L2 norm calculation
                if ((iter % nccheck) == 0 || (!csv && (iter % 100) == 0)) {
                    // as soon as computation is complete -> D2H-copy L2 norm
                    CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_bufs[curr].h, l2_norm_bufs[curr].d,
                                                 sizeof(real), cudaMemcpyDeviceToHost,
                                                 compute_stream));
                    CUDA_RT_CALL(cudaEventRecord(l2_norm_bufs[curr].copy_done, compute_stream));

                    // ensure previous D2H-copy is completed before using the
                    // data for calculation
                    CUDA_RT_CALL(cudaEventSynchronize(l2_norm_bufs[prev].copy_done));

/*
 * using atomics instead of critical sections caused a minimal (100ns /
 * iteration) performance gain
 */
#pragma omp atomic
                    l2_norms[prev] += *(l2_norm_bufs[prev].h);

#pragma omp barrier
                    const real l2_norm_prev = std::sqrt(l2_norms[prev]);
                    l2_norm_greater_than_tol = (l2_norm_prev > tol);

                    if (!csv && (iter % 100) == 0) {
#pragma omp single
                        printf("%5d, %0.6f\n", iter, l2_norm_prev);
                    }
#pragma omp barrier

                    // reset everything for next iteration
                    l2_norms[prev] = 0.0;
                    *(l2_norm_bufs[prev].h) = 0.0;
                    CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_bufs[prev].d, l2_norm_bufs[curr].h,
                                                 sizeof(real), cudaMemcpyHostToDevice,
                                                 reset_l2_norm_stream));
                    CUDA_RT_CALL(
                        cudaEventRecord(reset_l2_norm_done[prev][dev_id], reset_l2_norm_stream));
                } else {
#pragma omp barrier
                }

                std::swap(a_new[dev_id], a);
                iter++;
            }
            CUDA_RT_CALL(cudaDeviceSynchronize());
#pragma omp barrier
            double stop = omp_get_wtime();
            POP_RANGE

            CUDA_RT_CALL(
                cudaMemcpy(a_h + iy_start_global * nx, a + nx,
                           std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                           cudaMemcpyDeviceToHost));
#pragma omp barrier

#pragma omp master
            {
                result_correct = true;
                for (int iy = 1; result_correct && (iy < (ny - 1)); ++iy) {
                    for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
                        if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                            fprintf(stderr,
                                    "ERROR: a[%d * %d + %d] = %f does not "
                                    "match %f (reference)\n",
                                    iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                            result_correct = false;
                        }
                    }
                }
                if (result_correct) {
                    if (csv) {
                        printf(
                            "multi_threaded_p2p_opt, %d, %d, %d, %d, %d, 1, "
                            "%f, %f\n",
                            nx, ny, iter_max, nccheck, num_devices, (stop - start), runtime_serial);
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

            for (int i = 0; i < 2; ++i) {
                CUDA_RT_CALL(cudaFreeHost(l2_norm_bufs[i].h));
                CUDA_RT_CALL(cudaFree(l2_norm_bufs[i].d));
                CUDA_RT_CALL(cudaEventDestroy(l2_norm_bufs[i].copy_done));
            }

            CUDA_RT_CALL(cudaEventDestroy(reset_l2_norm_done[1][dev_id]));
            CUDA_RT_CALL(cudaEventDestroy(reset_l2_norm_done[0][dev_id]));
            CUDA_RT_CALL(cudaEventDestroy(compute_done[1][dev_id]));
            CUDA_RT_CALL(cudaEventDestroy(compute_done[0][dev_id]));
            CUDA_RT_CALL(cudaStreamDestroy(reset_l2_norm_stream));
            CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

            CUDA_RT_CALL(cudaFree(a_new[dev_id]));
            CUDA_RT_CALL(cudaFree(a));
            if (0 == dev_id) {
                CUDA_RT_CALL(cudaFreeHost(a_h));
                CUDA_RT_CALL(cudaFreeHost(a_ref_h));
            }
        }
    }

    return result_correct ? 0 : 1;
}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h,
                  const int nccheck, const bool print) {
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

    CUDA_RT_CALL(cudaMalloc(&a, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * ny * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * ny * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * ny * sizeof(real)));

    // Set diriclet boundary conditions on left and right boarder
    initialize_boundaries<<<ny / 128 + 1, 128>>>(a, a_new, PI, 0, nx, ny, ny);
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
        *(l2_norm_bufs[i].h) = 1.0;
    }

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (print)
        printf(
            "Single GPU Jacobi relaxation: %d iterations on %d x %d mesh with "
            "norm "
            "check every %d iterations\n",
            iter_max, ny, nx, nccheck);

    constexpr int dim_block_x = 32;
    constexpr int dim_block_y = 32;
    dim3 dim_grid((nx + dim_block_x - 1) / dim_block_x, (ny + dim_block_y - 1) / dim_block_y, 1);

    int iter = 0;
    for (int i = 0; i < 2; ++i) {
        l2_norms[i] = 1.0;
    }

    double start = omp_get_wtime();

    PUSH_RANGE("Jacobi solve", 0)

    bool l2_norm_greater_than_tol = true;
    while (l2_norm_greater_than_tol && iter < iter_max) {
        // on new iteration: old current vars are now previous vars, old
        // previous vars are no longer needed
        int prev = iter % 2;
        int curr = (iter + 1) % 2;

        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, reset_l2_norm_done[curr], 0));
        jacobi_kernel<dim_block_x, dim_block_y>
            <<<dim_grid, {dim_block_x, dim_block_y, 1}, 0, compute_stream>>>(
                a_new, a, l2_norm_bufs[curr].d, iy_start, iy_end, nx, a_new, iy_end, a_new,
                (iy_start - 1));
        CUDA_RT_CALL(cudaGetLastError());
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        // perform L2 norm calculation
        if ((iter % nccheck) == 0 || (print && (iter % 100) == 0)) {
            // as soon as computation is complete -> D2H-copy L2 norm
            CUDA_RT_CALL(cudaStreamWaitEvent(copy_l2_norm_stream, compute_done, 0));
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_bufs[curr].h, l2_norm_bufs[curr].d, sizeof(real),
                                         cudaMemcpyDeviceToHost, copy_l2_norm_stream));
            CUDA_RT_CALL(cudaEventRecord(l2_norm_bufs[curr].copy_done, copy_l2_norm_stream));

            // ensure previous D2H copy is completed before using the data for
            // calculation
            CUDA_RT_CALL(cudaEventSynchronize(l2_norm_bufs[prev].copy_done));

            l2_norms[prev] = *(l2_norm_bufs[prev].h);
            l2_norms[prev] = std::sqrt(l2_norms[prev]);
            l2_norm_greater_than_tol = (l2_norms[prev] > tol);

            if (print && (iter % 100) == 0) {
                printf("%5d, %0.6f\n", iter, l2_norms[prev]);
            }

            // reset everything for next iteration
            l2_norms[prev] = 0.0;
            *(l2_norm_bufs[prev].h) = 0.0;
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_bufs[prev].d, l2_norm_bufs[prev].h, sizeof(real),
                                         cudaMemcpyHostToDevice, reset_l2_norm_stream));
            CUDA_RT_CALL(cudaEventRecord(reset_l2_norm_done[prev], reset_l2_norm_stream));
        }

        std::swap(a_new, a);
        iter++;
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    POP_RANGE
    double stop = omp_get_wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

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

    return (stop - start);
}
