/* Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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

#include <mpi.h>

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (0 != mpi_status) {                                                        \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
        }                                                                             \
    }

#include <cuda_runtime.h>

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

#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
#else
typedef float real;
#define MPI_REAL_TYPE MPI_FLOAT
#endif

constexpr real tol = 1.0e-8;

const real PI = 2.0 * std::asin(1.0);

void launch_initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                  const real pi, const int offset, const int nx, const int my_ny,
                                  const int ny);

void launch_jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                          real* __restrict__ const l2_norm, const int iy_start, const int iy_end,
                          const int nx, const bool calculate_norm, cudaStream_t stream);

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

int main(int argc, char* argv[]) {
    MPI_CALL(MPI_Init(&argc, &argv));
    int rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");

    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    CUDA_RT_CALL(cudaSetDevice(local_rank));
    CUDA_RT_CALL(cudaFree(0));

    real* a_ref_h;
    CUDA_RT_CALL(cudaMallocHost(&a_ref_h, nx * ny * sizeof(real)));
    real* a_h;
    CUDA_RT_CALL(cudaMallocHost(&a_h, nx * ny * sizeof(real)));
    double runtime_serial = single_gpu(nx, ny, iter_max, a_ref_h, nccheck, !csv && (0 == rank));

    // ny - 2 rows are distributed amongst `size` ranks in such a way
    // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
    // This optimizes load balancing when (ny - 2) % size != 0
    int chunk_size;
    int chunk_size_low = (ny - 2) / size;
    int chunk_size_high = chunk_size_low + 1;
    // To calculate the number of ranks that need to compute an extra row,
    // the following formula is derived from this equation:
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low = size * chunk_size_low + size -
                        (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (rank < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    real* a;
    CUDA_RT_CALL(cudaMalloc(&a, nx * (chunk_size + 2) * sizeof(real)));
    real* a_new;
    CUDA_RT_CALL(cudaMalloc(&a_new, nx * (chunk_size + 2) * sizeof(real)));

    CUDA_RT_CALL(cudaMemset(a, 0, nx * (chunk_size + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * (chunk_size + 2) * sizeof(real)));

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (rank < num_ranks_low) {
        iy_start_global = rank * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array

    int iy_start = 1;
    int iy_end = iy_start + chunk_size;

    // Set diriclet boundary conditions on left and right boarder
    launch_initialize_boundaries(a, a_new, PI, iy_start_global - 1, nx, (chunk_size + 2), ny);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    cudaStream_t compute_stream;
    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));
    cudaEvent_t compute_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));

    real* l2_norm_d;
    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    real* l2_norm_h;
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

    PUSH_RANGE("MPI_Warmup", 5)
    for (int i = 0; i < 10; ++i) {
        const int top = rank > 0 ? rank - 1 : (size - 1);
        const int bottom = (rank + 1) % size;
        MPI_CALL(MPI_Sendrecv(a_new + iy_start * nx, nx, MPI_REAL_TYPE, top, 0,
                              a_new + (iy_end * nx), nx, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                              MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(a_new + (iy_end - 1) * nx, nx, MPI_REAL_TYPE, bottom, 0, a_new, nx,
                              MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        std::swap(a_new, a);
    }
    POP_RANGE

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (!csv && 0 == rank) {
        printf(
            "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
            "every %d iterations\n",
            iter_max, ny, nx, nccheck);
    }

    int iter = 0;
    real l2_norm = 1.0;
    bool calculate_norm;  // boolean to store whether l2 norm will be calculated in
                          //   an iteration or not

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    PUSH_RANGE("Jacobi solve", 0)
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

        calculate_norm = (iter % nccheck) == 0 || (!csv && (iter % 100) == 0);

        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx, calculate_norm,
                             compute_stream);
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        if (calculate_norm) {
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost,
                                         compute_stream));
        }

        const int top = rank > 0 ? rank - 1 : (size - 1);
        const int bottom = (rank + 1) % size;

        // Apply periodic boundary conditions
        CUDA_RT_CALL(cudaEventSynchronize(compute_done));
        PUSH_RANGE("MPI", 5)
        MPI_CALL(MPI_Sendrecv(a_new + iy_start * nx, nx, MPI_REAL_TYPE, top, 0,
                              a_new + (iy_end * nx), nx, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                              MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(a_new + (iy_end - 1) * nx, nx, MPI_REAL_TYPE, bottom, 0, a_new, nx,
                              MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        POP_RANGE

        if (calculate_norm) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            MPI_CALL(MPI_Allreduce(l2_norm_h, &l2_norm, 1, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD));
            l2_norm = std::sqrt(l2_norm);

            if (!csv && 0 == rank && (iter % 100) == 0) {
                printf("%5d, %0.6f\n", iter, l2_norm);
            }
        }

        std::swap(a_new, a);
        iter++;
    }
    double stop = MPI_Wtime();
    POP_RANGE

    CUDA_RT_CALL(cudaMemcpy(a_h + iy_start_global * nx, a + nx,
                            std::min((ny - iy_start_global) * nx, chunk_size * nx) * sizeof(real),
                            cudaMemcpyDeviceToHost));

    int result_correct = 1;
    for (int iy = iy_start_global; result_correct && (iy < iy_end_global); ++iy) {
        for (int ix = 1; result_correct && (ix < (nx - 1)); ++ix) {
            if (std::fabs(a_ref_h[iy * nx + ix] - a_h[iy * nx + ix]) > tol) {
                fprintf(stderr,
                        "ERROR on rank %d: a[%d * %d + %d] = %f does not match %f "
                        "(reference)\n",
                        rank, iy, nx, ix, a_h[iy * nx + ix], a_ref_h[iy * nx + ix]);
                result_correct = 0;
            }
        }
    }

    int global_result_correct = 1;
    MPI_CALL(MPI_Allreduce(&result_correct, &global_result_correct, 1, MPI_INT, MPI_MIN,
                           MPI_COMM_WORLD));
    result_correct = global_result_correct;

    if (rank == 0 && result_correct) {
        if (csv) {
            printf("mpi, %d, %d, %d, %d, %d, 1, %f, %f\n", nx, ny, iter_max, nccheck, size,
                   (stop - start), runtime_serial);
        } else {
            printf("Num GPUs: %d.\n", size);
            printf(
                "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
                "efficiency: %8.2f \n",
                ny, nx, runtime_serial, size, (stop - start), runtime_serial / (stop - start),
                runtime_serial / (size * (stop - start)) * 100);
        }
    }
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    CUDA_RT_CALL(cudaFree(a_new));
    CUDA_RT_CALL(cudaFree(a));

    CUDA_RT_CALL(cudaFreeHost(a_h));
    CUDA_RT_CALL(cudaFreeHost(a_ref_h));

    MPI_CALL(MPI_Finalize());
    return (result_correct == 1) ? 0 : 1;
}

double single_gpu(const int nx, const int ny, const int iter_max, real* const a_ref_h,
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
    launch_initialize_boundaries(a, a_new, PI, 0, nx, ny, ny);
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

    int iter = 0;
    real l2_norm = 1.0;
    bool calculate_norm;

    double start = MPI_Wtime();
    PUSH_RANGE("Jacobi solve", 0)
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync(l2_norm_d, 0, sizeof(real), compute_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));

        calculate_norm = (iter % nccheck) == 0 || (iter % 100) == 0;
        launch_jacobi_kernel(a_new, a, l2_norm_d, iy_start, iy_end, nx, calculate_norm,
                             compute_stream);
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
    double stop = MPI_Wtime();

    CUDA_RT_CALL(cudaMemcpy(a_ref_h, a, nx * ny * sizeof(real), cudaMemcpyDeviceToHost));

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
