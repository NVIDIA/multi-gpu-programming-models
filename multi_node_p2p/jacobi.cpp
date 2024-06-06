/* Copyright (c) 2017, 2024, NVIDIA CORPORATION. All rights reserved.
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
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>

#include <mpi.h>

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (MPI_SUCCESS != mpi_status) {                                              \
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
            exit(mpi_status);                                                         \
        }                                                                             \
    }

#include <cuda.h>

#define CUDA_CALL(call)                                                            \
    {                                                                              \
        CUresult cudaStatus = call;                                                \
        if (CUDA_SUCCESS != cudaStatus) {                                          \
            const char* error_string;                                              \
            cuGetErrorString(cudaStatus, &error_string);                           \
            fprintf(stderr,                                                        \
                    "ERROR: CUDA Driver call \"%s\" in line %d of file %s failed " \
                    "with "                                                        \
                    "%s (%d).\n",                                                  \
                    #call, __LINE__, __FILE__, error_string, cudaStatus);          \
            exit(cudaStatus);                                                      \
        }                                                                          \
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

// Group l2 norm and barrier counter for simplified handling of Multi Cast (MC) memory
struct real_int_pair {
    real value;
    unsigned int arrival_counter;
};

constexpr real tol = 1.0e-8;

const real PI = 2.0 * std::asin(1.0);

void launch_initialize_boundaries(real* __restrict__ const a_new, real* __restrict__ const a,
                                  const real pi, const int offset, const int nx, const int my_ny,
                                  const int ny);

void launch_jacobi_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                          real* __restrict__ const l2_norm, const int iy_start, const int iy_end,
                          const int nx, const bool calculate_norm, cudaStream_t stream);

void launch_jacobi_p2p_kernel(real* __restrict__ const a_new, const real* __restrict__ const a,
                              real* __restrict__ const l2_norm, const int iy_start,
                              const int iy_end, const int nx, real* __restrict__ const a_new_top,
                              const int top_iy, real* __restrict__ const a_new_bottom,
                              const int bottom_iy, const bool calculate_norm, cudaStream_t stream);

void launch_all_reduce_norm_barrier_kernel(real* __restrict__ const l2_norm,
                                           real_int_pair* __restrict__ partial_l2_norm_uc,
                                           real_int_pair* __restrict__ partial_l2_norm_mc,
                                           const int num_gpus, const int iter, cudaStream_t stream);

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

template <typename T>
T round_up(const T value, const T granularity) {
    return (value + granularity - 1) & ~(granularity - 1);
}

struct fabric_mem {
    CUdeviceptr ptr;
    size_t size;
    size_t aligned_size;
    size_t granularity;
    int device_id;
    CUmemGenericAllocationHandle generic_handle;
    CUmemFabricHandle fabric_handle;
    CUdeviceptr ptr_top;
    CUmemGenericAllocationHandle generic_handle_top;
    CUdeviceptr ptr_bottom;
    CUmemGenericAllocationHandle generic_handle_bottom;
};

struct uc_mc_pair {
    CUdeviceptr uc_ptr;
    CUdeviceptr mc_ptr;
    CUmemGenericAllocationHandle mc_handle;
    CUmemGenericAllocationHandle uc_handle;
    size_t mc_size;
    size_t uc_size;
};

/**
 * Allocate sharable fabric memory:
 * 1. Create a CUDA memory handle for the allocation (cuMemCreate)
 * 2. Export handle to fabric handle for later exchange (cuMemExportToShareableHandle)
 * 3. Reserve address range for allocation (cuMemAddressReserve)
 * 4. Map allocation into address range (cuMemMap)
 * 5. Make allocation accessible (cuMemSetAccess)
 */
void allocate_fabric_mem(fabric_mem& fm, const size_t size, const int device_id) {
    fm.device_id = device_id;
    // It is required to use a socket in the Unix domain with
    // CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR to exchange sharable handles between processes. To
    // enable communcation of shareable handles via MPI we need CU_MEM_HANDLE_TYPE_FABRIC so using
    // that here.
    const CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_FABRIC;
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = handle_type;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = fm.device_id;

    CUDA_CALL(cuMemGetAllocationGranularity(&fm.granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    fm.size = size;
    fm.aligned_size = round_up(fm.size, fm.granularity);

    CUDA_CALL(cuMemCreate(&fm.generic_handle, fm.aligned_size, &prop, 0 /*flags*/));

    CUDA_CALL(cuMemExportToShareableHandle(&fm.fabric_handle, fm.generic_handle, handle_type,
                                           0 /*flags*/));

    CUDA_CALL(cuMemAddressReserve(&fm.ptr, fm.aligned_size, fm.granularity, 0 /*baseVA*/, 0 /*flags*/));

    CUDA_CALL(cuMemMap(fm.ptr, fm.aligned_size, 0 /*offset*/, fm.generic_handle, 0 /*flags*/));

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = prop.location.id;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_CALL(cuMemSetAccess(fm.ptr, fm.aligned_size, &desc, 1 /*count*/));
}

void free_fabric_mem(fabric_mem& fm) {
    CUDA_CALL(cuMemUnmap(fm.ptr, fm.aligned_size));
    CUDA_CALL(cuMemRelease(fm.generic_handle));
    CUDA_CALL(cuMemAddressFree(fm.ptr, fm.aligned_size));
    // Fabric handle does not hold any resources so doe not need to be freed
}

/**
 * Map peer memory for direct load store access:
 * 1. Reserve address range for peer allocation (cuMemAddressReserve)
 * 2. Map peer memory into address range (cuMemMap)
 * 3. Make allocation accessible (cuMemSetAccess)
 */
void map_peers(fabric_mem& fm) {
    CUDA_CALL(cuMemAddressReserve(&fm.ptr_top, fm.aligned_size, fm.granularity, 0 /*baseVA*/,
                                  0 /*flags*/));
    CUDA_CALL(cuMemAddressReserve(&fm.ptr_bottom, fm.aligned_size, fm.granularity, 0 /*baseVA*/,
                                  0 /*flags*/));

    CUDA_CALL(cuMemMap(fm.ptr_top, fm.aligned_size, 0 /*offset*/, fm.generic_handle_top, 0 /*flags*/));
    CUDA_CALL(cuMemMap(fm.ptr_bottom, fm.aligned_size, 0 /*offset*/, fm.generic_handle_bottom,
                       0 /*flags*/));

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = fm.device_id;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUDA_CALL(cuMemSetAccess(fm.ptr_top, fm.aligned_size, &desc, 1 /*count*/));
    CUDA_CALL(cuMemSetAccess(fm.ptr_bottom, fm.aligned_size, &desc, 1 /*count*/));
}

void unmap_peers(fabric_mem& fm) {
    CUDA_CALL(cuMemUnmap(fm.ptr_top, fm.aligned_size));
    CUDA_CALL(cuMemUnmap(fm.ptr_bottom, fm.aligned_size));

    CUDA_CALL(cuMemRelease(fm.generic_handle_top));
    CUDA_CALL(cuMemRelease(fm.generic_handle_bottom));

    CUDA_CALL(cuMemAddressFree(fm.ptr_top, fm.aligned_size));
    CUDA_CALL(cuMemAddressFree(fm.ptr_bottom, fm.aligned_size));
}

int main(int argc, char* argv[]) {
    MPI_CALL(MPI_Init(&argc, &argv));
    int rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));
    int num_devices = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const bool csv = get_arg(argv, argv + argc, "-csv");
    const bool use_mc_red = get_arg(argv, argv + argc, "-use_mc_red");

    int local_rank = -1;
    int local_size = 1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CALL(MPI_Comm_size(local_comm, &local_size));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    const int device_id = local_rank % num_devices;
    CUDA_RT_CALL(cudaSetDevice(device_id));
    CUDA_RT_CALL(cudaFree(0));

    int fabric_handle_supported = 0;
    CUDA_CALL(cuDeviceGetAttribute(&fabric_handle_supported,
                                   CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device_id));
    if (!fabric_handle_supported) {
        cudaDeviceProp prop;
        CUDA_RT_CALL(cudaGetDeviceProperties(&prop, device_id));
        fprintf(stderr, "ERROR: Creating fabric handles is not supported on device %d (%s)\n",
                device_id, prop.name);
        MPI_CALL(MPI_Finalize());
        return -1;
    }

    if (use_mc_red) {
        int multicast_supported = 0;
        CUDA_CALL(cuDeviceGetAttribute(&multicast_supported,
                                       CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device_id));
        if (!multicast_supported) {
            cudaDeviceProp prop;
            CUDA_RT_CALL(cudaGetDeviceProperties(&prop, device_id));
            fprintf(stderr,
                    "ERROR: Creating Multicast Objects is not supported on device %d (%s)\n",
                    device_id, prop.name);
            MPI_CALL(MPI_Finalize());
            return -1;
        }

        if (1 < num_devices && num_devices < local_size) {
            fprintf(stderr,
                    "ERROR: Creating Multicast Objects is not supported when oversubscribing a GPU "
                    "with MPS: %d ranks using %d (< %d) devices!\n",
                    local_size, num_devices, local_size);
            MPI_CALL(MPI_Finalize());
            return 1;
        }
    }

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

    // Need to allocate with chunk_size_high on all ranks to ensure consistent sizes when mapping
    // peer memory
    fabric_mem a_fa;
    allocate_fabric_mem(a_fa, (nx * (chunk_size_high + 2) * sizeof(real)), device_id);
    real* a = reinterpret_cast<real*>(a_fa.ptr);
    fabric_mem a_new_fa;
    allocate_fabric_mem(a_new_fa, (nx * (chunk_size_high + 2) * sizeof(real)), device_id);
    real* a_new = reinterpret_cast<real*>(a_new_fa.ptr);

    CUDA_RT_CALL(cudaMemset(a, 0, nx * (chunk_size_high + 2) * sizeof(real)));
    CUDA_RT_CALL(cudaMemset(a_new, 0, nx * (chunk_size_high + 2) * sizeof(real)));

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

    int iy_end_top;
    {
        // Map memory of top and bottom peers to allow direct writes of halo data:
        // 1. Exchange shareable fabric handles with MPI
        // 2. Map fabric handles into local memory
        // 3. Exchange top peers bottom boundary index (`iy_end_top`)
        const int top = rank > 0 ? rank - 1 : (size - 1);
        const int bottom = (rank + 1) % size;

        CUmemFabricHandle fabric_handle_a_top;
        CUmemFabricHandle fabric_handle_a_bottom;
        MPI_CALL(MPI_Sendrecv(&a_fa.fabric_handle, sizeof(CUmemFabricHandle), MPI_BYTE, top, 0,
                              &fabric_handle_a_bottom, sizeof(CUmemFabricHandle), MPI_BYTE, bottom, 0, 
                              MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
        MPI_CALL(MPI_Sendrecv(&a_fa.fabric_handle, sizeof(CUmemFabricHandle), MPI_BYTE, bottom, 0,
                              &fabric_handle_a_top, sizeof(CUmemFabricHandle), MPI_BYTE, top, 0,
                              MPI_COMM_WORLD, MPI_STATUSES_IGNORE));

        MPI_CALL(MPI_Sendrecv(&iy_end, 1, MPI_INT, bottom, 0,
                              &iy_end_top, 1, MPI_INT, top, 0,
                              MPI_COMM_WORLD, MPI_STATUSES_IGNORE));

        CUmemFabricHandle fabric_handle_a_new_top;
        CUmemFabricHandle fabric_handle_a_new_bottom;
        MPI_CALL(MPI_Sendrecv(&a_new_fa.fabric_handle, sizeof(CUmemFabricHandle), MPI_BYTE, top, 0,
                              &fabric_handle_a_new_bottom, sizeof(CUmemFabricHandle), MPI_BYTE, bottom, 0,
                              MPI_COMM_WORLD, MPI_STATUSES_IGNORE));
        MPI_CALL(MPI_Sendrecv(&a_new_fa.fabric_handle, sizeof(CUmemFabricHandle), MPI_BYTE, bottom, 0,
                              &fabric_handle_a_new_top, sizeof(CUmemFabricHandle), MPI_BYTE, top, 0,
                              MPI_COMM_WORLD, MPI_STATUSES_IGNORE));

        const CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_FABRIC;

        CUDA_CALL(cuMemImportFromShareableHandle(&a_fa.generic_handle_top, &fabric_handle_a_top,
                                                 handle_type));
        CUDA_CALL(cuMemImportFromShareableHandle(&a_fa.generic_handle_bottom,
                                                 &fabric_handle_a_bottom, handle_type));
        CUDA_CALL(cuMemImportFromShareableHandle(&a_new_fa.generic_handle_top,
                                                 &fabric_handle_a_new_top, handle_type));
        CUDA_CALL(cuMemImportFromShareableHandle(&a_new_fa.generic_handle_bottom,
                                                 &fabric_handle_a_new_bottom, handle_type));

        map_peers(a_fa);
        map_peers(a_new_fa);
    }
    real* a_top = reinterpret_cast<real*>(a_fa.ptr_top);
    real* a_bottom = reinterpret_cast<real*>(a_fa.ptr_bottom);
    real* a_new_top = reinterpret_cast<real*>(a_new_fa.ptr_top);
    real* a_new_bottom = reinterpret_cast<real*>(a_new_fa.ptr_bottom);

    // Set Dirichlet boundary conditions on left and right borders
    launch_initialize_boundaries(a, a_new, PI, iy_start_global - 1, nx, (chunk_size + 2), ny);
    CUDA_RT_CALL(cudaDeviceSynchronize());

    cudaStream_t compute_stream;
    CUDA_RT_CALL(cudaStreamCreate(&compute_stream));

    real* l2_norm_d;
    CUDA_RT_CALL(cudaMalloc(&l2_norm_d, sizeof(real)));
    real* l2_norm_h;
    CUDA_RT_CALL(cudaMallocHost(&l2_norm_h, sizeof(real)));

    uc_mc_pair partial_l2_handles;
    real_int_pair* partial_l2_norm = nullptr;
    real_int_pair* partial_l2_norm_mc = nullptr;
    if (use_mc_red) {
        const CUmemAllocationHandleType handle_type = CU_MEM_HANDLE_TYPE_FABRIC;

        // Get the minimum/recommended granularity for the multicast object
        CUmulticastObjectProp mc_prop = {};
        mc_prop.numDevices = size;
        mc_prop.size = sizeof(real_int_pair);
        mc_prop.handleTypes = handle_type;

        size_t min_granularity;
        size_t granularity;
        CUDA_CALL(cuMulticastGetGranularity(&min_granularity, &mc_prop,
                                            CU_MULTICAST_GRANULARITY_MINIMUM));
        CUDA_CALL(cuMulticastGetGranularity(&granularity, &mc_prop,
                                            CU_MULTICAST_GRANULARITY_RECOMMENDED));

        mc_prop.size = round_up(mc_prop.size, granularity);
        partial_l2_handles.mc_size = mc_prop.size;

        partial_l2_handles.uc_size = round_up(sizeof(real_int_pair), min_granularity);

        CUmemFabricHandle fh;
        if (0 == rank) {
            // Allocate the multicast object
            CUDA_CALL(cuMulticastCreate(&partial_l2_handles.mc_handle, &mc_prop));

            CUDA_CALL(cuMemExportToShareableHandle(&fh, partial_l2_handles.mc_handle, handle_type, 0));
        }
        MPI_CALL(MPI_Bcast(&fh, sizeof(CUmemFabricHandle), MPI_BYTE, 0, MPI_COMM_WORLD));
        if (0 != rank) {
            CUDA_CALL(cuMemImportFromShareableHandle(&partial_l2_handles.mc_handle, &fh, handle_type));
        }
        CUDA_CALL(cuMulticastAddDevice(partial_l2_handles.mc_handle, device_id));

        // Ensure all devices in this process are added BEFORE binding mem on any device
        MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        prop.requestedHandleTypes = handle_type;

        CUDA_CALL(cuMemCreate(&partial_l2_handles.uc_handle, partial_l2_handles.uc_size, &prop,
                              0 /*flags*/));
        CUDA_CALL(cuMulticastBindMem(partial_l2_handles.mc_handle, 0, partial_l2_handles.uc_handle,
                                     0, partial_l2_handles.uc_size, 0));

        // MC Mapping
        CUDA_CALL(cuMemAddressReserve(&partial_l2_handles.mc_ptr, mc_prop.size, granularity,
                                      0 /*baseVA*/, 0 /*flags*/));
        CUDA_CALL(cuMemMap(partial_l2_handles.mc_ptr, mc_prop.size, 0 /*offset*/,
                           partial_l2_handles.mc_handle, 0 /*flags*/));
        CUmemAccessDesc desc = {};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = device_id;
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CUDA_CALL(cuMemSetAccess(partial_l2_handles.mc_ptr, mc_prop.size, &desc, 1 /*count*/));

        // UC Mapping
        CUDA_CALL(cuMemAddressReserve(&partial_l2_handles.uc_ptr, partial_l2_handles.uc_size,
                                      granularity, 0 /*baseVA*/, 0 /*flags*/));
        CUDA_CALL(cuMemMap(partial_l2_handles.uc_ptr, partial_l2_handles.uc_size, 0 /*offset*/,
                           partial_l2_handles.uc_handle, 0 /*flags*/));
        CUDA_CALL(cuMemSetAccess(partial_l2_handles.uc_ptr, partial_l2_handles.uc_size, &desc, 1 /*count*/));

        partial_l2_norm = reinterpret_cast<real_int_pair*>(partial_l2_handles.uc_ptr);
        partial_l2_norm_mc = reinterpret_cast<real_int_pair*>(partial_l2_handles.mc_ptr);

        real_int_pair partial_l2_norm_init;
        partial_l2_norm_init.value = 0.0;
        partial_l2_norm_init.arrival_counter = 0;
        CUDA_RT_CALL(cudaMemcpy(partial_l2_norm, &partial_l2_norm_init, sizeof(real_int_pair), cudaMemcpyHostToDevice));
    }

    if (!csv && 0 == rank) {
        printf(
            "Jacobi relaxation: %d iterations on %d x %d mesh with norm check "
            "every %d iterations\n",
            iter_max, ny, nx, nccheck);
    }

    int iter = 0;
    real l2_norm = 1.0;
    bool calculate_norm = true;  // boolean to store whether l2 norm will be calculated in
                                 // an iteration or not

    CUDA_RT_CALL(cudaDeviceSynchronize());
    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    PUSH_RANGE("Jacobi solve", 0)
    while (l2_norm > tol && iter < iter_max) {
        CUDA_RT_CALL(cudaMemsetAsync((use_mc_red ? &(partial_l2_norm->value) : l2_norm_d), 0,
                                     sizeof(real), compute_stream));

        calculate_norm = (iter % nccheck) == 0 || (!csv && (iter % 100) == 0);

        launch_jacobi_p2p_kernel(a_new, a, (use_mc_red ? &(partial_l2_norm->value) : l2_norm_d),
                                 iy_start, iy_end, nx, a_new_top, iy_end_top, a_new_bottom, 0,
                                 calculate_norm, compute_stream);

        if (calculate_norm) {
            if (use_mc_red) {
                launch_all_reduce_norm_barrier_kernel(l2_norm_d, partial_l2_norm, partial_l2_norm_mc, size, iter, compute_stream);
            }
            CUDA_RT_CALL(cudaMemcpyAsync(l2_norm_h, l2_norm_d, sizeof(real), cudaMemcpyDeviceToHost,
                                         compute_stream));
        }

        if (calculate_norm) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            if (!use_mc_red) {
                MPI_CALL(MPI_Allreduce(l2_norm_h, &l2_norm, 1, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD));
                l2_norm = std::sqrt(l2_norm);
            } else {
                // Need to ensure that partial_l2_norm is not reset by any GPU before all GPUs are
                // done reading it
                MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
                l2_norm = *l2_norm_h;
            }

            if (!csv && 0 == rank && (iter % 100) == 0) {
                printf("%5d, %0.6f\n", iter, l2_norm);
            }
        } else {
            MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
        }

        std::swap(a_new, a);
        std::swap(a_new_top, a_top);
        std::swap(a_new_bottom, a_bottom);
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
            printf("multi_node_p2p, %d, %d, %d, %d, %d, 1, %f, %f\n", nx, ny, iter_max, nccheck,
                   size, (stop - start), runtime_serial);
        } else {
            printf("Num GPUs: %d.\n", size);
            printf(
                "%dx%d: 1 GPU: %8.4f s, %d GPUs: %8.4f s, speedup: %8.2f, "
                "efficiency: %8.2f \n",
                ny, nx, runtime_serial, size, (stop - start), runtime_serial / (stop - start),
                runtime_serial / (size * (stop - start)) * 100);
        }
    }
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    CUDA_RT_CALL(cudaFreeHost(l2_norm_h));
    CUDA_RT_CALL(cudaFree(l2_norm_d));

    if (use_mc_red) {
        // Need to ensure that all processes are done using the MC object before unbinding it
        MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
        CUDA_CALL(cuMulticastUnbind(partial_l2_handles.mc_handle, device_id, 0,
                                    partial_l2_handles.uc_size));
        CUDA_CALL(cuMemUnmap(partial_l2_handles.mc_ptr, partial_l2_handles.mc_size));
        CUDA_CALL(cuMemUnmap(partial_l2_handles.uc_ptr, partial_l2_handles.uc_size));
        CUDA_CALL(cuMemRelease(partial_l2_handles.uc_handle));
        CUDA_CALL(cuMemRelease(partial_l2_handles.mc_handle));
        CUDA_CALL(cuMemAddressFree(partial_l2_handles.mc_ptr, partial_l2_handles.mc_size));
        CUDA_CALL(cuMemAddressFree(partial_l2_handles.uc_ptr, partial_l2_handles.uc_size));
    }

    unmap_peers(a_new_fa);
    unmap_peers(a_fa);
    free_fabric_mem(a_new_fa);
    free_fabric_mem(a_fa);

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
    bool calculate_norm = true;

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
