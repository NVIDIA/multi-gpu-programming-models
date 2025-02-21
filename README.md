# Multi GPU Programming Models
This project implements the well known multi GPU Jacobi solver with different multi GPU Programming Models:
* `single_threaded_copy`        Single Threaded using cudaMemcpy for inter GPU communication
* `multi_threaded_copy`         Multi Threaded with OpenMP using cudaMemcpy for inter GPU communication
* `multi_threaded_copy_overlap` Multi Threaded with OpenMP using cudaMemcpy for inter GPU communication with overlapping communication
* `multi_threaded_p2p`          Multi Threaded with OpenMP using GPUDirect P2P mappings for inter GPU communication
* `multi_threaded_p2p_opt`      Multi Threaded with OpenMP using GPUDirect P2P mappings for inter GPU communication with delayed norm execution
* `multi_threaded_um`           Multi Threaded with OpenMP relying on transparent peer mappings with Unified Memory for inter GPU communication
* `mpi`                         Multi Process with MPI using CUDA-aware MPI for inter GPU communication
* `mpi_overlap`                 Multi Process with MPI using CUDA-aware MPI for inter GPU communication with overlapping communication
* `nccl`                        Multi Process with MPI and NCCL using NCCL for inter GPU communication
* `nccl_overlap`                Multi Process with MPI and NCCL using NCCL for inter GPU communication with overlapping communication
* `nccl_graphs`                 Multi Process with MPI and NCCL using NCCL for inter GPU communication with overlapping communication combined with CUDA Graphs
* `nvshmem`                     Multi Process with MPI and NVSHMEM using NVSHMEM for inter GPU communication.
* `multi_node_p2p`              Multi Process Multi Node variant using the low level CUDA Driver [Virtual Memory Management](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#virtual-memory-management) and [Multicast Object Management](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MULTICAST.html#group__CUDA__MULTICAST) APIs. This example is for developers of libraries like NCCL or NVSHMEM. It shows how higher-level programming models like NVSHMEM work internally within a (multinode) NVLINK domain. Application developers generally should use the higher-level MPI, NCCL, or NVSHMEM interfaces instead of this API.

Each variant is a stand alone `Makefile` project and most variants have been discussed in various GTC Talks, e.g.:
* `single_threaded_copy`, `multi_threaded_copy`, `multi_threaded_copy_overlap`, `multi_threaded_p2p`, `multi_threaded_p2p_opt`, `mpi`, `mpi_overlap` and `nvshmem` on DGX-1V at GTC Europe 2017 in 23031 - Multi GPU Programming Models
* `single_threaded_copy`, `multi_threaded_copy`, `multi_threaded_copy_overlap`, `multi_threaded_p2p`, `multi_threaded_p2p_opt`, `mpi`, `mpi_overlap` and `nvshmem` on DGX-2 at GTC 2019 in S9139 - Multi GPU Programming Models
* `multi_threaded_copy`, `multi_threaded_copy_overlap`, `multi_threaded_p2p`, `multi_threaded_p2p_opt`, `mpi`, `mpi_overlap`, `nccl`, `nccl_overlap` and `nvshmem`  on DGX A100 at GTC 2021 in [A31140 - Multi-GPU Programming Models](https://www.nvidia.com/en-us/on-demand/session/gtcfall21-a31140/)

Some examples in this repository are the basis for an interactive tutorial: [FZJ-JSC/tutorial-multi-gpu](https://github.com/FZJ-JSC/tutorial-multi-gpu). 

# Requirements
* CUDA: version 11.0 (9.2 if build with `DISABLE_CUB=1`) or later is required by all variants.
  * `nccl_graphs` requires NCCL 2.15.1, CUDA 11.7 and CUDA Driver 515.65.01 or newer
  * `multi_node_p2p` requires CUDA 12.4, a CUDA Driver 550.54.14 or newer and the NVIDIA IMEX daemon running.
* OpenMP capable compiler: Required by the Multi Threaded variants. The examples have been developed and tested with gcc.
* MPI: The `mpi` and `mpi_overlap` variants require a CUDA-aware[^1] implementation. For NVSHMEM, NCCL and `multi_node_p2p`, a non CUDA-aware MPI is sufficient. The examples have been developed and tested with OpenMPI.
* NVSHMEM (version 0.4.1 or later): Required by the NVSHMEM variant.
* NCCL (version 2.8 or later): Required by the NCCL variant

# Building 
Each variant comes with a `Makefile` and can be built by simply issuing `make`, e.g. 
```sh
multi-gpu-programming-models$ cd multi_threaded_copy
multi_threaded_copy$ make
nvcc -DHAVE_CUB -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -ldl -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -std=c++14 jacobi.cu -o jacobi
multi_threaded_copy$ ls jacobi
jacobi
```

# Run instructions
All variants have the following command line options
* `-niter`: How many iterations to carry out (default 1000)
* `-nccheck`: How often to check for convergence (default 1)
* `-nx`: Size of the domain in x direction (default 16384)
* `-ny`: Size of the domain in y direction (default 16384)
* `-csv`: Print performance results as -csv
* `-use_hp_streams`: In `mpi_overlap` use high priority streams to hide kernel launch latencies of boundary kernels.

The `nvshmem` variant additionally provides
* `-use_block_comm`: Use block cooperative `nvshmemx_float_put_nbi_block` instead of `nvshmem_float_p` for communication.
* `-norm_overlap`: Enable delayed norm execution as also implemented in `multi_threaded_p2p_opt` 
* `-neighborhood_sync`: Use custom neighbor only sync instead of `nvshmemx_barrier_all_on_stream`

The `multi_node_p2p` variant additionally provides
* `-use_mc_red`: Use a device side barrier and allreduce leveraging Multicast Objects instead of MPI primitives

The `nccl` variants additionally provide
* `-user_buffer_reg`: Avoid extra internal copies in NCCL communication with [User Buffer Registration](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/bufferreg.html#user-buffer-registration). Required NCCL APIs are available with NCCL 2.19.1 or later. NCCL 2.23.4 added support for the used communication pattern.

The provided script `bench.sh` contains some examples executing all the benchmarks presented in the GTC Talks referenced above.

# Developers guide
The code applies the style guide implemented in [`.clang-format`](.clang-format) file. [`clang-format`](https://clang.llvm.org/docs/ClangFormat.html) version 7 or later should be used to format the code prior to submitting it. E.g. with
```sh
multi-gpu-programming-models$ cd multi_threaded_copy
multi_threaded_copy$ clang-format -style=file -i jacobi.cu
```

[^1]: A check for CUDA-aware support is done at compile and run time (see [the OpenMPI FAQ](https://www.open-mpi.org/faq/?category=runcuda#mpi-cuda-aware-support) for details). If your CUDA-aware MPI implementation does not support this check, which requires `MPIX_CUDA_AWARE_SUPPORT` and `MPIX_Query_cuda_support()` to be defined in `mpi-ext.h`, it can be skipped by setting `SKIP_CUDA_AWARENESS_CHECK=1`.
