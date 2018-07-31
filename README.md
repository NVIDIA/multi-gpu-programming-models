# Multi GPU Programming Models
This project implements the well known multi GPU Jacobi solver with different multi GPU Programming Models:
* Single Threaded using cudaMemcpy of inter GPU communication (`single_threaded_copy`)
* Multi Threaded with OpenMP using cudaMemcpy for inter GPU communication (`multi_threaded_copy`)
* Multi Threaded with OpenMP using cudaMemcpy for itner GPU communication with overlapping communication (`multi_threaded_copy_overlapp`)
* Multi Threaded with OpenMP using GPUDirect P2P mappings for inter GPU communication (`multi_threaded_p2p`)
* Multi Threaded with OpenMP using GPUDirect P2P mappings for inter GPU communication with delayed norm execution (`multi_threaded_p2p_opt`)
* Multi Threaded with OpenMP relying on transparent peer mappings with Unified Memory for inter GPU communication (`multi_threaded_um`)
* Multi Process with MPI using CUDA-aware MPI for inter GPU communication (`mpi`)
* Multi Process with MPI using CUDA-aware MPI for inter GPU communication with overlapping communication (`mpi_overlapp`)

Each variant is a stand alone Makefile project and all variants have been described in the GTC EU 2017 Talk [Multi GPU Programming Models](http://on-demand-gtc.gputechconf.com/gtc-quicklink/4rWBZ)

# Requirements
* CUDA: Required by all variants. The examples have been developed with CUDA 8 and tested with CUDA 9.1 and 9.2 but except the Unified Memory variant should also work with older CUDA version.
* OpenMP capable compiler: Required by the Multi Threaded variants. The examples have been developed and tested with gcc.
* CUDA-aware MPI: Required by teh MPI variants. The examples have been developed and tested with OpenMPI.
* CUB: Optional for optimized residual reductions. Set CUB_HOME to your cub installation directory. The examples have been developed and tested with cub 1.8.0.

# Building 
Each variant come with a Makefile and can be build by simply issuing make, e.g.
```sh
multi-gpu-programming-models$ cd multi_threaded_copy
multi_threaded_copy$ make CUB_HOME=../cub
nvcc -DHAVE_CUB -I../cub -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_70,code=compute_70  -std=c++11 jacobi.cu -o jacobi
multi_threaded_copy$ ls jacobi
jacobi
```

# Run instructions
All variant have the following command line options
* `-niter`: How many iterations to carry out (default 1000)
* `-nccheck`: How often to check for convergence (default 1)
* `-nx`: Size of the domain in x direction (default 7168)
* `-ny`: Size of the domain in y direction (default 7168)
* `-csv`: Print performance results as -csv

The provided script `bench.sh` contains some examples executing all the benchmarks presented in the GTC EU 2017 Talk Multi GPU Programming Models.
