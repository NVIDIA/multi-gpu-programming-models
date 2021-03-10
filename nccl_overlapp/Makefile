# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
NP ?= 1
NVCC=nvcc
MPICXX=mpicxx
MPIRUN ?= mpirun
CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)
ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif
NVCC_FLAGS += -lineinfo $(GENCODE_FLAGS) -std=c++14
MPICXX_FLAGS = -DUSE_NVTX -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -std=c++14
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt -lnccl
jacobi: Makefile jacobi.cpp jacobi_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) jacobi.cpp jacobi_kernels.o $(LD_FLAGS) -o jacobi

jacobi_kernels.o: Makefile jacobi_kernels.cu
	$(NVCC) $(NVCC_FLAGS) jacobi_kernels.cu -c

.PHONY.: clean
clean:
	rm -f jacobi jacobi_kernels.o *.qdrep jacobi.*.compute-sanitizer.log

sanitize: jacobi
	$(MPIRUN) -np $(NP) compute-sanitizer --log-file jacobi.%q{OMPI_COMM_WORLD_RANK}.compute-sanitizer.log ./jacobi -niter 10

run: jacobi
	$(MPIRUN) -np $(NP) ./jacobi

profile: jacobi
	$(MPIRUN) -np $(NP) nsys profile --trace=mpi,cuda,nvtx -o jacobi.%q{OMPI_COMM_WORLD_RANK} ./jacobi -niter 10
