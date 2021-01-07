# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
NP ?= 1
NVCC=nvcc
MPIRUN ?= mpirun
ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set)
endif
ifndef MPI_HOME
$(error MPI_HOME is not set)
endif
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)
ifdef CUB_HOME
        NVCC_FLAGS = -DHAVE_CUB -I$(CUB_HOME)
else
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
endif
NVCC_FLAGS += -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -lnvToolsExt $(GENCODE_FLAGS) -std=c++11 -I$(NVSHMEM_HOME)/include -I$(MPI_HOME)/include
NVCC_LDFLAGS = -ccbin=mpic++ -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(MPI_HOME)/lib -lmpi -L$(CUDA_HOME)/lib64 -lcuda -lcudart -lnvToolsExt
jacobi: Makefile jacobi.cu
	$(NVCC) $(NVCC_FLAGS) jacobi.cu -c -o jacobi.o
	$(NVCC) $(GENCODE_FLAGS) jacobi.o -o jacobi $(NVCC_LDFLAGS)

.PHONY.: clean
clean:
	rm -f jacobi jacobi.o

memcheck: jacobi
	$(MPIRUN) -np $(NP) cuda-memcheck ./jacobi -nx 512 -ny 512

run: jacobi
	$(MPIRUN) -np $(NP) ./jacobi
