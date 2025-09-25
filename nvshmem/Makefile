# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
NP ?= 1
NVCC=nvcc
MPIRUN ?= mpirun
CUDA_HOME ?= /usr/local/cuda
ifndef NVSHMEM_HOME
$(error NVSHMEM_HOME is not set)
endif
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_SM90    := -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
ifdef USE_LTO
	GENCODE_SM70 += -gencode arch=compute_70,code=lto_70
	GENCODE_SM80 += -gencode arch=compute_80,code=lto_80
	GENCODE_SM90 += -gencode arch=compute_90,code=lto_90
endif
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80) $(GENCODE_SM90)
ifdef BUILD_SM_ARCH
	GENCODE_FLAGS	:= $(GENCODE_SM$(BUILD_SM_ARCH))
endif

ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif
NVCC_FLAGS += -ccbin=mpic++ -dc -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -ldl $(GENCODE_FLAGS) -std=c++17 -I$(NVSHMEM_HOME)/include
NVCC_LDFLAGS = -ccbin=mpic++ -L$(NVSHMEM_HOME)/lib -lnvshmem -L$(CUDA_HOME)/lib64 -lcuda -lcudart -ldl -lnvidia-ml
ifdef USE_LTO
	NVCC_FLAGS += -maxrregcount=32
	NVCC_LDFLAGS += -maxrregcount=32 -dlto
endif
jacobi: Makefile jacobi.cu
	$(NVCC) $(NVCC_FLAGS) jacobi.cu -c -o jacobi.o
	$(NVCC) $(GENCODE_FLAGS) jacobi.o -o jacobi $(NVCC_LDFLAGS)

.PHONY.: clean
clean:
	rm -f jacobi jacobi.o *.nsys-rep jacobi.*.compute-sanitizer.log

sanitize: jacobi
	$(MPIRUN) -np $(NP) compute-sanitizer --log-file jacobi.%q{OMPI_COMM_WORLD_RANK}.compute-sanitizer.log ./jacobi -niter 10

run: jacobi
	$(MPIRUN) -np $(NP) ./jacobi

profile: jacobi
	$(MPIRUN) -np $(NP) nsys profile --trace=mpi,cuda,nvtx -o jacobi.%q{OMPI_COMM_WORLD_RANK} ./jacobi -niter 10
