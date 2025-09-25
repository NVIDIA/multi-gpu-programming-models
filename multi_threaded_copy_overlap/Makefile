# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
NVCC=nvcc
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80
GENCODE_SM90    := -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80) $(GENCODE_SM90)
ifdef BUILD_SM_ARCH
	GENCODE_FLAGS	:= $(GENCODE_SM$(BUILD_SM_ARCH))
endif
ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif
NVCC_FLAGS += -Xcompiler -fopenmp -lineinfo -DUSE_NVTX -ldl $(GENCODE_FLAGS) -std=c++14
jacobi: Makefile jacobi.cu
	$(NVCC) $(NVCC_FLAGS) jacobi.cu -o jacobi

.PHONY.: clean
clean:
	rm -f jacobi jacobi.nsys-rep

sanitize: jacobi
	compute-sanitizer ./jacobi -niter 10

run: jacobi
	./jacobi

profile: jacobi
	nsys profile --trace=cuda,nvtx -o jacobi ./jacobi -niter 10
