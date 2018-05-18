#!/bin/bash
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

CPUID=0
NREP=5
IFS=$'\n'

ARGS="-csv"

echo "type, nx, ny, iter_max, nccheck, num_devices, p2p, runtime, runtime_serial"

function find_best () {
    declare -a RESULTS
    for ((i=0; i<$NREP; i++)); do
        RESULTS+=($("$@"))
    done
    printf '%s\n' "${RESULTS[@]}" | sort -k8 -b -t',' | head -1
    unset RESULTS
}

#Single threaded copy - no P2P
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS} -nop2p

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS} -nop2p

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS} -nop2p

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS} -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS} -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS} -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS} -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS} -nop2p

fi

# Single threaded copy - P2P, no device reordering
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    #P2P not available for all comm pairs
    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    #P2P not available for all comm pairs
    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    #P2P not available for all comm pairs
    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    #P2P not available for all comm pairs
    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

fi

# Single threaded copy - device reordering DGX-1V
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"                 # 0xNV2, 0xNV1
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,3"               # 2xNV2, 0xNV1
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,3,2"             # 2xNV2, 1xNV1
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,3,2,1"           # 3xNV2, 1xNV1
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="3,2,1,5,7"         # 3xNV2, 2xNV1
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,4"       # 5xNV2, 1xNV1
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,4,7,6,5,1,2"     # 6xNV2, 1xNV1
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,6,7,4"   # 8xNV2, 0xNV1
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi ${ARGS}

fi

#multi threaded copy without thread pinning
if true; then

    export OMP_PROC_BIND=FALSE
    
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"                 # 0xNV2, 0xNV1
    unset OMP_PLACES
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,3"               # 2xNV2, 0xNV1
    unset OMP_PLACES
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,3,2"             # 2xNV2, 1xNV1
    unset OMP_PLACES
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,3,2,1"           # 3xNV2, 1xNV1
    unset OMP_PLACES
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="3,2,1,5,7"         # 3xNV2, 2xNV1
    unset OMP_PLACES
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,4"       # 5xNV2, 1xNV1
    unset OMP_PLACES
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,4,7,6,5,1,2"     # 6xNV2, 1xNV1
    unset OMP_PLACES
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,6,7,4"   # 8xNV2, 0xNV1
    unset OMP_PLACES
    find_best ./multi_threaded_copy/jacobi ${ARGS}

fi

export OMP_PROC_BIND=TRUE

#multi threaded copy
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"                 # 0xNV2, 0xNV1
    export OMP_PLACES="{0}"
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,3"               # 2xNV2, 0xNV1
    export OMP_PLACES="{0},{1}"
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,3,2"             # 2xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2}"
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,3,2,1"           # 3xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2},{3}"
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="3,2,1,5,7"         # 3xNV2, 2xNV1
    export OMP_PLACES="{0},{1},{2},{20},{21}"
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,4"       # 5xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2},{3},{20},{21}"
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,4,7,6,5,1,2"     # 6xNV2, 1xNV1
    export OMP_PLACES="{0},{20},{21},{22},{23},{1},{2}"
    find_best ./multi_threaded_copy/jacobi ${ARGS}

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,6,7,4"   # 8xNV2, 0xNV1
    export OMP_PLACES="{0},{1},{2},{3},{20},{21},{22},{23}"
    find_best ./multi_threaded_copy/jacobi ${ARGS}

fi

#multi threaded copy overlap
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"                 # 0xNV2, 0xNV1
    export OMP_PLACES="{0}"
    find_best ./multi_threaded_copy_overlapp/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,3"               # 2xNV2, 0xNV1
    export OMP_PLACES="{0},{1}"
    find_best ./multi_threaded_copy_overlapp/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,3,2"             # 2xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2}"
    find_best ./multi_threaded_copy_overlapp/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,3,2,1"           # 3xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2},{3}"
    find_best ./multi_threaded_copy_overlapp/jacobi ${ARGS}

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="3,2,1,5,7"         # 3xNV2, 2xNV1
    export OMP_PLACES="{0},{1},{2},{20},{21}"
    find_best ./multi_threaded_copy_overlapp/jacobi ${ARGS}

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,4"       # 5xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2},{3},{20},{21}"
    find_best ./multi_threaded_copy_overlapp/jacobi ${ARGS}

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,4,7,6,5,1,2"     # 6xNV2, 1xNV1
    export OMP_PLACES="{0},{20},{21},{22},{23},{1},{2}"
    find_best ./multi_threaded_copy_overlapp/jacobi ${ARGS}

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,6,7,4"   # 8xNV2, 0xNV1
    export OMP_PLACES="{0},{1},{2},{3},{20},{21},{22},{23}"
    find_best ./multi_threaded_copy_overlapp/jacobi ${ARGS}

fi

#multi threaded p2p
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"                 # 0xNV2, 0xNV1
    export OMP_PLACES="{0}"
    find_best ./multi_threaded_p2p/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,3"               # 2xNV2, 0xNV1
    export OMP_PLACES="{0},{1}"
    find_best ./multi_threaded_p2p/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,3,2"             # 2xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2}"
    find_best ./multi_threaded_p2p/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,3,2,1"           # 3xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2},{3}"
    find_best ./multi_threaded_p2p/jacobi ${ARGS}

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="3,2,1,5,7"         # 3xNV2, 2xNV1
    export OMP_PLACES="{0},{1},{2},{20},{21}"
    find_best ./multi_threaded_p2p/jacobi ${ARGS}

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,4"       # 5xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2},{3},{20},{21}"
    find_best ./multi_threaded_p2p/jacobi ${ARGS}

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,4,7,6,5,1,2"     # 6xNV2, 1xNV1
    export OMP_PLACES="{0},{20},{21},{22},{23},{1},{2}"
    find_best ./multi_threaded_p2p/jacobi ${ARGS}

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,6,7,4"   # 8xNV2, 0xNV1
    export OMP_PLACES="{0},{1},{2},{3},{20},{21},{22},{23}"
    find_best ./multi_threaded_p2p/jacobi ${ARGS}

fi

#multi threaded p2p with delayed check
if true; then
    
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"                 # 0xNV2, 0xNV1
    export OMP_PLACES="{0}"
    find_best ./multi_threaded_p2p_opt/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,3"               # 2xNV2, 0xNV1
    export OMP_PLACES="{0},{1}"
    find_best ./multi_threaded_p2p_opt/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,3,2"             # 2xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2}"
    find_best ./multi_threaded_p2p_opt/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,3,2,1"           # 3xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2},{3}"
    find_best ./multi_threaded_p2p_opt/jacobi ${ARGS}

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="3,2,1,5,7"         # 3xNV2, 2xNV1
    export OMP_PLACES="{0},{1},{2},{20},{21}"
    find_best ./multi_threaded_p2p_opt/jacobi ${ARGS}

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,4"       # 5xNV2, 1xNV1
    export OMP_PLACES="{0},{1},{2},{3},{20},{21}"
    find_best ./multi_threaded_p2p_opt/jacobi ${ARGS}

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,4,7,6,5,1,2"     # 6xNV2, 1xNV1
    export OMP_PLACES="{0},{20},{21},{22},{23},{1},{2}"
    find_best ./multi_threaded_p2p_opt/jacobi ${ARGS}

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,6,7,4"   # 8xNV2, 0xNV1
    export OMP_PLACES="{0},{1},{2},{3},{20},{21},{22},{23}"
    find_best ./multi_threaded_p2p_opt/jacobi ${ARGS}
fi

if true; then
    
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"                 # 0xNV2, 0xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,3"               # 2xNV2, 0xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,3,2"             # 2xNV2, 1xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,3,2,1"           # 3xNV2, 1xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi/jacobi ${ARGS}

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="3,2,1,5,7"         # 3xNV2, 2xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:3:socket --bind-to core ./mpi/jacobi ${ARGS}

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,4"       # 5xNV2, 1xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi/jacobi ${ARGS}

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,4,7,6,5,1,2"     # 6xNV2, 1xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi/jacobi ${ARGS}

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,6,7,4"   # 8xNV2, 0xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi/jacobi ${ARGS}
    
fi

if true; then
    
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"                 # 0xNV2, 0xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi_overlapp/jacobi ${ARGS}

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,3"               # 2xNV2, 0xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi_overlapp/jacobi ${ARGS}

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,3,2"             # 2xNV2, 1xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi_overlapp/jacobi ${ARGS}

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,3,2,1"           # 3xNV2, 1xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi_overlapp/jacobi ${ARGS}

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="3,2,1,5,7"         # 3xNV2, 2xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:3:socket --bind-to core ./mpi_overlapp/jacobi ${ARGS}

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,4"       # 5xNV2, 1xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi_overlapp/jacobi ${ARGS}

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,4,7,6,5,1,2"     # 6xNV2, 1xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi_overlapp/jacobi ${ARGS}

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,3,2,1,5,6,7,4"   # 8xNV2, 0xNV1
    find_best mpirun -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:4:socket --bind-to core ./mpi_overlapp/jacobi ${ARGS}
    
fi
