#!/bin/bash
# Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
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

NREP=5
NXNY="18432"

#DGX-1V
#CPUID=0-19
#MAX_NUM_GPUS=8
#CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,3" "0,3,2" "0,3,2,1" "3,2,1,5,7" "0,3,2,1,5,4" "0,4,7,6,5,1,2" "0,3,2,1,5,6,7,4" )

#DGX-2
CPUID=0-23
MAX_NUM_GPUS=16
CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,1" "0,1,2" "0,1,2,3" "0,1,2,3,4" "0,1,2,3,4,5" "0,1,2,3,4,5,6" "0,1,2,3,4,5,6,7" "0,1,2,3,4,5,6,7,8" "0,1,2,3,4,5,6,7,8,9" "0,1,2,3,4,5,6,7,8,9,10" "0,1,2,3,4,5,6,7,8,9,10,11" "0,1,2,3,4,5,6,7,8,9,10,11,12" "0,1,2,3,4,5,6,7,8,9,10,11,12,13" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" )

IFS=$'\n'
function find_best () {
    declare -a RESULTS
    for ((i=0; i<$NREP; i++)); do
        RESULTS+=($("$@"))
    done
    printf '%s\n' "${RESULTS[@]}" | sort -k8 -b -t',' | head -1
    unset RESULTS
}

#sudo nvidia-smi -ac 958,1597

#Single GPU
if false; then
    echo "type, nx, ny, iter_max, nccheck, runtime"
    export CUDA_VISIBLE_DEVICES="0"
    for (( nx=1024; nx <= 18*1024; nx+=1024 )); do
        find_best taskset -c ${CPUID} ./single_gpu/jacobi -csv -nx $nx -ny $nx
    done
fi

if false; then
    echo "type, nx, ny, iter_max, nccheck, runtime"
    export CUDA_VISIBLE_DEVICES="0"
    find_best taskset -c ${CPUID} ./single_gpu/jacobi -csv -nx ${NXNY} -ny ${NXNY}
fi

echo "type, nx, ny, iter_max, nccheck, num_devices, p2p, runtime, runtime_serial"

#Single threaded copy - no P2P
if false; then

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx ${NXNY} -ny ${NXNY} -nop2p
    done

fi

# Single threaded copy - P2P
if false; then

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#multi threaded copy without thread pinning
if false; then

    export OMP_PROC_BIND=FALSE
    unset OMP_PLACES
    
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best ./multi_threaded_copy/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

export OMP_PROC_BIND=TRUE

#multi threaded copy
if false; then

    OMP_PLACES="{0}"
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        if (( NUM_GPUS > 1 )); then
            OMP_PLACES="${OMP_PLACES},{$((NUM_GPUS-1))}"
        fi
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        export OMP_PLACES
        find_best ./multi_threaded_copy/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#multi threaded copy overlap
if false; then

    OMP_PLACES="{0}"
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        if (( NUM_GPUS > 1 )); then
            OMP_PLACES="${OMP_PLACES},{$((NUM_GPUS-1))}"
        fi
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        export OMP_PLACES
        find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#multi threaded p2p
if false; then

    OMP_PLACES="{0}"
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        if (( NUM_GPUS > 1 )); then
            OMP_PLACES="${OMP_PLACES},{$((NUM_GPUS-1))}"
        fi
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        export OMP_PLACES
        find_best ./multi_threaded_p2p/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

#multi threaded p2p with delayed check
if false; then

    OMP_PLACES="{0}"
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        if (( NUM_GPUS > 1 )); then
            OMP_PLACES="${OMP_PLACES},{$((NUM_GPUS-1))}"
        fi
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        export OMP_PLACES
        find_best ./multi_threaded_p2p_opt/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

if false; then

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

if false; then

    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi

if false; then

    export SHMEM_SYMMETRIC_SIZE=3221225472
    for (( NUM_GPUS=1; NUM_GPUS <= ${MAX_NUM_GPUS}; NUM_GPUS+=1 )); do
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NUM_GPUS}]}
        find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx ${NXNY} -ny ${NXNY}
    done

fi
