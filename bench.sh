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

CPUID=0-23
NREP=5
IFS=$'\n'

ARGS="-csv -nx 18432 -ny 18432"

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

if true; then
    echo "type, nx, ny, iter_max, nccheck, runtime"
    export CUDA_VISIBLE_DEVICES="0"
    find_best taskset -c ${CPUID} ./single_gpu/jacobi -csv -nx 18432 -ny 18432
fi

echo "type, nx, ny, iter_max, nccheck, num_devices, p2p, runtime, runtime_serial"

#Single threaded copy - no P2P
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p
    
    #P2P not available for all comm pairs
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p
    
    #P2P not available for all comm pairs
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p
    
    #P2P not available for all comm pairs
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p

    #P2P not available for all comm pairs
    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p
    
    #P2P not available for all comm pairs
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p
    
    #P2P not available for all comm pairs
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p
    
    #P2P not available for all comm pairs
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p
    
    #P2P not available for all comm pairs
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432 -nop2p
fi

# Single threaded copy - P2P
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    #P2P not available for all comm pairs
    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    #P2P not available for all comm pairs
    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    #P2P not available for all comm pairs
    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    #P2P not available for all comm pairs
    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    #P2P not available for all comm pairs
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    #P2P not available for all comm pairs
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    #P2P not available for all comm pairs
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    #P2P not available for all comm pairs
    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    #P2P not available for all comm pairs
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    #P2P not available for all comm pairs
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    #P2P not available for all comm pairs
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    #P2P not available for all comm pairs
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    find_best taskset -c ${CPUID} ./single_threaded_copy/jacobi -csv -nx 18432 -ny 18432

fi

#multi threaded copy without thread pinning
if true; then

    export OMP_PROC_BIND=FALSE
    unset OMP_PLACES

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

fi

export OMP_PROC_BIND=TRUE

#multi threaded copy
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    export OMP_PLACES="{0}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    export OMP_PLACES="{0},{1}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    export OMP_PLACES="{0},{1},{2}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export OMP_PLACES="{0},{1},{2},{3}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    export OMP_PLACES="{0},{1},{2},{3},{4}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}"
    find_best ./multi_threaded_copy/jacobi -csv -nx 18432 -ny 18432

fi

#multi threaded copy overlap
if true; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    export OMP_PLACES="{0}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    export OMP_PLACES="{0},{1}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    export OMP_PLACES="{0},{1},{2}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export OMP_PLACES="{0},{1},{2},{3}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    export OMP_PLACES="{0},{1},{2},{3},{4}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}"
    find_best ./multi_threaded_copy_overlapp/jacobi -csv -nx 18432 -ny 18432

fi

#multi threaded p2p
if false; then

    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    export OMP_PLACES="{0}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    export OMP_PLACES="{0},{1}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    export OMP_PLACES="{0},{1},{2}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export OMP_PLACES="{0},{1},{2},{3}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    export OMP_PLACES="{0},{1},{2},{3},{4}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}"
    find_best ./multi_threaded_p2p/jacobi -csv -nx 18432 -ny 18432

fi

#multi threaded p2p with delayed check
if true; then
    
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    export OMP_PLACES="{0}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    export OMP_PLACES="{0},{1}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    export OMP_PLACES="{0},{1},{2}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    export OMP_PLACES="{0},{1},{2},{3}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    export OMP_PLACES="{0},{1},{2},{3},{4}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15}"
    find_best ./multi_threaded_p2p_opt/jacobi -csv -nx 18432 -ny 18432
fi

if true; then
    
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi/jacobi -csv -nx 18432 -ny 18432
    
fi

if true; then
    
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES --map-by ppr:8:socket --bind-to core ./mpi_overlapp/jacobi -csv -nx 18432 -ny 18432
    
fi

if true; then
    
    export SHMEM_SYMMETRIC_SIZE=3221225472
    NUM_GPUS=1
    export CUDA_VISIBLE_DEVICES="0"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=2
    export CUDA_VISIBLE_DEVICES="0,1"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=3
    export CUDA_VISIBLE_DEVICES="0,1,2"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=4
    export CUDA_VISIBLE_DEVICES="0,1,2,3"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=5
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=6
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=7
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432

    NUM_GPUS=8
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=9
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=10
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=11
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=12
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=13
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=14
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=15
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
    NUM_GPUS=16
    export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
    find_best mpirun ${MPIRUN_ARGS} -np ${NUM_GPUS} -x CUDA_VISIBLE_DEVICES -x SHMEM_SYMMETRIC_SIZE --map-by ppr:8:socket --bind-to core ./nvshmem/jacobi -csv -nx 18432 -ny 18432
    
fi
