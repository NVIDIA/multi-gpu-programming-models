#!/bin/bash

#SBATCH -p batch
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 02:00:00

# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

: "${ENROOT_IMG_PATH:=.}"
: "${LUSTRE:=.}"

IMG=nvcr.io/nvidia/nvhpc:24.1-devel-cuda12.3-ubuntu22.04
SQUASHFS_IMG=$ENROOT_IMG_PATH/`echo "$IMG" | md5sum | cut -f1 -d " "`
CONTAINER_NAME=HPCSDK-CONTAINER

CONTAINER_MNTS=$LUSTRE/workspace/multi-gpu-programming-models:/mnt

start=`date`

if [[ -f "$SQUASHFS_IMG" ]]; then
    echo "Using: $SQUASHFS_IMG"
else
    echo "Fetching $IMG to $SQUASHFS_IMG"
    srun -n 1 -N 1 --ntasks-per-node=1 enroot import -o $SQUASHFS_IMG docker://$IMG
    echo "$IMG" > "${SQUASHFS_IMG}.url"
fi

CONTAINER_IMG=$SQUASHFS_IMG

if [[ ! -f "$CONTAINER_IMG" ]]; then
    echo "Falling back to $IMG"
    CONTAINER_IMG=$IMG
fi

# Pulling container on all nodes
srun -N ${SLURM_JOB_NUM_NODES} \
     -n ${SLURM_JOB_NUM_NODES} \
     --ntasks-per-node=1 \
     --container-image=$CONTAINER_IMG \
     --container-name=$CONTAINER_NAME \
     true

export SRUN_ARGS="--cpu-bind=none --mpi=none --no-container-remap-root --container-mounts=$CONTAINER_MNTS --container-workdir=/mnt --container-name=$CONTAINER_NAME"

# HCOLL is not used silence HCOLL warnings when running on a node without a IB HCA 
export OMPI_MCA_coll_hcoll_enable=0

export MPIRUN_ARGS="--oversubscribe"

#rebuild executables
srun $SRUN_ARGS -n 1 /bin/bash -c "./test.sh clean; sleep 1; ./test.sh"

srun -n 1 /bin/bash -c "sudo nvidia-smi -lgc 1980,1980"

srun $SRUN_ARGS -n 1 ./bench.sh

srun $SRUN_ARGS -n 1 /bin/bash -c "nvidia-smi; modinfo gdrdrv; env; nvcc --version; mpicxx --version"

