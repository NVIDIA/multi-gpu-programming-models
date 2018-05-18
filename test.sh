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

now=`date +"%Y%m%d%H%M"`
LOG="test-${now}.log"

if [ -e ${LOG} ]; then
  echo "ERROR log file ${LOG} already exists."
  exit 1
fi

CUDA_VISIBLE_DEVICES_SETTING=("0" "0" "0,3" "0,3,2" "0,3,2,1" "3,2,1,5,7" "0,3,2,1,5,4" "0,4,7,6,5,1,2" "0,3,2,1,5,6,7,4" )

errors=0

for entry in `ls -1`; do
    if [ -f ${entry}/Makefile ] ; then
        if [ "run" == "$1" ] ; then
            NUM_GPUS=`nvidia-smi -L | wc -l`
            for (( NP=1; NP<=${NUM_GPUS}; NP++ )) ; do
                export NP
                export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SETTING[${NP}]}
                CMD="make -C ${entry} $1"
                ${CMD} >> ${LOG} 2>&1
                if [ $? -ne 0 ]; then
                    echo "ERROR with ${CMD} (NP = ${NP}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}) see ${LOG} for details."
                    errors=1
                    break
                fi
            done
        else
            CMD="make -C ${entry} $1"
            ${CMD} >> ${LOG} 2>&1
            if [ $? -ne 0 ]; then
                echo "ERROR with ${CMD} see ${LOG} for details."
                errors=1
                break
            fi
        fi
    fi
done

if [ ${errors} -eq 0 ]; then
    echo "Passed."
    exit 0
else
    exit 1
fi
