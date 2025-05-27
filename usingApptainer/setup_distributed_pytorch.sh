#!/bin/bash

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo "The head node is ${head_node}"

# OPTIONAL: set to true if you want more details on NCCL communications
DEBUG=true
if [ "$DEBUG" == "true" ]; then
    export LOGLEVEL=INFO
    export NCCL_DEBUG=TRACE
    export TORCH_CPP_LG_LEVEL=INFO
else
    echo "Debug mode is off."
fi

# Define the file where PyTorch will make a snapshot in case a training is interrupted and will have to be restarted
snapshot_name="snapshot.pt"
snapshot_file="${PWD}/${snapshot_name}"

if [ -f "$snapshot_file" ]; then
    file_exists=true
    echo "snapshot file found"
else
    file_exists=false
    echo "no snapshot file was found"
fi

remove_snapshot=true
if [ "$remove_snapshot" == "true" ]; then
    if [ -f "$snapshot_file" ]; then
        rm ${snapshot_file}
        echo "snapshot file deleted"
    fi
fi

export NCCL_SOCKET_IFNAME=ib0
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8

# get free port
export random_port=$(python getPort.py)

echo "rdvz-endpoint is ${head_node_ip}:${random_port}"
endpoint="${head_node_ip}:${random_port}"

export NGPUS_PER_NODE=4

export CUDA_LAUNCH_BLOCKING=1
