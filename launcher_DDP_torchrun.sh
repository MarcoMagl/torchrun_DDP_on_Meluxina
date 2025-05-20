#!/bin/bash -l
#SBATCH --job-name Resnet50DDPSingleNode
#SBATCH --account lxp
#SBATCH --partition gpu
#SBATCH --qos short
#SBATCH --nodes 2
#SBATCH --time 00:30:00
#SBATCH --output %x%j.out
#SBATCH --error %x%j.err
#SBATCH -c 8

module load env/release/2023.1
## Load software environment
module load Python
#module load CUDA

module load PyTorch
module load torchvision

rm output_*.txt

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export FREE_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Free port: $FREE_PORT"

# srun -n2 -G8 -c8 torchrun --nnodes=2 --nproc_per_node=4 torchrun resnet50ScriptDDP_for_torchrun.py 3 1 256 2

export NCCL_SOCKET_IFNAME=ib0
# deprecated
# export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
# export NCCL_CROSS_NIC=1

endpoint="${head_node_ip}:${FREE_PORT}"
echo "rdvz-endpoint is ${endpoint}"

# pick a stable “head” node
export MASTER_ADDR=${head_node_ip}
# derive a port that is free for this job (29400-65535 is usually open)
export MASTER_PORT=${FREE_PORT}

lsof -i:${MASTER_PORT}

echo "port status check done"

NGPUS_PER_NODE=4

set -x

pkill -f torch.distributed.run

module load jemalloc
export JEMALLOC_PRELOAD=$(jemalloc-config --libdir)/libjemalloc.so.$(jemalloc-config --revision)

rank=0
for host in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    echo "Launching rank $rank on $host"

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    srun --nodes=1 --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
        --gres=gpu:4 \
        --exclusive \
        --exact \
        --nodelist=$host \
        LD_PRELOAD=${JEMALLOC_PRELOAD} \
        torchrun \
        --nnodes $SLURM_NNODES \
        --node_rank $rank \
        --nproc_per_node 4 \
        --rdzv_backend c10d \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
        resnet50ScriptDDP_for_torchrun.py 4 1 256 2 &
    rank=$((rank + 1))
done

wait
