# Marco Magliulo @ LuxProvide
# Emmanuel Kieffer @ LuxProvide
#!/bin/bash -l
#SBATCH --job-name Resnet50DDPApptainer
#SBATCH --account lxp
#SBATCH --partition gpu
#SBATCH --qos short
##SBATCH --reservation gpudev 
#SBATCH --nodes 2
#SBATCH --time 00:05:00
#SBATCH --output %x%j.out
#SBATCH --error %x%j.err
#SBATCH -c 8
#SBATCH --ntasks=4  

module load env/release/2024.1
module load Apptainer/1.3.6-GCCcore-13.3.0
export APPTAINER_CACHEDIR=/project/home/lxp/reframe_runtime/app-reframe
rm output_*.txt
nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
export head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

export FREE_PORT=$(srun -n 1 -c 1 -N 1 apptainer exec pytorch_22.08.sif python -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()')
echo "Free port: $FREE_PORT"
export NCCL_SOCKET_IFNAME=ib0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=8
endpoint="${head_node_ip}:${FREE_PORT}"
echo "rdvz-endpoint is ${endpoint}"

export MASTER_ADDR=${head_node_ip}
export MASTER_PORT=${FREE_PORT}
lsof -i:${MASTER_PORT}

echo "port status check done"
pkill -f torch.distributed.run

# Get the number of nodes from SLURM
NNODES=$SLURM_JOB_NUM_NODES
# Get the number of GPUs per node (assuming it's set as an environment variable)
NGPUS_PER_NODE=4
# Calculate the total number of tasks
NTASKS=$((NNODES * NGPUS_PER_NODE))
# Set the number of tasks per node
NTASKS_PER_NODE=$NGPUS_PER_NODE

echo "Number of Nodes: $NNODES"
echo "GPUs per Node: $NGPUS_PER_NODE"
echo "Total Number of Tasks: $NTASKS"
echo "Tasks per Node: $NTASKS_PER_NODE"

export CUDA_VISIBLE_DEVICES=0,1,2,3
HOST_DIR=${PWD}

module load jemalloc
export JEMALLOC_PRELOAD=$(jemalloc-config --libdir)/libjemalloc.so.$(jemalloc-config --revision)

rank=0
for host in $(scontrol show hostnames "$SLURM_JOB_NODELIST"); do
    echo "In the batch file: Launching rank $rank on $host"

    export CUDA_VISIBLE_DEVICES=0,1,2,3
    srun --nodes=1 --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
        --gres=gpu:4 \
        --exclusive \
        --exact \
        --nodelist=$host \
        apptainer \
        exec --nv \
        --env MASTER_ADDR=${MASTER_ADDR}\
        --env MASTER_PORT=${MASTER_PORT}\
        --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
        -B${PWD} --pwd ${PWD} \
        pytorch_22.08.sif \
        torchrun \
        --nnodes $SLURM_NNODES \
        --node_rank $rank \
        --nproc_per_node ${NGPUS_PER_NODE} \
        --rdzv_backend c10d \
        --rdzv_id $SLURM_JOB_ID \
        --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
	../resnet50ScriptDDP_for_torchrun.py 4 1 256 2 &

    rank=$((rank + 1))
done

wait

