#!/bin/bash
#SBATCH --job-name=modgpt-8gpu
#SBATCH --account=PAS2836
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1           # one launcher per node
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu               # verify this is the correct multi-node GPU partition
#SBATCH --time=04:00:00               # give yourself room; adjust as needed
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "Start: $(date)"
echo "Job $SLURM_JOB_ID on $SLURM_JOB_NODELIST"
cd "/fs/ess/PAS2836/mugezhang/code/modded-nanogpt"

# --- env (already built ahead of time) ---
module load miniconda3/24.1.2-py310
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate modded-nanogpt-test

# --- comms / threading ---
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME="^lo,docker0"

# rendezvous
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
MASTER_PORT=$((10000 + SLURM_JOB_ID % 20000))
NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID

srun --ntasks-per-node=1 python -m torch.distributed.run \
  --nnodes="${NNODES}" \
  --nproc-per-node=4 \
  --node_rank="${NODE_RANK}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  train_gpt.py

echo "End: $(date)"