#!/bin/bash -l
#SBATCH --output=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/training_logs/test-%j-3b-54880-wo-Lstatus.log
#SBATCH --error=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/training_logs/test-%j-3b-54880-wo-Lstatus.log
#SBATCH --job-name=inter2.5
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8  # 申请n块gpu
#SBATCH --ntasks-per-node=1  # 每个节点1个任务
#SBATCH --cpus-per-task=64  # 每个任务分配32个CPU核心,平均一卡16核
#SBATCH --mem=800G #300G     # 每个节点分配80G内存
#SBATCH --time=2-00:00     # runtime (D-HH:MM)
#SBATCH --mail-user=1312438276@qq.com   # email address
#SBATCH --mail-type=END   #get email when job ends
#SBATCH --mail-type=FAIL    #get email when job aborts


RUNPATH=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/LlamaFactory/
cd $RUNPATH

source activate inter2.5


#新添加
# export NCCL_DEBUG=INFO # 开启 NCCL 调试日志（关键）


#设置 Hugging Face 缓存目录到项目存储，避免用户配额限制
# export HF_HOME=/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/huggingface
# export HF_HUB_CACHE=/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/huggingface
# export TRANSFORMERS_CACHE=/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/huggingface

# # 禁用在线下载，强制使用本地缓存
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1

set -x
echo "JOB START $(date)"
hostname
pwd

which python
python -V
nvidia-smi || true

# 获取SLURM多节点环境变量
NUM_NODES=${SLURM_JOB_NUM_NODES:-1}
NODE_RANK=${SLURM_NODEID:-0}
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))

# 获取主节点地址（SLURM会自动设置）
if [ -z "$MASTER_ADDR" ]; then
    # 从SLURM节点列表获取第一个节点作为主节点
    MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
fi


# MASTER_PORT=${MASTER_PORT:-29500}
# 根据 Job ID 动态生成端口 (10000 + ID后四位)，防止端口冲突
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))


echo "=========================================="
echo "Multi-Node Multi-GPU Training Configuration:"
echo "  Number of nodes: $NUM_NODES"
echo "  Current node rank: $NODE_RANK"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Total GPUs: $TOTAL_GPUS"
echo "  Master address: $MASTER_ADDR"
echo "  Master port: $MASTER_PORT"
echo "  SLURM_NODELIST: $SLURM_JOB_NODELIST"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

# 设置环境变量以支持DeepSpeed多节点多卡训练
export FORCE_TORCHRUN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NNODES=$NUM_NODES
export NODE_RANK=$NODE_RANK
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export NPROC_PER_NODE=$GPUS_PER_NODE

#输出编译日志
#实时监控：确保你能看到它在干什么
export PYTHONUNBUFFERED=1
# export NCCL_DEBUG=INFO
export DS_BUILD_VERBOSE=1

# 使用本地临时目录作为编译缓存，速度快 10 倍
# export DEEPSPEED_CACHE_DIR="/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/ds_cache"
# mkdir -p $DEEPSPEED_CACHE_DIR
export BC_ROUND=1


# 禁用DeepSpeed编译避免磁盘问题
export DS_SKIP_CUDA_CHECK=1

# 使用llamafactory-cli进行多节点多GPU训练
echo "Starting multi-node multi-GPU training with llamafactory-cli..."
echo "Node $NODE_RANK of $NUM_NODES nodes"
# DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/mint-cot/qwen2vl_2b_lora_sft_interleaved.yaml
# DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/mint-cot/qwen2_5_vl_7b_full_sft_interleaved.yaml 
DISABLE_VERSION_CHECK=1 FORCE_TORCHRUN=1 llamafactory-cli train examples/mint-cot/qwen2_5_vl_3b_full_sft_interleaved.yaml 