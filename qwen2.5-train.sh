#!/bin/bash -l
#SBATCH --output=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/training_log/test-%j.log
#SBATCH --error=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/training_log/test-%j.log
#SBATCH --job-name=qwen2.5-train
#SBATCH --partition=interruptible_gpu
#SBATCH --gres=gpu:3 # 2 申请2块gpu
#SBATCH --ntasks-per-node=1  # 每个节点（机器）1个任务
#SBATCH --cpus-per-task=32  # 每个任务分配32个CPU核心,平均一卡16核
#SBATCH --mem=256G     # 所有节点共分配100G内存
#SBATCH --time=1-00:00     # runtime (D-HH:MM)
#SBATCH --mail-user=1312438276@qq.com   # email address
#SBATCH --mail-type=END   #get email when job ends
#SBATCH --mail-type=FAIL    #get email when job aborts


RUNPATH=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/LlamaFactory/
cd $RUNPATH

source activate inter2.5


# conda activate inter2.5
# VENV_BIN=/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/envs/inter2.5/bin
# source $VENV_BIN/activate

# 1. 定义你的工作路径
PROJ_DIR="/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN"
PROJ_DIR="/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN"
mkdir -p $PROJ_DIR/hf_cache/{home,datasets,ds_cache,triton_cache,pip_cache,pycache}


# 2. 强行重定向所有可能的缓存（如果不存在会自动创建）
export HF_HOME="$PROJ_DIR/hf_cache/home"
export HF_DATASETS_CACHE="$PROJ_DIR/hf_cache/datasets"
export DEEPSPEED_CACHE_DIR="$PROJ_DIR/hf_cache/ds_cache"
export TRITON_CACHE_DIR="$PROJ_DIR/hf_cache/triton_cache"
export PIP_CACHE_DIR="$PROJ_DIR/hf_cache/pip_cache"
export PYTHON_PYCACHE_PREFIX="$PROJ_DIR/hf_cache/pycache"
export TRANSFORMERS_CACHE=/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/huggingface


export HF_DATASETS_OFFLINE=1  # 既然模型下好了，设为 1 防止联网检查导致的卡顿
export TRANSFORMERS_VERBOSITY=info

# python -c "from datasets import config; print('Dataset Cache Path:', config.HF_DATASETS_CACHE)"
export PYTHONUNBUFFERED=1

#减少信息输出
#---------------------------------------------------------------------------
export PYTHONWARNINGS="ignore"
export NCCL_DEBUG=ERROR

# 屏蔽 Transformers 的建议性警告 (解决那些 🚨 输出)
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1 

# 将 Transformers 日志设为 error，只看报错 (Rank 0 也会变安静，只报关键错)
# export TRANSFORMERS_VERBOSITY=error 

# 屏蔽 Python 自身的警告
export PYTHONWARNINGS="ignore"

# 屏蔽 Tokenizers 的并行化警告
export TOKENIZERS_PARALLELISM=false

# 屏蔽 Accelerate 的冗余日志
export ACCELERATE_LOG_LEVEL=error

#显示进度条
export TQDM_DISABLE=0
# ---------------------------------------------------------------------------


export NCCL_P2P_DISABLE=1 #1
export NCCL_IB_DISABLE=1 #1

#确保端口不冲突
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500 # 换一个端口，防止之前的僵尸进程占用
# export MASTER_PORT=$(shuf -i 20000-30000 -n 1) # 随机端口避免冲突

# 解决环境兼容性报错 (备选，报错时开启)
export DS_SKIP_CUDA_CHECK=1 

export DISABLE_VERSION_CHECK=1
export FORCE_TORCHRUN=1

# 解决 DeepSpeed 编译缓存空间 (建议)
# export DEEPSPEED_CACHE_DIR="/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/hf_cache/ds_cache"


# 4. 单卡运行训练 (请根据你的实际 yaml 文件路径修改)
#llamafactory-cli train examples/mint-cot/qwen2_5_vl_7b_full_sft_interleaved.yaml 2>&1 | tee training_log_7b.txt

#多卡环境下的单卡训练
# export CUDA_VISIBLE_DEVICES=0
# export NPROC_PER_NODE=1
# llamafactory-cli train  examples/mint-cot/qwen2_5_vl_7b_full_sft_interleaved.yaml 2>&1 | tee training_log_7b.txt



# #5.多卡训练
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5  # 使用全部 4 块 A100
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
export NPROC_PER_NODE=$GPUS_PER_NODE         # 进程数与 GPU 数量一致
echo "  GPUs per node: $GPUS_PER_NODE"


# llamafactory-cli train examples/mint-cot/qwen2_5_vl_7b_full_sft_interleaved.yaml 2>&1 | tee training_log.log

llamafactory-cli train examples/mint-cot/qwen2_5_vl_7b_full_sft_interleaved.yaml 