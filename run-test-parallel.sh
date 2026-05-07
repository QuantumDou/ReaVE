#!/bin/bash -l
#SBATCH --job-name=qwen2.5-mimic-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2           # 显式启动 8 个分片
#SBATCH --gres=gpu:2                  # 申请 8 块 GPU
#SBATCH --cpus-per-task=16            # 每卡分配 8 核 CPU 提升预处理速度
#SBATCH --mem=300G                    # 调大内存，确保 8 个进程加载模型时不 OOM
#SBATCH --time=2-00:00               
#SBATCH --output=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/testing_logs/test-parallel-%j.log
#SBATCH --error=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/testing_logs/test-parallel-%j.log
#SBATCH --mail-user=1312438276@qq.com
#SBATCH --mail-type=END,FAIL

# --- 1. 环境准备 ---
# 工作目录路径
RUNPATH=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/
cd $RUNPATH

# 核心修复：显式将项目根目录加入 Python 搜索路径，防止 ModuleNotFoundError
export PYTHONPATH=$RUNPATH:$PYTHONPATH

# 激活 Conda 环境
eval "$(conda shell.bash hook)"
conda activate inter2.5

# 优化环境变量
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# 检查分片数
echo "🚀 开始并行评估，总分片数: $SLURM_NTASKS"

# --- 2. 分片运行逻辑 ---
# 使用 srun 启动并行任务
# --exclusive 确保每个任务分配独立的资源，避免进程冲突
# --label 在日志前增加 [0], [1] 标签方便定位报错
srun --label --exclusive bash -c "
    # 核心修复：强制实施 GPU 逻辑隔离
    # 将当前的进程绑定到本地 GPU ID (0-7)，防止所有进程抢 GPU 0
    export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID
    
    echo \"[Shard \$SLURM_PROCID] Starting on \$(hostname), Using GPU: \$CUDA_VISIBLE_DEVICES\"
    
    # 稍微延迟启动，缓解 8 个进程同时读磁盘的 I/O 压力
    sleep \$((SLURM_LOCALID * 2))
    
    python -u -m Med_eval_kit_mimic.run \
        --shard_id \$SLURM_PROCID \
        --num_shards \$SLURM_NTASKS
"

# --- 3. 结果合并逻辑 ---
# $? 获取 srun 整个任务组的退出状态
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "✅ 所有分片运行成功。正在合并结果..."
    echo "=========================================="
    
    # 等待文件 IO 刷新
    sleep 5
    
    # 运行合并脚本
    python -u -m Med_eval_kit_mimic.merge_shard_results \
        --num_shards $SLURM_NTASKS
    
    if [ $? -eq 0 ]; then
        echo "🎉 所有任务圆满完成！"
    else
        echo "❌ 合并阶段出错！"
        exit 1
    fi
else
    echo "❌ 并行评估阶段有进程崩溃，请检查日志中的 OOM 或路径错误！"
    exit 1
fi