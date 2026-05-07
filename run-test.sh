#!/bin/bash -l
#SBATCH --output=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/testing_logs/test-%j-_v1.log
#SBATCH --error=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/testing_logs/test-%j-_v1.log
#SBATCH --job-name=qwen2.5-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 # 每个节点分配4个GPU（可根据需要调整）
#SBATCH --ntasks-per-node=1  # 每个节点1个任务
#SBATCH --cpus-per-task=4  # 每个任务分配16个CPU核心
#SBATCH --mem=40G     # 每个节点分配80G内存
#SBATCH --time=1-00:00     # runtime (D-HH:MM)
#SBATCH --mail-user=1312438276@qq.com   # email address
#SBATCH --mail-type=END   #get email when job ends
#SBATCH --mail-type=FAIL    #get email when job aborts



RUNPATH=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/
cd $RUNPATH

# 激活 Conda
eval "$(conda shell.bash hook)"
conda activate inter2.5

# PYTHONUNBUFFERED=1 python -u -W ignore::UserWarning -m Med_eval_kit_templates.run
PYTHONUNBUFFERED=1 python -u -W ignore::UserWarning -m Med_eval_kit_mimic.run
# PYTHONUNBUFFERED=1 python download.py






#--------------生成测试集
# RUNPATH=/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/A-Chest-MINT-CoT
# cd $RUNPATH

# # source activate mint
# source /cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/WQS/env/anaconda3/etc/profile.d/conda.sh
# conda activate mint

# PYTHONUNBUFFERED=1 python -u -W ignore::UserWarning -m IU-test.generate_test_json