import os
from huggingface_hub import snapshot_download

# --- 关键修改 1: 强制重定向所有 HF 缓存到大空间目录 ---
# 这样哪怕是锁文件或临时块，都不会进 Home 目录
os.environ["HF_HOME"] = "/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/huggingface_cache"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 建议加上镜像站，国内环境更快

#下载qwen2.5vl
# model_id = "Qwen/Qwen2.5-VL-3B-Instruct" #"Qwen/Qwen2.5-VL-7B-Instruct"
# download_dir = "/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/huggingface/Qwen2.5-VL-3B-Instruct"

#下载qwen2vl
model_id = "Qwen/Qwen2-VL-2B-Instruct" #"Qwen/Qwen2.5-VL-7B-Instruct"
download_dir = "/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/huggingface/Qwen2-VL-2B-Instruct"



os.makedirs(download_dir, exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

print(f"正在下载 {model_id}...")
print(f"下载位置: {download_dir}")
print(f"缓存位置: {os.environ['HF_HOME']}")
print("=" * 60)

local_dir = snapshot_download(
    repo_id=model_id,
    local_dir=download_dir,
    local_dir_use_symlinks=False, # --- 关键修改 2: 禁用软链接，直接下载到目标路径 ---
    resume_download=True,
    max_workers=4                 # 限制线程数，在 CephFS 上更稳定
)

print("=" * 60)
print(f"✅ 下载完成！")
print(f"模型位置: {local_dir}")