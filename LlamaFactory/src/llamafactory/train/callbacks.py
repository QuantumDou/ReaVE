# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional
import subprocess
from transformers.trainer_utils import get_last_checkpoint


import torch
import transformers
from peft import PeftModel
from transformers import PreTrainedModel, ProcessorMixin, TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from typing_extensions import override

from ..extras import logging
from ..extras.constants import TRAINER_LOG, V_HEAD_SAFE_WEIGHTS_NAME, V_HEAD_WEIGHTS_NAME
from ..extras.misc import get_peak_memory, is_env_enabled, use_ray
from ..extras.packages import is_safetensors_available


if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file


if TYPE_CHECKING:
    from transformers import TrainerControl, TrainerState, TrainingArguments
    from trl import AutoModelForCausalLMWithValueHead

    from ..hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = logging.get_logger(__name__)


# class MedicalEvalCallback(TrainerCallback):
#     r"""在每个 epoch 结束后，使用当前 LoRA 权重跑一次 medical_eval_kit，并按 BLEU-4 选择最优 checkpoint。

#     说明：
#       - 为了避免在 LlamaFactory 内部做复杂集成，这里通过子进程调用 `medical_eval_kit/run.py`。
#       - 配置参数可以从 YAML 文件中读取，如果未提供则使用默认值。
#     """

#     def __init__(
#         self,
#         project_root: Optional[str] = None,
#         eval_script: Optional[str] = None,
#         dataset_path: Optional[str] = None,
#         image_base_dir: Optional[str] = None,
#         eval_root_dir: Optional[str] = None,
#         device: Optional[str] = None,
#         dtype: Optional[str] = None,
#     ) -> None:
#         """
#         Args:
#             project_root: 项目根目录路径（如果未提供，会尝试从默认路径推断）
#             eval_script: medical_eval_kit/run.py 的路径（如果未提供，会基于 project_root 推断）
#             dataset_path: 测试数据集 JSON 文件路径
#             image_base_dir: 图像文件的基础目录
#             eval_root_dir: 训练过程中评估结果的输出目录
#             device: 评测时使用的设备（"cpu", "cuda", "mps", "auto"），默认 "cpu"
#             dtype: 评测时使用的数据类型（"float16", "bfloat16", "float32", "auto"），默认 "bfloat16"
#         """
#         # 默认值（向后兼容）
#         default_project_root = "/Users/tsn/Documents/Files/MyProgram/Qwen2.5_Report_Generation/Qwen2_5_CoT"
#         default_dataset_path = "/Users/tsn/Documents/Files/MyProgram/Report Generation/MINT-CoT/IU-test/iu_test_demo.json"
#         default_image_base_dir = "/Users/tsn/Documents/Files/MyProgram/Report Generation/MINT-CoT/IU-test"
#         default_device = "cuda"
#         default_dtype = "bfloat16"

#         # 使用传入的参数或默认值
#         self.project_root = project_root or default_project_root
#         self.eval_script = eval_script or os.path.join(self.project_root, "medical_eval_kit", "run.py")
#         self.dataset_path = dataset_path or default_dataset_path
#         self.image_base_dir = image_base_dir or default_image_base_dir
#         self.eval_root_dir = eval_root_dir or os.path.join(self.project_root, "medical_eval_kit", "medical_eval_results_from_training")
#         self.device = device or default_device
#         self.dtype = dtype or default_dtype

#         os.makedirs(self.eval_root_dir, exist_ok=True)

#         # 记录当前最优 BLEU-4 以及对应的 LoRA checkpoint 目录
#         self.best_score: Optional[float] = None
#         self.best_ckpt_dir: Optional[str] = None

#     # def verify_vision_frozen(self, model):
#     #     """验证视觉编码器是否仍然被冻结"""
#     #     vision_params = [p for name, p in model.named_parameters() 
#     #                     if 'visual.patch_embed' in name or 'visual.blocks' in name]
#     #     for param in vision_params:
#     #         assert not param.requires_grad, f"Vision parameter {param} should be frozen!"
#     #     logger.info_rank0("✓ Vision encoder is still frozen")


#     def verify_vision_frozen(self, model):
#         """验证视觉编码器是否仍然被冻结"""
#         vision_params = [p for name, p in model.named_parameters() 
#                         if 'visual.patch_embed' in name or 'visual.blocks' in name]
#         for param in vision_params:
#             assert not param.requires_grad, f"Vision parameter {param} should be frozen!"
#         logger.info_rank0("✓ Vision encoder is still frozen")

#     def verify_language_model_trainable(self, model):
#         """验证语言模型是否正常可训练（未被意外冻结）"""
#         # Qwen2.5-VL 的语言模型参数路径
#         language_model_keys = [
#             'model.embed_tokens',  # 词嵌入
#             'model.layers',        # Transformer 层
#             'model.norm',          # 层归一化
#             'lm_head',             # 语言模型头
#         ]
        
#         # 需要排除的参数（这些可能被冻结或不是语言模型核心部分）
#         excluded_keys = [
#             'visual.',             # 视觉编码器
#             'visual.merger',       # 多模态投影器（如果被冻结）
#         ]
        
#         trainable_lm_params = []
#         frozen_lm_params = []
        
#         for name, param in model.named_parameters():
#             # 检查是否是语言模型参数
#             is_lm_param = any(lm_key in name for lm_key in language_model_keys)
#             # 排除视觉编码器
#             is_excluded = any(excluded_key in name for excluded_key in excluded_keys)
            
#             if is_lm_param and not is_excluded:
#                 if param.requires_grad:
#                     trainable_lm_params.append(name)
#                 else:
#                     frozen_lm_params.append(name)
        
#         # 验证：至少应该有一些语言模型参数是可训练的
#         if len(trainable_lm_params) == 0:
#             logger.warning_rank0("⚠️ 警告：没有发现可训练的语言模型参数！")
#             logger.warning_rank0(f"   冻结的语言模型参数数量: {len(frozen_lm_params)}")
#             if len(frozen_lm_params) > 0:
#                 logger.warning_rank0(f"   示例冻结参数: {frozen_lm_params[:3]}")
#         else:
#             logger.info_rank0(f"✓ Language model is trainable ({len(trainable_lm_params)} trainable params, {len(frozen_lm_params)} frozen params)")
#             if len(trainable_lm_params) <= 5:
#                 logger.info_rank0(f"   可训练参数: {trainable_lm_params}")
#             else:
#                 logger.info_rank0(f"   可训练参数示例: {trainable_lm_params[:3]} ... (共 {len(trainable_lm_params)} 个)")

#     def verify_special_params(self, model):
#         """验证特殊参数（interleave_token, s_projection, vis_projection, logit_scale）的状态"""
#         special_params_config = {
#             'interleave_token': None,  # None 表示检查是否存在，不强制要求可训练状态
#             's_projection': None,
#             'vis_projection': None,
#             'logit_scale': None,
#         }
        
#         found_params = {}
        
#         for name, param in model.named_parameters():
#             for param_name in special_params_config.keys():
#                 if param_name in name:
#                     found_params[param_name] = {
#                         'name': name,
#                         'requires_grad': param.requires_grad,
#                         'shape': list(param.shape),
#                         'dtype': str(param.dtype),
#                     }
#                     break
        
#         # 打印检查结果
#         if found_params:
#             logger.info_rank0("✓ Special parameters status:")
#             for param_name, info in found_params.items():
#                 status = "可训练" if info['requires_grad'] else "冻结"
#                 logger.info_rank0(f"   {param_name}: {status} (shape: {info['shape']}, dtype: {info['dtype']})")
#         else:
#             logger.warning_rank0("⚠️ 未找到任何特殊参数（interleave_token, s_projection, vis_projection, logit_scale）")
        
#         return found_params


#     @override
#     def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         # 1. 进程同步：确保所有卡都跑完了 save_model
#         if torch.distributed.is_initialized():
#             rank = torch.distributed.get_rank()
#             print(f"[Rank {rank}] 🛑 已到达同步点...", flush=True) # 调试时可开，平时可关
#             torch.distributed.barrier()

#         #########这里验证一下model的参数是否需要更新
#         model = kwargs.get("model")
#         if model is not None:
#             self.verify_vision_frozen(model)
#             self.verify_language_model_trainable(model)
#             self.verify_special_params(model)  # 【新增】验证特殊参数
#         #######################################

#         # 2. 【关键修改】只在"世界主进程"检查文件，防止多节点冲突
#         if state.is_world_process_zero:
#             print("所有卡已完成保存操作，主进程开始后处理...")

#             max_retries = 60
#             found_ckpt = None
            
#             for i in range(max_retries):
#                 current_ckpt = get_last_checkpoint(args.output_dir)
                
#                 # 检查基础目录
#                 if current_ckpt and os.path.isdir(current_ckpt):
#                     # 检查关键的元数据文件
#                     required_meta = ["config.json", "trainer_state.json", "vocab.json", "merges.txt"] #"preprocessor_config.json"
#                     for file in required_meta:
#                         if not os.path.exists(os.path.join(current_ckpt, file)):
#                             print(f"[MedicalEval] ⏳ 缺文件: {file}", flush=True)
#                             break
#                         else:
#                             print(f"[MedicalEval] ✅ 检测到文件: {file}", flush=True)
#                     if all(os.path.exists(os.path.join(current_ckpt, f)) for f in required_meta):
                        
#                         # === 分支 A：检查分片模型 ===
#                         index_file = os.path.join(current_ckpt, "model.safetensors.index.json")
                        
#                         if os.path.exists(index_file):
#                             try:
#                                 with open(index_file, 'r') as f:
#                                     index_data = json.load(f)
#                                 required_shards = set(index_data['weight_map'].values())
                                
#                                 all_shards_exist = True
#                                 for shard_name in required_shards:
#                                     if not os.path.exists(os.path.join(current_ckpt, shard_name)):
#                                         # print(f"[MedicalEval] ⏳ 缺分片: {shard_name}", flush=True)
#                                         all_shards_exist = False
#                                         break 
                                
#                                 if all_shards_exist:
#                                     found_ckpt = current_ckpt
#                                     print(f"[MedicalEval] ✅ 检测到完整分片模型，耗时 {i} 秒", flush=True)
#                                     break
                                    
#                             except (json.JSONDecodeError, KeyError):
#                                 pass
                        
#                         # === 分支 B：检查单文件模型 ===
#                         elif os.path.exists(os.path.join(current_ckpt, "model.safetensors")):
#                             found_ckpt = current_ckpt
#                             print(f"[MedicalEval] ✅ 检测到完整单文件模型，耗时 {i} 秒", flush=True)
#                             break
                        
#                         # === 分支 C：检查 LoRA 权重 ===
#                         elif os.path.exists(os.path.join(current_ckpt, "adapter_model.safetensors")):
#                             found_ckpt = current_ckpt
#                             print(f"[MedicalEval] ✅ 检测到 LoRA 权重，耗时 {i} 秒", flush=True)
#                             break

#                 # 还没好，等待 (减少打印频率，每5秒打一次)
#                 if i % 5 == 0:
#                     print(f"[MedicalEval] ⏳ 等待完整文件落盘... ({i+1}/{max_retries})", flush=True)
#                 time.sleep(1)
            
#             # 3. 拿到结果，开始干活
#             if found_ckpt:
#                 epoch_idx = int(state.epoch) if state.epoch is not None else 0
#                 print(f"**************** 开启评估 (Epoch {epoch_idx}) ******************")
#                 is_lora = os.path.exists(os.path.join(found_ckpt, "adapter_config.json")) # 用 found_ckpt 更严谨

#                 eval_output_dir = os.path.join(self.eval_root_dir, f"epoch_{epoch_idx}")
#                 eval_output_dir = os.path.abspath(eval_output_dir)
#                 os.makedirs(eval_output_dir, exist_ok=True)

#                 cmd = [
#                     sys.executable, "-m", "medical_eval_kit.run",
#                     "--dataset", os.path.abspath(self.dataset_path),
#                     "--image_base_dir", os.path.abspath(self.image_base_dir),
#                     "--output_dir", eval_output_dir,
#                     "--device", self.device,
#                     "--dtype", self.dtype,
#                     "--lora_path" if is_lora else "--base_model_path", found_ckpt
#                 ]
                
#                 # 执行评估
#                 eval_success = False
#                 try:
#                     subprocess.run(
#                         cmd, 
#                         check=True, 
#                         text=True, 
#                         cwd=self.project_root
#                     )
#                     eval_success = True
#                 except subprocess.CalledProcessError as e:
#                     print(f"❌ 评估脚本执行失败，退出码: {e.returncode}")
                
#                 # 只有评估成功才去读分数
#                 if eval_success:
#                     # 【修正 1】先定义 metrics_path
#                     metrics_path = os.path.join(eval_output_dir, "metrics.json")
                    
#                     if os.path.exists(metrics_path):
#                         try:
#                             # 【修正 2】文件存在后再读取
#                             with open(metrics_path, "r", encoding="utf-8") as f:
#                                 metrics = json.load(f)
                            
#                             #需要最佳模型的判断分数，只改这里即可
#                             current_score = float(metrics.get("BLEU-1", 0.0))
#                             best_prev = self.best_score if self.best_score is not None else 0.0
                            
#                             print(f"Epoch {epoch_idx}: BLEU-1 = {current_score:.4f} (Best: {best_prev:.4f})")
                                    
#                             # 去留决策
#                             import shutil
#                             best_dir = os.path.join(args.output_dir, "best_medical_bleu")
                            
#                             if self.best_score is None or current_score >= self.best_score:
#                                 print(f"🚀 New Best Model! Copying from {found_ckpt} to {best_dir}")
                                
#                                 # 【优化】使用 dirs_exist_ok=True，不需要先 rm 再 copy，原子性更好，不易出错
#                                 shutil.copytree(found_ckpt, best_dir, dirs_exist_ok=True)
                                
#                                 self.best_score = current_score
#                                 self.best_ckpt_dir = best_dir
#                         except Exception as e:
#                             print(f"❌ 读取 metrics.json 或复制模型时出错: {e}")
#                     else:
#                         print(f"⚠️ 未找到 metrics.json: {metrics_path}，跳过最佳模型更新。")

#             else:
#                 print("❌ 错误：60秒内未检测到完整模型，跳过评估！", flush=True)
            
#             print(f"**************** 评估结束 (Epoch {epoch_idx}) ******************")

#         # 4. 可选：再次同步，防止Rank 0还在复制文件，其他Rank就开始下一个epoch的计算了（虽然通常不需要）
#         if torch.distributed.is_initialized():
#              torch.distributed.barrier()
       
#         # import torch.distributed as dist
        


#         # # 1. 调试日志
#         # rank = dist.get_rank() if dist.is_initialized() else 0
#         # # logger.info(f"[Rank {rank}] MedicalEvalCallback.on_save: 被调用 (should_save={args.should_save}, is_world_process_zero={state.is_world_process_zero})")

#         # if dist.is_initialized():
#         #     # 只有 Rank 0 打印日志，避免刷屏
#         #     # if rank == 0:
#         #         # logger.info_rank0("MedicalEvalCallback: 进入 on_save，等待所有 Rank 同步...")
#         #     dist.barrier()
#         # # logger.info(f"MedicalEvalCallback: 所有 Rank 同步完成！！！")
        
     
        
        
#         # try:
#         #     if state.is_world_process_zero:
#         #         # logger.info_rank0("MedicalEvalCallback.on_save: Rank 0 开始执行评估...")
#         #         # 获取框架保存的 checkpoint 目录
#         #         # 框架会在 save_strategy: epoch 时保存到 checkpoint-{step} 目录
#         #         epoch_idx = int(state.epoch) if state.epoch is not None else 0
                
#         #         # 框架保存的 checkpoint 目录格式：checkpoint-{step}
#         #         checkpoint_dir = None
#         #         if hasattr(state, "global_step") and state.global_step is not None:
#         #             checkpoint_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
#         #         else:
#         #             # 如果没有 global_step，尝试找到最新的 checkpoint
#         #             import glob
#         #             checkpoint_pattern = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-*")
#         #             checkpoints = glob.glob(checkpoint_pattern)
#         #             if checkpoints:
#         #                 # 按修改时间排序，取最新的
#         #                 checkpoints.sort(key=os.path.getmtime, reverse=True)
#         #                 checkpoint_dir = checkpoints[0]
                
#         #         if checkpoint_dir is None or not os.path.exists(checkpoint_dir):
#         #             logger.warning_rank0(
#         #                 f"MedicalEvalCallback: 找不到 checkpoint 目录，跳过评估。"
#         #                 f"期望路径: {checkpoint_dir if checkpoint_dir else 'N/A'}"
#         #             )
#         #             return
                
#         #         # 确保 checkpoint_dir 是绝对路径
#         #         checkpoint_dir = os.path.abspath(checkpoint_dir)
                
#         #         # 等待 checkpoint 文件完全保存（检查关键文件是否存在）
#         #         import time
#         #         max_wait_time = 60  # 最多等待 60 秒
#         #         wait_interval = 1  # 每秒检查一次
#         #         waited_time = 0
                
#         #         # 检查关键文件是否存在（模型权重文件或 index 文件）
#         #         # checkpoint_ready = False
#         #         # while waited_time < max_wait_time:
#         #         #     # 检查是否有模型权重文件（safetensors 或 pytorch_model.bin）
#         #         #     has_model_files = (
#         #         #         os.path.exists(os.path.join(checkpoint_dir, "model.safetensors.index.json")) or
#         #         #         os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) or
#         #         #         os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or
#         #         #         os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")) or
#         #         #         os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
#         #         #     )
                    
#         #         #     if has_model_files:
#         #         #         checkpoint_ready = True
#         #         #         break
#         #         # 检查关键文件是否存在（模型权重文件、processor 文件等）
#         #         checkpoint_ready = False
#         #         while waited_time < max_wait_time:
#         #             # 检查是否有模型权重文件（safetensors 或 pytorch_model.bin）
#         #             has_model_files = (
#         #                 os.path.exists(os.path.join(checkpoint_dir, "model.safetensors.index.json")) or
#         #                 os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) or
#         #                 os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or
#         #                 os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")) or
#         #                 os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
#         #             )

#         #             # 检查所有4个分片文件是否存在
#         #             has_all_shards = all(
#         #                 os.path.exists(os.path.join(checkpoint_dir, f"model-{i:05d}-of-00004.safetensors"))
#         #                 for i in range(1, 5)
#         #             )
                    
#         #             # 检查 processor 相关文件（Qwen2.5-VL 需要）
#         #             has_processor = os.path.exists(os.path.join(checkpoint_dir, "preprocessor_config.json"))
#         #             has_tokenizer = os.path.exists(os.path.join(checkpoint_dir, "tokenizer_config.json"))
                    
#         #             # 模型文件必须存在，processor 或 tokenizer 至少有一个存在
#         #             if has_model_files and has_processor and has_tokenizer and has_all_shards:
#         #                 checkpoint_ready = True
#         #                 break
                    
#         #             # logger.info_rank0(f"MedicalEvalCallback: 等待 checkpoint 文件保存完成... ({waited_time}s/{max_wait_time}s)")
#         #             time.sleep(wait_interval)
#         #             waited_time += wait_interval
                
#         #         if not checkpoint_ready:
#         #             logger.warning_rank0(
#         #                 f"MedicalEvalCallback: checkpoint 文件在 {max_wait_time} 秒内未完全保存，跳过评估。"
#         #                 f"checkpoint_dir: {checkpoint_dir}"
#         #             )
#         #             return
                
#         #         # logger.info_rank0(f"MedicalEvalCallback: checkpoint 文件已就绪，继续评估")
                
#         #         logger.info_rank0(f"=====================  开启评估流程 (Epoch {epoch_idx})  ====================")
#         #         logger.info_rank0(f"MedicalEvalCallback: 使用框架保存的 checkpoint: {checkpoint_dir}")
                
#         #         # 检查是否是 LoRA 模型（通过检查 checkpoint 目录中是否有 adapter_config.json）
#         #         is_lora = os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
                
#         #         # 运行评估脚本（直接使用框架保存的 checkpoint）
#         #         if torch.cuda.is_available():
#         #             torch.cuda.empty_cache()
                
#         #         eval_output_dir = os.path.join(self.eval_root_dir, f"epoch_{epoch_idx}")
#         #         eval_output_dir = os.path.abspath(eval_output_dir)
#         #         os.makedirs(eval_output_dir, exist_ok=True)
                
#         #         # 确保所有路径都是绝对路径
#         #         dataset_path_abs = os.path.abspath(self.dataset_path) if os.path.exists(self.dataset_path) else self.dataset_path
#         #         image_base_dir_abs = os.path.abspath(self.image_base_dir) if os.path.exists(self.image_base_dir) else self.image_base_dir
                
#         #         cmd = [
#         #             sys.executable,
#         #             "-m",
#         #             "medical_eval_kit.run",
#         #             "--dataset",
#         #             dataset_path_abs,
#         #             "--image_base_dir",
#         #             image_base_dir_abs,
#         #             "--output_dir",
#         #             eval_output_dir,
#         #             "--device",
#         #             self.device,
#         #             "--dtype",
#         #             self.dtype,
#         #         ]
                
#         #         if is_lora:
#         #             cmd.extend(["--lora_path", checkpoint_dir])
#         #         else:
#         #             cmd.extend(["--base_model_path", checkpoint_dir])
                
#         #         logger.info_rank0(f"MedicalEvalCallback: 执行评估: {' '.join(cmd)}")
#         #         logger.info_rank0(f"MedicalEvalCallback: 工作目录: {self.project_root}")
#         #         logger.info_rank0(f"MedicalEvalCallback: 载入权重路径: {checkpoint_dir} (exists: {os.path.exists(checkpoint_dir)})")
                
#         #         try:
#         #             import subprocess
#         #             # 添加超时设置，避免无限等待（评估可能需要较长时间，设置 1 小时超时）
#         #             # logger.info_rank0("MedicalEvalCallback: *********开始执行评估子进程********")
#         #             result = subprocess.run(
#         #                 cmd, 
#         #                 check=True, 
#         #                 # capture_output=True, 
#         #                 text=True, 
#         #                 cwd=self.project_root
#         #             )
#         #             # logger.info_rank0("MedicalEvalCallback: *********评估子进程执行完成********")
#         #             # logger.info_rank0(f"MedicalEvalCallback eval stdout:\n{result.stdout}")
#         #             # if result.stderr:
#         #             #     logger.info_rank0(f"MedicalEvalCallback eval stderr:\n{result.stderr}")
                    
#         #             # 读取评估结果中的 BLEU-4 分数
#         #             metrics_path = os.path.join(eval_output_dir, "metrics.json")
#         #             if not os.path.exists(metrics_path):
#         #                 logger.warning_rank0(
#         #                     f"MedicalEvalCallback: metrics.json not found at {metrics_path}, skip best-model selection."
#         #                 )
#         #             else:
#         #                 try:
#         #                     with open(metrics_path, "r", encoding="utf-8") as f:
#         #                         metrics = json.load(f)
#         #                     bleu4 = float(metrics.get("bleu_4", 0.0))
                            
#         #                     logger.info_rank0(f"MedicalEvalCallback: epoch {epoch_idx} BLEU-4 = {bleu4:.4f}")
                            
#         #                     # 去留决策：如果是最优模型，将其复制到 best_medical_bleu；如果不是，可以选择删除
#         #                     import shutil
#         #                     best_dir = os.path.join(args.output_dir, "best_medical_bleu")
                            
#         #                     if self.best_bleu4 is None or bleu4 >= self.best_bleu4:
#         #                         logger.info_rank0(
#         #                             f"MedicalEvalCallback: new best BLEU-4 {bleu4:.4f} (prev: {self.best_bleu4}), "
#         #                             f"copying checkpoint from {checkpoint_dir} to {best_dir}"
#         #                         )
                                
#         #                         # 如果 best_dir 已存在，先删除
#         #                         if os.path.exists(best_dir):
#         #                             shutil.rmtree(best_dir)
                                
#         #                         # 复制 checkpoint 到 best_dir（保留原始 checkpoint 目录）
#         #                         shutil.copytree(checkpoint_dir, best_dir)
                                
#         #                         self.best_bleu4 = bleu4
#         #                         self.best_ckpt_dir = best_dir
#         #                         logger.info_rank0(f"Epoch {epoch_idx}: updated best BLEU-4 {bleu4:.4f}, saved to {best_dir}")
#         #                     else:
#         #                         logger.info_rank0(
#         #                             f"MedicalEvalCallback: BLEU-4 {bleu4:.4f} < best {self.best_bleu4:.4f}, "
#         #                             f"keeping checkpoint {checkpoint_dir} (framework managed)"
#         #                         )
#         #                         # 注意：这里不删除 checkpoint，因为它是框架管理的
#         #                         # 如果需要删除，可以取消下面的注释
#         #                         # if os.path.exists(checkpoint_dir):
#         #                         #     shutil.rmtree(checkpoint_dir)
#         #                 except Exception as e:
#         #                     logger.warning_rank0(f"MedicalEvalCallback: failed to read BLEU-4 from {metrics_path}: {e}")
                
#         #         except subprocess.CalledProcessError as e:
#         #             logger.warning_rank0(
#         #                 f"MedicalEvalCallback: evaluation failed for epoch {epoch_idx} "
#         #                 f"with return code {e.returncode}: {e}"
#         #             )
#         #             if e.stdout:
#         #                 logger.warning_rank0(f"MedicalEvalCallback eval stdout (error case):\n{e.stdout}")
#         #             if e.stderr:
#         #                 logger.warning_rank0(f"MedicalEvalCallback eval stderr (error case):\n{e.stderr}")

#         #     else:
#         #         # -----------------------------------------------------
#         #         # Rank 1-3 逻辑：什么都不做，直接通过
#         #         # -----------------------------------------------------
#         #         # 它们会直接跳到 finally 块去等待 Rank 0
#         #         pass

        
#         # except Exception as e:
#         #     logger.error_rank0(f"MedicalEvalCallback: 发生异常: {e}")
#         #     import traceback
#         #     logger.error_rank0(traceback.format_exc())
        
#         # finally:
#         #     # 最终同步：Rank 1-3 在这里等待 Rank 0 跑完
#         #     # logger.info_rank0(f"MedicalEvalCallback.on_save: 进入 finally 块 (is_world_process_zero={state.is_world_process_zero})")
#         #     if dist.is_initialized():
#         #         # logger.info_rank0("MedicalEvalCallback.on_save: 等待所有 rank 到达最终 barrier...")
#         #         dist.barrier()
#         #         # logger.info_rank0("MedicalEvalCallback.on_save: 最终 barrier 通过")
            
#         #     if state.is_world_process_zero:
#         #         logger.info_rank0("============================   评估结束   ===================================")
            


def fix_valuehead_checkpoint(
    model: "AutoModelForCausalLMWithValueHead", output_dir: str, safe_serialization: bool
) -> None:
    r"""Fix the valuehead checkpoint files.

    The model is already unwrapped.

    There are three cases:
    1. full tuning without ds_zero3: state_dict = {"model.layers.*": ..., "v_head.summary.*": ...}
    2. lora tuning without ds_zero3: state_dict = {"v_head.summary.*": ...}
    3. under deepspeed zero3: state_dict = {"pretrained_model.model.layers.*": ..., "v_head.summary.*": ...}

    We assume `stage3_gather_16bit_weights_on_model_save=true`.
    """
    if not isinstance(model.pretrained_model, (PreTrainedModel, PeftModel)):
        return

    if safe_serialization:
        path_to_checkpoint = os.path.join(output_dir, SAFE_WEIGHTS_NAME)
        with safe_open(path_to_checkpoint, framework="pt", device="cpu") as f:
            state_dict: dict[str, torch.Tensor] = {key: f.get_tensor(key).clone() for key in f.keys()}
    else:
        path_to_checkpoint = os.path.join(output_dir, WEIGHTS_NAME)
        state_dict: dict[str, torch.Tensor] = torch.load(path_to_checkpoint, map_location="cpu", weights_only=True)

    os.remove(path_to_checkpoint)
    decoder_state_dict, v_head_state_dict = {}, {}
    for name, param in state_dict.items():
        if name.startswith("v_head."):
            v_head_state_dict[name] = param
        else:
            decoder_state_dict[name.replace("pretrained_model.", "", 1)] = param

    model.pretrained_model.save_pretrained(
        output_dir, state_dict=decoder_state_dict or None, safe_serialization=safe_serialization
    )

    if safe_serialization:
        save_file(v_head_state_dict, os.path.join(output_dir, V_HEAD_SAFE_WEIGHTS_NAME), metadata={"format": "pt"})
    else:
        torch.save(v_head_state_dict, os.path.join(output_dir, V_HEAD_WEIGHTS_NAME))

    logger.info_rank0(f"Value head model saved at: {output_dir}")


class FixValueHeadModelCallback(TrainerCallback):
    r"""A callback for fixing the checkpoint for valuehead models."""

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            fix_valuehead_checkpoint(
                model=kwargs.pop("model"), output_dir=output_dir, safe_serialization=args.save_safetensors
            )


class SaveProcessorCallback(TrainerCallback):
    r"""A callback for saving the processor."""

    def __init__(self, processor: "ProcessorMixin") -> None:
        self.processor = processor

    @override
    def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            output_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
            try:
                self.processor.save_pretrained(output_dir)
                logger.info_rank0(f"Processor saved to {output_dir}")
            except Exception as e:
                logger.warning_rank0(f"Failed to save processor to {output_dir}: {e}")

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            try:
                self.processor.save_pretrained(args.output_dir)
                logger.info_rank0(f"Processor saved to {args.output_dir}")
            except Exception as e:
                logger.warning_rank0(f"Failed to save processor to {args.output_dir}: {e}")


class PissaConvertCallback(TrainerCallback):
    r"""A callback for converting the PiSSA adapter to a normal one."""

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            logger.info_rank0(f"Initial PiSSA adapter will be saved at: {pissa_init_dir}.")
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_init_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            model = kwargs.pop("model")
            pissa_init_dir = os.path.join(args.output_dir, "pissa_init")
            pissa_backup_dir = os.path.join(args.output_dir, "pissa_backup")
            pissa_convert_dir = os.path.join(args.output_dir, "pissa_converted")
            logger.info_rank0(f"Converted PiSSA adapter will be saved at: {pissa_convert_dir}.")
            # 1. save a pissa backup with init_lora_weights: True
            # 2. save a converted lora with init_lora_weights: pissa
            # 3. load the pissa backup with init_lora_weights: True
            # 4. delete the initial adapter and change init_lora_weights to pissa
            if isinstance(model, PeftModel):
                init_lora_weights = getattr(model.peft_config["default"], "init_lora_weights")
                setattr(model.peft_config["default"], "init_lora_weights", True)
                model.save_pretrained(pissa_backup_dir, safe_serialization=args.save_safetensors)
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)
                model.save_pretrained(
                    pissa_convert_dir,
                    safe_serialization=args.save_safetensors,
                    path_initial_model_for_weight_conversion=pissa_init_dir,
                )
                model.load_adapter(pissa_backup_dir, "default", is_trainable=True)
                model.set_adapter("default")
                setattr(model.peft_config["default"], "init_lora_weights", init_lora_weights)


class LogCallback(TrainerCallback):
    r"""A callback for logging training and evaluation status."""

    def __init__(self) -> None:
        # Progress
        self.start_time = 0
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        # Status
        self.aborted = False
        self.do_train = False
        # Web UI
        self.webui_mode = is_env_enabled("LLAMABOARD_ENABLED")
        if self.webui_mode and not use_ray():
            signal.signal(signal.SIGABRT, self._set_abort)
            self.logger_handler = logging.LoggerHandler(os.getenv("LLAMABOARD_WORKDIR"))
            logging.add_handler(self.logger_handler)
            transformers.logging.add_handler(self.logger_handler)

    def _set_abort(self, signum, frame) -> None:
        self.aborted = True

    def _reset(self, max_steps: int = 0) -> None:
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = max_steps
        self.elapsed_time = ""
        self.remaining_time = ""

    def _timing(self, cur_steps: int) -> None:
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / cur_steps if cur_steps != 0 else 0
        remaining_time = (self.max_steps - cur_steps) * avg_time_per_step
        self.cur_steps = cur_steps
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def _write_log(self, output_dir: str, logs: dict[str, Any]) -> None:
        with open(os.path.join(output_dir, TRAINER_LOG), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def _create_thread_pool(self, output_dir: str) -> None:
        os.makedirs(output_dir, exist_ok=True)
        self.thread_pool = ThreadPoolExecutor(max_workers=1)

    def _close_thread_pool(self) -> None:
        if self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None

    @override
    def on_init_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if (
            args.should_save
            and os.path.exists(os.path.join(args.output_dir, TRAINER_LOG))
            and args.overwrite_output_dir
        ):
            logger.warning_rank0_once("Previous trainer log in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, TRAINER_LOG))

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if args.should_save:
            self.do_train = True
            self._reset(max_steps=state.max_steps)
            self._create_thread_pool(output_dir=args.output_dir)

    @override
    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        self._close_thread_pool()

    @override
    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if self.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    @override
    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_predict(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not self.do_train:
            self._close_thread_pool()

    @override
    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not args.should_save:
            return

        self._timing(cur_steps=state.global_step)
        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss"),
            eval_loss=state.log_history[-1].get("eval_loss"),
            predict_loss=state.log_history[-1].get("predict_loss"),
            reward=state.log_history[-1].get("reward"),
            accuracy=state.log_history[-1].get("rewards/accuracies"),
            lr=state.log_history[-1].get("learning_rate"),
            epoch=state.log_history[-1].get("epoch"),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if state.num_input_tokens_seen:
            logs["throughput"] = round(state.num_input_tokens_seen / (time.time() - self.start_time), 2)
            logs["total_tokens"] = state.num_input_tokens_seen

        if is_env_enabled("RECORD_VRAM"):
            vram_allocated, vram_reserved = get_peak_memory()
            logs["vram_allocated"] = round(vram_allocated / (1024**3), 2)
            logs["vram_reserved"] = round(vram_reserved / (1024**3), 2)

        logs = {k: v for k, v in logs.items() if v is not None}
        if self.webui_mode and all(key in logs for key in ("loss", "lr", "epoch")):
            log_str = f"'loss': {logs['loss']:.4f}, 'learning_rate': {logs['lr']:2.4e}, 'epoch': {logs['epoch']:.2f}"
            for extra_key in ("reward", "accuracy", "throughput"):
                if logs.get(extra_key):
                    log_str += f", '{extra_key}': {logs[extra_key]:.2f}"

            logger.info_rank0("{" + log_str + "}")

        if self.thread_pool is not None:
            self.thread_pool.submit(self._write_log, args.output_dir, logs)

    @override
    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        if self.do_train:
            return

        if self.aborted:
            sys.exit(0)

        if not args.should_save:
            return

        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if has_length(eval_dataloader):
            if self.max_steps == 0:
                self._reset(max_steps=len(eval_dataloader))
                self._create_thread_pool(output_dir=args.output_dir)

            self._timing(cur_steps=self.cur_steps + 1)
            if self.cur_steps % 5 == 0 and self.thread_pool is not None:
                logs = dict(
                    current_steps=self.cur_steps,
                    total_steps=self.max_steps,
                    percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
                    elapsed_time=self.elapsed_time,
                    remaining_time=self.remaining_time,
                )
                self.thread_pool.submit(self._write_log, args.output_dir, logs)


class ReporterCallback(TrainerCallback):
    r"""A callback for reporting training status to external logger."""

    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
    ) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args
        os.environ["WANDB_PROJECT"] = os.getenv("WANDB_PROJECT", "llamafactory")

    @override
    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        if not state.is_world_process_zero:
            return

        if "wandb" in args.report_to:
            import wandb

            wandb.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )

        if self.finetuning_args.use_swanlab:
            import swanlab  # type: ignore

            swanlab.config.update(
                {
                    "model_args": self.model_args.to_dict(),
                    "data_args": self.data_args.to_dict(),
                    "finetuning_args": self.finetuning_args.to_dict(),
                    "generating_args": self.generating_args.to_dict(),
                }
            )
