from __future__ import annotations

import argparse
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .evaluator import MedFullEvaluator
from .model_adapter import register_model

DEFAULT_DATASET = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/A-Chest-MINT-CoT/IU-test/iu_test.json" #iu_test_demo.json
# DEFAULT_DATASET = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Med_eval_kit/dataset/iu_test_reconstructed.json"
DEFAULT_IMAGE_BASE = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/TriCL_iuxray/data/images/"
DEFAULT_OUTPUT = "/cephfs/volumes/hpc_data_usr/k1623928/c50db1e8-c7a8-4708-9be4-3db18a0ea062/scratch_tmp/TIAN/Qwen2.5_CoT/Med_eval_kit/result/qwen2_5_vl_3b_full_v4-3b-54880-1e-6-unfreeze/checkpoint-2830/0.8"
DEFAULT_MODEL_NAME = "med-full-qwen2.5vl-3b"  #"med-full-qwen-7b"
DEFAULT_BASE_MODEL_PATH = "/cephfs/volumes/hpc_data_prj/proj_loukides/bde7318a-93a3-4cd6-90a4-9007857ac4ea/haodi/TIAN/qwen2.5/saves/qwen2_5_vl_3b_full_v4-3b-54880-1e-6-unfreeze/checkpoint-2830"
DEFAULT_DEVICE = "auto"
DEFAULT_DTYPE = "bfloat16"
DEFAULT_GENERATION = {
    "max_new_tokens": 2048,
    "temperature": 0.1,  #0.1
    "top_p": 0.9,
    "repetition_penalty": 1.0, 
    "do_sample": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Medical Full-Model Evaluation Kit")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="测试集 JSON/JSONL 路径")
    parser.add_argument("--image_base_dir", default=DEFAULT_IMAGE_BASE, help="图像基路径")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT, help="评估结果输出目录")
    parser.add_argument("--model_name", default=DEFAULT_MODEL_NAME, help="模型别名(注册用)")
    parser.add_argument("--base_model_path", default=DEFAULT_BASE_MODEL_PATH, help="全量模型 checkpoint 路径")
    parser.add_argument("--generation_kwargs", default=None, help="JSON 字符串，覆盖生成参数")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="最大生成长度")
    parser.add_argument("--max_length", type=int, default=None, help="生成总长度(覆盖 max_new_tokens)")
    parser.add_argument("--min_length", type=int, default=None, help="生成最小长度")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="计算设备(auto/cuda/cpu/mps)")
    parser.add_argument("--dtype", default=DEFAULT_DTYPE, choices=["auto", "float16", "bfloat16", "float32"], help="推理精度")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    parser.add_argument("--prompt_template", default="full", choices=["full", "step_only"])
    parser.add_argument("--no_mulberry_prompt", action="store_true", help="不使用 Mulberry 风格提示")
    #新增
    parser.add_argument("--shard_id", type=int, default=None, help="当前节点 ID (0-based, 用于多节点并行)")
    parser.add_argument("--num_shards", type=int, default=None, help="总节点数 (用于多节点并行)")
    return parser.parse_args()


def _parse_dtype(dtype_str: str):
    import torch

    mapping = {
        "auto": None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_str]


def main() -> None:
    args = parse_args()
    print("prediction后处理")

    gen_kwargs = DEFAULT_GENERATION.copy()
    if args.max_new_tokens is not None:
        gen_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.max_length is not None:
        gen_kwargs["max_length"] = args.max_length
        gen_kwargs.pop("max_new_tokens", None)
    if args.min_length is not None:
        gen_kwargs["min_length"] = args.min_length
    if args.generation_kwargs:
        gen_kwargs.update(json.loads(args.generation_kwargs))

    print("===检查点路径:",DEFAULT_BASE_MODEL_PATH)
    print("===输出到output_dir:", args.output_dir)
    print("====使用数据集:", args.dataset)

    register_model(
        args.model_name,
        base_model_path=args.base_model_path,
        device=args.device,
        torch_dtype=_parse_dtype(args.dtype),
        generation_kwargs=gen_kwargs,
        verbose=args.verbose,
        prompt_template=args.prompt_template,
        use_mulberry_prompt=not args.no_mulberry_prompt,
    )

    # evaluator = MedFullEvaluator(
    #     dataset_path=args.dataset,
    #     image_base_dir=args.image_base_dir,
    #     output_dir=args.output_dir,
    #     model_name=args.model_name,
    # )
    evaluator = MedFullEvaluator(
        dataset_path=args.dataset,
        image_base_dir=args.image_base_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        shard_id=args.shard_id,  # 新增
        num_shards=args.num_shards,  # 新增
    )
    metrics = evaluator.run()

    print("\nEvaluation complete:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    


if __name__ == "__main__":
    main()