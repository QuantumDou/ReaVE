#!/usr/bin/env python3
"""
合并多个 GPU 节点评估结果的分片脚本
用法: python merge_shard_results.py --output_dir <主输出目录> --num_shards <节点数>
"""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from .run import DEFAULT_OUTPUT


from .metrics import compute_nlg_metrics


def merge_shard_results(output_dir: str, num_shards: int) -> Dict[str, Any]:
    """合并所有分片的结果"""
    all_records: List[Dict[str, Any]] = []
    all_predictions: List[str] = []
    all_references: List[str] = []
    
    # 收集所有分片的结果
    for shard_id in range(num_shards):
        shard_dir = os.path.join(output_dir, f"shard_{shard_id}")
        preds_path = os.path.join(shard_dir, "predictions.jsonl")
        
        if not os.path.exists(preds_path):
            print(f"Warning: Shard {shard_id} results not found at {preds_path}")
            continue
        
        # 解析 JSONL
        with open(preds_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            decoder = json.JSONDecoder()
            pos = 0
            n = len(content)
            while pos < n:
                obj, idx = decoder.raw_decode(content, pos)
                all_records.append(obj)
                all_predictions.append(obj["prediction"])
                all_references.append(obj["reference"])
                pos = idx
                while pos < n and content[pos].isspace():
                    pos += 1
        
        print(f"Loaded {len([r for r in all_records if r.get('_shard') == shard_id])} records from shard {shard_id}")
    
    # 按原始 ID 排序（如果有的话）
    if all_records and "id" in all_records[0]:
        all_records.sort(key=lambda x: x.get("id", ""))
        # 重新提取排序后的 predictions 和 references
        all_predictions = [r["prediction"] for r in all_records]
        all_references = [r["reference"] for r in all_records]
    
    # 计算全局 metrics
    print(f"\nComputing metrics on {len(all_predictions)} total samples...")
    global_metrics = compute_nlg_metrics(all_predictions, all_references)
    
    # 读取第一个分片的 metadata（模型配置等）
    metadata = {}
    for shard_id in range(num_shards):
        shard_dir = os.path.join(output_dir, f"shard_{shard_id}")
        metrics_path = os.path.join(shard_dir, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                shard_meta = json.load(f)
                if not metadata:
                    metadata = {
                        "model_name": shard_meta.get("model_name"),
                        "base_model_path": shard_meta.get("base_model_path"),
                        "generation_kwargs": shard_meta.get("generation_kwargs"),
                    }
            break
    
    # 保存合并后的结果
    merged_preds_path = os.path.join(output_dir, "predictions.jsonl")
    merged_metrics_path = os.path.join(output_dir, "metrics.json")
    
    with open(merged_preds_path, "w", encoding="utf-8") as f:
        for item in all_records:
            json_str = json.dumps(item, ensure_ascii=False, indent=2)
            f.write(json_str + "\n")
    
    merged_metadata = dict(metadata)
    merged_metadata.update(global_metrics)
    merged_metadata["num_shards"] = num_shards
    merged_metadata["total_samples"] = len(all_records)
    
    with open(merged_metrics_path, "w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Merged results saved to:")
    print(f"  - {merged_preds_path}")
    print(f"  - {merged_metrics_path}")
    print(f"\nGlobal metrics:")
    for key, value in global_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return global_metrics


def main():
    parser = argparse.ArgumentParser("Merge sharded evaluation results")
    # parser.add_argument("--output_dir", required=True, help="主输出目录（包含 shard_0, shard_1, ... 子目录）")
    parser.add_argument(
        "--output_dir", 
        default=DEFAULT_OUTPUT, 
        help=f"主输出目录 (默认: {DEFAULT_OUTPUT})"
    )
    parser.add_argument("--num_shards", type=int, required=True, help="总节点数")
    args = parser.parse_args()
    
    merge_shard_results(args.output_dir, args.num_shards)


if __name__ == "__main__":
    main()
