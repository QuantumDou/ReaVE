from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from tqdm import tqdm
from .dataset import FullModelReportDataset
from .metrics import compute_nlg_metrics
from .model_adapter import build_model


# @dataclass
# class MedFullEvaluator:
#     dataset_path: str
#     image_base_dir: Optional[str] = None
#     output_dir: str = "med_full_eval_results"
#     model_name: str = "med-full-qwen"
#     model_kwargs: Dict[str, Any] = field(default_factory=dict)

#     def run(self) -> Dict[str, Any]:
#         os.makedirs(self.output_dir, exist_ok=True)

#         dataset = FullModelReportDataset(self.dataset_path, self.image_base_dir)
#         model = build_model(self.model_name, **self.model_kwargs)

#         predictions: List[str] = []
#         references: List[str] = []
#         raw_records: List[Dict[str, Any]] = []

#         samples = dataset.to_model_inputs()
#         total = len(samples)

#         iterator = tqdm(samples, desc="Evaluating samples", unit="sample")
#         for idx, sample in enumerate(iterator, start=1):
#             output = model.generate(sample)

#             predictions.append(output["prediction"])
#             references.append(sample["reference"])
#             raw_records.append(
#                 {
#                     "id": sample["id"],
#                     "prediction": output["prediction"],
#                     "reference": sample["reference"],
#                     "raw_response": output["raw_response"],
#                     "aux": output["aux"]
#                 }
#             )

#         metrics = compute_nlg_metrics(predictions, references)
#         metadata = {
#             "model_name": self.model_name,
#             "base_model_path": getattr(model, "base_model_path", None),
#             "generation_kwargs": getattr(model, "generate_kwargs", None),
#         }
#         self._write_outputs(raw_records, metrics, metadata)
#         return metrics

    # def _write_outputs(self, records: List[Dict[str, Any]], metrics: Dict[str, float]) -> None:
    #     preds_path = os.path.join(self.output_dir, "predictions.jsonl")
    #     metrics_path = os.path.join(self.output_dir, "metrics.json")

    #     with open(preds_path, "w", encoding="utf-8") as f:
    #         for item in records:
    #             f.write(json.dumps(item, ensure_ascii=False) + "\n")

    #     with open(metrics_path, "w", encoding="utf-8") as f:
    #         json.dump(metrics, f, ensure_ascii=False, indent=2)



    # def _write_outputs(
    #     self,
    #     records: List[Dict[str, Any]],
    #     metrics: Dict[str, float],
    #     metadata: Dict[str, Any],
    # ) -> None:
    #     preds_path = os.path.join(self.output_dir, "predictions.jsonl")
    #     metrics_path = os.path.join(self.output_dir, "metrics.json")

    #     with open(preds_path, "w", encoding="utf-8") as f:
    #         for item in records:
    #             # 格式化每个字段为多行
    #             formatted_item = {
    #                 "id": item["id"],
    #                 "prediction": item["prediction"],
    #                 "reference": item["reference"], 
    #                 "raw_response": item["raw_response"],
    #                 "aux": item["aux"]
    #             }
    #             # 使用 indent 参数美化输出
    #             json_str = json.dumps(formatted_item, ensure_ascii=False, indent=2)
    #             f.write(json_str + "\n")  # 每个记录后加换行

    #     with open(metrics_path, "w", encoding="utf-8") as f:
    #         payload = dict(metrics)
    #         payload.update(metadata)
    #         json.dump(payload, f, ensure_ascii=False, indent=2)


@dataclass
class MedFullEvaluator:
    dataset_path: str
    image_base_dir: Optional[str] = None
    output_dir: str = "med_full_eval_results"
    model_name: str = "med-full-qwen"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    #新增
    shard_id: Optional[int] = None  # 当前节点 ID (0-based)
    num_shards: Optional[int] = None  # 总节点数

    def run(self) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)

        dataset = FullModelReportDataset(self.dataset_path, self.image_base_dir)
        model = build_model(self.model_name, **self.model_kwargs)

        predictions: List[str] = []
        references: List[str] = []
        raw_records: List[Dict[str, Any]] = []

        # samples = dataset.to_model_inputs()
        # total = len(samples)
        # 替换为分片逻辑
        samples = dataset.to_model_inputs()
        total = len(samples)
        if self.shard_id is not None and self.num_shards is not None:
            samples_per_shard = total // self.num_shards
            start_idx = self.shard_id * samples_per_shard
            if self.shard_id == self.num_shards - 1:
                end_idx = total  # 最后一个节点处理剩余所有样本
            else:
                end_idx = start_idx + samples_per_shard
            
            samples = samples[start_idx:end_idx]
            print(f"[Shard {self.shard_id}/{self.num_shards}] Processing samples {start_idx}-{end_idx} (total: {len(samples)})")
        else:
            print(f"[Single node] Processing all {total} samples")

        

        iterator = tqdm(samples, desc=f"Evaluating samples (shard {self.shard_id or 0})", unit="sample")
        for idx, sample in enumerate(iterator, start=1):
            output = model.generate(sample)

            predictions.append(output["prediction"])
            # reference = model._post_process_reference(sample["reference"])
            reference = sample["reference"]
            references.append(reference)
            raw_records.append(
                {
                    "id": sample["id"],
                    "prediction": output["prediction"],
                    "reference": reference,
                    "raw_response": output["raw_response"],
                    "aux": output["aux"]
                }
            )

        metrics = compute_nlg_metrics(predictions, references)
        metadata = {
            "model_name": self.model_name,
            "base_model_path": getattr(model, "base_model_path", None),
            "generation_kwargs": getattr(model, "generate_kwargs", None),
        }
        if self.shard_id is not None:
            # 分片模式：保存到子目录
            shard_output_dir = os.path.join(self.output_dir, f"shard_{self.shard_id}")
            os.makedirs(shard_output_dir, exist_ok=True)
            self._write_outputs(raw_records, metrics, metadata, output_dir=shard_output_dir)
        else:
            # 单节点模式：正常保存
            self._write_outputs(raw_records, metrics, metadata)
        return metrics

  
    def _write_outputs(
        self,
        records: List[Dict[str, Any]],
        metrics: Dict[str, float],
        metadata: Dict[str, Any],
        output_dir: Optional[str] = None,
    ) -> None:
        if output_dir is None:
            output_dir = self.output_dir
        preds_path = os.path.join(output_dir, "predictions.jsonl")  # ✅ 使用 output_dir
        metrics_path = os.path.join(output_dir, "metrics.json")  # ✅ 使用 output_dir

        with open(preds_path, "w", encoding="utf-8") as f:
            for item in records:
                # 格式化每个字段为多行
                formatted_item = {
                    "id": item["id"],
                    "prediction": item["prediction"],
                    "reference": item["reference"], 
                    "raw_response": item["raw_response"],
                    "aux": item["aux"]
                }
                # 使用 indent 参数美化输出
                json_str = json.dumps(formatted_item, ensure_ascii=False, indent=2)
                f.write(json_str + "\n")  # 每个记录后加换行

        with open(metrics_path, "w", encoding="utf-8") as f:
            payload = dict(metrics)
            payload.update(metadata)
            json.dump(payload, f, ensure_ascii=False, indent=2)