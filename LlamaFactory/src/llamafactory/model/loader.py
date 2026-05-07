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

import os
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoModelForSeq2SeqLM,
    AutoModelForTextToWaveform,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
)
from trl import AutoModelForCausalLMWithValueHead

from ..extras import logging
from ..extras.misc import count_parameters, skip_check_imports, try_download_model_from_other_hub
from ..extras.packages import is_torch_version_greater_than
from .adapter import init_adapter
from .model_utils.ktransformers import load_kt_pretrained_model
from .model_utils.liger_kernel import apply_liger_kernel
from .model_utils.misc import register_autoclass
from .model_utils.mod import convert_pretrained_model_to_mod, load_mod_pretrained_model
from .model_utils.unsloth import load_unsloth_pretrained_model
from .model_utils.valuehead import load_valuehead_params
from .patcher import patch_config, patch_model, patch_processor, patch_tokenizer, patch_valuehead_model


if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer, ProcessorMixin

    from ..hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)


class TokenizerModule(TypedDict):
    tokenizer: "PreTrainedTokenizer"
    processor: Optional["ProcessorMixin"]


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def load_tokenizer(model_args: "ModelArguments") -> "TokenizerModule":
    r"""Load pretrained tokenizer and optionally loads processor.

    Note: including inplace operation of model_args.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # try another one
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    patch_tokenizer(tokenizer, model_args)

    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            **init_kwargs,
        )
        logger.info_rank0(f"Processor loaded successfully: {processor.__class__.__name__}")

    except ValueError:  # try another one
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            **init_kwargs,
        )
        logger.info_rank0(f"except ValueError. Processor loaded successfully: {processor.__class__.__name__}")

    except Exception as e:
        logger.info_rank0(f"Failed to load processor: {e}.")
        processor = None

    # Avoid load tokenizer, see:
    # https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/models/auto/processing_auto.py#L324
    if processor is not None and "Processor" not in processor.__class__.__name__:
        logger.debug("The loaded processor is not an instance of Processor. Dropping it.")
        processor = None

    if processor is not None:
        patch_processor(processor, tokenizer, model_args)

    return {"tokenizer": tokenizer, "processor": processor}


def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""Load model config."""
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)


def load_model(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""Load pretrained model."""
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    apply_liger_kernel(config, model_args, is_trainable, require_logits=(finetuning_args.stage not in ["pt", "sft"]))

    model = None
    lazy_load = False
    if model_args.use_kt:
        from ktransformers.sft.monkey_patch_torch_module import install_patch

        install_patch()
        model = load_kt_pretrained_model(config, model_args)
    elif model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
        elif is_trainable:
            model = load_unsloth_pretrained_model(config, model_args, finetuning_args)

    if model is None and not lazy_load:
        init_kwargs["config"] = config
        init_kwargs["pretrained_model_name_or_path"] = model_args.model_name_or_path
        init_kwargs["torch_dtype"] = "auto"

        if model_args.mixture_of_depths == "load":
            model = load_mod_pretrained_model(**init_kwargs)
        else:
            # Debug: 检查 config 类型和模型映射
            config_type = type(config)  #<class 'transformers.models.qwen2_5_vl.configuration_qwen2_5_vl.Qwen2_5_VLConfig'>
            model_type = getattr(config, "model_type", None) #qwen2_5_vl
           
            if type(config) in AutoModelForImageTextToText._model_mapping.keys():  # image-text
                load_class = AutoModelForImageTextToText
                logger.info_rank0(f"[DEBUG] Using AutoModelForImageTextToText")
            elif type(config) in AutoModelForVision2Seq._model_mapping.keys():  # image-text
                load_class = AutoModelForVision2Seq
                logger.info_rank0(f"[DEBUG] Using AutoModelForVision2Seq")
            
            elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():  # audio-text
                load_class = AutoModelForSeq2SeqLM
                logger.info_rank0(f"[DEBUG] Using AutoModelForSeq2SeqLM")
            elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():  # audio hack for qwen omni
                load_class = AutoModelForTextToWaveform
                logger.info_rank0(f"[DEBUG] Using AutoModelForTextToWaveform")
            else:
                load_class = AutoModelForCausalLM
                logger.warning_rank0(
                    f"[DEBUG] Config type {config_type} not found in Vision2Seq mapping, "
                    f"using AutoModelForCausalLM. This may cause incorrect weight loading!"
                )

            if model_args.train_from_scratch: #从头训练
                model = load_class.from_config(config, trust_remote_code=model_args.trust_remote_code)
            else: #重新加载
                logger.info_rank0(f"Loading model with {load_class.__name__} from {model_args.model_name_or_path}")
                
                # Debug: 在加载前检查权重文件中的权重名称
                try:
                    from safetensors import safe_open
                    import os
                    from huggingface_hub import hf_hub_download
                    
                    # 尝试从缓存或下载权重索引文件
                    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
                    model_id = model_args.model_name_or_path
                    if "/" in model_id:
                        org, name = model_id.split("/", 1)
                    else:
                        org, name = None, model_id
                    
                    # 查找权重索引文件
                    index_file = None
                    for root, dirs, files in os.walk(cache_dir):
                        if "model.safetensors.index.json" in files:
                            full_path = os.path.join(root, "model.safetensors.index.json")
                            # 检查是否匹配模型ID
                            if org and name:
                                if org in root and name in root:
                                    index_file = full_path
                                    break
                    
                    if index_file:
                        import json
                        with open(index_file, 'r') as f:
                            index_data = json.load(f)
                        
                        # 检查权重文件中的 embed_tokens 相关权重名称
                        all_weight_names = []
                        if "weight_map" in index_data:
                            all_weight_names = list(index_data["weight_map"].keys())
                        
                        embed_weight_names = [k for k in all_weight_names if "embed_tokens" in k or "embed" in k.lower()]
                        logger.info_rank0(f"[DEBUG] Found {len(embed_weight_names)} embed-related keys in checkpoint:")
                        for key in embed_weight_names[:20]:  # 只打印前20个
                            logger.info_rank0(f"  - {key}")
                        if len(embed_weight_names) > 20:
                            logger.info_rank0(f"  ... and {len(embed_weight_names) - 20} more")
                except Exception as e:
                    logger.warning_rank0(f"[DEBUG] Could not inspect checkpoint weights: {e}")
                
                model = load_class.from_pretrained(**init_kwargs)
                if getattr(model.config, "model_type", None) in ["qwen2_5_omni", "qwen3_omni_moe"]:
                    model = getattr(model, "thinker")
                
                # # Debug: 检查加载的权重名称（仅检查 embed_tokens 相关的）
                # if hasattr(model, "state_dict"):
                #     state_dict_keys = list(model.state_dict().keys())
                #     embed_keys = [k for k in state_dict_keys if "embed_tokens" in k]
                #     logger.info_rank0(f"[DEBUG] Found {len(embed_keys)} embed_tokens keys in model state_dict:")
                #     for key in embed_keys[:10]:  # 只打印前10个
                #         logger.info_rank0(f"  - {key}")
                #     if len(embed_keys) > 10:
                #         logger.info_rank0(f"  ... and {len(embed_keys) - 10} more")
                
                # # Debug: 检查模型权重是否正常加载
                # if hasattr(model, "model") and hasattr(model.model, "language_model") and hasattr(model.model.language_model, "embed_tokens"):
                #     embed_tokens_weight = model.model.language_model.embed_tokens.weight
                #     if not embed_tokens_weight.is_meta:
                #         zeros_count = (embed_tokens_weight == 0).sum().item()
                #         total_count = embed_tokens_weight.numel()
                #         zeros_ratio = zeros_count / total_count * 100
                #         logger.info_rank0(
                #             f"[DEBUG] After model loading: embed_tokens.weight zeros: {zeros_count}/{total_count} ({zeros_ratio:.2f}%), "
                #             f"mean: {embed_tokens_weight.mean().item():.6f}, std: {embed_tokens_weight.std().item():.6f}"
                #         )
                #         if zeros_ratio > 50:
                #             logger.warning_rank0(
                #                 f"WARNING: embed_tokens.weight has {zeros_ratio:.2f}% zeros! "
                #                 "This suggests pretrained weights may not have loaded correctly."
                #             )
                #     else:
                #         logger.warning_rank0("[DEBUG] embed_tokens.weight is still meta tensor - weights not loaded yet")

        if model_args.mixture_of_depths == "convert":
            model = convert_pretrained_model_to_mod(model, config, model_args)

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)

    if add_valuehead:
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
        patch_valuehead_model(model)

        if model_args.adapter_name_or_path is not None:
            vhead_path = model_args.adapter_name_or_path[-1]
        else:
            vhead_path = model_args.model_name_or_path

        vhead_params = load_valuehead_params(vhead_path, model_args)
        if vhead_params is not None:
            model.load_state_dict(vhead_params, strict=False)
            logger.info_rank0(f"Loaded valuehead from checkpoint: {vhead_path}")

    # Conv3D is not recommended when using torch 2.9.x
    if is_torch_version_greater_than("2.9.0") and not is_torch_version_greater_than("2.10.0"):
        if any(isinstance(m, torch.nn.Conv3d) for m in model.modules()):
            raise ValueError(
                "Unsupported torch version detected: torch 2.9.x with Conv3D. "
                "This combination is known to cause severe performance regression. "
                "Please downgrade torch to <2.9 or remove Conv3D. "
                "See https://github.com/pytorch/pytorch/issues/166122"
            )

    if not is_trainable:
        model.requires_grad_(False)
        model.eval()
    else:
        model.train()

    # Borrowing the kernel plugins ability of v1 to temporarily apply the NPU fusion operator to v0,
    # it is turned off by default, and can be discarded after the transition period ends.
    if model_args.use_v1_kernels and is_trainable:
        logger.warning_rank0(
            "You are try to using future feature about kernels, please note that this feature "
            "is not supported for all models. If get any error, please disable this feature, or report the issue."
        )
        from ..v1.plugins.model_plugins.kernels.interface import apply_default_kernels

        model = apply_default_kernels(model, include_kernels=model_args.use_v1_kernels)

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = (
            f"trainable params: {trainable_params:,} || "
            f"all params: {all_param:,} || trainable%: {100 * trainable_params / all_param:.4f}"
        )
    else:
        param_stats = f"all params: {all_param:,}"

    logger.info_rank0(param_stats)

    if model_args.print_param_status and int(os.getenv("LOCAL_RANK", "0")) == 0:
        for name, param in model.named_parameters():
            print(f"name: {name}, dtype: {param.dtype}, device: {param.device}, trainable: {param.requires_grad}")

    return model
